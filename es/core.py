"""The ES optimizer: Algorithm 1 / Algorithm 2, plus the pieces only the code has.

Paper, Algorithm 1 (Evolution Strategies):
    1: Input: Learning rate alpha, noise standard deviation sigma, initial policy
       parameters theta_0
    2: for t = 0, 1, 2, ... do
    3:   Sample eps_1, ... eps_n ~ N(0, I)
    4:   Compute returns F_i = F(theta_t + sigma*eps_i) for i = 1, ..., n
    5:   Set theta_{t+1} <- theta_t + alpha * (1/(n*sigma)) * sum_i F_i eps_i
    6: end for

Paper, Sec. 2.1:
    "To reduce variance, we use antithetic sampling [...] also known as mirrored
     sampling [...]: that is, we always evaluate pairs of perturbations eps, -eps, for
     Gaussian noise vector eps."

Paper, Sec. 2.1, on sigma:
    "Unlike Wierstra et al. [2014] we did not see benefit from adapting sigma during
     training, and we therefore treat it as a fixed hyperparameter instead."

Hyperparameters below are from configurations/humanoid.json in the released code, NOT
from the paper (the paper has no hyperparameter appendix at all):
    noise_stdev 0.02 | l2coeff 0.005 | stepsize 0.01 | optimizer adam
    return_proc_mode centered_rank | hidden_dims [256,256] | ac_noise_std 0.01
Note hidden_dims [256,256] in that config contradicts Sec. 4.1's "two 64-unit hidden
layers"; humanoid was evidently run with a wider net than the MuJoCo suite. We default
to (64,64) to match the *text* for the Table-1 environments and record the deviation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import jax, jax.numpy as jnp
import numpy as np

from .shaping import shaped_weights, es_gradient


# ---------------------------------------------------------------- Adam (optimizers.py)
@dataclass
class Adam:
    """Verbatim port of es_distributed/optimizers.py:Adam.

    The reference calls `optimizer.update(-g + l2coeff*theta)` -- i.e. it MINIMIZES,
    so the ascent direction g enters negated and weight decay enters positively.
    """
    dim: int
    stepsize: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0
    m: jnp.ndarray = field(default=None)
    v: jnp.ndarray = field(default=None)

    def __post_init__(self):
        if self.m is None: self.m = jnp.zeros(self.dim, jnp.float32)
        if self.v is None: self.v = jnp.zeros(self.dim, jnp.float32)

    def step(self, theta: jnp.ndarray, globalg: jnp.ndarray):
        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (jnp.sqrt(self.v) + self.epsilon)
        new_theta = theta + step
        ratio = float(jnp.linalg.norm(step) / (jnp.linalg.norm(theta) + 1e-12))  # "UpdateRatio"
        return new_theta, ratio


@dataclass
class SGDMom:
    """optimizers.py:SGD -- momentum 0.9. Used only for the Alg.-1-literal ablation."""
    dim: int
    stepsize: float = 0.01
    momentum: float = 0.9
    v: jnp.ndarray = field(default=None)

    def __post_init__(self):
        if self.v is None: self.v = jnp.zeros(self.dim, jnp.float32)

    def step(self, theta, globalg):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * globalg
        step = -self.stepsize * self.v
        return theta + step, float(jnp.linalg.norm(step) / (jnp.linalg.norm(theta) + 1e-12))


# ------------------------------------------------------- observation stats (es.py:RunningStat)
class RunningStat:
    """es.py:28-48. Fed from a random subsample of observations (calc_obstat_prob=0.01)."""

    def __init__(self, shape, eps: float = 1e-2):
        self.sum = np.zeros(shape, np.float64)
        self.sumsq = np.full(shape, eps, np.float64)
        self.count = float(eps)

    def increment(self, s, ssq, c):
        self.sum += np.asarray(s, np.float64)
        self.sumsq += np.asarray(ssq, np.float64)
        self.count += float(c)

    @property
    def mean(self):
        return (self.sum / self.count).astype(np.float32)

    @property
    def std(self):
        # reference clamps the variance at 1e-2 before the sqrt
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.sum / self.count), 1e-2)).astype(np.float32)


# ------------------------------------------------------------------------ the ES config
@dataclass
class ESConfig:
    n_pairs: int = 128              # antithetic pairs; population = 2 * n_pairs
    sigma: float = 0.02             # noise_stdev
    stepsize: float = 0.01
    l2coeff: float = 0.005
    return_proc_mode: str = "centered_rank"
    optimizer: str = "adam"         # "adam" | "sgd"
    divide_by_sigma: bool = False   # False = follow the CODE; True = follow Alg. 1 literally
    antithetic: bool = True
    obs_norm: bool = True
    calc_obstat_prob: float = 0.01
    seed: int = 0


def make_optimizer(cfg: ESConfig, dim: int):
    return Adam(dim, cfg.stepsize) if cfg.optimizer == "adam" else SGDMom(dim, cfg.stepsize)


def build_population(theta, eps, sigma, antithetic=True):
    """theta (D,), eps (n,D) -> perturbed parameter matrix (P,D) with P = 2n (or n).

    Antithetic ordering is [all +eps, then all -eps] so that the two halves line up
    index-for-index when returns are split again in `es_update`.
    """
    if antithetic:
        return jnp.concatenate([theta[None] + sigma * eps, theta[None] - sigma * eps], axis=0)
    return theta[None] + sigma * eps


def es_update(theta, eps, returns, cfg: ESConfig, opt):
    """One iteration of Algorithm 1 lines 5 (with shaping + weight decay).

    returns: (P,) episode returns aligned with build_population's ordering.
    """
    if cfg.antithetic:
        n = eps.shape[0]
        r_pos, r_neg = returns[:n], returns[n:]
    else:
        r_pos, r_neg = returns, jnp.zeros_like(returns)
    w = shaped_weights(r_pos, r_neg, cfg.return_proc_mode)
    g = es_gradient(w, eps, cfg.sigma, cfg.divide_by_sigma)
    # es.py:249 -- optimizer.update(-g + config.l2coeff * theta)
    theta, ratio = opt.step(theta, -g + cfg.l2coeff * theta)
    return theta, g, ratio
