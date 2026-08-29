"""Fitness shaping and the ES gradient assembly.

Paper, Sec. 2.1:
    "We also find it useful to perform fitness shaping [Wierstra et al., 2014] by
     applying a rank transformation to the returns before computing each parameter
     update. Doing so removes the influence of outlier individuals in each population
     and decreases the tendency for ES to fall into local optima early in training.
     In addition, we apply weight decay to the parameters of our policy network: this
     prevents the parameters from growing very large compared to the perturbations."

Two details below are NOT in the paper; they come from the released reference
implementation (openai/evolution-strategies-starter, es_distributed/es.py) and both
change the effective step size, so a reproduction has to pin them down:

  1. `compute_centered_ranks` ranks over the FLATTENED (n,2) array -- i.e. the
     positive and negative antithetic halves are ranked jointly against each other,
     not separately. Reference: es.py:80-84.

  2. The reference update is
         g = sum_i (rank+_i - rank-_i) * eps_i  /  returns_n2.size      (es.py:242-247)
     where `returns_n2.size == 2n`. Note there is NO division by sigma, even though
     Algorithm 1 line 5 of the paper reads

         theta_{t+1} <- theta_t + alpha * (1 / (n*sigma)) * sum_i F_i eps_i.

     The released code folds 1/sigma away entirely. With Adam (which the paper also
     never mentions -- see configurations/humanoid.json) the update is invariant to
     a global rescaling of g, so this is harmless there; with plain SGD it is a
     1/sigma = 50x difference in learning rate at sigma=0.02. We follow the CODE and
     expose `divide_by_sigma` so the paper's literal Alg. 1 can be run as an ablation.
"""
from __future__ import annotations
import jax, jax.numpy as jnp


def compute_ranks(x: jnp.ndarray) -> jnp.ndarray:
    """Ranks in [0, len(x)). Verbatim port of es.py:69-77.

    Note the reference comment: "This is different from scipy.stats.rankdata, which
    returns ranks in [1, len(x)]."
    """
    flat = x.ravel()
    order = jnp.argsort(flat)
    ranks = jnp.zeros_like(flat).at[order].set(jnp.arange(flat.size, dtype=flat.dtype))
    return ranks.reshape(x.shape)


def compute_centered_ranks(x: jnp.ndarray) -> jnp.ndarray:
    """Verbatim port of es.py:80-84: rank -> /(size-1) -> -0.5, giving [-0.5, +0.5]."""
    y = compute_ranks(x).astype(jnp.float32)
    y = y / (x.size - 1)
    return y - 0.5


def shaped_weights(returns_pos: jnp.ndarray, returns_neg: jnp.ndarray, mode: str = "centered_rank"):
    """Return the per-pair scalar weight w_i that multiplies eps_i in the ES gradient.

    `centered_rank` is the paper's default (configurations/humanoid.json:
    "return_proc_mode": "centered_rank"). `raw` is Algorithm 1 as literally written,
    i.e. no shaping at all -- kept so the ablation in Sec. 2.1 can be run.
    """
    if mode == "centered_rank":
        stacked = jnp.stack([returns_pos, returns_neg], axis=1)          # (n, 2)
        proc = compute_centered_ranks(stacked)                            # joint ranking
        return proc[:, 0] - proc[:, 1]
    if mode == "sign":
        return jnp.sign(returns_pos) - jnp.sign(returns_neg)
    if mode == "raw":
        return returns_pos - returns_neg
    if mode == "centered":                                                # mean-baseline, no ranks
        both = jnp.concatenate([returns_pos, returns_neg])
        m, s = both.mean(), both.std() + 1e-8
        return (returns_pos - m) / s - (returns_neg - m) / s
    raise NotImplementedError(mode)


def es_gradient(weights: jnp.ndarray, eps: jnp.ndarray, sigma: float,
                divide_by_sigma: bool = False) -> jnp.ndarray:
    """g = sum_i w_i eps_i / (2n)   [es.py:242-247].

    eps: (n, D) the POSITIVE half of each antithetic pair; the negative half is -eps.
    Division is by 2n (= returns_n2.size), i.e. by the number of EPISODES, not pairs.
    """
    n = eps.shape[0]
    g = jnp.einsum("n,nd->d", weights, eps) / (2 * n)
    return g / sigma if divide_by_sigma else g
