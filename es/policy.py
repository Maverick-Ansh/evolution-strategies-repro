"""Feed-forward policy, ported from openai/evolution-strategies-starter MujocoPolicy.

Paper, Sec. 4.1:
    "We used both ES and TRPO to train policies with identical architectures:
     multilayer perceptrons with two 64-unit hidden layers separated by tanh
     nonlinearities."

Paper, Sec. 4.1 (the 'one binary hyperparameter' of key finding 5):
    "We found that ES occasionally benefited from discrete actions, since continuous
     actions could be too smooth with respect to parameter perturbation and could
     hamper exploration (see section 2.2). For the hopping and swimming tasks, we
     discretized the actions for ES into 10 bins for each action component."

Paper, Sec. 2.2, on why parameterization matters at all:
    "For ES to improve upon parameters theta, some members of the population must
     achieve better return than others: i.e. it is crucial that Gaussian perturbation
     vectors eps occasionally lead to new individuals theta+sigma*eps with better
     return."

Details taken from the code rather than the paper (policies.py MujocoPolicy._make_net):
  * observations are normalized and CLIPPED: clip((o - ob_mean)/ob_std, -5, 5).
    This is the MuJoCo counterpart of the virtual batch normalization the paper uses
    for Atari -- the paper's key finding 1 says these reparameterizations are what
    make ES non-brittle, so it is not an optional detail.
  * hidden layers use normc_initializer(1.0), the output layer normc_initializer(0.01)
    (tf_util.py:109-114): sample N(0,1) then rescale each COLUMN to unit norm.
  * an extra Gaussian action noise of std `ac_noise_std` (0.01 in humanoid.json) is
    added at rollout time, on top of the parameter perturbation. The paper never
    mentions this.

Everything is expressed as a function of a FLAT parameter vector theta (D,), because
ES only ever manipulates theta as a vector (get_trainable_flat / set_trainable_flat in
the reference). A whole population is then just a (P, D) array and one vmap.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import jax, jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    obs_dim: int
    act_dim: int
    hidden: Tuple[int, ...] = (64, 64)
    ac_bins: int = 0          # 0 => continuous; k>0 => uniform k-bin discretization
    ac_noise_std: float = 0.01
    act_low: float = -1.0
    act_high: float = 1.0
    obs_clip: float = 5.0

    @property
    def out_dim(self) -> int:
        # discretized policies emit one score per (action dim, bin) -- policies.py `bins`
        return self.act_dim * self.ac_bins if self.ac_bins > 0 else self.act_dim

    @property
    def layer_dims(self):
        return [self.obs_dim, *self.hidden, self.out_dim]

    @property
    def num_params(self) -> int:
        d = self.layer_dims
        return sum(d[i] * d[i + 1] + d[i + 1] for i in range(len(d) - 1))


def _normc(key, shape, std=1.0):
    """tf_util.normc_initializer: randn, then rescale each column to norm `std`."""
    out = jax.random.normal(key, shape, dtype=jnp.float32)
    return out * std / jnp.sqrt(jnp.square(out).sum(axis=0, keepdims=True))


def init_flat(key, spec: PolicySpec) -> jnp.ndarray:
    """Initial theta_0. normc(1.0) on hidden layers, normc(0.01) on the output layer."""
    d = spec.layer_dims
    keys = jax.random.split(key, len(d) - 1)
    parts = []
    for i, k in enumerate(keys):
        std = 1.0 if i < len(d) - 2 else 0.01
        parts.append(_normc(k, (d[i], d[i + 1]), std).ravel())
        parts.append(jnp.zeros((d[i + 1],), jnp.float32))
    return jnp.concatenate(parts)


def unflatten(theta: jnp.ndarray, spec: PolicySpec):
    d, o, out = spec.layer_dims, 0, []
    for i in range(len(d) - 1):
        nw = d[i] * d[i + 1]
        W = jax.lax.dynamic_slice(theta, (o,), (nw,)).reshape(d[i], d[i + 1]); o += nw
        b = jax.lax.dynamic_slice(theta, (o,), (d[i + 1],));                   o += d[i + 1]
        out.append((W, b))
    return out


def forward(theta, obs, ob_mean, ob_std, spec: PolicySpec):
    """Scores for one member. obs: (obs_dim,) -> (out_dim,)."""
    x = jnp.clip((obs - ob_mean) / ob_std, -spec.obs_clip, spec.obs_clip)
    layers = unflatten(theta, spec)
    for W, b in layers[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = layers[-1]
    return x @ W + b


def act(theta, obs, ob_mean, ob_std, spec: PolicySpec, key=None):
    """Map observation to a bounded action, matching MujocoPolicy._make_net."""
    z = forward(theta, obs, ob_mean, ob_std, spec)
    if spec.ac_bins > 0:
        # policies.py: aidx = argmax over bins; a = idx/(k-1) * (high-low) + low
        idx = jnp.argmax(z.reshape(spec.act_dim, spec.ac_bins), axis=-1).astype(jnp.float32)
        a = idx / (spec.ac_bins - 1.0) * (spec.act_high - spec.act_low) + spec.act_low
    else:
        a = z
    if key is not None and spec.ac_noise_std > 0:
        a = a + spec.ac_noise_std * jax.random.normal(key, a.shape)
    return jnp.clip(a, spec.act_low, spec.act_high)


def act_population(thetas, obs, ob_mean, ob_std, spec: PolicySpec, keys=None):
    """thetas (P,D), obs (P,obs_dim) -> (P,act_dim). One perturbed policy per env."""
    f = lambda th, ob, k: act(th, ob, ob_mean, ob_std, spec, k)
    if keys is None:
        return jax.vmap(lambda th, ob: f(th, ob, None))(thetas, obs)
    return jax.vmap(f)(thetas, obs, keys)
