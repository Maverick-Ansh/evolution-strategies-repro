"""The shared noise table -- the mechanism behind the paper's headline scaling claim.

Paper, Sec. 2.1:
    "The only information obtained by each worker is the scalar return of an episode:
     if we synchronize random seeds between workers before optimization, each worker
     knows what perturbations the other workers used, so each worker only needs to
     communicate a single scalar to and from each other worker to agree on a parameter
     update. ES thus requires extremely low bandwidth, in sharp contrast to policy
     gradient methods, which require workers to communicate entire gradients."

and:
    "In practice, we implement sampling by having each worker instantiate a large block
     of Gaussian noise at the start of training, and then perturbing its parameters by
     adding a randomly indexed subset of these noise variables at each iteration.
     Although this means that the perturbations are not strictly independent across
     iterations, we did not find this to be a problem in practice."

The reference implementation (es.py:50-66) allocates 250M float32 = 1 GB of shared
memory. We keep the same structure but size it to the device: the table lives on the
GPU once, and a perturbation is a *slice*, so a population of P perturbations costs P
int32 indices to describe rather than P*D floats.

This is what makes the bandwidth claim exact rather than rhetorical: see
scripts/bandwidth.py, which measures bytes-per-iteration for ES vs data-parallel SGD.
"""
from __future__ import annotations
import jax, jax.numpy as jnp
import numpy as np


class SharedNoiseTable:
    """A fixed block of N(0,1) floats, addressable by (index, dim) -> slice.

    Every worker constructs this from the same integer seed, so referring to a
    perturbation costs one int32 instead of D floats.
    """

    def __init__(self, seed: int = 123, count: int = 25_000_000, device=None):
        # reference uses count=250_000_000 (1 GB); we default to 25M (100 MB) so the
        # table plus a population of policies both fit in a 16 GB T4.
        self.seed, self.count = seed, count
        rs = np.random.RandomState(seed)
        block = rs.randn(count).astype(np.float32)      # 64->32 bit conversion, as in es.py:59
        self.noise = jax.device_put(block, device) if device is not None else jnp.asarray(block)

    @property
    def nbytes(self) -> int:
        return self.count * 4

    def get(self, i: int, dim: int) -> jnp.ndarray:
        """es.py:62-63 -- self.noise[i:i+dim]."""
        return jax.lax.dynamic_slice(self.noise, (i,), (dim,))

    def sample_index(self, rng: np.random.RandomState, dim: int) -> int:
        """es.py:65-66 -- stream.randint(0, len(noise) - dim + 1)."""
        return int(rng.randint(0, self.count - dim + 1))

    def batch(self, idxs: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Gather a whole population of perturbations as one (P, dim) array.

        This is the GPU restatement of Algorithm 2 lines 9-11: 'reconstruct all
        perturbations eps_j using known random seeds'. On a CPU cluster each worker
        rebuilds every other worker's eps_j locally; on one GPU the same
        reconstruction is a single batched dynamic_slice, and the thing that crossed
        the wire is still only the index + the scalar return.
        """
        return jax.vmap(lambda i: jax.lax.dynamic_slice(self.noise, (i,), (dim,)))(idxs)


def perturbation_descriptor_bytes(n_pairs: int) -> int:
    """Bytes ES must broadcast per iteration: one int32 index + one float32 return
    per antithetic pair (the reference sends returns for both halves)."""
    return n_pairs * (4 + 2 * 4)


def gradient_broadcast_bytes(n_workers: int, num_params: int, dtype_bytes: int = 4) -> int:
    """Bytes a data-parallel policy-gradient method must all-reduce per iteration:
    every worker contributes a full D-dimensional gradient."""
    return n_workers * num_params * dtype_bytes
