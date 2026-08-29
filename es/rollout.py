"""GPU-vectorized evaluation of an ES population -- the resize of the paper's cluster.

The paper's compute model (Sec. 2.1, 4.3) is one CPU worker per perturbation:
    "By parallelizing the evaluation of perturbed parameters across 720 CPUs on Amazon
     EC2, we can bring down the time required for the training process to about one
     hour per game."
    "when distributed across 80 machines and 1,440 CPU cores, ES can solve 3D Humanoid
     in just 10 minutes"

That is a statement about ES's *structure*: the population is embarrassingly parallel
and the workers only ever exchange scalars. In 2017 the only way to cash that structure
in was a CPU cluster, because MuJoCo ran on CPU. The structure is unchanged today, but
the hardware that exploits it is not: a GPU-resident batched simulator (Brax) evaluates
P perturbed policies as ONE batched program. So we keep the algorithm exactly and swap
the parallel substrate:

    paper:  P perturbations -> P CPU processes   -> all-gather P scalars
    here:   P perturbations -> P lanes of one    -> reduce P scalars
                               batched rollout

The claim under test (low communication; near-linear return on added parallelism) is
preserved. What is NOT preserved, and is reported as such, is the paper's wall-clock
numbers, which are facts about a 1,440-core cluster.

EPISODE CAPPING. Sec. 2.1:
    "Evolution Strategies, as presented above, works with full-length episodes. In some
     rare cases this can lead to low CPU utilization, as some episodes run for many more
     steps than others. For this reason, we cap episode length at a constant m steps for
     all workers, which we dynamically adjust as training progresses. For example, by
     setting m to be equal to twice the mean number of steps taken per episode, we can
     guarantee that CPU utilization stays above 50% in the worst case."

On a GPU the cap has to be the STATIC length of the scan, not a runtime comparison,
or the program still pays for every one of the 1000 possible steps even when every
lane died at step 20. Making m static means recompiling when m changes, so callers
bucket m to powers of two (see `cap_bucket`) and cache one compiled rollout per bucket
-- a handful of compilations for the whole run, in exchange for the ~30x speedup that
the paper got for free by simply not running those steps on its CPUs.
"""
from __future__ import annotations
import jax, jax.numpy as jnp

from .policy import act_population, PolicySpec


def make_pop_rollout(env, spec: PolicySpec, scan_T: int):
    """Build a jitted fn: (thetas (P,D), key, ob_mean, ob_std) -> per-lane statistics.

    `scan_T` is both the number of simulated steps and the paper's episode cap m.
    Reward delay is applied at the environment level (es/delayed.py), not here, so the
    ES and PPO arms consume literally the same environment.
    """

    def rollout(thetas, key, ob_mean, ob_std):
        P = thetas.shape[0]
        rk, sk = jax.random.split(key)
        state = env.reset(rk)

        def body(carry, _):
            state, ret, alive, length, osum, ossq, k = carry
            k, ak = jax.random.split(k)
            akeys = jax.random.split(ak, P)
            a = act_population(thetas, state.obs, ob_mean, ob_std, spec, akeys)
            nxt = env.step(state, a)
            # a lane stops contributing the moment it terminates
            ret = ret + nxt.reward * alive
            length = length + alive
            osum = osum + jnp.sum(state.obs * alive[:, None], axis=0)
            ossq = ossq + jnp.sum(jnp.square(state.obs) * alive[:, None], axis=0)
            alive = alive * (1.0 - nxt.done)
            return (nxt, ret, alive, length, osum, ossq, k), None

        init = (state, jnp.zeros(P), jnp.ones(P), jnp.zeros(P),
                jnp.zeros(spec.obs_dim), jnp.zeros(spec.obs_dim), sk)
        (state, ret, alive, length, osum, ossq, _), _ = jax.lax.scan(
            body, init, None, length=scan_T)
        return dict(returns=ret, lengths=length, obs_sum=osum, obs_sumsq=ossq,
                    obs_count=jnp.sum(length), truncated=jnp.mean(alive))

    return jax.jit(rollout)


def cap_bucket(mean_length: float, max_T: int, mult: float = 2.0, min_T: int = 32) -> int:
    """Sec. 2.1's m = 2 x mean episode length, rounded UP to a power of two.

    Bucketing keeps the number of distinct XLA compilations to log2(max_T/min_T) ~ 5
    for a whole run, instead of one per iteration.
    """
    m = max(min_T, mult * mean_length)
    b = min_T
    while b < m and b < max_T:
        b *= 2
    return int(min(b, max_T))


class RolloutCache:
    """One compiled rollout per episode-cap bucket, built lazily over a single env."""

    def __init__(self, env, spec: PolicySpec):
        self.env, self.spec, self._cache = env, spec, {}

    def get(self, scan_T: int):
        if scan_T not in self._cache:
            self._cache[scan_T] = make_pop_rollout(self.env, self.spec, scan_T)
        return self._cache[scan_T]

    @property
    def compiled(self):
        return sorted(self._cache)
