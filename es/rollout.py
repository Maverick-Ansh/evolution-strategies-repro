"""GPU-vectorized evaluation of an ES population -- the resize of the paper's cluster.

The paper's compute model (Sec. 2.1, 4.3) is one CPU worker per perturbation:
    "By parallelizing the evaluation of perturbed parameters across 720 CPUs on Amazon
     EC2, we can bring down the time required for the training process to about one
     hour per game."
    "when distributed across 80 machines and 1,440 CPU cores, ES can solve 3D Humanoid
     in just 10 minutes"

That is a statement about ES's *structure*: the population is embarrassingly parallel
and the workers only exchange scalars. In 2017 the only way to cash that structure in
was a CPU cluster, because MuJoCo ran on CPU. The structure is unchanged today, but
the hardware that exploits it is different: a GPU-resident batched simulator (Brax/MJX)
evaluates P perturbed policies as ONE batched program. So we keep the algorithm exactly
and swap the parallel substrate:

    paper:  P perturbations -> P CPU processes -> all-gather P scalars
    here:   P perturbations -> P lanes of one batched rollout -> reduce P scalars

The claim being tested (low communication, near-linear return on added parallelism) is
preserved; only the machine changes. What is NOT preserved, and is reported as such, is
the paper's wall-clock numbers, which are numbers about a 1,440-core cluster.

Episode-length capping is implemented as the paper describes it (Sec. 2.1):
    "we cap episode length at a constant m steps for all workers, which we dynamically
     adjust as training progresses. For example, by setting m to be equal to twice the
     mean number of steps taken per episode, we can guarantee that CPU utilization
     stays above 50% in the worst case."
"""
from __future__ import annotations
from functools import partial
import jax, jax.numpy as jnp

from .policy import act_population, PolicySpec


def make_pop_rollout(env, spec: PolicySpec, max_T: int):
    """Build a jitted fn: (thetas (P,D), key, ob_mean, ob_std, cap) -> stats.

    Reward delay is applied at the environment level (es/delayed.py), not here, so that
    the ES and PPO arms consume literally the same environment.
    """

    def rollout(thetas, key, ob_mean, ob_std, cap):
        P = thetas.shape[0]
        rk, sk = jax.random.split(key)
        state = env.reset(rk)

        def body(carry, t):
            state, ret, alive, length, osum, ossq, k = carry
            k, ak = jax.random.split(k)
            akeys = jax.random.split(ak, P)
            a = act_population(thetas, state.obs, ob_mean, ob_std, spec, akeys)
            nxt = env.step(state, a)
            # a lane stops contributing once it has terminated, or once we pass the cap
            live = alive * (t < cap)
            ret = ret + nxt.reward * live
            length = length + live
            osum = osum + jnp.sum(state.obs * live[:, None], axis=0)
            ossq = ossq + jnp.sum(jnp.square(state.obs) * live[:, None], axis=0)
            alive = live * (1.0 - nxt.done)
            return (nxt, ret, alive, length, osum, ossq, k), None

        init = (state,
                jnp.zeros(P), jnp.ones(P), jnp.zeros(P),
                jnp.zeros(spec.obs_dim), jnp.zeros(spec.obs_dim), sk)
        (state, ret, alive, length, osum, ossq, _), _ = jax.lax.scan(
            body, init, jnp.arange(max_T))
        return dict(returns=ret, lengths=length, obs_sum=osum, obs_sumsq=ossq,
                    obs_count=jnp.sum(length))

    return jax.jit(rollout)


def dynamic_cap(mean_length: float, max_T: int, mult: float = 2.0) -> int:
    """Sec. 2.1: m = 2 x mean episode length, clipped to the env's own limit."""
    return int(min(max_T, max(50, mult * mean_length)))
