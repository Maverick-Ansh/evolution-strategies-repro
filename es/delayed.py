"""Maximally delayed reward -- the substrate for the Sec. 3.3 claim.

Paper, Sec. 3.3:
    "Also, ES can deal with maximally sparse and delayed rewards; only the total return
     of an episode is used, whereas other methods use individual rewards and their exact
     timing."

The wrapper withholds every per-step reward and pays the entire accumulated sum on the
terminal step. The episode's total return is therefore bit-identical to the dense
environment's; what is destroyed is the *timing* information that temporal-difference
and discounted policy-gradient methods rely on.

That makes this a clean single-variable ablation:
  * ES sees no change whatsoever -- it consumes only sum_t r_t, so its objective F is
    literally the same function. `scripts/smoke_delayed.py` asserts this bit-exactly
    rather than taking it on faith.
  * PPO loses its entire credit-assignment signal: every advantage estimate before the
    final step is now computed against a zero reward, and discounting shrinks the one
    real reward by gamma^T before it reaches early actions.

If ES degrades here at all, it is an implementation bug, not a property of ES -- which
is exactly why the assertion is worth having.
"""
from __future__ import annotations
import jax.numpy as jnp
from brax.envs.base import Wrapper


class DelayedRewardWrapper(Wrapper):
    """Pay sum_t r_t at the end of the episode; pay 0 at every other step."""

    def __init__(self, env, episode_length: int):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info['acc_reward'] = jnp.zeros_like(state.reward)
        state.info['delay_step'] = jnp.zeros_like(state.reward)
        return state

    def step(self, state, action):
        acc = state.info['acc_reward']
        n = state.info['delay_step']
        nxt = self.env.step(state, action)
        acc = acc + nxt.reward
        n = n + 1.0
        # pay out on termination OR when the horizon is exhausted, so no reward is
        # ever silently dropped by truncation
        at_end = jnp.logical_or(nxt.done > 0.0, n >= self.episode_length)
        reward = jnp.where(at_end, acc, jnp.zeros_like(acc))
        nxt.info['acc_reward'] = jnp.where(at_end, jnp.zeros_like(acc), acc)
        nxt.info['delay_step'] = n
        return nxt.replace(reward=reward)
