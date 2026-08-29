"""Phase-4 gate: measure the FLOOR of every environment before running any sweep.

The failure this exists to catch: an environment where a degenerate policy already
scores near the top has no dynamic range left, so "ES solved it" and "ES did nothing"
produce the same number. The paper's Table 3 quotes TRPO's InvertedPendulum score as
1000.00 -- exactly the episode cap -- so this environment is saturating by construction
and any reproduction has to know where its floor sits.

Three floors, in increasing order of relevance to ES:
  zero    : constant zero action. The "do nothing" policy.
  random  : uniform random action each step. The "no information" policy.
  init    : a freshly normc-initialised network, i.e. ES's actual theta_0 -- this is
            the number an ES learning curve must be read against.

Also reports the mean episode LENGTH, which tells us whether termination fires at all.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from es.policy import PolicySpec, init_flat, act_population

TABLE1 = ['halfcheetah', 'hopper', 'inverted_double_pendulum',
          'inverted_pendulum', 'swimmer', 'walker2d']


def rollout_const(env, T, B, mode, key, spec=None, thetas=None):
    state = jax.jit(env.reset)(key)
    step = jax.jit(env.step)
    ret = jnp.zeros(B)
    alive = jnp.ones(B)
    length = jnp.zeros(B)
    k = key
    for _ in range(T):
        if mode == 'zero':
            a = jnp.zeros((B, env.action_size))
        elif mode == 'random':
            k, sk = jax.random.split(k)
            a = jax.random.uniform(sk, (B, env.action_size), minval=-1.0, maxval=1.0)
        else:  # freshly initialised network, no obs normalisation yet
            a = act_population(thetas, state.obs, jnp.zeros(spec.obs_dim),
                               jnp.ones(spec.obs_dim), spec, None)
        nxt = step(state, a)
        ret = ret + nxt.reward * alive
        length = length + alive
        alive = alive * (1.0 - nxt.done)
        state = nxt
    return float(ret.mean()), float(ret.std()), float(length.mean()), float(alive.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='generalized')
    p.add_argument('--episode-length', type=int, default=1000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--out', default='results/env_floor.json')
    a = p.parse_args()

    rows = {}
    print("{:26s} {:>10s} {:>10s} {:>10s} {:>8s} {:>7s}".format(
        "env", "zero", "random", "init-net", "len(rnd)", "alive%"))
    for nm in TABLE1:
        env = envs.create(nm, backend=a.backend, episode_length=a.episode_length,
                          batch_size=a.batch, auto_reset=False)
        spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size,
                          hidden=(64, 64), ac_bins=0, ac_noise_std=0.0)
        thetas = jnp.stack([init_flat(jax.random.PRNGKey(1000 + i), spec)
                            for i in range(a.batch)])
        z, zs, zl, _ = rollout_const(env, a.episode_length, a.batch, 'zero', jax.random.PRNGKey(0))
        r, rsd, rl, ral = rollout_const(env, a.episode_length, a.batch, 'random', jax.random.PRNGKey(1))
        i_, isd, il, _ = rollout_const(env, a.episode_length, a.batch, 'init', jax.random.PRNGKey(2),
                                       spec, thetas)
        rows[nm] = dict(zero=z, zero_len=zl, random=r, random_std=rsd, random_len=rl,
                        init=i_, init_std=isd, init_len=il, alive_frac=ral,
                        episode_length=a.episode_length, obs=env.observation_size,
                        act=env.action_size)
        print("{:26s} {:10.1f} {:10.1f} {:10.1f} {:8.0f} {:6.0f}%".format(
            nm, z, r, i_, rl, 100 * ral))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(rows, open(a.out, 'w'), indent=1)
    print("\nwrote", a.out)

    print("\nVERDICT")
    for nm, d in rows.items():
        # an environment where the no-information policy already survives the full
        # episode has no termination signal to learn from
        if d['random_len'] >= 0.95 * d['episode_length']:
            print("  [!] {}: random policy survives {:.0f}/{} steps -- termination never "
                  "fires, so 'staying alive' is free and the reward is pure shaping".format(
                      nm, d['random_len'], d['episode_length']))
        else:
            print("  [ok] {}: random policy terminates at {:.0f}/{} steps".format(
                nm, d['random_len'], d['episode_length']))


if __name__ == '__main__':
    main()
