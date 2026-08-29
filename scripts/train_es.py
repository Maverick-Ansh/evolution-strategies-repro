"""ES training on GPU-vectorized Brax. Algorithm 2, with one GPU standing in for the cluster.

Sample complexity is logged in ENV TIMESTEPS, because that is the x-axis of the claim
under test (Table 1: "Ratio of ES timesteps to TRPO timesteps needed to reach various
percentages of TRPO's learning progress at 5 million timesteps").
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from es.core import ESConfig, RunningStat, make_optimizer, build_population, es_update
from es.policy import PolicySpec, init_flat
from es.rollout import make_pop_rollout, dynamic_cap
from es.noise import SharedNoiseTable

# Sec. 4.1: "we discretized the actions for ES into 10 bins for each action component"
# for "the hopping and swimming tasks". This is key finding 5's "one binary
# hyperparameter, which has not been held constant between the different MuJoCo
# environments" -- the only per-env hyperparameter in the entire paper.
DISCRETIZE = {'hopper': 10, 'swimmer': 10}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='inverted_pendulum')
    p.add_argument('--backend', default='generalized')
    p.add_argument('--n-pairs', type=int, default=256)      # population = 2 * n_pairs
    p.add_argument('--sigma', type=float, default=0.02)     # noise_stdev, humanoid.json
    p.add_argument('--stepsize', type=float, default=0.01)  # humanoid.json
    p.add_argument('--l2coeff', type=float, default=0.005)  # humanoid.json
    p.add_argument('--hidden', default='64,64')             # Sec. 4.1
    p.add_argument('--episode-length', type=int, default=1000)
    p.add_argument('--action-repeat', type=int, default=1)  # Sec. 4.4 frame-skip
    p.add_argument('--delayed-reward', action='store_true')  # Sec. 3.3 sparse/delayed
    p.add_argument('--max-steps', type=int, default=5_000_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shaping', default='centered_rank')
    p.add_argument('--optimizer', default='adam')
    p.add_argument('--divide-by-sigma', action='store_true')
    p.add_argument('--no-obs-norm', action='store_true')
    p.add_argument('--no-antithetic', action='store_true')
    p.add_argument('--discretize', type=int, default=-1)     # -1 => use DISCRETIZE table
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-batch', type=int, default=256)
    p.add_argument('--dynamic-cap', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default=None)
    a = p.parse_args()

    tag = a.tag or "es_{}_s{}".format(a.env, a.seed)
    out = a.out or "results/{}.json".format(tag)
    if os.path.exists(out):
        print("[skip] {} exists".format(out))
        return

    hidden = tuple(int(x) for x in a.hidden.split(','))
    bins = DISCRETIZE.get(a.env, 0) if a.discretize < 0 else a.discretize
    P = 2 * a.n_pairs if not a.no_antithetic else a.n_pairs

    env = envs.create(a.env, backend=a.backend, episode_length=a.episode_length,
                      action_repeat=a.action_repeat, batch_size=P, auto_reset=False)
    eval_env = envs.create(a.env, backend=a.backend, episode_length=a.episode_length,
                           action_repeat=a.action_repeat, batch_size=a.eval_batch,
                           auto_reset=False)

    spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size, hidden=hidden,
                      ac_bins=bins, ac_noise_std=0.01)
    # The paper reports "deterministic policy evaluation" for its final numbers
    # (Table 2 caption), so evaluation drops the action noise.
    eval_spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size, hidden=hidden,
                           ac_bins=bins, ac_noise_std=0.0)

    cfg = ESConfig(n_pairs=a.n_pairs, sigma=a.sigma, stepsize=a.stepsize, l2coeff=a.l2coeff,
                   return_proc_mode=a.shaping, optimizer=a.optimizer,
                   divide_by_sigma=a.divide_by_sigma, antithetic=not a.no_antithetic,
                   obs_norm=not a.no_obs_norm, seed=a.seed)

    key = jax.random.PRNGKey(a.seed)
    key, ik = jax.random.split(key)
    theta = init_flat(ik, spec)
    D = spec.num_params
    opt = make_optimizer(cfg, D)
    noise = SharedNoiseTable(seed=123, count=max(25_000_000, D * 40))
    rs = np.random.RandomState(a.seed)
    obstat = RunningStat((spec.obs_dim,), eps=1e-2)

    rollout = make_pop_rollout(env, spec, a.episode_length, a.delayed_reward)
    evaluate = make_pop_rollout(eval_env, eval_spec, a.episode_length, False)

    hist = {'steps': [], 'eval': [], 'pop_mean': [], 'pop_max': [], 'iter': [],
            'wall': [], 'update_ratio': [], 'mean_len': []}
    total_steps, t0, it = 0, time.time(), 0
    cap = a.episode_length

    print("[{}] env={} backend={} D={} P={} bins={} obs={} act={} budget={:,}".format(
        tag, a.env, a.backend, D, P, bins, spec.obs_dim, spec.act_dim, a.max_steps), flush=True)

    while total_steps < a.max_steps:
        it += 1
        # Alg. 2 line 5: each worker samples eps_i -- here, one index into the shared table
        idxs = jnp.asarray([noise.sample_index(rs, D) for _ in range(a.n_pairs)])
        eps = noise.batch(idxs, D)

        om = jnp.asarray(obstat.mean) if cfg.obs_norm else jnp.zeros(spec.obs_dim)
        os_ = jnp.asarray(obstat.std) if cfg.obs_norm else jnp.ones(spec.obs_dim)

        pop = build_population(theta, eps, cfg.sigma, cfg.antithetic)
        key, rk = jax.random.split(key)
        r = rollout(pop, rk, om, os_, cap)
        returns = r['returns']
        lengths = np.asarray(r['lengths'])
        total_steps += int(lengths.sum()) * a.action_repeat

        if cfg.obs_norm:
            obstat.increment(np.asarray(r['obs_sum']), np.asarray(r['obs_sumsq']),
                             float(r['obs_count']))
        theta, g, ratio = es_update(theta, eps, returns, cfg, opt)

        if a.dynamic_cap:      # Sec. 2.1: m = 2 x mean episode length
            cap = dynamic_cap(float(lengths.mean()), a.episode_length)

        if it % a.eval_every == 0 or total_steps >= a.max_steps:
            key, ek = jax.random.split(key)
            ev = evaluate(jnp.tile(theta[None], (a.eval_batch, 1)), ek, om, os_,
                          a.episode_length)
            ev_ret = float(jnp.mean(ev['returns']))
            hist['steps'].append(total_steps)
            hist['eval'].append(ev_ret)
            hist['pop_mean'].append(float(jnp.mean(returns)))
            hist['pop_max'].append(float(jnp.max(returns)))
            hist['iter'].append(it)
            hist['wall'].append(time.time() - t0)
            hist['update_ratio'].append(ratio)
            hist['mean_len'].append(float(lengths.mean()))
            print("  it {:5d} steps {:>11,} eval {:9.2f} pop {:9.2f} len {:6.1f} {:7.1f}k sps".format(
                it, total_steps, ev_ret, float(jnp.mean(returns)), lengths.mean(),
                total_steps / (time.time() - t0) / 1e3), flush=True)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'args': vars(a), 'D': D, 'P': P, 'bins': bins, 'hist': hist,
               'wall_s': time.time() - t0, 'final_eval': hist['eval'][-1]},
              open(out, 'w'), indent=1)
    print("[{}] done in {:.0f}s -> {}".format(tag, time.time() - t0, out), flush=True)


if __name__ == '__main__':
    main()
