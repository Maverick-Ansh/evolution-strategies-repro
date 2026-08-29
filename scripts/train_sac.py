"""SAC baseline -- the strongest available gradient-based arm on hopper/walker2d.

Why this file exists. Brax's own reference notebook trains hopper and walker2d with SAC,
not PPO, and publishes no tuned PPO config for either. Running only PPO there would have
handed ES a win on the two environments where the paper reports its LARGEST ES penalty
(Table 1: hopper 6.94x, walker2d 7.88x) -- exactly the rows a reproduction most needs to
get right.

SAC is a legitimate opponent for this paper's thesis. The abstract frames ES as an
alternative to "popular MDP-based RL techniques such as Q-learning and Policy Gradients",
and Sec. 2.1 argues ES avoids value functions entirely:
    "It does not require value function approximations. RL with value function estimation
     is inherently sequential: To improve upon a given policy, multiple updates to the
     value function are typically needed to get enough signal."
SAC is squarely inside the family being argued against -- off-policy, replay-based, with
a learned Q function -- so beating it is a stronger result for ES than beating a weak PPO,
and losing to it is a more informative negative.

Hyperparameters are Brax's, verbatim from notebooks/training.ipynb.
"""
import argparse, functools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
from brax import envs

from es.jaxcache import enable as _enable_jax_cache
_enable_jax_cache()
from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks

def dump(out, payload):
    """Write results at every eval, atomically, so a run that is stopped early still
    yields a usable learning curve instead of nothing."""
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, out)


BRAX_SAC = {
    'hopper':   dict(reward_scaling=30, discounting=0.997, lr=6e-4, num_envs=128,
                     batch_size=512, grad_updates_per_step=64),
    'walker2d': dict(reward_scaling=5, discounting=0.997, lr=6e-4, num_envs=128,
                     batch_size=128, grad_updates_per_step=32),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='hopper')
    p.add_argument('--backend', default='generalized')
    p.add_argument('--max-steps', type=int, default=5_000_000)
    p.add_argument('--episode-length', type=int, default=1000)
    p.add_argument('--action-repeat', type=int, default=1)
    p.add_argument('--delayed-reward', action='store_true')
    p.add_argument('--hidden', default='64,64')
    p.add_argument('--num-evals', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default=None)
    a = p.parse_args()

    cfg = BRAX_SAC.get(a.env, BRAX_SAC['hopper'])
    tag = a.tag or "sac_{}_s{}".format(a.env, a.seed)
    out = a.out or "results/{}.json".format(tag)
    if os.path.exists(out):
        try:
            if json.load(open(out)).get('complete', True):
                print("[skip] {} already complete".format(out))
                return
            print("[restart] {} exists but is incomplete".format(out))
        except Exception:
            pass

    env = envs.get_environment(a.env, backend=a.backend)
    if a.delayed_reward:
        from es.delayed import DelayedRewardWrapper
        env = DelayedRewardWrapper(env, a.episode_length)

    hidden = tuple(int(x) for x in a.hidden.split(','))
    nf = functools.partial(sac_networks.make_sac_networks,
                           hidden_layer_sizes=hidden, activation=jnp.tanh)

    hist = {'steps': [], 'eval': [], 'wall': []}
    t0 = time.time()

    def progress(num_steps, metrics):
        r = float(metrics.get('eval/episode_reward', float('nan')))
        hist['steps'].append(int(num_steps))
        hist['eval'].append(r)
        hist['wall'].append(time.time() - t0)
        print("  steps {:>11,} eval {:9.2f}  ({:.0f}s)".format(int(num_steps), r,
                                                               time.time() - t0), flush=True)
        dump(out, {'args': vars(a), 'cfg': cfg, 'hist': hist, 'wall_s': time.time() - t0,
                   'final_eval': hist['eval'][-1], 'complete': False})

    print("[{}] SAC env={} budget={:,} cfg={}".format(tag, a.env, a.max_steps, cfg), flush=True)

    sac.train(
        environment=env,
        num_timesteps=a.max_steps,
        episode_length=a.episode_length,
        action_repeat=a.action_repeat,
        num_envs=cfg['num_envs'],
        batch_size=cfg['batch_size'],
        grad_updates_per_step=cfg['grad_updates_per_step'],
        learning_rate=cfg['lr'],
        discounting=cfg['discounting'],
        reward_scaling=cfg['reward_scaling'],
        normalize_observations=True,
        max_replay_size=1048576,
        min_replay_size=8192,
        num_evals=a.num_evals,
        seed=a.seed,
        network_factory=nf,
        progress_fn=progress,
    )

    dump(out, {'args': vars(a), 'cfg': cfg, 'hist': hist, 'wall_s': time.time() - t0,
               'final_eval': hist['eval'][-1], 'complete': True})
    print("[{}] done in {:.0f}s -> {}".format(tag, time.time() - t0, out), flush=True)


if __name__ == '__main__':
    main()
