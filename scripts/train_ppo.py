"""Policy-gradient baseline (Brax PPO) -- the modern stand-in for the paper's TRPO arm.

Deviation, stated up front: the paper compares against
    "a highly tuned implementation of Trust Region Policy Optimization [TRPO;
     Schulman et al., 2015], a policy gradient algorithm designed to efficiently
     optimize neural network policies"
We compare against PPO instead. PPO is TRPO's direct successor from the same authors,
is what a 2026 practitioner would actually reach for, and -- decisively for this
reproduction -- Brax ships a tuned, GPU-native implementation, so both arms run on the
same simulator and the same hardware. Substituting the baseline is legitimate here
because the quantity under test is the ES/PG *ratio* (Table 1), not either arm's
absolute score.

Architecture is matched to the paper's protocol (Sec. 4.1):
    "We used both ES and TRPO to train policies with identical architectures:
     multilayer perceptrons with two 64-unit hidden layers separated by tanh
     nonlinearities."
so --hidden defaults to 64,64 with tanh. Because a crippled baseline would silently
manufacture an ES win, --preset brax additionally runs Brax's own tuned default
network, and the report quotes both.

One fixed hyperparameter set is used for all six environments, mirroring the paper's
key finding 5: "we achieved the aforementioned results using fixed hyperparameters for
all the Atari environments, and a different set of fixed hyperparameters for all MuJoCo
environments".
"""
import argparse, functools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='inverted_pendulum')
    p.add_argument('--backend', default='generalized')
    p.add_argument('--max-steps', type=int, default=5_000_000)
    p.add_argument('--episode-length', type=int, default=1000)
    p.add_argument('--action-repeat', type=int, default=1)
    p.add_argument('--delayed-reward', action='store_true')
    p.add_argument('--hidden', default='64,64')
    p.add_argument('--activation', default='tanh')
    p.add_argument('--preset', default='paper', choices=['paper', 'brax'])
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--unroll-length', type=int, default=20)
    p.add_argument('--num-minibatches', type=int, default=32)
    p.add_argument('--updates-per-batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--entropy-cost', type=float, default=1e-3)
    p.add_argument('--discounting', type=float, default=0.97)
    p.add_argument('--num-evals', type=int, default=25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default=None)
    a = p.parse_args()

    tag = a.tag or "ppo_{}_s{}".format(a.env, a.seed)
    out = a.out or "results/{}.json".format(tag)
    if os.path.exists(out):
        print("[skip] {} exists".format(out))
        return

    env = envs.get_environment(a.env, backend=a.backend)
    if a.delayed_reward:
        from es.delayed import DelayedRewardWrapper
        env = DelayedRewardWrapper(env, a.episode_length)

    act = {'tanh': jnp.tanh, 'swish': jax.nn.swish, 'relu': jax.nn.relu}[a.activation]
    hidden = tuple(int(x) for x in a.hidden.split(','))
    if a.preset == 'paper':
        nf = functools.partial(ppo_networks.make_ppo_networks,
                               policy_hidden_layer_sizes=hidden,
                               value_hidden_layer_sizes=hidden,
                               activation=act)
    else:
        nf = ppo_networks.make_ppo_networks      # Brax's own tuned defaults

    hist = {'steps': [], 'eval': [], 'wall': []}
    t0 = time.time()

    def progress(num_steps, metrics):
        r = float(metrics.get('eval/episode_reward', float('nan')))
        hist['steps'].append(int(num_steps))
        hist['eval'].append(r)
        hist['wall'].append(time.time() - t0)
        print("  steps {:>11,} eval {:9.2f}  ({:.0f}s)".format(int(num_steps), r,
                                                               time.time() - t0), flush=True)

    print("[{}] env={} preset={} hidden={} act={} budget={:,}".format(
        tag, a.env, a.preset, hidden, a.activation, a.max_steps), flush=True)

    _, params, metrics = ppo.train(
        environment=env,
        num_timesteps=a.max_steps,
        episode_length=a.episode_length,
        action_repeat=a.action_repeat,
        num_envs=a.num_envs,
        batch_size=a.batch_size,
        unroll_length=a.unroll_length,
        num_minibatches=a.num_minibatches,
        num_updates_per_batch=a.updates_per_batch,
        learning_rate=a.lr,
        entropy_cost=a.entropy_cost,
        discounting=a.discounting,
        normalize_observations=True,
        reward_scaling=1.0,
        num_evals=a.num_evals,
        seed=a.seed,
        network_factory=nf,
        progress_fn=progress,
    )

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'args': vars(a), 'hist': hist, 'wall_s': time.time() - t0,
               'final_eval': hist['eval'][-1]}, open(out, 'w'), indent=1)
    print("[{}] done in {:.0f}s -> {}".format(tag, time.time() - t0, out), flush=True)


if __name__ == '__main__':
    main()
