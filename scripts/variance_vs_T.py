"""Direct measurement of the paper's central theoretical claim (Sec. 3.1).

The paper asserts, but never measures, this:

    "Here we have that grad_theta log p(a; theta) = sum_{t=1}^{T} grad_theta log p(a_t; theta)
     is a sum of T uncorrelated terms, so that the variance of the policy gradient
     estimator will grow nearly linearly with T. The corresponding term for evolution
     strategies, grad_theta log p(theta~; theta), is independent of T. Evolution
     strategies will thus have an advantage compared to policy gradients for long
     episodes with very many time steps."

and its own caveat:

    "In practice, the effective number of steps T is often reduced in policy gradient
     methods by discounting rewards. If the effects of actions are short-lasting, this
     allows us to dramatically reduce the variance in our gradient estimate [...]
     However, this discounting will bias our gradient estimate if actions have
     long-lasting effects."

This script measures both. It is probe-free: no learned critic, no auxiliary model, no
training run. We fix one policy theta, vary the horizon T, and directly estimate the
sampling distribution of each gradient estimator.

Metric: NORMALISED VARIANCE  =  trace(Cov[g]) / ||E[g]||^2.
This is scale-free, so it is comparable across estimators whose gradients live at
totally different magnitudes (ES ranks are O(1); REINFORCE scores are O(1/sigma_a^2)).
It answers the operational question: "how many independent estimates must I average
before the direction is trustworthy?"

Fairness: every estimator gets the SAME number of episodes per estimate (`--B`) and the
same environment, policy and horizon. HalfCheetah is used because it never terminates
early, so the horizon T is exactly the quantity being varied and nothing else changes.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from es.jaxcache import enable as _enable_jax_cache
_enable_jax_cache()

from es.policy import PolicySpec, init_flat, forward
from es.shaping import shaped_weights, es_gradient


def make_traj_rollout(env, spec, T, sigma_a):
    """Roll out a Gaussian policy and keep what REINFORCE needs: obs, actions, rewards."""
    def rollout(theta_pop, key):
        P = theta_pop.shape[0]
        rk, sk = jax.random.split(key)
        state = env.reset(rk)

        def body(carry, t):
            state, k = carry
            k, ak = jax.random.split(k)
            mu = jax.vmap(lambda th, ob: forward(th, ob, jnp.zeros(spec.obs_dim),
                                                 jnp.ones(spec.obs_dim), spec))(theta_pop, state.obs)
            a = mu + sigma_a * jax.random.normal(ak, mu.shape)
            a = jnp.clip(a, -1.0, 1.0)
            nxt = env.step(state, a)
            return (nxt, k), (state.obs, a, nxt.reward)

        (_, _), (obs, acts, rews) = jax.lax.scan(body, (state, sk), jnp.arange(T))
        return obs, acts, rews          # (T,P,obs), (T,P,act), (T,P)
    return jax.jit(rollout)


def reinforce_grads(theta, obs, acts, spec, sigma_a):
    """Per-trajectory grad_theta sum_t log pi(a_t|s_t) for a fixed-variance Gaussian policy.

    log pi = -||a - mu_theta(s)||^2 / (2 sigma_a^2) + const, so this is exactly the
    'sum of T uncorrelated terms' the paper's variance argument is about.
    """
    def logp_traj(th, ob_t, ac_t):
        mu = jax.vmap(lambda o: forward(th, o, jnp.zeros(spec.obs_dim),
                                        jnp.ones(spec.obs_dim), spec))(ob_t)
        return -jnp.sum((ac_t - mu) ** 2) / (2 * sigma_a ** 2)
    g = jax.grad(logp_traj)
    # vmap over trajectories: obs (T,P,obs) -> per-trajectory (P, T, obs)
    return jax.vmap(lambda o, a: g(theta, o, a))(jnp.swapaxes(obs, 0, 1),
                                                 jnp.swapaxes(acts, 0, 1))


def norm_var(estimates):
    """trace(Cov) / ||mean||^2 over a stack of (M, D) independent gradient estimates."""
    m = estimates.mean(0)
    return float(jnp.sum(jnp.var(estimates, axis=0)) / (jnp.sum(m ** 2) + 1e-30)), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='halfcheetah')   # never terminates early => clean T
    p.add_argument('--backend', default='generalized')
    p.add_argument('--horizons', default='8,16,32,64,128,256')
    p.add_argument('--M', type=int, default=64)      # independent estimates per horizon
    p.add_argument('--B', type=int, default=32)      # episodes per estimate (both arms)
    p.add_argument('--sigma-a', type=float, default=0.3)   # action-space noise (PG)
    p.add_argument('--sigma-p', type=float, default=0.02)  # parameter-space noise (ES)
    p.add_argument('--gammas', default='1.0,0.99')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/variance_vs_T.json')
    a = p.parse_args()

    Ts = [int(x) for x in a.horizons.split(',')]
    gammas = [float(x) for x in a.gammas.split(',')]
    P = a.M * a.B
    rows = {}

    print("env={} M={} B={} total episodes/horizon={}".format(a.env, a.M, a.B, P))
    print("{:>6s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
        "T", "PG(g=1.0)", "PG(g=0.99)", "ES(raw)", "ES(rank)"))

    for T in Ts:
        env = envs.create(a.env, backend=a.backend, episode_length=T,
                          batch_size=P, auto_reset=False)
        spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size,
                          hidden=(64, 64), ac_bins=0, ac_noise_std=0.0)
        theta = init_flat(jax.random.PRNGKey(a.seed), spec)
        D = spec.num_params
        roll = make_traj_rollout(env, spec, T, a.sigma_a)

        # ---- policy gradient arm: one theta, P stochastic trajectories -------------
        key = jax.random.PRNGKey(a.seed + 1)
        obs, acts, rews = roll(jnp.tile(theta[None], (P, 1)), key)
        glogp = reinforce_grads(theta, obs, acts, spec, a.sigma_a)      # (P, D)
        row = {}
        for gm in gammas:
            disc = (gm ** jnp.arange(T))[:, None]
            R = jnp.sum(rews * disc, axis=0)                            # (P,)
            R = R.reshape(a.M, a.B)
            gl = glogp.reshape(a.M, a.B, D)
            base = R.mean(axis=1, keepdims=True)                        # "a good baseline"
            est = jnp.einsum('mb,mbd->md', R - base, gl) / a.B          # (M, D)
            nv, mean = norm_var(est)
            row['pg_g{}'.format(gm)] = nv

        # ---- ES arm: P/2 antithetic pairs, same episode budget --------------------
        npairs = P // 2
        key2 = jax.random.PRNGKey(a.seed + 2)
        eps = jax.random.normal(key2, (npairs, D))
        pop = jnp.concatenate([theta[None] + a.sigma_p * eps,
                               theta[None] - a.sigma_p * eps], axis=0)
        _, _, rews_es = roll(pop, jax.random.PRNGKey(a.seed + 3))
        Res = jnp.sum(rews_es, axis=0)                                  # undiscounted, Sec. 3.3
        rp, rn = Res[:npairs], Res[npairs:]
        pairs_per_est = npairs // a.M
        for mode, label in [('centered', 'es_raw'), ('centered_rank', 'es_rank')]:
            ests = []
            for m in range(a.M):
                sl = slice(m * pairs_per_est, (m + 1) * pairs_per_est)
                w = shaped_weights(rp[sl], rn[sl], mode)
                ests.append(es_gradient(w, eps[sl], a.sigma_p, divide_by_sigma=True))
            nv, _ = norm_var(jnp.stack(ests))
            row[label] = nv

        rows[T] = row
        print("{:6d} {:14.3f} {:14.3f} {:14.3f} {:14.3f}".format(
            T, row['pg_g1.0'], row['pg_g0.99'], row['es_raw'], row['es_rank']))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'args': vars(a), 'rows': {str(k): v for k, v in rows.items()}},
              open(a.out, 'w'), indent=1)

    # ---- the actual verdict: fit normalised variance ~ T^k -----------------------
    print("\nSCALING EXPONENT  (normalised variance ~ T^k; Sec. 3.1 predicts k~1 for PG, k~0 for ES)")
    lt = np.log(np.array(Ts, dtype=float))
    for key_ in ['pg_g1.0', 'pg_g0.99', 'es_raw', 'es_rank']:
        y = np.log(np.array([rows[T][key_] for T in Ts]))
        k = np.polyfit(lt, y, 1)[0]
        print("  {:12s} k = {:+.3f}".format(key_, k))
    print("\nwrote", a.out)


if __name__ == '__main__':
    main()
