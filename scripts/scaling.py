"""Sec. 2.1 / 4.3: the parallelism and bandwidth claims, restated for a GPU.

Paper, Sec. 2.1:
    "ES thus requires extremely low bandwidth, in sharp contrast to policy gradient
     methods, which require workers to communicate entire gradients."

Paper, Sec. 4.3 / Fig. 1:
    "Solving 3D Humanoid with ES on one 18-core machine takes about 11 hours [...]
     However, when distributed across 80 machines and 1,440 CPU cores, ES can solve 3D
     Humanoid in just 10 minutes [...] Figure 1 shows that, for this task, ES is able to
     achieve linear speedup in the number of CPU cores."

We cannot rent 1,440 cores, and reporting a shrunken version of Fig. 1 would just be
reporting our own cluster's noise. Two things are measurable here instead, and both are
statements about the algorithm rather than about OpenAI's 2017 EC2 bill:

  (A) BANDWIDTH IS ARITHMETIC, so quote it exactly. Per iteration, ES broadcasts one
      int32 noise index and two float32 returns per antithetic pair; a data-parallel
      policy-gradient method all-reduces a full D-dimensional gradient per worker. The
      ratio is D-dependent and exact -- no experiment can disagree with it. We evaluate
      it at the paper's own scale (1,440 workers) and at its own network sizes.

  (B) PARALLEL SPEEDUP still has a physical meaning on a GPU: how does wall-clock per
      ES iteration grow as the population grows? Fig. 1's "linear speedup in the number
      of CPU cores" becomes "sublinear growth in time as the number of simultaneously
      evaluated perturbations grows" -- i.e. the region where extra population is
      nearly free. That region is the GPU's version of the paper's cluster scaling, and
      it ends where the device saturates, which is exactly the number worth reporting.

  (C) MULTI-DEVICE. ES splits a population across devices with literally zero
      communication during the rollout, so 2 T4s should give ~2x. We measure it.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax, jax.numpy as jnp
from brax import envs

from es.jaxcache import enable as _enable_jax_cache
_enable_jax_cache()

from es.policy import PolicySpec, init_flat
from es.rollout import make_pop_rollout
from es.noise import perturbation_descriptor_bytes, gradient_broadcast_bytes


def human(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return "{:.1f} {}".format(n, u)
        n /= 1024
    return "{:.1f} TB".format(n)


def bandwidth_table(out):
    """(A) Exact per-iteration communication, at the paper's own scale."""
    print("\n=== (A) BANDWIDTH PER ITERATION (Sec. 2.1) ===")
    print("Networks priced at the paper's own sizes. n = 1,440 workers (Sec. 4.3).")
    nets = [
        ("MuJoCo MLP 64-64 (Sec. 4.1), obs=17 act=6",
         PolicySpec(17, 6, (64, 64)).num_params),
        ("MuJoCo MLP 256-256 (humanoid.json), obs=244 act=17",
         PolicySpec(244, 17, (256, 256)).num_params),
        ("Atari CNN (Mnih et al. 2016 arch, approx)", 1_686_000),
    ]
    n_workers = 1440
    rows = []
    print("{:<52s} {:>10s} {:>12s} {:>14s} {:>10s}".format(
        "policy", "D", "ES bytes", "PG bytes", "ratio"))
    for name, D in nets:
        es_b = perturbation_descriptor_bytes(n_workers)
        pg_b = gradient_broadcast_bytes(n_workers, D)
        rows.append(dict(net=name, D=D, es_bytes=es_b, pg_bytes=pg_b, ratio=pg_b / es_b))
        print("{:<52s} {:>10,} {:>12s} {:>14s} {:>9.0f}x".format(
            name, D, human(es_b), human(pg_b), pg_b / es_b))
    print("\nES per-iteration traffic is independent of D: {} regardless of network size."
          .format(human(perturbation_descriptor_bytes(n_workers))))
    out['bandwidth'] = rows


def population_scaling(env_name, backend, T, out, pops):
    """(B) Wall-clock per ES iteration vs population size, on one GPU."""
    print("\n=== (B) POPULATION SCALING ON ONE T4 ({}, horizon {}) ===".format(env_name, T))
    print("{:>8s} {:>12s} {:>14s} {:>12s} {:>10s}".format(
        "P", "iter (s)", "env-steps/s", "s per 1k pop", "efficiency"))
    base = None
    rows = []
    for P in pops:
        env = envs.create(env_name, backend=backend, episode_length=T,
                          batch_size=P, auto_reset=False)
        spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size,
                          hidden=(64, 64), ac_noise_std=0.01)
        roll = make_pop_rollout(env, spec, T)
        th = jnp.tile(init_flat(jax.random.PRNGKey(0), spec)[None], (P, 1))
        om, os_ = jnp.zeros(spec.obs_dim), jnp.ones(spec.obs_dim)
        r = roll(th, jax.random.PRNGKey(0), om, os_); r['returns'].block_until_ready()
        reps = 3
        t = time.time()
        for _ in range(reps):
            r = roll(th, jax.random.PRNGKey(1), om, os_)
        r['returns'].block_until_ready()
        dt = (time.time() - t) / reps
        if base is None:
            base = dt / P
        eff = (base * P) / dt          # 1.0 => perfectly free extra population
        rows.append(dict(P=P, iter_s=dt, steps_per_s=P * T / dt, eff=eff))
        print("{:>8d} {:>12.3f} {:>14,.0f} {:>12.4f} {:>9.2f}x".format(
            P, dt, P * T / dt, dt / (P / 1000), eff))
    out['population_scaling'] = rows
    knee = max(rows, key=lambda r: r['steps_per_s'])
    print("\nPeak throughput at P={} ({:,.0f} env-steps/s). Below that, extra "
          "perturbations are close to free -- the GPU analogue of Fig. 1's linear "
          "speedup region.".format(knee['P'], knee['steps_per_s']))


def multi_device(env_name, backend, T, P, out):
    """(C) Population split across both T4s, zero inter-device communication."""
    devs = jax.devices()
    print("\n=== (C) MULTI-DEVICE SCALING ({} devices) ===".format(len(devs)))
    if len(devs) < 2:
        print("only one device visible; skipping"); return

    def timed(P_, dev):
        env = envs.create(env_name, backend=backend, episode_length=T,
                          batch_size=P_, auto_reset=False)
        spec = PolicySpec(obs_dim=env.observation_size, act_dim=env.action_size,
                          hidden=(64, 64), ac_noise_std=0.01)
        roll = make_pop_rollout(env, spec, T)
        th = jax.device_put(jnp.tile(init_flat(jax.random.PRNGKey(0), spec)[None], (P_, 1)), dev)
        om = jax.device_put(jnp.zeros(spec.obs_dim), dev)
        os_ = jax.device_put(jnp.ones(spec.obs_dim), dev)
        k = jax.device_put(jax.random.PRNGKey(0), dev)
        r = roll(th, k, om, os_); r['returns'].block_until_ready()
        return roll, th, k, om, os_

    r1 = timed(P, devs[0])
    t = time.time()
    o = r1[0](r1[1], r1[2], r1[3], r1[4]); o['returns'].block_until_ready()
    one = time.time() - t

    half = P // 2
    a0 = timed(half, devs[0]); a1 = timed(half, devs[1])
    t = time.time()
    o0 = a0[0](a0[1], a0[2], a0[3], a0[4])       # both dispatch asynchronously
    o1 = a1[0](a1[1], a1[2], a1[3], a1[4])
    o0['returns'].block_until_ready(); o1['returns'].block_until_ready()
    two = time.time() - t
    print("  P={} on 1 GPU : {:.3f}s".format(P, one))
    print("  P={} on 2 GPUs: {:.3f}s   speedup {:.2f}x".format(P, two, one / two))
    print("  (ES exchanges nothing between devices during a rollout; only {} of "
          "scalars are reduced afterwards)".format(human(perturbation_descriptor_bytes(P // 2))))
    out['multi_device'] = dict(P=P, one_gpu_s=one, two_gpu_s=two, speedup=one / two,
                               n_devices=len(devs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='hopper')
    p.add_argument('--backend', default='generalized')
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--pops', default='32,64,128,256,512,1024,2048,4096,8192')
    p.add_argument('--md-pop', type=int, default=4096)
    p.add_argument('--out', default='results/scaling.json')
    a = p.parse_args()

    out = {'args': vars(a), 'devices': [str(d) for d in jax.devices()]}
    bandwidth_table(out)
    population_scaling(a.env, a.backend, a.horizon, out,
                       [int(x) for x in a.pops.split(',')])
    multi_device(a.env, a.backend, a.horizon, a.md_pop, out)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print("\nwrote", a.out)


if __name__ == '__main__':
    main()
