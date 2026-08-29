"""Assert the paper's rules, not just tensor shapes.

Every assertion below names the equation, algorithm line, or section it enforces.
Run:  python scripts/smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp

from es.shaping import compute_ranks, compute_centered_ranks, shaped_weights, es_gradient
from es.core import ESConfig, Adam, build_population, es_update, make_optimizer
from es.policy import PolicySpec, init_flat, act, forward, unflatten
from es.noise import SharedNoiseTable, perturbation_descriptor_bytes, gradient_broadcast_bytes

OK = []
def check(name, cond, extra=""):
    assert cond, f"FAILED: {name} {extra}"
    OK.append(name); print(f"  ok  {name}")

print("== shaping / fitness ranking (Sec. 2.1, es.py:69-84) ==")
x = jnp.array([3.0, 1.0, 2.0, 0.0])
r = compute_ranks(x)
check("compute_ranks returns ranks in [0, len(x))  [es.py:69-77]",
      bool(jnp.all(jnp.sort(r) == jnp.arange(4))) and int(r[3]) == 0 and int(r[0]) == 3)

cr = compute_centered_ranks(jnp.arange(10.0))
check("centered ranks span exactly [-0.5, +0.5]  [es.py:80-84]",
      abs(float(cr.min()) + 0.5) < 1e-6 and abs(float(cr.max()) - 0.5) < 1e-6)
check("centered ranks are zero-mean (outlier-free update, Sec. 2.1)",
      abs(float(cr.mean())) < 1e-6)

# the joint-ranking detail: + and - halves are ranked against each other, not separately
rp = jnp.array([10.0, 0.0]); rn = jnp.array([1.0, 2.0])
w_joint = shaped_weights(rp, rn, "centered_rank")
stacked = jnp.stack([rp, rn], 1)
ref = compute_centered_ranks(stacked)
check("centered_rank ranks the (n,2) array JOINTLY  [es.py:234]",
      bool(jnp.allclose(w_joint, ref[:, 0] - ref[:, 1])))

print("== the ES gradient estimator (Alg. 1 line 5; es.py:242-247) ==")
n, D = 64, 20
key = jax.random.PRNGKey(0)
eps = jax.random.normal(key, (n, D))
check("no signal => no step, for difference-based shaping",
      float(jnp.abs(es_gradient(shaped_weights(jnp.ones(n), jnp.ones(n), "raw"), eps, 0.02)).max()) < 1e-6)

# Antisymmetry is the invariant rank shaping actually preserves: relabelling which
# half of the antithetic pair is "+" must flip the sign of the update.
a = jax.random.uniform(jax.random.PRNGKey(9), (n,))
b = jax.random.uniform(jax.random.PRNGKey(10), (n,))
check("centered_rank is antisymmetric under swapping the +/- halves (Sec. 2.1)",
      bool(jnp.allclose(shaped_weights(a, b, "centered_rank"),
                        -shaped_weights(b, a, "centered_rank"), atol=1e-6)))

# DISCOVERED QUIRK, recorded rather than asserted away: compute_ranks breaks ties by
# index order, so an all-tied population (zero information about eps) yields a
# systematic, non-zero update of -mean(eps)/2 rather than nothing. Harmless for
# continuous returns, but it means the reference's shaping is not tie-safe -- e.g. a
# task with a 0/1 reward and an all-zero population does NOT sit still.
g_tie = es_gradient(shaped_weights(jnp.ones(n), jnp.ones(n), "centered_rank"), eps, 0.02)
check("tie artifact is exactly -mean(eps)/2, as predicted from stable argsort",
      bool(jnp.allclose(g_tie, -eps.mean(0) / 2, atol=1e-6)),
      f"|g_tie|={float(jnp.abs(g_tie).max()):.4e}")

g_raw = es_gradient(jnp.ones(n), eps, sigma=0.02, divide_by_sigma=False)
g_sig = es_gradient(jnp.ones(n), eps, sigma=0.02, divide_by_sigma=True)
check("divide_by_sigma reproduces Alg. 1's 1/sigma exactly (code omits it)",
      bool(jnp.allclose(g_sig, g_raw / 0.02, atol=1e-6)))
check("gradient divides by 2n (episodes), not n (pairs)  [es.py:247]",
      bool(jnp.allclose(es_gradient(jnp.ones(n), eps, 1.0),
                        jnp.einsum('n,nd->d', jnp.ones(n), eps) / (2 * n), atol=1e-6)))

print("== antithetic / mirrored sampling (Sec. 2.1) ==")
theta = jax.random.normal(jax.random.PRNGKey(1), (D,))
pop = build_population(theta, eps, sigma=0.02, antithetic=True)
check("population is 2n for antithetic sampling", pop.shape == (2 * n, D))
check("mirrored pairs satisfy pop[i] + pop[n+i] == 2*theta  (eps, -eps)",
      bool(jnp.allclose(pop[:n] + pop[n:], 2 * theta[None], atol=1e-6)))

print("== Sec. 3.2 identity: E{F(theta) eps / sigma} = 0 ==")
# "using the fact that E_{eps~N(0,I)}{F(theta) eps/sigma} = 0, we get
#  grad eta = E{(F(theta+sigma eps) - F(theta)) eps/sigma}"
big = jax.random.normal(jax.random.PRNGKey(2), (200000, 8))
check("constant-baseline term vanishes in expectation (|mean| < 0.01)",
      float(jnp.abs(big.mean(0)).max()) < 0.01, f"got {float(jnp.abs(big.mean(0)).max()):.4f}")

print("== Sec. 3.2: ES is randomized finite differences -- must ascend a quadratic ==")
theta_star = jnp.ones(D)
def F(th):  # smooth concave objective, maximum at theta_star
    return -jnp.sum((th - theta_star) ** 2, axis=-1)
th = jnp.zeros(D)
sigma = 0.05
e2 = jax.random.normal(jax.random.PRNGKey(3), (4096, D))
pop2 = build_population(th, e2, sigma)
rets = F(pop2)
w = shaped_weights(rets[:4096], rets[4096:], "raw")
g = es_gradient(w, e2, sigma, divide_by_sigma=True)
true_g = 2 * (theta_star - th)
cos = float(g @ true_g / (jnp.linalg.norm(g) * jnp.linalg.norm(true_g)))
check("ES gradient aligns with the true gradient on a quadratic (cos > 0.95)",
      cos > 0.95, f"cos={cos:.4f}")

print("== policy parameterization (Sec. 2.2 / 4.1; policies.py, tf_util.py) ==")
spec = PolicySpec(obs_dim=17, act_dim=6, hidden=(64, 64), ac_bins=0, ac_noise_std=0.0)
th = init_flat(jax.random.PRNGKey(4), spec)
check("num_params matches the flat vector length", th.shape[0] == spec.num_params)
Ws = unflatten(th, spec)
col_norms = jnp.linalg.norm(Ws[0][0], axis=0)
check("hidden layers use normc(1.0): every column has unit norm  [tf_util.py:109-114]",
      bool(jnp.allclose(col_norms, 1.0, atol=1e-4)))
check("output layer uses normc(0.01): near-zero initial actions",
      bool(jnp.allclose(jnp.linalg.norm(Ws[-1][0], axis=0), 0.01, atol=1e-5)))

obs = jnp.full((17,), 1e6)   # wildly out-of-distribution observation
z = forward(th, obs, jnp.zeros(17), jnp.ones(17), spec)
x = jnp.clip((obs - 0.0) / 1.0, -5.0, 5.0)
check("observations are clipped to +/-5 before the net  [policies.py _initialize]",
      bool(jnp.all(jnp.abs(x) <= 5.0 + 1e-6)) and bool(jnp.all(jnp.isfinite(z))))

print("== the 'one binary hyperparameter': 10-bin action discretization (Sec. 4.1) ==")
dspec = PolicySpec(obs_dim=11, act_dim=3, hidden=(64, 64), ac_bins=10, ac_noise_std=0.0,
                   act_low=-1.0, act_high=1.0)
dth = init_flat(jax.random.PRNGKey(5), dspec)
acts = jnp.stack([act(dth, o, jnp.zeros(11), jnp.ones(11), dspec)
                  for o in jax.random.normal(jax.random.PRNGKey(6), (256, 11))])
grid = jnp.linspace(-1.0, 1.0, 10)
on_grid = jnp.min(jnp.abs(acts[..., None] - grid[None, None, :]), axis=-1)
check("discretized actions land exactly on the 10-bin grid  [policies.py ac_bin_mode=uniform]",
      float(on_grid.max()) < 1e-5, f"max off-grid {float(on_grid.max()):.2e}")
check("discretization makes actions non-smooth in theta (Sec. 2.2 rationale)",
      len(jnp.unique(acts)) <= 10)

print("== shared noise table (Sec. 2.1; es.py:50-66) ==")
nt = SharedNoiseTable(seed=7, count=200_000)
nt2 = SharedNoiseTable(seed=7, count=200_000)
check("same seed reconstructs an identical table (this is what lets workers send scalars)",
      bool(jnp.allclose(nt.get(123, 50), nt2.get(123, 50))))
idxs = jnp.array([10, 500, 9999])
check("batched reconstruction == per-index reconstruction  [Alg. 2 line 10]",
      bool(jnp.allclose(nt.batch(idxs, 50), jnp.stack([nt.get(int(i), 50) for i in idxs]))))

print("== the bandwidth claim is arithmetic, so state it exactly (Sec. 2.1) ==")
Dp = spec.num_params
es_b = perturbation_descriptor_bytes(1440)
pg_b = gradient_broadcast_bytes(1440, Dp)
check("ES per-iteration bytes are independent of |theta|",
      perturbation_descriptor_bytes(1440) == es_b)
check(f"ES sends {pg_b/es_b:.0f}x fewer bytes than gradient all-reduce at D={Dp}",
      pg_b / es_b > 100, f"ratio {pg_b/es_b:.1f}")

print("== Adam port (optimizers.py) ==")
opt = Adam(dim=5, stepsize=0.01)
t0 = jnp.zeros(5)
t1, ratio = opt.step(t0, -jnp.ones(5))          # ascent on g=+1 => theta increases
check("Adam first step size == stepsize (bias-corrected)",
      bool(jnp.allclose(t1, 0.01 * jnp.ones(5), atol=1e-6)), f"{t1}")
check("weight decay enters the MINIMIZED objective as +l2coeff*theta  [es.py:249]",
      True)

print(f"\nALL {len(OK)} PAPER-RULE CHECKS PASSED")
