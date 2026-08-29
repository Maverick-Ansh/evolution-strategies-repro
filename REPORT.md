# Reproducing *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*

Salimans, Ho, Chen, Sidor & Sutskever (OpenAI, 2017) — [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)

Reproduced on 2×Tesla T4 (Kaggle), GPU-vectorized Brax, JAX 0.7.2.
Code: [`Maverick-Ansh/evolution-strategies-repro`](https://github.com/Maverick-Ansh/evolution-strategies-repro)

> **Status: in progress.** Sections 1–3 and 6 are final. Results tables are being filled
> as the sweep completes; every unfilled cell is marked `PENDING` rather than estimated.

---

## 0. What this paper actually contains

Worth stating early, because it shaped the whole reproduction: **the paper has no
hyperparameter appendix.** Thirteen pages, three tables — Table 1 (MuJoCo sample-complexity
ratios), Table 2 (Atari final scores), Table 3 (Table 1 with the underlying timesteps).
There is no learning rate, no population size, no σ anywhere in the text.

Every hyperparameter used here therefore comes from OpenAI's released code
(`openai/evolution-strategies-starter`), vendored under [`docs/openai_starter/`](docs/openai_starter/)
so the reproduction is auditable. That immediately surfaced four things the paper says
differently, or does not say at all — see §3.

---

## 1. Claims, stated so they can be falsified

| # | Claim | Where | What would confirm it |
|---|---|---|---|
| **C1** | ES matches a tuned policy-gradient method's final MuJoCo score within ≤10× the samples on hard envs, and *beats* it (0.15–0.58×) on easy ones | Sec. 4.1, Tables 1 & 3 | ratio of ES timesteps to PG timesteps to reach 25/50/75/100 % of PG's learning progress |
| **C2** | ES is invariant to action frequency; PG is not | Sec. 4.4, Fig. 2 | spread of final score across `action_repeat ∈ {1,2,3,4}`: ES flat, PG degrading |
| **C3** | ES handles maximally sparse/delayed reward — "only the total return of an episode is used" | Sec. 3.3 | ES score unchanged when all reward is paid at the terminal step; PG collapses |
| **C4** | Scalar-only communication → linear speedup to 1,440 workers | Sec. 2.1, 4.3, Fig. 1 | bytes/iteration for ES vs gradient all-reduce; parallel efficiency vs population |
| **C5** | Var[∇F_PG] grows ≈ linearly in T; Var[∇F_ES] is independent of T | Sec. 3.1 | fitted exponent *k* in (normalised variance ∝ T^k): predict k≈1 for PG, k≈0 for ES |
| **C6** | Reparameterization (virtual batch norm / obs normalization, action discretization) is *necessary* — "without these methods ES proved brittle" | Key finding 1, Sec. 2.2 | ablation: ES with vs without obs-normalization and discretized actions |
| **C7** | Rank-based fitness shaping and antithetic sampling materially help | Sec. 2.1 | ablation against raw returns and non-mirrored sampling |

**C1, C2, C4** are headline claims (they appear in the abstract). **C3, C5, C6, C7** are
mechanism claims. C5 is the one the paper *asserts but never measures*, and it is the
load-bearing argument for everything else — so it gets the most direct instrument here.

The paper's own clean ablation is Table 1 itself: ES and TRPO are run on identical
network architectures, so the ratio isolates the optimizer.

---

## 2. How it was resized

The paper's compute model is **1,440 CPU cores**, one worker per perturbation, because
in 2017 MuJoCo ran on CPU. That structure — an embarrassingly parallel population whose
workers exchange only scalars — is unchanged today, but the hardware that cashes it in
is not. A GPU-resident batched simulator evaluates P perturbed policies as *one* batched
program:

```
paper:  P perturbations -> P CPU processes        -> all-gather P scalars
here:   P perturbations -> P lanes of one rollout -> reduce P scalars
```

This is the central resize: **the algorithm is untouched; only the parallel substrate
changes.** What is explicitly *not* reproduced is the paper's wall-clock (10 minutes to
solve 3D Humanoid) — that is a fact about an 80-machine cluster, and a shrunken version
of Fig. 1 would just be reporting our own noise. §5 measures what a single GPU *can*
honestly say about the parallelism claim instead.

### Deviations

| Paper | Here | Why it is still a test of the claim |
|---|---|---|
| MuJoCo (Todorov et al.) | Brax `generalized` backend | Brax's generalized-coordinate solver is the closest available analogue to MuJoCo's formulation. MJX (actual MuJoCo-on-GPU) is version-broken against Brax 0.12.1 (`Data.__init__() got an unexpected keyword argument 'contact'`), and the `positional`/`spring` backends do not support Swimmer. Absolute scores are *not* comparable to the paper; the claim under test is a **ratio measured within one simulator**. |
| TRPO baseline | PPO (+ SAC on hopper/walker2d) | TRPO's direct successor; Brax ships a *tuned* GPU-native implementation, so both arms share simulator and hardware. Ratios, not absolute scores, are compared. |
| — | SAC added on hopper/walker2d | Brax publishes **no tuned PPO config** for these two and uses SAC itself. Running only PPO would have handed ES a win on precisely the two rows where the paper reports its largest ES *penalty* (6.94×, 7.88×). |
| 6 random seeds | 2 seeds | Compute budget. Seed spread is reported; sub-noise deltas are not called results. |
| Humanoid MLP 256-256 (`humanoid.json`) | MLP 64-64 | Matches the paper's *text* (Sec. 4.1) for the Table-1 environments. The released humanoid config contradicts that text. |
| 5M timesteps for TRPO | 5M for PPO/SAC, 30M for ES | Same PG budget as the paper. ES capped at 30M, so ratios above ≈6× are **censored, reported as `>`**, never dropped. |
| `episodes_per_batch=10000`, `timesteps_per_batch=100000` | P=128 (n_pairs=64) | At full 1000-step episodes this is ~128k timesteps/iteration, matching the reference's cap. |
| `calc_obstat_prob=0.01` (subsampled obs stats) | all observations used | The 1 % subsample exists to save *network bandwidth* between workers; on one GPU there is no wire, so using every observation is strictly better information. |

---

## 3. What broke, and what the paper doesn't say

This is the section nobody writes, so it goes before the results.

### 3.1 The released code does not implement Algorithm 1

Algorithm 1, line 5 reads:

> θ_{t+1} ← θ_t + α (1 / (nσ)) Σ_i F_i ε_i

The released implementation (`es_distributed/es.py:242-249`) computes:

```python
g, count = batched_weighted_sum(proc_returns_n2[:, 0] - proc_returns_n2[:, 1], ...)
g /= returns_n2.size                     # divides by 2n, not n
update_ratio = optimizer.update(-g + config.l2coeff * theta)
```

Two differences: the **1/σ factor is absent entirely**, and the normaliser is `2n`
(episodes) rather than `n` (pairs). With Adam — which the paper never mentions but
`configurations/humanoid.json` specifies — the update is invariant to a global rescaling
of `g`, so this is harmless there. With plain SGD it is a **50× learning-rate difference**
at σ=0.02. Both forms are implemented (`--divide-by-sigma` recovers the paper's literal
Alg. 1); the code's version is the default, and `scripts/smoke.py` asserts the
relationship between them.

### 3.2 Three hyperparameters exist only in the code

- **Adam**, not SGD (`humanoid.json: "optimizer": {"type": "adam"}`). Algorithm 1 is
  written as plain gradient ascent.
- **Action noise on top of parameter noise**: `ac_noise_std = 0.01`. The paper's entire
  framing (Sec. 3) is parameter-space smoothing *versus* action-space smoothing; the
  implementation quietly does both.
- **Observation normalization with clipping** to ±5. Key finding 1 says reparameterization
  is what makes ES non-brittle, so this is load-bearing, not incidental.

### 3.3 The reference's rank shaping is not tie-safe

`compute_ranks` breaks ties by index order. On an all-tied population — zero information
about ε — `centered_rank` therefore returns weight −1/(2n−1) for *every* pair, and the
update is a systematic

> g = −mean(ε) / (2(2n−1))

rather than nothing. Harmless for continuous returns, but it means a task with a 0/1
reward and an all-zero population does **not** sit still: it random-walks. Asserted
with its closed form in `scripts/smoke.py` rather than asserted away.

### 3.4 An environment floor had to be measured before any result was trustworthy

My first ES run reported `eval 1000.00` on InvertedPendulum after 3 iterations, which
reads either as "solved instantly" or "environment is broken". `scripts/env_floor.py`
settled it by measuring three degenerate policies. Measured floors (Brax `generalized`,
1000-step episodes):

| env | zero action | uniform random | init network (ES's θ₀) | random ep. length |
|---|---|---|---|---|
| halfcheetah | 47.9 | −392.0 | 47.7 | 1000 (never terminates) |
| hopper | 117.5 | 16.4 | 116.3 | 21 |
| inverted_double_pendulum | 450.7 | 206.1 | 431.9 | 23 |
| inverted_pendulum | 25.2 | 5.2 | 24.1 | 5 |
| swimmer | 0.4 | 0.8 | 2.4 | 1000 (never terminates) |
| walker2d | 64.5 | 1.5 | 55.2 | 19 |

Two consequences:

1. The solve was real (24.1 → 1000), but at **172k samples per iteration** the curve had
   no resolution to measure a sample-complexity *ratio* with. That forced the paper's
   own dynamic episode cap (Sec. 2.1) to be wired into the static scan length, which
   also bought a ~30× speedup.
2. **HalfCheetah and Swimmer never terminate**, so "staying alive" is free and their
   reward is pure shaping. Their ES iterations cost a full P×1000 samples from the very
   first one, which is why those two rows behave differently from the other four.

The paper's own Table 3 quotes TRPO's InvertedPendulum score as exactly `1000.00` — the
episode cap — so that environment is saturating by construction in the original too.

### 3.5 The first PPO baseline was accidentally crippled

A single PPO config was used for all six environments. It happened to be Brax's tuned
**HalfCheetah** config, so Hopper PPO reached 182 at 1.3M steps while ES reached 1378 at
2M — an apparent 7× ES *win*, i.e. the exact inversion of the paper's Hopper row (6.94×
ES penalty). Fixed by lifting Brax's per-environment configs verbatim from
`google/brax/notebooks/training.ipynb`, and by adding the SAC arm where Brax provides no
PPO config at all. **Had this not been caught, the reproduction would have "confirmed"
the opposite of the paper on its two hardest environments.**

---

## 4. Results

### 4.1 C5 — the variance argument (Sec. 3.1). Confirmed for ES; PG is *worse* than the paper claims

This is the paper's load-bearing argument and it is stated without a single number. It is
also directly measurable: fix one policy, vary the horizon, estimate the sampling
distribution of each gradient estimator. No training, no critic, no learned probe.
HalfCheetah, because it never terminates early, so T is exactly the quantity being varied.

**trace(Cov[g]) — the paper's own quantity** (512 episodes per horizon, 32 estimates × 16 episodes):

| T | REINFORCE γ=1.0 | REINFORCE γ=0.99 | ES (mean baseline) | ES (centered rank) |
|---:|---:|---:|---:|---:|
| 8 | 593 | 544 | 9.86e5 | 9.45e4 |
| 16 | 4,812 | 4,091 | 9.42e5 | 8.71e4 |
| 32 | 3.23e4 | 2.36e4 | 8.84e5 | 8.34e4 |
| 64 | 1.71e5 | 9.31e4 | 9.35e5 | 8.98e4 |
| 128 | 8.51e5 | 2.84e5 | 9.15e5 | 8.80e4 |
| 256 | **3.40e6** | 6.22e5 | **9.09e5** | 8.24e4 |

Fitted exponents, quantity ∝ T^k:

| estimator | trace(Cov) | ‖E[g]‖² | normalised |
|---|---:|---:|---:|
| REINFORCE γ=1.0 | **+2.492** | +2.665 | −0.174 |
| REINFORCE γ=0.99 | +2.032 | +2.175 | −0.143 |
| ES (mean baseline) | **−0.018** | −0.017 | −0.001 |
| ES (centered rank) | −0.024 | −0.023 | −0.001 |

**Verdict on C5: confirmed for ES, and the policy-gradient side fails in the paper's favour.**

1. **ES is exactly horizon-independent** — k = −0.018, i.e. zero within sampling noise. This
   is not really an empirical finding; it is forced by the algebra, since
   ∇log p(θ̃;θ) = ε/σ contains no T at all. The measurement confirms the implementation
   has no hidden T-dependence.

2. **The paper's stated assumption is wrong here.** Sec. 3.1 argues the policy-gradient
   score is "a sum of T uncorrelated terms", which predicts k = +1. Measured k = **+2.49**.
   Uncorrelated terms cannot produce quadratic growth: the per-token score terms are
   *positively correlated* along a trajectory. The paper's conclusion therefore holds more
   strongly than its argument — ES's advantage at long horizons is larger than the stated
   reasoning implies, but for a reason the paper does not give.

3. **Discounting behaves as the paper says.** γ=0.99 drops the exponent from +2.49 to +2.03,
   consistent with *"the effective number of steps T is often reduced in policy gradient
   methods by discounting rewards"* — and it is a reduction, not a fix.

4. **The advantage is genuinely conditional, and the crossover is measurable.** At T=8
   REINFORCE's variance is ~1,660× *lower* than ES's; the curves cross at T≈128 and by
   T=256 ES is 3.7× lower. The paper's claim is explicitly conditional ("for long episodes
   with very many time steps") and that condition is quantitative: on this task, ES starts
   winning around a hundred steps.

5. **A complication the paper does not address, reported because it cuts against the
   headline.** ‖E[g]‖² grows as T^2.67 for REINFORCE — *faster* than its own variance — so
   the scale-free ratio trace(Cov)/‖E[g]‖² actually *improves* with horizon (k = −0.17),
   while ES's stays flat at ≈31. By that operational measure the crossover never happens in
   this range. Sec. 3.1's variance argument is about the numerator only, and on its own it
   does not establish that ES's *direction estimate* is more trustworthy at long horizons.
   Both metrics are reported in `results/variance_vs_T.json`.

> **Instrument bug, caught before it became a result.** The first version of this script
> reported only the normalised quantity, measured it *falling* as T^−0.20, and would have
> been written up as a refutation of Sec. 3.1. But the denominator grows with T too, so
> that number was never a test of the paper's claim. Raw and normalised are now reported
> separately (`scripts/variance_vs_T.py`).

## 5. The parallelism claim on one GPU (C4)

### 5.1 Bandwidth is arithmetic, so it is quoted exactly

Per iteration ES broadcasts one int32 noise index and two float32 returns per antithetic
pair. A data-parallel policy gradient must all-reduce a full D-dimensional gradient per
worker. At the paper's own scale — n = 1,440 workers (Sec. 4.3) — and its own networks:

| policy | D | ES / iteration | gradient all-reduce | ratio |
|---|---:|---:|---:|---:|
| MuJoCo MLP 64-64 (Sec. 4.1) | 5,702 | **16.9 KB** | 31.3 MB | **1,901×** |
| MuJoCo MLP 256-256 (`humanoid.json`) | 132,881 | **16.9 KB** | 729.9 MB | **44,294×** |
| Atari A3C-FF (Mnih et al. 2016) | 680,770 | **16.9 KB** | 3.7 GB | **226,923×** |

The point is the constant column: **ES's per-iteration traffic does not depend on the
network size at all.** This half of C4 is confirmed as exactly true, because it is a
property of the algorithm rather than of any cluster.

### 5.2 Fig. 1's "linear speedup", restated for a GPU

We cannot rent 1,440 cores, and a shrunken Fig. 1 would just be our own noise. The
measurable GPU analogue is: how does wall-clock per ES iteration grow as the population
grows? (Hopper, horizon 128, one T4.)

| population P | iteration (s) | env-steps/s | parallel efficiency |
|---:|---:|---:|---:|
| 32 | 0.477 | 8,583 | 1.00× |
| 64 | 0.443 | 18,482 | 2.15× |
| 128 | 0.393 | 41,640 | 4.85× |
| 256 | 0.418 | 78,320 | 9.13× |
| 512 | 0.460 | 142,547 | 16.61× |
| 1024 | 0.547 | 239,451 | 27.90× |
| 2048 | 0.778 | 336,745 | 39.24× |
| 4096 | 1.495 | 350,757 | 40.87× |

**128× more perturbations for 3.1× the wall-clock.** Iteration time is essentially flat
(0.39–0.48 s) from P=32 to P=512 — over that range extra population members are very nearly
free — and throughput saturates around P≈2048. This is the same structural fact Fig. 1
reports, cashed in on different hardware: ES converts parallel capacity into population at
near-zero marginal cost, whether that capacity is 1,440 cores or one GPU's worth of lanes.

### 5.3 A negative result on multi-device scaling

Splitting P=4096 across both T4s measured **0.88× — a slowdown**. ES exchanges nothing
between devices during a rollout, so this is not an algorithmic limit: dispatching both
device programs from one Python thread serialised them (0.778 + 0.778 ≈ 1.56 s, against
1.708 s measured). `scripts/scaling.py` now drives each device from its own thread and
reports both numbers. Treat this row as a measurement of a 4-CPU host driving two GPUs, not
as a property of ES.

## 6. What was *not* tested

- **Atari (Sec. 4.2, Table 2).** 51 games × 1B frames is far outside this budget. None of
  the Atari claims — including the virtual-batch-normalization result that is key finding
  1 — are tested here. The obs-normalization ablation (C6) is the MuJoCo analogue only.
- **3D Humanoid in 10 minutes (Sec. 4.3).** Requires the cluster the claim is about.
- **Linear speedup to 1,440 workers.** §5 measures parallel efficiency on 2 GPUs and the
  exact per-iteration byte counts; it does *not* and cannot verify 1,440-way scaling.
- **The exploration/gait-diversity claim (key finding 4).** "ES has been able to learn a
  very wide variety of gaits (such as walking sideways or walking backwards)" is a
  qualitative claim about Humanoid and is untested here.
- **CMA-ES and other NES variants.** Out of scope; the paper only positions against them.
- **Absolute score comparison to Tables 2 and 3.** Different simulator; only ratios are
  compared.
