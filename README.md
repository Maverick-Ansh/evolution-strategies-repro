# Evolution Strategies, reproduced on one GPU

A from-scratch reproduction of **[Evolution Strategies as a Scalable Alternative to
Reinforcement Learning](https://arxiv.org/abs/1703.03864)** (Salimans, Ho, Chen, Sidor &
Sutskever — OpenAI, 2017), resized from the paper's 1,440-CPU cluster onto a single
GPU-vectorized simulator.

**Read [`REPORT.md`](REPORT.md) first** — it holds the claims, the deviations, what broke,
and the verdicts. This file is just the map.

---

## The one-line version of the resize

The paper's headline is that ES parallelizes across a cluster because its workers exchange
only scalars. That is a claim about the *structure* of the algorithm, and the structure has
not changed — only the hardware that can exploit it:

```
paper:  P perturbations -> P CPU processes        -> all-gather P scalars
here:   P perturbations -> P lanes of one rollout -> reduce P scalars
```

The algorithm is untouched. Brax evaluates the whole population as one batched program.

## Why this paper is worth re-running in 2026

Sec. 3.1 says ES beats policy gradients when the horizon is long, actions have long-lasting
effects, and no good value function is available. That is a description of modern LLM
post-training with verifiable rewards — hundreds of generated tokens, one scalar reward from
a verifier at the end, and GRPO deliberately removing the value function. So the paper makes
a falsifiable prediction about a setting that did not exist when it was written.
`scripts/llm_variance.py` tests it directly.

## Layout

```
es/
  shaping.py    fitness shaping + the ES gradient (ported from the released code,
                with the paper-vs-code differences documented in the docstring)
  noise.py      the shared noise table -- the mechanism behind the bandwidth claim
  policy.py     MujocoPolicy port: normc init, obs clipping, 10-bin action discretization
  core.py       Adam/SGD ports, RunningStat, the ES update
  rollout.py    GPU-vectorized population rollout + the Sec. 2.1 dynamic episode cap
  delayed.py    maximally-delayed-reward wrapper (Sec. 3.3 substrate)
scripts/
  smoke.py            25 assertions of the paper's *rules*, not tensor shapes
  env_floor.py        Phase-4 gate: degenerate-policy floors for every environment
  train_es.py         ES on Brax
  train_ppo.py        PPO baseline, Brax's tuned per-env configs
  train_sac.py        SAC baseline (Brax's own choice on hopper/walker2d)
  variance_vs_T.py    direct measurement of the Sec. 3.1 variance claim
  llm_variance.py     the same claim, measured on an LLM in the RLVR regime
  scaling.py          exact bandwidth arithmetic + GPU population scaling
  analyze_table1.py   reproduces Table 1's ratios (censored cells reported as ">")
  launch_sweep.py     detached, resumable sweep runner
  make_figures.py
docs/openai_starter/  vendored reference implementation, so every hyperparameter
                      claim in REPORT.md is auditable
```

## Reproducing

```bash
pip install "mujoco==3.3.5" "mujoco-mjx==3.3.5" && pip install --no-deps brax==0.12.1 jaxopt
python scripts/smoke.py                 # 25 paper-rule checks
python scripts/env_floor.py             # measure the floors BEFORE trusting any curve
python scripts/launch_sweep.py --which table1 --seeds 2
python scripts/analyze_table1.py
python scripts/make_figures.py
```

Hardware used: 2×Tesla T4 (Kaggle), 4 CPUs, JAX 0.7.2, Brax 0.12.1, `generalized` backend.

## A warning that cost me an hour

Brax publishes **no tuned PPO config for hopper or walker2d** — its own notebook uses SAC
for both. Running one PPO config across all six environments (Brax's HalfCheetah one) made
PPO look 7× *worse* than ES on Hopper, which is the exact inversion of the paper's result.
If you compare against a policy-gradient baseline here, tune it per environment or you will
reproduce the opposite of the paper. See REPORT.md §3.5.
