"""Generate ES_reproduction.ipynb from source.

The notebook is a BUILD ARTIFACT, never edited by hand: prose and code would otherwise
drift apart, and committed cell outputs would rot. Run:

    python nbsrc/build_notebook.py     # writes ES_reproduction.ipynb at the repo root
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/Maverick-Ansh/evolution-strategies-repro"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip().splitlines(keepends=True)})


md(r"""
# Evolution Strategies as a Scalable Alternative to RL — a reproduction

Salimans, Ho, Chen, Sidor & Sutskever (OpenAI, 2017) · [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)

This notebook reproduces the paper on a **single GPU** instead of the 1,440 CPU cores it
was written for. The algorithm is unchanged; only the parallel substrate is different:

```
paper:  P perturbations -> P CPU processes        -> all-gather P scalars
here:   P perturbations -> P lanes of one rollout -> reduce P scalars
```

**Runtime → Change runtime type → GPU** before running anything.

Full write-up, including the deviations table and everything that broke:
[`REPORT.md`](""" + REPO + r"""/blob/master/REPORT.md)
""")

md("## 1. Setup")

code(r"""
!pip -q install "mujoco==3.3.5" "mujoco-mjx==3.3.5"
!pip -q install --no-deps brax==0.12.1 jaxopt
!pip -q install etils[epath] ml_collections trimesh pytinyrenderer tensorboardX dm_env orbax-checkpoint
""")

code(r"""
import os
os.environ['MUJOCO_GL'] = 'egl'                      # MUST precede any mujoco import
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
!git clone -q """ + REPO + r""".git
%cd evolution-strategies-repro
import jax; print(jax.__version__, jax.devices())
""")

md(r"""
## 2. The paper's rules, as assertions

Before any training, `scripts/smoke.py` asserts 25 properties of the *paper*, not of the
tensors: that mirrored pairs satisfy `pop[i] + pop[n+i] == 2θ` (Sec. 2.1), that centered
ranks span exactly [−0.5, +0.5] (`es.py:80-84`), that discretized actions land on the
10-bin grid (Sec. 4.1), that the ES gradient ascends a quadratic (Sec. 3.2), and so on.

Two of these encode differences between the paper and the code OpenAI released:

* Algorithm 1 line 5 divides by `nσ`; the released code divides by `2n` and **drops
  1/σ entirely**. Harmless under Adam (which the paper never mentions), a 50× learning
  rate difference under SGD.
* `compute_ranks` breaks ties by index order, so a fully-tied population produces a
  systematic update of `−mean(ε)/(2(2n−1))` rather than none.
""")

code("!python scripts/smoke.py")

md(r"""
## 3. Measure the floor before trusting any curve

The failure this prevents: an environment where a degenerate policy already scores near
the top has no dynamic range, so "ES solved it" and "ES did nothing" produce the same
number. The paper's own Table 3 quotes TRPO's InvertedPendulum score as exactly
`1000.00` — the episode cap — so that environment saturates by construction.

This also reveals that **HalfCheetah and Swimmer never terminate**: a uniform-random
policy survives all 1000 steps, so "staying alive" is free and their reward is pure
shaping.
""")

code("!python scripts/env_floor.py --batch 128")

md(r"""
## 4. The bandwidth claim is arithmetic, so quote it exactly

Sec. 2.1: *"ES thus requires extremely low bandwidth, in sharp contrast to policy
gradient methods, which require workers to communicate entire gradients."*

Per iteration ES broadcasts one int32 noise index and two float32 returns per antithetic
pair — **independent of the network size**. A data-parallel policy gradient must
all-reduce a full D-dimensional gradient per worker. No experiment can disagree with the
ratio, so it is computed rather than measured.
""")

code("!python scripts/scaling.py --env hopper --horizon 128")

md(r"""
## 5. The claim the paper asserts but never measures

Sec. 3.1 is the load-bearing argument of the whole paper:

> *"grad log p(a; θ) = Σ_t grad log p(a_t; θ) is a sum of T uncorrelated terms, so that
> the variance of the policy gradient estimator will grow nearly linearly with T. The
> corresponding term for evolution strategies is independent of T."*

It is stated without a single number. It is also directly measurable: fix one policy,
vary the horizon, and estimate the sampling distribution of each gradient estimator.
No training, no critic, no learned probe.
""")

code("!python scripts/variance_vs_T.py --M 32 --B 16")

md(r"""
## 6. Training: ES vs a tuned policy gradient

**Warning worth repeating:** Brax publishes no tuned PPO config for hopper or walker2d —
its own notebook uses SAC for both. Running one PPO config across all six environments
made PPO look 7× *worse* than ES on Hopper, the exact inversion of the paper's result.
Per-environment configs are lifted verbatim from Brax's notebook; see REPORT.md §3.5.
""")

code(r"""
# a short demo run; the full sweep is scripts/launch_sweep.py --which table1
!python scripts/train_es.py  --env hopper --n-pairs 64 --max-steps 3000000 --tag demo_es
!python scripts/train_ppo.py --env hopper --max-steps 1000000 --tag demo_ppo
""")

md(r"""
## 7. The same argument, in the paradigm that exists now

Sec. 3.1 says ES wins when the horizon is long, actions have long-lasting effects, and
no good value function is available. That is a description of contemporary LLM
post-training with verifiable rewards: hundreds of generated tokens, one scalar reward
from a verifier at the end, and GRPO deliberately removing the value function.

So the 2017 paper makes a falsifiable prediction about a setting that did not exist when
it was written. The ES side needs no experiment — `ε/σ` is T-independent by construction,
with `trace(Cov) = D/σ²` exactly. The whole question is whether `Σ_t ∇log π_t` really
grows like T on a real language model, which also tests the paper's *assumption* that
the per-token terms are uncorrelated.
""")

code("!python scripts/llm_variance.py --n-seq 32 --lengths 8,16,32,64,128,256")

md(r"""
## 8. Results

`analyze_table1.py` reproduces Table 1's ratio-of-timesteps table. Cells where an arm
never reached the target inside its budget are printed as `>x.xx` — censored, never
dropped, because dropping them biases every average toward the environments where ES
happened to win.
""")

code(r"""
!python scripts/analyze_table1.py
!python scripts/analyze_ablation.py
!python scripts/make_figures.py
from IPython.display import Image, display
import glob
for f in sorted(glob.glob('figures/*.png')):
    print(f); display(Image(f))
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(ROOT, 'ES_reproduction.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
