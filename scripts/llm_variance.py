"""Sec. 3.1, measured on a 2026 LLM instead of a 2017 MuJoCo robot.

WHY THIS IS THE RIGHT MODERN TEST OF THIS PAPER.

Sec. 3.1 says ES beats policy gradients when three things hold at once:

    "Evolution strategies is thus an attractive choice if the effective number of time
     steps T is long, actions have long-lasting effects, and if no good value function
     estimates are available."

That is a precise description of contemporary LLM post-training with verifiable rewards:
the "episode" is a generated completion hundreds of tokens long, every token is an action
whose effect is felt only at the end, the reward is a single scalar from a verifier, and
GRPO -- the current default -- *removes the value function on purpose*. The 2017 paper
therefore makes a falsifiable prediction about 2026 RLVR, and it can be checked directly.

WHAT IS MEASURED. The paper's argument is about the second factor in

    Var[grad F_PG] ~ Var[R(a)] * Var[grad log p(a; theta)]
    Var[grad F_ES] ~ Var[R(a)] * Var[grad log p(theta~; theta)]

Var[R] is common to both, so the entire claim rests on the score terms:

    PG:  grad log p(a;theta) = sum_{t=1}^{T} grad log pi(a_t | s_<t)   -- claimed ~ T
    ES:  grad log p(theta~;theta) = eps / sigma                        -- exactly const in T

The ES side needs no experiment at all: eps ~ N(0, I) in D dimensions gives
trace(Cov[eps/sigma]) = D/sigma^2 for every T. It is T-independent *by construction*.
So the whole question is empirical only on the PG side, and it is measurable with
forward/backward passes alone -- no RL run, no reward model, no training.

We also test the paper's ASSUMPTION, not just its conclusion. Sec. 3.1 asserts the score
is "a sum of T uncorrelated terms". If the per-token terms were positively correlated the
growth would be T^2, not T; if they were anti-correlated it would be sublinear. Measuring
the realised exponent settles which regime an actual language model is in -- and that is
what decides whether ES is a sensible optimiser for LLM post-training.

Design note: the same sampled completions are reused across every prefix length, so the
T-series is PAIRED. That removes sampling noise between horizons, which matters because
the effect being measured is a growth rate.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def target_params(model, mode):
    """theta = the parameters both estimators would act on.

    LoRA in practice adapts the attention projections, so q_proj/v_proj is the honest
    stand-in for 'the parameters an ES or GRPO run would actually perturb'.
    """
    ps = []
    for n, p in model.named_parameters():
        if mode == 'attn' and ('q_proj.weight' in n or 'v_proj.weight' in n):
            ps.append((n, p))
        elif mode == 'lastlayer' and 'layers.23.' in n and p.ndim == 2:
            ps.append((n, p))
    for _, p in ps:
        p.requires_grad_(True)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--lengths', default='8,16,32,64,128,256')
    ap.add_argument('--n-seq', type=int, default=48)
    ap.add_argument('--target', default='attn', choices=['attn', 'lastlayer'])
    ap.add_argument('--sigma', type=float, default=0.02)   # ES noise_stdev, humanoid.json
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/llm_variance.json')
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    dev = 'cuda'
    # T4 is compute capability 7.5: fp16, never bf16.
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev)
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad_(False)
    ps = target_params(model, a.target)
    D = sum(p.numel() for _, p in ps)
    Ts = [int(x) for x in a.lengths.split(',')]
    Tmax = max(Ts)
    print("model={} |theta|={:,} over {} tensors  lengths={}".format(a.model, D, len(ps), Ts))

    # A verifiable-reward style prompt set: short arithmetic with a checkable answer.
    # The prompts only need to elicit realistic on-policy completions -- the score-function
    # variance being measured does not depend on the reward, which is exactly why this
    # measurement is cheap (Var[R] factors out of both estimators identically).
    rng = np.random.RandomState(a.seed)
    prompts = []
    for _ in range(a.n_seq):
        x, y = rng.randint(11, 99), rng.randint(11, 99)
        msg = [{"role": "user",
                "content": "What is {} + {}? Think step by step.".format(x, y)}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    # ---- sample completions once, reuse for every prefix length (paired design) ------
    print("sampling {} completions of {} tokens ...".format(a.n_seq, Tmax), flush=True)
    seqs = []
    t0 = time.time()
    for i in range(a.n_seq):
        enc = tok(prompts[i], return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=a.temperature,
                                 top_p=1.0, min_new_tokens=Tmax, max_new_tokens=Tmax,
                                 pad_token_id=tok.eos_token_id)
        seqs.append((enc['input_ids'].shape[1], out[0]))
        if (i + 1) % 12 == 0:
            print("  {}/{}  ({:.0f}s)".format(i + 1, a.n_seq, time.time() - t0), flush=True)

    # ---- trace(Cov) of the PG score, accumulated without storing per-seq gradients ---
    # trace(Cov[g]) = E||g||^2 - ||E g||^2, so a running sum of ||g_i||^2 (a scalar) plus
    # a running mean vector suffices. Storing 48 gradients of 22M floats would not fit.
    rows = {}
    for T in Ts:
        sum_g = [torch.zeros_like(p, dtype=torch.float32) for _, p in ps]
        sum_sq = 0.0
        for plen, seq in seqs:
            model.zero_grad(set_to_none=True)
            ids = seq[:plen + T].unsqueeze(0)
            out = model(ids)
            logits = out.logits[:, plen - 1:plen + T - 1, :].float() / a.temperature
            tgt = ids[:, plen:plen + T]
            logp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp.sum().backward()          # exactly sum_t grad log pi(a_t | s_<t)
            n2 = 0.0
            for j, (_, p) in enumerate(ps):
                g = p.grad.detach().float()
                sum_g[j] += g
                n2 += float(torch.sum(g * g))
            sum_sq += n2
        N = len(seqs)
        mean_sq = sum(float(torch.sum((s / N) ** 2)) for s in sum_g)
        tr_cov = sum_sq / N - mean_sq
        rows[T] = dict(trace_cov_pg=tr_cov, mean_norm_sq=mean_sq, N=N)
        print("  T={:4d}  trace(Cov[PG score]) = {:.4e}   ||E g||^2 = {:.4e}".format(
            T, tr_cov, mean_sq), flush=True)
        del sum_g

    # ---- ES side is analytic --------------------------------------------------------
    tr_es = D / (a.sigma ** 2)
    lt = np.log(np.array(Ts, float))
    k = float(np.polyfit(lt, np.log([rows[T]['trace_cov_pg'] for T in Ts]), 1)[0])

    print("\n=== VERDICT (Sec. 3.1) ===")
    print("PG score trace(Cov) ~ T^{:+.3f}".format(k))
    print("   paper's assumption 'a sum of T uncorrelated terms' predicts k = +1.00")
    print("   k > 1 => per-token score terms are POSITIVELY correlated (worse than the")
    print("            paper assumes, and better for ES)")
    print("   k < 1 => they partially cancel (better than the paper assumes)")
    print("ES score trace(Cov) = D/sigma^2 = {:.4e}, identical at every T (k = 0 exactly)"
          .format(tr_es))
    xover = None
    for T in Ts:
        if rows[T]['trace_cov_pg'] > tr_es:
            xover = T
            break
    print("crossover: PG score noise first exceeds ES's at T = {}".format(
        xover if xover else "> {} (not reached in this range)".format(max(Ts))))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'args': vars(a), 'D': D, 'rows': {str(k_): v for k_, v in rows.items()},
               'es_trace_cov': tr_es, 'pg_exponent_k': k, 'crossover_T': xover},
              open(a.out, 'w'), indent=1)
    print("\nwrote", a.out)


if __name__ == '__main__':
    main()
