"""C6 / C7 ablations, plus the paper-vs-code discrepancy from REPORT.md 3.1.

Key finding 1 is an ablation claim and deserves to be read as one:
    "We found that the use of virtual batch normalization and other reparameterizations
     of the neural network policy (section 2.2) greatly improve the reliability of
     evolution strategies. Without these methods ES proved brittle in our experiments,
     but with these reparameterizations we achieved strong results over a wide variety
     of environments."

Every arm is scored against the measured init-network floor, not against zero, because
"ES did nothing" and "ES scored 116 on hopper" are the same event (env_floor.json).
An arm that lands within noise of its floor is reported as FLOOR, not as a small number.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

ARM_MEANING = {
    'base':      'full method (Sec. 2.1 + 2.2 as released)',
    'noobsnorm': 'C6: no observation normalisation (MuJoCo analogue of virtual batch norm)',
    'contact':   'C6: continuous actions where the paper discretised (Sec. 4.1)',
    'discact':   'C6: 10-bin discretised actions where the paper kept them continuous',
    'rawret':    'C7: raw returns, no rank-based fitness shaping',
    'noanti':    'C7: no antithetic / mirrored sampling',
    'alg1':      'Alg. 1 taken literally: plain SGD + the 1/sigma the released code drops',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--floor', default='results/env_floor.json')
    p.add_argument('--out', default='results/ablation.json')
    a = p.parse_args()

    floors = json.load(open(a.floor)) if os.path.exists(a.floor) else {}
    runs = defaultdict(list)
    for f in glob.glob('results/ab*_*.json'):
        base = os.path.basename(f)[2:-5]           # ab<arm>_<env>_s<seed>
        arm, env = base.split('_')[0], '_'.join(base.split('_')[1:-1])
        try:
            runs[(env, arm)].append(json.load(open(f)))
        except Exception:
            pass
    if not runs:
        print("no ablation results yet")
        return

    envs = sorted({e for e, _ in runs})
    out = {}
    for env in envs:
        fl = floors.get(env, {})
        floor = fl.get('init', 0.0)
        base_runs = runs.get((env, 'base'), [])
        base_m = np.mean([r['final_eval'] for r in base_runs]) if base_runs else float('nan')
        print("\n=== {} ===  init-net floor {:.1f} | full method {:.1f}".format(
            env, floor, base_m))
        print("{:<12s} {:>10s} {:>10s} {:>12s}  {}".format(
            "arm", "final", "above floor", "% of full", "what it removes"))
        out[env] = {'floor': floor, 'base': base_m, 'arms': {}}
        for arm in ['base', 'noobsnorm', 'contact', 'discact', 'rawret', 'noanti', 'alg1']:
            rs = runs.get((env, arm))
            if not rs:
                continue
            vals = [r['final_eval'] for r in rs]
            m, s = float(np.mean(vals)), float(np.std(vals))
            lift = m - floor
            base_lift = base_m - floor
            frac = 100.0 * lift / base_lift if base_lift > 1e-9 else float('nan')
            # "did this arm learn anything at all?" -- one seed-sigma above the floor
            tag = "FLOOR" if lift <= max(s, 1e-6) else "{:8.1f}%".format(frac)
            out[env]['arms'][arm] = dict(final=m, std=s, above_floor=lift,
                                         pct_of_full=frac, n=len(vals))
            print("{:<12s} {:>10.1f} {:>10.1f} {:>12s}  {}".format(
                arm, m, lift, tag, ARM_MEANING.get(arm, '')))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print("\nwrote", a.out)


if __name__ == '__main__':
    main()
