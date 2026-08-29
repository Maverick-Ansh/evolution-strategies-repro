"""Reproduce the paper's Table 1 / Table 3: ES-vs-PG sample-complexity ratios.

Table 1 caption:
    "MuJoCo tasks: Ratio of ES timesteps to TRPO timesteps needed to reach various
     percentages of TRPO's learning progress at 5 million timesteps."

"Percentage of learning progress" is not a percentage of the final score -- Table 3
makes the definition recoverable. For HalfCheetah it lists 25% -> -1.35 and
100% -> 2385.79; a quarter of 2385.79 is 596, not -1.35. The numbers only line up if
progress is interpolated between the score at timestep zero and the final score:

    target(p) = init + p * (final_PG - init)

Check on Hopper: 100% = 3403.46 and 25% = 877.45 imply init = 35.4, which then predicts
50% = 1719.4 against the paper's listed 1718.16. So `init` is the random-policy score,
which is why scripts/env_floor.py measures it explicitly rather than reading it off the
first noisy point of a learning curve.

Censoring is reported, never hidden: if an arm never reaches a target inside its budget
the cell reads ">N.NN" and the true ratio is larger. Dropping those rows would silently
bias every average toward the environments where ES happened to win.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

TABLE1_PAPER = {   # the paper's own Table 1, for side-by-side reporting
    'halfcheetah':              {25: 0.15, 50: 0.49, 75: 0.42, 100: 0.58},
    'hopper':                   {25: 0.53, 50: 3.64, 75: 6.05, 100: 6.94},
    'inverted_double_pendulum': {25: 0.46, 50: 0.48, 75: 0.49, 100: 1.23},
    'inverted_pendulum':        {25: 0.28, 50: 0.52, 75: 0.78, 100: 0.88},
    'swimmer':                  {25: 0.56, 50: 0.47, 75: 0.53, 100: 0.30},
    'walker2d':                 {25: 0.41, 50: 5.69, 75: 8.02, 100: 7.88},
}
PCTS = [25, 50, 75, 100]


def first_crossing(steps, evals, target):
    """Timesteps at which the curve first reaches `target`. None if it never does.

    Uses the running maximum: 'timesteps needed to reach X' is a statement about having
    got there, and a single noisy eval dipping back below should not un-reach it.
    """
    run = np.maximum.accumulate(np.asarray(evals, dtype=float))
    idx = np.argmax(run >= target)
    if run[idx] < target:
        return None
    return float(np.asarray(steps, dtype=float)[idx])


def load(pattern):
    runs = defaultdict(list)
    for f in glob.glob(pattern):
        d = json.load(open(f))
        runs[d['args']['env']].append(d)
    return runs


def curve_mean(runs):
    """Average eval curves across seeds on a common timestep grid."""
    grid = np.linspace(0, min(max(r['hist']['steps']) for r in runs), 200)
    ys = []
    for r in runs:
        ys.append(np.interp(grid, r['hist']['steps'], r['hist']['eval']))
    return grid, np.mean(ys, axis=0), np.std(ys, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--es-glob', default='results/es_*.json')
    p.add_argument('--pg-glob', default='results/ppo_*.json')
    p.add_argument('--floor', default='results/env_floor.json')
    p.add_argument('--out', default='results/table1.json')
    a = p.parse_args()

    es_runs, pg_runs = load(a.es_glob), load(a.pg_glob)
    floors = json.load(open(a.floor)) if os.path.exists(a.floor) else {}

    print("Table 1 reproduction -- ratio of ES timesteps to PG timesteps")
    print("(paper's own values in brackets; '>' means the arm never reached the target "
          "inside its budget, so the true ratio is larger)\n")
    hdr = "{:<26s} {:>6s} " + " ".join(["{:>16s}"] * len(PCTS))
    print(hdr.format("env", "seeds", *["{}%".format(p_) for p_ in PCTS]))

    out = {}
    for env in sorted(set(es_runs) & set(pg_runs)):
        es, pg = es_runs[env], pg_runs[env]
        init = floors.get(env, {}).get('init')
        if init is None:
            init = min(min(r['hist']['eval']) for r in es + pg)

        g_pg, m_pg, _ = curve_mean(pg)
        g_es, m_es, _ = curve_mean(es)
        final_pg = float(np.max(np.maximum.accumulate(m_pg)))

        row, cells = {}, []
        for pct in PCTS:
            target = init + (pct / 100.0) * (final_pg - init)
            t_pg = first_crossing(g_pg, m_pg, target)
            t_es = first_crossing(g_es, m_es, target)
            paper = TABLE1_PAPER.get(env, {}).get(pct)
            if t_pg is None:
                cells.append("{:>16s}".format("pg n/a"))
                row[pct] = None
                continue
            if t_es is None:
                lo = float(g_es[-1]) / t_pg
                row[pct] = {'ratio': None, 'lower_bound': lo, 'target': target,
                            't_pg': t_pg, 'paper': paper}
                cells.append("{:>16s}".format(">{:.2f} [{}]".format(
                    lo, "-" if paper is None else "{:.2f}".format(paper))))
            else:
                ratio = t_es / t_pg
                row[pct] = {'ratio': ratio, 'target': target, 't_pg': t_pg,
                            't_es': t_es, 'paper': paper}
                cells.append("{:>16s}".format("{:.2f} [{}]".format(
                    ratio, "-" if paper is None else "{:.2f}".format(paper))))
        out[env] = {'init': init, 'final_pg': final_pg, 'n_es': len(es), 'n_pg': len(pg),
                    'pcts': row}
        print(("{:<26s} {:>6s} " + " ".join(["{:>16s}"] * len(PCTS))).format(
            env, "{}/{}".format(len(es), len(pg)), *cells))

    print("\nFinal scores (mean over seeds), bracketed by the measured floors:")
    print("{:<26s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "env", "init-net", "random", "ES", "PG"))
    for env in sorted(out):
        fl = floors.get(env, {})
        es_f = float(np.mean([r['final_eval'] for r in es_runs[env]]))
        pg_f = float(np.mean([r['final_eval'] for r in pg_runs[env]]))
        out[env]['es_final'], out[env]['pg_final'] = es_f, pg_f
        print("{:<26s} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            env, fl.get('init', float('nan')), fl.get('random', float('nan')), es_f, pg_f))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print("\nwrote", a.out)


if __name__ == '__main__':
    main()
