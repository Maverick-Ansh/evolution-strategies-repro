"""Figures. Saves PNGs to figures/ -- never renders inline (notebook output goes back
through an MCP channel where a base64 PNG would swamp the tool result)."""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 8, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False})
ES_C, PG_C = '#c0392b', '#2471a3'


def load(pattern):
    runs = defaultdict(list)
    for f in glob.glob(pattern):
        try:
            runs[json.load(open(f))['args']['env']].append(json.load(open(f)))
        except Exception:
            pass
    return runs


def band(ax, runs, color, label):
    if not runs:
        return
    hi = min(max(r['hist']['steps']) for r in runs)
    grid = np.linspace(0, hi, 200)
    ys = np.stack([np.interp(grid, r['hist']['steps'], r['hist']['eval']) for r in runs])
    m, s = ys.mean(0), ys.std(0)
    ax.plot(grid, m, color=color, lw=1.4, label="{} (n={})".format(label, len(runs)))
    if len(runs) > 1:
        ax.fill_between(grid, m - s, m + s, color=color, alpha=0.18, lw=0)


def fig_curves(floors):
    es, pg = load('results/es_*.json'), load('results/ppo_*.json')
    envs = sorted(set(es) | set(pg))
    if not envs:
        return
    n = len(envs)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.0 * ((n + 1) // 2), 5.2))
    for ax, env in zip(np.ravel(axes), envs):
        band(ax, es.get(env, []), ES_C, 'ES')
        band(ax, pg.get(env, []), PG_C, 'PPO')
        f = floors.get(env, {})
        if f:
            ax.axhline(f['init'], color='0.35', ls='--', lw=0.8)
            ax.axhline(f['random'], color='0.65', ls=':', lw=0.8)
        ax.set_title(env, fontsize=8)
        ax.set_xlabel('env timesteps')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.legend(fontsize=6, frameon=False)
    for ax in np.ravel(axes)[n:]:
        ax.axis('off')
    np.ravel(axes)[0].set_ylabel('episode return')
    fig.suptitle('ES vs PPO on Brax (dashed = init-net floor, dotted = random floor)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/curves.png', bbox_inches='tight')
    print('figures/curves.png')


def fig_variance():
    p = 'results/variance_vs_T.json'
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    rows = d['rows']
    Ts = sorted(int(k) for k in rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    series = [('pg_g1.0', 'REINFORCE, no discount', PG_C, '-'),
              ('pg_g0.99', 'REINFORCE, gamma=0.99', PG_C, '--'),
              ('es_raw', 'ES (mean baseline)', ES_C, '-'),
              ('es_rank', 'ES (centered rank)', ES_C, '--')]
    lt = np.log(np.array(Ts, float))
    for k, lab, c, ls in series:
        y = np.array([rows[str(T)][k] for T in Ts], float)
        slope = np.polyfit(lt, np.log(y), 1)[0]
        a1.loglog(Ts, y, ls, color=c, marker='o', ms=3, lw=1.3,
                  label="{}  (k={:+.2f})".format(lab, slope))
    a1.set_xlabel('horizon T'); a1.set_ylabel('trace(Cov[g]) / ||E[g]||$^2$')
    a1.set_title('Gradient-estimator noise vs horizon\n(Sec. 3.1: PG ~ T, ES ~ const)',
                 fontsize=8)
    a1.legend(fontsize=6, frameon=False)

    for k, lab, c, ls in series:
        y = np.array([rows[str(T)][k] for T in Ts], float)
        a2.semilogx(Ts, y / y[0], ls, color=c, marker='o', ms=3, lw=1.3, label=lab)
    a2.set_xlabel('horizon T'); a2.set_ylabel('noise relative to T={}'.format(Ts[0]))
    a2.set_title('Same data, normalised at the shortest horizon', fontsize=8)
    a2.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig('figures/variance_vs_T.png', bbox_inches='tight')
    print('figures/variance_vs_T.png')


def fig_scaling():
    p = 'results/scaling.json'
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    rs = d.get('population_scaling', [])
    if not rs:
        return
    P = [r['P'] for r in rs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.loglog(P, [r['steps_per_s'] for r in rs], 'o-', color=ES_C, ms=3, lw=1.3)
    a1.set_xlabel('population size P (perturbations evaluated together)')
    a1.set_ylabel('env-steps / s')
    a1.set_title('GPU analogue of Fig. 1:\nthroughput vs parallel perturbations', fontsize=8)
    a2.semilogx(P, [r['eff'] for r in rs], 'o-', color=ES_C, ms=3, lw=1.3)
    a2.axhline(1.0, color='0.4', ls='--', lw=0.8)
    a2.set_xlabel('population size P'); a2.set_ylabel('parallel efficiency')
    a2.set_title('1.0 = extra perturbations are free', fontsize=8)
    fig.tight_layout(); fig.savefig('figures/scaling.png', bbox_inches='tight')
    print('figures/scaling.png')


def _final(tag_glob):
    out = defaultdict(list)
    for f in glob.glob(tag_glob):
        d = json.load(open(f))
        out[(d['args']['env'], d['args'].get('action_repeat', 1))].append(d['final_eval'])
    return out


def fig_frameskip(floors):
    es, pg = _final('results/fsES_*.json'), _final('results/fsPG_*.json')
    if not es and not pg:
        return
    envs = sorted({e for e, _ in list(es) + list(pg)})
    fig, axes = plt.subplots(1, max(1, len(envs)), figsize=(3.4 * max(1, len(envs)), 3.0),
                             squeeze=False)
    for ax, env in zip(axes[0], envs):
        ks = sorted({k for e, k in list(es) + list(pg) if e == env})
        for src, c, lab in [(es, ES_C, 'ES'), (pg, PG_C, 'PPO')]:
            m = [np.mean(src[(env, k)]) if src.get((env, k)) else np.nan for k in ks]
            s = [np.std(src[(env, k)]) if src.get((env, k)) else np.nan for k in ks]
            ax.errorbar(ks, m, yerr=s, marker='o', ms=4, lw=1.3, color=c, capsize=2, label=lab)
        f = floors.get(env, {})
        if f:
            ax.axhline(f['init'], color='0.35', ls='--', lw=0.8)
        ax.set_xticks(ks); ax.set_xlabel('action repeat (frame skip)')
        ax.set_ylabel('final return'); ax.set_title(env, fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle('Sec. 4.4: invariance to action frequency', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/frameskip.png', bbox_inches='tight')
    print('figures/frameskip.png')


def fig_delayed(floors):
    pairs = []
    for env in ['hopper', 'halfcheetah']:
        row = {}
        for pre, lab in [('es', 'ES dense'), ('ppo', 'PPO dense'),
                         ('dlES', 'ES delayed'), ('dlPG', 'PPO delayed')]:
            fs = glob.glob('results/{}_{}_s*.json'.format(pre, env))
            if fs:
                row[lab] = [json.load(open(f))['final_eval'] for f in fs]
        if row:
            pairs.append((env, row))
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 3.0), squeeze=False)
    for ax, (env, row) in zip(axes[0], pairs):
        labs = [l for l in ['ES dense', 'ES delayed', 'PPO dense', 'PPO delayed'] if l in row]
        xs = np.arange(len(labs))
        cols = [ES_C if l.startswith('ES') else PG_C for l in labs]
        hatch = ['' if 'dense' in l else '//' for l in labs]
        for x, l, c, h in zip(xs, labs, cols, hatch):
            ax.bar(x, np.mean(row[l]), yerr=np.std(row[l]), color=c, alpha=0.85,
                   hatch=h, edgecolor='w', capsize=3)
        f = floors.get(env, {})
        if f:
            ax.axhline(f['init'], color='0.35', ls='--', lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=20, ha='right', fontsize=6)
        ax.set_ylabel('final return'); ax.set_title(env, fontsize=8)
    fig.suptitle('Sec. 3.3: only the episode total survives (hatched = reward paid once, at the end)',
                 fontsize=9)
    fig.tight_layout(); fig.savefig('figures/delayed.png', bbox_inches='tight')
    print('figures/delayed.png')


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    floors = json.load(open('results/env_floor.json')) if os.path.exists('results/env_floor.json') else {}
    fig_curves(floors)
    fig_variance()
    fig_scaling()
    fig_frameskip(floors)
    fig_delayed(floors)
