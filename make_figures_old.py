"""
make_figures.py — publication figures at 300 dpi.
───────────────────────────────────────────────────────────────────────
Reads what run_all.py produced and draws every figure the paper needs:

  fig1_training_curves.png  HR@10 and loss vs epoch, per dataset
                            -> "does the model train stably?"
  fig2_tradeoff.png         DP vs HR and EO vs NDCG at K=10..40
                            -> same layout as FairIR's Fig. 4, so a
                               reviewer can overlay them mentally
  fig3_metrics_by_k.png     every metric as a function of K
  fig4_js_divergence.png    JS@20 / JS@50 bars vs FairIR's Table 7

Inputs (missing ones are skipped, not fatal):
  results/curves/{Model}_{dataset}.csv
  results/final/paper_table_{dataset}.json

USAGE:  python make_figures.py
Output: results/figures/*.png  (+ .pdf for LaTeX)
"""

import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

_ap = argparse.ArgumentParser()
_ap.add_argument('--run', default=None,
                 help='runs/<name> produced by run_all.py; '
                      'omit to use the newest, or pass "legacy" for results/')
_a = _ap.parse_args()

if _a.run == 'legacy':
    ROOT = 'results'
elif _a.run:
    ROOT = os.path.join('runs', _a.run)
else:
    _cand = sorted(glob.glob('runs/run_*'))
    ROOT = _cand[-1] if _cand else 'results'
print(f"Reading from: {ROOT}")

CURVES = os.path.join(ROOT, 'curves')
FINAL  = os.path.join(ROOT, 'final')
OUT    = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

COLOR = {'BPR': '#898781', 'SASRec': '#2a78d6', 'FairSASRec': '#eb6834',
         'LightGCN': '#1baf7a'}
MARK  = {'BPR': 's', 'SASRec': 'o', 'FairSASRec': 'D', 'LightGCN': '^'}
FAIRIR = {'HR': 0.2833, 'NDCG': 0.2777, 'DP': 0.5741, 'EO': 0.6258,
          'JS20': 0.1301, 'JS50': 0.0968}


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f"  {OUT}/{name}.png")


def load_tables():
    out = {}
    for p in glob.glob(os.path.join(FINAL, 'paper_table_*.json')):
        ds = os.path.basename(p).replace('paper_table_', '').replace('.json', '')
        out[ds] = json.load(open(p))
    return out


# ── fig 1: training curves ────────────────────────────────────────────
def fig_curves():
    files = sorted(glob.glob(os.path.join(CURVES, '*.csv')))
    if not files:
        print("  (no curves — skipping fig1)")
        return
    datasets = sorted({os.path.basename(f).split('_', 1)[1][:-4]
                       for f in files})
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.5*len(datasets), 6),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for f in files:
            base = os.path.basename(f)[:-4]
            model, fds = base.split('_', 1)
            if fds != ds:
                continue
            rows = list(csv_rows(f))
            if not rows:
                continue
            ep = [int(r['epoch']) for r in rows]
            hr = [float(r['hr10']) for r in rows]
            ls = [float(r['loss']) for r in rows]
            c, mk = COLOR.get(model, '#444'), MARK.get(model, 'o')
            axes[0][j].plot(ep, hr, marker=mk, ms=3, lw=1.6,
                            color=c, label=model)
            axes[1][j].plot(ep, ls, marker=mk, ms=3, lw=1.6,
                            color=c, label=model)
        axes[0][j].set_title(f'{ds} — probe HR@10')
        axes[0][j].set_xlabel('epoch'); axes[0][j].set_ylabel('HR@10')
        axes[0][j].legend()
        axes[1][j].set_title(f'{ds} — training loss')
        axes[1][j].set_xlabel('epoch'); axes[1][j].set_ylabel('loss')
        axes[1][j].legend()
    fig.tight_layout()
    save(fig, 'fig1_training_curves')


def csv_rows(path):
    import csv as _csv
    with open(path) as f:
        for r in _csv.DictReader(f):
            if r.get('hr10'):
                yield r


# ── fig 2: trade-off curves (FairIR Fig. 4 layout) ────────────────────
def fig_tradeoff(tables):
    for ds, res in tables.items():
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for model, r in res.items():
            ks = sorted(int(k) for k in r['fair'])
            hr = [r['acc'][str(k)]['HR'] for k in ks if str(k) in r['acc']]
            nd = [r['acc'][str(k)]['NDCG'] for k in ks if str(k) in r['acc']]
            dp = [r['fair'][str(k)]['DP'] for k in ks if str(k) in r['acc']]
            eo = [r['fair'][str(k)]['EO'] for k in ks if str(k) in r['acc']]
            c, mk = COLOR.get(model, '#444'), MARK.get(model, 'o')
            ax[0].plot(hr, dp, marker=mk, ms=5, lw=1.6, color=c, label=model)
            ax[1].plot(nd, eo, marker=mk, ms=5, lw=1.6, color=c, label=model)
            for x, y, k in zip(hr, dp, ks):
                ax[0].annotate(f'{k}', (x, y), fontsize=6,
                               xytext=(2, 3), textcoords='offset points')
        ax[0].axhline(FAIRIR['DP'], ls='--', lw=1, color='#d03b3b')
        ax[0].text(0.02, FAIRIR['DP'], ' FairIR best DP', fontsize=7,
                   color='#d03b3b', va='bottom', transform=ax[0].get_yaxis_transform())
        ax[1].axhline(FAIRIR['EO'], ls='--', lw=1, color='#d03b3b')
        ax[0].set_xlabel('HR@K (higher better)'); ax[0].set_ylabel('DP@K (lower better)')
        ax[1].set_xlabel('NDCG@K (higher better)'); ax[1].set_ylabel('EO@K (lower better)')
        ax[0].set_title(f'{ds}: accuracy–fairness trade-off')
        ax[1].set_title(f'{ds}: NDCG vs EO')
        ax[0].legend(); ax[1].legend()
        fig.tight_layout()
        save(fig, f'fig2_tradeoff_{ds}')


# ── fig 3: every metric vs K ──────────────────────────────────────────
def fig_by_k(tables):
    for ds, res in tables.items():
        metrics = [('HR', 'acc'), ('NDCG', 'acc'), ('DP', 'fair'),
                   ('EO', 'fair'), ('Gini', 'fair'), ('Cov', 'fair')]
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, (name, grp) in zip(axes.ravel(), metrics):
            for model, r in res.items():
                ks = sorted(int(k) for k in r[grp] if name in r[grp][k])
                ys = [r[grp][str(k)][name] for k in ks]
                if not ys:
                    continue
                ax.plot(ks, ys, marker=MARK.get(model, 'o'), ms=4, lw=1.6,
                        color=COLOR.get(model, '#444'), label=model)
            arrow = '↑' if name in ('HR', 'NDCG', 'Cov') else '↓'
            ax.set_title(f'{name}@K {arrow}')
            ax.set_xlabel('K')
        axes[0][0].legend()
        fig.suptitle(f'{ds}: all metrics across K', y=1.01)
        fig.tight_layout()
        save(fig, f'fig3_metrics_by_k_{ds}')


# ── fig 4: JS divergence bars ─────────────────────────────────────────
def fig_js(tables):
    for ds, res in tables.items():
        models = list(res.keys())
        js20 = [res[m]['fair'].get('20', {}).get('JS', np.nan) for m in models]
        js50 = [res[m]['fair'].get('50', {}).get('JS', np.nan) for m in models]
        x = np.arange(len(models)); w = 0.36
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(x - w/2, js20, w, label='JS@20',
               color=[COLOR.get(m, '#444') for m in models])
        ax.bar(x + w/2, js50, w, label='JS@50',
               color=[COLOR.get(m, '#444') for m in models], alpha=0.55)
        ax.axhline(FAIRIR['JS20'], ls='--', lw=1, color='#d03b3b')
        ax.text(len(models)-0.5, FAIRIR['JS20'], ' FairIR JS@20',
                fontsize=7, color='#d03b3b', va='bottom', ha='right')
        for i, (a, b) in enumerate(zip(js20, js50)):
            if not np.isnan(a): ax.text(i - w/2, a, f'{a:.3f}', ha='center',
                                        va='bottom', fontsize=7)
            if not np.isnan(b): ax.text(i + w/2, b, f'{b:.3f}', ha='center',
                                        va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel('JS divergence (lower = more balanced)')
        ax.set_title(f'{ds}: gender exposure divergence')
        ax.legend()
        fig.tight_layout()
        save(fig, f'fig4_js_divergence_{ds}')


print("Generating figures...")
tables = load_tables()
if not tables:
    print(f"  no paper_table_*.json in {FINAL} — run evaluate_final first")
fig_curves()
if tables:
    fig_tradeoff(tables); fig_by_k(tables); fig_js(tables)
print(f"\nDone. PNG for slides, PDF for LaTeX: {OUT}/")
