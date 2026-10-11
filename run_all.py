"""
run_all.py — one command, every model, both datasets.
───────────────────────────────────────────────────────────────────────
Trains BPR / SASRec / FairSASRec on ML-1M and ML-100K, captures the
per-epoch training curves, then runs the unified evaluation for each
dataset. Everything lands under results/ with dataset-tagged names, so
nothing overwrites anything.

ONE-LINE PATCH REQUIRED IN EACH TRAINER
───────────────────────────────────────
This script passes the dataset through environment variables. In
baselines/bpr_baseline.py, baselines/sasrec_baseline_200.py and
baselines/fairsasrec.py, replace the hard-coded path:

    DATA_DIR = 'data/ml-1m'
with:
    DATA_DIR = os.environ.get('DATA_DIR', 'data/ml-1m')

and in the two torch trainers replace the output directory:

    OUT_DIR = 'results/SASRec-200'          ->  os.environ.get('OUT_DIR', ...)
    'results/FairSASRec-ml1m'               ->  os.environ.get('OUT_DIR', ...)

Nothing else changes.

USAGE
─────
    python run_all.py                       # everything (~3 h on a T4)
    python run_all.py --datasets ml-1m      # one dataset
    python run_all.py --models sasrec fair  # skip BPR
    python run_all.py --skip-train          # evaluation + curves only
"""

import os, sys, csv, json, time, argparse, subprocess, re

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='+', default=['ml-1m', 'ml-100k'])
ap.add_argument('--models',   nargs='+', default=['bpr', 'sasrec', 'fair'],
                choices=['bpr', 'sasrec', 'fair'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--fairness-lambda', type=float, default=0.3)
ap.add_argument('--skip-train', action='store_true')
ap.add_argument('--run-name', default=None,
                help='folder under runs/ (default: timestamp)')
args = ap.parse_args()

# ── every artefact of this pipeline lives under one isolated folder ───
RUN = args.run_name or time.strftime('run_%Y%m%d_%H%M%S')
ROOT = os.path.join('runs', RUN)
for sub in ('models', 'curves', 'logs', 'final', 'figures'):
    os.makedirs(os.path.join(ROOT, sub), exist_ok=True)
print(f"\n>>> RUN DIRECTORY: {ROOT}\n")

TRAINERS = {
    'bpr':    'baselines/bpr_baseline.py',
    'sasrec': 'baselines/sasrec_baseline_200.py',
    'fair':   'baselines/fairsasrec.py',
}
LABEL = {'bpr': 'BPR', 'sasrec': 'SASRec', 'fair': 'FairSASRec'}


# "Epoch  10 | Loss: 0.88 (...) | HR@10: 0.53 | NDCG@10: 0.30"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Loss:?\s*([\d.]+).*?HR@10:?\s*([\d.]+)"
    r"(?:.*?NDCG@10:?\s*([\d.]+))?")


def run_trainer(model, ds):
    """Run one trainer, stream its output, save log + parsed curve."""
    out_dir = os.path.join(ROOT, "models", f"{LABEL[model]}-{ds}")
    env = dict(os.environ,
               DATA_DIR=f"data/{ds}",
               OUT_DIR=out_dir,
               SEED=str(args.seed))
    cmd = [sys.executable, TRAINERS[model]]
    if model == 'fair':
        cmd += ['--fairness_lambda', str(args.fairness_lambda)]

    print(f"\n{'='*66}\n  TRAIN  {LABEL[model]}  on  {ds}\n{'='*66}")
    t0 = time.time()
    lines, curve = [], []
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for ln in p.stdout:
        print(ln, end='')
        lines.append(ln)
        m = EPOCH_RE.search(ln)
        if m:
            curve.append({
                'epoch': int(m.group(1)),
                'loss':  float(m.group(2)),
                'hr10':  float(m.group(3)),
                'ndcg10': float(m.group(4)) if m.group(4) else '',
            })
    p.wait()
    mins = (time.time() - t0) / 60

    with open(os.path.join(ROOT, "logs", f"{LABEL[model]}_{ds}.log"), 'w') as f:
        f.writelines(lines)

    if curve:
        path = os.path.join(ROOT, "curves", f"{LABEL[model]}_{ds}.csv")
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss',
                                              'hr10', 'ndcg10'])
            w.writeheader(); w.writerows(curve)
        print(f"  curve -> {path}  ({len(curve)} points)")

    status = 'ok' if p.returncode == 0 else f'FAILED({p.returncode})'
    print(f"  {status} in {mins:.1f} min")
    return p.returncode == 0


def run_eval(ds):
    print(f"\n{'='*66}\n  EVALUATE  {ds}\n{'='*66}")
    cmd = [sys.executable, 'tests/evaluate_final.py', '--dataset', ds,
           '--out-dir', os.path.join(ROOT, 'final')]
    # point at the checkpoints this pipeline produced
    sas  = os.path.join(ROOT, 'models', f'SASRec-{ds}', 'sasrec_best.pt')
    fair = os.path.join(ROOT, 'models', f'FairSASRec-{ds}',
                        'fairsasrec_best.pt')
    if os.path.exists(sas):
        cmd += ['--sasrec-ckpt', sas]
    if os.path.exists(fair):
        cmd += ['--fair-ckpt', fair]
    r = subprocess.run(cmd)
    return r.returncode == 0


summary = []
for ds in args.datasets:
    if not os.path.isdir(f"data/{ds}"):
        print(f"!! data/{ds} not found — skipping dataset")
        continue
    if not args.skip_train:
        for m in args.models:
            if not os.path.exists(TRAINERS[m]):
                print(f"!! {TRAINERS[m]} not found — skipping")
                continue
            ok = run_trainer(m, ds)
            summary.append((ds, LABEL[m], 'trained' if ok else 'failed'))
    ok = run_eval(ds)
    summary.append((ds, 'evaluation', 'ok' if ok else 'failed'))

print(f"\n{'='*66}\n  PIPELINE SUMMARY\n{'='*66}")
for ds, what, st in summary:
    print(f"  {ds:<10} {what:<14} {st}")
# manifest so a run is self-describing months later
json.dump({'run': RUN, 'time': time.strftime('%F %T'),
           'datasets': args.datasets, 'models': args.models,
           'seed': args.seed, 'fairness_lambda': args.fairness_lambda,
           'summary': summary},
          open(os.path.join(ROOT, 'manifest.json'), 'w'), indent=2)

print(f"\nAll artefacts: {ROOT}/")
print(f"  models/  curves/  logs/  final/  figures/  manifest.json")
print(f"\nNext:  python make_figures.py --run {RUN}")
