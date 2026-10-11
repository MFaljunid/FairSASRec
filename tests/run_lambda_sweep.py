"""
tests/run_lambda_sweep.py
───────────────────────────────────────────────────────────────────────
Principled lambda selection for FairSASRec.

WHAT THIS IS — AND IS NOT
─────────────────────────
It is NOT a copy of fairsasrec.py. It is a RUNNER that:

  1. calls fairsasrec.py once per lambda (subprocess, same seed),
  2. evaluates every resulting checkpoint on the VALIDATION set,
  3. picks the winning lambda by VALIDATION only,
  4. prints that choice so you can report the TEST numbers for it
     (via evaluate_final.py) — and nothing else.

Why step 2 matters: fairsasrec.py evaluates on TEST. Choosing lambda by
test performance is p-hacking — the reported number stops being an
honest estimate of generalisation. Selection must happen on data the
final number does not come from. This script enforces that separation.

WHAT YOU REPORT IN THE PAPER
  - Table (ablation): validation HR/DP for every lambda  <- from here
  - Table (main):     test metrics for the winning lambda <- from
                      evaluate_final.py, run once, afterwards

USAGE
─────
    python tests/run_lambda_sweep.py                    # full sweep
    python tests/run_lambda_sweep.py --skip-train       # re-score only
    python tests/run_lambda_sweep.py --lambdas 0.0 0.1 0.3

Expect ~40 min per lambda on a T4.
"""

import sys, os, json, argparse, subprocess, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

ap = argparse.ArgumentParser()
ap.add_argument('--lambdas', type=float, nargs='+',
                default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--trainer', default='baselines/fairsasrec.py')
ap.add_argument('--dataset', default='ml-1m', choices=['ml-1m','ml-100k'])
ap.add_argument('--skip-train', action='store_true')
args = ap.parse_args()

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS       = args.dataset
DATA_DIR = f'data/{DS}'
SWEEP    = f'results/FairSASRec-sweep-{DS}'
W, K     = 20, 10
os.makedirs(SWEEP, exist_ok=True)
print(f"Device: {DEVICE} | dataset: {DS} | lambdas: {args.lambdas}\n")

# ── data (VALIDATION is the selection target) ─────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, r in train.iterrows():
    train_set[int(r['user_id'])].add(int(r['item_id']))
val_items = dict(zip(val['user_id'].astype(int), val['item_id'].astype(int)))
val_sets  = {u: {i} for u, i in val_items.items()}

hist = defaultdict(list)
for _, r in train.sort_values('timestamp').iterrows():
    hist[int(r['user_id'])].append(int(r['item_id']))

user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
if DS == 'ml-1m':
    raw_g = load_user_gender(f'{DATA_DIR}/users.dat')
else:
    raw_g = {}
    with open(f'{DATA_DIR}/u.user') as _f:
        for _ln in _f:
            _p = _ln.strip().split('|')
            if len(_p) >= 3: raw_g[int(_p[0])] = _p[2]
user_gender = {user2idx[u]: g for u, g in raw_g.items() if u in user2idx}


class FairSASRec(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.item_emb = nn.Embedding(n + 1, 64, padding_idx=0)
        s.pos_emb  = nn.Embedding(W + 1, 64)
        s.dropout  = nn.Dropout(0.2)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, dropout=0.2,
            batch_first=True, norm_first=True) for _ in range(2)])
        s.norm = nn.LayerNorm(64)
        s.fairness_alpha = nn.Parameter(torch.ones(1))
        s.fairness_proj  = nn.Linear(64, 1)
        s.explain_proj   = nn.Linear(64, 64)

    def scores(s, seq):
        B, Wn = seq.shape
        p = torch.arange(1, Wn + 1, device=seq.device).unsqueeze(0)
        x = s.dropout(s.item_emb(seq) + s.pos_emb(p))
        m = nn.Transformer.generate_square_subsequent_mask(Wn, device=seq.device)
        for l in s.layers:
            x = l(x, src_mask=m, is_causal=True)
        st = s.norm(x)[:, -1, :]
        return st @ s.item_emb.weight.T      # zero penalty at inference


def seq_of(uid):
    h = hist.get(uid, [])[-W:]
    return [0] * (W - len(h)) + h


def eval_validation(ckpt):
    """HR/NDCG (100-neg) + DP/EO/Gini/Cov on the VALIDATION set."""
    m = FairSASRec(N_ITEMS)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    m = m.to(DEVICE).eval()

    np.random.seed(args.seed)
    hits, ndcgs, recs = [], [], {}
    with torch.no_grad():
        for uid, pos in val_items.items():
            sc = m.scores(torch.LongTensor(seq_of(uid)).unsqueeze(0).to(DEVICE))
            sc = sc.squeeze(0)[:N_ITEMS].cpu().numpy()

            s2 = sc.copy()
            for x in train_set[uid]:
                s2[x] = -1e9
            recs[str(uid)] = np.argsort(s2)[::-1][:K].tolist()

            pool = [i for i in range(N_ITEMS)
                    if i not in train_set[uid] and i != pos]
            negs = np.random.choice(pool, 99, replace=False).tolist()
            cand = [pos] + negs
            order = sorted(cand, key=lambda c: sc[c], reverse=True)
            hit = pos in order[:K]
            hits.append(1.0 if hit else 0.0)
            ndcgs.append(1 / np.log2(order[:K].index(pos) + 2) if hit else 0.0)

    fr = compute_fairir_dp_eo(recs, user_gender, val_sets, N_ITEMS, K)
    ex = compute_exposure(recs, N_ITEMS, K)
    return {'HR': float(np.mean(hits)), 'NDCG': float(np.mean(ndcgs)),
            'DP': fr['DP'], 'EO': fr['EO'],
            'Gini': gini_coefficient(ex), 'Cov': coverage(recs, N_ITEMS, K)}


rows = []
for lam in args.lambdas:
    tag  = f"lambda_{lam}"
    dest = f"{SWEEP}/{tag}.pt"
    print(f"=== lambda = {lam} ===")

    if not args.skip_train:
        env = dict(os.environ, SEED=str(args.seed))
        r = subprocess.run(
            [sys.executable, args.trainer, '--fairness_lambda', str(lam)],
            env=env)
        if r.returncode != 0:
            print(f"  training failed for lambda={lam}; skipping")
            continue
        # the trainer writes to results/FairSASRec*/ — grab the newest .pt
        cands = []
        for root in ('results/FairSASRec-ml1m', 'results/FairSASRec'):
            if os.path.isdir(root):
                for f in os.listdir(root):
                    if f.endswith('.pt'):
                        p = os.path.join(root, f)
                        cands.append((os.path.getmtime(p), p))
        if not cands:
            print("  no checkpoint produced; skipping")
            continue
        shutil.copy(sorted(cands)[-1][1], dest)

    if not os.path.exists(dest):
        print(f"  {dest} missing (run without --skip-train first)")
        continue

    v = eval_validation(dest)
    v['lambda'] = lam
    rows.append(v)
    print(f"  VAL: HR {v['HR']:.4f} | NDCG {v['NDCG']:.4f} | "
          f"DP {v['DP']:.4f} | EO {v['EO']:.4f} | "
          f"Gini {v['Gini']:.4f} | Cov {v['Cov']:.3f}\n")

if not rows:
    sys.exit("no runs completed")

json.dump(rows, open(f'{SWEEP}/validation_sweep.json', 'w'), indent=2)

print("=" * 74)
print("VALIDATION SWEEP  (selection set — these numbers go in the ablation)")
print("=" * 74)
print(f"{'lambda':>7} {'HR@10':>8} {'NDCG@10':>8} {'DP@10':>8} "
      f"{'EO@10':>8} {'Gini':>8} {'Cov':>7}")
for v in rows:
    print(f"{v['lambda']:>7.2f} {v['HR']:>8.4f} {v['NDCG']:>8.4f} "
          f"{v['DP']:>8.4f} {v['EO']:>8.4f} {v['Gini']:>8.4f} {v['Cov']:>7.3f}")

base = next((v for v in rows if v['lambda'] == 0.0), None)
best_dp  = min(rows, key=lambda v: v['DP'])
best_hr  = max(rows, key=lambda v: v['HR'])

# balanced pick: lowest DP among runs that keep HR within 1% of lambda=0
if base:
    ok = [v for v in rows if v['HR'] >= base['HR'] * 0.99]
    balanced = min(ok, key=lambda v: v['DP']) if ok else best_dp
else:
    balanced = best_dp

print("-" * 74)
print(f"best HR (val)              : lambda = {best_hr['lambda']}")
print(f"best DP (val)              : lambda = {best_dp['lambda']}")
print(f"balanced (DP | HR>=99% of lambda=0): lambda = {balanced['lambda']}")
print("-" * 74)
print("Selection is complete. Now — and only now — run")
print(f"  tests/evaluate_final.py  with FAIR_CKPT = "
      f"{SWEEP}/lambda_{balanced['lambda']}.pt")
print("and report those TEST numbers as the paper's main result.")
print("Do not re-select lambda after seeing the test numbers.")
print("=" * 74)
