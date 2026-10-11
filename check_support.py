"""
check_support.py — validity check for the explanation statistics.
───────────────────────────────────────────────────────────────────────
The gender exposure gap of item i is |male_i - fem_i| / (male_i + fem_i),
which is undefined when the reference model never recommends i. In
make_explanation_figure.py such items are assigned a gap of zero, so a
replacement item that the backbone never surfaced receives a perfect
score for free. This script measures how much of the reported effect
comes from that artefact, and reports the effect restricted to items
with real exposure support.

Run it after make_explanation_figure.py, from the project root.

USAGE
    python check_support.py --run p1 --dataset ml-1m
"""

import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='p1')
ap.add_argument('--dataset', default='ml-1m')
ap.add_argument('--fair-ckpt', default=None)
ap.add_argument('--sasrec-ckpt', default=None)
ap.add_argument('--min-support', type=int, default=10)
a = ap.parse_args()

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = f'data/{a.dataset}'
ROOT     = os.path.join('runs', a.run)
W = 20

FAIR_CKPT = a.fair_ckpt or (
    'results/FairSASRec-ml1m/fairsasrec_final.pt' if a.dataset == 'ml-1m'
    else f'{ROOT}/models/FairSASRec-{a.dataset}/fairsasrec_best.pt')
SAS_CKPT = a.sasrec_ckpt or (
    'results/SASRec-200/sasrec_best.pt' if a.dataset == 'ml-1m'
    else f'{ROOT}/models/SASRec-{a.dataset}/sasrec_best.pt')

print(f"Device: {DEVICE} | dataset: {a.dataset}")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']

hist = defaultdict(list)
for _, r in train.sort_values('timestamp').iterrows():
    hist[int(r['user_id'])].append(int(r['item_id']))
seen = defaultdict(set)
for _, r in pd.concat([train, val]).iterrows():
    seen[int(r['user_id'])].add(int(r['item_id']))
users = sorted(set(test['user_id'].astype(int)))

user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_g = {}
if a.dataset == 'ml-1m':
    for line in open(f'{DATA_DIR}/users.dat'):
        p = line.strip().split('::')
        if len(p) >= 2: raw_g[int(p[0])] = p[1]
else:
    for line in open(f'{DATA_DIR}/u.user'):
        p = line.strip().split('|')
        if len(p) >= 3: raw_g[int(p[0])] = p[2]
gender = {user2idx[u]: g for u, g in raw_g.items() if u in user2idx}


class Encoder(nn.Module):
    def __init__(s, n, fair, d=64, h=256, heads=4, layers=2, w=W, p=0.2):
        super().__init__()
        s.item_emb = nn.Embedding(n + 1, d, padding_idx=0)
        s.pos_emb  = nn.Embedding(w + 1, d)
        s.dropout  = nn.Dropout(p)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=h, dropout=p,
            batch_first=True, norm_first=True) for _ in range(layers)])
        s.norm = nn.LayerNorm(d)
        if fair:
            s.fairness_alpha = nn.Parameter(torch.ones(1))
            s.fairness_proj  = nn.Linear(d, 1)
            s.explain_proj   = nn.Linear(d, d)

    def scores(s, seq):
        B, Wn = seq.shape
        pos = torch.arange(1, Wn + 1, device=seq.device).unsqueeze(0)
        x = s.dropout(s.item_emb(seq) + s.pos_emb(pos))
        m = nn.Transformer.generate_square_subsequent_mask(Wn,
                                                           device=seq.device)
        for l in s.layers: x = l(x, src_mask=m, is_causal=True)
        st = s.norm(x)[:, -1, :]
        return st @ s.item_emb.weight.T

def load(path, fair):
    m = Encoder(N_ITEMS, fair)
    m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    return m.to(DEVICE).eval()

fair_m, sas_m = load(FAIR_CKPT, True), load(SAS_CKPT, False)

def seq_of(uid):
    h = hist.get(uid, [])[-W:]
    return [0] * (W - len(h)) + h

def topk_and_top1(model, uid, k=10):
    t = torch.LongTensor(seq_of(uid)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sc = model.scores(t).squeeze(0)[:N_ITEMS].clone()
    for i in seen.get(uid, ()):
        if i < N_ITEMS: sc[i] = -1e9
    idx = sc.topk(k).indices.cpu().numpy()
    return idx, int(idx[0])

# ── exposure support and gap, from the BACKBONE's lists ──────────────
print("Building exposure support from the backbone...")
male = np.zeros(N_ITEMS, dtype=np.float32)
fem  = np.zeros(N_ITEMS, dtype=np.float32)
sas_top1 = {}
for uid in users:
    idx, t1 = topk_and_top1(sas_m, uid)
    sas_top1[uid] = t1
    arr = male if gender.get(uid, 'M') == 'M' else fem
    for it in idx:
        arr[int(it)] += 1

support = male + fem
gap = np.full(N_ITEMS, np.nan, dtype=np.float32)   # NaN where undefined
m = support > 0
gap[m] = np.abs(male[m] - fem[m]) / support[m]

print(f"  items with any exposure : {int(m.sum())} / {N_ITEMS} "
      f"({m.mean():.1%})")

# ── collect the changed recommendations ──────────────────────────────
print("Comparing top-1 recommendations...")
before, after = [], []
for uid in users:
    _, f1 = topk_and_top1(fair_m, uid)
    b1 = sas_top1[uid]
    if b1 != f1:
        before.append(b1); after.append(f1)
before, after = np.array(before), np.array(after)
n_changed = len(before)

gb_raw = np.nan_to_num(gap[before], nan=0.0)
ga_raw = np.nan_to_num(gap[after],  nan=0.0)
t_raw, p_raw = stats.ttest_rel(gb_raw, ga_raw)

zero_sup_after  = float((support[after]  == 0).mean())
zero_sup_before = float((support[before] == 0).mean())

# ── restricted comparison: both items must have real support ─────────
mask = (support[before] >= a.min_support) & (support[after] >= a.min_support)
res = {}
if mask.sum() >= 30:
    gb_r, ga_r = gap[before][mask], gap[after][mask]
    t_r, p_r = stats.ttest_rel(gb_r, ga_r)
    res = dict(n=int(mask.sum()),
               before=float(gb_r.mean()), after=float(ga_r.mean()),
               reduction_pct=float(100*(ga_r.mean()-gb_r.mean())/gb_r.mean()),
               t=float(t_r), p=float(p_r),
               pct_improved=float((ga_r < gb_r).mean()))

print("\n" + "="*66)
print("UNRESTRICTED (undefined gaps counted as zero)")
print("="*66)
print(f"  changed top-1                  : {n_changed} / {len(users)} "
      f"({n_changed/len(users):.1%})")
print(f"  displaced gap (mean)           : {gb_raw.mean():.4f}")
print(f"  replacement gap (mean)         : {ga_raw.mean():.4f}")
print(f"  t={t_raw:.2f}, p={p_raw:.2e}")
print(f"\n  displaced items with zero support  : {zero_sup_before:.1%}")
print(f"  replacement items with zero support: {zero_sup_after:.1%}   "
      f"<-- inflates the effect if large")

print("\n" + "="*66)
print(f"RESTRICTED (both items with >= {a.min_support} exposures)")
print("="*66)
if res:
    print(f"  cases                          : {res['n']}")
    print(f"  displaced gap (mean)           : {res['before']:.4f}")
    print(f"  replacement gap (mean)         : {res['after']:.4f}")
    print(f"  reduction                      : {res['reduction_pct']:+.1f}%")
    print(f"  t={res['t']:.2f}, p={res['p']:.2e}")
    print(f"  replacement less skewed in     : {res['pct_improved']:.1%}")
else:
    print(f"  too few cases ({int(mask.sum())}) for a reliable test")
print("="*66)

out = dict(unrestricted=dict(
              n_changed=n_changed, n_users=len(users),
              before=float(gb_raw.mean()), after=float(ga_raw.mean()),
              t=float(t_raw), p=float(p_raw),
              zero_support_before=zero_sup_before,
              zero_support_after=zero_sup_after),
           restricted=res, min_support=a.min_support)
os.makedirs(f'{ROOT}/final', exist_ok=True)
json.dump(out, open(f'{ROOT}/final/support_check.json', 'w'), indent=2)
print(f"\nSaved {ROOT}/final/support_check.json")
