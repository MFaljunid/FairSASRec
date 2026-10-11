
import sys, os, glob, json, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--dataset', default='ml-1m', choices=['ml-1m', 'ml-100k'])
_ap.add_argument('--sasrec-ckpt', default=None)
_ap.add_argument('--fair-ckpt',   default=None)
_a = _ap.parse_args()

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = f'data/{_a.dataset}'
DS       = _a.dataset

_DEFAULTS = {
    'ml-1m':   ('results/SASRec/sasrec_best.pt',
                'results/FairSASRec-ml1m/fairsasrec_final.pt'),
    'ml-100k': ('results/SASRec-100k/sasrec_best.pt',
                'results/FairSASRec-100k/fairsasrec_best.pt'),
}
SASREC_CKPT = _a.sasrec_ckpt or _DEFAULTS[DS][0]
FAIR_CKPT   = _a.fair_ckpt   or _DEFAULTS[DS][1]

K_ACC = [5,10,20,30,40,50]; K_FAIR = [10,20,30,40,50]; W = 20
print(f"Device: {DEVICE} | dataset: {DS}")
print(f"  SASRec ckpt: {SASREC_CKPT}")
print(f"  Fair   ckpt: {FAIR_CKPT}")

train = pd.read_csv(f'{DATA_DIR}/train.csv'); val = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv');  meta = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']; N_USERS = meta['n_users']

train_set = defaultdict(set); val_set = defaultdict(set)
for _, r in train.iterrows(): train_set[int(r['user_id'])].add(int(r['item_id']))
for _, r in val.iterrows():   val_set[int(r['user_id'])].add(int(r['item_id']))
test_items = dict(zip(test['user_id'].astype(int), test['item_id'].astype(int)))
test_sets  = {u: {i} for u, i in test_items.items()}
hist = defaultdict(list)
for _, r in train.sort_values('timestamp').iterrows():
    hist[int(r['user_id'])].append(int(r['item_id']))

user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
_gpath = f'{DATA_DIR}/users.dat' if DS == 'ml-1m' else f'{DATA_DIR}/u.user'
if DS == 'ml-1m':
    raw_g = load_user_gender(_gpath)
else:
    raw_g = {}
    with open(_gpath) as _f:
        for _ln in _f:
            _p = _ln.strip().split('|')
            if len(_p) >= 3: raw_g[int(_p[0])] = _p[2]
user_gender = {user2idx[u]: g for u, g in raw_g.items() if u in user2idx}

pop = np.zeros(N_ITEMS, dtype=np.float32)
for i in train['item_id']: pop[int(i)] += 1
pen = np.log1p(pop); pen = pen / (pen.max() + 1e-9)          # v1 penalty basis

def seq_of(uid, shift):
    h = [i + shift for i in hist.get(uid, [])][-W:]
    return [0]*(W-len(h)) + h

# ── model defs ────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(s, n, fair=False):
        super().__init__()
        s.item_emb = nn.Embedding(n+1, 64, padding_idx=0)
        s.pos_emb  = nn.Embedding(W+1, 64); s.dropout = nn.Dropout(0.2)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, dropout=0.2,
            batch_first=True, norm_first=True) for _ in range(2)])
        s.norm = nn.LayerNorm(64); s.fair_on = fair
        if fair:
            s.fairness_alpha = nn.Parameter(torch.ones(1))
            s.fairness_proj  = nn.Linear(64, 1)
            s.explain_proj   = nn.Linear(64, 64)
    def forward(s, seq):
        B, Wn = seq.shape
        p = torch.arange(1, Wn+1, device=seq.device).unsqueeze(0)
        x = s.dropout(s.item_emb(seq) + s.pos_emb(p))
        m = nn.Transformer.generate_square_subsequent_mask(Wn, device=seq.device)
        for l in s.layers: x = l(x, src_mask=m, is_causal=True)
        return s.norm(x)
    def scores(s, seq, pen_t=None):
        st = s.forward(seq)[:, -1, :]
        sc = st @ s.item_emb.weight.T
        if s.fair_on and pen_t is not None:
            a = F.softplus(s.fairness_alpha)
            rho = torch.sigmoid(s.fairness_proj(st))
            sc = sc - rho * (a * pen_t).unsqueeze(0)
        return sc

def load_torch(path, fair):
    m = SASRec(N_ITEMS, fair)
    sd = torch.load(path, map_location='cpu')
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load warn] {path}")
        print(f"    missing={missing[:3]}")
        print(f"    unexpected={unexpected[:3]}")
    return m.to(DEVICE).eval()

# ── generic evaluation given a full-score function (data-id space) ────
def eval_model(name, full_scores_fn):
    print(f"\n--- {name} ---")
    np.random.seed(42)
    accs = {k: {'h': [], 'n': []} for k in K_ACC}
    recs50 = {}
    for uid, pos in test_items.items():
        sc = full_scores_fn(uid)
        if sc is None: continue
        # Protocol B: full-rank top-50 (mask seen)
        s2 = sc.copy()
        for x in (train_set[uid] | val_set[uid]): s2[x] = -1e9
        recs50[str(uid)] = np.argsort(s2)[::-1][:max(K_FAIR)].tolist()
        # Protocol A: 100-neg
        seen = train_set[uid] | val_set[uid] | {pos}
        pool = [i for i in range(N_ITEMS) if i not in seen]
        negs = np.random.choice(pool, 99, replace=False).tolist()
        cand = [pos] + negs
        order = sorted(cand, key=lambda c: sc[c], reverse=True)
        for k in K_ACC:
            hit = 1.0 if pos in order[:k] else 0.0
            accs[k]['h'].append(hit)
            accs[k]['n'].append(1/np.log2(order[:k].index(pos)+2) if hit else 0.0)
    out = {'acc': {k: {'HR': float(np.mean(v['h'])),
                       'NDCG': float(np.mean(v['n']))} for k, v in accs.items()},
           'fair': {}}
    for k in K_FAIR:
        rk = {u: it[:k] for u, it in recs50.items()}
        fr = compute_fairir_dp_eo(rk, user_gender, test_sets, N_ITEMS, k)
        ex = compute_exposure(rk, N_ITEMS, k)
        me = np.zeros(N_ITEMS); fe = np.zeros(N_ITEMS)
        for u, items in rk.items():
            arr = me if user_gender.get(int(u), 'M') == 'M' else fe
            for it in items:
                if 0 <= it < N_ITEMS: arr[it] += 1
        p = me/ (me.sum()+1e-12); q = fe/(fe.sum()+1e-12)
        mm = 0.5*(p+q)
        def _kl(a, b):
            msk = a > 0
            return float(np.sum(a[msk]*np.log2(a[msk]/b[msk])))
        js = 0.5*_kl(p, mm) + 0.5*_kl(q, mm)
        out['fair'][k] = {'DP': fr['DP'], 'EO': fr['EO'],
                          'Gini': gini_coefficient(ex),
                          'Cov': coverage(rk, N_ITEMS, k), 'JS': js}
    print(f"  {'K':<4} {'HR':>8} {'NDCG':>8} {'DP':>8} {'EO':>8} "
          f"{'Gini':>8} {'Cov':>7} {'JS':>8}")
    for k in sorted(set(K_ACC) | set(K_FAIR)):
        a = out['acc'].get(k); f = out['fair'].get(k)
        hr = f"{a['HR']:>8.4f}" if a else f"{'--':>8}"
        nd = f"{a['NDCG']:>8.4f}" if a else f"{'--':>8}"
        if f:
            rest = (f"{f['DP']:>8.4f} {f['EO']:>8.4f} {f['Gini']:>8.4f} "
                    f"{f['Cov']:>7.3f} {f['JS']:>8.4f}")
        else:
            rest = f"{'--':>8} {'--':>8} {'--':>8} {'--':>7} {'--':>8}"
        print(f"  {k:<4} {hr} {nd} {rest}")
    return out

results = {}

# BPR
_bprdir = 'results/BPR' if DS == 'ml-1m' else 'results/BPR-100k'
pkls = sorted(glob.glob(f'{_bprdir}/*.pkl'))
if pkls:
    with open(pkls[-1], 'rb') as f: bpr = pickle.load(f)
    u2c = {int(k): v for k, v in bpr.uid_map.items()}
    i2c = {int(k): v for k, v in bpr.iid_map.items()}
    def bpr_scores(uid):
        if uid not in u2c: return None
        raw = np.asarray(bpr.score(u2c[uid]), dtype=np.float32)
        sc = np.full(N_ITEMS, -1e9, dtype=np.float32)
        for it, ci in i2c.items():
            if 0 <= ci < len(raw) and 0 <= it < N_ITEMS: sc[it] = raw[ci]
        return sc
    results['BPR'] = eval_model('BPR', bpr_scores)
else:
    print("BPR pkl not found — skipping")

# SASRec (plain) — shift +1
sas = load_torch(SASREC_CKPT, fair=False)
def sas_scores(uid):
    with torch.no_grad():
        sc = sas.scores(torch.LongTensor(seq_of(uid, 0)).unsqueeze(0).to(DEVICE))
    return sc.squeeze(0)[:N_ITEMS].cpu().numpy()
results['SASRec'] = eval_model('SASRec', sas_scores)

# FairSASRec — shift +1, popularity penalty
fp = FAIR_CKPT
fair = load_torch(fp, fair=True)
pen_t = torch.zeros(N_ITEMS + 1, device=DEVICE)

def fair_scores(uid):
    with torch.no_grad():
        sc = fair.scores(torch.LongTensor(seq_of(uid, 0)).unsqueeze(0).to(DEVICE),
                         pen_t)
    return sc.squeeze(0)[:N_ITEMS].cpu().numpy()
results['FairSASRec'] = eval_model(f'FairSASRec ({fp})', fair_scores)

# ── paper table ───────────────────────────────────────────────────────
os.makedirs('results/final', exist_ok=True)
json.dump(results, open(f'results/final/paper_table_{DS}.json', 'w'), indent=2)
print("\n" + "="*82)
print(f"PAPER TABLE ({DS}, leave-one-out | acc: 100-neg | fair: full-rank)")
print("="*82)
for k in [10, 20, 30, 40]:
    print(f"\n--- K = {k} ---")
    print(f"{'Model':<12} {'HR':>8} {'NDCG':>8} {'DP':>8} {'EO':>8} "
          f"{'Gini':>8} {'Cov':>7} {'JS':>8}")
    for name, r in results.items():
        a = r['acc'].get(k); f = r['fair'].get(k)
        if not (a and f):
            continue
        print(f"{name:<12} {a['HR']:>8.4f} {a['NDCG']:>8.4f} {f['DP']:>8.4f} "
              f"{f['EO']:>8.4f} {f['Gini']:>8.4f} {f['Cov']:>7.3f} "
              f"{f['JS']:>8.4f}")

print("\n--- JS divergence across K (FairIR Table 7 format) ---")
print(f"{'Model':<12} {'JS@20':>8} {'JS@50':>8}")
for name, r in results.items():
    print(f"{name:<12} {r['fair'][20]['JS']:>8.4f} {r['fair'][50]['JS']:>8.4f}")

print("-"*82)
print("FairIR published (their protocol, K=10): HR .2833 NDCG .2777 "
      "DP .5741 EO .6258")
print("FairIR published JS: @20 .1301  @50 .0968")
print(f"Saved: results/final/paper_table_{DS}.json")
