"""
evaluate_fairsasrec_fullrank.py
───────────────────────────────────────────────────────────────────────
Evaluates FairSASRec under BOTH protocols so its numbers are directly
comparable to evaluate_sasrec_fairness.py.

THE POINT
─────────
Item-side fairness (Gini, Coverage) is meaningless under 100-negative
sampling: the candidate pool is random, so exposure looks uniform by
construction. Only FULL RANKING reveals whether a model actually
concentrates its recommendations on a handful of blockbusters.

SASRec full-rank baseline (already measured):
    Gini@10 = 0.8112
    Cov@10  = 0.5270
    DP_exp  = 0.5560

The FairnessConstraintLayer is validated ONLY if FairSASRec achieves:
    Gini@10 < 0.8112   AND   Cov@10 > 0.5270

USAGE
─────
    python tests/evaluate_fairsasrec_fullrank.py --dataset ml-1m
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from metrics.fairness_metrics import (compute_exposure,
                                       gini_coefficient,
                                       coverage)
from metrics.user_fairness_metrics import (load_user_gender,
                                            compute_fairir_dp_eo)

# ── Arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m',
                    choices=['ml-1m', 'ml-100k'])
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()

DATA_DIR = f'data/{args.dataset}'
CKPT     = args.ckpt or ('results/FairSASRec/fairsasrec_best.pt'
                         if args.dataset == 'ml-1m'
                         else 'results/FairSASRec-100k/fairsasrec_best.pt')

K_LIST   = [5, 10, 20, 30, 40]
WINDOW   = 20 if args.dataset == 'ml-1m' else 10
EMB_DIM  = 64
HIDDEN   = 256
N_HEADS  = 4
N_LAYERS = 2
DROPOUT  = 0.2

print("=" * 62)
print("FairSASRec Evaluation — matched protocol vs SASRec")
print(f"dataset = {args.dataset}  |  window = {WINDOW}")
print(f"checkpoint = {CKPT}")
print("=" * 62)

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']
print(f"Users: {N_USERS} | Items: {N_ITEMS}")

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

test_items = dict(zip(test['user_id'].astype(int),
                      test['item_id'].astype(int)))

user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

# ── Gender ─────────────────────────────────────────────────────────────
user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}

if args.dataset == 'ml-100k':
    raw_gender = {}
    with open(f'{DATA_DIR}/u.user', 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                raw_gender[int(parts[0])] = parts[2]
else:
    raw_gender = load_user_gender(f'{DATA_DIR}/users.dat')

user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

n_m = sum(1 for g in user_gender.values() if g == 'M')
n_f = sum(1 for g in user_gender.values() if g == 'F')
print(f"Gender loaded — Male: {n_m} | Female: {n_f}")
assert n_m > 0 and n_f > 0, "Gender loading failed"


# ══════════════════════════════════════════════════════════════════════
#  FairSASRec — must match the architecture used at training time
# ══════════════════════════════════════════════════════════════════════
class FairSASRec(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(window + 1,  emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        # FairnessConstraintLayer
        self.fairness_alpha = nn.Parameter(torch.ones(1))
        self.fairness_proj  = nn.Linear(emb_dim, 1)

        # AttentionExplainer
        self.explain_proj  = nn.Linear(emb_dim, emb_dim)
        self.explain_scale = emb_dim ** 0.5

    def encode(self, seq):
        B, W = seq.shape
        pos  = torch.arange(1, W + 1, device=seq.device).unsqueeze(0)
        x    = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(
            W, device=seq.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.norm(x)

    def all_scores(self, seq, penalty_vec):
        out    = self.encode(seq)
        state  = out[:, -1, :]
        scores = state @ self.item_emb.weight.T

        pen_t = torch.as_tensor(penalty_vec, dtype=torch.float32,
                                device=seq.device)
        alpha = F.softplus(self.fairness_alpha)
        rho   = torch.sigmoid(self.fairness_proj(state))
        return scores - rho * (alpha * pen_t).unsqueeze(0)


model = FairSASRec(N_ITEMS, EMB_DIM, HIDDEN,
                   N_HEADS, N_LAYERS, WINDOW, DROPOUT)

state = torch.load(CKPT, map_location='cpu')
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  [warn] missing keys: {missing}")
if unexpected:
    print(f"  [warn] unexpected keys: {unexpected}")
model.eval()

alpha_val = F.softplus(model.fairness_alpha).item()
print(f"Loaded. Learned alpha = {alpha_val:.4f}")


# ══════════════════════════════════════════════════════════════════════
#  Reconstruct the penalty vector the model was trained with.
#
#  v1 (fairsasrec.py)    used: log1p(item_exposure), normalised
#  v2 (fairsasrec_v2.py) used: per-item gender exposure gap
#
#  We prefer saved exposure arrays if present; otherwise fall back to
#  training-set popularity, which is the closest available proxy.
# ══════════════════════════════════════════════════════════════════════
ckpt_dir = os.path.dirname(CKPT)
male_p   = os.path.join(ckpt_dir, 'male_exposure.npy')
female_p = os.path.join(ckpt_dir, 'female_exposure.npy')

if os.path.exists(male_p) and os.path.exists(female_p):
    male_exp   = np.load(male_p)
    female_exp = np.load(female_p)
    total = male_exp + female_exp
    penalty_vec = np.zeros_like(total)
    mask = total > 0
    penalty_vec[mask] = (np.abs(male_exp[mask] - female_exp[mask])
                         / total[mask])
    print("Penalty vector: gender exposure gap (from saved v2 arrays)")
else:
    pop = np.zeros(N_ITEMS + 1, dtype=np.float32)
    for _, row in train.iterrows():
        i = int(row['item_id'])
        if 0 <= i <= N_ITEMS:
            pop[i] += 1
    penalty_vec = np.log1p(pop)
    penalty_vec = penalty_vec / (penalty_vec.max() + 1e-9)
    print("Penalty vector: log-popularity (v1 fallback)")

print()


def get_seq(uid):
    h = user_history.get(uid, [])[-WINDOW:]
    if len(h) < WINDOW:
        h = [0] * (WINDOW - len(h)) + h
    return torch.LongTensor(h).unsqueeze(0)


def hit_at_k(ranked, pos, k):
    return 1.0 if pos in ranked[:k] else 0.0


def ndcg_at_k(ranked, pos, k):
    if pos in ranked[:k]:
        return 1.0 / np.log2(ranked[:k].index(pos) + 2)
    return 0.0


# ══════════════════════════════════════════════════════════════════════
#  PROTOCOL A — 100-negative sampling  (accuracy + DP-HR)
# ══════════════════════════════════════════════════════════════════════
print("--- Protocol A: 100-negative sampling ---")
np.random.seed(42)

res_a   = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}
recs_a  = {}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen = train_set[uid] | val_set[uid] | {pos_item}
        pool = [i for i in range(1, N_ITEMS + 1) if i not in seen]
        if len(pool) < 99:
            continue

        negs       = np.random.choice(pool, 99, replace=False).tolist()
        candidates = [pos_item] + negs

        seq_t  = get_seq(uid)
        scores = model.all_scores(seq_t, penalty_vec).squeeze(0)
        cand_scores = [scores[c].item() for c in candidates]

        order  = sorted(range(len(candidates)),
                        key=lambda i: cand_scores[i], reverse=True)
        ranked = [candidates[i] for i in order]

        recs_a[str(uid)] = ranked
        for k in K_LIST:
            res_a[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            res_a[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'Gini':>7} {'Cov':>7}")
print("-" * 48)

table_a = {}
for k in K_LIST:
    hr   = float(np.mean(res_a[k]['hits']))
    ndcg = float(np.mean(res_a[k]['ndcgs']))

    recs_k   = {u: it[:k] for u, it in recs_a.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)

    hm, hf = [], []
    for uid, pos_item in test_items.items():
        if str(uid) not in recs_a:
            continue
        hit = hit_at_k(recs_a[str(uid)], pos_item, k)
        (hm if user_gender.get(uid, 'M') == 'M' else hf).append(hit)
    dp = abs(np.mean(hm) - np.mean(hf)) if hm and hf else float('nan')

    table_a[k] = {'HR': hr, 'NDCG': ndcg, 'DP': dp,
                  'Gini': gini, 'Coverage': cov}
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f} {dp:>7.4f} "
          f"{gini:>7.4f} {cov:>7.4f}")


# ══════════════════════════════════════════════════════════════════════
#  PROTOCOL B — FULL RANKING  (item-fairness ground truth)
# ══════════════════════════════════════════════════════════════════════
print("\n--- Protocol B: FULL RANKING (item-fairness ground truth) ---")

recs_b = {}
with torch.no_grad():
    for uid in test_items.keys():
        seq_t  = get_seq(uid)
        scores = model.all_scores(seq_t, penalty_vec).squeeze(0).clone()

        seen = train_set[uid] | val_set[uid]
        if seen:
            idx = torch.LongTensor(list(seen))
            idx = idx[(idx >= 0) & (idx <= N_ITEMS)]
            if len(idx) > 0:
                scores[idx] = -1e9
        scores[0] = -1e9

        recs_b[str(uid)] = scores.topk(40).indices.numpy().tolist()

print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'Gini':>7} {'Cov':>7} "
      f"{'DP_exp':>8} {'EO_exp':>8}")
print("-" * 58)

table_b = {}
for k in K_LIST:
    hits, ndcgs = [], []
    for uid, pos_item in test_items.items():
        r = recs_b.get(str(uid), [])
        hits.append(hit_at_k(r, pos_item, k))
        ndcgs.append(ndcg_at_k(r, pos_item, k))

    hr   = float(np.mean(hits))
    ndcg = float(np.mean(ndcgs))

    recs_k   = {u: it[:k] for u, it in recs_b.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)

    fair   = compute_fairir_dp_eo(recs_k, user_gender,
                                   test_set, N_ITEMS, k)

    table_b[k] = {'HR': hr, 'NDCG': ndcg, 'Gini': gini,
                  'Coverage': cov,
                  'DP_exp': fair['DP'], 'EO_exp': fair['EO']}
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f} {gini:>7.4f} {cov:>7.4f} "
          f"{fair['DP']:>8.4f} {fair['EO']:>8.4f}")


# ── Save ───────────────────────────────────────────────────────────────
out_dir = f'results/FairSASRec-{args.dataset}-fullrank'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/fairsasrec_fairness.json', 'w') as f:
    json.dump({
        'model':          'FairSASRec (backbone + FairnessConstraintLayer)',
        'dataset':        args.dataset,
        'learned_alpha':  alpha_val,
        'sampled_100neg': {str(k): v for k, v in table_a.items()},
        'full_ranking':   {str(k): v for k, v in table_b.items()},
    }, f, indent=2)
print(f"\nSaved: {out_dir}/fairsasrec_fairness.json")


# ══════════════════════════════════════════════════════════════════════
#  VERDICT — the whole paper hinges on this block
# ══════════════════════════════════════════════════════════════════════
SASREC_FULL = {'Gini': 0.8112, 'Cov': 0.5270,
               'DP_exp': 0.5560, 'HR': 0.1772}

f10 = table_b[10]
d_gini = SASREC_FULL['Gini'] - f10['Gini']      # positive = better
d_cov  = f10['Coverage'] - SASREC_FULL['Cov']   # positive = better
d_dp   = SASREC_FULL['DP_exp'] - f10['DP_exp']  # positive = better

print("\n" + "=" * 62)
print("VERDICT — FairnessConstraintLayer, matched full-rank protocol")
print("=" * 62)
print(f"{'Metric':<14} {'SASRec':>9} {'FairSASRec':>12} {'Delta':>9}  Better?")
print("-" * 62)
print(f"{'Gini@10 (lo)':<14} {SASREC_FULL['Gini']:>9.4f} "
      f"{f10['Gini']:>12.4f} {d_gini:>+9.4f}  "
      f"{'YES' if d_gini > 0 else 'no'}")
print(f"{'Cov@10 (hi)':<14} {SASREC_FULL['Cov']:>9.4f} "
      f"{f10['Coverage']:>12.4f} {d_cov:>+9.4f}  "
      f"{'YES' if d_cov > 0 else 'no'}")
print(f"{'DP_exp (lo)':<14} {SASREC_FULL['DP_exp']:>9.4f} "
      f"{f10['DP_exp']:>12.4f} {d_dp:>+9.4f}  "
      f"{'YES' if d_dp > 0 else 'no'}")
print(f"{'HR@10 (hi)':<14} {SASREC_FULL['HR']:>9.4f} "
      f"{f10['HR']:>12.4f} {f10['HR']-SASREC_FULL['HR']:>+9.4f}")
print("-" * 62)

if d_gini > 0 and d_cov > 0:
    print(">> LAYER VALIDATED. Item-fairness claim is supported.")
    print("   Paper: 'FairSASRec: Item-Fairness-Aware Explainable")
    print("           Sequential Recommendation'")
elif d_gini > 0 or d_cov > 0:
    print(">> PARTIAL. One metric improves, one does not.")
    print("   Report both honestly; frame as a trade-off, not a win.")
else:
    print(">> LAYER NOT VALIDATED. Item-fairness claim is NOT supported.")
    print("   Paper: 'Dual-Explainable Sequential Recommendation'")
    print("   Contributions: SOTA accuracy + explainability")
    print("                  + honest negative result on fairness regularisation")
print("=" * 62)
