"""
evaluate_sasrec_fairness.py
───────────────────────────────────────────────────────────────────────
Measures Gini, Coverage, DP and EO for the PLAIN SASRec baseline
(i.e. WITHOUT the FairnessConstraintLayer).

WHY THIS MATTERS FOR THE PAPER
──────────────────────────────
Our headline claim is that the FairnessConstraintLayer improves
item-side fairness. To prove the *layer* is responsible — and not the
Transformer backbone alone — we must show:

    SASRec        (backbone only)      -> Gini_A, Coverage_A
    FairSASRec    (backbone + layer)   -> Gini_B, Coverage_B

and demonstrate Gini_B < Gini_A and Coverage_B > Coverage_A.

Without this number the ablation is incomplete and a reviewer will
reject the causal claim immediately.

USAGE
─────
    python tests/evaluate_sasrec_fairness.py --dataset ml-1m
    python tests/evaluate_sasrec_fairness.py --dataset ml-100k

Requires: results/SASRec/sasrec_best.pt  (from sasrec_baseline.py)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
parser.add_argument('--ckpt', type=str,
                    default='results/SASRec/sasrec_best.pt')
args = parser.parse_args()

DATA_DIR = f'data/{args.dataset}'
K_LIST   = [5, 10, 20, 30, 40]
WINDOW   = 20 if args.dataset == 'ml-1m' else 10
EMB_DIM  = 64
HIDDEN   = 256
N_HEADS  = 4
N_LAYERS = 2
DROPOUT  = 0.2

print("=" * 62)
print(f"SASRec Fairness Evaluation (baseline, NO fairness layer)")
print(f"dataset = {args.dataset}  |  window = {WINDOW}")
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

# ── Gender (format differs between ML-1M and ML-100K) ──────────────────
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


# ── SASRec (identical architecture to sasrec_baseline.py) ──────────────
class SASRec(nn.Module):
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
        self.norm   = nn.LayerNorm(emb_dim)
        self.window = window

    def forward(self, seq):
        B, W = seq.shape
        pos  = torch.arange(1, W + 1, device=seq.device).unsqueeze(0)
        x    = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(
            W, device=seq.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.norm(x)

    def all_scores(self, seq):
        """Score every item — needed for Gini/Coverage."""
        out   = self.forward(seq)
        state = out[:, -1, :]                      # (B, d)
        return state @ self.item_emb.weight.T      # (B, N+1)

    def predict(self, seq, item_idx):
        return self.all_scores(seq).gather(1, item_idx)


model = SASRec(N_ITEMS, EMB_DIM, HIDDEN,
               N_HEADS, N_LAYERS, WINDOW, DROPOUT)
model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
model.eval()
print(f"Loaded checkpoint: {args.ckpt}\n")


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
#  PART 1 — Accuracy + fairness under 100-negative sampling
#           (same protocol as evaluate_fairsasrec.py, so the numbers
#            are directly comparable to FairSASRec)
# ══════════════════════════════════════════════════════════════════════
print("--- Protocol A: 100-negative sampling ---")
np.random.seed(42)

results   = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}
recs_dict = {}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen = train_set[uid] | val_set[uid] | {pos_item}
        pool = [i for i in range(1, N_ITEMS + 1) if i not in seen]
        if len(pool) < 99:
            continue

        negs       = np.random.choice(pool, 99, replace=False).tolist()
        candidates = [pos_item] + negs

        seq_t  = get_seq(uid)
        cand_t = torch.LongTensor(candidates).unsqueeze(0)
        scores = model.predict(seq_t, cand_t).squeeze(0)

        order  = sorted(range(len(candidates)),
                        key=lambda i: scores[i].item(), reverse=True)
        ranked = [candidates[i] for i in order]

        recs_dict[str(uid)] = ranked
        for k in K_LIST:
            results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)

sampled_table = {}
for k in K_LIST:
    hr   = float(np.mean(results[k]['hits']))
    ndcg = float(np.mean(results[k]['ndcgs']))

    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)

    # DP as HR-difference between gender groups
    hm, hf = [], []
    for uid, pos_item in test_items.items():
        if str(uid) not in recs_dict:
            continue
        hit = hit_at_k(recs_dict[str(uid)], pos_item, k)
        if user_gender.get(uid, 'M') == 'M':
            hm.append(hit)
        else:
            hf.append(hit)
    dp = abs(np.mean(hm) - np.mean(hf)) if hm and hf else float('nan')

    sampled_table[k] = {'HR': hr, 'NDCG': ndcg, 'DP': dp,
                        'Gini': gini, 'Coverage': cov}
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f} "
          f"{dp:>7.4f} {dp:>7.4f} {gini:>7.4f} {cov:>7.4f}")


# ══════════════════════════════════════════════════════════════════════
#  PART 2 — Item-side fairness under FULL RANKING
#
#  This is the number that actually matters for the item-fairness claim.
#  Gini/Coverage computed on 100 sampled candidates is misleading: the
#  candidate pool is random, so exposure looks artificially uniform.
#  Only full ranking reveals whether the model truly concentrates its
#  recommendations on a handful of blockbusters.
# ══════════════════════════════════════════════════════════════════════
print("\n--- Protocol B: FULL RANKING (item-fairness ground truth) ---")

full_recs = {}
with torch.no_grad():
    for uid in test_items.keys():
        seq_t  = get_seq(uid)
        scores = model.all_scores(seq_t).squeeze(0).clone()

        # Mask items the user has already seen
        seen = train_set[uid] | val_set[uid]
        if seen:
            idx = torch.LongTensor(list(seen))
            idx = idx[(idx >= 0) & (idx <= N_ITEMS)]
            if len(idx) > 0:
                scores[idx] = -1e9
        scores[0] = -1e9   # padding index is not a real item

        top40 = scores.topk(40).indices.numpy().tolist()
        full_recs[str(uid)] = top40

print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'Gini':>7} {'Cov':>7} "
      f"{'DP_exp':>8} {'EO_exp':>8}")
print("-" * 58)

full_table = {}
for k in K_LIST:
    hits, ndcgs = [], []
    for uid, pos_item in test_items.items():
        r = full_recs.get(str(uid), [])
        hits.append(hit_at_k(r, pos_item, k))
        ndcgs.append(ndcg_at_k(r, pos_item, k))

    hr   = float(np.mean(hits))
    ndcg = float(np.mean(ndcgs))

    recs_k   = {uid: items[:k] for uid, items in full_recs.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)

    fair   = compute_fairir_dp_eo(recs_k, user_gender,
                                   test_set, N_ITEMS, k)
    dp_exp = fair['DP']
    eo_exp = fair['EO']

    full_table[k] = {'HR': hr, 'NDCG': ndcg, 'Gini': gini,
                     'Coverage': cov, 'DP_exp': dp_exp, 'EO_exp': eo_exp}
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f} {gini:>7.4f} {cov:>7.4f} "
          f"{dp_exp:>8.4f} {eo_exp:>8.4f}")


# ── Save ───────────────────────────────────────────────────────────────
out_dir = f'results/SASRec-{args.dataset}'
os.makedirs(out_dir, exist_ok=True)
out_path = f'{out_dir}/sasrec_fairness.json'

with open(out_path, 'w') as f:
    json.dump({
        'model':          'SASRec (no fairness layer)',
        'dataset':        args.dataset,
        'window':         WINDOW,
        'sampled_100neg': {str(k): v for k, v in sampled_table.items()},
        'full_ranking':   {str(k): v for k, v in full_table.items()},
    }, f, indent=2)

print(f"\nSaved: {out_path}")

# ── Ablation summary ───────────────────────────────────────────────────
print("\n" + "=" * 62)
print("ABLATION — paste these SASRec numbers into the paper")
print("=" * 62)
s10 = sampled_table[10]
f10 = full_table[10]
print(f"  [100-neg]      HR@10   = {s10['HR']:.4f}")
print(f"  [100-neg]      NDCG@10 = {s10['NDCG']:.4f}")
print(f"  [100-neg]      DP@10   = {s10['DP']:.4f}")
print(f"  [full-rank]    Gini@10 = {f10['Gini']:.4f}   <-- key number")
print(f"  [full-rank]    Cov@10  = {f10['Coverage']:.4f}   <-- key number")
print(f"  [full-rank]    DP_exp  = {f10['DP_exp']:.4f}")
print()
print("Compare against FairSASRec. The fairness layer is validated")
print("only if:   Gini(FairSASRec) < Gini(SASRec)")
print("     and:  Cov(FairSASRec)  > Cov(SASRec)")
print("=" * 62)
