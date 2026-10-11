"""
Run this in your project:
  py tests/evaluate_fairsasrec.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
from collections import defaultdict
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

DATA_DIR = 'data/ml-1m'
K_LIST   = [5, 10, 20, 30, 40]
WINDOW   = 20
LAMBDA   = 0.3   # best lambda

train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']
N_USERS = meta['n_users']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = dict(zip(test['user_id'].astype(int),
                      test['item_id'].astype(int)))

test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

user2idx   = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender = load_user_gender(f'{DATA_DIR}/users.dat')
user_gender= {user2idx[u]: g for u, g in raw_gender.items() if u in user2idx}

item_exposure = np.load(f'{DATA_DIR}/fairsasrec_item_embeddings.npy')
item_exposure_vec = np.zeros(N_ITEMS + 1, dtype=np.float32)

class FairSASRec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden_dim=256,
                 num_heads=4, num_layers=2, window=20,
                 dropout=0.2, fairness_lambda=0.3):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(window+1, emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.norm           = nn.LayerNorm(emb_dim)
        self.fairness_alpha = nn.Parameter(torch.ones(1))
        self.fairness_proj  = nn.Linear(emb_dim, 1)
        self.explain_proj   = nn.Linear(emb_dim, emb_dim)
        self.explain_scale  = emb_dim ** 0.5

    def encode(self, seq):
        B, W = seq.shape
        pos  = torch.arange(1, W+1, device=seq.device).unsqueeze(0)
        x    = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(W, device=seq.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.norm(x)

    def forward(self, seq, exp_vec):
        out   = self.encode(seq)
        state = out[:, -1, :]
        scores = state @ self.item_emb.weight.T
        exp_n  = torch.log1p(torch.FloatTensor(exp_vec))
        exp_n  = exp_n / (exp_n.max() + 1e-9)
        alpha  = F.softplus(self.fairness_alpha)
        pen    = exp_n * alpha
        sens   = torch.sigmoid(self.fairness_proj(state))
        return scores - sens * pen.unsqueeze(0), state, out

    def predict(self, seq, item_idx, exp_vec):
        fs, _, _ = self.forward(seq, exp_vec)
        return fs.gather(1, item_idx)

model = FairSASRec(N_ITEMS, fairness_lambda=LAMBDA)
model.load_state_dict(torch.load(
    'results/FairSASRec/fairsasrec_best.pt', map_location='cpu'))
model.eval()
print(f"FairSASRec loaded (lambda={LAMBDA})")

np.random.seed(42)
results   = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}
recs_dict = {}

def hit_at_k(ranked, pos, k):
    return 1.0 if pos in ranked[:k] else 0.0

def ndcg_at_k(ranked, pos, k):
    if pos in ranked[:k]:
        return 1.0 / np.log2(ranked[:k].index(pos) + 2)
    return 0.0

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen = train_set[uid] | val_set[uid] | {pos_item}
        pool = [i for i in range(1, N_ITEMS+1) if i not in seen]
        if len(pool) < 99:
            continue
        negs       = np.random.choice(pool, 99, replace=False).tolist()
        candidates = [pos_item] + negs

        history = user_history[uid][-WINDOW:]
        if len(history) < WINDOW:
            history = [0]*(WINDOW-len(history)) + history

        seq_t  = torch.LongTensor(history).unsqueeze(0)
        cand_t = torch.LongTensor(candidates).unsqueeze(0)
        scores = model.predict(seq_t, cand_t, item_exposure_vec).squeeze(0)

        ranked = sorted(range(len(candidates)),
                        key=lambda i: scores[i].item(), reverse=True)
        ranked_items = [candidates[i] for i in ranked]

        recs_dict[str(uid)] = ranked_items
        for k in K_LIST:
            results[k]['hits'].append(hit_at_k(ranked_items, pos_item, k))
            results[k]['ndcgs'].append(ndcg_at_k(ranked_items, pos_item, k))

print(f"\nFairSASRec Results (lambda={LAMBDA}):")
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} {'Gini':>7} {'Cov':>7}")
print("=" * 55)

for k in K_LIST:
    hr   = np.mean(results[k]['hits'])
    ndcg = np.mean(results[k]['ndcgs'])
    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_dp_eo(recs_k, user_gender,
                             {int(k): v for k,v in test_items.items()}, k)
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f} "
          f"{fairness['DP']:>7.4f} {fairness['EO']:>7.4f} "
          f"{gini:>7.4f} {cov:>7.4f}")
