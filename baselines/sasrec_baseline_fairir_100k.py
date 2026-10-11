"""
baselines/sasrec_baseline_fairir_100k.py
───────────────────────────────────────────────────────────────────────
SASRec on MovieLens-100K under the FairIR protocol (random 80/20,
full ranking, multi-item test sets). FAST ITERATION VERSION — a full
run takes ~10 minutes, so we use it to validate the recipe before
committing an hour to ML-1M.

KEY DESIGN POINT (the fix being tested here):
  Training and evaluation use DIFFERENT user representations.
    - Training:   last-position state only  -> clean next-item objective
    - Evaluation: 0.5*last + 0.5*mean-pooled state
  Rationale: the random split scatters test items across the user's
  whole timeline, so ground truth reflects GLOBAL taste. A pure
  last-position state captures only RECENT taste and collapses as
  training sharpens it. Blending at eval time only keeps the training
  objective intact while aligning inference with what the protocol
  actually measures.

What to watch: the probe HR must RISE monotonically across epochs.
If it rises here, apply the same recipe to ML-1M.

Outputs:
    results/SASRec-ml-100k-fairir/sasrec_best.pt / sasrec_final.pt
    results/SASRec-ml-100k-fairir/sasrec_fairir_results.json
    data/ml-100k-fairir/sasrec_item_embeddings.npy

USAGE
─────
    python baselines/sasrec_baseline_fairir_100k.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from metrics.fairness_metrics import (compute_exposure,
                                       gini_coefficient,
                                       coverage)
from metrics.user_fairness_metrics import compute_fairir_dp_eo

DATA_DIR = 'data/ml-100k-fairir'
OUT_DIR  = 'results/SASRec-ml-100k-fairir'
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {
    'emb_dim':    64,
    'hidden_dim': 256,
    'num_heads':  4,
    'num_layers': 2,
    'window':     20,      # 100k histories are shorter than 1M's
    'dropout':    0.2,
    'lr':         1e-3,
    'epochs':     100,
    'batch_size': 512,
    'l2':         1e-5,
    'eval_blend': 0.5,     # weight on last-position state at eval
}

K_LIST = [10, 20, 30, 40]

print("=" * 62)
print("SASRec — ML-100K, FairIR protocol (random 80/20, full ranking)")
print("=" * 62)

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']
print(f"Users: {N_USERS} | Items: {N_ITEMS}")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

SHIFT = 1   # model item id = data id + 1 ; 0 = padding

train_hist = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    train_hist[int(row['user_id'])].append(int(row['item_id']) + SHIFT)

eval_hist = defaultdict(list)
combined  = pd.concat([train, val]).sort_values('timestamp')
for _, row in combined.iterrows():
    eval_hist[int(row['user_id'])].append(int(row['item_id']) + SHIFT)

seen_items = defaultdict(set)
for _, row in combined.iterrows():
    seen_items[int(row['user_id'])].add(int(row['item_id']) + SHIFT)

test_sets = defaultdict(set)
for _, row in test.iterrows():
    test_sets[int(row['user_id'])].add(int(row['item_id']) + SHIFT)

# Gender — ml-100k u.user format: user_id|age|gender|occupation|zip
user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender = {}
with open(f'{DATA_DIR}/u.user', 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            raw_gender[int(parts[0])] = parts[2]
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}
n_m = sum(1 for g in user_gender.values() if g == 'M')
n_f = sum(1 for g in user_gender.values() if g == 'F')
print(f"Gender loaded — Male: {n_m} | Female: {n_f}")
assert n_m > 0 and n_f > 0, "Gender loading failed"


# ── SASRec ─────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(window + 1,  emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True, norm_first=True,
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

    # ── TRAINING path: last position only (clean next-item objective) ──
    def all_scores(self, seq):
        out   = self.forward(seq)
        state = out[:, -1, :]
        return state @ self.item_emb.weight.T

    def predict(self, seq, item_idx):
        return self.all_scores(seq).gather(1, item_idx)

    # ── EVALUATION path: blend last + mean-pooled (global taste) ───────
    def all_scores_eval(self, seq, blend=0.5):
        out  = self.forward(seq)                       # (B, W, d)
        mask = (seq > 0).unsqueeze(-1).float()         # (B, W, 1)
        state_mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        state_last = out[:, -1, :]
        state = blend * state_last + (1.0 - blend) * state_mean
        return state @ self.item_emb.weight.T


class SeqDataset(Dataset):
    def __init__(self, hist, window):
        self.samples = []
        for uid, h in hist.items():
            if len(h) < 2:
                continue
            for i in range(1, len(h)):
                seq = h[max(0, i - window):i]
                if len(seq) < window:
                    seq = [0] * (window - len(seq)) + seq
                self.samples.append((seq, h[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.LongTensor(seq), torch.LongTensor([target])


def get_eval_seq(uid):
    h = eval_hist.get(uid, [])[-CFG['window']:]
    if len(h) < CFG['window']:
        h = [0] * (CFG['window'] - len(h)) + h
    return torch.LongTensor(h).unsqueeze(0)


def fairir_hr_ndcg(topk_items, gt_set, k):
    topk = topk_items[:k]
    hits = sum(1 for it in topk if it in gt_set)
    hr_raw = hits / len(gt_set)
    hr_cap = hits / min(len(gt_set), k)
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, it in enumerate(topk) if it in gt_set)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(gt_set), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return hr_raw, hr_cap, ndcg


def full_rank_topk(model, uid, k_max=40):
    seq = get_eval_seq(uid)
    with torch.no_grad():
        scores = model.all_scores_eval(
            seq, blend=CFG['eval_blend']).squeeze(0).clone()
    scores[0] = -1e9
    seen = seen_items.get(uid, set())
    if seen:
        idx = torch.LongTensor(list(seen))
        scores[idx] = -1e9
    return scores.topk(k_max).indices.tolist()


def evaluate(model, users, k_list):
    model.eval()
    per_k = {k: {'hr_raw': [], 'hr_cap': [], 'ndcg': []} for k in k_list}
    recs  = {}
    for uid in users:
        gt = test_sets.get(uid)
        if not gt:
            continue
        topk = full_rank_topk(model, uid, k_max=max(k_list))
        recs[str(uid)] = topk
        for k in k_list:
            hr_r, hr_c, nd = fairir_hr_ndcg(topk, gt, k)
            per_k[k]['hr_raw'].append(hr_r)
            per_k[k]['hr_cap'].append(hr_c)
            per_k[k]['ndcg'].append(nd)
    out = {k: {m: float(np.mean(v)) for m, v in d.items()}
           for k, d in per_k.items()}
    return out, recs


# ── Train ──────────────────────────────────────────────────────────────
dataset    = SeqDataset(train_hist, CFG['window'])
dataloader = DataLoader(dataset, batch_size=CFG['batch_size'],
                        shuffle=True, num_workers=0)

model = SASRec(N_ITEMS, CFG['emb_dim'], CFG['hidden_dim'],
               CFG['num_heads'], CFG['num_layers'],
               CFG['window'], CFG['dropout'])

optimizer = optim.Adam(model.parameters(),
                       lr=CFG['lr'], weight_decay=CFG['l2'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['epochs'], eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(dataset)}\n")

all_test_users = sorted(test_sets.keys())
probe_users    = all_test_users[:400]
best_hr, best_state = 0.0, None

for epoch in range(CFG['epochs']):
    model.train()
    total, nb = 0.0, 0
    for seq, target in dataloader:
        neg = torch.randint(1, N_ITEMS + 1, target.shape)
        pos_s = model.predict(seq, target)
        neg_s = model.predict(seq, neg)
        loss  = (criterion(pos_s, torch.ones_like(pos_s)) +
                 criterion(neg_s, torch.zeros_like(neg_s)))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb += 1
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        probe, _ = evaluate(model, probe_users, [10])
        hr10 = probe[10]['hr_raw']
        print(f"Epoch {epoch+1:>3} | Loss {total/nb:.4f} | "
              f"HR@10(FairIR) {hr10:.4f} | "
              f"NDCG@10 {probe[10]['ndcg']:.4f}")
        if hr10 > best_hr:
            best_hr    = hr10
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

if best_state is None:
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

torch.save(best_state, f'{OUT_DIR}/sasrec_best.pt')
torch.save(model.state_dict(), f'{OUT_DIR}/sasrec_final.pt')
model.load_state_dict(best_state)
np.save(f'{DATA_DIR}/sasrec_item_embeddings.npy',
        model.item_emb.weight.data.numpy())

# ── Final full evaluation ──────────────────────────────────────────────
print("\nFinal evaluation on all test users (full ranking)...")
final, recs = evaluate(model, all_test_users, K_LIST)

print(f"\n{'K':<5} {'HR(raw)':>9} {'HR(cap)':>9} {'NDCG':>8} "
      f"{'DP':>8} {'EO':>8} {'Gini':>8} {'Cov':>8}")
print("-" * 68)

results_json = {}
for k in K_LIST:
    recs_k = {u: items[:k] for u, items in recs.items()}
    recs_k_data = {u: [i - SHIFT for i in items]
                   for u, items in recs_k.items()}
    test_sets_data = {u: {i - SHIFT for i in s}
                      for u, s in test_sets.items()}

    fair = compute_fairir_dp_eo(recs_k_data, user_gender,
                                 test_sets_data, N_ITEMS, k)
    exposure = compute_exposure(recs_k_data, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k_data, N_ITEMS, k)

    row = {**final[k], 'DP': fair['DP'], 'EO': fair['EO'],
           'Gini': gini, 'Coverage': cov}
    results_json[k] = row
    print(f"K={k:<3} {row['hr_raw']:>9.4f} {row['hr_cap']:>9.4f} "
          f"{row['ndcg']:>8.4f} {row['DP']:>8.4f} {row['EO']:>8.4f} "
          f"{gini:>8.4f} {cov:>8.4f}")

with open(f'{OUT_DIR}/sasrec_fairir_results.json', 'w') as f:
    json.dump({str(k): v for k, v in results_json.items()}, f, indent=2)

print("\n" + "=" * 62)
print("RECIPE CHECK (ML-100K is for iteration, not the paper table)")
print("=" * 62)
print("Did the probe HR RISE across epochs without collapsing?")
print("  YES -> apply the same recipe to ML-1M "
      "(sasrec_baseline_fairir.py)")
print("  NO  -> report the probe trajectory back before touching ML-1M")
print("=" * 62)
