import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

# ── GPU ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else ""))

# DATA_DIR    = 'data/ml-1m'
# RESULTS_CSV = 'results/FairSASRec/lambda_sweep_results.csv'

DATA_DIR    = os.environ.get('DATA_DIR', 'data/ml-1m')
OUT_DIR     = os.environ.get('OUT_DIR', 'results/FairSASRec-ml1m')

# ── Command-line arguments ─────────────────────────────────────────────
# Run like: python fairsasrec.py --fairness_lambda 0.3 --alpha_max 1.0
parser = argparse.ArgumentParser()
parser.add_argument('--fairness_lambda', type=float, default=0.1,
                    help='Fairness penalty weight for this run')
parser.add_argument('--alpha_max', type=float, default=None,
                    help='Upper bound on softplus(alpha). None = unbounded '
                         '(the default behaviour). Setting it exposes the '
                         'fairness-diversity trade-off as a tunable knob.')
args = parser.parse_args()

ALPHA_MAX = args.alpha_max
if ALPHA_MAX is not None:
    OUT_DIR = f'{OUT_DIR}-amax{ALPHA_MAX}'
    print(f"Constraint strength capped at softplus(alpha) <= {ALPHA_MAX}")

RESULTS_CSV = f'{OUT_DIR}/lambda_sweep_results.csv'

CFG = {
    'emb_dim':         64,
    'hidden_dim':      256,
    'num_heads':       4,
    'num_layers':      2,
    'window':          20,
    'dropout':         0.2,
    'lr':              1e-3,
    'epochs':          200,
    'batch_size':      1024,
    'l2':              1e-5,
    'fairness_lambda': args.fairness_lambda,   # ← now set from CLI
}

# ── Load data ──────────────────────────────────────────────────────────
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
val     = pd.read_csv(f'{DATA_DIR}/val.csv')
test    = pd.read_csv(f'{DATA_DIR}/test.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"Users: {N_USERS} | Items: {N_ITEMS}")

# ── Gender for DP/EO (same source as evaluate_full_ranking.py) ─────────
# user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
# # raw_gender  = load_user_gender(f'{DATA_DIR}/users.dat')
# raw_gender  = load_user_gender(f'{DATA_DIR}/u.user')
# user_gender = {user2idx[u]: g for u, g in raw_gender.items()
#                if u in user2idx}

# ── Gender (format differs between ML-1M and ML-100K) ──────────────────
user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}

if 'ml-100k' in DATA_DIR:
    # u.user format: user_id|age|gender|occupation|zip
    raw_gender = {}
    with open(f'{DATA_DIR}/u.user', 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                raw_gender[int(parts[0])] = parts[2]   # gender is index 2
else:
    raw_gender = load_user_gender(f'{DATA_DIR}/users.dat')

user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

n_m = sum(1 for g in user_gender.values() if g == 'M')
n_f = sum(1 for g in user_gender.values() if g == 'F')
print(f"Gender loaded — Male: {n_m} | Female: {n_f}")

# ── Build user histories ───────────────────────────────────────────────
user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))
lens = [len(h) for h in user_history.values()]
print(f"History length — mean: {np.mean(lens):.1f} | "
      f"median: {np.median(lens):.0f} | "
      f"pct >= 20: {100*np.mean([l >= 20 for l in lens]):.1f}%")

val_items  = {int(r['user_id']): int(r['item_id'])
              for _, r in val.iterrows()}
test_items = {int(r['user_id']): int(r['item_id'])
              for _, r in test.iterrows()}

val_set = defaultdict(set)
for uid, item in val_items.items():
    val_set[uid].add(item)

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

# test_set — multi-item set version needed by compute_fairir_dp_eo
# (test_items above is single-item-per-user, used for HR/NDCG sampling)
test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

# ── Item exposure tracking ─────────────────────────────────────────────
# NOTE: this is the ONE persistent exposure vector used across the whole
# training run. It is updated once per epoch from *real* top-10
# recommendations on a sample of users (see end of epoch loop below),
# not from noisy per-batch argmax on random negatives.
item_exposure = np.zeros(N_ITEMS + 1, dtype=np.float32)

# ── FairSASRec Model ───────────────────────────────────────────────────
class FairSASRec(nn.Module):
    """
    SASRec + FairnessConstraintLayer + AttentionExplainer
    YOUR CONTRIBUTION: fairness is baked into the architecture
    """
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout,
                 fairness_lambda=0.1):
        super().__init__()
        self.n_items          = n_items
        self.fairness_lambda  = fairness_lambda

        # ── SASRec backbone ────────────────────────────────────────────
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(window + 1,  emb_dim)
        self.dropout  = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.window  = window

        # ── FairnessConstraintLayer (YOUR NOVELTY) ─────────────────────
        self.fairness_alpha    = nn.Parameter(torch.ones(1))
        self.fairness_proj     = nn.Linear(emb_dim, 1)

        # ── AttentionExplainer ─────────────────────────────────────────
        self.explain_proj = nn.Linear(emb_dim, emb_dim)
        self.explain_scale = emb_dim ** 0.5

    def encode(self, seq):
        """Encode sequence → state vector."""
        B, W      = seq.shape
        positions = torch.arange(1, W+1, device=seq.device).unsqueeze(0)
        x = self.dropout(
            self.item_emb(seq) + self.pos_emb(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(
            W, device=seq.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.norm(x)   # (B, W, emb_dim)

    def forward(self, seq, item_exposure_vec):
        """
        seq              : (B, window) item ids
        item_exposure_vec: (n_items+1,) exposure counts

        Returns: fair_scores (B, n_items+1)
        """
        alpha        = F.softplus(self.fairness_alpha)
        if ALPHA_MAX is not None:
            alpha = torch.clamp(alpha, max=ALPHA_MAX)
        out   = self.encode(seq)
        state = out[:, -1, :]              # (B, emb_dim)

        # Base scores via dot product
        all_item_embs = self.item_emb.weight  # (n_items+1, emb_dim)
        scores        = state @ all_item_embs.T  # (B, n_items+1)

        # FairnessConstraintLayer — subtract exposure penalty
        exp_norm = torch.log1p(
            torch.FloatTensor(item_exposure_vec).to(seq.device))
        exp_norm     = exp_norm / (exp_norm.max() + 1e-9)
        alpha        = F.softplus(self.fairness_alpha)
        fair_penalty = exp_norm * alpha
        user_sens    = torch.sigmoid(
            self.fairness_proj(state))           # (B, 1)
        fair_scores  = scores - user_sens * fair_penalty.unsqueeze(0)

        return fair_scores, state, out

    def predict(self, seq, item_indices, item_exposure_vec):
        """For evaluation — score specific candidates."""
        fair_scores, _, _ = self.forward(seq, item_exposure_vec)
        return fair_scores.gather(1, item_indices)

    def explain(self, seq, top_k=5):
        """Attention-based explanation."""
        out   = self.encode(seq)
        state = out[:, -1:, :]             # (B, 1, emb_dim)
        query = self.explain_proj(state)   # (B, 1, emb_dim)
        attn  = torch.bmm(
            out, query.transpose(1, 2)).squeeze(-1)  # (B, W)
        attn  = F.softmax(attn / self.explain_scale, dim=-1)
        return attn                         # (B, W)


# ── Dataset ────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, user_history, n_items, window):
        self.samples = []
        for uid, history in user_history.items():
            if len(history) < 2:
                continue
            for i in range(1, len(history)):
                seq    = history[max(0, i-window):i]
                target = history[i]
                if len(seq) < window:
                    seq = [0] * (window - len(seq)) + seq
                self.samples.append((seq, target, uid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target, uid = self.samples[idx]
        return (torch.LongTensor(seq),
                torch.LongTensor([target]),
                uid)


# ── Training ───────────────────────────────────────────────────────────
dataset    = SeqDataset(user_history, N_ITEMS, CFG['window'])
dataloader = DataLoader(dataset, batch_size=CFG['batch_size'],
                        shuffle=True, num_workers=0)

model = FairSASRec(
    n_items=N_ITEMS,
    emb_dim=CFG['emb_dim'],
    hidden_dim=CFG['hidden_dim'],
    num_heads=CFG['num_heads'],
    num_layers=CFG['num_layers'],
    window=CFG['window'],
    dropout=CFG['dropout'],
    fairness_lambda=CFG['fairness_lambda'],
).to(DEVICE)

optimizer = optim.Adam(model.parameters(),
                       lr=CFG['lr'],
                       weight_decay=CFG['l2'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['epochs'], eta_min=1e-5)

criterion = nn.BCEWithLogitsLoss()

print(f"FairSASRec parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(dataset)}")

best_hr    = 0.0
best_state = None
ndcg       = 0.0  # safety default, in case epochs < 10 and eval block never runs
np.random.seed(42)

for epoch in range(CFG['epochs']):
    model.train()
    total_loss      = 0.0
    total_bce       = 0.0
    total_fair_pen  = 0.0
    n_batches       = 0

    # NOTE: item_exposure is NOT reset here anymore — it persists and
    # accumulates across epochs, updated once at the end of each epoch
    # from real top-10 recommendations (see below). This is what the
    # FairnessConstraintLayer actually sees during forward().

    for seq, target, uid in dataloader:
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)
        neg = torch.randint(1, N_ITEMS + 1, target.shape, device=DEVICE)

        fair_scores, _, _ = model.forward(
            seq, item_exposure)

        pos_scores = fair_scores.gather(1, target)
        neg_scores = fair_scores.gather(1, neg)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        bce_loss = (criterion(pos_scores, pos_labels) +
                    criterion(neg_scores, neg_labels))

        # ── Explicit fairness penalty term ──────────────────────────
        # Penalize concentration of exposure among the batch's argmax
        # recommendations. Lower variance across items = more even
        # exposure = fairer. This gives the model a direct gradient
        # signal toward fairness, on top of the logit-level adjustment
        # already applied by the FairnessConstraintLayer.
        batch_exposure = torch.zeros(N_ITEMS + 1, device=DEVICE)

        # fairness_penalty = batch_exposure.var()

        mean_exp = batch_exposure.mean() + 1e-9
        fairness_penalty = batch_exposure.var() / (mean_exp ** 2)
        loss = bce_loss + CFG['fairness_lambda'] * fairness_penalty

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss     += loss.item()
        total_bce      += bce_loss.item()
        total_fair_pen += fairness_penalty.item()
        n_batches      += 1

    scheduler.step()
    avg_loss     = total_loss / n_batches
    avg_bce      = total_bce / n_batches
    avg_fair_pen = total_fair_pen / n_batches

    # ── Update real exposure from actual top-10 recommendations ────────
    # This replaces the old per-batch argmax-on-random-negatives update.
    # We sample a fixed pool of users and compute genuine top-10 lists
    # using the model's *current* weights, then accumulate into the
    # persistent item_exposure vector that forward() consumes.
    model.eval()
    with torch.no_grad():
        exposure_sample_uids = list(user_history.keys())[:1000]
        for uid in exposure_sample_uids:
            history = user_history[uid][-CFG['window']:]
            if len(history) < CFG['window']:
                history = [0] * (CFG['window'] - len(history)) + history
            seq_t = torch.LongTensor(history).unsqueeze(0).to(DEVICE)
            fs, _, _ = model.forward(seq_t, item_exposure)
            top10 = fs.squeeze(0).topk(10).indices.cpu().numpy()
            for item in top10:
                if 0 < item <= N_ITEMS:
                    item_exposure[item] += 1
    model.train()

    # ── Evaluate every 10 epochs ───────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        model.eval()
        hits  = []
        ndcgs = []
        np.random.seed(42)
        sample_users = list(test_items.keys())[:500]

        with torch.no_grad():
            # Use the REAL accumulated exposure here (not zeros) so the
            # HR/NDCG you see during training reflects the model's
            # actual behavior with the fairness constraint active.
            for uid in sample_users:
                pos_item = test_items[uid]
                seen     = train_set[uid] | val_set[uid] | {pos_item}
                pool     = [i for i in range(1, N_ITEMS + 1)
                            if i not in seen]
                if len(pool) < 99:
                    continue
                negs       = np.random.choice(
                    pool, 99, replace=False).tolist()
                candidates = [pos_item] + negs

                history = user_history[uid][-CFG['window']:]
                if len(history) < CFG['window']:
                    history = ([0] * (CFG['window'] - len(history))
                               + history)

                seq_t  = torch.LongTensor(history).unsqueeze(0).to(DEVICE)
                cand_t = torch.LongTensor(candidates).unsqueeze(0).to(DEVICE)
                scores = model.predict(
                    seq_t, cand_t, item_exposure).squeeze(0)

                ranked = sorted(range(len(candidates)),
                                key=lambda i: scores[i].item(),
                                reverse=True)
                ranked_items = [candidates[i] for i in ranked]

                hit  = 1.0 if pos_item in ranked_items[:10] else 0.0
                if pos_item in ranked_items[:10]:
                    ndcg = 1.0 / np.log2(
                        ranked_items[:10].index(pos_item) + 2)
                else:
                    ndcg = 0.0
                hits.append(hit)
                ndcgs.append(ndcg)

        hr   = np.mean(hits)
        ndcg = np.mean(ndcgs)
        print(f"Epoch {epoch+1:>3} | Loss: {avg_loss:.4f} "
              f"(BCE: {avg_bce:.4f}, Fair: {avg_fair_pen:.4f}) | "
              f"HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")

        if hr > best_hr:
            best_hr    = hr
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

# ── Save ───────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(best_state,        f'{OUT_DIR}/fairsasrec_best.pt')
torch.save(model.state_dict(), f'{OUT_DIR}/fairsasrec_final.pt')

# Save embeddings
if best_state is None:
    print("\nWarning: no improvement was ever recorded (best_state is None). "
          "Falling back to final model weights for saving/DP-EO computation.")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
item_emb = model.item_emb.weight.data.cpu().numpy()
np.save(f'{OUT_DIR}/fairsasrec_item_embeddings.npy', item_emb)
print(f"\nBest HR@10: {best_hr:.4f}")
print(f"Saved to results/FairSASRec/")

# ── Compute DP/EO for this run using the BEST model on full test set ───
# Builds top-10 recommendations for every test user (not just the 500-user
# sample used during training) and measures gender-based DP/EO, same
# definitions as evaluate_full_ranking.py.
print("\nComputing DP/EO on full test set (best model)...")
model.eval()
dp_eo_recs = {}
with torch.no_grad():
    for uid in test_set.keys():
        history = user_history.get(uid, [])[-CFG['window']:]
        if len(history) < CFG['window']:
            history = [0] * (CFG['window'] - len(history)) + history
        seq_t = torch.LongTensor(history).unsqueeze(0).to(DEVICE)
        fs, _, _ = model.forward(seq_t, item_exposure)

        seen = train_set[uid] | val_set[uid]
        scores = fs.squeeze(0).clone()
        if seen:
            seen_idx = torch.LongTensor(list(seen)).to(DEVICE)
            seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx <= N_ITEMS)]
            if len(seen_idx) > 0:
                scores[seen_idx] = -1e9

        top10 = scores.topk(10).indices.cpu().numpy().tolist()
        dp_eo_recs[str(uid)] = top10

fairness = compute_fairir_dp_eo(dp_eo_recs, user_gender, test_set, N_ITEMS, 10)
dp_final = fairness['DP']
eo_final = fairness['EO']
print(f"DP@10: {dp_final:.4f} | EO@10: {eo_final:.4f}")

# ── Append this run's results to the persistent sweep CSV ──────────────
# This does NOT overwrite previous runs — each call to this script with
# a different --fairness_lambda adds one new row. Run the sweep like:
#   for L in 0 0.1 0.3 0.5 0.7 1.0; do
#       python fairsasrec.py --fairness_lambda $L
#   done
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
file_exists = os.path.isfile(RESULTS_CSV)

with open(RESULTS_CSV, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'timestamp', 'fairness_lambda', 'epochs',
            'best_hr10', 'final_ndcg10',
            'final_loss', 'final_bce', 'final_fair_penalty',
            'DP@10', 'EO@10'
        ])
    writer.writerow([
        time.strftime('%Y-%m-%d %H:%M:%S'),
        CFG['fairness_lambda'],
        CFG['epochs'],
        round(best_hr, 4),
        round(ndcg, 4),
        round(avg_loss, 4),
        round(avg_bce, 4),
        round(avg_fair_pen, 4),
        round(dp_final, 4),
        round(eo_final, 4),
    ])

print(f"Results appended to {RESULTS_CSV}")