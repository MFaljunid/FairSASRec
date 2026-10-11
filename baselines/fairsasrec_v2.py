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

# ══════════════════════════════════════════════════════════════════════
#  FairSASRec v2 — GENDER-AWARE fairness constraint
#
#  KEY CHANGE FROM v1:
#    v1 tracked a single global exposure vector and penalised its
#    variance. That is item-level fairness — it is BLIND to gender,
#    yet we were measuring gender-based DP/EO. Mismatch => no effect.
#
#    v2 tracks exposure SEPARATELY per gender group, penalises the
#    GAP between them, and adds a loss term that directly targets the
#    male/female hit-rate difference (which is exactly what DP is).
# ══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--fairness_lambda', type=float, default=0.3)
parser.add_argument('--dataset', type=str, default='ml-1m',
                    choices=['ml-1m', 'ml-100k'])
args = parser.parse_args()

DATA_DIR    = f'data/{args.dataset}'
RESULTS_DIR = ('results/FairSASRec-100k' if args.dataset == 'ml-100k'
               else 'results/FairSASRec')
RESULTS_CSV = f'{RESULTS_DIR}/lambda_sweep_results.csv'

CFG = {
    'emb_dim':         64,
    'hidden_dim':      256,
    'num_heads':       4,
    'num_layers':      2,
    'window':          20 if args.dataset == 'ml-1m' else 10,
    'dropout':         0.2,
    'lr':              1e-3,
    'epochs':          100,
    'batch_size':      1024,
    'l2':              1e-5,
    'fairness_lambda': args.fairness_lambda,
}

print("=" * 60)
print(f"FairSASRec v2 (gender-aware) — {args.dataset}")
print(f"lambda = {CFG['fairness_lambda']}  |  window = {CFG['window']}")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
val     = pd.read_csv(f'{DATA_DIR}/val.csv')
test    = pd.read_csv(f'{DATA_DIR}/test.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"Users: {N_USERS} | Items: {N_ITEMS}")

# ── Gender (format differs between ML-1M and ML-100K) ──────────────────
user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}

if args.dataset == 'ml-100k':
    # u.user format:  user_id|age|gender|occupation|zip
    raw_gender = {}
    with open(f'{DATA_DIR}/u.user', 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                raw_gender[int(parts[0])] = parts[2]
else:
    # users.dat format:  user_id::gender::age::occupation::zip
    raw_gender = load_user_gender(f'{DATA_DIR}/users.dat')

user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

n_m = sum(1 for g in user_gender.values() if g == 'M')
n_f = sum(1 for g in user_gender.values() if g == 'F')
print(f"Gender loaded — Male: {n_m} | Female: {n_f}")
assert n_m > 0 and n_f > 0, "Gender loading failed — DP/EO would be meaningless"

# Fast lookup tensor: 1.0 = male, 0.0 = female
gender_is_male = torch.zeros(N_USERS, dtype=torch.float32)
for uid, g in user_gender.items():
    if 0 <= uid < N_USERS:
        gender_is_male[uid] = 1.0 if g == 'M' else 0.0

# ── Build user histories ───────────────────────────────────────────────
user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

lens = [len(h) for h in user_history.values()]
print(f"History length — mean: {np.mean(lens):.1f} | "
      f"median: {np.median(lens):.0f} | "
      f"pct >= {CFG['window']}: "
      f"{100*np.mean([l >= CFG['window'] for l in lens]):.1f}%")

val_items  = {int(r['user_id']): int(r['item_id']) for _, r in val.iterrows()}
test_items = {int(r['user_id']): int(r['item_id']) for _, r in test.iterrows()}

val_set = defaultdict(set)
for uid, item in val_items.items():
    val_set[uid].add(item)

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

# ══════════════════════════════════════════════════════════════════════
#  GENDER-SEPARATED EXPOSURE  (the core fix)
#
#  v1 had:  item_exposure[i]  = how often item i was recommended
#  v2 has:  male_exposure[i]  = how often item i went to a male user
#           female_exposure[i]= how often item i went to a female user
#
#  The gap  |male[i] - female[i]| / (male[i] + female[i])  is exactly
#  the per-item disparity that DP aggregates. Penalising it moves DP.
# ══════════════════════════════════════════════════════════════════════
male_exposure   = np.zeros(N_ITEMS + 1, dtype=np.float32)
female_exposure = np.zeros(N_ITEMS + 1, dtype=np.float32)


def compute_exposure_gap(male_exp, female_exp):
    """
    Per-item gender exposure disparity in [0, 1].
    0 = item is shown equally to both groups (fair)
    1 = item is shown exclusively to one group (maximally unfair)
    Items never recommended get gap 0 (no disparity yet).
    """
    total = male_exp + female_exp
    gap   = np.zeros_like(total)
    mask  = total > 0
    gap[mask] = np.abs(male_exp[mask] - female_exp[mask]) / total[mask]
    return gap


# ══════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════
class FairSASRec(nn.Module):
    """
    SASRec backbone + gender-aware FairnessConstraintLayer
                    + AttentionExplainer

    FairnessConstraintLayer (v2):
        s_hat = s  -  rho_u * (alpha * gap)

      gap    : per-item gender exposure disparity (NOT global variance)
      alpha  : learned global penalty scale, softplus => alpha > 0
      rho_u  : learned per-user sensitivity gate, sigmoid => (0,1)

    When alpha -> 0 or rho -> 0 the layer vanishes and the model
    degenerates to plain SASRec. FairSASRec therefore strictly
    generalises its backbone.
    """
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout):
        super().__init__()
        self.n_items = n_items

        # ── SASRec backbone ────────────────────────────────────────────
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
        self.norm    = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.window  = window

        # ── FairnessConstraintLayer (NOVELTY) ──────────────────────────
        self.fairness_alpha = nn.Parameter(torch.ones(1))
        self.fairness_proj  = nn.Linear(emb_dim, 1)

        # ── AttentionExplainer ─────────────────────────────────────────
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

    def forward(self, seq, exposure_gap):
        """
        seq          : (B, W) item ids
        exposure_gap : (n_items+1,) per-item gender disparity in [0,1]
        """
        out    = self.encode(seq)
        state  = out[:, -1, :]                       # (B, d)
        scores = state @ self.item_emb.weight.T      # (B, N+1)

        gap_t = torch.as_tensor(exposure_gap,
                                dtype=torch.float32,
                                device=seq.device)
        alpha = F.softplus(self.fairness_alpha)      # > 0
        rho   = torch.sigmoid(self.fairness_proj(state))   # (B, 1)

        # Penalise items with high gender disparity
        fair_scores = scores - rho * (alpha * gap_t).unsqueeze(0)
        return fair_scores, state, out

    def predict(self, seq, item_idx, exposure_gap):
        fs, _, _ = self.forward(seq, exposure_gap)
        return fs.gather(1, item_idx)

    def explain(self, seq):
        """Attention weights over the history window."""
        out   = self.encode(seq)
        state = out[:, -1:, :]
        query = self.explain_proj(state)
        attn  = torch.bmm(out, query.transpose(1, 2)).squeeze(-1)
        return F.softmax(attn / self.explain_scale, dim=-1)


# ══════════════════════════════════════════════════════════════════════
#  Dataset — now carries user id so we can look up gender in the batch
# ══════════════════════════════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, user_history, window):
        self.samples = []
        for uid, history in user_history.items():
            if len(history) < 2:
                continue
            for i in range(1, len(history)):
                seq    = history[max(0, i - window):i]
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
                torch.LongTensor([uid]))


dataset    = SeqDataset(user_history, CFG['window'])
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
)

optimizer = optim.Adam(model.parameters(),
                       lr=CFG['lr'], weight_decay=CFG['l2'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['epochs'], eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss(reduction='none')

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(dataset)}\n")

best_hr    = 0.0
best_state = None
ndcg       = 0.0
np.random.seed(42)

exposure_gap = compute_exposure_gap(male_exposure, female_exposure)

for epoch in range(CFG['epochs']):
    model.train()
    total_loss = total_bce = total_fair = 0.0
    n_batches  = 0

    for seq, target, uid in dataloader:
        uid = uid.squeeze(1)                       # (B,)
        neg = torch.randint(1, N_ITEMS + 1, target.shape)

        fair_scores, _, _ = model.forward(seq, exposure_gap)

        pos_scores = fair_scores.gather(1, target)   # (B,1)
        neg_scores = fair_scores.gather(1, neg)      # (B,1)

        # ── Per-sample BCE (not reduced yet) ───────────────────────────
        pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
        neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
        per_sample = (pos_loss + neg_loss).squeeze(1)   # (B,)
        bce_loss   = per_sample.mean()

        # ══════════════════════════════════════════════════════════════
        #  GENDER-GAP LOSS  (the second core fix)
        #
        #  DP = |HR_male - HR_female|.  We cannot compute HR inside the
        #  training loop, but the per-sample BCE is a smooth surrogate:
        #  a group with systematically higher loss is a group the model
        #  serves worse. Penalising the gap between the two group means
        #  is therefore a differentiable proxy for DP.
        # ══════════════════════════════════════════════════════════════
        is_male   = gender_is_male[uid]                  # (B,)
        is_female = 1.0 - is_male

        n_male_b   = is_male.sum()
        n_female_b = is_female.sum()

        if n_male_b > 0 and n_female_b > 0:
            male_loss   = (per_sample * is_male).sum()   / n_male_b
            female_loss = (per_sample * is_female).sum() / n_female_b
            fairness_penalty = torch.abs(male_loss - female_loss)
        else:
            fairness_penalty = torch.tensor(0.0)

        loss = bce_loss + CFG['fairness_lambda'] * fairness_penalty

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce  += bce_loss.item()
        total_fair += fairness_penalty.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = total_loss / n_batches
    avg_bce  = total_bce  / n_batches
    avg_fair = total_fair / n_batches

    # ── Update GENDER-SEPARATED exposure from real top-10 lists ────────
    model.eval()
    with torch.no_grad():
        sample_uids = list(user_history.keys())[:1000]
        for uid in sample_uids:
            history = user_history[uid][-CFG['window']:]
            if len(history) < CFG['window']:
                history = [0] * (CFG['window'] - len(history)) + history
            seq_t    = torch.LongTensor(history).unsqueeze(0)
            fs, _, _ = model.forward(seq_t, exposure_gap)
            top10    = fs.squeeze(0).topk(10).indices.numpy()

            g = user_gender.get(uid, 'M')
            for item in top10:
                if 0 < item <= N_ITEMS:
                    if g == 'M':
                        male_exposure[item]   += 1
                    else:
                        female_exposure[item] += 1

    exposure_gap = compute_exposure_gap(male_exposure, female_exposure)
    model.train()

    # ── Evaluate ───────────────────────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        model.eval()
        hits, ndcgs = [], []
        hits_m, hits_f = [], []
        np.random.seed(42)
        sample_users = list(test_items.keys())[:500]

        with torch.no_grad():
            for uid in sample_users:
                pos_item = test_items[uid]
                seen = train_set[uid] | val_set[uid] | {pos_item}
                pool = [i for i in range(1, N_ITEMS + 1) if i not in seen]
                if len(pool) < 99:
                    continue
                negs       = np.random.choice(pool, 99, replace=False).tolist()
                candidates = [pos_item] + negs

                history = user_history[uid][-CFG['window']:]
                if len(history) < CFG['window']:
                    history = [0] * (CFG['window'] - len(history)) + history

                seq_t  = torch.LongTensor(history).unsqueeze(0)
                cand_t = torch.LongTensor(candidates).unsqueeze(0)
                scores = model.predict(seq_t, cand_t, exposure_gap).squeeze(0)

                order  = sorted(range(len(candidates)),
                                key=lambda i: scores[i].item(), reverse=True)
                ranked = [candidates[i] for i in order]

                hit = 1.0 if pos_item in ranked[:10] else 0.0
                nd  = (1.0 / np.log2(ranked[:10].index(pos_item) + 2)
                       if hit else 0.0)
                hits.append(hit)
                ndcgs.append(nd)

                if user_gender.get(uid, 'M') == 'M':
                    hits_m.append(hit)
                else:
                    hits_f.append(hit)

        hr   = np.mean(hits)
        ndcg = np.mean(ndcgs)
        # Live DP estimate — this is the number we actually care about
        dp_live = abs(np.mean(hits_m) - np.mean(hits_f)) \
                  if hits_m and hits_f else float('nan')

        print(f"Epoch {epoch+1:>3} | Loss {avg_loss:.4f} "
              f"(BCE {avg_bce:.4f}, Fair {avg_fair:.4f}) | "
              f"HR@10 {hr:.4f} | NDCG@10 {ndcg:.4f} | "
              f"DP@10 {dp_live:.4f}")

        if hr > best_hr:
            best_hr    = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()

# ── Save ───────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

if best_state is None:
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

torch.save(best_state, f'{RESULTS_DIR}/fairsasrec_best.pt')
torch.save(model.state_dict(), f'{RESULTS_DIR}/fairsasrec_final.pt')
np.save(f'{RESULTS_DIR}/male_exposure.npy',   male_exposure)
np.save(f'{RESULTS_DIR}/female_exposure.npy', female_exposure)

model.load_state_dict(best_state)
np.save(f'{DATA_DIR}/fairsasrec_item_embeddings.npy',
        model.item_emb.weight.data.numpy())

print(f"\nBest HR@10: {best_hr:.4f}")

# ── Final DP/EO on the FULL test set with the BEST model ───────────────
print("\nComputing DP/EO on full test set (best model)...")
model.eval()
dp_eo_recs = {}
hits_m_all, hits_f_all = [], []

with torch.no_grad():
    for uid in test_set.keys():
        history = user_history.get(uid, [])[-CFG['window']:]
        if len(history) < CFG['window']:
            history = [0] * (CFG['window'] - len(history)) + history
        seq_t    = torch.LongTensor(history).unsqueeze(0)
        fs, _, _ = model.forward(seq_t, exposure_gap)

        scores = fs.squeeze(0).clone()
        seen   = train_set[uid] | val_set[uid]
        if seen:
            idx = torch.LongTensor(list(seen))
            idx = idx[(idx >= 0) & (idx <= N_ITEMS)]
            if len(idx) > 0:
                scores[idx] = -1e9

        top10 = scores.topk(10).indices.numpy().tolist()
        dp_eo_recs[str(uid)] = top10

        # HR-difference DP (the metric reported in the paper)
        hit = 1.0 if test_items[uid] in top10 else 0.0
        if user_gender.get(uid, 'M') == 'M':
            hits_m_all.append(hit)
        else:
            hits_f_all.append(hit)

hr_male   = float(np.mean(hits_m_all)) if hits_m_all else 0.0
hr_female = float(np.mean(hits_f_all)) if hits_f_all else 0.0
dp_hr     = abs(hr_male - hr_female)

fairness  = compute_fairir_dp_eo(dp_eo_recs, user_gender,
                                 test_set, N_ITEMS, 10)
dp_expo   = fairness['DP']
eo_expo   = fairness['EO']

print(f"  HR@10 male   : {hr_male:.4f}  (n={len(hits_m_all)})")
print(f"  HR@10 female : {hr_female:.4f}  (n={len(hits_f_all)})")
print(f"  DP@10 (HR-diff)      : {dp_hr:.4f}   <-- report this")
print(f"  DP@10 (item-exposure): {dp_expo:.4f}")
print(f"  EO@10 (item-exposure): {eo_expo:.4f}")

# ── Append to sweep CSV ────────────────────────────────────────────────
file_exists = os.path.isfile(RESULTS_CSV)
with open(RESULTS_CSV, mode='a', newline='') as f:
    w = csv.writer(f)
    if not file_exists:
        w.writerow(['timestamp', 'dataset', 'fairness_lambda', 'epochs',
                    'best_hr10', 'final_ndcg10',
                    'final_loss', 'final_bce', 'final_fair_penalty',
                    'HR_male', 'HR_female',
                    'DP_hrdiff', 'DP_exposure', 'EO_exposure'])
    w.writerow([
        time.strftime('%Y-%m-%d %H:%M:%S'),
        args.dataset,
        CFG['fairness_lambda'],
        CFG['epochs'],
        round(best_hr, 4),
        round(ndcg, 4),
        round(avg_loss, 4),
        round(avg_bce, 4),
        round(avg_fair, 4),
        round(hr_male, 4),
        round(hr_female, 4),
        round(dp_hr, 4),
        round(dp_expo, 4),
        round(eo_expo, 4),
    ])

print(f"\nResults appended to {RESULTS_CSV}")
