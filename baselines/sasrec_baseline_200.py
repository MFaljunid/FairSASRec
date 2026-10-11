import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

import random

# ── Reproducibility ────────────────────────────────────────────────────
SEED = int(os.environ.get('SEED', 42))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type=='cuda' else "")
      + f" | seed: {SEED}")

# DATA_DIR = 'data/ml-1m'
# OUT_DIR  = 'results/SASRec-200'
DATA_DIR = os.environ.get('DATA_DIR', 'data/ml-1m')
OUT_DIR  = os.environ.get('OUT_DIR', 'results/SASRec-200')
CFG = {
    'emb_dim':    64,
    'hidden_dim': 256,
    'num_heads':  4,
    'num_layers': 2,
    'window':     20,      # ← reduce from 50
    'dropout':    0.2,
    'lr':         1e-3,
    'epochs':     200,     # matched to FairSASRec budget
    'batch_size': 1024,    # ← increase from 256
    'l2':         1e-5,
}

# ── Load data ──────────────────────────────────────────────────────────
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
val     = pd.read_csv(f'{DATA_DIR}/val.csv')
test    = pd.read_csv(f'{DATA_DIR}/test.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"Users: {N_USERS} | Items: {N_ITEMS}")

# ── Build user histories ───────────────────────────────────────────────
user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

val_items  = {int(r['user_id']): int(r['item_id'])
              for _, r in val.iterrows()}
test_items = {int(r['user_id']): int(r['item_id'])
              for _, r in test.iterrows()}

val_set  = defaultdict(set)
for uid, item in val_items.items():
    val_set[uid].add(item)

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

# ── SASRec Model ───────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout):
        super().__init__()
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

    def forward(self, seq):
        # seq: (B, window) item ids
        B, W = seq.shape
        positions = torch.arange(1, W+1,
                                  device=seq.device).unsqueeze(0)
        x = self.dropout(
            self.item_emb(seq) + self.pos_emb(positions))

        mask = nn.Transformer.generate_square_subsequent_mask(
            W, device=seq.device)

        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)

        return self.norm(x)  # (B, W, emb_dim)

    def predict(self, seq, item_indices):
        # seq: (B, W), item_indices: (B, n_candidates)
        out        = self.forward(seq)
        state      = out[:, -1, :]              # (B, emb_dim)
        item_embs  = self.item_emb(item_indices) # (B, n_cand, emb_dim)
        scores     = torch.bmm(item_embs,
                               state.unsqueeze(-1)).squeeze(-1)
        return scores

# ── Dataset ────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, user_history, n_items, window):
        self.samples  = []
        self.n_items  = n_items
        self.window   = window

        for uid, history in user_history.items():
            if len(history) < 2:
                continue
            for i in range(1, len(history)):
                seq    = history[max(0, i-window):i]
                target = history[i]
                # Pad
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

model = SASRec(
    n_items=N_ITEMS,
    emb_dim=CFG['emb_dim'],
    hidden_dim=CFG['hidden_dim'],
    num_heads=CFG['num_heads'],
    num_layers=CFG['num_layers'],
    window=CFG['window'],
    dropout=CFG['dropout'],
).to(DEVICE)

optimizer = optim.Adam(model.parameters(),
                       lr=CFG['lr'],
                       weight_decay=CFG['l2'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['epochs'], eta_min=1e-5)

criterion = nn.BCEWithLogitsLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(dataset)}")

best_hr   = 0.0
best_state = None

np.random.seed(42)

for epoch in range(CFG['epochs']):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for seq, target, uid in dataloader:
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)
        # Sample negatives
        neg = torch.randint(1, N_ITEMS+1, target.shape, device=DEVICE)

        pos_scores = model.predict(seq, target)
        neg_scores = model.predict(seq, neg)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        loss = (criterion(pos_scores, pos_labels) +
                criterion(neg_scores, neg_labels))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = total_loss / n_batches

    # Evaluate every 10 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        hits = []
        np.random.seed(42)

        sample_users = list(test_items.keys())[:200]

        with torch.no_grad():
            for uid in sample_users:
                pos_item = test_items[uid]
                seen     = train_set[uid] | val_set[uid] | {pos_item}
                pool     = [i for i in range(1, N_ITEMS+1)
                            if i not in seen]
                if len(pool) < 99:
                    continue

                negs = np.random.choice(pool, 99,
                                         replace=False).tolist()
                candidates = [pos_item] + negs

                history = user_history[uid][-CFG['window']:]
                if len(history) < CFG['window']:
                    history = [0]*(CFG['window']-len(history)) + history

                seq_t  = torch.LongTensor(history).unsqueeze(0).to(DEVICE)
                cand_t = torch.LongTensor(candidates).unsqueeze(0).to(DEVICE)
                scores = model.predict(seq_t, cand_t).squeeze(0)
                ranked = sorted(candidates,
                                key=lambda x: scores[
                                    candidates.index(x)].item(),
                                reverse=True)
                hits.append(1.0 if pos_item in ranked[:10] else 0.0)

        hr = np.mean(hits)
        print(f"Epoch {epoch+1:>3} | Loss: {avg_loss:.4f} | "
              f"HR@10: {hr:.4f}")

        if hr > best_hr:
            best_hr    = hr
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

# ── Save best model ────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if best_state is None:
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
torch.save(best_state,   f'{OUT_DIR}/sasrec_best.pt')
torch.save(model.state_dict(), f'{OUT_DIR}/sasrec_final.pt')

# Save embeddings for RL
model.load_state_dict(best_state)
item_emb = model.item_emb.weight.data.cpu().numpy()
np.save(f'{OUT_DIR}/sasrec_item_embeddings.npy', item_emb)

print(f"\nBest HR@10: {best_hr:.4f}")
print(f"Saved to {OUT_DIR}/")
print(f"Item embeddings saved to {OUT_DIR}/sasrec_item_embeddings.npy")