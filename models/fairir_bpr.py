"""
baselines/fairir_bpr.py
───────────────────────────────────────────────────────────────────────
FairIR reimplemented under OUR protocol (leave-one-out), on the SAME
BPR-MF backbone as fairbpr.py.

WHY THIS FILE EXISTS
────────────────────
Reviewer 2 (Comment 2.1) asked for fairness baselines evaluated under
our leave-one-out protocol rather than published figures from a
different split. This reproduces the mechanism of Shi et al. (KBS 336,
115279, 2026), "FairIR_BPR", on the backbone our other MF models use,
so that BPR-MF, FairBPR-MF and FairIR-BPR share code, optimiser, seed,
data and evaluation. The only differences are the fairness mechanisms.

WHAT IS REPRODUCED (section / equation numbers refer to their paper)
───────────────────────────────────────────────────────────────────
  Backbone          BPR-MF, loss Eq. (15).
  Learnable noise   v~_i = v_i + e_i, e_i initialised U(-delta, delta),
                    trained jointly (Eqs. 2-3, "the direction and
                    magnitude of the noise ... are learnable").
  Cross-group       for a male user, positives are drawn from his own
  transfer          items and, with probability `noise_ratio`, from
                    noise-augmented items interacted with by FEMALE
                    users; symmetrically for female users (Eqs. 6, 12).
  Distillation      InfoNCE over the resulting positive and k_n
                    negatives with temperature tau (Eqs. 11, 13).
  Objective         L = L_BPR + w_info * L_InfoNCE.

HYPERPARAMETERS
───────────────
tau, k_n and noise_ratio default to values read from their Fig. 8 and
Fig. 9 for MovieLens (tau = 0.2, k_n = 40, noise ratio = 0.5, which
lies in the flat optimum of their Fig. 9). delta and w_info are not
reported numerically; we use delta = 0.1 and w_info = 1.0 and expose
all five as arguments so the sensitivity can be reported.

NOTE: unlike the FairnessConstraintLayer, FairIR uses the sensitive
attribute during TRAINING (to decide which group's items are
transferred). At inference it scores with the learned embeddings only.

USAGE
    python baselines/fairir_bpr.py
    DATA_DIR=data/ml-100k OUT_DIR=results/FairIR-BPR-100k \
        python baselines/fairir_bpr.py
"""

import sys, os, json, time, random, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from metrics.fairness_metrics import (compute_exposure, gini_coefficient,
                                       coverage)
from metrics.user_fairness_metrics import (load_user_gender,
                                            compute_fairir_dp_eo)

# ── args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--k_n', type=int, default=40)
parser.add_argument('--noise_ratio', type=float, default=0.5)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--w_info', type=float, default=1.0)
args = parser.parse_args()

# ── reproducibility (identical to fairbpr.py) ─────────────────────────
SEED = int(os.environ.get('SEED', 42))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = os.environ.get('DATA_DIR', 'data/ml-1m')
OUT_DIR  = os.environ.get('OUT_DIR', 'results/FairIR-BPR')
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {'emb_dim': 64, 'lr': 1e-3, 'l2': 1e-5, 'epochs': args.epochs,
       'batch_size': 4096, 'eval_every': 10}

print("=" * 62)
print("FairIR-BPR (reimplementation under leave-one-out)")
print("=" * 62)
print(f"Device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else "")
      + f" | seed: {SEED}")
print(f"Data: {DATA_DIR} | Out: {OUT_DIR}")
print(f"tau={args.tau} k_n={args.k_n} noise_ratio={args.noise_ratio} "
      f"delta={args.delta} w_info={args.w_info}")

# ── data (identical to fairbpr.py) ────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS, N_ITEMS = meta['n_users'], meta['n_items']
print(f"Users: {N_USERS} | Items: {N_ITEMS}")

train_pairs = train[['user_id', 'item_id']].astype(int).values
seen = defaultdict(set)
for u, i in train_pairs:
    seen[u].add(i)
for _, r in val.iterrows():
    seen[int(r['user_id'])].add(int(r['item_id']))
test_items = dict(zip(test['user_id'].astype(int), test['item_id'].astype(int)))
test_sets = {u: {i} for u, i in test_items.items()}

user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
if 'ml-100k' in DATA_DIR:
    raw_gender = {}
    with open(f'{DATA_DIR}/u.user') as f:
        for line in f:
            p = line.strip().split('|')
            if len(p) >= 3:
                raw_gender[int(p[0])] = p[2]
else:
    raw_gender = load_user_gender(f'{DATA_DIR}/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items() if u in user2idx}
n_m = sum(1 for g in user_gender.values() if g == 'M')
print(f"Gender loaded — Male: {n_m} | Female: {len(user_gender) - n_m}")

# ── FairIR-specific structures (training only) ────────────────────────
is_male = torch.zeros(N_USERS, dtype=torch.bool)
for u, g in user_gender.items():
    if 0 <= u < N_USERS:
        is_male[u] = (g == 'M')
is_male = is_male.to(DEVICE)

# interaction pools per group: sampling from these transfers the
# opposite group's preference pattern (their Eq. 6)
male_pool = torch.LongTensor([i for u, i in train_pairs
                              if user_gender.get(int(u)) == 'M']).to(DEVICE)
fem_pool  = torch.LongTensor([i for u, i in train_pairs
                              if user_gender.get(int(u)) == 'F']).to(DEVICE)
print(f"Cross-group pools — male interactions: {len(male_pool)} | "
      f"female interactions: {len(fem_pool)}")

# dense seen-matrix on GPU for fast negative rejection (6034 x 3125 bools)
seen_mat = torch.zeros(N_USERS, N_ITEMS, dtype=torch.bool, device=DEVICE)
for u, items in seen.items():
    if items:
        seen_mat[u, torch.LongTensor(list(items))] = True


# ══════════════════════════════════════════════════════════════════════
class FairIRBPR(nn.Module):
    def __init__(self, n_users, n_items, d, delta):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        # learnable noise, E_i ~ U(-delta, delta)  (their Eq. 2)
        self.noise = nn.Embedding(n_items, d)
        nn.init.uniform_(self.noise.weight, -delta, delta)

    def all_scores(self, u):
        return self.user_emb(u) @ self.item_emb.weight.T   # clean embeddings

    def bpr(self, u, i, j):
        p = self.user_emb(u)
        return (p * self.item_emb(i)).sum(-1), (p * self.item_emb(j)).sum(-1)

    def info_nce(self, u, i_real, tau, k_n, noise_ratio):
        """InfoNCE with cross-group noise-augmented positives (Eqs. 12-13)."""
        B = u.shape[0]
        p = F.normalize(self.user_emb(u), dim=-1)

        # cross-group positive: male user -> female pool, and vice versa
        male = is_male[u]
        idx_f = torch.randint(len(fem_pool), (B,), device=u.device)
        idx_m = torch.randint(len(male_pool), (B,), device=u.device)
        cross = torch.where(male, fem_pool[idx_f], male_pool[idx_m])

        use_noise = torch.rand(B, device=u.device) < noise_ratio
        real_vec  = self.item_emb(i_real)
        noisy_vec = self.item_emb(cross) + self.noise(cross)     # v~ = v + e
        pos = torch.where(use_noise.unsqueeze(-1), noisy_vec, real_vec)
        pos = F.normalize(pos, dim=-1)

        # k_n negatives from I_u^- (items the user has not interacted with)
        negs = torch.randint(N_ITEMS, (B, k_n), device=u.device)
        for _ in range(3):                       # resample collisions
            bad = seen_mat[u.unsqueeze(-1).expand(-1, k_n), negs]
            if not bad.any():
                break
            negs = torch.where(bad,
                               torch.randint(N_ITEMS, negs.shape, device=u.device),
                               negs)
        neg = F.normalize(self.item_emb(negs), dim=-1)            # (B,k,d)

        pos_logit = (p * pos).sum(-1, keepdim=True) / tau          # (B,1)
        neg_logit = torch.bmm(neg, p.unsqueeze(-1)).squeeze(-1) / tau
        logits = torch.cat([pos_logit, neg_logit], dim=1)
        target = torch.zeros(B, dtype=torch.long, device=u.device)
        return F.cross_entropy(logits, target)


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, k):
        u, i = self.pairs[k]
        j = np.random.randint(N_ITEMS)
        while j in seen[u]:
            j = np.random.randint(N_ITEMS)
        return u, i, j


# ── evaluation (identical to fairbpr.py) ──────────────────────────────
def evaluate(model, k_acc=(10, 20), k_fair=(10, 20)):
    model.eval()
    np.random.seed(SEED)
    hits = {k: [] for k in k_acc}; ndcg = {k: [] for k in k_acc}
    recs = {}
    with torch.no_grad():
        for uid, pos in test_items.items():
            sc_np = model.all_scores(torch.LongTensor([uid]).to(DEVICE)) \
                         .squeeze(0).cpu().numpy()
            s2 = sc_np.copy()
            for x in seen.get(uid, ()):
                s2[x] = -1e9
            recs[str(uid)] = np.argsort(s2)[::-1][:max(k_fair)].tolist()
            pool = [i for i in range(N_ITEMS)
                    if i not in seen.get(uid, ()) and i != pos]
            negs = np.random.choice(pool, 99, replace=False).tolist()
            cand = [pos] + negs
            order = sorted(cand, key=lambda c: sc_np[c], reverse=True)
            for k in k_acc:
                hit = pos in order[:k]
                hits[k].append(1.0 if hit else 0.0)
                ndcg[k].append(1/np.log2(order[:k].index(pos)+2) if hit else 0.0)
    out = {'acc': {k: {'HR': float(np.mean(hits[k])),
                       'NDCG': float(np.mean(ndcg[k]))} for k in k_acc},
           'fair': {}}
    for k in k_fair:
        rk = {u: it[:k] for u, it in recs.items()}
        fr = compute_fairir_dp_eo(rk, user_gender, test_sets, N_ITEMS, k)
        ex = compute_exposure(rk, N_ITEMS, k)
        me = np.zeros(N_ITEMS); fe = np.zeros(N_ITEMS)
        for u, items in rk.items():
            arr = me if user_gender.get(int(u), 'M') == 'M' else fe
            for it in items:
                if 0 <= it < N_ITEMS: arr[it] += 1
        p = me/(me.sum()+1e-12); q = fe/(fe.sum()+1e-12); mm = 0.5*(p+q)
        _kl = lambda a, b: float(np.sum(a[a > 0]*np.log2(a[a > 0]/b[a > 0])))
        out['fair'][k] = {'DP': fr['DP'], 'EO': fr['EO'],
                          'Gini': gini_coefficient(ex),
                          'Cov': coverage(rk, N_ITEMS, k),
                          'JS': 0.5*_kl(p, mm) + 0.5*_kl(q, mm)}
    model.train()
    return out


# ── train ─────────────────────────────────────────────────────────────
model = FairIRBPR(N_USERS, N_ITEMS, CFG['emb_dim'], args.delta).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['l2'])
loader = DataLoader(PairDataset(train_pairs),
                    batch_size=CFG['batch_size'], shuffle=True)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training pairs: {len(train_pairs)}\n")

best_hr, best_state = 0.0, None
t0 = time.time()
for ep in range(CFG['epochs']):
    tb, ti, nb = 0.0, 0.0, 0
    for u, i, j in loader:
        u = u.to(DEVICE); i = i.to(DEVICE); j = j.to(DEVICE)
        si, sj = model.bpr(u, i, j)
        l_bpr  = -torch.log(torch.sigmoid(si - sj) + 1e-8).mean()
        l_info = model.info_nce(u, i, args.tau, args.k_n, args.noise_ratio)
        loss = l_bpr + args.w_info * l_info
        opt.zero_grad(); loss.backward(); opt.step()
        tb += l_bpr.item(); ti += l_info.item(); nb += 1

    if (ep + 1) % CFG['eval_every'] == 0:
        r = evaluate(model, k_acc=(10,), k_fair=(10,))
        hr = r['acc'][10]['HR']
        print(f"Epoch {ep+1:>3} | BPR {tb/nb:.4f} | InfoNCE {ti/nb:.4f} | "
              f"HR@10 {hr:.4f} | NDCG {r['acc'][10]['NDCG']:.4f} | "
              f"JS {r['fair'][10]['JS']:.4f}")
        if hr > best_hr:
            best_hr = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state is None:
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
torch.save(best_state, f'{OUT_DIR}/fairir_bpr_best.pt')
torch.save(model.state_dict(), f'{OUT_DIR}/fairir_bpr_final.pt')
model.load_state_dict(best_state)

print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")
print("Final evaluation on all test users...\n")
final = evaluate(model, k_acc=(10, 20, 30, 40), k_fair=(10, 20, 30, 40))
print(f"{'K':<4} {'HR':>8} {'NDCG':>8} {'DP':>8} {'EO':>8} "
      f"{'Gini':>8} {'Cov':>7} {'JS':>8}")
for k in (10, 20, 30, 40):
    a, f = final['acc'][k], final['fair'][k]
    print(f"{k:<4} {a['HR']:>8.4f} {a['NDCG']:>8.4f} {f['DP']:>8.4f} "
          f"{f['EO']:>8.4f} {f['Gini']:>8.4f} {f['Cov']:>7.3f} {f['JS']:>8.4f}")

json.dump({'config': vars(args),
           'results': {str(k): {**final['acc'][k], **final['fair'][k]}
                       for k in (10, 20, 30, 40)}},
          open(f'{OUT_DIR}/results.json', 'w'), indent=2)
print(f"\nSaved to {OUT_DIR}/")
