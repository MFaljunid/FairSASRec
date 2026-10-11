"""
baselines/fairbpr.py
───────────────────────────────────────────────────────────────────────
FairBPR = matrix factorisation + the FairnessConstraintLayer.

WHY THIS FILE EXISTS
────────────────────
Reviewer 2 observed that the paper claims the layer is backbone-agnostic
while every reported result uses SASRec. This file tests the claim on a
structurally different scorer: plain matrix factorisation trained with
the BPR objective, which has no sequence modelling, no attention, and
no notion of order.

The layer is transplanted UNCHANGED. It reads a user representation and
a score vector, which is all any scorer provides, and applies

    s_hat_ui = s_ui - rho(p_u) * softplus(alpha) * c_tilde_i

with alpha initialised dormant exactly as in fairsasrec.py.

CONTROLLED COMPARISON
─────────────────────
Running this file with --fairness_lambda 0 and --no-layer gives the
BPR backbone; running it with the layer gives FairBPR. Same code, same
optimiser, same seed, same data, same number of epochs. The only
difference is the layer, mirroring the SASRec/FairSASRec comparison.

USAGE
    # backbone
    python baselines/fairbpr.py --no-layer
    # with the layer
    python baselines/fairbpr.py

    DATA_DIR=data/ml-100k OUT_DIR=results/FairBPR-100k \
        python baselines/fairbpr.py

Outputs
    <OUT_DIR>/fairbpr_best.pt, fairbpr_final.pt
    <OUT_DIR>/item_embeddings.npy
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
parser.add_argument('--no-layer', action='store_true',
                    help='train the plain BPR backbone (no fairness layer)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--alpha_max', type=float, default=None,
                    help='optional upper bound on softplus(alpha)')
args = parser.parse_args()

USE_LAYER = not args.no_layer
ALPHA_MAX = args.alpha_max

# ── reproducibility ───────────────────────────────────────────────────
SEED = int(os.environ.get('SEED', 42))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = os.environ.get('DATA_DIR', 'data/ml-1m')
_default_out = 'results/FairBPR' if USE_LAYER else 'results/BPR-torch'
OUT_DIR = os.environ.get('OUT_DIR', _default_out)
if not USE_LAYER and os.environ.get('OUT_DIR'):
    OUT_DIR = f'{OUT_DIR}-backbone'
if ALPHA_MAX is not None:
    OUT_DIR = f'{OUT_DIR}-amax{ALPHA_MAX}'
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {
    'emb_dim':    64,
    'lr':         1e-3,
    'l2':         1e-5,
    'epochs':     args.epochs,
    'batch_size': 4096,
    'eval_every': 10,
}

print("=" * 62)
print(f"{'FairBPR (layer active)' if USE_LAYER else 'BPR backbone (no layer)'}")
print("=" * 62)
print(f"Device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else "")
      + f" | seed: {SEED}")
print(f"Data: {DATA_DIR} | Out: {OUT_DIR}")
if ALPHA_MAX is not None:
    print(f"Constraint strength capped at softplus(alpha) <= {ALPHA_MAX}")

# ── data ──────────────────────────────────────────────────────────────
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

test_items = dict(zip(test['user_id'].astype(int),
                      test['item_id'].astype(int)))
test_sets = {u: {i} for u, i in test_items.items()}

# gender, for evaluation only
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
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}
n_m = sum(1 for g in user_gender.values() if g == 'M')
print(f"Gender loaded — Male: {n_m} | Female: {len(user_gender) - n_m}")


# ══════════════════════════════════════════════════════════════════════
#  The layer, transplanted verbatim from fairsasrec.py
# ══════════════════════════════════════════════════════════════════════
class FairnessConstraintLayer(nn.Module):
    """s_hat = s - rho(h) * softplus(alpha) * c_tilde

    Identical in form and initialisation to the module used with the
    sequential backbone. It reads only a user representation and a score
    vector, so nothing about it is specific to self-attention."""

    def __init__(self, state_dim, init_alpha=-5.0):
        super().__init__()
        self.alpha     = nn.Parameter(torch.tensor([init_alpha]))
        self.user_proj = nn.Linear(state_dim, 1)
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.zeros_(self.user_proj.bias)

    def strength(self):
        a = F.softplus(self.alpha)
        if ALPHA_MAX is not None:
            a = torch.clamp(a, max=ALPHA_MAX)
        return a

    def forward(self, scores, state, exposure):
        exp_norm = torch.log1p(exposure)
        exp_norm = exp_norm / (exp_norm.max() + 1e-9)
        rho = torch.sigmoid(self.user_proj(state))
        return scores - rho * (self.strength() * exp_norm).unsqueeze(0)


class FairBPR(nn.Module):
    def __init__(self, n_users, n_items, d, use_layer):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.use_layer = use_layer
        self.fair = FairnessConstraintLayer(d) if use_layer else None

    def state(self, u):
        return self.user_emb(u)                       # (B, d)

    def all_scores(self, u, exposure=None):
        p = self.state(u)
        s = p @ self.item_emb.weight.T                # (B, n_items)
        if self.use_layer and exposure is not None:
            s = self.fair(s, p, exposure)
        return s

    def pair_scores(self, u, i, j, exposure=None):
        """Scores for one positive and one negative per row."""
        p  = self.state(u)
        qi = self.item_emb(i)
        qj = self.item_emb(j)
        si = (p * qi).sum(-1)
        sj = (p * qj).sum(-1)
        if self.use_layer and exposure is not None:
            a   = self.fair.strength()
            en  = torch.log1p(exposure)
            en  = en / (en.max() + 1e-9)
            rho = torch.sigmoid(self.fair.user_proj(p)).squeeze(-1)
            si = si - rho * a.squeeze() * en[i]
            sj = sj - rho * a.squeeze() * en[j]
        return si, sj


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


# ── exposure, accumulated from the model's own lists ─────────────────
exposure = torch.zeros(N_ITEMS, device=DEVICE)

def refresh_exposure(model, sample=800):
    model.eval()
    counts = torch.zeros(N_ITEMS, device=DEVICE)
    uids = sorted(test_items.keys())[:sample]
    with torch.no_grad():
        for start in range(0, len(uids), 256):
            chunk = uids[start:start + 256]
            u = torch.LongTensor(chunk).to(DEVICE)
            s = model.all_scores(u, exposure).clone()
            for row, uid in enumerate(chunk):
                idx = torch.LongTensor(list(seen.get(uid, ()))).to(DEVICE)
                if len(idx): s[row, idx] = -1e9
            for it in s.topk(10, dim=-1).indices.flatten():
                counts[it] += 1
    model.train()
    return counts


# ── evaluation, matching evaluate_final.py ───────────────────────────
def evaluate(model, k_acc=(10, 20), k_fair=(10, 20)):
    model.eval()
    np.random.seed(SEED)
    hits = {k: [] for k in k_acc}
    ndcg = {k: [] for k in k_acc}
    recs = {}
    zero = torch.zeros(N_ITEMS, device=DEVICE)   # deployed setting
    with torch.no_grad():
        for uid, pos in test_items.items():
            u = torch.LongTensor([uid]).to(DEVICE)
            sc = model.all_scores(u, zero).squeeze(0)
            sc_np = sc.cpu().numpy()

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
model = FairBPR(N_USERS, N_ITEMS, CFG['emb_dim'], USE_LAYER).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'],
                       weight_decay=CFG['l2'])
loader = DataLoader(PairDataset(train_pairs),
                    batch_size=CFG['batch_size'], shuffle=True)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
if USE_LAYER:
    print(f"alpha0 = {model.fair.strength().item():.4f}")
print(f"Training pairs: {len(train_pairs)}\n")

best_hr, best_state = 0.0, None
t0 = time.time()

for ep in range(CFG['epochs']):
    total, nb = 0.0, 0
    for u, i, j in loader:
        u = u.to(DEVICE); i = i.to(DEVICE); j = j.to(DEVICE)
        si, sj = model.pair_scores(u, i, j, exposure if USE_LAYER else None)
        loss = -torch.log(torch.sigmoid(si - sj) + 1e-8).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); nb += 1

    if USE_LAYER:
        exposure = refresh_exposure(model)

    if (ep + 1) % CFG['eval_every'] == 0:
        r = evaluate(model, k_acc=(10,), k_fair=(10,))
        hr = r['acc'][10]['HR']
        line = (f"Epoch {ep+1:>3} | Loss {total/nb:.4f} | "
                f"HR@10 {hr:.4f} | NDCG {r['acc'][10]['NDCG']:.4f} | "
                f"JS {r['fair'][10]['JS']:.4f}")
        if USE_LAYER:
            line += f" | alpha {model.fair.strength().item():.4f}"
        print(line)
        if hr > best_hr:
            best_hr = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state is None:
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
torch.save(best_state, f'{OUT_DIR}/fairbpr_best.pt')
torch.save(model.state_dict(), f'{OUT_DIR}/fairbpr_final.pt')
model.load_state_dict(best_state)
np.save(f'{OUT_DIR}/item_embeddings.npy',
        model.item_emb.weight.data.cpu().numpy())

# ── final evaluation ──────────────────────────────────────────────────
print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")
print("Final evaluation on all test users...\n")
final = evaluate(model, k_acc=(10, 20, 30, 40), k_fair=(10, 20, 30, 40))

print(f"{'K':<4} {'HR':>8} {'NDCG':>8} {'DP':>8} {'EO':>8} "
      f"{'Gini':>8} {'Cov':>7} {'JS':>8}")
for k in (10, 20, 30, 40):
    a, f = final['acc'][k], final['fair'][k]
    print(f"{k:<4} {a['HR']:>8.4f} {a['NDCG']:>8.4f} {f['DP']:>8.4f} "
          f"{f['EO']:>8.4f} {f['Gini']:>8.4f} {f['Cov']:>7.3f} "
          f"{f['JS']:>8.4f}")

if USE_LAYER:
    print(f"\nLearned constraint strength: "
          f"softplus(alpha) = {model.fair.strength().item():.4f} "
          f"(initialised at 0.0067)")

json.dump({str(k): {**final['acc'][k], **final['fair'][k]}
           for k in (10, 20, 30, 40)},
          open(f'{OUT_DIR}/results.json', 'w'), indent=2)
print(f"\nSaved to {OUT_DIR}/")
