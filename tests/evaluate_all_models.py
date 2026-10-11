import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR    = 'data/ml-1m'
RESULTS_DIR = 'results/ml-1m'
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)

print("=" * 60)
print("Evaluating ALL models at K = 5, 10, 20, 30, 40")
print("=" * 60)

K_LIST  = [5, 10, 20, 30, 40]
N_NEG   = 99
EMB_DIM = 64
HIDDEN  = 256
WINDOW  = 10

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

# ── Gender for DP/EO ───────────────────────────────────────────────────
user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender(f'{DATA_DIR}/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

np.random.seed(42)

# ── Helper functions ───────────────────────────────────────────────────
def sample_candidates(uid, pos_item, cornac_item_map=None):
    seen = train_set[uid] | val_set[uid] | {pos_item}
    if cornac_item_map:
        pool = [i for i in range(N_ITEMS)
                if i not in seen and i in cornac_item_map]
    else:
        pool = list(set(range(N_ITEMS)) - seen)
    if len(pool) < N_NEG:
        return None
    negs = np.random.choice(pool, size=N_NEG, replace=False).tolist()
    return [pos_item] + negs

def hit_at_k(ranked, pos, k):
    return 1.0 if pos in ranked[:k] else 0.0

def ndcg_at_k(ranked, pos, k):
    if pos in ranked[:k]:
        return 1.0 / np.log2(ranked[:k].index(pos) + 2)
    return 0.0

def compute_all_metrics(recs_dict, k):
    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_dp_eo(recs_k, user_gender, test_items, k)
    return {
        'Gini':     round(gini, 4),
        'Coverage': round(cov,  4),
        'DP':       round(fairness['DP'], 4),
        'EO':       round(fairness['EO'], 4),
    }

ALL_RESULTS = {}

# ══════════════════════════════════════════════════════════════════════
# MODEL 1: BPR
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating BPR ---")
pkl_files = glob.glob('results/BPR/*.pkl')
assert len(pkl_files) > 0, "Run bpr_baseline.py first"

with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

bpr_recs    = {}
bpr_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

for uid, pos_item in test_items.items():
    if uid not in bpr_u2c or pos_item not in bpr_i2c:
        continue
    candidates = sample_candidates(uid, pos_item, bpr_i2c)
    if candidates is None:
        continue
    try:
        scores = bpr.score(bpr_u2c[uid])
    except Exception:
        continue
    ranked = sorted(candidates,
                    key=lambda x: float(scores[bpr_i2c[x]])
                    if x in bpr_i2c else 0.0,
                    reverse=True)
    bpr_recs[str(uid)] = ranked
    for k in K_LIST:
        bpr_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
        bpr_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

bpr_final = {}
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)
for k in K_LIST:
    hr   = np.mean(bpr_results[k]['hits'])
    ndcg = np.mean(bpr_results[k]['ndcgs'])
    fair = compute_all_metrics(bpr_recs, k)
    bpr_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
    r = bpr_final[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
ALL_RESULTS['BPR'] = bpr_final

# ══════════════════════════════════════════════════════════════════════
# MODEL 2: LightGCN
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating LightGCN ---")
lgcn_pkl = glob.glob('results/LightGCN/*.pkl')

if lgcn_pkl:
    with open(sorted(lgcn_pkl)[-1], 'rb') as f:
        lgcn = pickle.load(f)

    lgcn_u2c = {int(k): v for k, v in lgcn.uid_map.items()}
    lgcn_i2c = {int(k): v for k, v in lgcn.iid_map.items()}

    lgcn_recs    = {}
    lgcn_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

    for uid, pos_item in test_items.items():
        if uid not in lgcn_u2c or pos_item not in lgcn_i2c:
            continue
        candidates = sample_candidates(uid, pos_item, lgcn_i2c)
        if candidates is None:
            continue
        try:
            scores = lgcn.score(lgcn_u2c[uid])
        except Exception:
            continue
        ranked = sorted(candidates,
                        key=lambda x: float(scores[lgcn_i2c[x]])
                        if x in lgcn_i2c else 0.0,
                        reverse=True)
        lgcn_recs[str(uid)] = ranked
        for k in K_LIST:
            lgcn_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            lgcn_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

    lgcn_final = {}
    print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
          f"{'Gini':>7} {'Cov':>7}")
    print("-" * 55)
    for k in K_LIST:
        hr   = np.mean(lgcn_results[k]['hits'])
        ndcg = np.mean(lgcn_results[k]['ndcgs'])
        fair = compute_all_metrics(lgcn_recs, k)
        lgcn_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
        r = lgcn_final[k]
        print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
              f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
              f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
    ALL_RESULTS['LightGCN'] = lgcn_final
else:
    print("LightGCN not found — skipping")

# ══════════════════════════════════════════════════════════════════════
# MODEL 3: Your RL model
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating Your RL Model ---")

env = RecEnv(f'{DATA_DIR}/train.csv', f'{DATA_DIR}/meta.json',
             emb_dim=EMB_DIM, window=WINDOW)

emb_path = f'{DATA_DIR}/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    bpr_emb = np.load(emb_path)
    if bpr_emb.shape[0] < N_ITEMS:
        pad     = np.zeros((N_ITEMS - bpr_emb.shape[0],
                            bpr_emb.shape[1]), dtype=np.float32)
        bpr_emb = np.vstack([bpr_emb, pad])
    env.load_pretrained_embeddings(bpr_emb)

policy = ActorCriticPolicy(emb_dim=EMB_DIM, n_items=N_ITEMS,
                            hidden_dim=HIDDEN)
policy.load_state_dict(
    torch.load('results/ml-1m/policy_final.pt', map_location='cpu'))
policy.eval()

def get_item_seq(uid):
    history = env._gt_history[uid]
    recent  = history[-WINDOW:]
    if len(recent) < WINDOW:
        pad    = [0] * (WINDOW - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

rl_recs    = {}
rl_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        candidates = sample_candidates(uid, pos_item)
        if candidates is None:
            continue
        seq_t           = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
        exp_t           = torch.zeros(N_ITEMS)
        logits, _, _, _ = policy.forward(seq_t, exp_t)
        scores          = logits.squeeze(0).numpy()
        ranked          = sorted(candidates,
                                 key=lambda x: scores[x], reverse=True)
        rl_recs[str(uid)] = ranked
        for k in K_LIST:
            rl_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            rl_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

rl_final = {}
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)
for k in K_LIST:
    hr   = np.mean(rl_results[k]['hits'])
    ndcg = np.mean(rl_results[k]['ndcgs'])
    fair = compute_all_metrics(rl_recs, k)
    rl_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
    r = rl_final[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
ALL_RESULTS['Our RL'] = rl_final

# ── Save all results ───────────────────────────────────────────────────
with open(f'{RESULTS_DIR}/all_models_tradeoff.json', 'w') as f:
    json.dump({m: {str(k): v for k, v in res.items()}
               for m, res in ALL_RESULTS.items()},
              f, indent=2)
print(f"\nSaved: {RESULTS_DIR}/all_models_tradeoff.json")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE at K=10
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL COMPARISON TABLE (K=10)")
print("=" * 65)
print(f"{'Model':<12} {'HR@10':>7} {'NDCG@10':>9} "
      f"{'DP':>7} {'EO':>7} {'Gini':>7} {'Cov':>7}")
print("-" * 65)
for model, results in ALL_RESULTS.items():
    r = results[10]
    print(f"{model:<12} {r['HR']:>7.4f} {r['NDCG']:>9.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
print("=" * 65)

# ── Trade-off figure ───────────────────────────────────────────────────
COLORS  = {'BPR': '#4C72B0', 'LightGCN': '#55A868', 'Our RL': '#E8A838'}
MARKERS = {'BPR': 'o',       'LightGCN': 's',        'Our RL': '*'}
SIZES   = {'BPR': 80,        'LightGCN': 80,          'Our RL': 200}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Accuracy-Fairness Trade-off Curves — MovieLens-1M\n'
             'Points labeled by K = 5, 10, 20, 30, 40',
             fontsize=13, fontweight='bold')

configs = [
    ('HR',   'DP',   'HR (higher=better)',   'DP (lower=better)',   'DP vs HR'),
    ('NDCG', 'EO',   'NDCG (higher=better)', 'EO (lower=better)',   'EO vs NDCG'),
]

for ax, (xm, ym, xlabel, ylabel, title) in zip(axes, configs):
    for model, results in ALL_RESULTS.items():
        xs = [results[k][xm] for k in K_LIST]
        ys = [results[k][ym] for k in K_LIST]
        ax.plot(xs, ys, color=COLORS[model],
                linewidth=2, alpha=0.8, zorder=3)
        ax.scatter(xs, ys, color=COLORS[model],
                   s=SIZES[model], marker=MARKERS[model],
                   zorder=5, edgecolors='black', linewidth=0.5)
        for k, x, y in zip(K_LIST, xs, ys):
            ax.annotate(f'K={k}', xy=(x, y),
                        xytext=(6, 4),
                        textcoords='offset points',
                        fontsize=8, color=COLORS[model],
                        fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [
    mlines.Line2D([], [], color=COLORS[m],
                  marker=MARKERS[m],
                  markersize=10 if m == 'Our RL' else 7,
                  linewidth=2, label=m)
    for m in ALL_RESULTS.keys()
]
fig.legend(handles=handles, loc='lower center',
           ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/figures/tradeoff_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {RESULTS_DIR}/figures/tradeoff_curves.png")
print("\nDone.")