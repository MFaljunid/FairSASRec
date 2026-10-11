import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

import pandas as pd
import numpy as np
import cornac
from cornac.eval_methods import BaseMethod
from cornac.models import BPR
from cornac.metrics import NDCG, Recall, Precision
import json, pickle
from collections import defaultdict
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage

os.makedirs('results', exist_ok=True)

DATA_DIR = os.environ.get('DATA_DIR', 'data/ml-1m')
BPR_OUT = os.environ.get('OUT_DIR', 'results/BPR')
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta     = json.load(open(f'{DATA_DIR}/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Users: {N_USERS} | Items: {N_ITEMS}")

# ── Build Cornac format ────────────────────────────────────────────────
train_data = list(zip(
    train_df['user_id'].astype(str),
    train_df['item_id'].astype(str),
    train_df['feedback'].astype(float)
))
test_data = list(zip(
    test_df['user_id'].astype(str),
    test_df['item_id'].astype(str),
    test_df['feedback'].astype(float)
))

eval_method = BaseMethod.from_splits(
    train_data=train_data,
    test_data=test_data,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    seed=42
)

# ── BPR model ──────────────────────────────────────────────────────────
bpr = BPR(
    k=64,
    max_iter=200,
    learning_rate=0.01,
    lambda_reg=0.001,
    seed=42,
    verbose=True
)

exp = cornac.Experiment(
    eval_method=eval_method,
    models=[bpr],
    metrics=[
        NDCG(k=10), NDCG(k=20), NDCG(k=30), NDCG(k=40),
        Recall(k=10), Recall(k=20),
        Precision(k=10)
    ],
    user_based=True,
    save_dir='results/'
)
exp.run()

# ── Save model and embeddings ──────────────────────────────────────────
os.makedirs(BPR_OUT, exist_ok=True)
from datetime import datetime
timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
model_path = f'{BPR_OUT}/{timestamp}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(bpr, f)
print(f"BPR model saved to {model_path}")

item_embeddings = bpr.i_factors
user_embeddings = bpr.u_factors
np.save(f'{DATA_DIR}/bpr_item_embeddings.npy', item_embeddings)
np.save(f'{DATA_DIR}/bpr_user_embeddings.npy', user_embeddings)
print(f"Item embeddings: {item_embeddings.shape}")

# ── Full ranking evaluation at K = 5, 10, 20, 30, 40 ──────────────────
print("\n" + "=" * 60)
print("BPR Full Evaluation at K = 5, 10, 20, 30, 40")
print("=" * 60)

K_LIST = [5, 10, 20, 30, 40]

# Build lookup sets
train_set = defaultdict(set)
for _, row in train_df.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

# Build test items — multiple per user (80/20 split)
test_set = defaultdict(set)
for _, row in test_df.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

val_df  = pd.read_csv(f'{DATA_DIR}/val.csv')
val_set = defaultdict(set)
for _, row in val_df.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

# Gender for DP/EO
user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
if 'ml-100k' in DATA_DIR:
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
print(f"Gender loaded — Male: {sum(1 for g in user_gender.values() if g=='M')} "
      f"| Female: {sum(1 for g in user_gender.values() if g=='F')}")
    
bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

# Get full ranking for each user
print("Generating recommendations...")
recs_dict = {}

for uid in test_set.keys():
    if uid not in bpr_u2c:
        continue
    cornac_uid = bpr_u2c[uid]
    try:
        scores = bpr.score(cornac_uid)
    except Exception:
        continue

    seen = train_set[uid] | val_set[uid]
    all_scores = []
    for item in range(N_ITEMS):
        if item in seen:
            continue
        cornac_item = bpr_i2c.get(item, -1)
        if 0 <= cornac_item < len(scores):
            all_scores.append((item, float(scores[cornac_item])))

    ranked = [item for item, score in
              sorted(all_scores, key=lambda x: x[1], reverse=True)]
    recs_dict[str(uid)] = ranked
# ── Compute metrics ────────────────────────────────────────────────────
def hit_at_k(ranked, relevant, k):
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(ranked, relevant, k):
    dcg  = sum(1/np.log2(i+2) for i, item in enumerate(ranked[:k])
               if item in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg > 0 else 0.0

print(f"\n{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("=" * 55)

bpr_results = {}
for k in K_LIST:
    hits, ndcgs = [], []
    for uid, relevant in test_set.items():
        uid_str = str(uid)
        if uid_str not in recs_dict:
            continue
        ranked = recs_dict[uid_str]
        hits.append(hit_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}  # ← inside loop
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_fairir_dp_eo(recs_k, user_gender, test_set, N_ITEMS, k)  # ← fixed

    bpr_results[k] = {
        'HR':       round(np.mean(hits),  4),
        'NDCG':     round(np.mean(ndcgs), 4),
        'DP':       round(fairness['DP'], 4),
        'EO':       round(fairness['EO'], 4),
        'Gini':     round(gini, 4),
        'Coverage': round(cov,  4),
    }
    r = bpr_results[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")

with open('results/bpr_full_results.json', 'w') as f:
    json.dump({str(k): v for k, v in bpr_results.items()}, f, indent=2)

print("\nSaved: results/bpr_full_results.json")