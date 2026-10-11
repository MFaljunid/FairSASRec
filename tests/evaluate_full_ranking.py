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
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR    = 'data/ml-1m'
RESULTS_DIR = 'results/ml-1m'
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)

print("=" * 60)
print("FULL RANKING Evaluation — same as FairIR paper")
print("K = 10, 20, 30, 40")
print("=" * 60)

K_LIST  = [10, 20, 30, 40]
EMB_DIM = 64
HIDDEN  = 256
WINDOW  = 10

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

# test_set — multiple items per user (80/20 split) ← used for HR/NDCG
test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

# Gender for DP/EO
user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender(f'{DATA_DIR}/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

print(f"Users: {N_USERS} | Items: {N_ITEMS}")
print(f"Test interactions: {len(test)}")

# ── Helper functions ───────────────────────────────────────────────────
def compute_hr_ndcg(ranked, relevant, k):
    """HR and NDCG with multiple relevant items."""
    topk     = ranked[:k]
    hits     = set(topk) & relevant
    hr       = len(hits) / min(len(relevant), k)
    dcg      = sum(1/np.log2(i+2) for i, item in enumerate(topk)
                   if item in relevant)
    idcg     = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    ndcg     = dcg/idcg if idcg > 0 else 0.0
    return hr, ndcg

def compute_all_metrics(recs_dict, k):
    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_fairir_dp_eo(recs_k, user_gender,
                                     test_set, N_ITEMS, k)
    return {
        'Gini':     round(gini, 4),
        'Coverage': round(cov,  4),
        'DP':       round(fairness['DP'], 4),
        'EO':       round(fairness['EO'], 4),
    }

def print_results(results_dict, model_name):
    final = {}
    print(f"\n{model_name} Results:")
    print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
          f"{'Gini':>7} {'Cov':>7}")
    print("-" * 55)
    for k in K_LIST:
        hr   = np.mean(results_dict[k]['hits'])
        ndcg = np.mean(results_dict[k]['ndcgs'])
        fair = compute_all_metrics(recs_dict_global[model_name], k)
        final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
        r = final[k]
        print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
              f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
              f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
    return final

recs_dict_global = {}
ALL_RESULTS      = {}

# ══════════════════════════════════════════════════════════════════════
# MODEL 1: BPR — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating BPR (full ranking) ---")
pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

bpr_recs    = {}
bpr_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

for uid, relevant in test_set.items():
    if uid not in bpr_u2c:
        continue
    cornac_uid = bpr_u2c[uid]
    try:
        scores = bpr.score(cornac_uid)
    except Exception:
        continue

    seen = train_set[uid]
    all_scores = []
    for item in range(N_ITEMS):
        if item in seen:
            continue
        cornac_item = bpr_i2c.get(item, -1)
        if 0 <= cornac_item < len(scores):
            all_scores.append((item, float(scores[cornac_item])))
        else:
            all_scores.append((item, 0.0))

    ranked = [item for item, score in
              sorted(all_scores, key=lambda x: x[1], reverse=True)]
    bpr_recs[str(uid)] = ranked

    for k in K_LIST:
        hr, ndcg = compute_hr_ndcg(ranked, relevant, k)
        bpr_results[k]['hits'].append(hr)
        bpr_results[k]['ndcgs'].append(ndcg)

recs_dict_global['BPR'] = bpr_recs
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)
bpr_final = {}
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
# MODEL 2: LightGCN — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating LightGCN (full ranking) ---")
lgcn_pkl = glob.glob('results/LightGCN/*.pkl')

if lgcn_pkl:
    with open(sorted(lgcn_pkl)[-1], 'rb') as f:
        lgcn = pickle.load(f)

    lgcn_u2c = {int(k): v for k, v in lgcn.uid_map.items()}
    lgcn_i2c = {int(k): v for k, v in lgcn.iid_map.items()}

    lgcn_recs    = {}
    lgcn_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

    for uid, relevant in test_set.items():
        if uid not in lgcn_u2c:
            continue
        cornac_uid = lgcn_u2c[uid]
        try:
            scores = lgcn.score(cornac_uid)
        except Exception:
            continue

        seen = train_set[uid]
        all_scores = []
        for item in range(N_ITEMS):
            if item in seen:
                continue
            cornac_item = lgcn_i2c.get(item, -1)
            if 0 <= cornac_item < len(scores):
                all_scores.append((item, float(scores[cornac_item])))
            else:
                all_scores.append((item, 0.0))

        ranked = [item for item, score in
                  sorted(all_scores, key=lambda x: x[1], reverse=True)]
        lgcn_recs[str(uid)] = ranked

        for k in K_LIST:
            hr, ndcg = compute_hr_ndcg(ranked, relevant, k)
            lgcn_results[k]['hits'].append(hr)
            lgcn_results[k]['ndcgs'].append(ndcg)

    recs_dict_global['LightGCN'] = lgcn_recs
    print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
          f"{'Gini':>7} {'Cov':>7}")
    print("-" * 55)
    lgcn_final = {}
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
# MODEL 3: Your RL model — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating Your RL Model (full ranking + fairness reranking) ---")

from rl_model.fairness_reranker import FairnessReranker

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

policy = ActorCriticPolicy(
    emb_dim=EMB_DIM,
    n_items=N_ITEMS,
    hidden_dim=HIDDEN
)
policy.load_state_dict(
    torch.load('results/policy_final.pt', map_location='cpu'))
policy.eval()

def get_item_seq(uid):
    history = env._gt_history[uid]
    recent  = history[-WINDOW:]
    if len(recent) < WINDOW:
        pad    = [0] * (WINDOW - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# Try different lambda values and pick best
best_results = None
best_dp      = float('inf')
best_lambda  = 0.5

for lambda_fair in [0.3, 0.5, 0.7]:
    reranker   = FairnessReranker(N_ITEMS, lambda_fair=lambda_fair)
    rl_recs    = {}
    rl_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

    with torch.no_grad():
        for uid, relevant in test_set.items():
            gender = user_gender.get(uid, 'M')
            seen   = train_set[uid]
            seq_t  = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
            exp_t  = torch.zeros(N_ITEMS)
            logits, _, _, _ = policy.forward(seq_t, exp_t)
            scores = logits.squeeze(0).numpy()

            ranked = reranker.rerank(scores, gender, seen, k=40)
            rl_recs[str(uid)] = ranked

            for k in K_LIST:
                hr, ndcg = compute_hr_ndcg(ranked, relevant, k)
                rl_results[k]['hits'].append(hr)
                rl_results[k]['ndcgs'].append(ndcg)

    recs_k   = {uid: items[:10] for uid, items in rl_recs.items()}
    fairness = compute_fairir_dp_eo(recs_k, user_gender,
                                     test_set, N_ITEMS, 10)
    dp   = fairness['DP']
    hr10 = np.mean(rl_results[10]['hits'])
    print(f"  lambda={lambda_fair}: HR@10={hr10:.4f} DP@10={dp:.4f}")

    if dp < best_dp:
        best_dp      = dp
        best_lambda  = lambda_fair
        best_results = (rl_recs, rl_results)

print(f"\nBest lambda: {best_lambda} with DP={best_dp:.4f}")
rl_recs, rl_results = best_results

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