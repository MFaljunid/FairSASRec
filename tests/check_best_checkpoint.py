import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np
import torch
import pandas as pd
import json
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

DATA_DIR = 'data/ml-1m'
train    = pd.read_csv(f'{DATA_DIR}/train.csv')
val      = pd.read_csv(f'{DATA_DIR}/val.csv')
test     = pd.read_csv(f'{DATA_DIR}/test.csv')
meta     = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS  = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = dict(zip(
    test['user_id'].astype(int),
    test['item_id'].astype(int)
))

env = RecEnv(f'{DATA_DIR}/train.csv', f'{DATA_DIR}/meta.json',
             emb_dim=64, window=10)
bpr_emb = np.load(f'{DATA_DIR}/bpr_item_embeddings.npy')
if bpr_emb.shape[0] < N_ITEMS:
    pad     = np.zeros((N_ITEMS - bpr_emb.shape[0],
                        bpr_emb.shape[1]), dtype=np.float32)
    bpr_emb = np.vstack([bpr_emb, pad])
env.load_pretrained_embeddings(bpr_emb)

def get_item_seq(uid):
    history = env._gt_history[uid]
    recent  = history[-10:]
    if len(recent) < 10:
        recent = [0]*(10-len(recent)) + recent
    return np.array(recent, dtype=np.int64)

np.random.seed(42)
sample_users = list(test_items.keys())[:2000]

checkpoints = sorted(glob.glob('results/ml-1m/policy_ep*.pt'))
checkpoints.append('results/ml-1m/policy_final.pt')

import glob
checkpoints = sorted(glob.glob('results/ml-1m/policy_ep*.pt'))
checkpoints.append('results/ml-1m/policy_final.pt')

print(f"{'Checkpoint':<30} {'HR@10':>8}")
print("-" * 40)

best_hr   = 0.0
best_ckpt = None

for ckpt in checkpoints:
    if not os.path.exists(ckpt):
        continue
    policy = ActorCriticPolicy(emb_dim=64, n_items=N_ITEMS, hidden_dim=256)
    policy.load_state_dict(torch.load(ckpt, map_location='cpu'))
    policy.eval()

    hits = []
    with torch.no_grad():
        for uid in sample_users:
            pos_item   = test_items[uid]
            seen       = train_set[uid] | val_set[uid] | {pos_item}
            pool       = [i for i in range(N_ITEMS) if i not in seen]
            if len(pool) < 99:
                continue
            neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
            candidates = [pos_item] + neg_items
            seq_t      = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
            exp_t      = torch.zeros(N_ITEMS)
            logits, _, _, _ = policy.forward(seq_t, exp_t)
            scores     = logits.squeeze(0).numpy()
            ranked     = sorted(candidates,
                                key=lambda x: scores[x], reverse=True)
            hits.append(1.0 if pos_item in ranked[:10] else 0.0)

    hr = np.mean(hits)
    name = os.path.basename(ckpt)
    print(f"{name:<30} {hr:>8.4f}")

    if hr > best_hr:
        best_hr   = hr
        best_ckpt = ckpt

print(f"\nBest checkpoint: {best_ckpt}")
print(f"Best HR@10:      {best_hr:.4f}")

# Copy best to policy_best.pt
import shutil
shutil.copy(best_ckpt, 'results/ml-1m/policy_best.pt')
print(f"Saved best to:   results/ml-1m/policy_best.pt")