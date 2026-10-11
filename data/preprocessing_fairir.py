"""
data/preprocessing_fairir.py  (v2 — RANDOM split)
───────────────────────────────────────────────────────────────────────
Rebuilds MovieLens-1M under the EXACT FairIR protocol.

VERIFIED FROM THE PAPER (Section 4.1, quoted):
    "The dataset was RANDOMLY split into 80% training set and
     20% test set to evaluate our models."

So the split is RANDOM per user — NOT temporal. Our first attempt used
a temporal split, which is a much harder protocol and explains why our
HR looked low next to their Table 3. This version fixes that.

Protocol:
  - rating > 3 -> implicit positive          (paper, Sec 4.1)
  - per-user RANDOM 80/20 split, seed=42     (paper, Sec 4.1)
  - small val slice carved from TRAIN only, for checkpoint selection
  - full-ranking evaluation downstream

Writes to data/ml-1m-fairir/  (overwrites the temporal split — that
experiment is already recorded in results/SASRec-fairir/).

USAGE
─────
    python data/preprocessing_fairir.py
"""

import pandas as pd
import numpy as np
import os
import json
import shutil

SRC_DIR = os.path.join('data', 'ml-1m')
OUT_DIR = os.path.join('data', 'ml-1m-fairir')
os.makedirs(OUT_DIR, exist_ok=True)

SEED             = 42
MIN_INTERACTIONS = 5
TEST_RATIO       = 0.20
VAL_FROM_TRAIN   = 0.05

rng = np.random.RandomState(SEED)

# ── 1. Load raw ratings ────────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(SRC_DIR, 'ratings.dat'),
    sep='::',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)
print(f"Raw interactions: {len(df)}")

# ── 2. rating > 3 -> implicit positive ────────────────────────────────
df = df[df['rating'] > 3].copy()
print(f"After rating > 3 filter: {len(df)}  (paper reports 513,112 "
      f"after their processing)")

df = df[['user_id', 'item_id', 'timestamp']].drop_duplicates(
    subset=['user_id', 'item_id'], keep='last')
df['feedback'] = 1

# ── 3. Cold-start filter ───────────────────────────────────────────────
prev_len = -1
while prev_len != len(df):
    prev_len = len(df)
    uc = df['user_id'].value_counts()
    ic = df['item_id'].value_counts()
    df = df[df['user_id'].isin(uc[uc >= MIN_INTERACTIONS].index)]
    df = df[df['item_id'].isin(ic[ic >= MIN_INTERACTIONS].index)]
print(f"After cold-start filter (min {MIN_INTERACTIONS}): {len(df)}")

# ── 4. Re-index ────────────────────────────────────────────────────────
sorted_users = sorted(df['user_id'].unique())
sorted_items = sorted(df['item_id'].unique())
user2idx = {int(u): int(i) for i, u in enumerate(sorted_users)}
item2idx = {int(it): int(i) for i, it in enumerate(sorted_items)}

df['user_id'] = df['user_id'].map(user2idx)
df['item_id'] = df['item_id'].map(item2idx)

n_users = len(sorted_users)
n_items = len(sorted_items)
print(f"Re-indexed — Users: {n_users} | Items: {n_items}")

# ── 5. Per-user RANDOM 80/20 split (FairIR, verified) ─────────────────
train_rows, val_rows, test_rows = [], [], []

for uid, group in df.groupby('user_id'):
    idx    = group.index.to_numpy()
    rng.shuffle(idx)
    n      = len(idx)
    n_test = max(1, int(round(n * TEST_RATIO)))

    test_idx  = idx[:n_test]
    keep_idx  = idx[n_test:]

    n_val = max(1, int(round(len(keep_idx) * VAL_FROM_TRAIN))) \
            if len(keep_idx) >= 3 else 0
    val_idx   = keep_idx[:n_val]
    train_idx = keep_idx[n_val:]

    if len(train_idx) == 0:
        train_idx, val_idx = keep_idx, keep_idx[:0]

    train_rows.append(group.loc[train_idx])
    if n_val > 0:
        val_rows.append(group.loc[val_idx])
    test_rows.append(group.loc[test_idx])

train_df = pd.concat(train_rows).reset_index(drop=True)
val_df   = (pd.concat(val_rows).reset_index(drop=True)
            if val_rows else pd.DataFrame(columns=df.columns))
test_df  = pd.concat(test_rows).reset_index(drop=True)

# Keep temporal order inside each split (sequences need it)
train_df = train_df.sort_values(['user_id', 'timestamp'])
val_df   = val_df.sort_values(['user_id', 'timestamp'])
test_df  = test_df.sort_values(['user_id', 'timestamp'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Test items per user — mean: "
      f"{test_df.groupby('user_id').size().mean():.1f} | "
      f"min: {test_df.groupby('user_id').size().min()} | "
      f"max: {test_df.groupby('user_id').size().max()}")

# ── 6. Sanity ──────────────────────────────────────────────────────────
overlap = pd.merge(train_df[['user_id', 'item_id']],
                   test_df[['user_id', 'item_id']],
                   on=['user_id', 'item_id'])
assert len(overlap) == 0, "train/test overlap detected!"
print("Train/test overlap check: 0 (OK)")

# ── 7. Save ────────────────────────────────────────────────────────────
cols = ['user_id', 'item_id', 'feedback', 'timestamp']
train_df[cols].to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
val_df[cols].to_csv(  os.path.join(OUT_DIR, 'val.csv'),   index=False)
test_df[cols].to_csv( os.path.join(OUT_DIR, 'test.csv'),  index=False)

meta = {
    'n_users':  n_users,
    'n_items':  n_items,
    'user2idx': user2idx,
    'item2idx': item2idx,
    'split':    'random-80-20 seed=42 (FairIR protocol, verified Sec 4.1)',
    'filter':   f'rating > 3, min {MIN_INTERACTIONS} interactions',
}
with open(os.path.join(OUT_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

shutil.copy(os.path.join(SRC_DIR, 'users.dat'),
            os.path.join(OUT_DIR, 'users.dat'))

print(f"\nSaved to {OUT_DIR}/  (random split, seed={SEED})")
print("\nNext:  python baselines/sasrec_baseline_fairir.py")
print("\nFairIR reference (MovieLens, K=10) to beat:")
print("  BPR        : HR 0.2478  NDCG 0.2492  DP 0.6541  EO 0.7158")
print("  FairIR_BPR : HR 0.2833  NDCG 0.2777  DP 0.5741  EO 0.6258")
print("  FairIR_GCCF: HR 0.2864  NDCG 0.2878  DP 0.5986  EO 0.6501")
