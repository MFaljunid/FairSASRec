import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

DATA_DIR = 'data/ml-100k'
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load raw ratings ────────────────────────────────────────────────
df = pd.read_csv(
    f'{DATA_DIR}/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
)
print(f"Raw interactions: {len(df)}")

# ── 2. Keep ratings > 3 ───────────────────────────────────────────────
df = df[df['rating'] > 3].copy()
print(f"After rating > 3 filter: {len(df)}")

# ── 3. Dedup ──────────────────────────────────────────────────────────
df = df[['user_id', 'item_id', 'timestamp']].drop_duplicates(
    subset=['user_id', 'item_id'], keep='last')
df['feedback'] = 1

# ── 4. Cold-start filter ──────────────────────────────────────────────
MIN_INTERACTIONS = 5
prev_len = -1
while prev_len != len(df):
    prev_len = len(df)
    uc = df['user_id'].value_counts()
    ic = df['item_id'].value_counts()
    df = df[df['user_id'].isin(uc[uc >= MIN_INTERACTIONS].index)]
    df = df[df['item_id'].isin(ic[ic >= MIN_INTERACTIONS].index)]
print(f"After cold-start filter: {len(df)}")

# ── 5. Sort ───────────────────────────────────────────────────────────
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ── 6. Re-index ───────────────────────────────────────────────────────
sorted_users = sorted(df['user_id'].unique())
sorted_items = sorted(df['item_id'].unique())
user2idx = {int(u): int(i) for i, u in enumerate(sorted_users)}
item2idx = {int(it): int(i) for i, it in enumerate(sorted_items)}

df['user_id'] = df['user_id'].map(user2idx)
df['item_id'] = df['item_id'].map(item2idx)

n_users = len(sorted_users)
n_items = len(sorted_items)
print(f"Users: {n_users} | Items: {n_items}")

# ── 7. Leave-one-out split ────────────────────────────────────────────
train_rows, val_rows, test_rows = [], [], []

for uid, group in df.groupby('user_id'):
    items = group.sort_values('timestamp')
    if len(items) < 3:
        train_rows.append(items)
        continue
    train_rows.append(items.iloc[:-2])
    val_rows.append(items.iloc[-2:-1])
    test_rows.append(items.iloc[-1:])

train_df = pd.concat(train_rows).reset_index(drop=True)
val_df   = pd.concat(val_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 8. Save ───────────────────────────────────────────────────────────
cols = ['user_id', 'item_id', 'feedback', 'timestamp']
train_df[cols].to_csv(f'{DATA_DIR}/train.csv', index=False)
val_df[cols].to_csv(  f'{DATA_DIR}/val.csv',   index=False)
test_df[cols].to_csv( f'{DATA_DIR}/test.csv',  index=False)

meta = {
    'n_users':  n_users,
    'n_items':  n_items,
    'user2idx': user2idx,
    'item2idx': item2idx,
    'split':    'leave-one-out',
    'filter':   'rating > 3, min 5 interactions',
}
with open(f'{DATA_DIR}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved to {DATA_DIR}/")
print(f"  train.csv ({len(train_df)} rows)")
print(f"  val.csv   ({len(val_df)} rows)")
print(f"  test.csv  ({len(test_df)} rows)")
print(f"  meta.json (n_users={n_users}, n_items={n_items})")