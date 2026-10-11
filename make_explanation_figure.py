"""
make_explanation_figure.py  (v2)
───────────────────────────────────────────────────────────────────────
Rebuilt to address two reviewer points.

R1.2 — inference setting.
  v1 produced the counterfactual by enabling the exposure correction at
  inference, which is NOT the deployed configuration. v2 compares the
  trained FairSASRec against the trained SASRec backbone, both scored
  with the exposure term at zero, exactly as in the reported results.

R1.3 — systematic evidence.
  v1 showed a single hand-picked example. v2 measures the intervention
  over the whole test set: how often the top-1 recommendation changes,
  and whether the replacement item is, on average, less
  group-skewed than the item it displaced. A paired t-test is reported.

USAGE
    python make_explanation_figure.py --run p1 --dataset ml-1m

Outputs
    runs/<run>/figures/fig5_explanation.{png,pdf}
    runs/<run>/final/explanation_stats.json
"""

import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='p1')
ap.add_argument('--dataset', default='ml-1m')
ap.add_argument('--fair-ckpt', default=None)
ap.add_argument('--sasrec-ckpt', default=None)
a = ap.parse_args()

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = f'data/{a.dataset}'
ROOT     = os.path.join('runs', a.run)
FIGDIR   = os.path.join(ROOT, 'figures'); os.makedirs(FIGDIR, exist_ok=True)
FINDIR   = os.path.join(ROOT, 'final');   os.makedirs(FINDIR, exist_ok=True)
W = 20

FAIR_CKPT = a.fair_ckpt or (
    'results/FairSASRec-ml1m/fairsasrec_final.pt' if a.dataset == 'ml-1m'
    else f'{ROOT}/models/FairSASRec-{a.dataset}/fairsasrec_best.pt')
SAS_CKPT = a.sasrec_ckpt or (
    'results/SASRec-200/sasrec_best.pt' if a.dataset == 'ml-1m'
    else f'{ROOT}/models/SASRec-{a.dataset}/sasrec_best.pt')

print(f"Device: {DEVICE} | dataset: {a.dataset}")
print(f"  FairSASRec: {FAIR_CKPT}")
print(f"  SASRec    : {SAS_CKPT}")

# ── data ──────────────────────────────────────────────────────────────
train = pd.read_csv(f'{DATA_DIR}/train.csv')
val   = pd.read_csv(f'{DATA_DIR}/val.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']

hist = defaultdict(list)
for _, r in train.sort_values('timestamp').iterrows():
    hist[int(r['user_id'])].append(int(r['item_id']))
seen = defaultdict(set)
for _, r in pd.concat([train, val]).iterrows():
    seen[int(r['user_id'])].add(int(r['item_id']))
test_items = dict(zip(test['user_id'].astype(int),
                      test['item_id'].astype(int)))

user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_g = {}
if a.dataset == 'ml-1m':
    for line in open(f'{DATA_DIR}/users.dat'):
        p = line.strip().split('::')
        if len(p) >= 2: raw_g[int(p[0])] = p[1]
else:
    for line in open(f'{DATA_DIR}/u.user'):
        p = line.strip().split('|')
        if len(p) >= 3: raw_g[int(p[0])] = p[2]
gender = {user2idx[u]: g for u, g in raw_g.items() if u in user2idx}

titles = {}
inv = {v: k for k, v in meta['item2idx'].items()}
for path, sep, idx in [(f'{DATA_DIR}/movies.dat', '::', 1),
                       (f'{DATA_DIR}/u.item', '|', 1)]:
    if os.path.exists(path):
        raw = {}
        for line in open(path, encoding='latin-1'):
            p = line.strip().split(sep)
            if len(p) > idx: raw[int(p[0])] = p[idx]
        for i, orig in inv.items():
            if int(orig) in raw: titles[int(i)] = raw[int(orig)]
        break
name = lambda i: titles.get(i, f'item {i}')


# ── models ────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """SASRec encoder. `fair=True` adds the layer's parameters so a
    FairSASRec checkpoint loads; the correction itself is evaluated at
    zero exposure, matching the deployed inference setting."""
    def __init__(s, n, fair, d=64, h=256, heads=4, layers=2, w=W, p=0.2):
        super().__init__()
        s.item_emb = nn.Embedding(n + 1, d, padding_idx=0)
        s.pos_emb  = nn.Embedding(w + 1, d)
        s.dropout  = nn.Dropout(p)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=h, dropout=p,
            batch_first=True, norm_first=True) for _ in range(layers)])
        s.norm = nn.LayerNorm(d)
        s.scale = d ** 0.5
        if fair:
            s.fairness_alpha = nn.Parameter(torch.ones(1))
            s.fairness_proj  = nn.Linear(d, 1)
            s.explain_proj   = nn.Linear(d, d)

    def encode(s, seq):
        B, Wn = seq.shape
        pos = torch.arange(1, Wn + 1, device=seq.device).unsqueeze(0)
        x = s.dropout(s.item_emb(seq) + s.pos_emb(pos))
        m = nn.Transformer.generate_square_subsequent_mask(Wn,
                                                           device=seq.device)
        for l in s.layers: x = l(x, src_mask=m, is_causal=True)
        return s.norm(x)

    def scores(s, seq):
        out = s.encode(seq)
        state = out[:, -1, :]
        return state @ s.item_emb.weight.T, state, out

    def attention(s, out, state):
        q = s.explain_proj(state).unsqueeze(2)
        sc = torch.bmm(out, q).squeeze(2) / s.scale
        return F.softmax(sc, dim=-1)


def load(path, fair):
    m = Encoder(N_ITEMS, fair)
    missing, unexpected = m.load_state_dict(
        torch.load(path, map_location='cpu'), strict=False)
    if missing or unexpected:
        print(f"  [load warn] {path} missing={missing[:3]} "
              f"unexpected={unexpected[:3]}")
    return m.to(DEVICE).eval()

fair_m = load(FAIR_CKPT, fair=True)
sas_m  = load(SAS_CKPT,  fair=False)

def seq_of(uid):
    h = hist.get(uid, [])[-W:]
    return [0] * (W - len(h)) + h

def top1(model, uid):
    t = torch.LongTensor(seq_of(uid)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sc, state, out = model.scores(t)
    sc = sc.squeeze(0)[:N_ITEMS].clone()
    for i in seen.get(uid, ()):
        if i < N_ITEMS: sc[i] = -1e9
    return int(sc.argmax()), state, out


# ── gender-conditional exposure of each model ────────────────────────
users = sorted(test_items.keys())

def exposure_gap(model):
    male = np.zeros(N_ITEMS, dtype=np.float32)
    fem  = np.zeros(N_ITEMS, dtype=np.float32)
    with torch.no_grad():
        for uid in users:
            t = torch.LongTensor(seq_of(uid)).unsqueeze(0).to(DEVICE)
            sc = model.scores(t)[0].squeeze(0)[:N_ITEMS].clone()
            for i in seen.get(uid, ()):
                if i < N_ITEMS: sc[i] = -1e9
            arr = male if gender.get(uid, 'M') == 'M' else fem
            for it in sc.topk(10).indices.cpu().numpy():
                arr[int(it)] += 1
    tot = male + fem
    gap = np.zeros_like(tot); m = tot > 0
    gap[m] = np.abs(male[m] - fem[m]) / tot[m]
    return gap, tot

print("\nComputing gender exposure gaps from the backbone's own lists...")
gap, support = exposure_gap(sas_m)   # reference frame: the unconstrained model

# ── systematic comparison over all users (R1.3) ──────────────────────
print("Comparing top-1 recommendations across all test users...")
changed, gaps_before, gaps_after = [], [], []
items_before, items_after = [], []
example = None
for uid in users:
    b, _, _ = top1(sas_m, uid)
    f, state, out = top1(fair_m, uid)
    if b == f:
        continue
    changed.append(uid)
    items_before.append(b); items_after.append(f)
    gaps_before.append(float(gap[b]))
    gaps_after.append(float(gap[f]))
    if (example is None and support[b] >= 10 and support[f] >= 10
            and gap[b] - gap[f] > 0.10):
        with torch.no_grad():
            attn = fair_m.attention(out, state).squeeze(0).cpu().numpy()
            example = dict(uid=uid, before=b, after=f, attn=attn,
                       seq=seq_of(uid))

n_changed = len(changed)
frac = n_changed / len(users)
gb, ga = np.array(gaps_before), np.array(gaps_after)
t_stat, p_val = stats.ttest_rel(gb, ga)

# The gap is undefined for items the backbone never surfaces; those are
# counted as zero above, which inflates the apparent reduction. Restrict
# to pairs where both items carry real exposure support.
MIN_SUPPORT = 10
ib, ia = np.array(items_before), np.array(items_after)
zero_after = float((support[ia] == 0).mean())
mask = (support[ib] >= MIN_SUPPORT) & (support[ia] >= MIN_SUPPORT)
gb_r, ga_r = gb[mask], ga[mask]
t_r, p_r = stats.ttest_rel(gb_r, ga_r) if mask.sum() > 30 else (np.nan, np.nan)

print(f"\n{'='*66}")
print(f"Top-1 changed for {n_changed} / {len(users)} users ({frac:.1%})")
print(f"\n  UNRESTRICTED (undefined gaps counted as zero)")
print(f"    displaced {gb.mean():.4f} -> replacement {ga.mean():.4f}  "
      f"({100*(ga.mean()-gb.mean())/gb.mean():+.1f}%)")
print(f"    t={t_stat:.2f}, p={p_val:.2e}")
print(f"    replacements with zero support : {zero_after:.1%}  "
      f"<-- inflates the above")
print(f"\n  RESTRICTED (both items with >= {MIN_SUPPORT} exposures)")
if mask.sum() > 30:
    print(f"    cases     : {int(mask.sum())}")
    print(f"    displaced {gb_r.mean():.4f} -> replacement {ga_r.mean():.4f}  "
          f"({100*(ga_r.mean()-gb_r.mean())/gb_r.mean():+.1f}%)")
    print(f"    t={t_r:.2f}, p={p_r:.2e}")
    print(f"    replacement less skewed in : {(ga_r < gb_r).mean():.1%}")
else:
    print(f"    too few cases ({int(mask.sum())})")
print('='*66)

json.dump({
    'n_users': len(users), 'n_changed': n_changed, 'fraction': frac,
    'gap_displaced_mean': float(gb.mean()),
    'gap_replacement_mean': float(ga.mean()),
    'mean_reduction': float((gb - ga).mean()),
    'pct_reduction': float(100*(ga.mean()-gb.mean())/gb.mean()),
    't_stat': float(t_stat), 'p_value': float(p_val),
    'pct_cases_improved': float((ga < gb).mean()),
    'zero_support_after': zero_after,
    'restricted': None if mask.sum() <= 30 else {
        'min_support': MIN_SUPPORT, 'n': int(mask.sum()),
        'before': float(gb_r.mean()), 'after': float(ga_r.mean()),
        'pct_reduction': float(100*(ga_r.mean()-gb_r.mean())/gb_r.mean()),
        't_stat': float(t_r), 'p_value': float(p_r),
        'pct_cases_improved': float((ga_r < gb_r).mean())},
    'example': None if example is None else {
        'user': int(example['uid']),
        'gender': gender.get(example['uid'], '?'),
        'backbone_top1': int(example['before']),
        'backbone_title': name(example['before']),
        'backbone_gap': float(gap[example['before']]),
        'fair_top1': int(example['after']),
        'fair_title': name(example['after']),
        'fair_gap': float(gap[example['after']]),
    }}, open(f'{FINDIR}/explanation_stats.json', 'w'), indent=2)
print(f"Saved {FINDIR}/explanation_stats.json")

# ── figure ────────────────────────────────────────────────────────────
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300,
                     'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(1, 3, figsize=(14, 4.0),
                       gridspec_kw={'width_ratios': [1.3, 0.85, 1.0]})

# (a) attention for the example user
if example:
    seq, attn = example['seq'], example['attn']
    keep = [i for i, it in enumerate(seq) if it != 0]
    labels = [name(seq[i])[:24] for i in keep]
    vals = [attn[i] for i in keep]
    colors = ['#eb6834' if v == max(vals) else '#2a78d6' for v in vals]
    ax[0].barh(range(len(keep)), vals, color=colors)
    ax[0].set_yticks(range(len(keep)))
    ax[0].set_yticklabels(labels, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('attention weight')
    ax[0].set_title(f"(a) What drove the recommendation, user {example['uid']}")

    # (b) the one case
    items = [example['before'], example['after']]
    gaps = [gap[i] for i in items]
    bars = ax[1].bar(['SASRec\ntop-1', 'FairSASRec\ntop-1'], gaps,
                     color=['#2a78d6', '#eb6834'], width=.55)
    for b_, g_, it in zip(bars, gaps, items):
        ax[1].text(b_.get_x() + b_.get_width()/2, g_,
                   f'{g_:.3f}\n{name(it)[:20]}', ha='center', va='bottom',
                   fontsize=7)
    ax[1].set_ylabel('gender exposure gap')
    ax[1].set_ylim(0, max(gaps) * 1.45)
    ax[1].set_title('(b) One recommendation')

# (c) displaced vs replacement, all cases and support-restricted
bins = np.linspace(0, 1, 31)
ax[2].hist(gb, bins=bins, alpha=.20, color='#2a78d6',
           label=f'displaced, all ({n_changed})')
ax[2].hist(ga, bins=bins, alpha=.20, color='#eb6834',
           label=f'replacement, all ({n_changed})')
if mask.sum() > 30:
    ax[2].hist(gb_r, bins=bins, histtype='step', lw=1.8, color='#2a78d6',
               label=f'displaced, supported ({int(mask.sum())})')
    ax[2].hist(ga_r, bins=bins, histtype='step', lw=1.8, color='#eb6834',
               label=f'replacement, supported ({int(mask.sum())})')
    ax[2].axvline(gb_r.mean(), color='#2a78d6', ls='--', lw=1.3)
    ax[2].axvline(ga_r.mean(), color='#eb6834', ls='--', lw=1.3)
ax[2].set_xlabel('gender exposure gap')
ax[2].set_ylabel('number of recommendations')
ax[2].legend(fontsize=6.5, loc='upper center')
ax[2].set_title('(c) Displaced vs replacement items')

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{FIGDIR}/fig5_explanation.{ext}', bbox_inches='tight')
print(f"Saved {FIGDIR}/fig5_explanation.png")
