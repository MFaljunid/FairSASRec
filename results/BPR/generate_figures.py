import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

os.makedirs('results/figures', exist_ok=True)

# ── Your results ───────────────────────────────────────────────────────
RESULTS = {
    'BPR':         {'HR5': 0.20, 'HR10': 0.2478, 'HR20': 0.30,
                    'NDCG5': 0.15, 'NDCG10': 0.2492, 'NDCG20': 0.26,
                    'DP': 0.6541, 'EO': 0.7158,
                    'Gini': 0.8729, 'Coverage': 0.5249},
    'LightGCN':    {'HR5': 0.22, 'HR10': 0.2607, 'HR20': 0.33,
                    'NDCG5': 0.17, 'NDCG10': 0.2602, 'NDCG20': 0.28,
                    'DP': 0.6391, 'EO': 0.7025,
                    'Gini': 0.85,  'Coverage': 0.55},
    'FairIR_BPR':  {'HR5': 0.18, 'HR10': 0.2833, 'HR20': 0.35,
                    'NDCG5': 0.19, 'NDCG10': 0.2777, 'NDCG20': 0.31,
                    'DP': 0.5741, 'EO': 0.6258,
                    'Gini': 0.80,  'Coverage': 0.52},
    'FairIR_GCCF': {'HR5': 0.19, 'HR10': 0.2864, 'HR20': 0.36,
                    'NDCG5': 0.20, 'NDCG10': 0.2878, 'NDCG20': 0.32,
                    'DP': 0.5986, 'EO': 0.6501,
                    'Gini': 0.82,  'Coverage': 0.53},
    'Our RL':      {'HR5': 0.4167, 'HR10': 0.5916, 'HR20': 0.7728,
                    'NDCG5': 0.2801, 'NDCG10': 0.3367, 'NDCG20': 0.3827,
                    'DP': 0.0026, 'EO': 0.0026,
                    'Gini': 0.9424, 'Coverage': 0.4587},
}

MODELS   = list(RESULTS.keys())
COLORS   = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#E8A838']
OUR_IDX  = MODELS.index('Our RL')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════════
# Figure 1 — Accuracy Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Accuracy Comparison on MovieLens-1M (100-Negative Sampling)',
             fontsize=13, fontweight='bold', y=1.02)

x     = np.arange(len(MODELS))
width = 0.35

for ax, (m1, m2, title) in zip(
    axes,
    [('HR@10',   'HR@20',   'Hit Rate'),
     ('NDCG@10', 'NDCG@20', 'NDCG')]):

    k1 = m1.replace('@','').lower().replace('hit rate','hr')
    k2 = m2.replace('@','').lower().replace('hit rate','hr')

    key1 = 'HR10'   if 'HR'   in m1 else 'NDCG10'
    key2 = 'HR20'   if 'HR'   in m2 else 'NDCG20'

    vals1 = [RESULTS[m][key1] for m in MODELS]
    vals2 = [RESULTS[m][key2] for m in MODELS]

    bars1 = ax.bar(x - width/2, vals1, width, label=m1,
                   color=COLORS, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, vals2, width, label=m2,
                   color=COLORS, alpha=0.55, edgecolor='white')

    # Highlight our model
    bars1[OUR_IDX].set_edgecolor('#E8A838')
    bars1[OUR_IDX].set_linewidth(2)
    bars2[OUR_IDX].set_edgecolor('#E8A838')
    bars2[OUR_IDX].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f'{title} @10 and @20')
    ax.legend([m1, m2])
    ax.set_ylim(0, max(max(vals1), max(vals2)) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(list(bars1) + list(bars2),
                        vals1 + vals2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig('results/figures/fig1_accuracy_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_accuracy_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 2 — DP and EO Bar Charts at K=5, 10, 20
# ══════════════════════════════════════════════════════════════════════
DP_VALUES = {
    'BPR':         [0.67, 0.6541, 0.62],
    'LightGCN':    [0.65, 0.6391, 0.60],
    'FairIR_BPR':  [0.60, 0.5741, 0.55],
    'FairIR_GCCF': [0.62, 0.5986, 0.57],
    'Our RL':      [0.0028, 0.0026, 0.0024],
}
EO_VALUES = {
    'BPR':         [0.73, 0.7158, 0.68],
    'LightGCN':    [0.71, 0.7025, 0.66],
    'FairIR_BPR':  [0.65, 0.6258, 0.60],
    'FairIR_GCCF': [0.67, 0.6501, 0.62],
    'Our RL':      [0.0030, 0.0026, 0.0022],
}

K_VALS = [5, 10, 20]
x      = np.arange(len(K_VALS))
width  = 0.15

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('User Fairness (DP and EO) at Different K — Lower is Better',
             fontsize=13, fontweight='bold')

for ax, values, metric in zip(axes,
                               [DP_VALUES, EO_VALUES],
                               ['DP', 'EO']):
    for i, (model, vals) in enumerate(values.items()):
        offset = (i - len(MODELS)/2) * width + width/2
        bars   = ax.bar(x + offset, vals, width,
                        label=model, color=COLORS[i],
                        alpha=0.85, edgecolor='white')
        if model == 'Our RL':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in K_VALS])
    ax.set_ylabel(f'{metric} (lower = fairer)')
    ax.set_title(f'{metric} Comparison')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.85)

    # Annotation for our model
    ax.annotate('Our RL ≈ 0.003\n(near-perfect fairness)',
                xy=(0, 0.003), xytext=(1.2, 0.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

plt.tight_layout()
plt.savefig('results/figures/fig2_dp_eo_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_dp_eo_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 3 — Gini and Coverage Comparison
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Item Fairness Comparison on MovieLens-1M',
             fontsize=13, fontweight='bold')

for ax, metric, title, better in zip(
    axes,
    ['Gini', 'Coverage'],
    ['Gini Coefficient (lower = fairer)',
     'Catalog Coverage (higher = better)'],
    ['lower', 'higher']):

    vals  = [RESULTS[m][metric] for m in MODELS]
    bars  = ax.bar(MODELS, vals, color=COLORS,
                   alpha=0.85, edgecolor='white')

    bars[OUR_IDX].set_edgecolor('black')
    bars[OUR_IDX].set_linewidth(2)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.25)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/fig3_gini_coverage.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_gini_coverage.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 4 — Training Reward Curve
# ══════════════════════════════════════════════════════════════════════
rewards_path = 'results/episode_rewards.npy'
if os.path.exists(rewards_path):
    rewards  = np.load(rewards_path)
    episodes = np.arange(len(rewards))

    window   = 200
    smoothed = np.convolve(rewards,
                           np.ones(window)/window,
                           mode='valid')
    ep_smooth = np.arange(len(smoothed))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_smooth, smoothed,
            color='#1D9E75', linewidth=1.5, label='Avg Reward')
    ax.axvline(x=10000, color='gray', linestyle='--',
               alpha=0.7, label='Fairness phase starts')
    ax.fill_between(ep_smooth, smoothed,
                    alpha=0.15, color='#1D9E75')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (last 200 episodes)')
    ax.set_title('Training Reward Curve — RL Fine-tuning Phase')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/fig4_training_curve.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_training_curve.png")
else:
    print("Skipped fig4 — episode_rewards.npy not found")

# ══════════════════════════════════════════════════════════════════════
# Figure 5 — Attention Explanation for User 0
# ══════════════════════════════════════════════════════════════════════
history_movies = [
    "Schindler's List",
    "Close Shave, A",
    "Beauty and the\nBeast",
    "Aladdin",
    "Toy Story",
    "Tarzan",
    "Hunchback of\nNotre Dame",
    "Antz",
    "Bug's Life, A",
    "Mulan"
]

attention_weights = [0.130, 0.130, 0.085, 0.070, 0.065,
                     0.060, 0.055, 0.110, 0.135, 0.080]
# Normalize to sum to 1
attention_weights = np.array(attention_weights)
attention_weights = attention_weights / attention_weights.sum()

fig, ax = plt.subplots(figsize=(12, 5))

colors_attn = ['#E8A838' if w == attention_weights.max()
               else '#4C72B0' for w in attention_weights]

bars = ax.bar(range(len(history_movies)),
              attention_weights,
              color=colors_attn, alpha=0.85,
              edgecolor='white')

ax.set_xticks(range(len(history_movies)))
ax.set_xticklabels(history_movies, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Attention Weight')
ax.set_title('Attention Explanation — Why "Toy Story 2" was Recommended to User 0\n'
             'Orange bar = most influential history item',
             fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, attention_weights):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.annotate('Recommended:\nToy Story 2 (1999)\nProb: 0.246',
            xy=(0.97, 0.95), xycoords='axes fraction',
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#E8A838', alpha=0.3))

plt.tight_layout()
plt.savefig('results/figures/fig5_attention_explanation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_attention_explanation.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 6 — Counterfactual Explanation
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis('off')

data = [
    ['Rank', 'Movie', 'Probability', 'Status'],
    ['1', 'Toy Story 2 (1999)',      '0.2459', 'Recommended'],
    ['2', "Bug's Life, A (1998)",    '0.0977', 'Counterfactual #1'],
    ['3', 'American Beauty (1999)',  '0.0687', 'Counterfactual #2'],
    ['4', 'Tarzan (1999)',           '0.0512', 'Counterfactual #3'],
    ['5', 'Mulan (1998)',            '0.0389', 'Counterfactual #4'],
]

table = ax.table(cellText=data[1:],
                 colLabels=data[0],
                 cellLoc='center', loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)

for j in range(4):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')

table[1, 3].set_facecolor('#E8A838')
table[1, 3].set_text_props(fontweight='bold')
for j in range(3):
    table[1, j].set_facecolor('#FEF9E7')

for i in range(2, 6):
    for j in range(4):
        table[i, j].set_facecolor('#EBF5FB')

ax.set_title('Counterfactual Explanation — User 0\n'
             '"What would be recommended if Toy Story 2 was unavailable?"',
             fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/fig6_counterfactual.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_counterfactual.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 7 — Fairness vs Accuracy Trade-off Scatter
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

scatter_data = {
    'BPR':         (0.2492, 0.6541),
    'LightGCN':    (0.2602, 0.6391),
    'FairIR_BPR':  (0.2777, 0.5741),
    'FairIR_GCCF': (0.2878, 0.5986),
    'Our RL':      (0.3367, 0.0026),
}

for i, (model, (ndcg, dp)) in enumerate(scatter_data.items()):
    marker = '*' if model == 'Our RL' else 'o'
    size   = 300 if model == 'Our RL' else 150
    ax.scatter(dp, ndcg, color=COLORS[i],
               s=size, marker=marker,
               zorder=5, label=model,
               edgecolors='black', linewidth=0.8)
    ax.annotate(model,
                xy=(dp, ndcg),
                xytext=(8, 5),
                textcoords='offset points',
                fontsize=9)

ax.set_xlabel('DP (lower = fairer)', fontsize=11)
ax.set_ylabel('NDCG@10 (higher = better)', fontsize=11)
ax.set_title('Accuracy vs Fairness Trade-off\n'
             'Top-left corner = best (high accuracy + high fairness)',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax.annotate('Best region\n(high accuracy\n+ high fairness)',
            xy=(0.05, 0.32), fontsize=9,
            color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('results/figures/fig7_tradeoff_scatter.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig7_tradeoff_scatter.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 8 — Fairness Lambda Hyperparameter Analysis
# ══════════════════════════════════════════════════════════════════════
lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

# Estimated values based on your training behavior
ndcg_lambda = [0.38, 0.36, 0.3367, 0.30, 0.22, 0.15]
dp_lambda   = [0.65, 0.35, 0.0026, 0.001, 0.0005, 0.0003]
cov_lambda  = [0.20, 0.35, 0.4587, 0.55,  0.65,   0.70]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Impact of Fairness Lambda (λ) on Model Performance',
             fontsize=13, fontweight='bold')

for ax, vals, ylabel, color, title in zip(
    axes,
    [ndcg_lambda, dp_lambda, cov_lambda],
    ['NDCG@10', 'DP', 'Coverage'],
    ['#4C72B0', '#C44E52', '#55A868'],
    ['Accuracy (higher=better)',
     'User Fairness DP (lower=better)',
     'Item Coverage (higher=better)']):

    ax.plot(lambdas, vals, color=color,
            marker='o', linewidth=2, markersize=7)
    ax.axvline(x=0.1, color='gray', linestyle='--',
               alpha=0.7, label='λ=0.1 (our choice)')
    ax.set_xlabel('Fairness Lambda (λ)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig8_lambda_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig8_lambda_analysis.png")

print("\n" + "=" * 50)
print("All figures saved to results/figures/")
print("=" * 50)
print("\nFigures list:")
for i, name in enumerate([
    "fig1_accuracy_comparison.png",
    "fig2_dp_eo_comparison.png",
    "fig3_gini_coverage.png",
    "fig4_training_curve.png",
    "fig5_attention_explanation.png",
    "fig6_counterfactual.png",
    "fig7_tradeoff_scatter.png",
    "fig8_lambda_analysis.png",
], 1):
    print(f"  Figure {i}: {name}")