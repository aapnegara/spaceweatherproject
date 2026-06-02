"""
feature_stability.py
====================
Stage 1 of the two-stage ML pipeline for explainable solar flare nowcasting.

Assesses the stability of Random Forest feature importances across k=5 and k=10
StratifiedGroupKFold cross-validation folds using all 24 SHARP parameters from the
SDO/HMI MVTS dataset. Active region (HARPNUM) grouping prevents data leakage
between temporally correlated observations of the same AR.

The final 8 SHARP parameters selected for Stage 2 are those consistently ranked
in the top quartile (low average rank, low rank variance) across both k values:
    R_VALUE, TOTUSJH, USFLUX, TOTPOT, SAVNCPP, MEANSHR, ABSNJZH, MEANPOT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix


# =============================================================================
# Data Loading
# =============================================================================

file_path = '/STER/agungp/space-weather/clustering_ar_sf_hbaeke/Data/AllData_sharp.csv'

all_24_params = [
    'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45',
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY',
    'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE'
]

cols_to_load = all_24_params + ['MFLARE', 'XFLARE', 'SHARPnum']

# Dataset is ~2.6M rows; chunked loading avoids memory overflow.
chunks = []
for chunk in pd.read_csv(
    file_path,
    sep='\t',
    on_bad_lines='skip',
    engine='python',
    usecols=cols_to_load,
    chunksize=200_000
):
    # Binary label: 1 if any M- or X-class flare is concurrent with this SHARP timestep
    chunk['flare_label'] = ((chunk['MFLARE'] == 1) | (chunk['XFLARE'] == 1)).astype(int)
    chunks.append(chunk[all_24_params + ['SHARPnum', 'flare_label']].dropna())

df_full = pd.concat(chunks, ignore_index=True)
X_all = df_full[all_24_params].values
y_all = df_full['flare_label'].values
groups_all = df_full['SHARPnum'].values

print(f"Total samples: {len(y_all):,}")
print(f"Total flares: {y_all.sum()}")


# =============================================================================
# Log Transformation
# =============================================================================
# Several SHARP parameters span multiple orders of magnitude (extensive quantities
# such as TOTUSJH and TOTPOT), producing heavily right-skewed distributions.
# Log-transformation reduces skewness and improves RF split quality.
#
# Shift formula: ln(x + |min| + epsilon) handles negative and zero values.
# epsilon = 0.01 for standard params; 0.0001 for MEANPOT (smaller dynamic range).
#
# R_VALUE and MEANSHR are not transformed — their distributions are
# approximately symmetric (verified by the skewness diagnostic below).
# =============================================================================

log_params = ['TOTUSJH', 'TOTPOT', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTBSQ', 'TOTUSJZ']
log_params_small = ['MEANPOT']

df_log = df_full[all_24_params].copy()
for col in log_params:
    col_min = df_log[col].min()
    df_log[col] = np.log(df_log[col] + abs(col_min) + 0.01)
for col in log_params_small:
    col_min = df_log[col].min()
    df_log[col] = np.log(df_log[col] + abs(col_min) + 0.0001)

X_log = df_log.values


# =============================================================================
# Skewness Diagnostic
# =============================================================================
# Verifies that the log-transform choices above are justified by the data.
# Parameters with |skewness| > 2 are flagged as strong candidates for
# log-transformation; inconsistencies (high skew but not logged, or vice versa)
# are printed for review.
# =============================================================================

print("\n" + "="*65)
print("Skewness diagnostic — raw (untransformed) distributions")
print("="*65)

skew_results = []
for col in all_24_params:
    vals = df_full[col].dropna().values
    s = skew(vals)
    skew_results.append({'parameter': col, 'skewness': s})

skew_df = pd.DataFrame(skew_results).sort_values('skewness', key=abs, ascending=False)

currently_logged = set(log_params + log_params_small)
skew_df['log_transformed'] = skew_df['parameter'].apply(
    lambda p: 'YES' if p in currently_logged else 'no'
)

def flag(row):
    high_skew = abs(row['skewness']) > 2
    logged = row['log_transformed'] == 'YES'
    if high_skew and not logged:
        return '⚠ skewed but NOT logged'
    if not high_skew and logged:
        return '⚠ logged but low skew'
    return ''

skew_df['note'] = skew_df.apply(flag, axis=1)

print(f"\n{'Parameter':<12} {'Skewness':>10} {'Log-trans?':>12}   Note")
print("-" * 65)
for _, row in skew_df.iterrows():
    print(f"  {row['parameter']:<10} {row['skewness']:>10.2f} "
          f"{row['log_transformed']:>12}   {row['note']}")

skew_df.to_csv('skewness_diagnostic.csv', index=False)
print("\nSaved: skewness_diagnostic.csv")

# Histogram grid: blue = log-transformed, orange = kept raw
fig, axes = plt.subplots(6, 4, figsize=(16, 18))
axes = axes.flatten()

for i, col in enumerate(all_24_params):
    vals = df_full[col].dropna().values
    s = skew(vals)
    color = 'steelblue' if col in currently_logged else 'darkorange'
    axes[i].hist(vals, bins=80, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[i].set_title(f'{col}\nskew = {s:.2f}', fontsize=10)
    axes[i].set_ylabel('count', fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].set_yscale('log')

plt.suptitle(
    'Raw distributions of 24 SHARP parameters (y-axis log-scaled)\n'
    'Blue = log-transformed in pipeline   |   Orange = kept raw',
    fontsize=12, y=1.00
)
plt.tight_layout()
plt.savefig('skewness_diagnostic.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: skewness_diagnostic.png")
print("="*65 + "\n")


# =============================================================================
# Cross-Validation Stability Analysis
# =============================================================================

def run_cv_stability(X, y, groups, n_splits, random_state=42):
    """
    Run k-fold RF importance stability analysis.

    Uses StratifiedGroupKFold to preserve class balance across folds while
    ensuring all timesteps of a given AR (HARPNUM) remain in a single fold,
    preventing temporal leakage.

    Class imbalance is addressed per fold:
      1. RandomUnderSampler caps the majority class at 5000 samples.
      2. SMOTE oversamples the minority class to match the majority cap.
    Both resampling steps are applied exclusively to training data.

    Feature importance ranks (not raw scores) are tracked across folds.
    Rank variance quantifies stability: a parameter ranked consistently near
    the top across all folds is a reliable predictor.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    groups : ndarray, shape (n_samples,)  — HARPNUM identifiers
    n_splits : int  — number of CV folds (5 or 10)
    random_state : int

    Returns
    -------
    rank_avg : dict  — mean importance rank per parameter
    rank_var : dict  — rank variance per parameter
    tss_per_fold : list of float  — TSS per fold
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    importance_ranks = {p: [] for p in all_24_params}
    tss_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        X_train_f, X_test_f = X[train_idx], X[test_idx]
        y_train_f, y_test_f = y[train_idx], y[test_idx]

        # Fit scaler on training fold only to prevent data leakage
        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_test_f = scaler.transform(X_test_f)

        # Resampling: undersample majority, then SMOTE minority up to 5000 each
        n_flares = y_train_f.sum()
        n_keep = min(5000, (y_train_f == 0).sum())

        under = RandomUnderSampler(
            sampling_strategy={0: n_keep, 1: n_flares}, random_state=42)
        X_u, y_u = under.fit_resample(X_train_f, y_train_f)

        smote = SMOTE(
            sampling_strategy={1: n_keep}, random_state=42,
            k_neighbors=min(5, n_flares - 1))
        X_bal, y_bal = smote.fit_resample(X_u, y_u)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_bal, y_bal)

        # Evaluate on held-out fold (unmodified test set)
        y_pred = rf.predict(X_test_f)
        TN, FP, FN, TP = confusion_matrix(y_test_f, y_pred).ravel()
        tss_per_fold.append((TP / (TP + FN)) - (FP / (FP + TN)))

        # Convert importances to ranks (rank 1 = most important)
        importances = rf.feature_importances_
        ranks = len(all_24_params) - importances.argsort().argsort()
        for i, param in enumerate(all_24_params):
            importance_ranks[param].append(ranks[i])

    rank_avg = {p: np.mean(importance_ranks[p]) for p in all_24_params}
    rank_var = {p: np.var(importance_ranks[p]) for p in all_24_params}
    return rank_avg, rank_var, tss_per_fold


print("\n--- Running k=5 ---")
avg5, var5, tss5 = run_cv_stability(X_log, y_all, groups_all, n_splits=5)
print(f"k=5  TSS: {np.mean(tss5):.3f} ± {np.std(tss5):.3f}")

print("\n--- Running k=10 ---")
avg10, var10, tss10 = run_cv_stability(X_log, y_all, groups_all, n_splits=10)
print(f"k=10 TSS: {np.mean(tss10):.3f} ± {np.std(tss10):.3f}")

print("\n--- Comparison: k=5 vs k=10 ---")
print(f"{'Param':<12} {'k=5 rank':>10} {'k=5 var':>10} | {'k=10 rank':>11} {'k=10 var':>10}")
print("-" * 65)
all_sorted = sorted(all_24_params, key=lambda p: avg5[p])
for p in all_sorted:
    print(f"  {p:<10} {avg5[p]:>10.2f} {var5[p]:>10.2f} | "
          f"{avg10[p]:>11.2f} {var10[p]:>10.2f}")


# =============================================================================
# Stability Scatter Plot
# =============================================================================

def plot_stability(rank_avg, rank_var, k_value, save_path):
    """
    Plot RF importance stability: average rank vs. rank variance across folds.

    Parameters in the bottom-left region (low rank = important, low variance =
    stable) are consistently informative across folds and are selected for Stage 2.
    The dashed ellipse marks the selection region corresponding to the final 8
    SHARP parameters. Visualization style follows Ran et al. (2022, ApJ).
    """
    from matplotlib.patches import Ellipse

    sorted_params = sorted(all_24_params, key=lambda p: rank_avg[p])
    palette = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Set3').colors[:4])
    color_map = {p: palette[i] for i, p in enumerate(sorted_params)}

    fig, ax = plt.subplots(figsize=(13, 8))

    for p in sorted_params:
        ax.scatter(
            rank_avg[p], rank_var[p],
            s=140, color=color_map[p],
            edgecolor='black', linewidth=0.7,
            zorder=5,
            label=f'{p:<10}  (rank {rank_avg[p]:>4.1f}, var {rank_var[p]:>4.2f})'
        )

    good_region = Ellipse(
        xy=(6.5, 0.3), width=12.5, height=2.4, angle=15,
        edgecolor='navy', facecolor='none',
        linestyle='--', linewidth=1.8, alpha=0.7, zorder=2
    )
    ax.add_patch(good_region)
    ax.text(6.5, 1.7, 'Final 8 (selection region)',
            ha='center', va='bottom', fontsize=10, style='italic',
            color='navy', fontweight='bold')

    ax.set_xlabel('Average Rank  (lower = more important)', fontsize=12)
    ax.set_ylabel('Rank Variance  (lower = more stable)', fontsize=12)
    ax.set_title(
        f'Feature Importance Stability — RF across {k_value}-fold CV\n'
        f'(Bottom-left: consistently important across folds)',
        fontsize=13
    )
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc='upper left',
        fontsize=8.5, frameon=True,
        title='Parameters (sorted by avg rank)',
        title_fontsize=9, handletextpad=0.4, labelspacing=0.5
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


plot_stability(avg5,  var5,  k_value=5,  save_path='feature_stability_k5.png')
plot_stability(avg10, var10, k_value=10, save_path='feature_stability_k10.png')


# =============================================================================
# Export Stability Table
# =============================================================================

print("\n--- Feature Stability Ranking (k=5) ---")
print(f"{'Parameter':<12} {'Avg Rank':>10} {'Rank Variance':>15}")
print("-" * 40)

stability_df = pd.DataFrame({
    'parameter': all_24_params,
    'avg_rank': [avg5[p] for p in all_24_params],
    'rank_variance': [var5[p] for p in all_24_params]
}).sort_values('avg_rank')

print(stability_df.to_string(index=False))
stability_df.to_csv('feature_stability.csv', index=False)
print("\nSaved: feature_stability.csv")