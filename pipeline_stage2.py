"""
pipeline_stage2.py
==================
Stage 2 of the two-stage ML pipeline for explainable solar flare nowcasting.

Takes the 8 SHARP parameters selected in Stage 1 (feature_stability.py) and runs:
  1.  Data loading and label construction (Case 2 and Case 3)
  2.  Pearson correlation matrix of the final 8 parameters
  3.  Log-transform + StandardScaler preprocessing
  4.  PCA: scree plot, loadings, 2D projection coloured by flare label
  5.  K-Means clustering with elbow plot and Kneedle-based K selection
  6.  t-SNE 2D visualization coloured by flare label and cluster
  7.  Random Forest — Leave-One-Year-Out cross-validation × 2 cases
  8.  Feature importance comparison: Case 2 vs Case 3
  9. Summary

Classification scheme (Baeke et al. 2025, MNRAS):
  Case 2 — Flare (C+M+X) vs No-Flare
  Case 3 — Alert (M+X)   vs No-Alert

AR-aware splitting: all timesteps of a given active region (HARPNUM) are
assigned exclusively to either the training or the test set, preventing leakage
from temporally correlated observations of the same AR.
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from kneed import KneeLocator

print("=" * 65)
print("SOLAR FLARE NOWCASTING PIPELINE — STAGE 2")
print("=" * 65)


# =============================================================================
# Section 1 — Data Loading
# =============================================================================
# AllData_sharp.csv (~2.6M rows) is read in 200k-row chunks to avoid OOM.
# Labels are constructed at row level: a timestep is positive if any flare
# of the relevant class is concurrent with that SHARP measurement.
# The year column is extracted from the timestamp for LOYO-CV splitting.
# =============================================================================

file_path = '/STER/agungp/space-weather/clustering_ar_sf_hbaeke/Data/AllData_sharp.csv'

final_params = [
    'TOTUSJH', 'TOTPOT', 'SAVNCPP', 'USFLUX',
    'MEANPOT', 'MEANSHR', 'R_VALUE', 'ABSNJZH',
]

group_col = 'SHARPnum'
time_col  = 'Timestamp'

cols_to_load = final_params + ['CFLARE', 'MFLARE', 'XFLARE', group_col, time_col]

print(f"\nLoading dataset: {file_path}")

chunks = []
for chunk in pd.read_csv(
    file_path,
    sep='\t',
    on_bad_lines='skip',
    engine='python',
    usecols=cols_to_load,
    chunksize=200_000
):
    chunk = chunk[final_params + ['CFLARE', 'MFLARE', 'XFLARE', group_col, time_col]]
    chunk = chunk.dropna(subset=final_params + ['CFLARE', 'MFLARE', 'XFLARE', group_col, time_col])
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True).reset_index(drop=True)

df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col])
df['year'] = df[time_col].dt.year

# Check for ARs whose observations span multiple calendar years.
# These are handled in the LOYO split by excluding their training-set rows
# from the test year's AR list (see Section 8).
ar_years = df.groupby('SHARPnum')['year'].nunique()
crossing_ar = ar_years[ar_years > 1]

print(f"\nTotal ARs: {df['SHARPnum'].nunique():,}")
print(f"ARs crossing year boundary: {len(crossing_ar):,}")
if len(crossing_ar) > 0:
    examples = df[df['SHARPnum'].isin(crossing_ar.index)].groupby('SHARPnum')['year'].unique().head(5)
    for ar, yrs in examples.items():
        print(f"  AR {ar}: {list(yrs)}")

# Binary labels
df['flare_mx']  = ((df['MFLARE'] == 1) | (df['XFLARE'] == 1)).astype(int)
df['flare_cmx'] = ((df['CFLARE'] == 1) | (df['MFLARE'] == 1) | (df['XFLARE'] == 1)).astype(int)

y = df['flare_mx'].values   # primary label for exploration sections (Case 3)

n_ar    = df[group_col].nunique()
n_ar_mx = int(df.groupby(group_col)['flare_mx'].max().sum())

print(f"\nDataset loaded:")
print(f"  Total rows:                {len(df):,}")
print(f"  Year range:                {df['year'].min()}–{df['year'].max()}")
print(f"  Unique active regions:     {n_ar:,}")
print(f"  ARs with M/X flares:       {n_ar_mx:,} / {n_ar:,}  ({n_ar_mx/n_ar*100:.2f}%)")
print(f"  M/X flare rows (Case 3):   {df['flare_mx'].sum():,}  ({df['flare_mx'].mean()*100:.4f}%)")
print(f"  C/M/X flare rows (Case 2): {df['flare_cmx'].sum():,}  ({df['flare_cmx'].mean()*100:.4f}%)")

X_raw = df[final_params].values


# =============================================================================
# Section 2 — Correlation Matrix
# =============================================================================
# Pearson correlation is scale-invariant, so raw values are used here.
# The purpose is to verify that the final 8 parameters have acceptably low
# inter-correlation after removing TOTUSJZ (r=0.99 with TOTUSJH) in Stage 1.
# Remaining correlated pairs (e.g. TOTUSJH–ABSNJZH) are retained because they
# represent physically distinct quantities: unsigned total helicity vs. net
# helicity — both informative for flare forecasting.
# =============================================================================

print("\n" + "="*65)
print("Section 2 — Correlation Matrix")
print("="*65)

corr_matrix = df[final_params].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size': 9})
plt.title('Pearson Correlation Matrix — Final 8 SHARP Parameters\n', fontsize=12)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/correlation_matrix_final8.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: correlation_matrix_final8.png")

print("\nHighest absolute correlations among final 8:")
corr_pairs = [
    (final_params[i], final_params[j], corr_matrix.iloc[i, j])
    for i in range(len(final_params))
    for j in range(i+1, len(final_params))
]
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for p1, p2, r in corr_pairs[:5]:
    print(f"  {p1} — {p2}: r = {r:.3f}")


# =============================================================================
# Section 3 — Log-transform + StandardScaler (Exploration Preprocessing)
# =============================================================================
# Several SHARP parameters (extensive quantities: TOTUSJH, TOTPOT, USFLUX,
# ABSNJZH, SAVNCPP) are heavily right-skewed due to a small number of
# magnetically complex ARs with extreme values. Log-transformation reduces
# skewness and prevents these outliers from dominating PCA variance.
#
# Shift formula: ln(x + |min(x)| + ε) ensures all inputs are strictly positive.
# ε = 0.01 for most parameters; ε = 0.0001 for MEANPOT (values near zero).
# MEANSHR and R_VALUE are not transformed — their distributions are
# approximately symmetric (verified in feature_stability.py).
#
# Reference: Baeke et al. (2025), Table C1.
#
# For exploration (Sections 3–7): scaler is fit on the full dataset.
# For RF classification (Section 8): scaler is fit on training folds only.
# =============================================================================

print("\n" + "="*65)
print("Section 3 — Log-transform + Normalization")
print("="*65)

log_params       = ['TOTUSJH', 'TOTPOT', 'USFLUX', 'ABSNJZH', 'SAVNCPP']
log_params_small = ['MEANPOT']

df_log = df[final_params].copy()

for col in log_params:
    col_min = df_log[col].min()
    shift   = abs(col_min) + 0.01
    df_log[col] = np.log(df_log[col] + shift)
    print(f"  {col}: ln(x + {shift:.4f})")

for col in log_params_small:
    col_min = df_log[col].min()
    shift   = abs(col_min) + 0.0001
    df_log[col] = np.log(df_log[col] + shift)
    print(f"  {col}: ln(x + {shift:.6f})  [small ε]")

print("  MEANSHR, R_VALUE: no transform (approximately symmetric)")

scaler_explore = StandardScaler()
X_scaled = scaler_explore.fit_transform(df_log.values)

# X_log: log-transformed but NOT yet scaled — used as input to Section 8,
# where scaling is applied per fold on training data only.
X_log = df_log.values.copy()


# =============================================================================
# Section 4 — PCA
# =============================================================================
# PCA is applied to the standardized 8D feature space to:
#   (a) Quantify how much variance each PC captures (scree plot).
#   (b) Identify which physical parameters dominate each PC (loadings).
#   (c) Provide a reduced-dimensional space for K-Means clustering.
#   (d) Visualize the flare/no-flare separation in 2D (projection only).
#
# The number of PCs for clustering is chosen as the minimum retaining ≥90%
# cumulative variance. The 2D projection for visualization is separate and
# explicitly labeled with the fraction of variance lost.
#
# PCA is unsupervised — flare labels play no role in the decomposition.
# Flare/no-flare separation in PCA space therefore reflects genuine
# physical structure in the magnetic parameters, not a classifier artifact.
# =============================================================================

print("\n" + "="*65)
print("Section 4 — PCA")
print("="*65)

pca_full  = PCA()
pca_full.fit(X_scaled)
explained  = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\nPCA Explained Variance:")
for i, (ind, cum) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {ind*100:.1f}%  |  Cumulative: {cum*100:.1f}%")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(1, len(explained)+1), explained*100, alpha=0.6, color='steelblue', label='Individual')
ax.plot(range(1, len(explained)+1), cumulative*100, 'ro-', linewidth=2, label='Cumulative')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Scree Plot — 8 SHARP Parameters')
ax.set_xticks(range(1, len(explained)+1))
ax.legend()
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/pca_scree.png',
    dpi=150, bbox_inches='tight'
)
print("\nSaved: pca_scree.png")

n_components = int(np.argmax(cumulative >= 0.90)) + 1
var_retained = cumulative[n_components - 1] * 100
var_lost     = 100 - var_retained

print(f"\n→ {n_components} PCs retain {var_retained:.1f}% variance ({var_lost:.1f}% lost)")

pca_n = PCA(n_components=n_components)
X_pca = pca_n.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca_n.components_.T,
    index=final_params,
    columns=[f'PC{i+1}' for i in range(n_components)]
)
print(f"\nPCA Loadings:")
print(loadings.round(3).to_string())

plt.figure(figsize=(10, 5))
sns.heatmap(loadings, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 9})
plt.title(f'PCA Loadings — 8 SHARP Parameters → {n_components} PCs')
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/pca_loadings.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: pca_loadings.png")

# 2D projection for visualization only
pca_2d  = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_scaled)
var_2d  = pca_2d.explained_variance_ratio_
info_2d = sum(var_2d) * 100
loss_2d = 100 - info_2d

print(f"\n2D visualization: PC1={var_2d[0]*100:.1f}%, PC2={var_2d[1]*100:.1f}%")
print(f"  Retained: {info_2d:.1f}%  |  Lost: {loss_2d:.1f}%")

mask_nf = y == 0
mask_f  = y == 1

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(X_pca2d[mask_nf, 0], X_pca2d[mask_nf, 1],
           c='lightblue', alpha=0.2, s=1, label='No flare', rasterized=True)
ax.scatter(X_pca2d[mask_f, 0], X_pca2d[mask_f, 1],
           c='red', alpha=0.8, s=20, label='M/X flare', zorder=5)
ax.set_xlabel(f'PC1 ({var_2d[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({var_2d[1]*100:.1f}% variance)')
ax.set_title(
    f'PCA 2D Projection — colored by flare class (Case 3: M/X vs No-Alert)\n'
    f'Retained: {info_2d:.1f}%  |  Lost: {loss_2d:.1f}%'
)
ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/pca_2d_flares.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: pca_2d_flares.png")


# =============================================================================
# Section 5 — K-Means Clustering
# =============================================================================
# K-Means is applied in the n_components-dimensional PCA space rather than
# the raw 8D space. This is preferable because K-Means relies on Euclidean
# distances, which are distorted in correlated high-dimensional spaces;
# PCA removes inter-feature correlations and reduces noise, producing more
# geometrically meaningful cluster boundaries.
#
# The optimal K is determined via the Kneedle algorithm, which formalizes
# the elbow criterion by finding the point of maximum curvature on the
# normalized inertia curve. This is more reproducible than visual inspection.
#
# The resulting cluster flare rates provide the unsupervised risk hierarchy
# used in the explainability narrative: if K-Means (which has no access to
# flare labels) produces a cluster with elevated flare rate, it confirms that
# the physical signal is captured by the magnetic parameters alone.
# =============================================================================

print("\n" + "="*65)
print("Section 5 — K-Means Clustering")
print("="*65)

inertias = []
K_range  = range(2, 11)

print("\nElbow method — K=2 to 10:")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    print(f"  K={k:>2}  |  Inertia = {km.inertia_:.0f}")

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.title('K-Means Elbow Plot')
plt.xticks(list(K_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/kmeans_elbow.png',
    dpi=150, bbox_inches='tight'
)
print("\nSaved: kmeans_elbow.png")

kneedle = KneeLocator(
    x=list(K_range),
    y=inertias,
    curve='convex',
    direction='decreasing'
)
K_chosen = kneedle.knee
print(f"\n→ Kneedle detected elbow at K={K_chosen}")

km_final       = KMeans(n_clusters=K_chosen, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_pca)

print(f"\nK-Means K={K_chosen} — cluster flare rates:")
for k in range(K_chosen):
    n_k      = (cluster_labels == k).sum()
    flares_k = y[cluster_labels == k].sum()
    rate_k   = flares_k / n_k * 100
    print(f"  Cluster {k}: {n_k:>8,} samples  |  {flares_k:>4} M/X flares  |  rate = {rate_k:.3f}%")

import matplotlib.cm as mplcm
colors_k = [mplcm.tab10(i) for i in range(K_chosen)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for k in range(K_chosen):
    mask_k = cluster_labels == k
    axes[0].scatter(X_pca2d[mask_k, 0], X_pca2d[mask_k, 1],
                    c=colors_k[k], alpha=0.2, s=2,
                    label=f'Cluster {k}  (n={mask_k.sum():,})', rasterized=True)
axes[0].set_xlabel(f'PC1 ({var_2d[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({var_2d[1]*100:.1f}%)')
axes[0].set_title(f'K-Means Clusters (K={K_chosen})\n'
                  f'[Clustered in {n_components}D — visualized in 2D, {loss_2d:.1f}% info lost]')
axes[0].legend(markerscale=5, fontsize=8)

axes[1].scatter(X_pca2d[mask_nf, 0], X_pca2d[mask_nf, 1],
                c='lightblue', alpha=0.15, s=1, label='No flare', rasterized=True)
axes[1].scatter(X_pca2d[mask_f, 0], X_pca2d[mask_f, 1],
                c='red', alpha=0.8, s=20, label='M/X flare', zorder=5)
axes[1].set_xlabel(f'PC1 ({var_2d[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({var_2d[1]*100:.1f}%)')
axes[1].set_title('Actual Flare Labels (Case 3: M/X)')
axes[1].legend(markerscale=4)

plt.suptitle(
    'K-Means Clusters vs Actual Flare Labels\n'
    '(High-flare-rate cluster overlapping red region = physical signal validated)',
    fontsize=13
)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/kmeans_vs_flares.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: kmeans_vs_flares.png")


# =============================================================================
# Section 6 — t-SNE Visualization
# =============================================================================
# t-SNE (van der Maaten & Hinton 2008) provides a non-linear 2D projection
# that preserves local neighborhood structure. It complements PCA by
# revealing whether flaring ARs form a genuine local neighborhood in 8D
# SHARP space, which is invisible in the global linear PCA projection.
#
# Important limitations (must be stated when presenting results):
#   - Distances between clusters are not interpretable.
#   - Absolute positions are random (initialization-dependent).
#   - Only local cluster membership is meaningful.
#   - t-SNE is used for visualization only; no quantitative results derive from it.
#
# Sampling: t-SNE scales as O(n²). We use 5000 randomly sampled no-flare
# points + all flare points. This intentional overrepresentation of flares
# ensures they are visible. Sample size is reported explicitly in the figure.
# =============================================================================

print("\n" + "="*65)
print("Section 6 — t-SNE Visualization")
print("="*65)

np.random.seed(42)
noflare_idx = np.where(y == 0)[0]
flare_idx   = np.where(y == 1)[0]
sample_nf   = np.random.choice(noflare_idx, size=min(5000, len(noflare_idx)), replace=False)
tsne_idx    = np.unique(np.concatenate([sample_nf, flare_idx]))

X_tsne_input = X_scaled[tsne_idx]
y_tsne       = y[tsne_idx]
cluster_tsne = cluster_labels[tsne_idx]

print(f"\nt-SNE input: {len(tsne_idx):,} samples")
print(f"  No-flare (sampled): {(y_tsne==0).sum():,}")
print(f"  M/X flare (all):    {(y_tsne==1).sum():,}")
print("Running t-SNE...")

tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_tsne_input)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

nf_t = y_tsne == 0
f_t  = y_tsne == 1
axes[0].scatter(X_tsne[nf_t, 0], X_tsne[nf_t, 1],
                c='lightblue', alpha=0.3, s=4, label='No flare')
axes[0].scatter(X_tsne[f_t, 0], X_tsne[f_t, 1],
                c='red', alpha=0.8, s=30, label='M/X flare', zorder=5)
axes[0].set_title('t-SNE — colored by flare label (Case 3)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
axes[0].legend(markerscale=3)

for k in range(K_chosen):
    mk = cluster_tsne == k
    axes[1].scatter(X_tsne[mk, 0], X_tsne[mk, 1],
                    c=colors_k[k], alpha=0.3, s=4, label=f'Cluster {k}')
axes[1].set_title('t-SNE — colored by K-Means cluster')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].legend(markerscale=3)

plt.suptitle(
    't-SNE Visualization (local structure only — inter-cluster distances not meaningful)\n'
    f'Sample: {len(tsne_idx):,} points ({(y_tsne==1).sum()} M/X flares)',
    fontsize=12
)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/tsne.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: tsne.png")



# =============================================================================
# Section 7 — Random Forest: Leave-One-Year-Out Cross-Validation × 2 Cases
# =============================================================================
# Each calendar year (2010–2019) serves once as the test set while all other
# years are used for training. This evaluates inter-year generalization and
# avoids temporal mixing between training and test data.
#
# AR-aware splitting: when a test year is selected, any AR (HARPNUM) whose
# observations appear in that year is excluded from the training set entirely.
# This prevents leakage from ARs whose multi-year activity spans both sets.
#
# Anti-leakage pipeline order (applied per fold):
#   1. AR-aware year split
#   2. StandardScaler fit on training fold only
#   3. RandomUnderSampler on training fold only (majority class capped at 5000)
#   4. SMOTE on training fold only (minority class oversampled to match cap)
#   5. RF trained on balanced training fold
#   6. Evaluated on real, unbalanced test year
#
# TSS = TPR − FPR (True Skill Statistic; Hanssen & Kuipers 1965)
# Years where the test set contains zero positive samples produce undefined
# TSS (denominator = 0) and are excluded from the mean via np.nanmean.
# This is a known solar-minimum effect (2018–2019 for Case 3), not a bug.
# =============================================================================

print("\n" + "="*65)
print("Section 8 — Random Forest: Leave-One-Year-Out CV")
print("="*65)

n_keep = 5000


def compute_metrics(y_true, y_pred):
    """
    Compute confusion matrix, TSS, HSS, and FAR.

    Returns np.nan for any metric that is mathematically undefined
    (e.g. TSS when the test set contains only one class).
    labels=[0, 1] ensures a 2×2 matrix even for single-class test years.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    tss = tpr - fpr if not (np.isnan(tpr) or np.isnan(fpr)) else np.nan

    denom_hss = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss = 2 * (TP * TN - FP * FN) / denom_hss if denom_hss > 0 else np.nan

    far = FP / (TP + FP) if (TP + FP) > 0 else np.nan

    return cm, tss, hss, far


years = sorted(df['year'].unique())
print("\nYears:", years)

cases = {
    'Case2_FlareVsNoFlare': {
        'y': df['flare_cmx'].values,
        'pos_label': 'Flare (C+M+X)',
        'neg_label': 'No-Flare',
    },
    'Case3_AlertVsNoAlert': {
        'y': df['flare_mx'].values,
        'pos_label': 'Alert (M+X)',
        'neg_label': 'No-Alert',
    },
}

results_summary  = {}
case_importances = {}

for case_name, case_cfg in cases.items():

    y_case    = case_cfg['y']
    pos_label = case_cfg['pos_label']
    neg_label = case_cfg['neg_label']

    print(f"\n{'─'*65}")
    print(f"  {case_name}")
    print(f"  Positive ({pos_label}): {y_case.sum():,}")
    print(f"  Negative ({neg_label}): {(y_case == 0).sum():,}")
    print(f"  Positive rate: {y_case.mean()*100:.4f}%")
    print(f"{'─'*65}")

    all_tss         = []
    all_hss         = []
    all_far         = []
    all_cm          = []
    all_importances = []
    split_names     = []

    for test_year in years:

        test_mask = df['year'] == test_year
        # Exclude any AR that has observations in the test year from training,
        # regardless of which year those training observations belong to.
        test_ar    = df.loc[test_mask, group_col].unique()
        train_mask = (df['year'] != test_year) & (~df[group_col].isin(test_ar))

        X_tr = X_log[train_mask]
        X_te = X_log[test_mask]
        y_tr = y_case[train_mask]
        y_te = y_case[test_mask]

        if len(np.unique(y_tr)) < 2:
            print(f"\nSkipping test year {test_year}: training set has only one class.")
            continue
        if len(np.unique(y_te)) < 2:
            print(f"\nTest year {test_year}: only one class in test set — some metrics undefined.")

        train_years = sorted(df.loc[train_mask, 'year'].unique())
        print(f"\nTest year: {test_year} | Train: {min(train_years)}–{max(train_years)} excl. {test_year}")
        print(f"  Train: {len(y_tr):,} rows, {int(y_tr.sum()):,} positives")
        print(f"  Test:  {len(y_te):,} rows, {int(y_te.sum()):,} positives")

        # Scale: fit on training fold only
        scaler_r = StandardScaler()
        X_tr_sc  = scaler_r.fit_transform(X_tr)
        X_te_sc  = scaler_r.transform(X_te)

        # Undersample majority class
        n_pos_tr = int(y_tr.sum())
        n_neg_tr = int(min(n_keep, (y_tr == 0).sum()))
        under_r  = RandomUnderSampler(
            sampling_strategy={0: n_neg_tr, 1: n_pos_tr}, random_state=42)
        X_u, y_u = under_r.fit_resample(X_tr_sc, y_tr)

        # SMOTE minority class up to match majority cap
        n_pos_after = int(y_u.sum())
        n_neg_after = int((y_u == 0).sum())
        if n_pos_after < n_neg_after:
            smote_r = SMOTE(
                sampling_strategy={1: n_neg_after},
                random_state=42,
                k_neighbors=min(5, n_pos_after - 1)
            )
            X_bal, y_bal = smote_r.fit_resample(X_u, y_u)
        else:
            X_bal, y_bal = X_u, y_u

        print(f"  Balanced train: {(y_bal==1).sum():,} pos / {(y_bal==0).sum():,} neg")

        rf_r = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r.fit(X_bal, y_bal)

        y_pred_r = rf_r.predict(X_te_sc)
        cm_r, tss_r, hss_r, far_r = compute_metrics(y_te, y_pred_r)
        TN, FP, FN, TP = cm_r.ravel()

        all_tss.append(tss_r)
        all_hss.append(hss_r)
        all_far.append(far_r)
        all_cm.append(cm_r)
        all_importances.append(rf_r.feature_importances_)
        split_names.append(str(test_year))

        print(f"  TSS={tss_r:.3f}  HSS={hss_r:.3f}  FAR={far_r:.3f}  "
              f"| TP={TP} FP={FP} FN={FN} TN={TN}")

    # Aggregate metrics
    tss_arr = np.array(all_tss, dtype=float)
    hss_arr = np.array(all_hss, dtype=float)
    far_arr = np.array(all_far, dtype=float)
    imp_arr = np.array(all_importances)

    tss_mean = np.nanmean(tss_arr)
    tss_std  = np.nanstd(tss_arr)
    hss_mean = np.nanmean(hss_arr)
    hss_std  = np.nanstd(hss_arr)
    far_mean = np.nanmean(far_arr)
    far_std  = np.nanstd(far_arr)

    valid_years    = [split_names[i] for i in range(len(tss_arr)) if not np.isnan(tss_arr[i])]
    excluded_years = [split_names[i] for i in range(len(tss_arr)) if np.isnan(tss_arr[i])]

    print(f"\n  Summary — {case_name}")
    print(f"  Years in TSS mean: {', '.join(valid_years)}")
    if excluded_years:
        print(f"  Excluded (TSS undefined — no positives in test year): {', '.join(excluded_years)}")
    print(f"  TSS: {tss_mean:.3f} ± {tss_std:.3f}")
    print(f"  HSS: {hss_mean:.3f} ± {hss_std:.3f}")
    print(f"  FAR: {far_mean:.3f} ± {far_std:.3f}")

    results_summary[case_name] = {
        'TSS': f"{tss_mean:.3f} ± {tss_std:.3f}",
        'HSS': f"{hss_mean:.3f} ± {hss_std:.3f}",
        'FAR': f"{far_mean:.3f} ± {far_std:.3f}",
        'tss_arr': tss_arr, 'hss_arr': hss_arr, 'far_arr': far_arr,
        'split_names': split_names,
        'valid_tss_years': valid_years,
        'excluded_tss_years': excluded_years,
    }

    # Representative confusion matrix: test year with TSS closest to nanmean
    finite_idx   = np.where(~np.isnan(tss_arr))[0]
    rep_idx      = int(finite_idx[np.argmin(np.abs(tss_arr[finite_idx] - tss_mean))]) \
                   if len(finite_idx) > 0 else 0
    cm_rep       = all_cm[rep_idx]
    rep_year     = split_names[rep_idx]
    TN, FP, FN, TP = cm_rep.ravel()

    print(f"\n  Representative CM: test year {rep_year} (TSS={tss_arr[rep_idx]:.3f} ≈ mean {tss_mean:.3f})")
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rep, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Pred: {neg_label}', f'Pred: {pos_label}'],
                yticklabels=[f'True: {neg_label}', f'True: {pos_label}'])
    plt.title(f'Confusion Matrix — {case_name}\n'
              f'representative test year: {rep_year}\n'
              f'TSS = {tss_arr[rep_idx]:.3f} | Mean = {tss_mean:.3f} ± {tss_std:.3f}')
    plt.tight_layout()
    plt.savefig(
        f'/STER/agungp/space-weather/output-full-pipeline-cg/rf_cm_{case_name}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"  Saved: rf_cm_{case_name}.png")

    # Feature importance averaged over LOYO folds
    imp_mean = imp_arr.mean(axis=0)
    imp_std  = imp_arr.std(axis=0)
    imp_df   = pd.DataFrame({
        'parameter': final_params,
        'mean': imp_mean,
        'std': imp_std,
    }).sort_values('mean', ascending=False)

    case_importances[case_name] = imp_df

    print(f"\n  Feature importance (mean ± std over LOYO folds):")
    for _, row in imp_df.iterrows():
        print(f"    {row['parameter']:<12} {row['mean']:.4f} ± {row['std']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    imp_sorted = imp_df.sort_values('mean', ascending=True)
    ax.barh(imp_sorted['parameter'], imp_sorted['mean'], xerr=imp_sorted['std'],
            color='steelblue', alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 4, 'capthick': 1.2})
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'RF Feature Importance — {case_name}\nTSS = {tss_mean:.3f} ± {tss_std:.3f}')
    plt.tight_layout()
    plt.savefig(
        f'/STER/agungp/space-weather/output-full-pipeline-cg/rf_importance_{case_name}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"  Saved: rf_importance_{case_name}.png")


# =============================================================================
# Section 8 — Feature Importance: Case 2 vs Case 3 Comparison
# =============================================================================
# Compares RF feature importance rankings across the two case definitions.
# Consistency between cases indicates the ranking reflects genuine physical
# signal rather than an artifact of how the positive class is defined.
# This convergence — RF importance (here), PCA loadings (Section 4), and
# K-Means cluster profiles (Section 5) agreeing on the same top parameters —
# forms the core explainability argument of the project.
# =============================================================================

print("\n" + "="*65)
print("Section 9 — Feature Importance: Case 2 vs Case 3")
print("="*65)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, (case_name, imp_df) in zip(axes, case_importances.items()):
    imp_sorted = imp_df.sort_values('mean', ascending=True)
    ax.barh(imp_sorted['parameter'], imp_sorted['mean'], xerr=imp_sorted['std'],
            color='steelblue', alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 4})
    tss_str = results_summary[case_name]['TSS']
    ax.set_title(f'{case_name}\nTSS = {tss_str}')
    ax.set_xlabel('Feature Importance')

plt.suptitle('RF Feature Importance — Case 2 vs Case 3\n', fontsize=12)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/rf_importance_comparison.png',
    dpi=150, bbox_inches='tight'
)
print("Saved: rf_importance_comparison.png")


# =============================================================================
# Section 9 — Summary
# =============================================================================

print("\n" + "="*65)
print("PIPELINE COMPLETE — SUMMARY")
print("="*65)

print(f"\n  Dataset:         {len(y):,} samples, {df['year'].min()}–{df['year'].max()}")
print(f"  Parameters:      {len(final_params)} final SHARP parameters")
print(f"  PCA components:  {n_components} ({var_retained:.1f}% variance retained)")
print(f"  K-Means K:       {K_chosen} (Kneedle)")
print(f"  Evaluation:      Leave-One-Year-Out CV with AR-aware splitting")

print(f"\n  {'Case':<30} {'TSS':>15} {'HSS':>15} {'FAR':>12}")
print(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*12}")
for case_name, metrics in results_summary.items():
    print(f"  {case_name:<30} {metrics['TSS']:>15} {metrics['HSS']:>15} {metrics['FAR']:>12}")


# =============================================================================
# Per-year TSS Bar Plot
# =============================================================================
# Visualizes TSS across LOYO folds for both cases to identify years where
# model performance degrades. Low-solar-activity years (late solar cycle 24,
# 2018–2019) show degraded or undefined TSS for Case 3 due to near-zero
# M/X flare occurrence — a known limitation of the dataset, not the model.
# =============================================================================

years_case2 = [int(y) for y in results_summary['Case2_FlareVsNoFlare']['split_names']]
tss_case2   = results_summary['Case2_FlareVsNoFlare']['tss_arr']

years_case3 = [int(y) for y in results_summary['Case3_AlertVsNoAlert']['split_names']]
tss_case3   = results_summary['Case3_AlertVsNoAlert']['tss_arr']

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

axes[0].bar(years_case2, tss_case2, color='steelblue', alpha=0.85,
            edgecolor='black', linewidth=0.5)
axes[0].axhline(y=np.mean(tss_case2), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean TSS = {np.mean(tss_case2):.3f}')
axes[0].set_xlabel('Test Year')
axes[0].set_ylabel('TSS')
axes[0].set_title('Case 2 — Flare vs No-Flare')
axes[0].set_xticks(years_case2)
axes[0].set_xticklabels(years_case2, rotation=45)
axes[0].set_ylim(0, 1.0)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].text(0.98, 0.08, '2019: excluded (no positives)',
             transform=axes[0].transAxes, ha='right', fontsize=8, color='grey')

axes[1].bar(years_case3, tss_case3, color='steelblue', alpha=0.85,
            edgecolor='black', linewidth=0.5)
axes[1].axhline(y=np.mean(tss_case3), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean TSS = {np.mean(tss_case3):.3f}')
axes[1].set_xlabel('Test Year')
axes[1].set_ylabel('TSS')
axes[1].set_title('Case 3 — Alert vs No-Alert')
axes[1].set_xticks(years_case3)
axes[1].set_xticklabels(years_case3, rotation=45)
axes[1].set_ylim(0, 1.0)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].text(0.98, 0.08, '2018–2019: excluded (no positives)',
             transform=axes[1].transAxes, ha='right', fontsize=8, color='grey')

plt.suptitle(
    'TSS per Test Year — Leave-One-Year-Out Cross-Validation\n'
    'Red dashed line = mean TSS across valid years',
    fontsize=12
)
plt.tight_layout()
plt.savefig(
    '/STER/agungp/space-weather/output-full-pipeline-cg/tss_per_year.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: tss_per_year.png")
print("\nDone.")
