# Explainable Solar Flare Forecasting with SHARP Parameters

Group 1 project for the Space Weather course at KU Leuven (2025–2026)
Supervisors: Panagiotis Gonidakis

Pipeline
Stage 1: Feature Selection (feature_stability.py)
Runs Random Forest on all 24 SHARP parameters across k=5 and k=10 StratifiedGroupKFold CV to select the 8 most stable and important parameters: R_VALUE, TOTUSJH, USFLUX, TOTPOT, SAVNCPP, MEANSHR, ABSNJZH, MEANPOT.

Stage 2: Full Analysis (pipeline_stage2.py)
Using the final 8 parameters: correlation matrix, PCA, K-Means clustering, t-SNE, and Random Forest classification evaluated with Leave-One-Year-Out cross-validation across two binary classification cases (Case 2: Flare vs No-Flare, Case 3: Alert vs No-Alert).
Notes

All scripts were run on the Pleiades HPC cluster via SLURM.
The dataset (AllData_sharp.csv) is not included due to file size.
