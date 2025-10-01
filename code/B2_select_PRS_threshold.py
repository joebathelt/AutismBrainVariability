# %%
# ==============================================================================
# Title: B2_select_PRS_threshold.py
# ==============================================================================
# Description: This script selects polygenic risk score (PRS) thresholds based on
# their association with social difficulty scores. It includes steps such as
# loading data, defining thresholds, and testing their effects.
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, zscore
import seaborn as sns
import statsmodels.formula.api as smf
# %%
project_folder = Path(__file__).resolve().parents[1]
prs_df = pd.read_csv(project_folder / 'data/plink/Neuro_Chip_unrelated_prsice.all_score', sep=' ')
social_df = pd.read_csv(project_folder / 'data/cfa_factor_scores_full_sample.csv')
merged_df = pd.merge(prs_df, social_df, left_on='IID', right_on='Subject')

# Load the PCA data
pca_df = pd.read_csv(project_folder / 'data/plink/Neuro_Chip_full_sample_pca.eigenvec', sep=' ', header=None)
pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
pca_df = pca_df[['IID'] + [f'PC{i}' for i in range(1, 11)]]
merged_df = pd.merge(merged_df, pca_df, on='IID')

prs_thresholds = [column for column in merged_df if column.startswith('Pt')]

print("Starting 5-fold cross-validation...")
# Select the best PRS threshold based on 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_threshold = None
best_avg_corr = -np.inf
results = {}

for prs in prs_thresholds:
    print(f"Evaluating PRS threshold: {prs}", end="\r")
    correlations = []

    for train_idx, test_idx in kf.split(merged_df):
        # Regress the PCs out of the PRS scores
        merged_df['thresholded_PRS'] = merged_df[prs]
        smf.ols('thresholded_PRS ~ ' + ' + '.join([f'PC{i}' for i in range(1, 11)]), data=merged_df.iloc[train_idx]).fit()
        merged_df.loc[:, 'thresholded_PRS'] = smf.ols('thresholded_PRS ~ ' + ' + '.join([f'PC{i}' for i in range(1, 11)]), data=merged_df).fit().predict(merged_df.iloc[test_idx])

        train_data, test_data = merged_df.iloc[train_idx], merged_df.iloc[test_idx]

        # Compute Pearson correlation in validation set
        corr, _ = pearsonr(test_data['thresholded_PRS'], test_data['Social_Score'])
        correlations.append(corr)

    # Average correlation across folds
    avg_corr = np.mean(correlations)
    results[prs] = avg_corr

    # Track the best threshold
    if avg_corr > best_avg_corr:
        best_avg_corr = avg_corr
        best_threshold = prs

# Output results
print("Best PRS threshold:", best_threshold)
print("Average correlations for this threshold:", results[best_threshold])

results_df = pd.DataFrame(results, index=[0]).T
results_df = results_df.reset_index()
results_df.columns = ['PRS_Threshold', 'Average_Correlation']

stats_df = merged_df[['IID', 'FID', 'Social_Score', best_threshold]].copy()
stats_df.columns = ['IID', 'FID', 'Social_Score', 'PRS']
stats_df['PRS'] = zscore(stats_df['PRS'])
sns.regplot(data=stats_df, x='PRS', y='Social_Score')
plt.savefig(project_folder / 'results/unrelated_PRS_vs_Social_Score.png')
plt.clf()

print(smf.ols('Social_Score ~ PRS', data=stats_df).fit().summary())

out_file = project_folder / 'data/plink/unrelated_prs_scores.txt'
stats_df[['FID', 'IID', 'PRS']].to_csv(out_file, index=False, header=False, sep=' ')

print("PRS scores written to", out_file)
print("Done!")