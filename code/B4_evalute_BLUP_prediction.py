# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, zscore
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from pingouin import compute_effsize_from_t

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701
# %%
project_folder = Path(__file__).resolve().parents[1]
plink_folder = project_folder / 'data/plink/'

prs_unrelated_sample = pd.read_csv(plink_folder / 'unrelated_prs_scores.txt', sep=' ', header=None)
prs_unrelated_sample.columns = ['FID', 'IID', 'original_PRS']

prs_full_sample = pd.read_csv(plink_folder / 'full_prs_scores.snp.blp.profile', sep='\s+')
prs_full_sample = prs_full_sample[['IID', 'SCORESUM']]
prs_full_sample.columns = ['IID', 'blup_PRS']

# Load PCA data - using only first 5 PCs
pca_df = pd.read_csv(project_folder / 'data/plink/full_prs_scores.snp.blp.pca.eigenvec', sep=' ', header=None)
pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
pca_df = pca_df[['IID'] + [f'PC{i}' for i in range(1, 6)]]  # Only use first 5 PCs

# Merge the dataframes
merged_df = pd.merge(prs_unrelated_sample, prs_full_sample, on='IID')
merged_df = pd.merge(merged_df, pca_df, on='IID')

# Add the behavioural data
phenotypic_df = pd.read_csv(project_folder / "data/phenotypic_data_anonymised.csv")
behavioural_df = pd.read_csv(project_folder / "data/behavioural_data_anonymised.csv")
behavioural_df = pd.merge(behavioural_df, phenotypic_df, on='Subject')

merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender']], left_on='IID', right_on='Subject')
merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')

# Add the social scores
social_df = pd.read_csv(project_folder / 'data/cfa_factor_scores_full_sample.csv')
merged_df = pd.merge(social_df, merged_df, on='Subject')

# Add the unrelated sample information
merged_df.loc[:, 'unrelated'] = merged_df['IID'].isin(prs_unrelated_sample['IID'])
merged_df.replace({'unrelated': {True: 'Unrelated', False: 'Related'}}, inplace=True)

# Ensure that the data types are correct
merged_df['blup_PRS'] = pd.to_numeric(merged_df['blup_PRS'], errors='coerce')
merged_df['original_PRS'] = pd.to_numeric(merged_df['original_PRS'], errors='coerce')
merged_df['Social_Score'] = pd.to_numeric(merged_df['Social_Score'], errors='coerce')
merged_df['Age_in_Yrs'] = pd.to_numeric(merged_df['Age_in_Yrs'], errors='coerce')
# %%
# First, calculate simple correlation between original and BLUP PRS
r, p = pearsonr(merged_df['original_PRS'], merged_df['blup_PRS'])
print(f'Pearson correlation between original and BLUP PRS: {r:.2f}, p={p:.2g}')

# Standardize PRS values for easier interpretation
merged_df['original_PRS_z'] = zscore(merged_df['original_PRS'])
merged_df['blup_PRS_z'] = zscore(merged_df['blup_PRS'])

# Analyze original PRS with proper covariate adjustment (5 PCs)
print("Model 1: Original PRS in unrelated individuals")
model1 = smf.ols('Social_Score ~ original_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
print(model1.summary().tables[1])  # Print just the coefficients table

# Analyze BLUP PRS in the full sample (5 PCs)
print("\nModel 2: BLUP PRS in full sample")
model2 = smf.ols('Social_Score ~ blup_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
print(model2.summary().tables[1])  # Print just the coefficients table

# Testing if relatedness moderates the effect (5 PCs)
print("\nModel 3: Testing if relatedness moderates PRS effect")
model3 = smf.ols('Social_Score ~ blup_PRS_z * C(unrelated) + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
print(model3.summary().tables[1])  # Print just the coefficients table

r, p = pearsonr(merged_df['original_PRS'], merged_df['blup_PRS'])
print(f'Pearson correlation between original and blup PRS: {r:.2f}, p={p:.2f}')

# %%
# Plotting the PRS score correlation
fig, ax = plt.subplots(figsize=(40*mm2inches, 40*mm2inches))
sns.regplot(
    x='original_PRS_z',
    y='blup_PRS_z',
    data=merged_df,
    ax=ax,
    scatter_kws={'color': '#E64B35', 's': 5, 'alpha': 0.8, 'linewidths': 0.1},
    line_kws={'color': 'k', 'linewidth': 1}
)
sns.despine(offset=8)
plt.xlabel('Original PGS')
plt.ylabel('BLUP PGS')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.gca().set_yticks([-3, 0, 3])
plt.gca().set_xticks([-3, 0, 3])
plt.text(0.05, 0.95, f'$r$={r:.2f}, \n $p$={p:.2g}',
         horizontalalignment='left', verticalalignment='bottom',
         transform=ax.transAxes, fontsize=8)
plt.tight_layout()
plt.savefig(project_folder / 'figures/PGS_BLUP_correlation.png', dpi=300)
plt.show()

# %%
# Comparing the distribution of the PRS scores
fig, ax = plt.subplots(2, 1, figsize=(40*mm2inches, 45*mm2inches), sharex=True, sharey=True)
sns.histplot(merged_df['original_PRS_z'],
            alpha=0.9,
            bins=50,
            color='lightgrey',
            ax=ax[0])
sns.histplot(merged_df['blup_PRS_z'],
            alpha=0.9,
            bins=50,
            color='lightgrey',
            ax=ax[1])
plt.xlim([-3, 3])
plt.ylim([0, 25])
ax[0].set_title('Original PGS', fontsize=8)
ax[1].set_title('BLUP PGS', fontsize=8)
ax[1].set_xlabel('PGS [z-score]')
ax[1].set_ylabel('Count')
ax[0].set_ylabel('')
sns.despine(offset=8)
plt.tight_layout()
plt.savefig(project_folder / 'figures/PRS_comparison.png', dpi=300)
plt.show()

# %%
# Run adjusted model with 5 PCs
adjusted_model = smf.ols('Social_Score ~ blup_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
print(adjusted_model.summary())

# Get residualized variables for plotting (with 5 PCs)
residual_model_prs = smf.ols('blup_PRS_z ~ C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
merged_df['blup_PRS_residuals'] = residual_model_prs.resid

residual_model_social = smf.ols('Social_Score ~ C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()
merged_df['Social_Score_residuals'] = residual_model_social.resid
plotting_df = merged_df.copy()

beta = adjusted_model.params["blup_PRS_z"]
pval = adjusted_model.pvalues["blup_PRS_z"]
# %%
# Extend to the full sample
prs_full_sample = pd.read_csv(plink_folder / 'full_prs_scores.snp.blp.profile', sep='\s+')
prs_full_sample = prs_full_sample[['IID', 'SCORESUM']]
prs_full_sample.columns = ['IID', 'blup_PRS']

# Load PCA data - using only first 5 PCs
pca_df = pd.read_csv(project_folder / 'data/plink/full_prs_scores.snp.blp.pca.eigenvec', sep=' ', header=None)
pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
pca_df = pca_df[['IID'] + [f'PC{i}' for i in range(1, 6)]]  # Only use first 5 PCs
merged_df = pd.merge(prs_full_sample, pca_df, on='IID')

# Add the behavioural data
phenotypic_df = pd.read_csv(project_folder / "data/phenotypic_data_anonymised.csv")
behavioural_df = pd.read_csv(project_folder / "data/behavioural_data_anonymised.csv")
behavioural_df = pd.merge(behavioural_df, phenotypic_df, on='Subject')

merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender']], left_on='IID', right_on='Subject')
merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')

# Regress the first 5 PCs out of the PRS scores
residual_model_prs_full = smf.ols('blup_PRS ~ C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5', data=merged_df).fit()

merged_df.loc[:, 'blup_PRS_residuals'] = residual_model_prs_full.resid

# Save the PGS to CSV
merged_df = merged_df[['IID', 'blup_PRS', 'blup_PRS_residuals']]
merged_df.rename(columns={'IID': 'Subject'}, inplace=True)
merged_df.to_csv(project_folder / 'data/prs_residuals.csv', index=False)
# %%
merged_df['blup_PRS_residuals'] = zscore(merged_df['blup_PRS_residuals'])

fig, ax = plt.subplots(figsize=(40*mm2inches, 40*mm2inches))

# Add an annotation with the number of subjects in each category
n_middle = ((merged_df['blup_PRS_residuals'] > -0.5) & (merged_df['blup_PRS_residuals'] < 0.5)).sum()
n_low = (merged_df['blup_PRS_residuals'] < -1).sum()
n_high = (merged_df['blup_PRS_residuals'] > 1).sum()

# Add a shaded areas per group
ax.axvspan(1, 3, color='#8491B499', edgecolor='none', linewidth=0, alpha=0.5, label=f'high n={n_high}', zorder=0)
ax.axvspan(-0.5, 0.5, color='#91D1C299', edgecolor='none', linewidth=0, alpha=0.5, label=f'middle n={n_middle}', zorder=0)
ax.axvspan(-3, -1, color='#4DBBD599', edgecolor='none', linewidth=0, alpha=0.5, label=f'low n={n_low}', zorder=0)

# Show the distribution of the residualized PRS scores in the full sample
counts, bins = np.histogram(merged_df['blup_PRS_residuals'], bins=50, range=(-3, 3))
bin_width = bins[1] - bins[0]
ax.bar(bins[:-1], counts, width=bin_width, color='lightgrey', edgecolor='black', linewidth=0.3, zorder=3)
plt.xlim([-3, 3])
ax.set_xlabel('PGS [$z$]')
ax.set_ylabel('Count')
sns.despine(offset=8)

# Add the legend
#legend = ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=8)
#legend.get_frame().set_facecolor('white')     # Set background colour
#legend.get_frame().set_alpha(1)               # Fully opaque
#legend.get_frame().set_edgecolor('black')
#legend.get_frame().set_linewidth(0.5) 
plt.tight_layout()

plt.savefig(project_folder / 'figures/BLUP_PRS_residuals_full_sample.png', dpi=300)
# %%
plotting_df = merged_df.copy()
plotting_df['pgs_group'] = np.select(
    [
        plotting_df['blup_PRS_residuals'] <= -1,
        (plotting_df['blup_PRS_residuals'] >= -0.5) & (plotting_df['blup_PRS_residuals'] <= 0.5),
        plotting_df['blup_PRS_residuals'] >= 1
    ],
    ['low', 'middle', 'high'],
    default='other'
)
social_df = pd.read_csv(project_folder / 'data/cfa_factor_scores_full_sample.csv')
merged_df = pd.merge(plotting_df, social_df, on='Subject')

# %%
# Create the plot with residualized variables (for a cleaner visualization)
palette = {'middle': '#91D1C299', 'high':'#8491B499', 'low': '#4DBBD599', 'other': 'lightgrey'}
fig, ax = plt.subplots(figsize=(40*mm2inches, 40*mm2inches))
sns.scatterplot(x='blup_PRS_residuals', y='Social_Score', data=merged_df,
                hue='pgs_group', palette=palette, s=5, edgecolor=None, linewidth=0.1, alpha=0.8, ax=ax)
sns.regplot(x='blup_PRS_residuals', y='Social_Score',
            color='k', scatter=False, line_kws={'linewidth': 1}, data=merged_df, ax=ax)

plt.xlabel('PGS [$z$]')
plt.ylabel('SDS [$z$]')
plt.legend([], frameon=False)
sns.despine(offset=8)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.gca().set_yticks([-3, 0, 3])
plt.gca().set_xticks([-3, 0, 3])
# Add the beta and p-value from the full adjusted model
plt.text(0.0, 0.90, f'$\\beta$={beta:.2f},\n$p$={pval:.3f}',
         horizontalalignment='left', verticalalignment='bottom',
         transform=ax.transAxes, fontsize=8)

plt.tight_layout()
plt.savefig(project_folder / 'figures/PRS_SocialScore_adjusted_association.png', dpi=300)

# %%
# Compare the groups
groups = []
for comparison in [1, 1.5, -1, -1.5]:

    if comparison > 0:
        group1 = merged_df[merged_df['blup_PRS_residuals'] >= comparison].copy()
    elif comparison < 0:
        group1 = merged_df[merged_df['blup_PRS_residuals'] <= comparison].copy()
    group2 = merged_df[(merged_df['blup_PRS_residuals'] >= -0.5) & (merged_df['blup_PRS_residuals'] <= 0.5)].copy()
    if comparison == 1:
        groups.append(group2)
    groups.append(group1)

    group1 = group1['Social_Score']
    group2 = group2['Social_Score']

    # Remove NaNs before calculations
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    t_stat, p_val = ttest_ind(group1_clean, group2_clean)

    n1 = len(group1_clean)
    n2 = len(group2_clean)
    df = n1 + n2 - 2

    mean1 = np.mean(group1_clean)
    mean2 = np.mean(group2_clean)
    sd1 = np.std(group1_clean, ddof=1)
    sd2 = np.std(group2_clean, ddof=1)

    d = compute_effsize_from_t(t_stat, n1, n2, eftype='cohen')
    corrected_p_val = min(p_val * 4, 1.0)

    print(f"Comparison >{comparison}SD (n={n1}, mean={mean1:.3f}, sd={sd1:.3f}) vs middle (n={n2}, mean={mean2:.3f}, sd={sd2:.3f}): t({df})={t_stat:.3f}, p={p_val:.3f}, corrected p={corrected_p_val:.3f}, d={d:.3f}")
# %%
plotting_df = pd.concat([groups[0], groups[1], groups[3]], axis=0)
plotting_df['group'] = np.hstack([
    np.repeat('low', len(groups[3])),
    np.repeat('middle', len(groups[0])),
    np.repeat('high', len(groups[1])),
])

# %%
fig, ax = plt.subplots(figsize=(50*mm2inches, 38*mm2inches))
palette = {'middle': '#91D1C299', 'high':'#8491B499', 'low': '#4DBBD599'}

sns.swarmplot(x='group', y='Social_Score', alpha=0.9, data=plotting_df,
              s=1.5, edgecolor=None, linewidth=0.1, ax=ax, zorder=-1, palette=palette)
sns.boxplot(x='group', y='Social_Score', data=plotting_df, showcaps=False,
            width=0.5,
            boxprops={'facecolor':'None'}, showfliers=False,
            whiskerprops={'linewidth':1})
sns.despine(offset=8, trim=True)
plt.xlabel('PGS group')
plt.ylabel('SDS [$z$]')
plt.tight_layout()
plt.savefig(project_folder / 'figures/PRS_SocialScore_group_comparison.png', dpi=300)
# %%
