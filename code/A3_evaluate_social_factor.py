# ==============================================================================
# Title: A3_evaluate_social_factor.py
# ==============================================================================
# Description: This script evaluates the social factors affecting cognitive
# performance in the HCP dataset. It includes steps such as correlation analysis,
# regression modeling, and visualization of results.
# ==============================================================================

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import zscore, pearsonr

project_folder = Path(__file__).resolve().parents[1]
factor_df = pd.read_csv(Path(project_folder) / 'data' / 'cfa_factor_scores_full_sample.csv')
behaviour_df = pd.read_csv(Path(project_folder) / 'data' / 'behavioural_data_anonymised.csv')
merged_df = pd.merge(factor_df, behaviour_df, on='Subject', how='left')

# %%
# Compute correlations between social factors and cognitive performance
correlations = {}
for var in ['CogFluidComp_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj', 'CardSort_AgeAdj']:
    temp_df = merged_df.dropna(subset=['Social_Score', var]).copy().dropna()
    corr, p_value = pearsonr(temp_df['Social_Score'], temp_df[var])
    correlations[var] = {'correlation': corr, 'p_value': p_value, 'n': len(temp_df)}

correlations_df = pd.DataFrame(correlations).transpose()
correlations_df['correlation'] = correlations_df['correlation'].round(2)
# Show pvalues in scientific notation with 3 significant figures
correlations_df['p_value'] = correlations_df['p_value'].apply(lambda x: f'{x:.3g}')
correlations_df['n'] = correlations_df['n'].astype(int)

print(correlations_df)

# %%
