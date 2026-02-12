"""
B5_evalute_BLUP_prediction.py
Evaluate BLUP-extended PRS prediction accuracy

This script evaluates how well the BLUP-extended PRS predicts social cognition
scores, comparing original PRS in unrelated individuals vs BLUP PRS in the
full sample. It also generates residualized PRS scores for downstream analyses.

Usage:
    python B5_evalute_BLUP_prediction.py \
        --blup-prs <path> \
        --original-prs <path> \
        --social-scores <path> \
        --phenotypic <path> \
        --behavioural <path> \
        --pca <path> \
        --output-residuals <path> \
        --output-plot <path> \
        --project <path>
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore, ttest_ind
import seaborn as sns
import statsmodels.formula.api as smf
from pingouin import compute_effsize_from_t

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701


def evaluate_blup_prediction(blup_prs_file, original_prs_file, social_scores_file,
                              phenotypic_file, behavioural_file, pca_file,
                              output_residuals, output_plot, report_file, figures_dir):
    """
    Evaluate BLUP-extended PRS prediction accuracy.

    Parameters:
        blup_prs_file (Path): Path to BLUP PRS profile file.
        original_prs_file (Path): Path to original unrelated PRS scores.
        social_scores_file (Path): Path to social factor scores.
        phenotypic_file (Path): Path to phenotypic data.
        behavioural_file (Path): Path to behavioural data.
        pca_file (Path): Path to PCA eigenvector file.
        output_residuals (Path): Path to save residualized PRS output.
        output_plot (Path): Path to save main evaluation plot.
        report_file (Path): Path to save the processing report.
        figures_dir (Path): Path to figures directory.

    Returns:
        dict: Results dictionary with model statistics.
    """
    # Initialize report content
    report = []
    report.append("=" * 80)
    report.append("B5: BLUP PREDICTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Load data
    report.append("LOADING DATA:")
    report.append("-" * 80)

    # Load original PRS for unrelated sample
    prs_unrelated = pd.read_csv(original_prs_file, sep=' ', header=None)
    prs_unrelated.columns = ['FID', 'IID', 'original_PRS']
    report.append(f"Original PRS (unrelated) shape: {prs_unrelated.shape}")

    # Load BLUP PRS for full sample
    prs_blup = pd.read_csv(blup_prs_file, sep=r'\s+')
    prs_blup = prs_blup[['IID', 'SCORESUM']]
    prs_blup.columns = ['IID', 'blup_PRS']
    report.append(f"BLUP PRS (full sample) shape: {prs_blup.shape}")

    # Load PCA data - using only first 5 PCs
    pca_df = pd.read_csv(pca_file, sep=' ', header=None)
    pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
    pca_df = pca_df[['IID'] + [f'PC{i}' for i in range(1, 6)]]
    report.append(f"PCA data shape: {pca_df.shape}")

    # Load social scores
    social_df = pd.read_csv(social_scores_file)
    report.append(f"Social scores shape: {social_df.shape}")

    # Load phenotypic and behavioural data
    phenotypic_df = pd.read_csv(phenotypic_file)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    behavioural_df = pd.read_csv(behavioural_file)
    behavioural_df = pd.merge(behavioural_df, phenotypic_df, on='Subject')
    report.append(f"Behavioural data shape: {behavioural_df.shape}")
    report.append("")

    # =========================================================================
    # PART 1: Comparison analysis (unrelated sample with both PRS types)
    # =========================================================================
    report.append("PART 1: COMPARISON ANALYSIS (UNRELATED SAMPLE)")
    report.append("-" * 80)

    # Merge data for unrelated sample comparison
    merged_df = pd.merge(prs_unrelated, prs_blup, on='IID')
    merged_df = pd.merge(merged_df, pca_df, on='IID')
    merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender']],
                         left_on='IID', right_on='Subject')
    merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_df = pd.merge(social_df, merged_df, on='Subject')

    # Add relatedness indicator
    merged_df['unrelated'] = merged_df['IID'].isin(prs_unrelated['IID'])
    merged_df['unrelated'] = merged_df['unrelated'].map({True: 'Unrelated', False: 'Related'})

    report.append(f"Merged comparison data shape: {merged_df.shape}")
    report.append("")

    # Correlation between original and BLUP PRS
    r_prs, p_prs = pearsonr(merged_df['original_PRS'], merged_df['blup_PRS'])
    report.append(f"Correlation between original and BLUP PRS: r={r_prs:.4f}, p={p_prs:.2e}")
    report.append("")

    # Standardize PRS values
    merged_df['original_PRS_z'] = zscore(merged_df['original_PRS'])
    merged_df['blup_PRS_z'] = zscore(merged_df['blup_PRS'])

    # Model 1: Original PRS in unrelated individuals
    report.append("Model 1: Original PRS predicting Social Score (unrelated sample)")
    model1 = smf.ols('Social_Score ~ original_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  Original PRS beta: {model1.params['original_PRS_z']:.4f}")
    report.append(f"  Original PRS p-value: {model1.pvalues['original_PRS_z']:.4e}")
    report.append(f"  Model R-squared: {model1.rsquared:.4f}")
    report.append("")

    # Model 2: BLUP PRS in unrelated individuals
    report.append("Model 2: BLUP PRS predicting Social Score (unrelated sample)")
    model2 = smf.ols('Social_Score ~ blup_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  BLUP PRS beta: {model2.params['blup_PRS_z']:.4f}")
    report.append(f"  BLUP PRS p-value: {model2.pvalues['blup_PRS_z']:.4e}")
    report.append(f"  Model R-squared: {model2.rsquared:.4f}")
    report.append("")

    # Model 3: Testing relatedness moderation
    report.append("Model 3: Testing if relatedness moderates PRS effect")
    model3 = smf.ols('Social_Score ~ blup_PRS_z * C(unrelated) + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  Interaction p-value: {model3.pvalues.get('blup_PRS_z:C(unrelated)[T.Unrelated]', 'N/A')}")
    report.append("")

    # =========================================================================
    # PART 2: Full sample analysis with BLUP PRS
    # =========================================================================
    report.append("PART 2: FULL SAMPLE ANALYSIS (BLUP PRS)")
    report.append("-" * 80)

    # Rebuild merged data for full sample
    full_merged = pd.merge(prs_blup, pca_df, on='IID')
    full_merged = pd.merge(full_merged, behavioural_df[['Subject', 'Gender']],
                           left_on='IID', right_on='Subject')
    full_merged = pd.merge(full_merged, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    full_merged = pd.merge(social_df, full_merged, on='Subject')

    report.append(f"Full sample merged data shape: {full_merged.shape}")

    # Adjusted model with 5 PCs
    full_merged['blup_PRS_z'] = zscore(full_merged['blup_PRS'])
    adjusted_model = smf.ols('Social_Score ~ blup_PRS_z + C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                             data=full_merged).fit()

    beta = adjusted_model.params['blup_PRS_z']
    pval = adjusted_model.pvalues['blup_PRS_z']

    report.append(f"Full sample BLUP PRS beta: {beta:.4f}")
    report.append(f"Full sample BLUP PRS p-value: {pval:.4e}")
    report.append(f"Full sample Model R-squared: {adjusted_model.rsquared:.4f}")
    report.append("")

    # =========================================================================
    # PART 3: Generate residualized PRS scores
    # =========================================================================
    report.append("PART 3: GENERATING RESIDUALIZED PRS SCORES")
    report.append("-" * 80)

    # Regress covariates out of PRS scores
    residual_model = smf.ols('blup_PRS ~ C(Gender) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                             data=full_merged).fit()
    full_merged['blup_PRS_residuals'] = residual_model.resid

    # Save residualized PRS
    output_df = full_merged[['Subject', 'blup_PRS', 'blup_PRS_residuals']].copy()
    output_df.to_csv(output_residuals, index=False)
    report.append(f"Residualized PRS saved to: {output_residuals}")
    report.append(f"Output shape: {output_df.shape}")
    report.append("")

    # =========================================================================
    # PART 4: Group comparisons
    # =========================================================================
    report.append("PART 4: GROUP COMPARISONS")
    report.append("-" * 80)

    # Standardize residuals for group assignment
    full_merged['blup_PRS_residuals_z'] = zscore(full_merged['blup_PRS_residuals'])

    # Group counts
    n_low = (full_merged['blup_PRS_residuals_z'] < -1).sum()
    n_middle = ((full_merged['blup_PRS_residuals_z'] >= -0.5) &
                (full_merged['blup_PRS_residuals_z'] <= 0.5)).sum()
    n_high = (full_merged['blup_PRS_residuals_z'] > 1).sum()

    report.append(f"Low PRS group (<-1 SD): n={n_low}")
    report.append(f"Middle PRS group (-0.5 to 0.5 SD): n={n_middle}")
    report.append(f"High PRS group (>1 SD): n={n_high}")
    report.append("")

    # T-tests for group comparisons
    middle_group = full_merged[(full_merged['blup_PRS_residuals_z'] >= -0.5) &
                               (full_merged['blup_PRS_residuals_z'] <= 0.5)]['Social_Score'].dropna()

    for threshold, label in [(1, 'High (>1 SD)'), (1.5, 'High (>1.5 SD)'),
                             (-1, 'Low (<-1 SD)'), (-1.5, 'Low (<-1.5 SD)')]:
        if threshold > 0:
            group = full_merged[full_merged['blup_PRS_residuals_z'] >= threshold]['Social_Score'].dropna()
        else:
            group = full_merged[full_merged['blup_PRS_residuals_z'] <= threshold]['Social_Score'].dropna()

        if len(group) > 0 and len(middle_group) > 0:
            t_stat, p_val = ttest_ind(group, middle_group)
            n1, n2 = len(group), len(middle_group)
            d = compute_effsize_from_t(t_stat, n1, n2, eftype='cohen')
            corrected_p = min(p_val * 4, 1.0)

            report.append(f"{label} vs Middle: t={t_stat:.3f}, p={p_val:.4f}, "
                          f"corrected p={corrected_p:.4f}, d={d:.3f}, n={n1}")
    report.append("")

    # =========================================================================
    # Generate figures
    # =========================================================================
    report.append("GENERATING FIGURES:")
    report.append("-" * 80)

    # Figure 1: Main evaluation plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(180 * mm2inches, 140 * mm2inches), dpi=300)

    # Plot 1a: PRS correlation
    ax1 = axes[0, 0]
    sns.regplot(x='original_PRS', y='blup_PRS', data=merged_df, ax=ax1,
                scatter_kws={'color': '#E64B35', 's': 5, 'alpha': 0.8, 'linewidths': 0.1},
                line_kws={'color': 'k', 'lw': 1})
    ax1.set_xlabel('Original PRS')
    ax1.set_ylabel('BLUP PRS')
    ax1.set_title(f'PRS Correlation (r={r_prs:.2f})')
    sns.despine(ax=ax1, offset=8)

    # Plot 1b: PRS distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(merged_df['original_PRS'], bins=30, alpha=0.5, label='Original', color='steelblue')
    ax2.hist(merged_df['blup_PRS'], bins=30, alpha=0.5, label='BLUP', color='coral')
    ax2.set_xlabel('PRS Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('PRS Distribution Comparison')
    ax2.legend(fontsize=7)
    sns.despine(ax=ax2, offset=8)

    # Plot 1c: BLUP PRS vs Social Score
    ax3 = axes[1, 0]
    full_merged['pgs_group'] = np.select(
        [full_merged['blup_PRS_residuals_z'] <= -1,
         (full_merged['blup_PRS_residuals_z'] >= -0.5) & (full_merged['blup_PRS_residuals_z'] <= 0.5),
         full_merged['blup_PRS_residuals_z'] >= 1],
        ['low', 'middle', 'high'],
        default='other'
    )
    palette = {'middle': '#91D1C299', 'high': '#8491B499', 'low': '#4DBBD599', 'other': 'lightgrey'}
    sns.scatterplot(x='blup_PRS_residuals_z', y='Social_Score', data=full_merged,
                    hue='pgs_group', palette=palette, s=5, alpha=0.8, ax=ax3, legend=False)
    sns.regplot(x='blup_PRS_residuals_z', y='Social_Score', data=full_merged,
                scatter=False, line_kws={'linewidth': 1, 'color': 'k'}, ax=ax3)
    ax3.set_xlabel('BLUP PRS (residualized, z)')
    ax3.set_ylabel('Social Score')
    ax3.set_title(f'PRS-Social Association (β={beta:.3f}, p={pval:.3f})')
    sns.despine(ax=ax3, offset=8)

    # Plot 1d: Residualized PRS distribution with group shading
    ax4 = axes[1, 1]
    ax4.axvspan(1, 3, color='#8491B499', alpha=0.5, label=f'High (n={n_high})')
    ax4.axvspan(-0.5, 0.5, color='#91D1C299', alpha=0.5, label=f'Middle (n={n_middle})')
    ax4.axvspan(-3, -1, color='#4DBBD599', alpha=0.5, label=f'Low (n={n_low})')
    ax4.hist(full_merged['blup_PRS_residuals_z'], bins=50, range=(-3, 3),
             color='lightgrey', edgecolor='black', linewidth=0.3, zorder=3)
    ax4.set_xlabel('BLUP PRS (residualized, z)')
    ax4.set_ylabel('Count')
    ax4.set_title('PRS Distribution with Groups')
    ax4.set_xlim([-3, 3])
    ax4.legend(fontsize=6, loc='upper right')
    sns.despine(ax=ax4, offset=8)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    report.append(f"Main evaluation plot saved to: {output_plot}")

    # Figure 2: Group comparison boxplot
    plotting_df = full_merged[full_merged['pgs_group'].isin(['low', 'middle', 'high'])].copy()
    if len(plotting_df) > 0:
        fig2, ax = plt.subplots(figsize=(80 * mm2inches, 60 * mm2inches), dpi=300)
        order = ['low', 'middle', 'high']
        palette_box = {'middle': '#91D1C299', 'high': '#8491B499', 'low': '#4DBBD599'}

        sns.boxplot(x='pgs_group', y='Social_Score', data=plotting_df, order=order,
                    width=0.5, boxprops={'facecolor': 'None'}, showfliers=False,
                    whiskerprops={'linewidth': 1}, ax=ax)
        sns.swarmplot(x='pgs_group', y='Social_Score', data=plotting_df, order=order,
                      palette=palette_box, s=2, alpha=0.7, ax=ax)
        ax.set_xlabel('PRS Group')
        ax.set_ylabel('Social Score')
        ax.set_title('Social Score by PRS Group')
        sns.despine(ax=ax, offset=8, trim=True)
        plt.tight_layout()

        group_plot = figures_dir / 'B5_prs_group_comparison.png'
        plt.savefig(group_plot, dpi=300, bbox_inches='tight')
        plt.close()
        report.append(f"Group comparison plot saved to: {group_plot}")

    report.append("")

    # Write report to file
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    # Return results dictionary
    results = {
        'prs_correlation': r_prs,
        'original_prs_beta': model1.params['original_PRS_z'],
        'original_prs_pval': model1.pvalues['original_PRS_z'],
        'blup_prs_beta': beta,
        'blup_prs_pval': pval,
        'n_unrelated': len(merged_df),
        'n_full_sample': len(full_merged)
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BLUP-extended PRS prediction accuracy'
    )
    parser.add_argument('--blup-prs', required=True,
                        help='Path to BLUP PRS profile file')
    parser.add_argument('--original-prs', required=True,
                        help='Path to original unrelated PRS scores')
    parser.add_argument('--social-scores', required=True,
                        help='Path to social factor scores CSV')
    parser.add_argument('--phenotypic', required=True,
                        help='Path to phenotypic data CSV')
    parser.add_argument('--behavioural', required=True,
                        help='Path to behavioural data CSV')
    parser.add_argument('--pca', required=True,
                        help='Path to PCA eigenvector file')
    parser.add_argument('--output-residuals', required=True,
                        help='Path to save residualized PRS output')
    parser.add_argument('--output-plot', required=True,
                        help='Path to save main evaluation plot')
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    report_file = reports_dir / 'B5_evaluate_blup_prediction_report.txt'

    results = evaluate_blup_prediction(
        blup_prs_file=Path(args.blup_prs),
        original_prs_file=Path(args.original_prs),
        social_scores_file=Path(args.social_scores),
        phenotypic_file=Path(args.phenotypic),
        behavioural_file=Path(args.behavioural),
        pca_file=Path(args.pca),
        output_residuals=Path(args.output_residuals),
        output_plot=Path(args.output_plot),
        report_file=report_file,
        figures_dir=figures_dir
    )


if __name__ == "__main__":
    main()
