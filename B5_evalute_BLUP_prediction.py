"""
B5_evalute_BLUP_prediction.py
Evaluate BLUP-extended PGS prediction accuracy

This script evaluates how well the BLUP-extended PGS predicts social cognition
scores, comparing original PGS in unrelated individuals vs BLUP PGS in the
full sample. It also generates residualized PGS scores for downstream analyses.

Usage:
    python B5_evalute_BLUP_prediction.py \
        --blup-pgs <path> \
        --original-pgs <path> \
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


def evaluate_blup_prediction(blup_pgs_file, original_pgs_file, social_scores_file,
                              phenotypic_file, behavioural_file, pca_file,
                              output_residuals, output_plot, report_file, figures_dir):
    """
    Evaluate BLUP-extended PGS prediction accuracy.

    Parameters:
        blup_pgs_file (Path): Path to BLUP PGS profile file.
        original_pgs_file (Path): Path to original unrelated PGS scores.
        social_scores_file (Path): Path to social factor scores.
        phenotypic_file (Path): Path to phenotypic data.
        behavioural_file (Path): Path to behavioural data.
        pca_file (Path): Path to PCA eigenvector file.
        output_residuals (Path): Path to save residualized PGS output.
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

    # Load original PGS for unrelated sample
    pgs_unrelated = pd.read_csv(original_pgs_file, sep=' ', header=None)
    pgs_unrelated.columns = ['FID', 'IID', 'original_PGS']
    report.append(f"Original PGS (unrelated) shape: {pgs_unrelated.shape}")

    # Load BLUP PGS for full sample
    pgs_blup = pd.read_csv(blup_pgs_file, sep=r'\s+')
    pgs_blup = pgs_blup[['IID', 'SCORESUM']]
    pgs_blup.columns = ['IID', 'blup_PGS']
    report.append(f"BLUP PGS (full sample) shape: {pgs_blup.shape}")

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
    behavioural_df = behavioural_df[behavioural_df['Gender'] == 'M']
    behavioural_df = pd.merge(behavioural_df, phenotypic_df, on='Subject')
    report.append(f"Behavioural data shape: {behavioural_df.shape}")
    report.append("")

    # =========================================================================
    # PART 1: Comparison analysis (unrelated sample with both PGS types)
    # =========================================================================
    report.append("PART 1: COMPARISON ANALYSIS (UNRELATED SAMPLE)")
    report.append("-" * 80)

    # Merge data for unrelated sample comparison
    merged_df = pd.merge(pgs_unrelated, pgs_blup, on='IID')
    merged_df = pd.merge(merged_df, pca_df, on='IID')
    merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender']],
                         left_on='IID', right_on='Subject')
    merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_df = pd.merge(social_df, merged_df, on='Subject')

    # Add relatedness indicator
    merged_df['unrelated'] = merged_df['IID'].isin(pgs_unrelated['IID'])
    merged_df['unrelated'] = merged_df['unrelated'].map({True: 'Unrelated', False: 'Related'})

    report.append(f"Merged comparison data shape: {merged_df.shape}")
    report.append("")

    # Correlation between original and BLUP PGS
    r_pgs, p_pgs = pearsonr(merged_df['original_PGS'], merged_df['blup_PGS'])
    report.append(f"Correlation between original and BLUP PGS: r={r_pgs:.4f}, p={p_pgs:.2e}")
    report.append("")

    # Standardize PGS values
    merged_df['original_PGS_z'] = zscore(merged_df['original_PGS'])
    merged_df['blup_PGS_z'] = zscore(merged_df['blup_PGS'])

    # Model 1: Original PGS in unrelated individuals
    report.append("Model 1: Original PGS predicting Social Score (unrelated sample)")
    model1 = smf.ols('Social_Score ~ original_PGS_z + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  Original PGS beta: {model1.params['original_PGS_z']:.4f}")
    report.append(f"  Original PGS p-value: {model1.pvalues['original_PGS_z']:.4e}")
    report.append(f"  Model R-squared: {model1.rsquared:.4f}")
    report.append("")

    # Model 2: BLUP PGS in unrelated individuals
    report.append("Model 2: BLUP PGS predicting Social Score (unrelated sample)")
    model2 = smf.ols('Social_Score ~ blup_PGS_z + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  BLUP PGS beta: {model2.params['blup_PGS_z']:.4f}")
    report.append(f"  BLUP PGS p-value: {model2.pvalues['blup_PGS_z']:.4e}")
    report.append(f"  Model R-squared: {model2.rsquared:.4f}")
    report.append("")

    # Model 3: Testing relatedness moderation
    report.append("Model 3: Testing if relatedness moderates PGS effect")
    model3 = smf.ols('Social_Score ~ blup_PGS_z * C(unrelated) + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                     data=merged_df).fit()
    report.append(f"  Interaction p-value: {model3.pvalues.get('blup_PGS_z:C(unrelated)[T.Unrelated]', 'N/A')}")
    report.append("")

    # =========================================================================
    # PART 2: Full sample analysis with BLUP PGS
    # =========================================================================
    report.append("PART 2: FULL SAMPLE ANALYSIS (BLUP PGS)")
    report.append("-" * 80)

    # Rebuild merged data for full sample
    full_merged = pd.merge(pgs_blup, pca_df, on='IID')
    full_merged = pd.merge(full_merged, behavioural_df[['Subject', 'Gender']],
                           left_on='IID', right_on='Subject')
    full_merged = pd.merge(full_merged, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    full_merged = pd.merge(social_df, full_merged, on='Subject')

    report.append(f"Full sample merged data shape: {full_merged.shape}")

    # Adjusted model with 5 PCs
    full_merged['blup_PGS_z'] = zscore(full_merged['blup_PGS'])
    adjusted_model = smf.ols('Social_Score ~ blup_PGS_z + Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                             data=full_merged).fit()

    beta = adjusted_model.params['blup_PGS_z']
    pval = adjusted_model.pvalues['blup_PGS_z']

    report.append(f"Full sample BLUP PGS beta: {beta:.4f}")
    report.append(f"Full sample BLUP PGS p-value: {pval:.4e}")
    report.append(f"Full sample Model R-squared: {adjusted_model.rsquared:.4f}")
    report.append("")

    # =========================================================================
    # PART 3: Generate residualized PGS scores
    # =========================================================================
    report.append("PART 3: GENERATING RESIDUALIZED PGS SCORES")
    report.append("-" * 80)

    # Regress covariates out of PGS scores
    residual_model = smf.ols('blup_PGS ~ Age_in_Yrs + PC1 + PC2 + PC3 + PC4 + PC5',
                             data=full_merged).fit()
    full_merged['blup_PGS_residuals'] = residual_model.resid

    # Save residualized PGS
    output_df = full_merged[['Subject', 'blup_PGS', 'blup_PGS_residuals']].copy()
    output_df.to_csv(output_residuals, index=False)
    report.append(f"Residualized PGS saved to: {output_residuals}")
    report.append(f"Output shape: {output_df.shape}")
    report.append("")

    # =========================================================================
    # PART 4: Group comparisons
    # =========================================================================
    report.append("PART 4: GROUP COMPARISONS")
    report.append("-" * 80)

    # Standardize residuals for group assignment
    full_merged['blup_PGS_residuals_z'] = zscore(full_merged['blup_PGS_residuals'])

    # Group counts
    n_low = (full_merged['blup_PGS_residuals_z'] < -1).sum()
    n_middle = ((full_merged['blup_PGS_residuals_z'] >= -0.5) &
                (full_merged['blup_PGS_residuals_z'] <= 0.5)).sum()
    n_high = (full_merged['blup_PGS_residuals_z'] > 1).sum()

    report.append(f"Low PGS group (<-1 SD): n={n_low}")
    report.append(f"Middle PGS group (-0.5 to 0.5 SD): n={n_middle}")
    report.append(f"High PGS group (>1 SD): n={n_high}")
    report.append("")

    # T-tests for group comparisons
    middle_group = full_merged[(full_merged['blup_PGS_residuals_z'] >= -0.5) &
                               (full_merged['blup_PGS_residuals_z'] <= 0.5)]['Social_Score'].dropna()

    for threshold, label in [(1, 'High (>1 SD)'), (1.5, 'High (>1.5 SD)'),
                             (-1, 'Low (<-1 SD)'), (-1.5, 'Low (<-1.5 SD)')]:
        if threshold > 0:
            group = full_merged[full_merged['blup_PGS_residuals_z'] >= threshold]['Social_Score'].dropna()
        else:
            group = full_merged[full_merged['blup_PGS_residuals_z'] <= threshold]['Social_Score'].dropna()

        if len(group) > 0 and len(middle_group) > 0:
            t_stat, p_val = ttest_ind(group, middle_group)
            n1, n2 = len(group), len(middle_group)
            d = compute_effsize_from_t(t_stat, n1, n2, eftype='cohen')
            corrected_p = min(p_val * 4, 1.0)

            report.append(f"{label} vs Middle: t={t_stat:.3f}, p={p_val:.4f}, "
                          f"corrected p={corrected_p:.4f}, d={d:.3f}, n={n1}")

    # High vs Low direct comparisons
    report.append("")
    report.append("High vs Low direct comparisons:")
    for hi_thresh, lo_thresh, label in [(1, -1, '>1 SD vs <-1 SD'),
                                         (1.5, -1.5, '>1.5 SD vs <-1.5 SD')]:
        high_group = full_merged[full_merged['blup_PGS_residuals_z'] >= hi_thresh]['Social_Score'].dropna()
        low_group = full_merged[full_merged['blup_PGS_residuals_z'] <= lo_thresh]['Social_Score'].dropna()
        if len(high_group) > 0 and len(low_group) > 0:
            t_stat, p_val = ttest_ind(high_group, low_group)
            n1, n2 = len(high_group), len(low_group)
            d = compute_effsize_from_t(t_stat, n1, n2, eftype='cohen')
            report.append(f"  High ({label}): t={t_stat:.3f}, p={p_val:.4f}, "
                          f"d={d:.3f}, n_high={n1}, n_low={n2}")
    report.append("")

    # =========================================================================
    # Generate figures
    # =========================================================================
    report.append("GENERATING FIGURES:")
    report.append("-" * 80)

    # Figure 1: Main evaluation plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(180 * mm2inches, 140 * mm2inches), dpi=300)

    # Plot 1a: PGS correlation
    ax1 = axes[0, 0]
    sns.regplot(x='original_PGS', y='blup_PGS', data=merged_df, ax=ax1,
                scatter_kws={'color': '#E64B35', 's': 5, 'alpha': 0.8, 'linewidths': 0.1},
                line_kws={'color': 'k', 'lw': 1})
    ax1.set_xlabel('Original PGS')
    ax1.set_ylabel('BLUP PGS')
    ax1.set_title(f'PGS Correlation (r={r_pgs:.2f})')
    sns.despine(ax=ax1, offset=8)

    # Plot 1b: PGS distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(merged_df['original_PGS'], bins=30, alpha=0.5, label='Original', color='steelblue')
    ax2.hist(merged_df['blup_PGS'], bins=30, alpha=0.5, label='BLUP', color='coral')
    ax2.set_xlabel('PGS Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('PGS Distribution Comparison')
    ax2.legend(fontsize=7)
    sns.despine(ax=ax2, offset=8)

    # Plot 1c: BLUP PGS vs Social Score
    ax3 = axes[1, 0]
    full_merged['pgs_group'] = np.select(
        [full_merged['blup_PGS_residuals_z'] <= -1,
         (full_merged['blup_PGS_residuals_z'] >= -0.5) & (full_merged['blup_PGS_residuals_z'] <= 0.5),
         full_merged['blup_PGS_residuals_z'] >= 1],
        ['low', 'middle', 'high'],
        default='other'
    )
    palette = {'middle': '#91D1C299', 'high': '#8491B499', 'low': '#4DBBD599', 'other': 'lightgrey'}
    sns.scatterplot(x='blup_PGS_residuals_z', y='Social_Score', data=full_merged,
                    hue='pgs_group', palette=palette, s=5, alpha=0.8, ax=ax3, legend=False)
    sns.regplot(x='blup_PGS_residuals_z', y='Social_Score', data=full_merged,
                scatter=False, line_kws={'linewidth': 1, 'color': 'k'}, ax=ax3)
    ax3.set_xlabel('BLUP PGS (residualized, z)')
    ax3.set_ylabel('Social Score')
    ax3.set_title(f'PGS-Social Association (β={beta:.3f}, p={pval:.3f})')
    sns.despine(ax=ax3, offset=8)

    # Plot 1d: Residualized PGS distribution with group shading
    ax4 = axes[1, 1]
    ax4.axvspan(1, 3, color='#8491B499', alpha=0.5, label=f'High (n={n_high})')
    ax4.axvspan(-0.5, 0.5, color='#91D1C299', alpha=0.5, label=f'Middle (n={n_middle})')
    ax4.axvspan(-3, -1, color='#4DBBD599', alpha=0.5, label=f'Low (n={n_low})')
    ax4.hist(full_merged['blup_PGS_residuals_z'], bins=50, range=(-3, 3),
             color='lightgrey', edgecolor='black', linewidth=0.3, zorder=3)
    ax4.set_xlabel('BLUP PGS (residualized, z)')
    ax4.set_ylabel('Count')
    ax4.set_title('PGS Distribution with Groups')
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
        ax.set_xlabel('PGS Group')
        ax.set_ylabel('Social Score')
        ax.set_title('Social Score by PGS Group')
        sns.despine(ax=ax, offset=8, trim=True)
        plt.tight_layout()

        group_plot = figures_dir / 'B5_pgs_group_comparison.png'
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
        'pgs_correlation': r_pgs,
        'original_pgs_beta': model1.params['original_PGS_z'],
        'original_pgs_pval': model1.pvalues['original_PGS_z'],
        'blup_pgs_beta': beta,
        'blup_pgs_pval': pval,
        'n_unrelated': len(merged_df),
        'n_full_sample': len(full_merged)
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BLUP-extended PGS prediction accuracy'
    )
    parser.add_argument('--blup-pgs', required=True,
                        help='Path to BLUP PGS profile file')
    parser.add_argument('--original-pgs', required=True,
                        help='Path to original unrelated PGS scores')
    parser.add_argument('--social-scores', required=True,
                        help='Path to social factor scores CSV')
    parser.add_argument('--phenotypic', required=True,
                        help='Path to phenotypic data CSV')
    parser.add_argument('--behavioural', required=True,
                        help='Path to behavioural data CSV')
    parser.add_argument('--pca', required=True,
                        help='Path to PCA eigenvector file')
    parser.add_argument('--output-residuals', required=True,
                        help='Path to save residualized PGS output')
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
        blup_pgs_file=Path(args.blup_pgs),
        original_pgs_file=Path(args.original_pgs),
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
