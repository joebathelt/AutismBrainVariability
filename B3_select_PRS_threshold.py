"""
B3_select_PRS_threshold.py
Select PRS threshold for analysis

This script uses a fixed p-value threshold (p=0.1) for PRS, which was the
optimal threshold identified in Grove et al. (2019) - the discovery GWAS
for autism that this PRS is based on. Using the discovery GWAS threshold
is the most principled approach as it:
1. Avoids overfitting to sample-specific noise
2. Ensures reproducibility across runs
3. Is scientifically justified by the original GWAS

The script also runs cross-validation to report performance metrics for
transparency, but does not use CV results to select the threshold.

Usage:
    python B3_select_PRS_threshold.py \
        --prs <path> \
        --phenotype <path> \
        --pca <path> \
        --output-plot <path> \
        --output-threshold <path> \
        --project <path>
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701


def select_prs_threshold(prs_file, phenotype_file, pca_file, output_plot, output_threshold,
                         report_file, figures_dir, target_threshold=0.1):
    """
    Use a fixed PRS threshold (default p=0.1) for reproducibility.

    Using p=0.1 because:
    1. This was the optimal threshold in Grove et al. (2019), the discovery GWAS
       for autism that this PRS is based on
    2. Using the discovery GWAS threshold is the most principled approach
    3. Avoids overfitting to sample-specific noise in threshold selection

    Steps:
    1. Load PRS scores at multiple thresholds
    2. Load social cognition scores
    3. Load PCA data for population stratification correction
    4. Use fixed p=0.1 threshold (from Grove et al. 2019)
    5. Run CV to report performance metrics (not for selection)
    6. Generate visualization and report

    Parameters:
        prs_file (Path): Path to PRS scores file (all_score format).
        phenotype_file (Path): Path to phenotype data with social scores.
        pca_file (Path): Path to PCA eigenvector file.
        output_plot (Path): Path to save the threshold evaluation plot.
        output_threshold (Path): Path to save the selected threshold info.
        report_file (Path): Path to save the processing report.
        figures_dir (Path): Path to figures directory.
        target_threshold (float): Target p-value threshold (default 0.05).

    Returns:
        tuple: (selected_threshold, results dictionary)
    """
    # Initialize report content
    report = []
    report.append("=" * 80)
    report.append("B3: PRS THRESHOLD REPORT (FIXED p=0.1)")
    report.append("=" * 80)
    report.append("")
    report.append(f"Using fixed threshold p={target_threshold}")
    report.append("Rationale: p=0.1 was the optimal threshold in Grove et al. (2019),")
    report.append("the discovery GWAS for autism that this PRS is based on.")
    report.append("Using the discovery GWAS threshold is the most principled approach.")
    report.append("")

    # Load PGS scores for unrelated individuals
    report.append("LOADING DATA:")
    report.append("-" * 80)
    PGS_df = pd.read_csv(prs_file, sep=' ')
    report.append(f"PRS data shape: {PGS_df.shape}")

    # Load social cognition scores
    social_df = pd.read_csv(phenotype_file)
    report.append(f"Phenotype data shape: {social_df.shape}")

    # Load PCA data (using first 5 PCs)
    pca_df = pd.read_csv(pca_file, sep=' ', header=None)
    pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
    report.append(f"PCA data shape: {pca_df.shape}")
    report.append("")

    # Merge all data
    merged_df = pd.merge(PGS_df, social_df, left_on='IID', right_on='Subject')
    merged_df = pd.merge(merged_df, pca_df, on='IID')
    report.append(f"Merged data shape: {merged_df.shape}")
    report.append("")

    # Get all available PRS thresholds
    prs_thresholds = [col for col in merged_df.columns if col.startswith('Pt_')]
    report.append(f"Available PRS thresholds: {len(prs_thresholds)}")
    report.append("")

    # Find the threshold closest to target (p=0.05)
    def get_pvalue(thresh_name):
        try:
            return float(thresh_name.split('_')[1])
        except:
            return float('inf')

    # Sort thresholds by distance to target
    thresholds_with_pvals = [(t, get_pvalue(t)) for t in prs_thresholds]
    thresholds_with_pvals.sort(key=lambda x: abs(x[1] - target_threshold))
    selected_threshold = thresholds_with_pvals[0][0]
    selected_pval = thresholds_with_pvals[0][1]

    report.append("THRESHOLD SELECTION:")
    report.append("-" * 80)
    report.append(f"Target p-value: {target_threshold}")
    report.append(f"Selected threshold: {selected_threshold} (p={selected_pval})")
    report.append("")

    # Run cross-validation to report performance (not for selection)
    report.append("CROSS-VALIDATION PERFORMANCE:")
    report.append("-" * 80)
    report.append("Running 5-fold CV to evaluate selected threshold...")
    report.append(f"Using first 10 PCs for population stratification correction")
    report.append("")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    pc_cols = [f'PC{i}' for i in range(1, 11)]
    pc_formula = ' + '.join(pc_cols)

    # Evaluate selected threshold
    correlations = []
    for train_idx, test_idx in kf.split(merged_df):
        merged_df['thresholded_PRS'] = merged_df[selected_threshold]
        train_data = merged_df.iloc[train_idx]
        test_data = merged_df.iloc[test_idx]

        pc_model = smf.ols(f'thresholded_PRS ~ {pc_formula}', data=train_data).fit()
        test_prs_corrected = test_data['thresholded_PRS'] - pc_model.predict(test_data)
        corr, _ = pearsonr(test_prs_corrected, test_data['Social_Score'])
        correlations.append(corr)

    avg_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    results[selected_threshold] = {
        'avg_correlation': avg_corr,
        'std_correlation': std_corr,
        'fold_correlations': correlations
    }

    report.append(f"Selected threshold: {selected_threshold}")
    report.append(f"CV correlation: {avg_corr:.4f} (+/- {std_corr:.4f})")
    report.append(f"Fold correlations: {[f'{c:.3f}' for c in correlations]}")
    report.append("")

    # For compatibility, keep best_threshold variable
    best_threshold = selected_threshold
    best_avg_corr = avg_corr

    # Save selected threshold info
    with open(output_threshold, 'w') as f:
        f.write(f"{best_threshold}\n")
        f.write(f"{best_avg_corr:.6f}\n")
        f.write(f"{results[best_threshold]['std_correlation']:.6f}\n")

    report.append(f"Selected threshold saved to: {output_threshold}")
    report.append("")

    # Save z-scored PRS for unrelated individuals (for BLUP in B3)
    # This ensures B3 uses the same filtered sample as threshold selection
    # Note: FID may be renamed to FID_x after merge with pca_df
    fid_col = 'FID_x' if 'FID_x' in merged_df.columns else 'FID'
    prs_for_blup = merged_df[[fid_col, 'IID', best_threshold]].copy()
    prs_for_blup.columns = ['FID', 'IID', 'PRS']
    prs_for_blup['PRS'] = zscore(prs_for_blup['PRS'])

    # Save to PLINK directory for B3 to use
    plink_dir = Path(prs_file).parent
    unrelated_prs_file = plink_dir / 'unrelated_prs_scores.txt'
    prs_for_blup.to_csv(unrelated_prs_file, index=False, header=False, sep=' ')
    report.append(f"Z-scored PRS for BLUP saved to: {unrelated_prs_file}")
    report.append(f"  Individuals: {len(prs_for_blup)}")
    report.append(f"  PRS mean: {prs_for_blup['PRS'].mean():.6f}, std: {prs_for_blup['PRS'].std():.6f}")
    report.append("")

    # Get PC-corrected PRS for the selected threshold (using full data)
    merged_df['best_PRS'] = merged_df[best_threshold]
    pc_model_full = smf.ols(f'best_PRS ~ {pc_formula}', data=merged_df).fit()
    prs_corrected = merged_df['best_PRS'] - pc_model_full.predict(merged_df)

    # Calculate full-sample correlation for display
    full_corr, full_pval_stat = pearsonr(prs_corrected, merged_df['Social_Score'])
    best_pval = float(best_threshold.split('_')[1])

    report.append(f"Full-sample correlation: r={full_corr:.4f}, p={full_pval_stat:.4e}")
    report.append("")

    # =========================================================================
    # Figure 1: Main PRS-Social relationship plot (2 panels)
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(180 * mm2inches, 70 * mm2inches), dpi=300)

    # Plot 1: Scatter plot for selected threshold
    ax1 = axes1[0]
    ax1.scatter(prs_corrected, merged_df['Social_Score'], alpha=0.5, s=10, c='steelblue')
    z = np.polyfit(prs_corrected, merged_df['Social_Score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(prs_corrected.min(), prs_corrected.max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=1.5)
    ax1.set_xlabel('PC-corrected PRS')
    ax1.set_ylabel('Social Score')
    ax1.set_title(f'{best_threshold}\nCV r={best_avg_corr:.3f}, Full r={full_corr:.3f}')

    # Plot 2: Fold-wise correlations
    ax2 = axes1[1]
    fold_corrs = results[best_threshold]['fold_correlations']
    ax2.bar(range(1, 6), fold_corrs, color='steelblue', alpha=0.7)
    ax2.axhline(y=best_avg_corr, color='red', linestyle='--', linewidth=1, label=f'Mean: {best_avg_corr:.3f}')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Correlation')
    ax2.set_title(f'5-Fold CV Performance\n(Fixed threshold p={target_threshold})')
    ax2.set_xticks(range(1, 6))
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    report.append(f"PRS evaluation plot saved to: {output_plot}")

    # =========================================================================
    # Figure 2: Detailed distribution plots
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(180 * mm2inches, 140 * mm2inches), dpi=300)

    # Plot 2a: Distribution of raw PRS scores
    ax2a = axes2[0, 0]
    ax2a.hist(merged_df[best_threshold], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax2a.set_xlabel('Raw PRS')
    ax2a.set_ylabel('Frequency')
    ax2a.set_title(f'Distribution of Raw PRS ({best_threshold})')

    # Plot 2b: Distribution of PC-corrected PRS
    ax2b = axes2[0, 1]
    ax2b.hist(prs_corrected, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax2b.set_xlabel('PC-corrected PRS')
    ax2b.set_ylabel('Frequency')
    ax2b.set_title('Distribution of PC-corrected PRS')

    # Plot 2c: Distribution of z-scored PRS (for BLUP)
    ax2c = axes2[1, 0]
    ax2c.hist(prs_for_blup['PRS'], bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax2c.set_xlabel('Z-scored PRS')
    ax2c.set_ylabel('Frequency')
    ax2c.set_title('Distribution of Z-scored PRS (for BLUP)')

    # Plot 2d: Distribution of Social Scores
    ax2d = axes2[1, 1]
    ax2d.hist(merged_df['Social_Score'], bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax2d.set_xlabel('Social Score')
    ax2d.set_ylabel('Frequency')
    ax2d.set_title('Distribution of Social Scores')

    plt.tight_layout()
    cv_performance_plot = figures_dir / 'B3_cv_performance.png'
    plt.savefig(cv_performance_plot, dpi=300, bbox_inches='tight')
    plt.close()

    report.append(f"CV performance plot saved to: {cv_performance_plot}")
    report.append("")

    # Write report to file
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    return best_threshold, results


def main():
    parser = argparse.ArgumentParser(
        description='Select optimal PRS threshold based on 5-fold cross-validation'
    )
    parser.add_argument('--prs', required=True,
                        help='Path to PRS scores file (all_score format)')
    parser.add_argument('--phenotype', required=True,
                        help='Path to phenotype data with social scores')
    parser.add_argument('--pca', required=True,
                        help='Path to PCA eigenvector file')
    parser.add_argument('--output-plot', required=True,
                        help='Path to save the threshold evaluation plot')
    parser.add_argument('--output-threshold', required=True,
                        help='Path to save the selected threshold info')
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    report_file = reports_dir / 'B3_select_prs_threshold_report.txt'

    best_threshold, _ = select_prs_threshold(
        prs_file=Path(args.prs),
        phenotype_file=Path(args.phenotype),
        pca_file=Path(args.pca),
        output_plot=Path(args.output_plot),
        output_threshold=Path(args.output_threshold),
        report_file=report_file,
        figures_dir=figures_dir
    )


if __name__ == "__main__":
    main()
