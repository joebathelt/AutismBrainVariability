# %%
"""
C3b_continuous_heteroscedasticity_analysis.py
Continuous Heteroscedasticity Analysis - Testing Landscape Theory

This script tests whether the variance of brain network organization
changes continuously with polygenic score (PGS), using formal
heteroscedasticity tests rather than group comparisons:

1. Breusch-Pagan test for heteroscedasticity
2. White test for heteroscedasticity (general form)
3. Quantile regression at multiple quantiles (fan-shape analysis)
4. Decile variability visualization

All analyses applied to both modularity (primary) and global_efficiency (comparison).

Parcellation resolution is fixed upstream by C2b_evaluate_communities.py;
n_nodes is derived from the number of rows in the partition file.

Usage:
    python C3b_continuous_heteroscedasticity_analysis.py \
        --project <path> \
        --pgs <path> \
        --social <path> \
        --behavioural <path> \
        --phenotypic <path> \
        --movement <path> \
        --ids <path> \
        --matrices-dir <path> \
        --partition <path> \
        --threshold 0.2
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore, pearsonr, norm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.nonparametric.smoothers_lowess import lowess
import networkx as nx
import bct

from utils.covariates import COVARIATES, regress_out_covariates

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701
FIGURE_DPI = 300


# %%
# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(project_folder, pgs_file, social_file, behavioural_file,
                          phenotypic_file, movement_file, id_file, matrices_dir,
                          n_nodes, motion_threshold, report):
    """Load all data and create PGS groups."""
    msg = "Loading and preparing data..."
    print(msg)
    report.append(msg)

    # Load connectivity matrices (full correlation)
    matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'

    mats_df = pd.read_csv(matrix_file, header=None, sep=r'\s+')
    mats_df.columns = [f'conn_{i+1}' for i in range(mats_df.shape[1])]
    ids = pd.read_csv(id_file, header=None)[0].tolist()
    mats_df.index = ids

    report.append(f"Loaded connectivity matrices: {mats_df.shape}")

    # Load other datasets
    pgs_df = pd.read_csv(pgs_file)
    social_df = pd.read_csv(social_file)
    behavioural_df = pd.read_csv(behavioural_file)
    behavioural_df = behavioural_df[behavioural_df['Gender'] == 'M']
    phenotypic_df = pd.read_csv(phenotypic_file)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    movement_df = pd.read_csv(movement_file)

    # Merge datasets
    merged_df = pd.merge(pgs_df, mats_df, left_on='Subject', right_index=True)
    merged_df = pd.merge(merged_df, social_df[['Subject', 'Social_Score']], on='Subject')
    merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']], on='Subject')
    merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_df = pd.merge(merged_df, movement_df[['Subject', 'Movement_RelativeRMS_mean']], on='Subject')

    report.append(f"Merged data shape: {merged_df.shape}")

    # Quality control
    merged_df = merged_df[merged_df['Movement_RelativeRMS_mean'] < motion_threshold]
    merged_df = merged_df.dropna()

    msg = f"Final sample after QC: {len(merged_df)} subjects"
    print(msg)
    report.append(msg)

    # Create PGS groups
    # Groups: low (<-1 SD), middle (-0.5 to 0.5 SD), high (>+1 SD)
    # Subjects between -1 and -0.5, and between 0.5 and 1 are excluded
    merged_df['pgs_z'] = zscore(merged_df['blup_PGS_residuals'])
    merged_df['pgs_group'] = pd.cut(
        merged_df['pgs_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['low', 'exclude_low', 'middle', 'exclude_high', 'high']
    )

    report.append("")
    report.append("PGS group sizes:")
    for group in ['low', 'middle', 'high']:
        n = len(merged_df[merged_df['pgs_group'] == group])
        report.append(f"  {group.capitalize()}: {n}")

    return merged_df


# %%
# =============================================================================
# 2. NETWORK METRICS CALCULATION
# =============================================================================

def calculate_network_metrics(merged_df, partition_file, n_nodes, threshold, report):
    """Calculate modularity and global efficiency for all subjects."""
    msg = f"\nCalculating network metrics ({n_nodes} nodes, {threshold*100:.0f}% threshold)..."
    print(msg)
    report.append("")
    report.append(msg)

    # Load modularity partition
    partition_df = None
    if partition_file and partition_file.exists():
        partition_df = pd.read_csv(partition_file)
        report.append(f"Loaded partition from: {partition_file}")
    else:
        report.append("Warning: Using default community detection (no partition file)")

    results = []
    n_subjects = len(merged_df)

    for i, (subject_id, row) in enumerate(merged_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_subjects} subjects")

        # Extract connectivity matrix
        conn_data = row[[col for col in row.index if col.startswith('conn_')]].values
        mat = np.reshape(conn_data, (n_nodes, n_nodes)).astype(np.float64)
        mat = mat / 100  # Normalize
        mat = bct.threshold_proportional(mat, threshold)
        mat = np.nan_to_num(mat, nan=0.0)

        # Calculate modularity
        if partition_df is None:
            raise FileNotFoundError(f"Partition file not found: {partition_file}. "
                                    "Run C3 first or provide --partition argument.")
        else:
            _, modularity = bct.modularity_und_sign(mat, partition_df['community_id'].values)

        # Calculate global efficiency
        mat_pos = mat.copy()
        mat_pos[mat_pos < 0] = 0
        G = nx.from_numpy_array(mat_pos)
        global_efficiency = nx.global_efficiency(G)

        results.append({
            'Subject': subject_id,
            'modularity': modularity,
            'global_efficiency': global_efficiency,
            'Social_Score': row['Social_Score'],
            'pgs_group': row['pgs_group'],
            'pgs_z': row['pgs_z'],
            'Age_in_Yrs': row['Age_in_Yrs'],
            'FS_IntraCranial_Vol': row['FS_IntraCranial_Vol'],
            'Movement_RelativeRMS_mean': row['Movement_RelativeRMS_mean'],
        })

    report.append(f"Computed metrics for {len(results)} subjects")

    network_df = pd.DataFrame(results)

    # Residualise brain metrics for age, sex, ICV, and head motion so the
    # heteroscedasticity tests reflect PGS-related variance beyond demographic
    # and motion confounds. Matches C1's covariate handling; keep raw values
    # under *_raw for audit.
    network_df['modularity_raw'] = network_df['modularity']
    network_df['global_efficiency_raw'] = network_df['global_efficiency']
    network_df[['modularity', 'global_efficiency']] = regress_out_covariates(
        network_df[['modularity', 'global_efficiency']],
        network_df[list(COVARIATES)],
    )
    report.append(f"Residualised modularity & global_efficiency for: "
                  f"{', '.join(COVARIATES)}")

    return network_df


# %%
# =============================================================================
# 3. HETEROSCEDASTICITY ANALYSES
# =============================================================================

def test_breusch_pagan(df, report):
    """
    Breusch-Pagan test for heteroscedasticity.

    Tests whether residual variance from regressing brain metric on PGS
    is related to PGS level.
    """
    report.append("")
    report.append("=" * 60)
    report.append("BREUSCH-PAGAN TEST FOR HETEROSCEDASTICITY")
    report.append("=" * 60)

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")
        report.append(f"  OLS: {metric} ~ pgs_z")

        y = df[metric].values
        X = sm.add_constant(df['pgs_z'].values)

        # Fit OLS
        ols_model = sm.OLS(y, X).fit()
        residuals = ols_model.resid

        # Breusch-Pagan test
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X)

        report.append(f"  Lagrange Multiplier stat = {lm_stat:.3f}, p = {lm_pvalue:.3e}")
        report.append(f"  F-statistic = {f_stat:.3f}, p = {f_pvalue:.3e}")

        if lm_pvalue < 0.05:
            report.append(f"  *** Significant heteroscedasticity detected (p < 0.05)")
        else:
            report.append(f"  No significant heteroscedasticity (p >= 0.05)")

        results[metric] = {
            'lm_stat': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'residuals': residuals,
            'ols_model': ols_model
        }

    return results


def test_white(df, report):
    """
    White test for heteroscedasticity (general form).

    More general than Breusch-Pagan: tests for heteroscedasticity without
    assuming a specific functional form.
    """
    report.append("")
    report.append("=" * 60)
    report.append("WHITE TEST FOR HETEROSCEDASTICITY")
    report.append("=" * 60)

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")
        report.append(f"  OLS: {metric} ~ pgs_z")

        y = df[metric].values
        X = sm.add_constant(df['pgs_z'].values)

        # Fit OLS
        ols_model = sm.OLS(y, X).fit()
        residuals = ols_model.resid

        # White test
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(residuals, X)

        report.append(f"  Lagrange Multiplier stat = {lm_stat:.3f}, p = {lm_pvalue:.3e}")
        report.append(f"  F-statistic = {f_stat:.3f}, p = {f_pvalue:.3e}")

        if lm_pvalue < 0.05:
            report.append(f"  *** Significant heteroscedasticity detected (p < 0.05)")
        else:
            report.append(f"  No significant heteroscedasticity (p >= 0.05)")

        results[metric] = {
            'lm_stat': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue
        }

    return results


def test_quantile_regression(df, report):
    """
    Quantile regression at multiple quantiles.

    Fits quantile regression of brain metric on PGS at quantiles
    [0.1, 0.25, 0.5, 0.75, 0.9]. If PGS affects variance, the slopes
    at different quantiles will diverge (fan-shape pattern).
    """
    report.append("")
    report.append("=" * 60)
    report.append("QUANTILE REGRESSION ANALYSIS")
    report.append("=" * 60)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    results = {}

    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")
        report.append(f"  {'Quantile':<10} {'Slope':<10} {'95% CI':<25} {'p-value':<10}")
        report.append(f"  {'-'*55}")

        y = df[metric].values
        X = sm.add_constant(df['pgs_z'].values)

        slopes = []
        intercepts = []
        slope_cis = []
        pvalues = []
        slope_ses = []

        for q in quantiles:
            qr_model = QuantReg(y, X).fit(q=q)
            slope = qr_model.params[1]
            intercept = qr_model.params[0]
            ci = qr_model.conf_int()[1]  # CI for slope
            pval = qr_model.pvalues[1]
            se = qr_model.bse[1]

            slopes.append(slope)
            intercepts.append(intercept)
            slope_cis.append((ci[0], ci[1]))
            pvalues.append(pval)
            slope_ses.append(se)

            report.append(f"  {q:<10.2f} {slope:<10.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
                         f"{'':>5} {pval:<10.4f}")

        # Slope range
        slope_range = max(slopes) - min(slopes)
        report.append("")
        report.append(f"  Slope range (max - min): {slope_range:.4f}")

        # Interquantile test (Q90 vs Q10)
        slope_diff = slopes[-1] - slopes[0]  # Q90 - Q10
        se_diff = np.sqrt(slope_ses[-1]**2 + slope_ses[0]**2)
        iq_tstat = slope_diff / se_diff if se_diff > 0 else 0.0
        # Two-sided p-value from normal approximation
        iq_pvalue = 2 * (1 - norm.cdf(abs(iq_tstat)))

        report.append(f"  Interquantile test (Q0.9 vs Q0.1):")
        report.append(f"    Slope difference = {slope_diff:.4f}")
        report.append(f"    t = {iq_tstat:.3f}, p = {iq_pvalue:.4f}")

        if iq_pvalue < 0.05:
            report.append(f"  *** Fan-shape pattern detected: slopes diverge across quantiles")
        else:
            report.append(f"  No significant slope divergence across quantiles")

        results[metric] = {
            'quantiles': quantiles,
            'slopes': slopes,
            'intercepts': intercepts,
            'slope_cis': slope_cis,
            'pvalues': pvalues,
            'slope_ses': slope_ses,
            'slope_range': slope_range,
            'interquantile_test': {
                'slope_diff': slope_diff,
                't_stat': iq_tstat,
                'p_value': iq_pvalue
            }
        }

    return results


def compute_decile_variability(df, report, n_bootstrap=1000):
    """
    Compute variability of brain metrics within PGS deciles with bootstrap CIs.

    Splits PGS into 10 equal-sized bins (deciles) and computes the SD and IQR
    of modularity and global efficiency within each decile. Bootstrap resampling
    within each decile provides confidence intervals for the SD estimates.
    """
    report.append("")
    report.append("=" * 60)
    report.append("DECILE VARIABILITY ANALYSIS (WITH BOOTSTRAP CIs)")
    report.append("=" * 60)

    # Create decile bins
    df = df.copy()
    df['pgs_decile'] = pd.qcut(df['pgs_z'], q=10, labels=False) + 1

    results = {}
    np.random.seed(42)

    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")
        report.append(f"  {'Decile':<8} {'N':<6} {'PGS center':<12} {'SD':<10} "
                     f"{'SD 95% CI':<22} {'IQR':<10}")
        report.append(f"  {'-'*68}")

        decile_labels = []
        decile_centers = []
        sds = []
        sd_cis = []
        iqrs = []
        n_per_decile = []

        for d in range(1, 11):
            decile_data = df[df['pgs_decile'] == d]
            values = decile_data[metric].values
            pgs_center = decile_data['pgs_z'].mean()

            sd = np.std(values, ddof=1)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            n = len(values)

            # Bootstrap CI for SD
            boot_sds = []
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(values, size=n, replace=True)
                boot_sds.append(np.std(boot_sample, ddof=1))
            ci_lower, ci_upper = np.percentile(boot_sds, [2.5, 97.5])

            decile_labels.append(d)
            decile_centers.append(pgs_center)
            sds.append(sd)
            sd_cis.append((ci_lower, ci_upper))
            iqrs.append(iqr)
            n_per_decile.append(n)

            report.append(f"  {d:<8} {n:<6} {pgs_center:<12.3f} {sd:<10.4f} "
                         f"[{ci_lower:.4f}, {ci_upper:.4f}]  {iqr:<10.4f}")

        # Linear trend test
        trend_r, trend_p = pearsonr(decile_centers, sds)
        report.append("")
        report.append(f"  Linear trend (SD ~ PGS decile center):")
        report.append(f"    r = {trend_r:.3f}, p = {trend_p:.4f}")

        if trend_p < 0.05 and trend_r > 0:
            report.append(f"  *** Increasing variability with PGS - supports landscape theory")
        elif trend_p < 0.05 and trend_r < 0:
            report.append(f"  *** Decreasing variability with PGS")
        else:
            report.append(f"  No significant linear trend in variability")

        results[metric] = {
            'decile_labels': decile_labels,
            'decile_centers': decile_centers,
            'sds': sds,
            'sd_cis': sd_cis,
            'iqrs': iqrs,
            'n_per_decile': n_per_decile,
            'trend_r': trend_r,
            'trend_p': trend_p
        }

    return results


def test_bootstrap_heteroscedasticity(df, report, n_bootstrap=1000, n_bins=5):
    """
    Bootstrap-based heteroscedasticity test with PGS-balanced resampling.

    Addresses the problem of unequal sample sizes across the PGS range by:
    1. Dividing PGS into equal-N quantile bins (ensuring adequate sample per bin)
    2. Resampling an equal number of subjects from each bin per iteration
    3. Computing the variance of the brain metric within each bin
    4. Testing whether variance increases with PGS bin

    This ensures that the extremes of the PGS distribution contribute
    equally to the test, avoiding the dominance of middle-range subjects.
    """
    report.append("")
    report.append("=" * 60)
    report.append("BOOTSTRAP HETEROSCEDASTICITY TEST (PGS-BALANCED)")
    report.append("=" * 60)
    report.append(f"  N bootstrap iterations: {n_bootstrap}")
    report.append(f"  N PGS bins: {n_bins}")

    results = {}
    np.random.seed(42)

    # Create equal-N quantile bins (each bin has ~N/n_bins subjects)
    df = df.copy()
    df['pgs_bin'] = pd.qcut(df['pgs_z'], q=n_bins, labels=False)

    # Compute bin edges from actual quantile boundaries
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = df['pgs_z'].min()
    bin_edges[-1] = df['pgs_z'].max()
    for b in range(1, n_bins):
        bin_edges[b] = df[df['pgs_bin'] == b]['pgs_z'].min()

    # Get bin sizes (should be approximately equal)
    bin_sizes = df.groupby('pgs_bin').size()
    min_bin_size = bin_sizes.min()

    report.append("")
    report.append("  PGS bin sizes (quantile-based, approximately equal):")
    for b in range(n_bins):
        n = bin_sizes.get(b, 0)
        bin_data = df[df['pgs_bin'] == b]['pgs_z']
        pgs_range = f"[{bin_data.min():.2f}, {bin_data.max():.2f}]"
        report.append(f"    Bin {b+1} {pgs_range}: n = {n}")
    report.append(f"  Balanced sample size per bin: {min_bin_size}")

    # Compute bin centers (mean PGS per bin)
    bin_centers = []
    for b in range(n_bins):
        bin_centers.append(df[df['pgs_bin'] == b]['pgs_z'].mean())

    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")

        # Observed variance per bin
        observed_vars = []
        for b in range(n_bins):
            bin_data = df[df['pgs_bin'] == b][metric].values
            observed_vars.append(np.var(bin_data, ddof=1))

        # Observed trend: correlation between bin center and variance
        observed_trend_r, _ = pearsonr(bin_centers, observed_vars)

        # Bootstrap: resample equal numbers from each bin
        boot_trends = []
        boot_var_ratios = []
        for _ in range(n_bootstrap):
            boot_vars = []
            for b in range(n_bins):
                bin_data = df[df['pgs_bin'] == b][metric].values
                boot_sample = np.random.choice(bin_data, size=min_bin_size,
                                               replace=True)
                boot_vars.append(np.var(boot_sample, ddof=1))

            # Trend: correlation between bin center and variance
            trend_r, _ = pearsonr(bin_centers, boot_vars)
            boot_trends.append(trend_r)

            # Variance ratio: highest bin / lowest bin
            var_ratio = boot_vars[-1] / boot_vars[0] if boot_vars[0] > 0 else np.nan
            boot_var_ratios.append(var_ratio)

        # Bootstrap CI for trend
        trend_ci = np.percentile(boot_trends, [2.5, 97.5])
        # p-value: proportion of bootstrap samples with trend <= 0
        trend_p = np.mean(np.array(boot_trends) <= 0)

        # Bootstrap CI for variance ratio
        var_ratio_ci = np.nanpercentile(boot_var_ratios, [2.5, 97.5])
        median_var_ratio = np.nanmedian(boot_var_ratios)

        report.append(f"  Observed variance per bin:")
        for b in range(n_bins):
            report.append(f"    Bin {b+1}: var = {observed_vars[b]:.6f}")

        report.append(f"  Observed trend (var ~ bin): r = {observed_trend_r:.3f}")
        report.append(f"  Bootstrap trend:")
        report.append(f"    95% CI: [{trend_ci[0]:.3f}, {trend_ci[1]:.3f}]")
        report.append(f"    p-value (one-tailed, H1: increasing): {trend_p:.4f}")
        report.append(f"  Variance ratio (highest/lowest PGS bin):")
        report.append(f"    Median: {median_var_ratio:.3f}")
        report.append(f"    95% CI: [{var_ratio_ci[0]:.3f}, {var_ratio_ci[1]:.3f}]")

        if trend_p < 0.05:
            report.append(f"  *** Significant increasing variance with PGS (balanced bootstrap)")
        else:
            report.append(f"  No significant variance trend (balanced bootstrap)")

        results[metric] = {
            'observed_vars': observed_vars,
            'observed_trend_r': observed_trend_r,
            'boot_trend_ci': (trend_ci[0], trend_ci[1]),
            'boot_trend_p': trend_p,
            'median_var_ratio': median_var_ratio,
            'var_ratio_ci': (var_ratio_ci[0], var_ratio_ci[1]),
            'bin_centers': bin_centers,
            'min_bin_size': min_bin_size
        }

    return results


# %%
# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def create_main_figure(df, bp_results, white_results, qr_results,
                       decile_results, bootstrap_results, figures_dir):
    """Create 8-panel figure: 4 rows x 2 columns (modularity | global_efficiency)."""

    fig, axes = plt.subplots(4, 2, figsize=(280 * mm2inches, 320 * mm2inches), dpi=FIGURE_DPI)

    metrics = ['modularity', 'global_efficiency']
    metric_labels = ['Modularity (residualised)',
                     'Global Efficiency (residualised)']

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):

        # --- Row 1: Residual scatter with LOWESS ---
        ax = axes[0, col]
        residuals = bp_results[metric]['residuals']
        pgs_z = df['pgs_z'].values

        ax.scatter(pgs_z, residuals, alpha=0.3, s=10, color='gray', edgecolors='none')

        # LOWESS of |residuals| to show variance trend
        abs_resid = np.abs(residuals)
        lowess_result = lowess(abs_resid, pgs_z, frac=0.4)
        ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'k-', linewidth=2,
                label='LOWESS |residuals|')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Annotations
        bp_p = bp_results[metric]['lm_pvalue']
        wh_p = white_results[metric]['lm_pvalue']
        ax.text(0.05, 0.95, f'BP p = {bp_p:.3e}\nWhite p = {wh_p:.3e}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow' if bp_p < 0.05 else 'white',
                         alpha=0.8))

        ax.set_xlabel('PGS (z-scored)')
        ax.set_ylabel('OLS Residuals')
        ax.set_title(f'{label}: Residuals vs PGS')
        ax.legend(fontsize=7, loc='upper right')

        # --- Row 2: Quantile regression fan plot ---
        ax = axes[1, col]
        ax.scatter(pgs_z, df[metric].values, alpha=0.2, s=8, color='gray',
                  edgecolors='none')

        quantiles = qr_results[metric]['quantiles']
        slopes = qr_results[metric]['slopes']
        intercepts = qr_results[metric]['intercepts']
        cmap = plt.cm.coolwarm
        x_range = np.linspace(pgs_z.min(), pgs_z.max(), 100)

        for i, (q, slope, intercept) in enumerate(zip(quantiles, slopes, intercepts)):
            color = cmap(i / (len(quantiles) - 1))
            y_pred = intercept + slope * x_range
            ax.plot(x_range, y_pred, color=color, linewidth=1.5,
                   label=f'Q{q:.2f} (b={slope:.3f})')

        ax.set_xlabel('PGS (z-scored)')
        ax.set_ylabel(label)
        ax.set_title(f'{label}: Quantile Regression')
        ax.legend(fontsize=6, loc='upper left')

        # Annotation
        iq_p = qr_results[metric]['interquantile_test']['p_value']
        slope_range = qr_results[metric]['slope_range']
        ax.text(0.95, 0.05, f'Slope range = {slope_range:.4f}\nIQ p = {iq_p:.4f}',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round',
                         facecolor='yellow' if iq_p < 0.05 else 'white', alpha=0.8))

        # --- Row 3: Decile variability bar plot with bootstrap CIs ---
        ax = axes[2, col]
        decile_labels = decile_results[metric]['decile_labels']
        sds = decile_results[metric]['sds']
        sd_cis = decile_results[metric]['sd_cis']

        # Color gradient from blue (low PGS) to red (high PGS)
        cmap_decile = plt.cm.coolwarm
        colors = [cmap_decile(i / 9) for i in range(10)]

        # Error bars from bootstrap CIs
        ci_lower = [sd - ci[0] for sd, ci in zip(sds, sd_cis)]
        ci_upper = [ci[1] - sd for sd, ci in zip(sds, sd_cis)]

        bars = ax.bar(decile_labels, sds, color=colors, alpha=0.8, edgecolor='white',
                     linewidth=0.5, yerr=[ci_lower, ci_upper], capsize=3,
                     error_kw={'linewidth': 0.8, 'color': 'black'})

        # Linear trend line
        z_fit = np.polyfit(decile_labels, sds, 1)
        p_fit = np.poly1d(z_fit)
        ax.plot(decile_labels, p_fit(decile_labels), 'k--', linewidth=1.5,
               label='Linear trend')

        ax.set_xlabel('PGS Decile')
        ax.set_ylabel(f'{label} SD')
        ax.set_title(f'{label}: Variability by Decile (bootstrap CI)')
        ax.set_xticks(decile_labels)

        # Annotation
        trend_r = decile_results[metric]['trend_r']
        trend_p = decile_results[metric]['trend_p']
        ax.text(0.05, 0.95, f'r = {trend_r:.3f}, p = {trend_p:.4f}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round',
                         facecolor='yellow' if trend_p < 0.05 else 'white', alpha=0.8))
        ax.legend(fontsize=7)

        # --- Row 4: Bootstrap balanced variance by PGS bin ---
        ax = axes[3, col]
        n_bins = len(bootstrap_results[metric]['observed_vars'])
        bin_centers = bootstrap_results[metric]['bin_centers']
        observed_vars = bootstrap_results[metric]['observed_vars']

        # Bin labels as PGS center values
        bin_labels = [f'{c:.2f}' for c in bin_centers]
        bin_positions = range(1, n_bins + 1)

        cmap_bins = plt.cm.coolwarm
        bin_colors = [cmap_bins(i / (n_bins - 1)) for i in range(n_bins)]

        ax.bar(bin_positions, observed_vars, color=bin_colors, alpha=0.8,
              edgecolor='white', linewidth=0.5)

        # Trend line
        z_fit = np.polyfit(list(bin_positions), observed_vars, 1)
        p_fit = np.poly1d(z_fit)
        ax.plot(bin_positions, p_fit(list(bin_positions)), 'k--', linewidth=1.5,
               label='Linear trend')

        ax.set_xlabel('PGS Quintile (center z-score)')
        ax.set_ylabel(f'{label} Variance')
        ax.set_title(f'{label}: Balanced Bootstrap\n(n={bootstrap_results[metric]["min_bin_size"]}/bin)')
        ax.set_xticks(list(bin_positions))
        ax.set_xticklabels(bin_labels, fontsize=7)

        # Annotation
        boot_p = bootstrap_results[metric]['boot_trend_p']
        boot_r = bootstrap_results[metric]['observed_trend_r']
        var_ratio = bootstrap_results[metric]['median_var_ratio']
        ax.text(0.05, 0.95,
                f'r = {boot_r:.3f}\nBoot p = {boot_p:.4f}\nVar ratio = {var_ratio:.2f}x',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round',
                         facecolor='yellow' if boot_p < 0.05 else 'white', alpha=0.8))
        ax.legend(fontsize=7)

    plt.tight_layout()
    output_file = figures_dir / 'C3b_continuous_heteroscedasticity.png'
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to: {output_file}")

    return fig


# %%
# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete continuous heteroscedasticity analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Continuous heteroscedasticity analysis testing landscape theory'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--pgs', required=False,
                        help='Path to PGS/PGS residuals CSV')
    parser.add_argument('--social', required=False,
                        help='Path to social factor scores CSV')
    parser.add_argument('--behavioural', required=False,
                        help='Path to behavioural data CSV')
    parser.add_argument('--phenotypic', required=False,
                        help='Path to phenotypic data CSV')
    parser.add_argument('--movement', required=False,
                        help='Path to movement data CSV')
    parser.add_argument('--ids', required=False,
                        help='Path to subject IDs file')
    parser.add_argument('--matrices-dir', required=False,
                        help='Path to connectivity matrices directory')
    parser.add_argument('--partition', required=True,
                        help='Path to community partition CSV (from C2b). '
                             'n_nodes is derived from the number of rows.')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Network density threshold (default: 0.2)')
    parser.add_argument('--motion-threshold', type=float, default=0.2,
                        help='Motion threshold for subject exclusion (default: 0.2)')
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'

    for dir_path in [figures_dir, reports_dir, results_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = [
        "=" * 80,
        "C4b: CONTINUOUS HETEROSCEDASTICITY ANALYSIS - LANDSCAPE THEORY",
        "=" * 80,
        ""
    ]

    # Set up file paths with defaults
    pgs_file = (Path(args.pgs) if args.pgs
                else project_folder / "data/pgs_residuals.csv")
    social_file = (Path(args.social) if args.social
                   else project_folder / 'data/cfa_factor_scores_full_sample.csv')
    behavioural_file = (Path(args.behavioural) if args.behavioural
                        else project_folder / "data/behavioural_data_anonymised.csv")
    phenotypic_file = (Path(args.phenotypic) if args.phenotypic
                       else project_folder / "data/phenotypic_data_anonymised.csv")
    movement_file = (Path(args.movement) if args.movement
                     else project_folder / 'data/movement_data_anonymised.csv')
    id_file = (Path(args.ids) if args.ids
               else project_folder / 'data/subjectIDs_anonymised.txt')
    matrices_dir = (Path(args.matrices_dir) if args.matrices_dir
                    else project_folder / 'data/HCP_PTN1200/netmats')
    partition_file = Path(args.partition)

    # Parcellation size is inherited from the C2b-selected partition
    n_nodes = len(pd.read_csv(partition_file))

    report.append(f"Project folder: {project_folder}")
    report.append(f"N nodes (from C2b partition): {n_nodes}")
    report.append(f"Network threshold: {args.threshold}")
    report.append(f"Motion threshold: {args.motion_threshold}")
    report.append(f"Brain metrics residualised for: {', '.join(COVARIATES)}")
    report.append("  (applied before all heteroscedasticity tests)")
    report.append("")

    # Load data
    merged_df = load_and_prepare_data(
        project_folder, pgs_file, social_file, behavioural_file,
        phenotypic_file, movement_file, id_file, matrices_dir,
        n_nodes, args.motion_threshold, report
    )

    # Calculate network metrics
    network_df = calculate_network_metrics(
        merged_df, partition_file, n_nodes, args.threshold, report
    )

    # Run heteroscedasticity analyses (on full sample)
    bp_results = test_breusch_pagan(network_df, report)
    white_results = test_white(network_df, report)
    qr_results = test_quantile_regression(network_df, report)
    decile_results = compute_decile_variability(network_df, report)
    bootstrap_results = test_bootstrap_heteroscedasticity(network_df, report)

    # Create visualization
    create_main_figure(network_df, bp_results, white_results,
                       qr_results, decile_results, bootstrap_results, figures_dir)

    # Final summary
    report.append("")
    report.append("=" * 80)
    report.append("FINAL SUMMARY")
    report.append("=" * 80)

    report.append("")
    report.append("Breusch-Pagan test:")
    for metric in ['modularity', 'global_efficiency']:
        p = bp_results[metric]['lm_pvalue']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: LM p = {p:.3e} [{sig}]")

    report.append("")
    report.append("White test:")
    for metric in ['modularity', 'global_efficiency']:
        p = white_results[metric]['lm_pvalue']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: LM p = {p:.3e} [{sig}]")

    report.append("")
    report.append("Quantile regression slope divergence:")
    for metric in ['modularity', 'global_efficiency']:
        sr = qr_results[metric]['slope_range']
        p = qr_results[metric]['interquantile_test']['p_value']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: slope range = {sr:.4f}, IQ p = {p:.4f} [{sig}]")

    report.append("")
    report.append("Decile SD trend:")
    for metric in ['modularity', 'global_efficiency']:
        r = decile_results[metric]['trend_r']
        p = decile_results[metric]['trend_p']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: r = {r:.3f}, p = {p:.4f} [{sig}]")

    report.append("")
    report.append("Bootstrap balanced heteroscedasticity:")
    for metric in ['modularity', 'global_efficiency']:
        p = bootstrap_results[metric]['boot_trend_p']
        vr = bootstrap_results[metric]['median_var_ratio']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: boot p = {p:.4f}, var ratio = {vr:.3f} [{sig}]")

    # Landscape theory assessment
    report.append("")
    report.append("*** LANDSCAPE THEORY ASSESSMENT ***")

    mod_heteroscedastic = (bp_results['modularity']['lm_pvalue'] < 0.05 or
                           white_results['modularity']['lm_pvalue'] < 0.05 or
                           qr_results['modularity']['interquantile_test']['p_value'] < 0.05 or
                           bootstrap_results['modularity']['boot_trend_p'] < 0.05)
    eff_heteroscedastic = (bp_results['global_efficiency']['lm_pvalue'] < 0.05 or
                           white_results['global_efficiency']['lm_pvalue'] < 0.05 or
                           qr_results['global_efficiency']['interquantile_test']['p_value'] < 0.05 or
                           bootstrap_results['global_efficiency']['boot_trend_p'] < 0.05)

    if mod_heteroscedastic and not eff_heteroscedastic:
        report.append("SUPPORTED WITH SPECIFICITY: Modularity shows heteroscedasticity")
        report.append("with PGS but global efficiency does not.")
        report.append("Compensation occurs through modular reorganization while")
        report.append("preserving global integration.")
    elif mod_heteroscedastic and eff_heteroscedastic:
        report.append("PARTIALLY SUPPORTED: Both metrics show heteroscedasticity.")
        report.append("High genetic risk increases variability broadly, not specific")
        report.append("to modular organization.")
    elif not mod_heteroscedastic and not eff_heteroscedastic:
        report.append("NOT SUPPORTED: No evidence for heteroscedasticity in either metric.")
        report.append("Variance of brain network organization does not change with PGS.")
    else:
        report.append("UNEXPECTED PATTERN: Global efficiency shows heteroscedasticity")
        report.append("but modularity does not.")

    # Save results
    results_file = results_dir / 'C3b_heteroscedasticity_results.csv'
    network_df.to_csv(results_file, index=False)
    report.append("")
    report.append(f"Results saved to: {results_file}")

    # Save quantile regression coefficients
    qr_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for i, q in enumerate(qr_results[metric]['quantiles']):
            qr_rows.append({
                'metric': metric,
                'quantile': q,
                'slope': qr_results[metric]['slopes'][i],
                'intercept': qr_results[metric]['intercepts'][i],
                'slope_lower_ci': qr_results[metric]['slope_cis'][i][0],
                'slope_upper_ci': qr_results[metric]['slope_cis'][i][1],
                'pvalue': qr_results[metric]['pvalues'][i]
            })
    qr_coefs_file = results_dir / 'C3b_quantile_regression_coefficients.csv'
    pd.DataFrame(qr_rows).to_csv(qr_coefs_file, index=False)
    report.append(f"Quantile regression coefficients saved to: {qr_coefs_file}")

    # Save decile variability
    decile_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for i in range(10):
            decile_rows.append({
                'metric': metric,
                'decile': decile_results[metric]['decile_labels'][i],
                'n': decile_results[metric]['n_per_decile'][i],
                'pgs_center': decile_results[metric]['decile_centers'][i],
                'sd': decile_results[metric]['sds'][i],
                'sd_ci_lower': decile_results[metric]['sd_cis'][i][0],
                'sd_ci_upper': decile_results[metric]['sd_cis'][i][1],
                'iqr': decile_results[metric]['iqrs'][i]
            })
    decile_file = results_dir / 'C3b_decile_variability.csv'
    pd.DataFrame(decile_rows).to_csv(decile_file, index=False)
    report.append(f"Decile variability saved to: {decile_file}")

    # Write report to file
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_file = reports_dir / 'C3b_continuous_heteroscedasticity_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_file}")

    return network_df, bp_results, white_results, qr_results, decile_results, bootstrap_results


# %%
if __name__ == "__main__":
    main()

# %%
