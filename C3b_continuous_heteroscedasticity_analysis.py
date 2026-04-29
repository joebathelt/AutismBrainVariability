# %%
"""
C3b_continuous_heteroscedasticity_analysis.py
Continuous heteroscedasticity analysis on SDS (Social_Score)

Tests whether the variance of brain network organisation (modularity, global
efficiency, residualised for age/ICV/motion) varies continuously with the
Social Difficulty Score (SDS = Social_Score, where higher z = worse social
performance):

  1. Breusch-Pagan
  2. White
  3. Quantile regression at q in {.1, .25, .5, .75, .9}, plus interquantile slope test
  4. Within-decile SD trend with bootstrap CIs
  5. Balanced-bootstrap heteroscedasticity (equal-N quintiles)
  6. DGLM (joint mean + log-variance MLE) with SDS in the variance submodel

PGS is intentionally not loaded; the SDS~PGS test is in C6.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore, pearsonr, norm, chi2
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.nonparametric.smoothers_lowess import lowess
import networkx as nx
import bct

from utils.covariates import COVARIATES, regress_out_covariates

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

PREDICTOR = 'sds_z'  # used as x in all tests below


# %%
# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_and_prepare_data(project_folder, social_file, behavioural_file,
                          phenotypic_file, movement_file, id_file, matrices_dir,
                          n_nodes, motion_threshold, report):
    msg = "Loading and preparing data..."
    print(msg); report.append(msg)

    matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
    mats_df = pd.read_csv(matrix_file, header=None, sep=r'\s+')
    mats_df.columns = [f'conn_{i+1}' for i in range(mats_df.shape[1])]
    ids = pd.read_csv(id_file, header=None)[0].tolist()
    mats_df.index = ids
    report.append(f"Loaded connectivity matrices: {mats_df.shape}")

    social_df = pd.read_csv(social_file)
    behavioural_df = pd.read_csv(behavioural_file)
    # No sex filter: Gender is residualised out via COVARIATES.
    phenotypic_df = pd.read_csv(phenotypic_file)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    movement_df = pd.read_csv(movement_file)

    merged_df = pd.merge(social_df[['Subject', 'Social_Score']], mats_df,
                         left_on='Subject', right_index=True)
    merged_df = pd.merge(merged_df,
                         behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']],
                         on='Subject')
    merged_df = pd.merge(merged_df,
                         phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_df = pd.merge(merged_df,
                         movement_df[['Subject', 'Movement_RelativeRMS_mean']],
                         on='Subject')

    report.append(f"Merged data shape: {merged_df.shape}")
    merged_df = merged_df[merged_df['Movement_RelativeRMS_mean'] < motion_threshold]
    merged_df = merged_df.dropna()

    msg = f"Final sample after QC: {len(merged_df)} subjects"
    print(msg); report.append(msg)

    merged_df['sds_z'] = zscore(merged_df['Social_Score'])
    return merged_df


# %%
# =============================================================================
# 2. NETWORK METRICS
# =============================================================================

def calculate_network_metrics(merged_df, partition_file, n_nodes, threshold, report):
    msg = f"\nCalculating network metrics ({n_nodes} nodes, {threshold*100:.0f}% threshold)..."
    print(msg); report.append(""); report.append(msg)

    if partition_file and partition_file.exists():
        partition_df = pd.read_csv(partition_file)
        report.append(f"Loaded partition from: {partition_file}")
    else:
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    results = []
    n_subjects = len(merged_df)

    for i, (_, row) in enumerate(merged_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_subjects} subjects")

        conn_data = row[[c for c in row.index if c.startswith('conn_')]].values
        mat = np.reshape(conn_data, (n_nodes, n_nodes)).astype(np.float64) / 100
        mat = bct.threshold_proportional(mat, threshold)
        mat = np.nan_to_num(mat, nan=0.0)

        _, modularity = bct.modularity_und_sign(mat, partition_df['community_id'].values)

        mat_pos = mat.copy()
        mat_pos[mat_pos < 0] = 0
        global_efficiency = nx.global_efficiency(nx.from_numpy_array(mat_pos))

        results.append({
            'Subject': row['Subject'],
            'modularity': modularity,
            'global_efficiency': global_efficiency,
            'Social_Score': row['Social_Score'],
            'sds_z': row['sds_z'],
            'Age_in_Yrs': row['Age_in_Yrs'],
            'FS_IntraCranial_Vol': row['FS_IntraCranial_Vol'],
            'Movement_RelativeRMS_mean': row['Movement_RelativeRMS_mean'],
            'Gender': row['Gender'],
        })

    network_df = pd.DataFrame(results)
    report.append(f"Computed metrics for {len(network_df)} subjects")

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
# 3. HETEROSCEDASTICITY TESTS
# =============================================================================

def test_breusch_pagan(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("BREUSCH-PAGAN TEST FOR HETEROSCEDASTICITY")
    report.append("=" * 60)

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        report.append(f"  OLS: {metric} ~ {PREDICTOR}")
        y = df[metric].values
        X = sm.add_constant(df[PREDICTOR].values)
        ols_model = sm.OLS(y, X).fit()
        residuals = ols_model.resid
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X)
        report.append(f"  LM stat = {lm_stat:.3f}, p = {lm_pvalue:.3e}")
        report.append(f"  F-stat  = {f_stat:.3f}, p = {f_pvalue:.3e}")
        report.append("  *** Heteroscedasticity (p<0.05)" if lm_pvalue < 0.05 else
                      "  No heteroscedasticity")
        results[metric] = {
            'lm_stat': lm_stat, 'lm_pvalue': lm_pvalue,
            'f_stat': f_stat, 'f_pvalue': f_pvalue,
            'residuals': residuals, 'ols_model': ols_model,
        }
    return results


def test_white(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("WHITE TEST FOR HETEROSCEDASTICITY")
    report.append("=" * 60)

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        report.append(f"  OLS: {metric} ~ {PREDICTOR}")
        y = df[metric].values
        X = sm.add_constant(df[PREDICTOR].values)
        ols_model = sm.OLS(y, X).fit()
        residuals = ols_model.resid
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(residuals, X)
        report.append(f"  LM stat = {lm_stat:.3f}, p = {lm_pvalue:.3e}")
        report.append(f"  F-stat  = {f_stat:.3f}, p = {f_pvalue:.3e}")
        report.append("  *** Heteroscedasticity (p<0.05)" if lm_pvalue < 0.05 else
                      "  No heteroscedasticity")
        results[metric] = {
            'lm_stat': lm_stat, 'lm_pvalue': lm_pvalue,
            'f_stat': f_stat, 'f_pvalue': f_pvalue,
        }
    return results


def test_quantile_regression(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("QUANTILE REGRESSION ANALYSIS")
    report.append("=" * 60)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        report.append(f"  {'Quantile':<10} {'Slope':<10} {'95% CI':<25} {'p-value':<10}")
        report.append(f"  {'-'*55}")
        y = df[metric].values
        X = sm.add_constant(df[PREDICTOR].values)
        slopes, intercepts, slope_cis, pvalues, slope_ses = [], [], [], [], []
        for q in quantiles:
            qr = QuantReg(y, X).fit(q=q)
            slopes.append(qr.params[1]); intercepts.append(qr.params[0])
            ci = qr.conf_int()[1]; slope_cis.append((ci[0], ci[1]))
            pvalues.append(qr.pvalues[1]); slope_ses.append(qr.bse[1])
            report.append(f"  {q:<10.2f} {qr.params[1]:<10.4f} "
                          f"[{ci[0]:.4f}, {ci[1]:.4f}]{'':>5} {qr.pvalues[1]:<10.4f}")

        slope_range = max(slopes) - min(slopes)
        slope_diff = slopes[-1] - slopes[0]
        se_diff = np.sqrt(slope_ses[-1]**2 + slope_ses[0]**2)
        iq_t = slope_diff / se_diff if se_diff > 0 else 0.0
        iq_p = 2 * (1 - norm.cdf(abs(iq_t)))
        report.append(""); report.append(f"  Slope range: {slope_range:.4f}")
        report.append(f"  Interquantile (Q0.9 vs Q0.1): diff = {slope_diff:.4f}, "
                      f"t = {iq_t:.3f}, p = {iq_p:.4f}")
        report.append("  *** Fan-shape (p<0.05)" if iq_p < 0.05 else
                      "  No significant slope divergence")

        results[metric] = {
            'quantiles': quantiles, 'slopes': slopes, 'intercepts': intercepts,
            'slope_cis': slope_cis, 'pvalues': pvalues, 'slope_ses': slope_ses,
            'slope_range': slope_range,
            'interquantile_test': {'slope_diff': slope_diff, 't_stat': iq_t,
                                   'p_value': iq_p},
        }
    return results


def compute_decile_variability(df, report, n_bootstrap=1000):
    report.append("")
    report.append("=" * 60)
    report.append("DECILE VARIABILITY ANALYSIS (with bootstrap CIs)")
    report.append("=" * 60)

    df = df.copy()
    df['sds_decile'] = pd.qcut(df[PREDICTOR], q=10, labels=False) + 1

    results = {}
    np.random.seed(42)
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        report.append(f"  {'Decile':<8} {'N':<6} {'SDS center':<12} {'SD':<10} "
                      f"{'SD 95% CI':<22} {'IQR':<10}")
        report.append(f"  {'-'*68}")
        decile_labels, decile_centers, sds, sd_cis, iqrs, n_per = [], [], [], [], [], []
        for d in range(1, 11):
            decile_data = df[df['sds_decile'] == d]
            values = decile_data[metric].values
            sds_center = decile_data[PREDICTOR].mean()
            sd = np.std(values, ddof=1)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            n = len(values)
            boot_sds = [np.std(np.random.choice(values, size=n, replace=True), ddof=1)
                        for _ in range(n_bootstrap)]
            ci_lo, ci_hi = np.percentile(boot_sds, [2.5, 97.5])
            decile_labels.append(d); decile_centers.append(sds_center)
            sds.append(sd); sd_cis.append((ci_lo, ci_hi)); iqrs.append(iqr)
            n_per.append(n)
            report.append(f"  {d:<8} {n:<6} {sds_center:<12.3f} {sd:<10.4f} "
                          f"[{ci_lo:.4f}, {ci_hi:.4f}]  {iqr:<10.4f}")

        trend_r, trend_p = pearsonr(decile_centers, sds)
        report.append("")
        report.append(f"  Linear trend (SD ~ decile center): r = {trend_r:.3f}, "
                      f"p = {trend_p:.4f}")
        if trend_p < 0.05 and trend_r > 0:
            report.append("  *** Increasing variability with SDS - supports H1")
        elif trend_p < 0.05 and trend_r < 0:
            report.append("  *** Decreasing variability with SDS")
        else:
            report.append("  No significant linear trend")

        results[metric] = {
            'decile_labels': decile_labels, 'decile_centers': decile_centers,
            'sds': sds, 'sd_cis': sd_cis, 'iqrs': iqrs, 'n_per_decile': n_per,
            'trend_r': trend_r, 'trend_p': trend_p,
        }
    return results


def test_bootstrap_heteroscedasticity(df, report, n_bootstrap=1000, n_bins=5):
    report.append("")
    report.append("=" * 60)
    report.append("BOOTSTRAP HETEROSCEDASTICITY TEST (SDS-balanced quintiles)")
    report.append("=" * 60)
    report.append(f"  N bootstrap iterations: {n_bootstrap}")
    report.append(f"  N SDS bins: {n_bins}")

    df = df.copy()
    df['sds_bin'] = pd.qcut(df[PREDICTOR], q=n_bins, labels=False)
    bin_sizes = df.groupby('sds_bin').size()
    min_bin_size = int(bin_sizes.min())
    report.append("")
    report.append("  SDS bin sizes (quantile-based):")
    for b in range(n_bins):
        bd = df[df['sds_bin'] == b][PREDICTOR]
        report.append(f"    Bin {b+1} [{bd.min():.2f}, {bd.max():.2f}]: n = {len(bd)}")
    report.append(f"  Balanced sample size per bin: {min_bin_size}")

    bin_centers = [df[df['sds_bin'] == b][PREDICTOR].mean() for b in range(n_bins)]

    results = {}
    np.random.seed(42)
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        observed_vars = [np.var(df[df['sds_bin'] == b][metric].values, ddof=1)
                         for b in range(n_bins)]
        observed_trend_r, _ = pearsonr(bin_centers, observed_vars)

        boot_trends = np.empty(n_bootstrap)
        boot_var_ratios = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            boot_vars = []
            for b in range(n_bins):
                data = df[df['sds_bin'] == b][metric].values
                sample = np.random.choice(data, size=min_bin_size, replace=True)
                boot_vars.append(np.var(sample, ddof=1))
            boot_trends[k], _ = pearsonr(bin_centers, boot_vars)
            boot_var_ratios[k] = boot_vars[-1] / boot_vars[0] if boot_vars[0] > 0 else np.nan
        trend_ci = np.percentile(boot_trends, [2.5, 97.5])
        trend_p = float(np.mean(boot_trends <= 0))
        var_ratio_ci = np.nanpercentile(boot_var_ratios, [2.5, 97.5])
        median_var_ratio = float(np.nanmedian(boot_var_ratios))

        for b in range(n_bins):
            report.append(f"    Bin {b+1}: var = {observed_vars[b]:.6f}")
        report.append(f"  Observed trend r = {observed_trend_r:.3f}")
        report.append(f"  Bootstrap trend 95% CI: [{trend_ci[0]:.3f}, {trend_ci[1]:.3f}]")
        report.append(f"  Bootstrap one-tailed p (H1: increasing): {trend_p:.4f}")
        report.append(f"  Variance ratio (highest/lowest bin): "
                      f"median = {median_var_ratio:.3f}, "
                      f"95% CI [{var_ratio_ci[0]:.3f}, {var_ratio_ci[1]:.3f}]")
        report.append("  *** Significant increasing variance with SDS"
                      if trend_p < 0.05 else "  No significant variance trend")

        results[metric] = {
            'observed_vars': observed_vars,
            'observed_trend_r': observed_trend_r,
            'boot_trend_ci': (trend_ci[0], trend_ci[1]),
            'boot_trend_p': trend_p,
            'median_var_ratio': median_var_ratio,
            'var_ratio_ci': (var_ratio_ci[0], var_ratio_ci[1]),
            'bin_centers': bin_centers,
            'min_bin_size': min_bin_size,
        }
    return results


def fit_dglm(y_in, X_mu, X_sigma):
    """Joint mean-variance Gaussian DGLM fit by direct MLE.

    Returns the variance-coefficient on the predictor of interest (assumed
    to be the second column of X_sigma after the intercept), the LR
    statistic vs. an intercept-only variance submodel, and the LRT p-value.
    """
    y_std = float(np.std(y_in, ddof=1))
    if y_std == 0:
        return np.nan, np.nan, np.nan
    y = (y_in - np.mean(y_in)) / y_std
    p_mu = X_mu.shape[1]

    def neg_ll(params, X_s):
        bm = params[:p_mu]
        bs = params[p_mu:p_mu + X_s.shape[1]]
        log_var = np.clip(X_s @ bs, -30, 30)
        var = np.exp(log_var)
        mu = X_mu @ bm
        return 0.5 * np.sum(log_var + (y - mu) ** 2 / var)

    def fit(X_s):
        bm0 = np.linalg.lstsq(X_mu, y, rcond=None)[0]
        resid = y - X_mu @ bm0
        bs0 = np.zeros(X_s.shape[1])
        bs0[0] = float(np.log(max(np.var(resid), 1e-8)))
        x0 = np.concatenate([bm0, bs0])
        r1 = minimize(lambda p: neg_ll(p, X_s), x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        r2 = minimize(lambda p: neg_ll(p, X_s), r1.x, method='BFGS',
                      options={'maxiter': 1000, 'gtol': 1e-7})
        return r2

    full = fit(X_sigma)
    red = fit(X_sigma[:, :1])
    lr_stat = float(2 * (red.fun - full.fun))
    df_diff = X_sigma.shape[1] - 1
    p_lrt = float(chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
    alpha = float(full.x[p_mu + 1]) if X_sigma.shape[1] > 1 else np.nan
    return alpha, lr_stat, p_lrt


def test_dglm(df, report):
    """DGLM with SDS in the variance submodel (and in the mean for the
    location parameter, alongside age/ICV/motion already absorbed into the
    residualised metric — pass them as additional mean covariates for safety
    in case any residual structure remains)."""
    report.append("")
    report.append("=" * 60)
    report.append("DGLM (joint mean & log-variance, SDS in variance submodel)")
    report.append("=" * 60)

    Xmu_cols = [PREDICTOR, 'Age_in_Yrs', 'FS_IntraCranial_Vol',
                'Movement_RelativeRMS_mean']
    Xmu_df = df[Xmu_cols].copy()
    if 'Gender' in df.columns:
        Xmu_df['Gender_M'] = (df['Gender'] == 'M').astype(int)
    Xmu = sm.add_constant(Xmu_df.values.astype(float))
    Xsig = sm.add_constant(df[[PREDICTOR]].values.astype(float))

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append(""); report.append(f"{metric.upper()}:")
        try:
            alpha, lr_stat, p_lrt = fit_dglm(df[metric].values, Xmu, Xsig)
        except Exception as e:
            report.append(f"  DGLM fit failed: {e}")
            alpha, lr_stat, p_lrt = np.nan, np.nan, np.nan
        report.append(f"  alpha_SDS (log-variance coef) = {alpha:+.4f}")
        report.append(f"  LR statistic = {lr_stat:.3f}, p = {p_lrt:.4f}")
        if np.isfinite(p_lrt) and p_lrt < 0.05 and alpha > 0:
            report.append("  *** Variance increases with SDS (DGLM)")
        elif np.isfinite(p_lrt) and p_lrt < 0.05 and alpha < 0:
            report.append("  *** Variance decreases with SDS (DGLM)")
        else:
            report.append("  No significant variance-SDS effect (DGLM)")
        results[metric] = {'alpha': alpha, 'lr_stat': lr_stat, 'p_lrt': p_lrt}
    return results


# %%
# =============================================================================
# 4. VISUALISATION
# =============================================================================

def create_main_figure(df, bp_results, white_results, qr_results,
                       decile_results, bootstrap_results, dglm_results,
                       figures_dir):
    fig, axes = plt.subplots(4, 2, figsize=(280 * mm2inches, 320 * mm2inches),
                             dpi=FIGURE_DPI)
    metrics = ['modularity', 'global_efficiency']
    metric_labels = ['Modularity (residualised)', 'Global Efficiency (residualised)']

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Row 1: residual scatter + LOWESS of |residuals|
        ax = axes[0, col]
        residuals = bp_results[metric]['residuals']
        sds_z = df[PREDICTOR].values
        ax.scatter(sds_z, residuals, alpha=0.3, s=10, color='gray', edgecolors='none')
        abs_resid = np.abs(residuals)
        lowess_result = lowess(abs_resid, sds_z, frac=0.4)
        ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'k-', linewidth=2,
                label='LOWESS |residuals|')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        bp_p = bp_results[metric]['lm_pvalue']
        wh_p = white_results[metric]['lm_pvalue']
        dg_p = dglm_results[metric]['p_lrt']
        dg_a = dglm_results[metric]['alpha']
        ax.text(0.05, 0.95,
                f'BP p = {bp_p:.3e}\nWhite p = {wh_p:.3e}\n'
                f'DGLM alpha = {dg_a:+.3f}, p = {dg_p:.3f}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round',
                          facecolor='yellow' if bp_p < 0.05 else 'white', alpha=0.8))
        ax.set_xlabel('SDS [z]'); ax.set_ylabel('OLS Residuals')
        ax.set_title(f'{label}: Residuals vs SDS')
        ax.legend(fontsize=7, loc='upper right')

        # Row 2: quantile regression fan plot
        ax = axes[1, col]
        ax.scatter(sds_z, df[metric].values, alpha=0.2, s=8, color='gray',
                   edgecolors='none')
        quantiles = qr_results[metric]['quantiles']
        slopes = qr_results[metric]['slopes']
        intercepts = qr_results[metric]['intercepts']
        cmap = plt.cm.coolwarm
        x_range = np.linspace(sds_z.min(), sds_z.max(), 100)
        for i, (q, slope, intercept) in enumerate(zip(quantiles, slopes, intercepts)):
            color = cmap(i / (len(quantiles) - 1))
            ax.plot(x_range, intercept + slope * x_range, color=color, linewidth=1.5,
                    label=f'Q{q:.2f} (b={slope:.3f})')
        ax.set_xlabel('SDS [z]'); ax.set_ylabel(label)
        ax.set_title(f'{label}: Quantile Regression')
        ax.legend(fontsize=6, loc='upper left')
        iq_p = qr_results[metric]['interquantile_test']['p_value']
        slope_range = qr_results[metric]['slope_range']
        ax.text(0.95, 0.05, f'Slope range = {slope_range:.4f}\nIQ p = {iq_p:.4f}',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round',
                          facecolor='yellow' if iq_p < 0.05 else 'white', alpha=0.8))

        # Row 3: decile SDs with bootstrap CIs
        ax = axes[2, col]
        decile_labels = decile_results[metric]['decile_labels']
        sds_l = decile_results[metric]['sds']
        sd_cis = decile_results[metric]['sd_cis']
        cmap_decile = plt.cm.coolwarm
        colors = [cmap_decile(i / 9) for i in range(10)]
        ci_lower = [sd - ci[0] for sd, ci in zip(sds_l, sd_cis)]
        ci_upper = [ci[1] - sd for sd, ci in zip(sds_l, sd_cis)]
        bars = ax.bar(decile_labels, sds_l, color=colors, alpha=0.8, edgecolor='white',
                      linewidth=0.5, yerr=[ci_lower, ci_upper], capsize=3,
                      error_kw={'linewidth': 0.8, 'color': 'black'})
        z_fit = np.polyfit(decile_labels, sds_l, 1)
        p_fit = np.poly1d(z_fit)
        ax.plot(decile_labels, p_fit(decile_labels), 'k--', linewidth=1.5,
                label='Linear trend')
        ax.set_xlabel('SDS Decile'); ax.set_ylabel(f'{label} SD')
        ax.set_title(f'{label}: Variability by Decile')
        ax.set_xticks(decile_labels)
        trend_r = decile_results[metric]['trend_r']
        trend_p = decile_results[metric]['trend_p']
        ax.text(0.05, 0.95, f'r = {trend_r:.3f}, p = {trend_p:.4f}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round',
                          facecolor='yellow' if trend_p < 0.05 else 'white', alpha=0.8))
        ax.legend(fontsize=7)

        # Row 4: balanced bootstrap by SDS quintile
        ax = axes[3, col]
        n_bins = len(bootstrap_results[metric]['observed_vars'])
        bin_centers = bootstrap_results[metric]['bin_centers']
        observed_vars = bootstrap_results[metric]['observed_vars']
        bin_labels = [f'{c:.2f}' for c in bin_centers]
        bin_positions = list(range(1, n_bins + 1))
        cmap_bins = plt.cm.coolwarm
        bin_colors = [cmap_bins(i / (n_bins - 1)) for i in range(n_bins)]
        ax.bar(bin_positions, observed_vars, color=bin_colors, alpha=0.8,
               edgecolor='white', linewidth=0.5)
        z_fit = np.polyfit(bin_positions, observed_vars, 1)
        p_fit = np.poly1d(z_fit)
        ax.plot(bin_positions, p_fit(bin_positions), 'k--', linewidth=1.5,
                label='Linear trend')
        ax.set_xlabel('SDS Quintile (center z)'); ax.set_ylabel(f'{label} Variance')
        ax.set_title(f'{label}: Balanced Bootstrap '
                     f'(n={bootstrap_results[metric]["min_bin_size"]}/bin)')
        ax.set_xticks(bin_positions); ax.set_xticklabels(bin_labels, fontsize=7)
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
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Continuous heteroscedasticity analysis on SDS.'
    )
    parser.add_argument('--project', required=True)
    parser.add_argument('--social', required=True,
                        help='Path to social factor scores CSV (Social_Score column)')
    parser.add_argument('--behavioural', required=True)
    parser.add_argument('--phenotypic', required=True)
    parser.add_argument('--movement', required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--matrices-dir', required=True)
    parser.add_argument('--partition', required=True,
                        help='Path to community partition CSV (from C2b).')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--motion-threshold', type=float, default=0.2)
    args = parser.parse_args()

    project_folder = Path(args.project)
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'
    for d in (figures_dir, reports_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 80,
        "C3b: CONTINUOUS HETEROSCEDASTICITY ANALYSIS (SDS predictor)",
        "=" * 80,
        ""
    ]

    social_file = Path(args.social)
    behavioural_file = Path(args.behavioural)
    phenotypic_file = Path(args.phenotypic)
    movement_file = Path(args.movement)
    id_file = Path(args.ids)
    matrices_dir = Path(args.matrices_dir)
    partition_file = Path(args.partition)

    n_nodes = len(pd.read_csv(partition_file))
    report.append(f"Project folder: {project_folder}")
    report.append(f"N nodes (from C2b partition): {n_nodes}")
    report.append(f"Network threshold: {args.threshold}")
    report.append(f"Motion threshold: {args.motion_threshold}")
    report.append(f"Brain metrics residualised for: {', '.join(COVARIATES)}")
    report.append("PGS is NOT loaded here.")
    report.append("")

    merged_df = load_and_prepare_data(
        project_folder, social_file, behavioural_file, phenotypic_file,
        movement_file, id_file, matrices_dir, n_nodes, args.motion_threshold,
        report
    )

    network_df = calculate_network_metrics(
        merged_df, partition_file, n_nodes, args.threshold, report
    )

    bp_results = test_breusch_pagan(network_df, report)
    white_results = test_white(network_df, report)
    qr_results = test_quantile_regression(network_df, report)
    decile_results = compute_decile_variability(network_df, report)
    bootstrap_results = test_bootstrap_heteroscedasticity(network_df, report)
    dglm_results = test_dglm(network_df, report)

    create_main_figure(network_df, bp_results, white_results, qr_results,
                       decile_results, bootstrap_results, dglm_results,
                       figures_dir)

    # Final summary block
    report.append(""); report.append("=" * 80)
    report.append("FINAL SUMMARY"); report.append("=" * 80)

    for label, results, key in [
        ('Breusch-Pagan', bp_results, 'lm_pvalue'),
        ('White', white_results, 'lm_pvalue'),
    ]:
        report.append(""); report.append(f"{label} test:")
        for metric in ['modularity', 'global_efficiency']:
            p = results[metric][key]
            report.append(f"  {metric}: p = {p:.3e} [{'SIG' if p<0.05 else 'NS'}]")

    report.append(""); report.append("Quantile regression slope divergence:")
    for metric in ['modularity', 'global_efficiency']:
        sr = qr_results[metric]['slope_range']
        p = qr_results[metric]['interquantile_test']['p_value']
        report.append(f"  {metric}: slope range = {sr:.4f}, IQ p = {p:.4f} "
                      f"[{'SIG' if p<0.05 else 'NS'}]")

    report.append(""); report.append("Decile SD trend:")
    for metric in ['modularity', 'global_efficiency']:
        r = decile_results[metric]['trend_r']
        p = decile_results[metric]['trend_p']
        report.append(f"  {metric}: r = {r:.3f}, p = {p:.4f} "
                      f"[{'SIG' if p<0.05 else 'NS'}]")

    report.append(""); report.append("Balanced bootstrap heteroscedasticity:")
    for metric in ['modularity', 'global_efficiency']:
        p = bootstrap_results[metric]['boot_trend_p']
        vr = bootstrap_results[metric]['median_var_ratio']
        report.append(f"  {metric}: boot p = {p:.4f}, var ratio = {vr:.3f} "
                      f"[{'SIG' if p<0.05 else 'NS'}]")

    report.append(""); report.append("DGLM (log-variance ~ SDS):")
    for metric in ['modularity', 'global_efficiency']:
        a = dglm_results[metric]['alpha']
        p = dglm_results[metric]['p_lrt']
        report.append(f"  {metric}: alpha = {a:+.4f}, LR p = {p:.4f} "
                      f"[{'SIG' if (np.isfinite(p) and p<0.05) else 'NS'}]")

    # Save outputs
    network_df.to_csv(results_dir / 'C3b_heteroscedasticity_results.csv', index=False)
    report.append(""); report.append(
        f"Results saved to: {results_dir / 'C3b_heteroscedasticity_results.csv'}"
    )

    qr_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for i, q in enumerate(qr_results[metric]['quantiles']):
            qr_rows.append({
                'metric': metric, 'quantile': q,
                'slope': qr_results[metric]['slopes'][i],
                'intercept': qr_results[metric]['intercepts'][i],
                'slope_lower_ci': qr_results[metric]['slope_cis'][i][0],
                'slope_upper_ci': qr_results[metric]['slope_cis'][i][1],
                'pvalue': qr_results[metric]['pvalues'][i],
            })
    pd.DataFrame(qr_rows).to_csv(
        results_dir / 'C3b_quantile_regression_coefficients.csv', index=False)
    report.append(f"Saved: {results_dir / 'C3b_quantile_regression_coefficients.csv'}")

    decile_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for i in range(10):
            decile_rows.append({
                'metric': metric,
                'decile': decile_results[metric]['decile_labels'][i],
                'n': decile_results[metric]['n_per_decile'][i],
                'sds_center': decile_results[metric]['decile_centers'][i],
                'sd': decile_results[metric]['sds'][i],
                'sd_ci_lower': decile_results[metric]['sd_cis'][i][0],
                'sd_ci_upper': decile_results[metric]['sd_cis'][i][1],
                'iqr': decile_results[metric]['iqrs'][i],
            })
    pd.DataFrame(decile_rows).to_csv(
        results_dir / 'C3b_decile_variability.csv', index=False)
    report.append(f"Saved: {results_dir / 'C3b_decile_variability.csv'}")

    report.append(""); report.append("=" * 80)
    report.append("END OF REPORT"); report.append("=" * 80)

    report_file = reports_dir / 'C3b_continuous_heteroscedasticity_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"\nReport saved to: {report_file}")

    return network_df, bp_results, white_results, qr_results, decile_results, \
           bootstrap_results, dglm_results


# %%
if __name__ == "__main__":
    main()

# %%
