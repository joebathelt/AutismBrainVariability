# %%
"""
C3_continuous_heteroscedasticity_analysis.py
Sex-modulated heteroscedasticity of brain network organisation as a
function of polygenic score (PGS).

Mean-level association (reported first):
    pgs_z ~ metric + Gender + metric:Gender
  Tests whether each graph metric is associated with PGS, and whether
  that association differs by sex.

Primary heteroscedasticity test: variance regression with sex × PGS
interaction. Squared residuals from the mean model
`metric ~ pgs_z + Gender + pgs_z:Gender` are regressed on the same
design; the pgs_z:Gender coefficient is the formal test for whether
the PGS->variance slope differs by sex.

Supporting (sex-stratified) analyses:
  - Decile-SD trend per sex + bootstrap test for trend difference.
  - Balanced-bootstrap variance trend per sex + bootstrap test for
    trend difference.

All analyses applied to both modularity (primary) and global_efficiency
(comparison).

Parcellation resolution is fixed upstream by C2b_evaluate_communities.py;
n_nodes is derived from the number of rows in the partition file.

Usage:
    python C3_continuous_heteroscedasticity_analysis.py \
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
from pathlib import Path
from scipy.stats import zscore, pearsonr, chi2
import statsmodels.api as sm
import networkx as nx
import bct

from utils.covariates import regress_out_covariates

# C3b-specific covariate set: Gender is excluded from residualisation and
# instead enters the heteroscedasticity OLS as a main effect plus a
# pgs_z:Gender interaction (see _build_main_design below).
RESIDUAL_COVARIATES = ("Age_in_Yrs", "FS_IntraCranial_Vol",
                       "Movement_RelativeRMS_mean")


def _build_main_design(df):
    """Main-model design matrix: [const, pgs_z, Gender_dummy, pgs_z:Gender]."""
    pgs_z = df['pgs_z'].values.astype(float)
    gender_dummy = pd.get_dummies(df['Gender'],
                                  drop_first=True).iloc[:, 0].astype(float).values
    interaction = pgs_z * gender_dummy
    return sm.add_constant(np.column_stack([pgs_z, gender_dummy, interaction]))


_MAIN_COEF_NAMES = ('const', 'pgs_z', 'Gender', 'pgs_z:Gender')

# Set up plotting
plt.style.use('default')

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

    merged_df['pgs_z'] = zscore(merged_df['blup_PGS_residuals'])

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
            'pgs_z': row['pgs_z'],
            'Age_in_Yrs': row['Age_in_Yrs'],
            'Gender': row['Gender'],
            'FS_IntraCranial_Vol': row['FS_IntraCranial_Vol'],
            'Movement_RelativeRMS_mean': row['Movement_RelativeRMS_mean'],
        })

    report.append(f"Computed metrics for {len(results)} subjects")

    network_df = pd.DataFrame(results)

    # Residualise brain metrics for age, ICV, and head motion. Sex is
    # deliberately excluded here and enters the heteroscedasticity OLS as a
    # main effect plus a pgs_z:Gender interaction (see _build_main_design),
    # so any sex-dependent PGS effect on the mean is modelled rather than
    # residualised away. Keep raw values under *_raw for audit.
    network_df['modularity_raw'] = network_df['modularity']
    network_df['global_efficiency_raw'] = network_df['global_efficiency']
    network_df[['modularity', 'global_efficiency']] = regress_out_covariates(
        network_df[['modularity', 'global_efficiency']],
        network_df[list(RESIDUAL_COVARIATES)],
    )
    report.append(f"Residualised modularity & global_efficiency for: "
                  f"{', '.join(RESIDUAL_COVARIATES)}")

    return network_df


# %%
# =============================================================================
# 3. PGS ~ GRAPH METRIC ASSOCIATION (MEAN-LEVEL)
# =============================================================================

_ASSOC_COEF_NAMES = ('const', 'metric', 'Gender', 'metric:Gender')


def test_pgs_graph_metric_association(df, report):
    """
    OLS regression of PGS on each graph metric with sex and a
    metric × sex interaction.

        pgs_z ~ const + metric + Gender + metric:Gender

    Reports the metric main effect (overall association), the Gender
    main effect, and the metric:Gender interaction (does the
    metric->PGS slope differ by sex?).
    """
    report.append("")
    report.append("=" * 60)
    report.append("PGS ~ GRAPH METRIC ASSOCIATION (sex × metric interaction)")
    report.append("=" * 60)
    report.append("  Model: pgs_z ~ metric + Gender + metric:Gender")

    y = df['pgs_z'].values.astype(float)
    gender_dummy = pd.get_dummies(df['Gender'],
                                  drop_first=True).iloc[:, 0].astype(float).values

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")

        m = df[metric].values.astype(float)
        interaction = m * gender_dummy
        X = sm.add_constant(np.column_stack([m, gender_dummy, interaction]))
        model = sm.OLS(y, X).fit()

        report.append("  Coefficients:")
        for i, name in enumerate(_ASSOC_COEF_NAMES):
            report.append(f"    {name:<14} beta = {model.params[i]:.6e}, "
                          f"p = {model.pvalues[i]:.4f}")
        report.append(f"  Model R^2 = {model.rsquared:.4f}, "
                      f"overall F p = {model.f_pvalue:.4e}")

        interaction_p = model.pvalues[3]
        if interaction_p < 0.05:
            report.append(f"  *** Sex-modulated PGS-{metric} association: "
                          f"metric:Gender p = {interaction_p:.4f}")
        else:
            report.append(f"  No sex-modulated PGS-{metric} association: "
                          f"metric:Gender p = {interaction_p:.4f}")

        results[metric] = {
            'model': model,
            'coefs': dict(zip(_ASSOC_COEF_NAMES, model.params)),
            'pvalues': dict(zip(_ASSOC_COEF_NAMES, model.pvalues)),
            'r_squared': model.rsquared,
            'f_pvalue': model.f_pvalue,
        }

    return results


# %%
# =============================================================================
# 4. SEX-MODULATED HETEROSCEDASTICITY
# =============================================================================

def test_variance_regression_by_sex(df, report):
    """
    Generalised Breusch-Pagan with sex × PGS interaction on the variance.

    Mean model:   metric ~ const + pgs_z + Gender + pgs_z:Gender
    Variance:     resid^2 ~ const + pgs_z + Gender + pgs_z:Gender (OLS)

    The pgs_z:Gender coefficient on squared residuals is the formal test for
    sex-modulated heteroscedasticity (does the PGS->variance slope differ by
    sex?). We also report the omnibus LM = N*R^2 of the auxiliary regression,
    which is the standard BP statistic for this design.
    """
    report.append("")
    report.append("=" * 60)
    report.append("VARIANCE REGRESSION WITH SEX × PGS INTERACTION")
    report.append("=" * 60)
    report.append("  Mean:     metric ~ pgs_z + Gender + pgs_z:Gender")
    report.append("  Variance: resid^2 ~ pgs_z + Gender + pgs_z:Gender")

    X = _build_main_design(df)
    n = X.shape[0]
    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")

        y = df[metric].values
        mean_model = sm.OLS(y, X).fit()
        e_squared = mean_model.resid ** 2

        var_model = sm.OLS(e_squared, X).fit()

        report.append("  Auxiliary (resid^2 ~ ...) coefficients:")
        for i, name in enumerate(_MAIN_COEF_NAMES):
            report.append(f"    {name:<14} beta = {var_model.params[i]:.6e}, "
                          f"p = {var_model.pvalues[i]:.4f}")

        # BP-style LM stat: N * R^2 of auxiliary regression, chi^2(k-1).
        lm_stat = n * var_model.rsquared
        # k - 1 = number of regressors excluding constant
        df_resid = X.shape[1] - 1
        lm_pvalue = 1.0 - chi2.cdf(lm_stat, df_resid)

        report.append(f"  Omnibus LM = {lm_stat:.3f}, "
                      f"chi^2({df_resid}) p = {lm_pvalue:.4e}")

        interaction_p = var_model.pvalues[3]
        interaction_beta = var_model.params[3]
        if interaction_p < 0.05:
            report.append(f"  *** Sex-modulated heteroscedasticity: "
                          f"pgs_z:Gender beta = {interaction_beta:.4e}, "
                          f"p = {interaction_p:.4f}")
        else:
            report.append(f"  No sex-modulated heteroscedasticity: "
                          f"pgs_z:Gender p = {interaction_p:.4f}")

        results[metric] = {
            'mean_model': mean_model,
            'var_model': var_model,
            'coefs': dict(zip(_MAIN_COEF_NAMES, var_model.params)),
            'pvalues': dict(zip(_MAIN_COEF_NAMES, var_model.pvalues)),
            'r_squared': var_model.rsquared,
            'lm_stat': lm_stat,
            'lm_pvalue': lm_pvalue,
            'interaction_beta': interaction_beta,
            'interaction_p': interaction_p,
        }

    return results


def compute_decile_variability_by_sex(df, report, n_bootstrap=1000):
    """
    Decile-SD trend stratified by sex.

    Within each sex, splits PGS into 10 equal-N bins and computes the SD
    of the brain metric per decile (with bootstrap CIs). Reports the
    SD-vs-PGS-decile correlation per sex and a bootstrap two-tailed test
    for the difference in trends between sexes.
    """
    report.append("")
    report.append("=" * 60)
    report.append("DECILE-SD TREND BY SEX")
    report.append("=" * 60)

    np.random.seed(42)
    sex_groups = sorted(df['Gender'].dropna().unique())

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")

        per_sex = {}
        sex_decile_values = {}

        for sex in sex_groups:
            sub = df[df['Gender'] == sex].copy()
            sub['pgs_decile'] = pd.qcut(sub['pgs_z'], q=10, labels=False) + 1

            decile_centers, sds, sd_cis, n_per_decile, decile_values = (
                [], [], [], [], [])
            for d in range(1, 11):
                values = sub.loc[sub['pgs_decile'] == d, metric].values
                pgs_center = sub.loc[sub['pgs_decile'] == d, 'pgs_z'].mean()
                n = len(values)
                sd = np.std(values, ddof=1)

                boot_sds = [np.std(np.random.choice(values, size=n, replace=True),
                                   ddof=1)
                            for _ in range(n_bootstrap)]
                ci_lower, ci_upper = np.percentile(boot_sds, [2.5, 97.5])

                decile_centers.append(pgs_center)
                sds.append(sd)
                sd_cis.append((ci_lower, ci_upper))
                n_per_decile.append(n)
                decile_values.append(values)

            trend_r, trend_p = pearsonr(decile_centers, sds)
            report.append(f"  {sex} (median n/decile = {int(np.median(n_per_decile))}): "
                          f"trend r = {trend_r:.3f}, p = {trend_p:.4f}")

            per_sex[sex] = {
                'decile_centers': decile_centers,
                'sds': sds,
                'sd_cis': sd_cis,
                'n_per_decile': n_per_decile,
                'trend_r': trend_r,
                'trend_p': trend_p,
            }
            sex_decile_values[sex] = decile_values

        # Bootstrap two-tailed test for trend difference (only if 2 sexes)
        difference = None
        if len(sex_groups) == 2:
            sa, sb = sex_groups[0], sex_groups[1]
            obs_diff = per_sex[sb]['trend_r'] - per_sex[sa]['trend_r']

            boot_diffs = []
            for _ in range(n_bootstrap):
                trend_per_sex = {}
                for sex in sex_groups:
                    boot_sds_dec = [
                        np.std(np.random.choice(v, size=len(v), replace=True),
                               ddof=1)
                        for v in sex_decile_values[sex]
                    ]
                    r, _ = pearsonr(per_sex[sex]['decile_centers'], boot_sds_dec)
                    trend_per_sex[sex] = r
                boot_diffs.append(trend_per_sex[sb] - trend_per_sex[sa])

            diff_arr = np.asarray(boot_diffs)
            diff_ci = np.percentile(diff_arr, [2.5, 97.5])
            diff_p = 2.0 * min(np.mean(diff_arr >= 0), np.mean(diff_arr <= 0))

            report.append(f"  Trend difference ({sb} - {sa}):")
            report.append(f"    Observed: {obs_diff:.3f}")
            report.append(f"    Bootstrap 95% CI: [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]")
            report.append(f"    Two-tailed p: {diff_p:.4f}")
            if diff_p < 0.05:
                report.append(f"  *** Sex-modulated decile-SD trend detected")
            else:
                report.append(f"  No significant difference in decile-SD trend")

            difference = {
                'comparison': f"{sb} - {sa}",
                'observed_diff': obs_diff,
                'ci': (float(diff_ci[0]), float(diff_ci[1])),
                'p_value': diff_p,
            }

        results[metric] = {
            'per_sex': per_sex,
            'difference': difference,
            'sex_groups': sex_groups,
        }

    return results


def test_bootstrap_heteroscedasticity_by_sex(df, report, n_bootstrap=1000,
                                             n_bins=5):
    """
    Bootstrap-balanced heteroscedasticity test stratified by sex.

    Within each sex, divides PGS into equal-N quantile bins and bootstraps
    the variance-vs-bin trend with equal-N resampling per bin. Reports the
    per-sex observed trend with bootstrap 95% CI and a one-tailed
    "increasing variance" p-value, plus a two-tailed bootstrap test for the
    difference in trends between sexes.
    """
    report.append("")
    report.append("=" * 60)
    report.append("BOOTSTRAP HETEROSCEDASTICITY BY SEX (PGS-BALANCED)")
    report.append("=" * 60)
    report.append(f"  N bootstrap iterations: {n_bootstrap}")
    report.append(f"  N PGS bins per sex: {n_bins}")

    np.random.seed(42)
    sex_groups = sorted(df['Gender'].dropna().unique())

    # Pre-compute per-sex bin centers and bin values per metric
    bin_centers_per_sex = {}
    min_bin_size_per_sex = {}
    bin_values_per_sex = {sex: {m: [] for m in ['modularity',
                                                 'global_efficiency']}
                          for sex in sex_groups}

    for sex in sex_groups:
        sub = df[df['Gender'] == sex].copy()
        sub['pgs_bin'] = pd.qcut(sub['pgs_z'], q=n_bins, labels=False)
        centers = []
        sizes = []
        for b in range(n_bins):
            bin_df = sub[sub['pgs_bin'] == b]
            centers.append(bin_df['pgs_z'].mean())
            sizes.append(len(bin_df))
            for m in ['modularity', 'global_efficiency']:
                bin_values_per_sex[sex][m].append(bin_df[m].values)
        bin_centers_per_sex[sex] = centers
        min_bin_size_per_sex[sex] = min(sizes)
        report.append(f"  {sex}: bin sizes {sizes} (balanced n = {min(sizes)})")

    results = {}
    for metric in ['modularity', 'global_efficiency']:
        report.append("")
        report.append(f"{metric.upper()}:")

        per_sex = {}
        for sex in sex_groups:
            obs_vars = [np.var(v, ddof=1)
                        for v in bin_values_per_sex[sex][metric]]
            obs_trend, _ = pearsonr(bin_centers_per_sex[sex], obs_vars)

            boot_trends = []
            for _ in range(n_bootstrap):
                boot_vars = [
                    np.var(np.random.choice(v,
                                            size=min_bin_size_per_sex[sex],
                                            replace=True), ddof=1)
                    for v in bin_values_per_sex[sex][metric]
                ]
                r, _ = pearsonr(bin_centers_per_sex[sex], boot_vars)
                boot_trends.append(r)

            trend_ci = np.percentile(boot_trends, [2.5, 97.5])
            trend_p_one = float(np.mean(np.array(boot_trends) <= 0))

            report.append(f"  {sex}: obs trend r = {obs_trend:.3f}, "
                          f"95% CI [{trend_ci[0]:.3f}, {trend_ci[1]:.3f}], "
                          f"one-tailed p = {trend_p_one:.4f}")

            per_sex[sex] = {
                'observed_trend': obs_trend,
                'boot_trend_ci': (float(trend_ci[0]), float(trend_ci[1])),
                'boot_trend_p_increasing': trend_p_one,
                'observed_vars': obs_vars,
                'bin_centers': bin_centers_per_sex[sex],
                'min_bin_size': min_bin_size_per_sex[sex],
            }

        difference = None
        if len(sex_groups) == 2:
            sa, sb = sex_groups[0], sex_groups[1]
            obs_diff = (per_sex[sb]['observed_trend']
                        - per_sex[sa]['observed_trend'])

            boot_diffs = []
            for _ in range(n_bootstrap):
                trends = {}
                for sex in sex_groups:
                    boot_vars = [
                        np.var(np.random.choice(
                            v, size=min_bin_size_per_sex[sex], replace=True),
                               ddof=1)
                        for v in bin_values_per_sex[sex][metric]
                    ]
                    r, _ = pearsonr(bin_centers_per_sex[sex], boot_vars)
                    trends[sex] = r
                boot_diffs.append(trends[sb] - trends[sa])

            diff_arr = np.asarray(boot_diffs)
            diff_ci = np.percentile(diff_arr, [2.5, 97.5])
            diff_p = 2.0 * min(np.mean(diff_arr >= 0), np.mean(diff_arr <= 0))

            report.append(f"  Trend difference ({sb} - {sa}):")
            report.append(f"    Observed: {obs_diff:.3f}")
            report.append(f"    Bootstrap 95% CI: [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]")
            report.append(f"    Two-tailed p: {diff_p:.4f}")
            if diff_p < 0.05:
                report.append(f"  *** Sex-modulated balanced-variance trend detected")
            else:
                report.append(f"  No significant difference in balanced-variance trend")

            difference = {
                'comparison': f"{sb} - {sa}",
                'observed_diff': obs_diff,
                'ci': (float(diff_ci[0]), float(diff_ci[1])),
                'p_value': diff_p,
            }

        results[metric] = {
            'per_sex': per_sex,
            'difference': difference,
            'sex_groups': sex_groups,
        }

    return results


# %%
# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def create_sex_stratified_figure(decile_sex_results, bootstrap_sex_results,
                                 figures_dir):
    """2x2 figure: rows = metrics, cols = (decile-SD by sex | balanced var by sex)."""
    fig, axes = plt.subplots(2, 2,
                             figsize=(220 * mm2inches, 180 * mm2inches),
                             dpi=FIGURE_DPI)

    metrics = ['modularity', 'global_efficiency']
    metric_labels = ['Modularity', 'Global Efficiency']
    sex_colors = {'F': '#E64B35', 'M': '#4DBBD5'}

    for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # ---- Decile SD by sex ----
        ax = axes[row, 0]
        per_sex = decile_sex_results[metric]['per_sex']
        for sex in decile_sex_results[metric]['sex_groups']:
            d = per_sex[sex]
            yerr_lo = [s - ci[0] for s, ci in zip(d['sds'], d['sd_cis'])]
            yerr_hi = [ci[1] - s for s, ci in zip(d['sds'], d['sd_cis'])]
            ax.errorbar(d['decile_centers'], d['sds'],
                        yerr=[yerr_lo, yerr_hi], fmt='-o',
                        color=sex_colors.get(sex, 'gray'),
                        label=f"{sex} (r={d['trend_r']:.2f}, "
                              f"p={d['trend_p']:.3f})",
                        capsize=2, linewidth=1, markersize=4)
        ax.set_xlabel('PGS decile center (z)')
        ax.set_ylabel(f'{label} SD')
        ax.set_title(f'{label}: decile-SD trend by sex')
        ax.legend(fontsize=7)

        diff = decile_sex_results[metric]['difference']
        if diff is not None:
            ax.text(0.05, 0.95,
                    f"diff = {diff['observed_diff']:.2f}\n"
                    f"p = {diff['p_value']:.3f}",
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round',
                              facecolor=('yellow' if diff['p_value'] < 0.05
                                         else 'white'), alpha=0.8))

        # ---- Balanced variance by sex ----
        ax = axes[row, 1]
        per_sex = bootstrap_sex_results[metric]['per_sex']
        for sex in bootstrap_sex_results[metric]['sex_groups']:
            b = per_sex[sex]
            ax.plot(b['bin_centers'], b['observed_vars'], '-o',
                    color=sex_colors.get(sex, 'gray'),
                    label=f"{sex} (r={b['observed_trend']:.2f}, "
                          f"one-tailed p={b['boot_trend_p_increasing']:.3f})",
                    linewidth=1, markersize=4)
        ax.set_xlabel('PGS quintile center (z)')
        ax.set_ylabel(f'{label} variance')
        ax.set_title(f'{label}: balanced variance by sex')
        ax.legend(fontsize=7)

        diff = bootstrap_sex_results[metric]['difference']
        if diff is not None:
            ax.text(0.05, 0.95,
                    f"diff = {diff['observed_diff']:.2f}\n"
                    f"p = {diff['p_value']:.3f}",
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round',
                              facecolor=('yellow' if diff['p_value'] < 0.05
                                         else 'white'), alpha=0.8))

    plt.tight_layout()
    output_file = figures_dir / 'C3_sex_stratified_heteroscedasticity.png'
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"Sex-stratified figure saved to: {output_file}")
    return fig


# %%
# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete continuous heteroscedasticity analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Sex-modulated heteroscedasticity analysis (PGS × sex on variance)'
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
        "C3: CONTINUOUS HETEROSCEDASTICITY ANALYSIS - SEX × PGS VARIANCE",
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
    report.append(f"Brain metrics residualised for: {', '.join(RESIDUAL_COVARIATES)}")
    report.append("  (applied before all heteroscedasticity tests)")
    report.append("Association model: pgs_z ~ metric + Gender + metric:Gender")
    report.append("Heteroscedasticity mean model: metric ~ pgs_z + Gender + pgs_z:Gender")
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

    # Mean-level association: PGS ~ graph metric (with sex × metric interaction)
    assoc_results = test_pgs_graph_metric_association(network_df, report)

    # Sex-modulated heteroscedasticity (variance regression + stratified)
    vr_sex_results = test_variance_regression_by_sex(network_df, report)
    decile_sex_results = compute_decile_variability_by_sex(network_df, report)
    bootstrap_sex_results = test_bootstrap_heteroscedasticity_by_sex(
        network_df, report)

    # Sex-stratified visualisation
    create_sex_stratified_figure(decile_sex_results, bootstrap_sex_results,
                                 figures_dir)

    # Final summary
    report.append("")
    report.append("=" * 80)
    report.append("FINAL SUMMARY")
    report.append("=" * 80)

    report.append("")
    report.append("PGS ~ graph metric association (mean level):")
    for metric in ['modularity', 'global_efficiency']:
        m_beta = assoc_results[metric]['coefs']['metric']
        m_p = assoc_results[metric]['pvalues']['metric']
        i_beta = assoc_results[metric]['coefs']['metric:Gender']
        i_p = assoc_results[metric]['pvalues']['metric:Gender']
        m_sig = "SIG" if m_p < 0.05 else "NS"
        i_sig = "SIG" if i_p < 0.05 else "NS"
        report.append(f"  {metric}: metric beta = {m_beta:.4e}, "
                      f"p = {m_p:.4f} [{m_sig}]; "
                      f"metric:Gender beta = {i_beta:.4e}, "
                      f"p = {i_p:.4f} [{i_sig}]")

    report.append("")
    report.append("Variance regression — sex × PGS interaction (resid^2):")
    for metric in ['modularity', 'global_efficiency']:
        p = vr_sex_results[metric]['interaction_p']
        beta = vr_sex_results[metric]['interaction_beta']
        sig = "SIG" if p < 0.05 else "NS"
        report.append(f"  {metric}: pgs_z:Gender beta = {beta:.4e}, "
                      f"p = {p:.4f} [{sig}]")

    report.append("")
    report.append("Decile-SD trend difference between sexes:")
    for metric in ['modularity', 'global_efficiency']:
        diff = decile_sex_results[metric]['difference']
        if diff is None:
            report.append(f"  {metric}: only one sex in sample (skipped)")
            continue
        sig = "SIG" if diff['p_value'] < 0.05 else "NS"
        report.append(f"  {metric}: {diff['comparison']} = "
                      f"{diff['observed_diff']:.3f}, p = {diff['p_value']:.4f} "
                      f"[{sig}]")

    report.append("")
    report.append("Bootstrap balanced-variance trend difference between sexes:")
    for metric in ['modularity', 'global_efficiency']:
        diff = bootstrap_sex_results[metric]['difference']
        if diff is None:
            report.append(f"  {metric}: only one sex in sample (skipped)")
            continue
        sig = "SIG" if diff['p_value'] < 0.05 else "NS"
        report.append(f"  {metric}: {diff['comparison']} = "
                      f"{diff['observed_diff']:.3f}, p = {diff['p_value']:.4f} "
                      f"[{sig}]")

    # Save results
    results_file = results_dir / 'C3_heteroscedasticity_results.csv'
    network_df.to_csv(results_file, index=False)
    report.append("")
    report.append(f"Results saved to: {results_file}")

    # Save variance-regression coefficients (sex × PGS interaction test)
    vr_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for term in _MAIN_COEF_NAMES:
            vr_rows.append({
                'metric': metric,
                'term': term,
                'beta': vr_sex_results[metric]['coefs'][term],
                'pvalue': vr_sex_results[metric]['pvalues'][term],
            })
        vr_rows.append({
            'metric': metric,
            'term': 'omnibus_LM',
            'beta': vr_sex_results[metric]['lm_stat'],
            'pvalue': vr_sex_results[metric]['lm_pvalue'],
        })
    vr_file = results_dir / 'C3_variance_regression_sex.csv'
    pd.DataFrame(vr_rows).to_csv(vr_file, index=False)
    report.append(f"Variance-regression coefficients saved to: {vr_file}")

    # Save PGS ~ graph metric association coefficients
    assoc_rows = []
    for metric in ['modularity', 'global_efficiency']:
        for term in _ASSOC_COEF_NAMES:
            assoc_rows.append({
                'metric': metric,
                'term': term,
                'beta': assoc_results[metric]['coefs'][term],
                'pvalue': assoc_results[metric]['pvalues'][term],
            })
        assoc_rows.append({
            'metric': metric,
            'term': 'model_R2',
            'beta': assoc_results[metric]['r_squared'],
            'pvalue': assoc_results[metric]['f_pvalue'],
        })
    assoc_file = results_dir / 'C3_pgs_metric_association.csv'
    pd.DataFrame(assoc_rows).to_csv(assoc_file, index=False)
    report.append(f"PGS-metric association coefficients saved to: {assoc_file}")

    # Write report to file
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_file = reports_dir / 'C3_continuous_heteroscedasticity_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_file}")

    return (network_df, assoc_results, vr_sex_results, decile_sex_results,
            bootstrap_sex_results)


# %%
if __name__ == "__main__":
    main()

# %%
