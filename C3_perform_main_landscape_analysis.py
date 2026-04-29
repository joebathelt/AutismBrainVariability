# %%
"""
C3_perform_main_landscape_analysis.py
Behaviour-stratified landscape analysis (SDS-primary)

Tests whether participants with worse social performance (higher SDS =
Social_Score, where higher z = slower/worse RT-derived social cognition
factor) show GREATER variability in graph metrics (modularity, global
efficiency), with global efficiency expected to be relatively preserved.

PGS is intentionally NOT loaded here. Joining PGS would restrict the cohort
to genotyped participants and substantially reduce N. The exploratory
SDS-PGS test lives in C6_exploratory_sds_pgs.py and operates on the
genotyped subset of this script's output.

Stratification: low/middle/high SDS by z-cut with buffer
  low_sds   : sds_z < -1
  middle    : -0.5 <= sds_z <= 0.5
  high_sds  : sds_z > 1
  (subjects in [-1, -0.5) and (0.5, 1] are dropped as buffer)

Tests on graph metrics (residualised for age / ICV / motion):
  - Pearson r between metric and Social_Score (full-sample coherence check)
  - One-tailed Levene's: Var(metric | high_sds) > Var(metric | low_sds)
  - Bootstrap CI for the High/Low SD ratio

Parcellation resolution is fixed upstream by C2b_evaluate_communities.py.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, levene, zscore
import networkx as nx
import bct
import warnings

from utils.covariates import COVARIATES, regress_out_covariates

warnings.filterwarnings('ignore')

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

SDS_GROUPS = ['low_sds', 'middle', 'high_sds']
SDS_GROUP_LABELS = {'low_sds': 'Low SDS', 'middle': 'Middle', 'high_sds': 'High SDS'}
SDS_GROUP_COLORS = {'low_sds': '#3498db', 'middle': '#f39c12', 'high_sds': '#e74c3c'}


# %%
# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(args, report):
    """Load behavioural / phenotypic / movement / connectivity inputs and
    create SDS groups. PGS is deliberately not loaded."""
    report.append("=" * 80)
    report.append("DATA LOADING AND PREPARATION")
    report.append("=" * 80)

    social_df = pd.read_csv(args.social)
    behavioural_df = pd.read_csv(args.behavioural)
    # No sex filter: Gender is residualised out via COVARIATES.
    phenotypic_df = pd.read_csv(args.phenotypic)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    movement_df = pd.read_csv(args.movement)
    ids = pd.read_csv(args.ids, header=None)[0].tolist()

    report.append("\nLoaded data files:")
    report.append(f"  Social scores: {len(social_df)} subjects")
    report.append(f"  Behavioural data (M+F): {len(behavioural_df)} subjects")
    report.append(f"  Phenotypic data: {len(phenotypic_df)} subjects")
    report.append(f"  Movement data: {len(movement_df)} subjects")
    report.append(f"  Subject IDs: {len(ids)}")

    merged_base = pd.merge(social_df[['Subject', 'Social_Score']],
                           behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']],
                           on='Subject')
    merged_base = pd.merge(merged_base,
                           phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_base = pd.merge(merged_base,
                           movement_df[['Subject', 'Movement_RelativeRMS_mean']],
                           on='Subject')

    report.append(f"\nAfter merging base data: {len(merged_base)} subjects")

    n_before_motion = len(merged_base)
    merged_base = merged_base[merged_base['Movement_RelativeRMS_mean'] < args.motion_threshold]
    report.append(f"After motion exclusion (threshold={args.motion_threshold}): "
                  f"{len(merged_base)} subjects "
                  f"(excluded {n_before_motion - len(merged_base)})")

    merged_base = merged_base.dropna()
    report.append(f"After removing missing data: {len(merged_base)} subjects")

    # SDS groups: low (<-1 SD), middle (-0.5 to 0.5 SD), high (>+1 SD)
    # Buffer zones [-1, -0.5) and (0.5, 1] are dropped from group analyses but
    # kept in the dataframe so continuous tests can use them.
    merged_base['sds_z'] = zscore(merged_base['Social_Score'])
    merged_base['sds_group'] = pd.cut(
        merged_base['sds_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['low_sds', 'buffer_low', 'middle', 'buffer_high', 'high_sds']
    )

    report.append("\nSDS group sizes:")
    for group in SDS_GROUPS:
        n = (merged_base['sds_group'] == group).sum()
        report.append(f"  {SDS_GROUP_LABELS[group]}: {n}")
    n_buffer = ((merged_base['sds_group'] == 'buffer_low').sum()
                + (merged_base['sds_group'] == 'buffer_high').sum())
    report.append(f"  (buffer dropped from group tests: {n_buffer})")

    data_by_parcellation = {}
    matrices_dir = Path(args.matrices_dir)

    report.append("\nLoading connectivity matrices:")
    for n_nodes in args.parcellations:
        matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
        try:
            mats_df = pd.read_csv(matrix_file, header=None, sep=r'\s+')
            mats_df.columns = [f'conn_{i+1}' for i in range(mats_df.shape[1])]
            mats_df.index = ids
            merged_df = pd.merge(merged_base, mats_df, left_on='Subject', right_index=True)
            data_by_parcellation[n_nodes] = merged_df
            report.append(f"  {n_nodes} nodes: {len(merged_df)} subjects "
                          f"(matrix shape: {mats_df.shape})")
        except FileNotFoundError:
            report.append(f"  {n_nodes} nodes: FILE NOT FOUND - skipping")
            continue

    report.append(f"\nLoaded data for {len(data_by_parcellation)} parcellations")
    return data_by_parcellation


# %%
# =============================================================================
# 2. NETWORK METRICS CALCULATION
# =============================================================================

def calculate_network_metrics_all(data_by_parcellation, args, report):
    """Calculate network metrics for all parcellation x threshold configurations."""
    report.append("\n" + "=" * 80)
    report.append("NETWORK METRICS CALCULATION")
    report.append("=" * 80)

    sensitivity_config = {}
    for n_nodes in args.parcellations:
        if n_nodes == args.main_nodes:
            sensitivity_config[n_nodes] = args.thresholds
        else:
            sensitivity_config[n_nodes] = [args.main_threshold]

    total_configs = sum(len(t) for t in sensitivity_config.values())
    report.append(f"\nConfigurations to process: {total_configs}")
    for n_nodes, thresholds in sensitivity_config.items():
        report.append(f"  {n_nodes} nodes: thresholds {thresholds}")

    all_results = {}

    for n_nodes in args.parcellations:
        if n_nodes not in data_by_parcellation:
            report.append(f"\nSkipping {n_nodes} nodes (data not available)")
            continue

        merged_df = data_by_parcellation[n_nodes]
        thresholds_to_test = sensitivity_config[n_nodes]

        for threshold in thresholds_to_test:
            config_key = f"{n_nodes}nodes_{threshold:.2f}thresh"
            report.append(f"\nProcessing {config_key}...")

            partition_file = Path(args.partition)
            partition_df = pd.read_csv(partition_file)
            report.append(f"  Using partition from {partition_file.name}")

            results = []
            n_subjects = len(merged_df)

            for i, (_, row) in enumerate(merged_df.iterrows()):
                if (i + 1) % 200 == 0:
                    print(f"    {config_key}: Processed {i+1}/{n_subjects} subjects")

                conn_data = row[[c for c in row.index if c.startswith('conn_')]].values
                mat = np.reshape(conn_data, (n_nodes, n_nodes)).astype(np.float64)
                mat = mat / 100
                mat = bct.threshold_proportional(mat, threshold)
                mat = np.nan_to_num(mat, nan=0.0)

                try:
                    if partition_df is not None and len(partition_df) == n_nodes:
                        _, modularity = bct.modularity_und_sign(
                            mat, partition_df['community_id'].values)
                    else:
                        _, modularity = bct.modularity_und_sign(mat)
                except Exception:
                    modularity = np.nan

                try:
                    mat_pos = mat.copy()
                    mat_pos[mat_pos < 0] = 0
                    G = nx.from_numpy_array(mat_pos)
                    global_efficiency = nx.global_efficiency(G)
                except Exception:
                    global_efficiency = np.nan

                results.append({
                    'Subject': row['Subject'],
                    'modularity': modularity,
                    'global_efficiency': global_efficiency,
                    'Social_Score': row['Social_Score'],
                    'sds_group': row['sds_group'],
                    'sds_z': row['sds_z'],
                    'Age_in_Yrs': row['Age_in_Yrs'],
                    'FS_IntraCranial_Vol': row['FS_IntraCranial_Vol'],
                    'Movement_RelativeRMS_mean': row['Movement_RelativeRMS_mean'],
                    'Gender': row['Gender'],
                    'n_nodes': n_nodes,
                    'threshold': threshold,
                    'config': config_key
                })

            config_df = pd.DataFrame(results)
            n_before = len(config_df)
            config_df = config_df.dropna(subset=['modularity', 'global_efficiency'])

            config_df['modularity_raw'] = config_df['modularity']
            config_df['global_efficiency_raw'] = config_df['global_efficiency']
            config_df[['modularity', 'global_efficiency']] = regress_out_covariates(
                config_df[['modularity', 'global_efficiency']],
                config_df[list(COVARIATES)],
            )

            all_results[config_key] = config_df
            report.append(f"  Final n = {len(config_df)} "
                          f"(excluded {n_before - len(config_df)} with missing metrics)")
            report.append(f"  Residualised modularity & global_efficiency for: "
                          f"{', '.join(COVARIATES)}")

    return all_results


# %%
# =============================================================================
# 3. MAIN ANALYSES
# =============================================================================

def test_brain_behavior_relationships(all_results, args, report):
    """Pearson correlations of graph metrics with SDS (full sample, residualised
    metrics). Coherence check, not the primary test."""
    report.append("\n" + "=" * 80)
    report.append("BRAIN-BEHAVIOUR COHERENCE CHECK (Pearson r vs Social_Score)")
    report.append("=" * 80)

    sensitivity_results = {}

    for config_key, df in all_results.items():
        if len(df) == 0:
            report.append(f"\nWARNING: {config_key} has no data - skipping")
            continue
        n_nodes = df['n_nodes'].iloc[0]
        threshold = df['threshold'].iloc[0]

        r_mod, p_mod = pearsonr(df['modularity'], df['Social_Score'])
        r_eff, p_eff = pearsonr(df['global_efficiency'], df['Social_Score'])

        is_main = (n_nodes == args.main_nodes and threshold == args.main_threshold)

        sensitivity_results[config_key] = {
            'n_nodes': n_nodes,
            'threshold': threshold,
            'n_subjects': len(df),
            'mod_r': r_mod, 'mod_p': p_mod,
            'eff_r': r_eff, 'eff_p': p_eff,
            'is_main': is_main
        }

        status = "*** MAIN ***" if is_main else ""
        report.append(f"\n{config_key} {status}")
        report.append(f"  n = {len(df)}")
        report.append(f"  Modularity-SDS: r = {r_mod:.3f}, p = {p_mod:.3e}")
        report.append(f"  Efficiency-SDS: r = {r_eff:.3f}, p = {p_eff:.3e}")

    return sensitivity_results


def test_variability_hypothesis(all_results, args, report):
    """Primary test: variance of graph metrics across SDS groups.

    Directional hypothesis: Var(metric | high_sds) > Var(metric | low_sds).
    Reports a one-tailed Levene's p (folded from the two-tailed scipy output,
    sign read from the observed variance ratio) and a bootstrap CI for the
    SD ratio.
    """
    report.append("\n" + "=" * 80)
    report.append("VARIABILITY ANALYSIS (PRIMARY): graph metric SD by SDS group")
    report.append("Directional H1: Var(High SDS) > Var(Low SDS)")
    report.append("=" * 80)

    sensitivity_results = {}

    for config_key, df in all_results.items():
        if len(df) == 0:
            report.append(f"\nWARNING: {config_key} has no data - skipping")
            continue
        n_nodes = df['n_nodes'].iloc[0]
        threshold = df['threshold'].iloc[0]
        is_main = (n_nodes == args.main_nodes and threshold == args.main_threshold)

        config_results = {}

        for metric in ['modularity', 'global_efficiency']:
            high_data = df[df['sds_group'] == 'high_sds'][metric].values
            low_data = df[df['sds_group'] == 'low_sds'][metric].values
            middle_data = df[df['sds_group'] == 'middle'][metric].values

            if len(high_data) < 10 or len(low_data) < 10:
                continue

            group_stats = {}
            for group, data in [('low_sds', low_data),
                                ('middle', middle_data),
                                ('high_sds', high_data)]:
                group_stats[group] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'var': np.var(data),
                    'n': len(data)
                }

            var_ratio = np.var(high_data) / np.var(low_data)

            # Two-tailed Levene's; convert to one-tailed in the predicted
            # direction (Var_high > Var_low). If observed var_ratio <= 1 the
            # one-tailed p is 1 - p_two/2.
            levene_stat, levene_p_two = levene(high_data, low_data)
            if var_ratio > 1:
                levene_p_one = levene_p_two / 2
            else:
                levene_p_one = 1 - levene_p_two / 2

            np.random.seed(42)
            sd_ratios = []
            var_diffs = []
            for _ in range(1000):
                boot_high = np.random.choice(high_data, size=len(high_data), replace=True)
                boot_low = np.random.choice(low_data, size=len(low_data), replace=True)
                sd_low_b = np.std(boot_low, ddof=1)
                sd_ratios.append(np.std(boot_high, ddof=1) / sd_low_b
                                 if sd_low_b > 0 else np.nan)
                var_diffs.append(np.var(boot_high) - np.var(boot_low))

            ci_lower, ci_upper = np.nanpercentile(sd_ratios, [2.5, 97.5])
            # One-tailed bootstrap p: Pr(SD_high <= SD_low) under resampling
            bootstrap_p = float(np.mean(np.array(var_diffs) <= 0))

            config_results[metric] = {
                'group_stats': group_stats,
                'var_ratio': var_ratio,
                'sd_ratio_ci': (ci_lower, ci_upper),
                'levene_stat': levene_stat,
                'levene_p_two': levene_p_two,
                'levene_p_one': levene_p_one,
                'bootstrap_p': bootstrap_p,
                'n_high': len(high_data),
                'n_low': len(low_data),
            }

        sensitivity_results[config_key] = {
            'n_nodes': n_nodes,
            'threshold': threshold,
            'metrics': config_results,
            'is_main': is_main
        }

        status = "*** MAIN ***" if is_main else ""
        report.append(f"\n{config_key} {status}")

        for metric in ['modularity', 'global_efficiency']:
            if metric not in config_results:
                continue
            result = config_results[metric]
            report.append(f"  {metric.upper()}:")
            if is_main:
                for group in SDS_GROUPS:
                    stats = result['group_stats'][group]
                    report.append(f"    {SDS_GROUP_LABELS[group]} (n={stats['n']}): "
                                  f"M = {stats['mean']:.3f}, SD = {stats['std']:.3f}")
            report.append(f"    Variance ratio (High SDS / Low SDS): "
                          f"{result['var_ratio']:.3f}")
            report.append(f"    Bootstrap 95% CI for SD ratio: "
                          f"[{result['sd_ratio_ci'][0]:.3f}, "
                          f"{result['sd_ratio_ci'][1]:.3f}]")
            report.append(f"    Levene's F = {result['levene_stat']:.3f}, "
                          f"two-tailed p = {result['levene_p_two']:.4f}, "
                          f"one-tailed (High>Low) p = {result['levene_p_one']:.4f}")
            report.append(f"    Bootstrap one-tailed p (Var_high <= Var_low): "
                          f"{result['bootstrap_p']:.4f}")

            if result['var_ratio'] > 1 and result['levene_p_one'] < 0.05:
                report.append("    *** SUPPORTS hypothesis: High-SDS more variable")
            else:
                report.append("    No significant variability difference in predicted direction")

    return sensitivity_results


# %%
# =============================================================================
# 4. SENSITIVITY SUMMARY
# =============================================================================

def summarize_sensitivity_results(brain_behavior_results, variability_results,
                                  args, report):
    report.append("\n" + "=" * 80)
    report.append("THRESHOLD SENSITIVITY SUMMARY")
    report.append("=" * 80)

    total_configs = len(brain_behavior_results)

    report.append("\nBRAIN-BEHAVIOUR PEARSON r (coherence check):")
    report.append("-" * 70)
    report.append(f"{'Config':<24} | {'Mod-SDS r':>10} | {'p':>8} | {'Eff-SDS r':>10} | {'p':>8}")
    report.append("-" * 70)
    for config_key, r in brain_behavior_results.items():
        marker = "*" if r['is_main'] else " "
        report.append(f"{config_key:<24} |{marker}{r['mod_r']:>9.3f} | {r['mod_p']:>8.3e}"
                      f" | {r['eff_r']:>10.3f} | {r['eff_p']:>8.3e}")

    report.append("\nVARIABILITY DIFFERENCES (High SDS vs Low SDS, residualised):")
    report.append("-" * 90)
    report.append(f"{'Config':<24} | {'Mod var ratio':>14} | "
                  f"{'Levene p1':>10} | {'Boot p':>8} | "
                  f"{'Eff var ratio':>14} | {'Levene p1':>10} | {'Boot p':>8}")
    report.append("-" * 90)
    for config_key, result in variability_results.items():
        marker = "*" if result['is_main'] else " "
        if 'modularity' in result['metrics'] and 'global_efficiency' in result['metrics']:
            m = result['metrics']['modularity']
            e = result['metrics']['global_efficiency']
            report.append(f"{config_key:<24} |{marker}{m['var_ratio']:>13.2f} | "
                          f"{m['levene_p_one']:>10.4f} | {m['bootstrap_p']:>8.4f} | "
                          f"{e['var_ratio']:>14.2f} | "
                          f"{e['levene_p_one']:>10.4f} | {e['bootstrap_p']:>8.4f}")

    sig_mod = sum(
        1 for r in variability_results.values()
        if 'modularity' in r['metrics']
        and r['metrics']['modularity']['var_ratio'] > 1
        and r['metrics']['modularity']['levene_p_one'] < 0.05
    )
    sig_eff = sum(
        1 for r in variability_results.values()
        if 'global_efficiency' in r['metrics']
        and r['metrics']['global_efficiency']['var_ratio'] > 1
        and r['metrics']['global_efficiency']['levene_p_one'] < 0.05
    )

    report.append(f"\nConsistency across {total_configs} configurations:")
    report.append(f"  Modularity High>Low directional Levene's: {sig_mod}/{total_configs}")
    report.append(f"  Efficiency High>Low directional Levene's: {sig_eff}/{total_configs}")

    return {
        'modularity_var_consistency': sig_mod / total_configs if total_configs else 0.0,
        'efficiency_var_consistency': sig_eff / total_configs if total_configs else 0.0,
    }


# %%
# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def create_main_figure(all_results, brain_behavior_results, variability_results,
                       args, figures_dir):
    main_config_key = f"{args.main_nodes}nodes_{args.main_threshold:.2f}thresh"
    if main_config_key not in all_results:
        print("Main configuration not found - skipping main figure")
        return None

    df = all_results[main_config_key]
    df_groups = df[df['sds_group'].isin(SDS_GROUPS)]

    fig, axes = plt.subplots(2, 2, figsize=(220 * mm2inches, 180 * mm2inches), dpi=FIGURE_DPI)
    colors = [SDS_GROUP_COLORS[g] for g in SDS_GROUPS]
    group_names = [SDS_GROUP_LABELS[g] for g in SDS_GROUPS]

    # Panel A: brain-behaviour scatter (residualised modularity vs SDS)
    ax = axes[0, 0]
    for i, group in enumerate(SDS_GROUPS):
        gd = df_groups[df_groups['sds_group'] == group]
        ax.scatter(gd['Social_Score'], gd['modularity'],
                   alpha=0.6, color=colors[i], label=SDS_GROUP_LABELS[group], s=20)
    z = np.polyfit(df['Social_Score'], df['modularity'], 1)
    xs = np.linspace(df['Social_Score'].min(), df['Social_Score'].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'k-', alpha=0.8, linewidth=1.5)
    bb = brain_behavior_results[main_config_key]
    ax.text(0.05, 0.95, f"r = {bb['mod_r']:.3f}, p = {bb['mod_p']:.2e}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('SDS [z]')
    ax.set_ylabel('Modularity (residualised)')
    ax.set_title('A. Modularity vs SDS (coherence check)')
    ax.legend(fontsize=7)

    # Panel B: modularity SD per SDS group (the primary test)
    ax = axes[0, 1]
    if 'modularity' in variability_results[main_config_key]['metrics']:
        mr = variability_results[main_config_key]['metrics']['modularity']
        sds = [mr['group_stats'][g]['std'] for g in SDS_GROUPS]
        bars = ax.bar(group_names, sds, color=colors, alpha=0.7)
        ax.set_ylabel('Modularity SD (residualised)')
        ax.set_title('B. PRIMARY: Modularity Variability by SDS')
        for bar, sd in zip(bars, sds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{sd:.3f}', ha='center', va='bottom', fontsize=7)
        face = 'yellow' if (mr['var_ratio'] > 1 and mr['levene_p_one'] < 0.05) else 'white'
        ax.text(0.5, 0.98,
                f"High/Low: {mr['var_ratio']:.2f}x var\n"
                f"Levene p1 = {mr['levene_p_one']:.4f}\n"
                f"SD ratio CI [{mr['sd_ratio_ci'][0]:.2f}, {mr['sd_ratio_ci'][1]:.2f}]",
                transform=ax.transAxes, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor=face, alpha=0.8))

    # Panel C: efficiency SD per SDS group
    ax = axes[1, 0]
    if 'global_efficiency' in variability_results[main_config_key]['metrics']:
        er = variability_results[main_config_key]['metrics']['global_efficiency']
        sds = [er['group_stats'][g]['std'] for g in SDS_GROUPS]
        bars = ax.bar(group_names, sds, color=colors, alpha=0.7)
        ax.set_ylabel('Global Efficiency SD (residualised)')
        ax.set_title('C. Efficiency Variability by SDS')
        for bar, sd in zip(bars, sds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{sd:.3f}', ha='center', va='bottom', fontsize=7)
        face = 'yellow' if (er['var_ratio'] > 1 and er['levene_p_one'] < 0.05) else 'white'
        ax.text(0.5, 0.98,
                f"High/Low: {er['var_ratio']:.2f}x var\n"
                f"Levene p1 = {er['levene_p_one']:.4f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor=face, alpha=0.8))

    # Panel D: network organization space (modularity vs efficiency by SDS group)
    ax = axes[1, 1]
    for i, group in enumerate(SDS_GROUPS):
        gd = df_groups[df_groups['sds_group'] == group]
        ax.scatter(gd['global_efficiency'], gd['modularity'],
                   alpha=0.6, color=colors[i], label=SDS_GROUP_LABELS[group], s=20)
    ax.set_xlabel('Global Efficiency (residualised)')
    ax.set_ylabel('Modularity (residualised)')
    ax.set_title('D. Network Organization Space')
    ax.legend(fontsize=7)

    plt.tight_layout()
    output_file = figures_dir / 'C3_landscape_theory_graph_analysis.png'
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Main figure saved to: {output_file}")
    return fig


def create_sensitivity_figure(brain_behavior_results, variability_results,
                              args, figures_dir):
    """Threshold-sensitivity figure (parcellation fixed by C2b)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    thresh_configs = [f"{args.main_nodes}nodes_{t:.2f}thresh" for t in args.thresholds]
    bar_colors = ['lightblue', 'orange', 'lightpink']

    def _star(p):
        if np.isnan(p):
            return ''
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    def _highlight_main(bars, thresholds):
        for bar, t in zip(bars, thresholds):
            if t == args.main_threshold:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)

    # Panel A: Pearson r modularity-SDS across thresholds
    ax = axes[0, 0]
    rs = [brain_behavior_results[c]['mod_r'] if c in brain_behavior_results else np.nan
          for c in thresh_configs]
    ps = [brain_behavior_results[c]['mod_p'] if c in brain_behavior_results else np.nan
          for c in thresh_configs]
    bars = ax.bar(range(len(args.thresholds)), rs,
                  color=bar_colors[:len(args.thresholds)], alpha=0.7)
    ax.set_xticks(range(len(args.thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.thresholds])
    ax.set_xlabel('Edge threshold')
    ax.set_ylabel('Modularity-SDS r')
    ax.set_title(f'A. Brain-Behaviour ({args.main_nodes} nodes)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                _star(p), ha='center', va='bottom', fontweight='bold')
    _highlight_main(bars, args.thresholds)

    # Panel B: modularity variance ratio (High SDS / Low SDS)
    ax = axes[0, 1]
    ratios = [
        variability_results[c]['metrics']['modularity']['var_ratio']
        if c in variability_results and 'modularity' in variability_results[c]['metrics']
        else np.nan for c in thresh_configs
    ]
    ps = [
        variability_results[c]['metrics']['modularity']['levene_p_one']
        if c in variability_results and 'modularity' in variability_results[c]['metrics']
        else np.nan for c in thresh_configs
    ]
    bars = ax.bar(range(len(args.thresholds)), ratios,
                  color=bar_colors[:len(args.thresholds)], alpha=0.7)
    ax.set_xticks(range(len(args.thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.thresholds])
    ax.set_xlabel('Edge threshold')
    ax.set_ylabel('Modularity variance ratio\n(High SDS / Low SDS)')
    ax.set_title('B. Modularity variability')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    for bar, ratio, p in zip(bars, ratios, ps):
        if not np.isnan(ratio):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}\n{_star(p)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    _highlight_main(bars, args.thresholds)

    # Panel C: efficiency variance ratio
    ax = axes[1, 0]
    ratios = [
        variability_results[c]['metrics']['global_efficiency']['var_ratio']
        if c in variability_results and 'global_efficiency' in variability_results[c]['metrics']
        else np.nan for c in thresh_configs
    ]
    ps = [
        variability_results[c]['metrics']['global_efficiency']['levene_p_one']
        if c in variability_results and 'global_efficiency' in variability_results[c]['metrics']
        else np.nan for c in thresh_configs
    ]
    bars = ax.bar(range(len(args.thresholds)), ratios,
                  color=bar_colors[:len(args.thresholds)], alpha=0.7)
    ax.set_xticks(range(len(args.thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.thresholds])
    ax.set_xlabel('Edge threshold')
    ax.set_ylabel('Efficiency variance ratio\n(High SDS / Low SDS)')
    ax.set_title('C. Efficiency variability')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    for bar, ratio, p in zip(bars, ratios, ps):
        if not np.isnan(ratio):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}\n{_star(p)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    _highlight_main(bars, args.thresholds)

    ax = axes[1, 1]
    ax.axis('off')
    total = len(thresh_configs)
    mod_sig = sum(
        1 for c in thresh_configs
        if c in variability_results
        and 'modularity' in variability_results[c]['metrics']
        and variability_results[c]['metrics']['modularity']['var_ratio'] > 1
        and variability_results[c]['metrics']['modularity']['levene_p_one'] < 0.05
    )
    eff_sig = sum(
        1 for c in thresh_configs
        if c in variability_results
        and 'global_efficiency' in variability_results[c]['metrics']
        and variability_results[c]['metrics']['global_efficiency']['var_ratio'] > 1
        and variability_results[c]['metrics']['global_efficiency']['levene_p_one'] < 0.05
    )
    summary_text = (
        f"\nTHRESHOLD SENSITIVITY SUMMARY\n\n"
        f"Parcellation: {args.main_nodes} nodes (fixed by C2b)\n"
        f"Thresholds tested: {len(args.thresholds)}\n\n"
        f"Modularity High>Low (Levene one-tailed): {mod_sig}/{total}\n"
        f"Efficiency High>Low (Levene one-tailed): {eff_sig}/{total}\n\n"
        f"Red borders = main threshold ({args.main_threshold})\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    output_path = figures_dir / 'C4_sensitivity_analysis.png'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity figure saved to: {output_path}")
    return fig


# %%
# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

def save_results(all_results, brain_behavior_results, variability_results,
                 results_dir, report):
    report.append("\n" + "=" * 80)
    report.append("SAVING RESULTS")
    report.append("=" * 80)

    for config_key, df in all_results.items():
        path = results_dir / f'C4_network_metrics_{config_key}.csv'
        df.to_csv(path, index=False)
        report.append(f"Saved: {path.name}")

    summary_df = pd.DataFrame([
        {
            'config': config,
            'n_nodes': r['n_nodes'],
            'threshold': r['threshold'],
            'n_subjects': r['n_subjects'],
            'modularity_sds_r': r['mod_r'],
            'modularity_sds_p': r['mod_p'],
            'efficiency_sds_r': r['eff_r'],
            'efficiency_sds_p': r['eff_p'],
            'is_main_config': r['is_main']
        }
        for config, r in brain_behavior_results.items()
    ])

    for config, r in variability_results.items():
        idx = summary_df['config'] == config
        if 'modularity' in r['metrics']:
            m = r['metrics']['modularity']
            summary_df.loc[idx, 'modularity_var_ratio'] = m['var_ratio']
            summary_df.loc[idx, 'modularity_levene_p_one'] = m['levene_p_one']
            summary_df.loc[idx, 'modularity_bootstrap_p'] = m['bootstrap_p']
        if 'global_efficiency' in r['metrics']:
            e = r['metrics']['global_efficiency']
            summary_df.loc[idx, 'efficiency_var_ratio'] = e['var_ratio']
            summary_df.loc[idx, 'efficiency_levene_p_one'] = e['levene_p_one']
            summary_df.loc[idx, 'efficiency_bootstrap_p'] = e['bootstrap_p']

    summary_path = results_dir / 'C4_sensitivity_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    report.append(f"Saved: {summary_path.name}")

    # Stable downstream filenames consumed by C5, C6, generate_publication_figures
    for config, r in brain_behavior_results.items():
        if r['is_main']:
            for name in ('C3_graph_theory_landscape_results.csv',
                         'C4_main_network_metrics.csv'):
                path = results_dir / name
                all_results[config].to_csv(path, index=False)
                report.append(f"Saved: {path.name}")
            break


# %%
# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SDS-stratified landscape analysis (no PGS join in primary).'
    )
    parser.add_argument('--project', required=True)
    parser.add_argument('--social', required=True,
                        help='Path to CFA social factor scores CSV (column: Social_Score)')
    parser.add_argument('--behavioural', required=True)
    parser.add_argument('--phenotypic', required=True)
    parser.add_argument('--movement', required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--matrices-dir', required=True)
    parser.add_argument('--partition', required=True,
                        help='Community partition CSV from C2b. n_nodes is '
                             'derived from the number of rows.')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.15, 0.20, 0.25])
    parser.add_argument('--main-threshold', type=float, default=0.20)
    parser.add_argument('--motion-threshold', type=float, default=0.2)
    args = parser.parse_args()

    partition_df_for_size = pd.read_csv(args.partition)
    n_nodes_from_partition = len(partition_df_for_size)
    args.parcellations = [n_nodes_from_partition]
    args.main_nodes = n_nodes_from_partition

    project_folder = Path(args.project)
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'
    for d in (figures_dir, reports_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 80,
        "C3: SDS-STRATIFIED LANDSCAPE ANALYSIS",
        "=" * 80,
        "",
        f"Project folder: {project_folder}",
        f"Parcellation (from C2b): {args.main_nodes} nodes",
        f"Main threshold: {args.main_threshold}",
        f"Threshold sensitivity sweep: {args.thresholds}",
        f"Motion threshold: {args.motion_threshold}",
        f"Brain metrics residualised for: {', '.join(COVARIATES)}",
        "  (applied once per threshold config; tests operate on residuals)",
        "PGS is NOT loaded here (kept out of primary analysis to preserve N).",
        "  See C6_exploratory_sds_pgs.py for the SDS~PGS test on the genotyped subset.",
        ""
    ]

    print("=" * 80)
    print("C3: SDS-STRATIFIED LANDSCAPE ANALYSIS")
    print("=" * 80)

    data_by_parcellation = load_and_prepare_data(args, report)
    if not data_by_parcellation:
        report.append("\nERROR: No data loaded. Check file paths.")
        print("No data loaded. Check file paths.")
        return None

    all_results = calculate_network_metrics_all(data_by_parcellation, args, report)
    if not all_results:
        report.append("\nERROR: No network metrics calculated.")
        print("No network metrics calculated.")
        return None

    brain_behavior_results = test_brain_behavior_relationships(all_results, args, report)
    variability_results = test_variability_hypothesis(all_results, args, report)

    consistency = summarize_sensitivity_results(
        brain_behavior_results, variability_results, args, report
    )

    report.append("\n" + "=" * 80)
    report.append("GENERATING FIGURES")
    report.append("=" * 80)
    create_main_figure(all_results, brain_behavior_results, variability_results,
                       args, figures_dir)
    create_sensitivity_figure(brain_behavior_results, variability_results,
                              args, figures_dir)

    save_results(all_results, brain_behavior_results, variability_results,
                 results_dir, report)

    report.append("\n" + "=" * 80)
    report.append("FINAL SUMMARY")
    report.append("=" * 80)
    report.append(f"\nModularity High>Low directional Levene's: "
                  f"{consistency['modularity_var_consistency']:.0%} of configurations")
    report.append(f"Efficiency High>Low directional Levene's: "
                  f"{consistency['efficiency_var_consistency']:.0%} of configurations")

    if consistency['modularity_var_consistency'] >= 0.5:
        report.append("\n*** SDS-STRATIFIED VARIABILITY EFFECT SUPPORTED ***")
    else:
        report.append("\n*** LIMITED SUPPORT FOR SDS-STRATIFIED VARIABILITY EFFECT ***")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_file = reports_dir / 'C3_perform_main_landscape_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_file}")
    print(f"Figures: {figures_dir}/C3_*.png, {figures_dir}/C4_*.png")
    print(f"Results: {results_dir}/C3_*.csv, {results_dir}/C4_*.csv")

    return {
        'all_results': all_results,
        'brain_behavior_results': brain_behavior_results,
        'variability_results': variability_results,
        'consistency': consistency,
    }


# %%
if __name__ == "__main__":
    main()

# %%
