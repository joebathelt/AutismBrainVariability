# %%
"""
C3_perform_main_landscape_analysis.py
Graph Theory Analysis - Testing Landscape Theory

This script tests landscape theory predictions using network-level metrics:
1. Modularity predicts social functioning (establishes brain-behavior relationship)
2. High PGS individuals show more variability in modularity (compensation diversity)
3. Global efficiency remains stable (preserved integration constraint)

The parcellation resolution is fixed upstream by C2b_evaluate_communities.py
(the most consistent / stable parcellation for resting-state community
structure). C3 receives that partition via --partition and derives n_nodes
from its row count. Sensitivity is assessed across edge thresholds only
(default 0.15 / 0.20 / 0.25).

Usage:
    python C3_perform_main_landscape_analysis.py \
        --project <path> \
        --pgs <path> \
        --social <path> \
        --behavioural <path> \
        --phenotypic <path> \
        --movement <path> \
        --ids <path> \
        --matrices-dir <path> \
        --partition <path> \
        --thresholds 0.15 0.20 0.25 \
        --main-threshold 0.20
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, levene, zscore, ttest_ind
import networkx as nx
import bct
import warnings

from utils.covariates import COVARIATES, regress_out_covariates

warnings.filterwarnings('ignore')

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

def load_and_prepare_data(args, report):
    """Load all data for multiple parcellations and create PGS groups."""
    report.append("=" * 80)
    report.append("DATA LOADING AND PREPARATION")
    report.append("=" * 80)

    project_dir = Path(args.project)

    # Load behavioral and genetic data first
    pgs_df = pd.read_csv(args.pgs)
    social_df = pd.read_csv(args.social)
    behavioural_df = pd.read_csv(args.behavioural)
    behavioural_df = behavioural_df[behavioural_df['Gender'] == 'M']
    phenotypic_df = pd.read_csv(args.phenotypic)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    movement_df = pd.read_csv(args.movement)
    ids = pd.read_csv(args.ids, header=None)[0].tolist()

    report.append(f"\nLoaded data files:")
    report.append(f"  PGS data: {len(pgs_df)} subjects")
    report.append(f"  Social scores: {len(social_df)} subjects")
    report.append(f"  Behavioural data: {len(behavioural_df)} subjects")
    report.append(f"  Phenotypic data: {len(phenotypic_df)} subjects")
    report.append(f"  Movement data: {len(movement_df)} subjects")
    report.append(f"  Subject IDs: {len(ids)}")

    # Merge non-connectivity data
    merged_base = pd.merge(pgs_df, social_df[['Subject', 'Social_Score']], on='Subject')
    merged_base = pd.merge(merged_base, behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']], on='Subject')
    merged_base = pd.merge(merged_base, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_base = pd.merge(merged_base, movement_df[['Subject', 'Movement_RelativeRMS_mean']], on='Subject')

    report.append(f"\nAfter merging base data: {len(merged_base)} subjects")

    # Quality control
    n_before_motion = len(merged_base)
    merged_base = merged_base[merged_base['Movement_RelativeRMS_mean'] < args.motion_threshold]
    report.append(f"After motion exclusion (threshold={args.motion_threshold}): {len(merged_base)} subjects")
    report.append(f"  Excluded for motion: {n_before_motion - len(merged_base)}")

    merged_base = merged_base.dropna()
    report.append(f"After removing missing data: {len(merged_base)} subjects")

    # Create PGS groups
    # Groups: low (<-1 SD), middle (-0.5 to 0.5 SD), high (>+1 SD)
    # Subjects between -1 and -0.5, and between 0.5 and 1 are excluded
    merged_base['pgs_z'] = zscore(merged_base['blup_PGS_residuals'])
    merged_base['pgs_group'] = pd.cut(
        merged_base['pgs_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['low', 'exclude_low', 'middle', 'exclude_high', 'high']
    )

    report.append(f"\nPGS group sizes:")
    for group in ['low', 'middle', 'high']:
        n = len(merged_base[merged_base['pgs_group'] == group])
        report.append(f"  {group.capitalize()}: {n}")

    # Load connectivity matrices for each parcellation
    data_by_parcellation = {}
    matrices_dir = Path(args.matrices_dir)

    report.append(f"\nLoading connectivity matrices:")

    for n_nodes in args.parcellations:
        matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'

        try:
            mats_df = pd.read_csv(matrix_file, header=None, sep=r'\s+')
            mats_df.columns = [f'conn_{i+1}' for i in range(mats_df.shape[1])]
            mats_df.index = ids

            # Merge with base data
            merged_df = pd.merge(merged_base, mats_df, left_on='Subject', right_index=True)
            data_by_parcellation[n_nodes] = merged_df

            report.append(f"  {n_nodes} nodes: {len(merged_df)} subjects (matrix shape: {mats_df.shape})")

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

    project_dir = Path(args.project)

    # Determine configurations: threshold sensitivity for main parcellation,
    # single threshold for others
    sensitivity_config = {}
    for n_nodes in args.parcellations:
        if n_nodes == args.main_nodes:
            sensitivity_config[n_nodes] = args.thresholds
        else:
            sensitivity_config[n_nodes] = [args.main_threshold]

    total_configs = sum(len(thresholds) for thresholds in sensitivity_config.values())
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

            # Partition is fixed upstream by C2b
            partition_file = Path(args.partition)
            partition_df = pd.read_csv(partition_file)
            report.append(f"  Using partition from {partition_file.name}")

            results = []
            n_subjects = len(merged_df)

            for i, (subject_id, row) in enumerate(merged_df.iterrows()):
                if (i + 1) % 200 == 0:
                    print(f"    {config_key}: Processed {i+1}/{n_subjects} subjects")

                # Extract connectivity matrix
                conn_data = row[[col for col in row.index if col.startswith('conn_')]].values
                mat = np.reshape(conn_data, (n_nodes, n_nodes)).astype(np.float64)
                mat = mat / 100  # Normalize

                # Apply threshold
                mat = bct.threshold_proportional(mat, threshold)
                mat = np.nan_to_num(mat, nan=0.0)

                # Calculate modularity
                try:
                    if partition_df is not None and len(partition_df) == n_nodes:
                        _, modularity = bct.modularity_und_sign(mat, partition_df['community_id'].values)
                    else:
                        _, modularity = bct.modularity_und_sign(mat)
                except Exception:
                    modularity = np.nan

                # Calculate global efficiency
                try:
                    mat_pos = mat.copy()
                    mat_pos[mat_pos < 0] = 0
                    G = nx.from_numpy_array(mat_pos)
                    global_efficiency = nx.global_efficiency(G)
                except Exception:
                    global_efficiency = np.nan

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
                    'n_nodes': n_nodes,
                    'threshold': threshold,
                    'config': config_key
                })

            config_df = pd.DataFrame(results)
            n_before = len(config_df)
            config_df = config_df.dropna(subset=['modularity', 'global_efficiency'])

            # Residualise brain metrics for age, sex, ICV, and head motion.
            # Keep raw values under *_raw for audit; all downstream tests read
            # the 'modularity' / 'global_efficiency' columns, so they now
            # operate on residuals (matches C1's covariate handling).
            config_df['modularity_raw'] = config_df['modularity']
            config_df['global_efficiency_raw'] = config_df['global_efficiency']
            config_df[['modularity', 'global_efficiency']] = regress_out_covariates(
                config_df[['modularity', 'global_efficiency']],
                config_df[list(COVARIATES)],
            )

            all_results[config_key] = config_df
            report.append(f"  Final n = {len(config_df)} (excluded {n_before - len(config_df)} with missing metrics)")
            report.append(f"  Residualised modularity & global_efficiency for: {', '.join(COVARIATES)}")

    return all_results


# %%
# =============================================================================
# 3. MAIN ANALYSES
# =============================================================================

def test_brain_behavior_relationships(all_results, args, report):
    """Test brain-behavior relationships across all configurations."""
    report.append("\n" + "=" * 80)
    report.append("BRAIN-BEHAVIOR RELATIONSHIPS")
    report.append("=" * 80)

    sensitivity_results = {}

    for config_key, df in all_results.items():
        if len(df) == 0:
            report.append(f"\nWARNING: {config_key} has no data - skipping")
            continue
        n_nodes = df['n_nodes'].iloc[0]
        threshold = df['threshold'].iloc[0]

        # Overall sample correlations
        r_mod, p_mod = pearsonr(df['modularity'], df['Social_Score'])
        r_eff, p_eff = pearsonr(df['global_efficiency'], df['Social_Score'])

        is_main = (n_nodes == args.main_nodes and threshold == args.main_threshold)

        sensitivity_results[config_key] = {
            'n_nodes': n_nodes,
            'threshold': threshold,
            'n_subjects': len(df),
            'mod_r': r_mod,
            'mod_p': p_mod,
            'eff_r': r_eff,
            'eff_p': p_eff,
            'is_main': is_main
        }

        status = "*** MAIN ***" if is_main else ""
        report.append(f"\n{config_key} {status}")
        report.append(f"  n = {len(df)}")
        report.append(f"  Modularity-Social: r = {r_mod:.3f}, p = {p_mod:.3e}")
        report.append(f"  Efficiency-Social: r = {r_eff:.3f}, p = {p_eff:.3e}")

    return sensitivity_results


def test_variability_hypothesis(all_results, args, report):
    """Test variability differences across PGS groups for all configurations."""
    report.append("\n" + "=" * 80)
    report.append("VARIABILITY ANALYSIS")
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
            # Get group data
            high_data = df[df['pgs_group'] == 'high'][metric].values
            low_data = df[df['pgs_group'] == 'low'][metric].values
            middle_data = df[df['pgs_group'] == 'middle'][metric].values

            if len(high_data) < 10 or len(low_data) < 10:
                continue

            # Group statistics
            group_stats = {}
            for group, data in [('low', low_data), ('middle', middle_data), ('high', high_data)]:
                group_stats[group] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'var': np.var(data),
                    'n': len(data)
                }

            # Variance ratio
            var_ratio = np.var(high_data) / np.var(low_data)

            # Levene's test
            levene_stat, levene_p = levene(high_data, low_data)

            # Bootstrap test
            np.random.seed(42)
            var_diffs = []
            for _ in range(1000):
                boot_high = np.random.choice(high_data, size=len(high_data), replace=True)
                boot_low = np.random.choice(low_data, size=len(low_data), replace=True)
                var_diffs.append(np.var(boot_high) - np.var(boot_low))

            ci_lower, ci_upper = np.percentile(var_diffs, [2.5, 97.5])
            bootstrap_p = np.mean(np.array(var_diffs) <= 0)

            config_results[metric] = {
                'group_stats': group_stats,
                'var_ratio': var_ratio,
                'levene_stat': levene_stat,
                'levene_p': levene_p,
                'bootstrap_p': bootstrap_p,
                'ci': (ci_lower, ci_upper),
                'n_high': len(high_data),
                'n_low': len(low_data)
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
            if metric in config_results:
                result = config_results[metric]
                report.append(f"  {metric.upper()}:")
                if is_main:
                    for group in ['low', 'middle', 'high']:
                        stats = result['group_stats'][group]
                        report.append(f"    {group.capitalize()}: M = {stats['mean']:.3f}, SD = {stats['std']:.3f}")
                report.append(f"    Variance ratio (High/Low): {result['var_ratio']:.3f}")
                report.append(f"    Levene's test: F = {result['levene_stat']:.3f}, p = {result['levene_p']:.3f}")
                report.append(f"    Bootstrap 95% CI: [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]")
                report.append(f"    Bootstrap p-value: {result['bootstrap_p']:.3f}")

                if result['bootstrap_p'] < 0.05 and result['var_ratio'] > 1:
                    report.append(f"    *** SUPPORTS landscape theory: High PGS more variable")
                elif result['bootstrap_p'] >= 0.05:
                    report.append(f"    *** No significant variability difference")

    return sensitivity_results


def test_compensation_strategies(all_results, args, report):
    """Test for different compensation strategies in high PGS group (main config only)."""
    report.append("\n" + "=" * 80)
    report.append("COMPENSATION STRATEGY ANALYSIS")
    report.append("=" * 80)

    main_config_key = f"{args.main_nodes}nodes_{args.main_threshold:.2f}thresh"
    if main_config_key not in all_results:
        report.append("Main configuration not found - skipping compensation analysis")
        return None

    df = all_results[main_config_key]
    high_pgs = df[df['pgs_group'] == 'high'].copy()

    if len(high_pgs) < 20:
        report.append(f"Insufficient high PGS subjects ({len(high_pgs)}) for strategy analysis")
        return None

    # Split high PGS group by modularity (median split)
    med_mod = high_pgs['modularity'].median()

    high_mod_strategy = high_pgs[high_pgs['modularity'] > med_mod]
    low_mod_strategy = high_pgs[high_pgs['modularity'] <= med_mod]

    report.append(f"High PGS group (n={len(high_pgs)}) split by modularity:")
    report.append("")
    report.append(f"High Modularity Strategy (n={len(high_mod_strategy)}):")
    report.append(f"  Modularity: {high_mod_strategy['modularity'].mean():.3f} +/- {high_mod_strategy['modularity'].std():.3f}")
    report.append(f"  Efficiency: {high_mod_strategy['global_efficiency'].mean():.3f} +/- {high_mod_strategy['global_efficiency'].std():.3f}")
    report.append(f"  Social Score: {high_mod_strategy['Social_Score'].mean():.3f} +/- {high_mod_strategy['Social_Score'].std():.3f}")

    report.append("")
    report.append(f"Low Modularity Strategy (n={len(low_mod_strategy)}):")
    report.append(f"  Modularity: {low_mod_strategy['modularity'].mean():.3f} +/- {low_mod_strategy['modularity'].std():.3f}")
    report.append(f"  Efficiency: {low_mod_strategy['global_efficiency'].mean():.3f} +/- {low_mod_strategy['global_efficiency'].std():.3f}")
    report.append(f"  Social Score: {low_mod_strategy['Social_Score'].mean():.3f} +/- {low_mod_strategy['Social_Score'].std():.3f}")

    # Test if strategies achieve similar social outcomes
    t_stat, p_val = ttest_ind(high_mod_strategy['Social_Score'], low_mod_strategy['Social_Score'])

    report.append("")
    report.append(f"Social outcome comparison (High vs Low Modularity strategies):")
    report.append(f"  t = {t_stat:.3f}, p = {p_val:.3f}")

    if p_val > 0.05:
        report.append("  *** Different strategies achieve similar outcomes - SUPPORTS compensation hypothesis")
    else:
        report.append("  *** Strategies differ in outcomes")

    return {
        'high_mod_strategy': high_mod_strategy,
        'low_mod_strategy': low_mod_strategy,
        'social_comparison': {'t': t_stat, 'p': p_val}
    }


def test_variability_alternative_groups(all_results, args, report):
    """Test variability using alternative group definitions (low < -0.75, high > 0.75)."""
    report.append("\n" + "=" * 80)
    report.append("VARIABILITY ANALYSIS (ALTERNATIVE GROUP DEFINITIONS)")
    report.append("Groups: Low PGS < -0.75 SD, Middle -0.5 to 0.5 SD, High PGS > 0.75 SD")
    report.append("=" * 80)

    sensitivity_results = {}

    for config_key, df in all_results.items():
        if len(df) == 0:
            report.append(f"\nWARNING: {config_key} has no data - skipping")
            continue
        n_nodes = df['n_nodes'].iloc[0]
        threshold = df['threshold'].iloc[0]
        is_main = (n_nodes == args.main_nodes and threshold == args.main_threshold)

        # Assign alternative groups
        alt_group = pd.cut(
            df['pgs_z'],
            bins=[-np.inf, -0.75, -0.5, 0.5, 0.75, np.inf],
            labels=['low', 'exclude_low', 'middle', 'exclude_high', 'high']
        )

        config_results = {}

        for metric in ['modularity', 'global_efficiency']:
            high_data = df[alt_group == 'high'][metric].values
            low_data = df[alt_group == 'low'][metric].values
            middle_data = df[alt_group == 'middle'][metric].values

            if len(high_data) < 10 or len(low_data) < 10:
                continue

            # Group statistics
            group_stats = {}
            for group, data in [('low', low_data), ('middle', middle_data), ('high', high_data)]:
                group_stats[group] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'var': np.var(data),
                    'n': len(data)
                }

            # Variance ratio
            var_ratio = np.var(high_data) / np.var(low_data)

            # Levene's test
            levene_stat, levene_p = levene(high_data, low_data)

            # Bootstrap test
            np.random.seed(42)
            var_diffs = []
            for _ in range(1000):
                boot_high = np.random.choice(high_data, size=len(high_data), replace=True)
                boot_low = np.random.choice(low_data, size=len(low_data), replace=True)
                var_diffs.append(np.var(boot_high) - np.var(boot_low))

            ci_lower, ci_upper = np.percentile(var_diffs, [2.5, 97.5])
            bootstrap_p = np.mean(np.array(var_diffs) <= 0)

            config_results[metric] = {
                'group_stats': group_stats,
                'var_ratio': var_ratio,
                'levene_stat': levene_stat,
                'levene_p': levene_p,
                'bootstrap_p': bootstrap_p,
                'ci': (ci_lower, ci_upper),
                'n_high': len(high_data),
                'n_low': len(low_data)
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
            if metric in config_results:
                result = config_results[metric]
                report.append(f"  {metric.upper()}:")
                report.append(f"    Low (n={result['n_low']}): M = {result['group_stats']['low']['mean']:.3f}, SD = {result['group_stats']['low']['std']:.3f}")
                report.append(f"    High (n={result['n_high']}): M = {result['group_stats']['high']['mean']:.3f}, SD = {result['group_stats']['high']['std']:.3f}")
                report.append(f"    Variance ratio (High/Low): {result['var_ratio']:.3f}")
                report.append(f"    Levene's test: F = {result['levene_stat']:.3f}, p = {result['levene_p']:.3f}")
                report.append(f"    Bootstrap 95% CI: [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]")
                report.append(f"    Bootstrap p-value: {result['bootstrap_p']:.3f}")

                if result['bootstrap_p'] < 0.05 and result['var_ratio'] > 1:
                    report.append(f"    *** SUPPORTS landscape theory: High PGS more variable")
                elif result['bootstrap_p'] >= 0.05:
                    report.append(f"    *** No significant variability difference")

    return sensitivity_results


# %%
# =============================================================================
# 4. SENSITIVITY SUMMARY
# =============================================================================

def summarize_sensitivity_results(brain_behavior_results, variability_results,
                                  variability_alt_results, args, report):
    """Create summary of sensitivity analysis results."""
    report.append("\n" + "=" * 80)
    report.append("SENSITIVITY ANALYSIS SUMMARY")
    report.append("=" * 80)

    total_configs = len(brain_behavior_results)

    # Brain-behavior relationships summary
    report.append("\nBRAIN-BEHAVIOR RELATIONSHIPS:")
    report.append("-" * 70)
    report.append(f"{'Config':<24} | {'Modularity-Social r':>18} | {'p-value':>10} | {'Significant':>10}")
    report.append("-" * 70)

    for config_key, result in brain_behavior_results.items():
        significant = "YES" if result['mod_p'] < 0.05 else "NO"
        main_marker = "*" if result['is_main'] else " "
        report.append(f"{config_key:<24} |{main_marker}{result['mod_r']:>17.3f} | {result['mod_p']:>10.3e} | {significant:>10}")

    # Variability summary (main groups: low -1.0 to -0.5, high 0.5 to 1.0)
    report.append(f"\nVARIABILITY DIFFERENCES - MAIN GROUPS (low: -1.0 to -0.5 SD, high: 0.5 to 1.0 SD):")
    report.append("-" * 80)
    report.append(f"{'Config':<24} | {'Mod Ratio':>10} | {'p-value':>8} | {'Eff Ratio':>10} | {'p-value':>8}")
    report.append("-" * 80)

    for config_key, result in variability_results.items():
        main_marker = "*" if result['is_main'] else " "

        if 'modularity' in result['metrics'] and 'global_efficiency' in result['metrics']:
            mod_ratio = result['metrics']['modularity']['var_ratio']
            mod_p = result['metrics']['modularity']['bootstrap_p']
            eff_ratio = result['metrics']['global_efficiency']['var_ratio']
            eff_p = result['metrics']['global_efficiency']['bootstrap_p']

            report.append(f"{config_key:<24} |{main_marker}{mod_ratio:>9.2f} | {mod_p:>8.3f} | {eff_ratio:>10.2f} | {eff_p:>8.3f}")

    # Variability summary (alternative groups: low < -0.75, high > 0.75)
    report.append(f"\nVARIABILITY DIFFERENCES - ALTERNATIVE GROUPS (low: < -0.75 SD, high: > 0.75 SD):")
    report.append("-" * 80)
    report.append(f"{'Config':<24} | {'Mod Ratio':>10} | {'p-value':>8} | {'Eff Ratio':>10} | {'p-value':>8}")
    report.append("-" * 80)

    for config_key, result in variability_alt_results.items():
        main_marker = "*" if result['is_main'] else " "

        if 'modularity' in result['metrics'] and 'global_efficiency' in result['metrics']:
            mod_ratio = result['metrics']['modularity']['var_ratio']
            mod_p = result['metrics']['modularity']['bootstrap_p']
            eff_ratio = result['metrics']['global_efficiency']['var_ratio']
            eff_p = result['metrics']['global_efficiency']['bootstrap_p']

            report.append(f"{config_key:<24} |{main_marker}{mod_ratio:>9.2f} | {mod_p:>8.3f} | {eff_ratio:>10.2f} | {eff_p:>8.3f}")

    # Count significant results
    sig_brain_behavior = sum(1 for r in brain_behavior_results.values() if r['mod_p'] < 0.05)
    sig_mod_var = sum(1 for r in variability_results.values()
                     if 'modularity' in r['metrics'] and r['metrics']['modularity']['bootstrap_p'] < 0.05)
    sig_eff_var = sum(1 for r in variability_results.values()
                     if 'global_efficiency' in r['metrics'] and r['metrics']['global_efficiency']['bootstrap_p'] < 0.05)

    sig_mod_var_alt = sum(1 for r in variability_alt_results.values()
                         if 'modularity' in r['metrics'] and r['metrics']['modularity']['bootstrap_p'] < 0.05)
    sig_eff_var_alt = sum(1 for r in variability_alt_results.values()
                         if 'global_efficiency' in r['metrics'] and r['metrics']['global_efficiency']['bootstrap_p'] < 0.05)

    report.append(f"\nOVERALL CONCLUSIONS:")
    report.append(f"  Brain-behavior relationships: {sig_brain_behavior}/{total_configs} configurations significant")
    report.append(f"  Modularity variability (main groups): {sig_mod_var}/{total_configs} configurations significant")
    report.append(f"  Efficiency variability (main groups): {sig_eff_var}/{total_configs} configurations significant")
    report.append(f"  Modularity variability (alt groups):  {sig_mod_var_alt}/{total_configs} configurations significant")
    report.append(f"  Efficiency variability (alt groups):  {sig_eff_var_alt}/{total_configs} configurations significant")

    return {
        'brain_behavior_consistency': sig_brain_behavior / total_configs,
        'modularity_var_consistency': sig_mod_var / total_configs,
        'efficiency_var_consistency': sig_eff_var / total_configs,
        'modularity_var_alt_consistency': sig_mod_var_alt / total_configs,
        'efficiency_var_alt_consistency': sig_eff_var_alt / total_configs,
    }


# %%
# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def create_main_figure(all_results, brain_behavior_results, variability_results,
                       compensation_results, args, figures_dir):
    """Create comprehensive figure showing main analysis findings."""
    main_config_key = f"{args.main_nodes}nodes_{args.main_threshold:.2f}thresh"
    if main_config_key not in all_results:
        print("Main configuration not found - skipping main figure")
        return None

    df = all_results[main_config_key]
    # Filter to PGS groups for plotting
    df_groups = df[df['pgs_group'].isin(['low', 'middle', 'high'])]

    fig, axes = plt.subplots(2, 3, figsize=(280 * mm2inches, 180 * mm2inches), dpi=FIGURE_DPI)
    colors = ['#3498db', '#f39c12', '#e74c3c']  # Blue, Orange, Red

    # 1. Brain-behavior relationships
    ax = axes[0, 0]
    for i, group in enumerate(['low', 'middle', 'high']):
        group_data = df_groups[df_groups['pgs_group'] == group]
        ax.scatter(group_data['modularity'], group_data['Social_Score'],
                  alpha=0.6, color=colors[i], label=f'{group.capitalize()} PGS', s=20)

    # Overall regression line (full sample)
    z = np.polyfit(df['modularity'], df['Social_Score'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['modularity'].min(), df['modularity'].max(), 100)
    ax.plot(x_range, p(x_range), 'k-', alpha=0.8, linewidth=1.5)

    overall_r = brain_behavior_results[main_config_key]['mod_r']
    overall_p = brain_behavior_results[main_config_key]['mod_p']
    ax.text(0.05, 0.95, f'r = {overall_r:.3f}, p = {overall_p:.2e}',
           transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           verticalalignment='top')

    ax.set_xlabel('Modularity (residualised)')
    ax.set_ylabel('Social Score')
    ax.set_title('Finding 1: Modularity-Social\nFunctioning Association')
    ax.legend(fontsize=7)

    # 2. Modularity variability
    ax = axes[0, 1]
    group_names = ['Low', 'Middle', 'High']

    if main_config_key in variability_results and 'modularity' in variability_results[main_config_key]['metrics']:
        mod_result = variability_results[main_config_key]['metrics']['modularity']
        mod_stds = [mod_result['group_stats'][g]['std'] for g in ['low', 'middle', 'high']]

        bars = ax.bar(group_names, mod_stds, color=colors, alpha=0.7)
        ax.set_ylabel('Modularity SD (residualised)')
        ax.set_title('Finding 2: Modularity Variability\nby PGS Group')

        var_ratio = mod_result['var_ratio']
        p_val = mod_result['bootstrap_p']
        ax.text(0.5, 0.95, f'High vs Low:\n{var_ratio:.2f}x variance\np = {p_val:.3f}',
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round',
                        facecolor='yellow' if p_val < 0.05 else 'white', alpha=0.8))

        for bar, std in zip(bars, mod_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=7)

    # 3. Efficiency variability
    ax = axes[0, 2]

    if main_config_key in variability_results and 'global_efficiency' in variability_results[main_config_key]['metrics']:
        eff_result = variability_results[main_config_key]['metrics']['global_efficiency']
        eff_stds = [eff_result['group_stats'][g]['std'] for g in ['low', 'middle', 'high']]

        bars = ax.bar(group_names, eff_stds, color=colors, alpha=0.7)
        ax.set_ylabel('Global Efficiency SD (residualised)')
        ax.set_title('Finding 3: Efficiency Variability\n(No Group Differences)')

        var_ratio = eff_result['var_ratio']
        p_val = eff_result['bootstrap_p']
        ax.text(0.5, 0.95, f'High vs Low:\n{var_ratio:.2f}x variance\np = {p_val:.3f}',
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        for bar, std in zip(bars, eff_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=7)

    # 4. Network organization space
    ax = axes[1, 0]
    for i, group in enumerate(['low', 'middle', 'high']):
        group_data = df_groups[df_groups['pgs_group'] == group]
        ax.scatter(group_data['global_efficiency'], group_data['modularity'],
                  alpha=0.6, color=colors[i], label=f'{group.capitalize()} PGS', s=20)

    ax.set_xlabel('Global Efficiency (residualised)')
    ax.set_ylabel('Modularity (residualised)')
    ax.set_title('Network Organization Space')
    ax.legend(fontsize=7)

    # 5. High PGS compensation strategies
    ax = axes[1, 1]
    if compensation_results is not None:
        high_mod = compensation_results['high_mod_strategy']
        low_mod = compensation_results['low_mod_strategy']

        ax.scatter(high_mod['global_efficiency'], high_mod['Social_Score'],
                  color='red', alpha=0.7, label='High Modularity Strategy', s=30)
        ax.scatter(low_mod['global_efficiency'], low_mod['Social_Score'],
                  color='blue', alpha=0.7, label='Low Modularity Strategy', s=30)

        ax.set_xlabel('Global Efficiency (residualised)')
        ax.set_ylabel('Social Score')
        ax.set_title('High PGS: Different Strategies\nSimilar Outcomes')
        ax.legend(fontsize=7)

        p_val = compensation_results['social_comparison']['p']
        ax.text(0.05, 0.95, f'Strategy comparison:\np = {p_val:.3f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round',
                        facecolor='lightgreen' if p_val > 0.05 else 'white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Insufficient high PGS\nsubjects for strategy\nanalysis',
               transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.set_title('Compensation Strategy Analysis')

    # 6. Empty subplot
    ax = axes[1, 2]
    ax.axis('off')

    plt.tight_layout()
    output_file = figures_dir / 'C3_landscape_theory_graph_analysis.png'
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"Main figure saved to: {output_file}")
    return fig


def create_sensitivity_figure(brain_behavior_results, variability_results,
                              all_results, args, figures_dir):
    """Threshold-sensitivity figure (parcellation is fixed upstream by C2b)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    configs = list(brain_behavior_results.keys())
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

    # Panel A: Brain-behavior correlation across thresholds
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
    ax.set_ylabel('Modularity-Social r')
    ax.set_title(f'A. Brain-Behavior ({args.main_nodes} nodes)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                _star(p), ha='center', va='bottom', fontweight='bold')
    _highlight_main(bars, args.thresholds)

    # Panel B: Modularity variance ratio across thresholds
    ax = axes[0, 1]
    ratios = [
        variability_results[c]['metrics']['modularity']['var_ratio']
        if c in variability_results and 'modularity' in variability_results[c]['metrics']
        else np.nan
        for c in thresh_configs
    ]
    ps = [
        variability_results[c]['metrics']['modularity']['bootstrap_p']
        if c in variability_results and 'modularity' in variability_results[c]['metrics']
        else np.nan
        for c in thresh_configs
    ]
    bars = ax.bar(range(len(args.thresholds)), ratios,
                  color=bar_colors[:len(args.thresholds)], alpha=0.7)
    ax.set_xticks(range(len(args.thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.thresholds])
    ax.set_xlabel('Edge threshold')
    ax.set_ylabel('Modularity variance ratio\n(High vs Low PGS)')
    ax.set_title('B. Modularity variability')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    for bar, ratio, p in zip(bars, ratios, ps):
        if not np.isnan(ratio):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}\n{_star(p)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    _highlight_main(bars, args.thresholds)

    # Panel C: Global efficiency variance ratio across thresholds
    ax = axes[1, 0]
    ratios = [
        variability_results[c]['metrics']['global_efficiency']['var_ratio']
        if c in variability_results and 'global_efficiency' in variability_results[c]['metrics']
        else np.nan
        for c in thresh_configs
    ]
    ps = [
        variability_results[c]['metrics']['global_efficiency']['bootstrap_p']
        if c in variability_results and 'global_efficiency' in variability_results[c]['metrics']
        else np.nan
        for c in thresh_configs
    ]
    bars = ax.bar(range(len(args.thresholds)), ratios,
                  color=bar_colors[:len(args.thresholds)], alpha=0.7)
    ax.set_xticks(range(len(args.thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.thresholds])
    ax.set_xlabel('Edge threshold')
    ax.set_ylabel('Efficiency variance ratio\n(High vs Low PGS)')
    ax.set_title('C. Efficiency variability')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    for bar, ratio, p in zip(bars, ratios, ps):
        if not np.isnan(ratio):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}\n{_star(p)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    _highlight_main(bars, args.thresholds)

    # Panel D: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    total_configs = len(configs)
    bb_sig = sum(1 for c in configs if brain_behavior_results[c]['mod_p'] < 0.05)
    mod_var_sig = sum(
        1 for c in configs
        if 'modularity' in variability_results[c]['metrics']
        and variability_results[c]['metrics']['modularity']['bootstrap_p'] < 0.05
    )
    eff_var_sig = sum(
        1 for c in configs
        if 'global_efficiency' in variability_results[c]['metrics']
        and variability_results[c]['metrics']['global_efficiency']['bootstrap_p'] < 0.05
    )

    bb_consistency = bb_sig / total_configs
    mod_consistency = mod_var_sig / total_configs
    eff_consistency = eff_var_sig / total_configs

    if bb_consistency >= 0.6 and mod_consistency >= 0.8 and eff_consistency < 0.4:
        interpretation, bg = "SUPPORTED", 'lightgreen'
    elif mod_consistency >= 0.8:
        interpretation, bg = "PARTIALLY SUPPORTED", 'lightyellow'
    else:
        interpretation, bg = "LIMITED SUPPORT", 'lightcoral'

    summary_text = f"""
THRESHOLD SENSITIVITY SUMMARY

Parcellation: {args.main_nodes} nodes (fixed by C2b)
Thresholds tested: {len(args.thresholds)}

Results consistency:
- Brain-behavior:           {bb_consistency:.0%} ({bb_sig}/{total_configs})
- Modularity variability:   {mod_consistency:.0%} ({mod_var_sig}/{total_configs})
- Efficiency variability:   {eff_consistency:.0%} ({eff_var_sig}/{total_configs})

CONCLUSION: {interpretation}

Red borders = main threshold ({args.main_threshold})
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=bg, alpha=0.7))

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

def save_results(all_results, brain_behavior_results, variability_results, results_dir, report):
    """Save all results to files."""
    report.append("\n" + "=" * 80)
    report.append("SAVING RESULTS")
    report.append("=" * 80)

    # Save individual configuration results
    for config_key, df in all_results.items():
        output_path = results_dir / f'C4_network_metrics_{config_key}.csv'
        df.to_csv(output_path, index=False)
        report.append(f"Saved: {output_path.name}")

    # Save sensitivity analysis summary
    sensitivity_summary_df = pd.DataFrame([
        {
            'config': config,
            'n_nodes': result['n_nodes'],
            'threshold': result['threshold'],
            'n_subjects': result['n_subjects'],
            'modularity_social_r': result['mod_r'],
            'modularity_social_p': result['mod_p'],
            'efficiency_social_r': result['eff_r'],
            'efficiency_social_p': result['eff_p'],
            'is_main_config': result['is_main']
        }
        for config, result in brain_behavior_results.items()
    ])

    # Add variability results
    for config, result in variability_results.items():
        idx = sensitivity_summary_df['config'] == config
        if 'modularity' in result['metrics']:
            sensitivity_summary_df.loc[idx, 'modularity_var_ratio'] = result['metrics']['modularity']['var_ratio']
            sensitivity_summary_df.loc[idx, 'modularity_var_p'] = result['metrics']['modularity']['bootstrap_p']
        if 'global_efficiency' in result['metrics']:
            sensitivity_summary_df.loc[idx, 'efficiency_var_ratio'] = result['metrics']['global_efficiency']['var_ratio']
            sensitivity_summary_df.loc[idx, 'efficiency_var_p'] = result['metrics']['global_efficiency']['bootstrap_p']

    summary_path = results_dir / 'C4_sensitivity_summary.csv'
    sensitivity_summary_df.to_csv(summary_path, index=False)
    report.append(f"Saved: {summary_path.name}")

    # Also save main config results at fixed paths consumed by C5 and
    # generate_publication_figures. C3_graph_theory_landscape_results.csv
    # is kept for backwards compatibility; C4_main_network_metrics.csv is the
    # stable downstream name (independent of the C2b-selected parcellation).
    for config, result in brain_behavior_results.items():
        if result['is_main']:
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
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Run graph theory analysis testing landscape theory with sensitivity analyses'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--pgs', required=True,
                        help='Path to PGS/PGS residuals CSV')
    parser.add_argument('--social', required=True,
                        help='Path to social factor scores CSV')
    parser.add_argument('--behavioural', required=True,
                        help='Path to behavioural data CSV')
    parser.add_argument('--phenotypic', required=True,
                        help='Path to phenotypic data CSV')
    parser.add_argument('--movement', required=True,
                        help='Path to movement data CSV')
    parser.add_argument('--ids', required=True,
                        help='Path to subject IDs file')
    parser.add_argument('--matrices-dir', required=True,
                        help='Path to connectivity matrices directory')
    parser.add_argument('--partition', required=True,
                        help='Path to community partition CSV (selected by C2b). '
                             'n_nodes is derived from the number of rows.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.15, 0.20, 0.25],
                        help='Edge thresholds for sensitivity sweep (default: 0.15 0.20 0.25)')
    parser.add_argument('--main-threshold', type=float, default=0.20,
                        help='Main analysis threshold (default: 0.20)')
    parser.add_argument('--motion-threshold', type=float, default=0.2,
                        help='Motion threshold for subject exclusion (default: 0.2)')
    args = parser.parse_args()

    # Derive parcellation resolution from the C2b-selected partition
    partition_df_for_size = pd.read_csv(args.partition)
    n_nodes_from_partition = len(partition_df_for_size)
    args.parcellations = [n_nodes_from_partition]
    args.main_nodes = n_nodes_from_partition

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'

    for dir_path in [figures_dir, reports_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = [
        "=" * 80,
        "C3: GRAPH THEORY ANALYSIS - LANDSCAPE THEORY TEST",
        "=" * 80,
        "",
        f"Project folder: {project_folder}",
        f"Parcellation (from C2b): {args.main_nodes} nodes",
        f"Main threshold: {args.main_threshold}",
        f"Threshold sensitivity sweep: {args.thresholds}",
        f"Motion threshold: {args.motion_threshold}",
        f"Brain metrics residualised for: {', '.join(COVARIATES)}",
        "  (applied once per threshold config; all association and",
        "   variability tests operate on residuals)",
        ""
    ]

    print("=" * 80)
    print("C3: GRAPH THEORY ANALYSIS - LANDSCAPE THEORY TEST")
    print("=" * 80)

    # Load data for all parcellations
    data_by_parcellation = load_and_prepare_data(args, report)

    if not data_by_parcellation:
        report.append("\nERROR: No data loaded. Please check file paths.")
        print("No data loaded. Please check file paths.")
        return None

    # Calculate network metrics for all configurations
    all_results = calculate_network_metrics_all(data_by_parcellation, args, report)

    if not all_results:
        report.append("\nERROR: No network metrics calculated.")
        print("No network metrics calculated. Please check data and parameters.")
        return None

    # Run brain-behavior analysis on full sample for all configurations
    brain_behavior_results = test_brain_behavior_relationships(all_results, args, report)

    # Run variability analysis (PGS groups) for all configurations
    variability_results = test_variability_hypothesis(all_results, args, report)

    # Run compensation strategy analysis (main config only)
    compensation_results = test_compensation_strategies(all_results, args, report)

    # Run variability analysis with alternative group definitions
    variability_alt_results = test_variability_alternative_groups(all_results, args, report)

    # Summarize sensitivity results
    consistency_summary = summarize_sensitivity_results(
        brain_behavior_results, variability_results, variability_alt_results, args, report
    )

    # Create visualizations
    report.append("\n" + "=" * 80)
    report.append("GENERATING FIGURES")
    report.append("=" * 80)

    create_main_figure(all_results, brain_behavior_results, variability_results,
                      compensation_results, args, figures_dir)
    create_sensitivity_figure(brain_behavior_results, variability_results,
                             all_results, args, figures_dir)

    # Save all results
    save_results(all_results, brain_behavior_results, variability_results, results_dir, report)

    # Final summary
    report.append("\n" + "=" * 80)
    report.append("FINAL SUMMARY")
    report.append("=" * 80)
    report.append(f"\nBrain-behavior consistency: {consistency_summary['brain_behavior_consistency']:.1%}")
    report.append(f"Modularity variability consistency: {consistency_summary['modularity_var_consistency']:.1%}")
    report.append(f"Efficiency variability consistency: {consistency_summary['efficiency_var_consistency']:.1%}")

    if (consistency_summary['modularity_var_consistency'] >= 0.8 and
        consistency_summary['brain_behavior_consistency'] >= 0.6):
        report.append("\n*** LANDSCAPE THEORY SUPPORTED ***")
        report.append("Strong modularity variability effects across thresholds")
        report.append(f"Brain-behavior relationships robust at {args.main_nodes}-node parcellation")
    elif consistency_summary['modularity_var_consistency'] >= 0.8:
        report.append("\n*** CORE LANDSCAPE THEORY SUPPORTED ***")
        report.append("Robust modularity-based compensation mechanism")
    else:
        report.append("\n*** LIMITED SUPPORT FOR LANDSCAPE THEORY ***")
        report.append("Results may be threshold-dependent")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Write report to file
    report_file = reports_dir / 'C3_perform_main_landscape_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_file}")
    print(f"Figures saved to: {figures_dir}/C3_*.png, {figures_dir}/C4_*.png")
    print(f"Results saved to: {results_dir}/C3_*.csv, {results_dir}/C4_*.csv")

    return {
        'all_results': all_results,
        'brain_behavior_results': brain_behavior_results,
        'variability_results': variability_results,
        'variability_alt_results': variability_alt_results,
        'compensation_results': compensation_results,
        'consistency_summary': consistency_summary
    }


# %%
if __name__ == "__main__":
    main()

# %%
