# %%
# ==============================================================================
# Title: C4_perform_main_landscape_analysis.py
# ==============================================================================
# Description: This script performs the main analyses to test landscape
# theory predictions using network-level metrics (modularity and global
# efficiency). It includes brain-behavior relationships, variability analyses,
# and compensation strategy evaluations.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, levene, zscore
import statsmodels.formula.api as smf
import networkx as nx
import bct

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# Configuration
project_folder = Path(__file__).resolve().parents[1]
n_nodes = 100  # Main analysis
threshold = 0.20  # 20% network density

# %%
# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load all data and create PGS groups"""
    print("Loading and preparing data...")
    
    # Load connectivity matrices
    matrix_file = project_folder / f'data/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
    id_file = project_folder / 'data/subjectIDs_anonymised.txt'
    
    mats_df = pd.read_csv(matrix_file, header=None, sep='\s+')
    mats_df.columns = [f'conn_{i+1}' for i in range(mats_df.shape[1])]
    ids = pd.read_csv(id_file, header=None)[0].values
    mats_df.set_index(ids, inplace=True)
    
    # Load other datasets
    pgs_df = pd.read_csv(project_folder / "data/prs_residuals.csv")
    social_df = pd.read_csv(project_folder / 'data/cfa_factor_scores_full_sample.csv')
    behavioural_df = pd.read_csv(project_folder / "data/behavioural_data_anonymised.csv")
    phenotypic_df = pd.read_csv(project_folder / "data/phenotypic_data_anonymised.csv")
    movement_df = pd.read_csv(project_folder / 'data/movement_data_anonymised.csv')
    
    # Merge datasets
    merged_df = pd.merge(pgs_df, mats_df, left_on='Subject', right_index=True)
    merged_df = pd.merge(merged_df, social_df[['Subject', 'Social_Score']], on='Subject')
    merged_df = pd.merge(merged_df, behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']], on='Subject')
    merged_df = pd.merge(merged_df, phenotypic_df[['Subject', 'Age_in_Yrs']], on='Subject')
    merged_df = pd.merge(merged_df, movement_df[['Subject', 'Movement_RelativeRMS_mean']], on='Subject')
    
    # Quality control
    merged_df = merged_df[merged_df['Movement_RelativeRMS_mean'] < 0.2]
    merged_df = merged_df.dropna()
    print(f"Final sample: {len(merged_df)} subjects")
    
    # Create PGS groups
    merged_df['pgs_z'] = zscore(merged_df['blup_PRS_residuals'])
    merged_df['pgs_group'] = pd.cut(
        merged_df['pgs_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['exclude_low', 'low', 'middle', 'high', 'exclude_high']
    )
    
    # Keep main three groups
    merged_df = merged_df[merged_df['pgs_group'].isin(['low', 'middle', 'high'])]
    
    print("PGS group sizes:")
    print(merged_df['pgs_group'].value_counts().sort_index())
    
    return merged_df

# %%
# =============================================================================
# 2. NETWORK METRICS CALCULATION
# =============================================================================

def calculate_network_metrics(merged_df):
    """Calculate modularity and global efficiency for all subjects"""
    print(f"\nCalculating network metrics ({n_nodes} nodes, {threshold*100:.0f}% threshold)...")
    
    # Load modularity partition
    partition_df = pd.read_csv(project_folder / f'data/final_partition_{n_nodes}Nodes.csv')
    
    results = []
    for i, (subject_id, row) in enumerate(merged_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(merged_df)} subjects")
        
        # Extract connectivity matrix
        conn_data = row[[col for col in row.index if col.startswith('conn_')]].values
        mat = np.reshape(conn_data, (n_nodes, n_nodes)).astype(np.float64)
        mat = mat / 100  # Normalize
        mat = bct.threshold_proportional(mat, threshold)
        mat = np.nan_to_num(mat, nan=0.0)
        
        # Calculate modularity
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
            'pgs_z': row['pgs_z']
        })
    
    return pd.DataFrame(results)

# %%
# =============================================================================
# 3. MAIN ANALYSES
# =============================================================================

def test_brain_behavior_relationships(df):
    """Test brain-behavior relationships (Finding 1)"""
    print("\n" + "="*60)
    print("FINDING 1: BRAIN-BEHAVIOR RELATIONSHIPS")
    print("="*60)
    
    # Overall sample correlations
    r_mod, p_mod = pearsonr(df['modularity'], df['Social_Score'])
    r_eff, p_eff = pearsonr(df['global_efficiency'], df['Social_Score'])
    
    print(f"Whole sample (n={len(df)}):")
    print(f"  Modularity - Social Score:     r = {r_mod:.3f}, p = {p_mod:.3e}")
    print(f"  Global Efficiency - Social Score: r = {r_eff:.3f}, p = {p_eff:.3e}")
    
    # Within-group correlations
    print(f"\nWithin-group correlations:")
    group_results = {}
    for group in ['low', 'middle', 'high']:
        group_data = df[df['pgs_group'] == group]
        r_mod_grp, p_mod_grp = pearsonr(group_data['modularity'], group_data['Social_Score'])
        r_eff_grp, p_eff_grp = pearsonr(group_data['global_efficiency'], group_data['Social_Score'])
        
        print(f"  {group.capitalize()} PGS (n={len(group_data)}):")
        print(f"    Modularity: r = {r_mod_grp:.3f}, p = {p_mod_grp:.3f}")
        print(f"    Efficiency: r = {r_eff_grp:.3f}, p = {p_eff_grp:.3f}")
        
        group_results[group] = {
            'n': len(group_data),
            'mod_r': r_mod_grp, 'mod_p': p_mod_grp,
            'eff_r': r_eff_grp, 'eff_p': p_eff_grp
        }
    
    return {
        'overall': {'mod_r': r_mod, 'mod_p': p_mod, 'eff_r': r_eff, 'eff_p': p_eff},
        'groups': group_results
    }

    # Regression model
    model = smf.ols('Social_Score ~ modularity + global_efficiency + modularity*pgs_z + global_efficiency*pgs_z', data=results[0]).fit()
    print("\nRegression model summary:")
    print(model.summary())

def test_variability_hypothesis(df):
    """Test variability differences across PGS groups (Finding 2 & 3)"""
    print("\n" + "="*60)
    print("FINDING 2 & 3: VARIABILITY ANALYSIS")
    print("="*60)
    
    results = {}
    for metric in ['modularity', 'global_efficiency']:
        print(f"\n{metric.upper()} VARIABILITY:")
        
        group_stats = {}
        group_values = {}
        
        for group in ['low', 'middle', 'high']:
            data = df[df['pgs_group'] == group][metric]
            group_values[group] = data.values
            
            stats = {
                'mean': np.mean(data),
                'std': np.std(data),
                'var': np.var(data),
                'n': len(data)
            }
            group_stats[group] = stats
            
            print(f"  {group.capitalize()}: M = {stats['mean']:.3f}, SD = {stats['std']:.3f}")
        
        # Test variance differences (High vs Low PGS)
        high_data = group_values['high']
        low_data = group_values['low']
        
        # Levene's test
        levene_stat, levene_p = levene(high_data, low_data)
        
        # Variance ratio
        var_ratio = group_stats['high']['var'] / group_stats['low']['var']
        pct_increase = (var_ratio - 1) * 100
        
        print(f"  High vs Low PGS:")
        print(f"    Variance ratio = {var_ratio:.3f} ({pct_increase:+.1f}% difference)")
        print(f"    Levene's test: F = {levene_stat:.3f}, p = {levene_p:.3f}")
        
        # Bootstrap confidence interval for variance difference
        var_diffs = []
        for _ in range(1000):
            boot_high = np.random.choice(high_data, size=len(high_data), replace=True)
            boot_low = np.random.choice(low_data, size=len(low_data), replace=True)
            var_diffs.append(np.var(boot_high) - np.var(boot_low))
        
        ci_lower, ci_upper = np.percentile(var_diffs, [2.5, 97.5])
        p_bootstrap = np.mean(np.array(var_diffs) <= 0)
        
        print(f"    Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    Bootstrap p-value: {p_bootstrap:.3f}")
        
        # Landscape theory interpretation
        if p_bootstrap < 0.05 and var_ratio > 1:
            print(f"    *** SUPPORTS landscape theory: High PGS more variable")
        elif p_bootstrap >= 0.05:
            print(f"    *** No significant variability difference")
        
        results[metric] = {
            'group_stats': group_stats,
            'var_ratio': var_ratio,
            'levene_p': levene_p,
            'bootstrap_p': p_bootstrap,
            'ci': (ci_lower, ci_upper)
        }
    
    return results

def test_compensation_strategies(df):
    """Test for different compensation strategies in high PGS group"""
    print("\n" + "="*60)
    print("COMPENSATION STRATEGY ANALYSIS")
    print("="*60)
    
    high_pgs = df[df['pgs_group'] == 'high'].copy()
    
    # Split high PGS group by modularity (median split)
    med_mod = high_pgs['modularity'].median()
    
    high_mod_strategy = high_pgs[high_pgs['modularity'] > med_mod]
    low_mod_strategy = high_pgs[high_pgs['modularity'] <= med_mod]
    
    print(f"High PGS group (n={len(high_pgs)}) split by modularity:")
    print(f"\nHigh Modularity Strategy (n={len(high_mod_strategy)}):")
    print(f"  Modularity: {high_mod_strategy['modularity'].mean():.3f} ± {high_mod_strategy['modularity'].std():.3f}")
    print(f"  Efficiency: {high_mod_strategy['global_efficiency'].mean():.3f} ± {high_mod_strategy['global_efficiency'].std():.3f}")
    print(f"  Social Score: {high_mod_strategy['Social_Score'].mean():.3f} ± {high_mod_strategy['Social_Score'].std():.3f}")
    
    print(f"\nLow Modularity Strategy (n={len(low_mod_strategy)}):")
    print(f"  Modularity: {low_mod_strategy['modularity'].mean():.3f} ± {low_mod_strategy['modularity'].std():.3f}")
    print(f"  Efficiency: {low_mod_strategy['global_efficiency'].mean():.3f} ± {low_mod_strategy['global_efficiency'].std():.3f}")
    print(f"  Social Score: {low_mod_strategy['Social_Score'].mean():.3f} ± {low_mod_strategy['Social_Score'].std():.3f}")
    
    # Test if strategies achieve similar social outcomes
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(high_mod_strategy['Social_Score'], low_mod_strategy['Social_Score'])
    
    print(f"\nSocial outcome comparison (High vs Low Modularity strategies):")
    print(f"  t = {t_stat:.3f}, p = {p_val:.3f}")
    
    if p_val > 0.05:
        print("  *** Different strategies achieve similar outcomes - SUPPORTS compensation hypothesis")
    else:
        print("  *** Strategies differ in outcomes")
    
    return {
        'high_mod_strategy': high_mod_strategy,
        'low_mod_strategy': low_mod_strategy,
        'social_comparison': {'t': t_stat, 'p': p_val}
    }

# %%
# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def create_main_figure(df, brain_behavior_results, variability_results, compensation_results):
    """Create comprehensive figure showing all main findings"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#3498db', '#f39c12', '#e74c3c']  # Blue, Orange, Red
    
    # 1. Brain-behavior relationships
    ax = axes[0, 0]
    for i, group in enumerate(['low', 'middle', 'high']):
        group_data = df[df['pgs_group'] == group]
        ax.scatter(group_data['modularity'], group_data['Social_Score'], 
                  alpha=0.6, color=colors[i], label=f'{group.capitalize()} PGS', s=30)
    
    # Overall regression line
    z = np.polyfit(df['modularity'], df['Social_Score'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['modularity'].min(), df['modularity'].max(), 100)
    ax.plot(x_range, p(x_range), 'k-', alpha=0.8, linewidth=2)
    
    overall_r = brain_behavior_results['overall']['mod_r']
    overall_p = brain_behavior_results['overall']['mod_p']
    ax.text(0.05, 0.95, f'r = {overall_r:.3f}, p = {overall_p:.2e}', 
           transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Modularity')
    ax.set_ylabel('Social Score')
    ax.set_title('Finding 1: Modularity-Social Functioning\nAssociation')
    ax.legend()
    
    # 2. Modularity variability
    ax = axes[0, 1]
    group_names = ['Low', 'Middle', 'High']
    mod_stds = [variability_results['modularity']['group_stats'][g]['std'] for g in ['low', 'middle', 'high']]
    
    bars = ax.bar(group_names, mod_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Modularity Standard Deviation')
    ax.set_title('Finding 2: Modularity Variability\nby PGS Group')
    
    # Add significance annotation
    var_ratio = variability_results['modularity']['var_ratio']
    p_val = variability_results['modularity']['bootstrap_p']
    ax.text(0.5, 0.95, f'High vs Low:\n{var_ratio:.2f}× variance\np = {p_val:.3f}', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow' if p_val < 0.05 else 'white', alpha=0.8))
    
    # Add values on bars
    for bar, std in zip(bars, mod_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Efficiency variability (comparison)
    ax = axes[0, 2]
    eff_stds = [variability_results['global_efficiency']['group_stats'][g]['std'] for g in ['low', 'middle', 'high']]
    
    bars = ax.bar(group_names, eff_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Global Efficiency Standard Deviation')
    ax.set_title('Finding 3: Efficiency Variability\n(No Group Differences)')
    
    var_ratio = variability_results['global_efficiency']['var_ratio']
    p_val = variability_results['global_efficiency']['bootstrap_p']
    ax.text(0.5, 0.95, f'High vs Low:\n{var_ratio:.2f}× variance\np = {p_val:.3f}', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add values on bars
    for bar, std in zip(bars, eff_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Network organization space
    ax = axes[1, 0]
    for i, group in enumerate(['low', 'middle', 'high']):
        group_data = df[df['pgs_group'] == group]
        ax.scatter(group_data['global_efficiency'], group_data['modularity'], 
                  alpha=0.6, color=colors[i], label=f'{group.capitalize()} PGS', s=30)
    
    ax.set_xlabel('Global Efficiency')
    ax.set_ylabel('Modularity')
    ax.set_title('Network Organization Space')
    ax.legend()
    
    # 5. High PGS compensation strategies
    ax = axes[1, 1]
    if compensation_results is not None:
        high_mod = compensation_results['high_mod_strategy']
        low_mod = compensation_results['low_mod_strategy']
        
        ax.scatter(high_mod['global_efficiency'], high_mod['Social_Score'], 
                  color='red', alpha=0.7, label='High Modularity Strategy', s=40)
        ax.scatter(low_mod['global_efficiency'], low_mod['Social_Score'], 
                  color='blue', alpha=0.7, label='Low Modularity Strategy', s=40)
        
        ax.set_xlabel('Global Efficiency')
        ax.set_ylabel('Social Score')
        ax.set_title('High PGS: Different Strategies\nSimilar Outcomes')
        ax.legend()
        
        p_val = compensation_results['social_comparison']['p']
        ax.text(0.05, 0.95, f'Strategy comparison:\np = {p_val:.3f}', 
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen' if p_val > 0.05 else 'white', alpha=0.8))

    
    plt.tight_layout()
    plt.savefig(project_folder / 'figures/landscape_theory_graph_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.clf()
    
    return fig

# %%
# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete analysis pipeline"""
    print("="*80)
    print("GRAPH THEORY ANALYSIS - LANDSCAPE THEORY TEST")
    print("="*80)
    
    # Load data
    merged_df = load_and_prepare_data()
    
    # Calculate network metrics
    network_df = calculate_network_metrics(merged_df)
    
    # Run main analyses
    brain_behavior_results = test_brain_behavior_relationships(network_df)
    variability_results = test_variability_hypothesis(network_df)
    compensation_results = test_compensation_strategies(network_df)
    
    # Create visualization
    fig = create_main_figure(network_df, brain_behavior_results, variability_results, compensation_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    mod_var_ratio = variability_results['modularity']['var_ratio']
    mod_p = variability_results['modularity']['bootstrap_p']
    eff_var_ratio = variability_results['global_efficiency']['var_ratio']
    eff_p = variability_results['global_efficiency']['bootstrap_p']
    
    overall_mod_r = brain_behavior_results['overall']['mod_r']
    overall_mod_p = brain_behavior_results['overall']['mod_p']
    
    print(f"1. Modularity-Social Functioning: r = {overall_mod_r:.3f}, p = {overall_mod_p:.2e}")
    print(f"2. Modularity variability (High vs Low PGS): {mod_var_ratio:.2f}× variance, p = {mod_p:.3f}")
    print(f"3. Efficiency variability (High vs Low PGS): {eff_var_ratio:.2f}× variance, p = {eff_p:.3f}")
    
    if mod_p < 0.05 and eff_p >= 0.05:
        print("\n*** LANDSCAPE THEORY SUPPORTED WITH NETWORK-LEVEL SPECIFICITY ***")
        print("Compensation occurs through modular reorganization while preserving global integration")
    elif mod_p < 0.05 and eff_p < 0.05:
        print("\n*** GENERAL LANDSCAPE THEORY SUPPORTED ***")
        print("High genetic risk increases variability in both network measures")
    else:
        print("\n*** LANDSCAPE THEORY NOT SUPPORTED ***")
        print("No evidence for increased neural variability with genetic risk")
    
    # Save results
    network_df.to_csv(project_folder / 'results/graph_theory_landscape_results.csv', index=False)
    
    return network_df, brain_behavior_results, variability_results, compensation_results

# %%
if __name__ == "__main__":
    results = main()
# %%
