#!/usr/bin/env python3
"""
generate_publication_figures.py
===============================

Generates standalone publication-ready figures as SVG files for the brain
compensation analysis. Each figure is saved as a separate SVG for easy
combination into multi-panel figures.

Usage:
    python generate_publication_figures.py --project /path/to/project

Output:
    figures/publication/
        - fig_pgs_social_scatter.svg
        - fig_pgs_distribution.svg
        - fig_pgs_group_boxplot.svg
        - fig_modularity_social_scatter.svg
        - fig_modularity_variability_bar.svg
        - fig_efficiency_variability_bar.svg
        - fig_network_organization_space.svg
        - fig_compensation_strategies.svg
        - fig_bootstrap_density_modularity.svg
        - fig_bootstrap_density_efficiency.svg
        - fig_bootstrap_ellipse_extent.svg
        - fig_connectivity_matrix_{group}.svg
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, levene, zscore
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Set publication-quality defaults
rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
})

# Unit conversion
mm2inches = 0.0393701

# Color palettes
PGS_COLORS = {
    'low': '#4DBBD5',      # Cyan
    'middle': '#91D1C2',   # Teal
    'high': '#8491B4',     # Slate blue
    'other': '#E5E5E5'     # Light gray
}

COMMUNITY_COLORS = {
    0: '#7B2D8E',  # Visual - deep purple
    1: '#C85450',  # DMN - warm red
    2: '#A8B8C8',  # Sensorimotor - blue-gray
    3: '#D17A47',  # FPN - warm orange
    4: '#2DB574',  # VAN - green
    5: '#4FB3D9'   # Other - cyan
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_significance_symbol(p_value):
    """Convert p-value to significance symbol."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def setup_output_dir(project_dir):
    """Create output directory for publication figures."""
    output_dir = project_dir / 'figures' / 'publication'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# FIGURE 1: PGS vs Social Score Association
# =============================================================================

def fig_pgs_social_scatter(df, output_dir, width_mm=55, height_mm=45):
    """
    Create scatter plot of PGS vs Social Score with regression line.

    Parameters:
    -----------
    df : DataFrame
        Must contain 'blup_PGS_residuals_z', 'Social_Score', 'pgs_group'
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    # Plot points by group
    for group in ['low', 'middle', 'high', 'other']:
        if group == 'other':
            mask = ~df['pgs_group'].isin(['low', 'middle', 'high'])
        else:
            mask = df['pgs_group'] == group

        if mask.sum() == 0:
            continue

        ax.scatter(
            df.loc[mask, 'blup_PGS_residuals_z'],
            df.loc[mask, 'Social_Score'],
            c=PGS_COLORS.get(group, PGS_COLORS['other']),
            s=8,
            alpha=0.7,
            edgecolors='none',
            label=group.capitalize() if group != 'other' else None
        )

    # Add regression line
    x = df['blup_PGS_residuals_z'].dropna()
    y = df.loc[x.index, 'Social_Score']

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, p(x_range), 'k-', linewidth=1.5, zorder=10)

    # Statistics
    r, pval = pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {r:.2f}\np = {pval:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('PGS [z]')
    ax.set_ylabel('SDS [z]')
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-4, 4])
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_pgs_social_scatter.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_pgs_social_scatter.svg")


# =============================================================================
# FIGURE 2: PGS Distribution with Group Shading
# =============================================================================

def fig_pgs_distribution(df, output_dir, width_mm=55, height_mm=40):
    """
    Create histogram of PGS distribution with group shading.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    pgs_z = df['blup_PGS_residuals_z'].dropna()

    # Count subjects in each group
    n_low = (pgs_z < -1).sum()
    n_middle = ((pgs_z >= -0.5) & (pgs_z <= 0.5)).sum()
    n_high = (pgs_z > 1).sum()

    # Add shaded regions for groups
    ax.axvspan(1, 4, color=PGS_COLORS['high'], alpha=0.3, label=f'High (n={n_high})')
    ax.axvspan(-0.5, 0.5, color=PGS_COLORS['middle'], alpha=0.3, label=f'Middle (n={n_middle})')
    ax.axvspan(-4, -1, color=PGS_COLORS['low'], alpha=0.3, label=f'Low (n={n_low})')

    # Histogram
    counts, bins, _ = ax.hist(pgs_z, bins=40, range=(-3.5, 3.5),
                               color='#666666', edgecolor='white', linewidth=0.3,
                               zorder=5)

    ax.set_xlabel('PGS [z]')
    ax.set_ylabel('Count')
    ax.set_xlim([-3.5, 3.5])
    ax.set_xticks([-3, -1, 0, 1, 3])

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_pgs_distribution.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_pgs_distribution.svg")


# =============================================================================
# FIGURE 3: PGS Group Comparison Boxplot
# =============================================================================

def fig_pgs_group_boxplot(df, output_dir, width_mm=55, height_mm=40):
    """
    Create boxplot with swarm overlay comparing Social Scores across PGS groups.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    # Filter to main groups only
    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()
    plot_df['pgs_group'] = pd.Categorical(plot_df['pgs_group'],
                                           categories=['low', 'middle', 'high'],
                                           ordered=True)

    # Swarm plot
    sns.swarmplot(x='pgs_group', y='Social_Score', data=plot_df,
                  palette=PGS_COLORS, size=2, alpha=0.6, ax=ax, zorder=1)

    # Box plot overlay
    sns.boxplot(x='pgs_group', y='Social_Score', data=plot_df,
                showcaps=False, width=0.4,
                boxprops={'facecolor': 'none', 'edgecolor': 'black'},
                whiskerprops={'color': 'black'},
                medianprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, zorder=2)

    ax.set_xlabel('PGS Group')
    ax.set_ylabel('SDS [z]')
    ax.set_xticklabels(['Low', 'Middle', 'High'])

    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_pgs_group_boxplot.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_pgs_group_boxplot.svg")


# =============================================================================
# FIGURE 4: Modularity vs Social Score
# =============================================================================

def fig_modularity_social_scatter(df, output_dir, width_mm=50, height_mm=45):
    """
    Create scatter plot of modularity vs social score with group colors.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()

    # Plot by group
    for group in ['low', 'middle', 'high']:
        mask = plot_df['pgs_group'] == group
        ax.scatter(
            plot_df.loc[mask, 'modularity'],
            plot_df.loc[mask, 'Social_Score'],
            c=PGS_COLORS[group],
            s=15,
            alpha=0.6,
            label=group.capitalize(),
            edgecolors='none'
        )

    # Overall regression line
    x = plot_df['modularity']
    y = plot_df['Social_Score']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, p(x_range), 'k-', linewidth=1.5, zorder=10)

    # Statistics
    r, pval = pearsonr(x, y)
    sig = get_significance_symbol(pval)
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.2e}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('Q (res)')
    ax.set_ylabel('SDS [z]')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=7)

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_modularity_social_scatter.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_modularity_social_scatter.svg")


# =============================================================================
# FIGURE 5: Modularity Variability by Group
# =============================================================================

def fig_modularity_variability_bar(df, output_dir, width_mm=45, height_mm=40):
    """
    Create bar chart showing modularity standard deviation by PGS group.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()

    # Calculate statistics
    stats_dict = {}
    for group in ['low', 'middle', 'high']:
        data = plot_df[plot_df['pgs_group'] == group]['modularity']
        stats_dict[group] = {'std': data.std(), 'var': data.var(), 'n': len(data)}

    # Bar plot
    groups = ['Low', 'Middle', 'High']
    stds = [stats_dict[g.lower()]['std'] for g in groups]
    colors = [PGS_COLORS[g.lower()] for g in groups]

    bars = ax.bar(groups, stds, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add values on bars
    for bar, std in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{std:.3f}', ha='center', va='bottom', fontsize=7)

    # Variance ratio annotation
    var_ratio = stats_dict['high']['var'] / stats_dict['low']['var']

    # Levene's test
    high_data = plot_df[plot_df['pgs_group'] == 'high']['modularity']
    low_data = plot_df[plot_df['pgs_group'] == 'low']['modularity']
    _, levene_p = levene(high_data, low_data)

    sig = get_significance_symbol(levene_p)
    ax.text(0.95, 0.95, f'Var ratio: {var_ratio:.2f}x\np = {levene_p:.3f} {sig}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if levene_p < 0.05 else 'white',
                     alpha=0.8, edgecolor='none'))

    ax.set_ylabel('Q SD (res)')
    ax.set_xlabel('PGS Group')

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_modularity_variability_bar.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_modularity_variability_bar.svg")


# =============================================================================
# FIGURE 6: Global Efficiency Variability by Group
# =============================================================================

def fig_efficiency_variability_bar(df, output_dir, width_mm=45, height_mm=40):
    """
    Create bar chart showing global efficiency standard deviation by PGS group.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()

    # Calculate statistics
    stats_dict = {}
    for group in ['low', 'middle', 'high']:
        data = plot_df[plot_df['pgs_group'] == group]['global_efficiency']
        stats_dict[group] = {'std': data.std(), 'var': data.var(), 'n': len(data)}

    # Bar plot
    groups = ['Low', 'Middle', 'High']
    stds = [stats_dict[g.lower()]['std'] for g in groups]
    colors = [PGS_COLORS[g.lower()] for g in groups]

    bars = ax.bar(groups, stds, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add values on bars
    for bar, std in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{std:.3f}', ha='center', va='bottom', fontsize=7)

    # Variance ratio annotation
    var_ratio = stats_dict['high']['var'] / stats_dict['low']['var']

    # Levene's test
    high_data = plot_df[plot_df['pgs_group'] == 'high']['global_efficiency']
    low_data = plot_df[plot_df['pgs_group'] == 'low']['global_efficiency']
    _, levene_p = levene(high_data, low_data)

    sig = get_significance_symbol(levene_p)
    ax.text(0.95, 0.95, f'Var ratio: {var_ratio:.2f}x\np = {levene_p:.3f} {sig}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='none'))

    ax.set_ylabel('E SD (res)')
    ax.set_xlabel('PGS Group')

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_efficiency_variability_bar.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_efficiency_variability_bar.svg")


# =============================================================================
# FIGURE 7: Network Organization Space
# =============================================================================

def fig_network_organization_space(df, output_dir, width_mm=50, height_mm=45):
    """
    Create scatter plot of efficiency vs modularity showing network organization space.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()

    # Plot by group
    for group in ['low', 'middle', 'high']:
        mask = plot_df['pgs_group'] == group
        ax.scatter(
            plot_df.loc[mask, 'global_efficiency'],
            plot_df.loc[mask, 'modularity'],
            c=PGS_COLORS[group],
            s=15,
            alpha=0.6,
            label=group.capitalize(),
            edgecolors='none'
        )

    ax.set_xlabel('E (res)')
    ax.set_ylabel('Q (res)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_network_organization_space.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_network_organization_space.svg")


# =============================================================================
# FIGURE 8: Compensation Strategies in High PGS
# =============================================================================

def fig_compensation_strategies(df, output_dir, width_mm=50, height_mm=45):
    """
    Create scatter plot showing different compensation strategies in high PGS group.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    high_pgs = df[df['pgs_group'] == 'high'].copy()

    if len(high_pgs) < 10:
        print("  Skipped: fig_compensation_strategies.svg (insufficient data)")
        plt.close(fig)
        return

    # Split by median modularity
    med_mod = high_pgs['modularity'].median()
    high_mod = high_pgs[high_pgs['modularity'] > med_mod]
    low_mod = high_pgs[high_pgs['modularity'] <= med_mod]

    # Plot strategies (use high PGS color with different shades)
    ax.scatter(high_mod['global_efficiency'], high_mod['Social_Score'],
               c=PGS_COLORS['high'], alpha=0.9, s=20, label='High Modularity', edgecolors='none')
    ax.scatter(low_mod['global_efficiency'], low_mod['Social_Score'],
               c=PGS_COLORS['low'], alpha=0.9, s=20, label='Low Modularity', edgecolors='none')

    # T-test for social score comparison
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(high_mod['Social_Score'], low_mod['Social_Score'])

    ax.text(0.05, 0.95, f'Strategy comparison:\np = {p_val:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round',
                     facecolor='lightgreen' if p_val > 0.05 else 'white',
                     alpha=0.8, edgecolor='none'))

    ax.set_xlabel('E (res)')
    ax.set_ylabel('SDS [z]')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=7,
              title='High PGS:', title_fontsize=7)

    sns.despine(offset=5)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_compensation_strategies.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_compensation_strategies.svg")


# =============================================================================
# FIGURE 9 & 10: Bootstrap Density Plots
# =============================================================================

def fig_bootstrap_density(df, measure, output_dir, width_mm=55, height_mm=42,
                          n_bootstrap=1000, sample_size=90):
    """
    Bootstrap density of mean absolute deviation from each PGS group's mean —
    visualises the spread that Levene's test compares across groups.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()

    positions = [0, 1.2, 2.4]
    colors = [PGS_COLORS['low'], PGS_COLORS['middle'], PGS_COLORS['high']]

    for i, (group, pos, color) in enumerate(zip(['low', 'middle', 'high'], positions, colors)):
        data = plot_df[plot_df['pgs_group'] == group][measure].values
        dev = np.abs(data - data.mean())

        bootstrap_mads = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(dev, size=min(sample_size, len(dev)), replace=True)
            bootstrap_mads.append(np.mean(sample))

        bootstrap_mads = np.array(bootstrap_mads)

        density = stats.gaussian_kde(bootstrap_mads)
        xs = np.linspace(bootstrap_mads.min(), bootstrap_mads.max(), 200)
        density_curve = density(xs)
        density_curve = density_curve / density_curve.max() * 0.4

        ax.fill_betweenx(xs, pos - density_curve, pos, alpha=0.6, color=color)

        ax.boxplot([bootstrap_mads], positions=[pos + 0.2], widths=0.1,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color='black', linewidth=1.5),
                   showfliers=False)

    if measure == 'modularity':
        label = 'Q'
    elif measure == 'global_efficiency':
        label = 'E'

    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Low', 'Middle', 'High'])
    ax.set_ylabel(f'Mean |{label} − μ|')
    ax.set_xlabel('PGS Group')
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    measure_name = measure.replace('_', '_')
    fig.savefig(output_dir / f'fig_bootstrap_density_{measure_name}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig_bootstrap_density_{measure_name}.svg")


# =============================================================================
# FIGURE 11: Bootstrap Ellipse Extent Plot
# =============================================================================

def fig_bootstrap_ellipse_extent(df, output_dir, width_mm=120, height_mm=75,
                                  n_bootstrap=1000, sample_size=90):
    """
    Create plot showing bootstrapped confidence ellipses for each PGS group.
    """
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

    plot_df = df[df['pgs_group'].isin(['low', 'middle', 'high'])].copy()
    colors = [PGS_COLORS['low'], PGS_COLORS['middle'], PGS_COLORS['high']]

    for i, group in enumerate(['low', 'middle', 'high']):
        group_data = plot_df[plot_df['pgs_group'] == group]
        mod_data = group_data['modularity'].values
        eff_data = group_data['global_efficiency'].values

        if len(mod_data) < 10:
            continue

        # Bootstrap ellipse parameters
        ellipse_params = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(mod_data), size=min(sample_size, len(mod_data)), replace=True)
            boot_mod = mod_data[idx]
            boot_eff = eff_data[idx]

            # Calculate covariance matrix
            data_2d = np.column_stack([boot_mod, boot_eff])
            cov = np.cov(data_2d.T)

            # Calculate ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]

            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * 1.96 * np.sqrt(eigenvals[0])
            height = 2 * 1.96 * np.sqrt(eigenvals[1])
            center = (np.mean(boot_mod), np.mean(boot_eff))

            ellipse_params.append({
                'center': center,
                'width': width,
                'height': height,
                'angle': angle
            })

        # Get median parameters
        widths = [p['width'] for p in ellipse_params]
        heights = [p['height'] for p in ellipse_params]
        centers_x = [p['center'][0] for p in ellipse_params]
        centers_y = [p['center'][1] for p in ellipse_params]
        angles = [p['angle'] for p in ellipse_params]

        width_med = np.median(widths)
        height_med = np.median(heights)
        center_x_med = np.median(centers_x)
        center_y_med = np.median(centers_y)
        angle_med = np.median(angles)

        cx_lo, cx_hi = np.percentile(centers_x, [2.5, 97.5])
        cy_lo, cy_hi = np.percentile(centers_y, [2.5, 97.5])

        # Draw confidence ellipse with subtle Venn-style shading
        ellipse = Ellipse(
            xy=(center_x_med, center_y_med),
            width=width_med,
            height=height_med,
            angle=angle_med,
            facecolor=mcolors.to_rgba(colors[i], 0.18),
            edgecolor=mcolors.to_rgba(colors[i], 0.9),
            linewidth=2,
            linestyle='-',
        )
        ax.add_patch(ellipse)

        # 95% CI crosshair on the ellipse centroid
        ax.plot([cx_lo, cx_hi], [center_y_med, center_y_med],
                color='black', linewidth=1, alpha=0.9, zorder=10)
        ax.plot([center_x_med, center_x_med], [cy_lo, cy_hi],
                color='black', linewidth=1, alpha=0.9, zorder=10)

        # Add centroid marker
        ax.scatter(center_x_med, center_y_med, marker='o', s=80, alpha=0.7,
                   color=colors[i], edgecolors='black', linewidth=0.5,
                   zorder=9, label=group.capitalize())

        # Rug plots: per-subject ticks colour-coded by group
        sns.rugplot(x=mod_data, ax=ax, color=colors[i], height=0.03,
                    alpha=0.6, linewidth=0.6)
        sns.rugplot(y=eff_data, ax=ax, color=colors[i], height=0.03,
                    alpha=0.6, linewidth=0.6)

    ax.set_xlabel('Q (res)')
    ax.set_ylabel('E (res)')
    ax.set_ylim([-0.075, 0.051])
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)
    ax.set_aspect('equal', adjustable='datalim')

    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    fig.savefig(output_dir / 'fig_bootstrap_ellipse_extent.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_bootstrap_ellipse_extent.svg")


# =============================================================================
# FIGURE 12: Connectivity Matrices
# =============================================================================

def fig_connectivity_matrices(project_dir, output_dir, width_mm=50, height_mm=50):
    """
    Create connectivity matrix heatmaps for each PGS group.
    """
    data_dir = project_dir / 'archive' / 'data'

    for group in ['low', 'middle', 'high']:
        matrix_file = data_dir / f'avg_connectivity_{group}_pgs_bootstrap.npy'

        if not matrix_file.exists():
            print(f"  Skipped: fig_connectivity_matrix_{group}.svg (file not found)")
            continue

        matrix = np.load(matrix_file)

        # Load partition for reordering (selected by C2b)
        partition_file = project_dir / 'results' / 'C2b_selected_partition.csv'
        if partition_file.exists():
            partition_df = pd.read_csv(partition_file)
            community_order = np.argsort(partition_df['community_id'].values)
            matrix = matrix[community_order][:, community_order]

            # Community boundaries
            community_sizes = partition_df['community_id'].value_counts().sort_index()
            boundaries = np.cumsum([0] + list(community_sizes))[:-1]
        else:
            boundaries = []

        fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))

        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)

        # Add community boundaries
        for boundary in boundaries[1:]:
            ax.axhline(boundary - 0.5, color='black', linewidth=0.5)
            ax.axvline(boundary - 0.5, color='black', linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        fig.savefig(output_dir / f'fig_connectivity_matrix_{group}.svg', format='svg',
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  Saved: fig_connectivity_matrix_{group}.svg")


# =============================================================================
# FIGURE 13: CONSORT-style Subject Retention Diagram (TikZ snippet)
# =============================================================================

def _parse_consort_counts(project_dir):
    """Parse subject-retention counts from pipeline report files.

    Reads A1, B1, B5 and C1 reports and returns the counts needed to
    populate the CONSORT diagram boxes. Raises a clear RuntimeError if a
    file is missing or a value cannot be located.
    """
    reports_dir = project_dir / 'reports'
    a1_path = reports_dir / 'A1_preprocess_phenotypic_data_report.txt'
    b1_path = reports_dir / 'B1_plinkQC_genotype_qc_report.txt'
    b5_path = reports_dir / 'B5_evaluate_blup_prediction_report.txt'
    c1_path = reports_dir / 'C1_run_univariate_fMRI_prediction_report.txt'

    for p in (a1_path, b1_path, b5_path, c1_path):
        if not p.exists():
            raise RuntimeError(f"CONSORT: report file not found: {p}")

    a1_text = a1_path.read_text()
    b1_text = b1_path.read_text()
    b5_text = b5_path.read_text()
    c1_text = c1_path.read_text()

    def grab_int(pattern, text, source):
        m = re.search(pattern, text)
        if m is None:
            raise RuntimeError(
                f"CONSORT: could not parse {pattern!r} from {source}")
        return int(m.group(1))

    def grab_float(pattern, text, source):
        m = re.search(pattern, text)
        if m is None:
            raise RuntimeError(
                f"CONSORT: could not parse {pattern!r} from {source}")
        return float(m.group(1))

    original = grab_int(
        r'Behavioural data shape: \((\d+),', a1_text, a1_path)

    plinkqc_failed = grab_int(
        r'Total unique individuals failing:\s*(\d+)', b1_text, b1_path)
    plinkqc_final = grab_int(
        r'FINAL CLEANED DATA:[^\n]*\n[^\n]*\n\s*Individuals:\s*(\d+)',
        b1_text, b1_path)
    after_genetic = plinkqc_final
    excl_genetic_total = original - after_genetic
    excl_failed_qc = plinkqc_failed
    excl_no_genetics = max(0, excl_genetic_total - excl_failed_qc)

    # Per-category plinkQC fails — read line counts directly from the
    # plinkQC_output/*.fail-*.IDs files. Sub-counts may overlap (a
    # subject can fail more than one criterion); the unique total is
    # ``excl_failed_qc`` parsed above.
    name_match = re.search(r'Input data:\s+(\S+)', b1_text)
    if name_match is None:
        raise RuntimeError(
            f"CONSORT: could not parse 'Input data:' from {b1_path}")
    plink_name = name_match.group(1)
    plinkqc_dir = project_dir / 'data' / 'plinkQC_output'
    if not plinkqc_dir.exists():
        raise RuntimeError(
            f"CONSORT: plinkQC output dir not found: {plinkqc_dir}")

    def count_fail(suffix):
        fpath = plinkqc_dir / f"{plink_name}.fail-{suffix}.IDs"
        if not fpath.exists():
            return 0
        return sum(1 for line in fpath.open() if line.strip())

    fail_sex = count_fail('sexcheck')
    fail_het = count_fail('het')
    fail_relatedness = count_fail('IBD')
    fail_imiss = count_fail('imiss')
    fail_ancestry = count_fail('ancestry')
    ancestry_skipped = bool(
        re.search(r'ANCESTRY CHECK:\s*\n\s*SKIPPED', b1_text))

    after_behaviour = grab_int(
        r'Full sample merged data shape: \((\d+),', b5_text, b5_path)
    excl_behaviour = after_genetic - after_behaviour

    after_fmri = grab_int(
        r'Final sample size: (\d+) subjects', c1_text, c1_path)
    excl_fmri_total = after_behaviour - after_fmri
    motion_threshold = grab_float(
        r'Motion threshold:\s*([\d.]+)', c1_text, c1_path)

    pgs_low = grab_int(
        r'Low PGS \(<-1 SD\):\s*(\d+)', c1_text, c1_path)
    pgs_middle = grab_int(
        r'Middle PGS \(>-0\.5 SD & <0\.5 SD\):\s*(\d+)', c1_text, c1_path)
    pgs_high = grab_int(
        r'High PGS \(>\+1 SD\):\s*(\d+)', c1_text, c1_path)
    pgs_other = after_fmri - pgs_low - pgs_middle - pgs_high

    return {
        'original': original,
        'excl_no_genetics': excl_no_genetics,
        'excl_failed_qc': excl_failed_qc,
        'excl_genetic_total': excl_genetic_total,
        'fail_sex': fail_sex,
        'fail_het': fail_het,
        'fail_relatedness': fail_relatedness,
        'fail_imiss': fail_imiss,
        'fail_ancestry': fail_ancestry,
        'ancestry_skipped': ancestry_skipped,
        'after_genetic': after_genetic,
        'excl_behaviour': excl_behaviour,
        'after_behaviour': after_behaviour,
        'excl_fmri_total': excl_fmri_total,
        'motion_threshold': motion_threshold,
        'after_fmri': after_fmri,
        'pgs_low': pgs_low,
        'pgs_middle': pgs_middle,
        'pgs_high': pgs_high,
        'pgs_other': pgs_other,
    }


def fig_consort_diagram(project_dir, output_dir, include_behaviour_step=True):
    """Generate a TikZ snippet for the CONSORT subject-retention diagram.

    Writes ``fig_consort_diagram.tex`` containing a single ``tikzpicture``
    environment. The snippet expects the parent document to define a ``\\h``
    indentation macro (see the manuscript preamble).

    Each exclusion box sits in the same matrix row as the step it is
    being excluded *from* (matching the user's hand-drawn template).
    """
    counts = _parse_consort_counts(project_dir)

    def fmt(x):
        return f"{x:,}"

    rows = []

    qc_breakdown_lines = [
        f"        \\h No genetics: n={fmt(counts['excl_no_genetics'])} \\\\",
        f"        \\h Failed sex check: n={fmt(counts['fail_sex'])} \\\\",
        f"        \\h Failed het/missingness: n={fmt(counts['fail_het'])} \\\\",
        f"        \\h Failed relatedness: n={fmt(counts['fail_relatedness'])} \\\\",
    ]
    if counts['ancestry_skipped']:
        qc_breakdown_lines.append(
            "        \\h Ancestry check: not run")
    else:
        qc_breakdown_lines.append(
            f"        \\h Failed ancestry: n={fmt(counts['fail_ancestry'])}")

    rows.append(
        f"      \\node [block_center] (start) {{Original: n={fmt(counts['original'])}}}; &\n"
        f"      \\node [block_left] (excluded1) {{Missing data: n={fmt(counts['excl_genetic_total'])} \\\\\n"
        + "\n".join(qc_breakdown_lines) + "\n"
        "      }; \\\\"
    )

    if include_behaviour_step and counts['excl_behaviour'] > 0:
        rows.append(
            f"      \\node [block_center] (step1) {{Genetic: n={fmt(counts['after_genetic'])}}}; &\n"
            f"      \\node [block_left] (excluded2) {{Missing behaviour: n={fmt(counts['excl_behaviour'])}}}; \\\\"
        )
        before_fmri_row = (
            f"      \\node [block_center] (step2) "
            f"{{Behaviour: n={fmt(counts['after_behaviour'])}}}; &\n"
            f"      \\node [block_left] (excluded3) {{fMRI exclusion: n={fmt(counts['excl_fmri_total'])} \\\\\n"
            f"        \\h Motion threshold: {counts['motion_threshold']:g} (rel. RMS)\n"
            "      }; \\\\"
        )
        extra_paths = [
            "      \\path (step1) -- (excluded2);",
            "      \\path (step1) -- (step2);",
            "      \\path (step2) -- (excluded3);",
            "      \\path (step2) -- (step3);",
        ]
    elif include_behaviour_step:
        rows.append(
            f"      \\node [block_center] (step1) {{Genetic: n={fmt(counts['after_genetic'])}}}; \\\\"
        )
        before_fmri_row = (
            f"      \\node [block_center] (step2) "
            f"{{Behaviour: n={fmt(counts['after_behaviour'])}\\\\"
            "(missing values imputed via MICE)}; &\n"
            f"      \\node [block_left] (excluded3) {{fMRI exclusion: n={fmt(counts['excl_fmri_total'])} \\\\\n"
            f"        \\h Motion threshold: {counts['motion_threshold']:g} (rel. RMS)\n"
            "      }; \\\\"
        )
        extra_paths = [
            "      \\path (step1) -- (step2);",
            "      \\path (step2) -- (excluded3);",
            "      \\path (step2) -- (step3);",
        ]
    else:
        before_fmri_row = (
            f"      \\node [block_center] (step1) {{Genetic: n={fmt(counts['after_genetic'])}}}; &\n"
            f"      \\node [block_left] (excluded3) {{fMRI exclusion: n={fmt(counts['excl_fmri_total'])} \\\\\n"
            f"        \\h Motion threshold: {counts['motion_threshold']:g} (rel. RMS)\n"
            "      }; \\\\"
        )
        extra_paths = [
            "      \\path (step1) -- (excluded3);",
            "      \\path (step1) -- (step3);",
        ]

    rows.append(before_fmri_row)
    rows.append(
        f"      \\node [block_center] (step3) {{rs-fMRI: n={fmt(counts['after_fmri'])}}}; \\\\"
    )
    rows.append(
        "      \\node [block_large] (step4) {PGS groups:\\\\\n"
        f"      \\h Low PGS (\\textless -1 SD): n={fmt(counts['pgs_low'])}\\\\\n"
        f"      \\h Middle PGS (±0.5 SD): n={fmt(counts['pgs_middle'])}\\\\\n"
        f"      \\h High PGS (\\textgreater +1 SD): n={fmt(counts['pgs_high'])}\\\\\n"
        f"      \\h Other: n={fmt(counts['pgs_other'])}}}; \\\\"
    )

    paths = [
        "      \\path (start) -- (excluded1);",
        "      \\path (start) -- (step1);",
        *extra_paths,
        "      \\path (step3) -- (step4);",
    ]

    snippet = (
        "% CONSORT subject-retention diagram (auto-generated by\n"
        "% generate_publication_figures.py:fig_consort_diagram).\n"
        "% Parent document must define \\h (indentation macro), e.g.:\n"
        "%   \\newcommand*{\\h}{\\hspace{5pt}}\n"
        "\\begin{tikzpicture}[\n"
        "    auto,\n"
        "    block_center/.style ={rectangle, draw=black, thick, fill=white,\n"
        "      text width=12em, align=center, minimum height=4em},\n"
        "    block_left/.style ={rectangle, draw=black, thick, fill=white,\n"
        "      text width=16em, align=left, minimum height=4em, inner sep=6pt},\n"
        "    block_large/.style ={rectangle, draw=black, thick, fill=white,\n"
        "      text width=16em, align=left, minimum height=4em, inner sep=6pt},\n"
        "    line/.style ={draw, -Latex, thick, shorten >=0pt},\n"
        "  ]\n"
        "    \\matrix [column sep=5mm,row sep=3mm] {\n"
        + "\n".join(rows) + "\n"
        "    };\n"
        "    \\begin{scope}[every path/.style=line]\n"
        + "\n".join(paths) + "\n"
        "    \\end{scope}\n"
        "  \\end{tikzpicture}\n"
    )

    output_path = output_dir / 'fig_consort_diagram.tex'
    output_path.write_text(snippet)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready SVG figures"
    )
    parser.add_argument("--project", required=True, help="Project directory path")
    parser.add_argument("--results-file", help="Path to results CSV (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project)

    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    # Setup output directory
    output_dir = setup_output_dir(project_dir)
    print(f"\nOutput directory: {output_dir}")

    # Find results file
    results_file = None
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        # Try common locations. The stable downstream name is
        # C4_main_network_metrics.csv (written by C3 for the C2b-selected
        # parcellation at the main threshold).
        candidates = [
            project_dir / 'results' / 'C4_main_network_metrics.csv',
            project_dir / 'results' / 'C3_graph_theory_landscape_results.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                results_file = candidate
                break

    if results_file is None or not results_file.exists():
        print("ERROR: Could not find results file. Please specify with --results-file")
        return

    print(f"\nLoading data from: {results_file}")
    df = pd.read_csv(results_file)
    print(f"  Loaded {len(df)} subjects")

    # Check for required columns and rename if necessary
    if 'blup_PGS_residuals_z' not in df.columns:
        if 'pgs_z' in df.columns:
            df['blup_PGS_residuals_z'] = df['pgs_z']
        else:
            print("WARNING: No PGS z-score column found")

    # Generate figures
    print("\nGenerating individual figures:")

    # B4-style figures
    print("\n--- PGS-Behavior Association Figures ---")
    fig_pgs_social_scatter(df, output_dir)
    fig_pgs_distribution(df, output_dir)
    fig_pgs_group_boxplot(df, output_dir)

    # C4-style figures
    print("\n--- Landscape Theory Figures ---")
    fig_modularity_social_scatter(df, output_dir)
    fig_modularity_variability_bar(df, output_dir)
    fig_efficiency_variability_bar(df, output_dir)
    fig_network_organization_space(df, output_dir)
    fig_compensation_strategies(df, output_dir)

    # C6-style figures
    print("\n--- Network Visualization Figures ---")
    fig_bootstrap_density(df, 'modularity', output_dir)
    fig_bootstrap_density(df, 'global_efficiency', output_dir)
    fig_bootstrap_ellipse_extent(df, output_dir)
    fig_connectivity_matrices(project_dir, output_dir)

    # CONSORT subject-retention diagram (LaTeX/TikZ snippet)
    print("\n--- CONSORT Diagram ---")
    fig_consort_diagram(project_dir, output_dir)

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated SVG files can be combined using tools like:")
    print("  - Inkscape (free, cross-platform)")
    print("  - svgutils (Python library)")
    print("  - Adobe Illustrator")


if __name__ == "__main__":
    main()
