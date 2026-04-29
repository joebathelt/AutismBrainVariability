#!/usr/bin/env python3
"""
generate_publication_figures.py
===============================

Standalone publication-ready figures for the SDS-stratified landscape
analysis. Reads the C3 main-threshold metrics CSV (which contains
sds_group, residualised modularity / global_efficiency, Social_Score) and
the C6 SDS~PGS regression CSV.

Outputs (figures/publication/):
    fig_sds_distribution.svg
    fig_sds_group_boxplot.svg
    fig_modularity_sds_scatter.svg
    fig_efficiency_sds_scatter.svg
    fig_modularity_variability_bar.svg
    fig_efficiency_variability_bar.svg
    fig_network_organization_space.svg
    fig_bootstrap_density_modularity.svg
    fig_bootstrap_density_global_efficiency.svg
    fig_bootstrap_ellipse_extent.svg
    fig_sds_vs_pgs_scatter.svg            (exploratory, from C6 outputs)

Usage:
    python generate_publication_figures.py \
        --project /path/to/project \
        --results-file results/C4_main_network_metrics.csv
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import levene, pearsonr

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

mm2inches = 0.0393701

SDS_GROUPS = ['low_sds', 'middle', 'high_sds']
SDS_GROUP_LABELS = {'low_sds': 'Low SDS', 'middle': 'Middle', 'high_sds': 'High SDS'}
SDS_COLORS = {'low_sds': '#4DBBD5', 'middle': '#91D1C2', 'high_sds': '#8491B4'}


def setup_output_dir(project_dir):
    out = project_dir / 'figures' / 'publication'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _star(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_sds_distribution(df, output_dir, width_mm=55, height_mm=40):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    ax.hist(df['Social_Score'], bins=30, color='#3a3a3a', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    for cut, color in [(-1, '#4DBBD5'), (1, '#8491B4')]:
        ax.axvline(cut, color=color, linewidth=1, linestyle='--', alpha=0.8)
    ax.set_xlabel('SDS (Social_Score) [z]')
    ax.set_ylabel('Count')
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_sds_distribution.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_sds_distribution.svg")


def fig_sds_group_boxplot(df, output_dir, width_mm=55, height_mm=40):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    sub = df[df['sds_group'].isin(SDS_GROUPS)].copy()
    data = [sub.loc[sub['sds_group'] == g, 'Social_Score'].values for g in SDS_GROUPS]
    bp = ax.boxplot(data, labels=[SDS_GROUP_LABELS[g] for g in SDS_GROUPS],
                    patch_artist=True, widths=0.5)
    for patch, g in zip(bp['boxes'], SDS_GROUPS):
        patch.set_facecolor(SDS_COLORS[g])
        patch.set_alpha(0.8)
    ax.set_ylabel('SDS [z]')
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_sds_group_boxplot.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_sds_group_boxplot.svg")


def _metric_sds_scatter(df, metric, ylabel, fname, output_dir, width_mm=50, height_mm=45):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    sub = df[df['sds_group'].isin(SDS_GROUPS)]
    for group in SDS_GROUPS:
        gd = sub[sub['sds_group'] == group]
        ax.scatter(gd['Social_Score'], gd[metric], s=10, alpha=0.6,
                   color=SDS_COLORS[group], edgecolors='none',
                   label=SDS_GROUP_LABELS[group])
    z = np.polyfit(df['Social_Score'], df[metric], 1)
    xs = np.linspace(df['Social_Score'].min(), df['Social_Score'].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'k-', linewidth=1.2, alpha=0.85)
    r, p = pearsonr(df['Social_Score'], df[metric])
    ax.text(0.04, 0.96, f'r = {r:.3f}\np = {p:.2e}\nn = {len(df)}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel('SDS [z]')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / fname, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_modularity_sds_scatter(df, output_dir):
    _metric_sds_scatter(df, 'modularity', 'Modularity (residualised)',
                        'fig_modularity_sds_scatter.svg', output_dir)


def fig_efficiency_sds_scatter(df, output_dir):
    _metric_sds_scatter(df, 'global_efficiency',
                        'Global efficiency (residualised)',
                        'fig_efficiency_sds_scatter.svg', output_dir)


def _variability_bar(df, metric, ylabel, fname, output_dir,
                     width_mm=45, height_mm=40):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    stds, ns = [], []
    for g in SDS_GROUPS:
        data = df[df['sds_group'] == g][metric].values
        stds.append(np.std(data, ddof=1))
        ns.append(len(data))
    bars = ax.bar([SDS_GROUP_LABELS[g] for g in SDS_GROUPS], stds,
                  color=[SDS_COLORS[g] for g in SDS_GROUPS], alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    high = df[df['sds_group'] == 'high_sds'][metric].values
    low = df[df['sds_group'] == 'low_sds'][metric].values
    if len(high) >= 10 and len(low) >= 10:
        var_ratio = np.var(high) / np.var(low)
        levene_two = levene(high, low)[1]
        levene_one = levene_two / 2 if var_ratio > 1 else 1 - levene_two / 2
        face = 'lightyellow' if (var_ratio > 1 and levene_one < 0.05) else 'white'
        ax.text(0.5, 0.98,
                f'High/Low: {var_ratio:.2f}x var\nLevene p1 = {levene_one:.3f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor=face, alpha=0.85))
    for bar, sd in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{sd:.3f}', ha='center', va='bottom', fontsize=6)
    ax.set_ylabel(ylabel)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / fname, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_modularity_variability_bar(df, output_dir):
    _variability_bar(df, 'modularity', 'Modularity SD (resid.)',
                     'fig_modularity_variability_bar.svg', output_dir)


def fig_efficiency_variability_bar(df, output_dir):
    _variability_bar(df, 'global_efficiency', 'Efficiency SD (resid.)',
                     'fig_efficiency_variability_bar.svg', output_dir)


def fig_network_organization_space(df, output_dir, width_mm=50, height_mm=45):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    sub = df[df['sds_group'].isin(SDS_GROUPS)]
    for g in SDS_GROUPS:
        gd = sub[sub['sds_group'] == g]
        ax.scatter(gd['global_efficiency'], gd['modularity'], s=10, alpha=0.6,
                   color=SDS_COLORS[g], edgecolors='none',
                   label=SDS_GROUP_LABELS[g])
    ax.set_xlabel('Global efficiency (residualised)')
    ax.set_ylabel('Modularity (residualised)')
    ax.legend(fontsize=6, frameon=False)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_network_organization_space.svg', format='svg',
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_network_organization_space.svg")


def fig_bootstrap_density(df, measure, output_dir,
                           n_bootstrap=1000, sample_size=90,
                           width_mm=55, height_mm=42):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    np.random.seed(42)
    sub = df[df['sds_group'].isin(SDS_GROUPS)].copy()
    all_vals = sub[measure].values
    mean_all = np.mean(all_vals); std_all = np.std(all_vals, ddof=1)

    positions = [0, 1.2, 2.4]
    for pos, group in zip(positions, SDS_GROUPS):
        data = (sub[sub['sds_group'] == group][measure].values - mean_all) / std_all
        boots = [np.mean(np.random.choice(data, size=min(sample_size, len(data)),
                                          replace=True))
                 for _ in range(n_bootstrap)]
        boots = np.array(boots)
        kde = stats.gaussian_kde(boots)
        xs = np.linspace(boots.min(), boots.max(), 200)
        density = kde(xs) / kde(xs).max() * 0.4
        ax.fill_betweenx(xs, pos - density, pos, alpha=0.7,
                         color=SDS_COLORS[group])
        ax.boxplot([boots], positions=[pos + 0.2], widths=0.1, patch_artist=True,
                   boxprops=dict(facecolor=SDS_COLORS[group], alpha=0.7),
                   medianprops=dict(color='black', linewidth=1.5),
                   showfliers=False)
    ax.set_xlim(-0.5, 3.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([SDS_GROUP_LABELS[g] for g in SDS_GROUPS])
    label = 'Modularity' if measure == 'modularity' else 'Global efficiency'
    ax.set_ylabel(f'{label} (residualised) [z]')
    ax.set_xlabel('SDS Group')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    fname = f'fig_bootstrap_density_{measure}.svg'
    fig.savefig(output_dir / fname, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_bootstrap_ellipse_extent(df, output_dir,
                                  n_bootstrap=1000, sample_size=90,
                                  width_mm=120, height_mm=75):
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    np.random.seed(42)
    for g in SDS_GROUPS:
        gd = df[df['sds_group'] == g]
        if len(gd) == 0:
            continue
        mod = gd['modularity'].values
        eff = gd['global_efficiency'].values
        params = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(mod), size=min(sample_size, len(mod)),
                                   replace=True)
            cov = np.cov(np.column_stack([mod[idx], eff[idx]]).T)
            eigs, vecs = np.linalg.eigh(cov)
            order = eigs.argsort()[::-1]
            eigs = eigs[order]; vecs = vecs[:, order]
            params.append({
                'cx': np.mean(mod[idx]), 'cy': np.mean(eff[idx]),
                'w': 2 * 1.96 * np.sqrt(eigs[0]),
                'h': 2 * 1.96 * np.sqrt(eigs[1]),
                'angle': np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])),
            })
        cxm = np.median([p['cx'] for p in params])
        cym = np.median([p['cy'] for p in params])
        whi = np.percentile([p['w'] for p in params], 97.5)
        hhi = np.percentile([p['h'] for p in params], 97.5)
        am = np.median([p['angle'] for p in params])
        ax.add_patch(Ellipse(xy=(cxm, cym), width=whi, height=hhi, angle=am,
                             facecolor='none', edgecolor=SDS_COLORS[g],
                             linewidth=2, alpha=0.85,
                             label=SDS_GROUP_LABELS[g]))
        ax.scatter(cxm, cym, marker='o', s=80, alpha=0.6, color=SDS_COLORS[g],
                   edgecolors='black', linewidth=0.5, zorder=10)
    ax.set_xlabel('Modularity (residualised)')
    ax.set_ylabel('Global efficiency (residualised)')
    ax.legend(fontsize=7, frameon=False, loc='best')
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_bootstrap_ellipse_extent.svg', format='svg',
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_bootstrap_ellipse_extent.svg")


def fig_sds_vs_pgs_scatter(c6_results_csv, output_dir,
                            width_mm=80, height_mm=60):
    """Re-renders C6's SDS~PGS scatter as SVG. If the C6 CSV is missing
    (e.g. PGS not yet run on this branch), draws a placeholder text box."""
    fig, ax = plt.subplots(figsize=(width_mm * mm2inches, height_mm * mm2inches))
    if not c6_results_csv.exists():
        ax.text(0.5, 0.5, "C6 results not available\n(run exploratory_sds_pgs)",
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.axis('off')
    else:
        c6 = pd.read_csv(c6_results_csv)
        m1 = c6[c6['model'] == 'M1'].iloc[0]
        ax.errorbar([0], [m1['beta']],
                    yerr=[[m1['beta'] - m1['ci_lower']],
                          [m1['ci_upper'] - m1['beta']]],
                    fmt='o', color='#c84e3b', capsize=6, markersize=7)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.7)
        ax.set_xticks([0])
        ax.set_xticklabels(['blup_PGS_residuals_z'])
        ax.set_ylabel('beta on Social_Score')
        ax.set_title(f"M1: SDS ~ PGS  (n = {int(m1['n'])}, p = {m1['p']:.3g})")
        sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_sds_vs_pgs_scatter.svg', format='svg',
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_sds_vs_pgs_scatter.svg")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication SVGs (SDS-stratified)")
    parser.add_argument("--project", required=True)
    parser.add_argument("--results-file",
                        help="Path to results CSV; default: "
                             "results/C4_main_network_metrics.csv")
    parser.add_argument("--c6-results",
                        help="C6 SDS~PGS regression CSV; default: "
                             "results/C6_sds_pgs_regression.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project)
    output_dir = setup_output_dir(project_dir)
    print(f"Output directory: {output_dir}")

    if args.results_file:
        results_file = Path(args.results_file)
    else:
        results_file = project_dir / 'results' / 'C4_main_network_metrics.csv'

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    print(f"Loading: {results_file}")
    df = pd.read_csv(results_file)
    print(f"  {len(df)} subjects")

    if 'sds_group' not in df.columns:
        raise ValueError(
            f"{results_file} has no 'sds_group' column. Re-run C3 to produce "
            "the SDS-stratified main metrics CSV.")

    c6_csv = (Path(args.c6_results) if args.c6_results
              else project_dir / 'results' / 'C6_sds_pgs_regression.csv')

    print("\nGenerating figures:")
    fig_sds_distribution(df, output_dir)
    fig_sds_group_boxplot(df, output_dir)
    fig_modularity_sds_scatter(df, output_dir)
    fig_efficiency_sds_scatter(df, output_dir)
    fig_modularity_variability_bar(df, output_dir)
    fig_efficiency_variability_bar(df, output_dir)
    fig_network_organization_space(df, output_dir)
    fig_bootstrap_density(df, 'modularity', output_dir)
    fig_bootstrap_density(df, 'global_efficiency', output_dir)
    fig_bootstrap_ellipse_extent(df, output_dir)
    fig_sds_vs_pgs_scatter(c6_csv, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
