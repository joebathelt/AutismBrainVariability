#!/usr/bin/env python3
"""
D3_variability_figure.py
========================

Pooled modularity-variability figure (panels a-f) on the MALE subsample.

Recreates the presentation of the original ``manuscript/figures/Figure_2.pdf``
correctly restricted to males (``Gender == 'M'``). The main result of the paper
(autism polygenic load broadens modularity *variability*) is sex-dependent and
present in males; this figure presents it as a single, pooled PGS-group view
(Low / Middle / High) rather than the sex-stratified Men/Women split of
``D2_publication_figures.py``.

Panels (pooled by PGS group; NOT sex-stratified):
    (a) PGS distribution histogram with Low/Middle/High group shading
    (b) Social difficulty score (SDS) by PGS group
    (c) SDS vs PGS scatter with regression line + Pearson r
    (d) Bootstrap distribution of mean |modularity - mean| by PGS group
    (e) Bootstrap distribution of mean |global efficiency - mean| by PGS group
    (f) Joint Q-E bootstrap covariance ellipses per PGS group, with rug/carpet

Panel styling is ported from the retired ``generate_publication_figures.py``
(the script that produced the original Figure_2.pdf), refactored to draw onto
shared axes and composed into one figure via GridSpec so the whole figure is
reproducible in-code (no manual Illustrator compositing step).

Data source:
    results/C3_heteroscedasticity_results.csv  (one row per subject; carries
        Gender, pgs_z, Social_Score, and the covariate-residualised modularity
        & global_efficiency columns). `pgs_group` (low/middle/high/exclude_*)
        is derived from pgs_z if the column is absent.

Outputs:
    figures/D3_variability_figure_<sex>.pdf   (pipeline working copy)
    manuscript/figures/Figure_Variability.pdf (manuscript figure slot;
        overwrite only the missing/stale manuscript figure, never Figure_2.pdf)
    reports/D3_variability_figure_report.txt
"""

# %%
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, levene
import warnings
warnings.filterwarnings('ignore')


# %%
# =============================================================================
# STYLE (ported from generate_publication_figures.py)
# =============================================================================
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
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
})

mm2inches = 0.0393701

PGS_COLORS = {
    'low': '#4DBBD5',      # Cyan
    'middle': '#91D1C2',   # Teal
    'high': '#8491B4',     # Slate blue
    'other': '#E5E5E5',    # Light gray
}

GROUP_ORDER = ['low', 'middle', 'high']
GROUP_LABELS = ['Low', 'Middle', 'High']


def get_significance_symbol(p_value):
    """Convert p-value to significance symbol."""
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def derive_pgs_group(pgs_z):
    """Standardised PGS-group labels from the z-score, matching the cutoffs
    used elsewhere (B5, the variability-figure caption):

        low:          z < -1
        exclude_low:  -1 <= z < -0.5
        middle:       -0.5 <= z <= 0.5
        exclude_high: 0.5 < z <= 1
        high:         z > 1
    """
    z = pgs_z.to_numpy()
    conditions = [z < -1, z < -0.5, z <= 0.5, z <= 1]
    choices = ['low', 'exclude_low', 'middle', 'exclude_high']
    return pd.Series(np.select(conditions, choices, default='high'),
                     index=pgs_z.index)


def _despine(ax):
    """Hide top/right spines (composite-safe equivalent of sns.despine)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_label(ax, letter):
    """Bold panel label at the top-left of an axis."""
    ax.text(-0.22, 1.08, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


# %%
# =============================================================================
# PANEL DRAWERS (each draws onto a passed Axes; no fig creation / savefig)
# =============================================================================

def draw_pgs_distribution(ax, df):
    """(a) PGS histogram with Low/Middle/High shaded regions."""
    pgs_z = df['blup_PGS_residuals_z'].dropna()
    n_low = int((pgs_z < -1).sum())
    n_middle = int(((pgs_z >= -0.5) & (pgs_z <= 0.5)).sum())
    n_high = int((pgs_z > 1).sum())

    ax.axvspan(1, 4, color=PGS_COLORS['high'], alpha=0.3)
    ax.axvspan(-0.5, 0.5, color=PGS_COLORS['middle'], alpha=0.3)
    ax.axvspan(-4, -1, color=PGS_COLORS['low'], alpha=0.3)

    ax.hist(pgs_z, bins=40, range=(-3.5, 3.5), color='#666666',
            edgecolor='white', linewidth=0.3, zorder=5)

    ax.set_xlabel('PGS [z]')
    ax.set_ylabel('Count')
    ax.set_xlim([-3.5, 3.5])
    ax.set_xticks([-3, -1, 0, 1, 3])
    return {'n_low': n_low, 'n_middle': n_middle, 'n_high': n_high}


def draw_pgs_group_boxplot(ax, df):
    """(b) SDS by PGS group (swarm + box)."""
    plot_df = df[df['pgs_group'].isin(GROUP_ORDER)].copy()
    plot_df['pgs_group'] = pd.Categorical(plot_df['pgs_group'],
                                          categories=GROUP_ORDER, ordered=True)

    sns.swarmplot(data=plot_df, x='pgs_group', y='Social_Score',
                  order=GROUP_ORDER, hue='pgs_group', palette=PGS_COLORS,
                  size=2, alpha=0.6, ax=ax, zorder=1, legend=False)
    sns.boxplot(data=plot_df, x='pgs_group', y='Social_Score',
                order=GROUP_ORDER, showcaps=False, width=0.4,
                boxprops={'facecolor': 'none', 'edgecolor': 'black'},
                whiskerprops={'color': 'black'},
                medianprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, zorder=2)

    ax.set_xlabel('PGS Group')
    ax.set_ylabel('SDS [z]')
    ax.set_xticks(range(len(GROUP_LABELS)))
    ax.set_xticklabels(GROUP_LABELS)


def draw_pgs_social_scatter(ax, df):
    """(c) SDS vs PGS scatter with regression line; returns (r, p)."""
    for group in ['low', 'middle', 'high', 'other']:
        if group == 'other':
            mask = ~df['pgs_group'].isin(GROUP_ORDER)
        else:
            mask = df['pgs_group'] == group
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, 'blup_PGS_residuals_z'],
                   df.loc[mask, 'Social_Score'],
                   c=PGS_COLORS.get(group, PGS_COLORS['other']),
                   s=8, alpha=0.7, edgecolors='none')

    x = df['blup_PGS_residuals_z'].dropna()
    y = df.loc[x.index, 'Social_Score']
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, poly(x_range), 'k-', linewidth=1.5, zorder=10)

    r, pval = pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {r:.2f}\np = {pval:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='none'))

    ax.set_xlabel('PGS [z]')
    ax.set_ylabel('SDS [z]')
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-4, 4])
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    return float(r), float(pval)


def draw_bootstrap_density(ax, df, measure, rng, n_bootstrap=1000,
                           sample_size=None):
    """(d/e) Bootstrap distribution of the mean absolute deviation of `measure`
    from its group mean, per PGS group (half-violin + boxplot raincloud).

    Resamples the full group size by default (proper non-parametric bootstrap);
    set `sample_size` to cap it (the original used a fixed 90).
    """
    plot_df = df[df['pgs_group'].isin(GROUP_ORDER)].copy()
    positions = [0, 1.2, 2.4]
    colors = [PGS_COLORS[g] for g in GROUP_ORDER]

    for group, pos, color in zip(GROUP_ORDER, positions, colors):
        data = plot_df[plot_df['pgs_group'] == group][measure].values
        dev = np.abs(data - data.mean())
        size = len(dev) if sample_size is None else min(sample_size, len(dev))
        boot = np.array([rng.choice(dev, size=size, replace=True).mean()
                         for _ in range(n_bootstrap)])

        density = stats.gaussian_kde(boot)
        xs = np.linspace(boot.min(), boot.max(), 200)
        curve = density(xs)
        curve = curve / curve.max() * 0.4
        ax.fill_betweenx(xs, pos - curve, pos, alpha=0.6, color=color)

        ax.boxplot([boot], positions=[pos + 0.2], widths=0.1,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color='black', linewidth=1.5),
                   showfliers=False)

    label = 'Q' if measure == 'modularity' else 'E'
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylabel(f'Mean |{label} − μ|')
    ax.set_xlabel('PGS Group')
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)


def draw_bootstrap_ellipse_extent(ax, df, rng, n_bootstrap=1000,
                                  sample_size=None):
    """(f) Bootstrapped covariance ellipses of (modularity, global efficiency)
    per PGS group, with 95% CI crosshairs on the centroid and per-subject rug
    (carpet) plots on both axes."""
    plot_df = df[df['pgs_group'].isin(GROUP_ORDER)].copy()
    colors = [PGS_COLORS[g] for g in GROUP_ORDER]

    for i, group in enumerate(GROUP_ORDER):
        group_data = plot_df[plot_df['pgs_group'] == group]
        mod_data = group_data['modularity'].values
        eff_data = group_data['global_efficiency'].values
        if len(mod_data) < 10:
            continue

        centers_x, centers_y, widths, heights, angles = [], [], [], [], []
        for _ in range(n_bootstrap):
            size = (len(mod_data) if sample_size is None
                    else min(sample_size, len(mod_data)))
            idx = rng.choice(len(mod_data), size=size, replace=True)
            boot_mod, boot_eff = mod_data[idx], eff_data[idx]
            cov = np.cov(np.column_stack([boot_mod, boot_eff]).T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angles.append(np.degrees(np.arctan2(eigenvecs[1, 0],
                                                eigenvecs[0, 0])))
            widths.append(2 * 1.96 * np.sqrt(max(eigenvals[0], 0)))
            heights.append(2 * 1.96 * np.sqrt(max(eigenvals[1], 0)))
            centers_x.append(boot_mod.mean())
            centers_y.append(boot_eff.mean())

        cx, cy = np.median(centers_x), np.median(centers_y)
        cx_lo, cx_hi = np.percentile(centers_x, [2.5, 97.5])
        cy_lo, cy_hi = np.percentile(centers_y, [2.5, 97.5])

        ax.add_patch(Ellipse(
            xy=(cx, cy), width=np.median(widths), height=np.median(heights),
            angle=np.median(angles),
            facecolor=mcolors.to_rgba(colors[i], 0.18),
            edgecolor=mcolors.to_rgba(colors[i], 0.9),
            linewidth=2, linestyle='-'))
        ax.plot([cx_lo, cx_hi], [cy, cy], color='black', linewidth=1,
                alpha=0.9, zorder=10)
        ax.plot([cx, cx], [cy_lo, cy_hi], color='black', linewidth=1,
                alpha=0.9, zorder=10)
        ax.scatter(cx, cy, marker='o', s=80, alpha=0.7, color=colors[i],
                   edgecolors='black', linewidth=0.5, zorder=9,
                   label=group.capitalize())
        sns.rugplot(x=mod_data, ax=ax, color=colors[i], height=0.03,
                    alpha=0.6, linewidth=0.6)
        sns.rugplot(y=eff_data, ax=ax, color=colors[i], height=0.03,
                    alpha=0.6, linewidth=0.6)

    ax.set_xlabel('Q (res)')
    ax.set_ylabel('E (res)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)


def variability_summary(df):
    """Per-group SD/var + high/low variance ratio and Levene test, per metric.
    Used for the text/report (the panel-d/e result)."""
    plot_df = df[df['pgs_group'].isin(GROUP_ORDER)].copy()
    out = {}
    for metric in ['modularity', 'global_efficiency']:
        by_group = {g: plot_df[plot_df['pgs_group'] == g][metric]
                    for g in GROUP_ORDER}
        stds = {g: float(by_group[g].std()) for g in GROUP_ORDER}
        var_ratio = (by_group['high'].var() / by_group['low'].var()
                     if by_group['low'].var() > 0 else float('nan'))
        _, lev_p = levene(by_group['high'], by_group['low'])
        out[metric] = {'stds': stds, 'n': {g: int(len(by_group[g]))
                                           for g in GROUP_ORDER},
                       'var_ratio_high_low': float(var_ratio),
                       'levene_p_high_vs_low': float(lev_p)}
    return out


# %%
# =============================================================================
# MAIN
# =============================================================================

def build_figure(df, rng, n_bootstrap, sample_size):
    """Compose panels a-f into one GridSpec figure. Returns (fig, panel_stats)."""
    fig = plt.figure(figsize=(180 * mm2inches, 120 * mm2inches))
    gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.5)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[1:, 1:])

    dist_counts = draw_pgs_distribution(ax_a, df)
    draw_pgs_group_boxplot(ax_b, df)
    r_c, p_c = draw_pgs_social_scatter(ax_c, df)
    draw_bootstrap_density(ax_d, df, 'modularity', rng,
                           n_bootstrap=n_bootstrap, sample_size=sample_size)
    draw_bootstrap_density(ax_e, df, 'global_efficiency', rng,
                           n_bootstrap=n_bootstrap, sample_size=sample_size)
    draw_bootstrap_ellipse_extent(ax_f, df, rng, n_bootstrap=n_bootstrap,
                                  sample_size=sample_size)

    for ax, letter in [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'),
                       (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')]:
        _despine(ax)
        _panel_label(ax, letter)

    panel_stats = {'dist_counts': dist_counts,
                   'scatter_r': r_c, 'scatter_p': p_c}
    return fig, panel_stats


def main():
    parser = argparse.ArgumentParser(
        description='D3: pooled modularity-variability figure (male subsample).')
    parser.add_argument('--project', required=True, help='Project directory')
    parser.add_argument('--input', default=None,
                        help='Metrics CSV (default: '
                             'results/C3_heteroscedasticity_results.csv). '
                             'pgs_group is derived from pgs_z if absent.')
    parser.add_argument('--sex', default='M', choices=['M', 'F', 'all'],
                        help="Restrict to this sex (default: M).")
    parser.add_argument('--output', default=None,
                        help='Pipeline figure PDF path (default: '
                             'figures/D3_variability_figure_<sex>.pdf)')
    parser.add_argument('--manuscript-output', default=None,
                        help='Manuscript figure path (default: '
                             'manuscript/figures/Figure_Variability.pdf)')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Cap bootstrap resample size per group '
                             '(default: full group size).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project = Path(args.project)
    figures_dir = project / 'figures'
    reports_dir = project / 'reports'
    for d in (figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    input_file = (Path(args.input) if args.input
                  else project / 'results' / 'C3_heteroscedasticity_results.csv')
    df = pd.read_csv(input_file)
    n_total = len(df)

    # PGS z column used by the panels (mirror generate_publication_figures.py)
    if 'blup_PGS_residuals_z' not in df.columns:
        if 'pgs_z' in df.columns:
            df['blup_PGS_residuals_z'] = df['pgs_z']
        else:
            raise SystemExit('ERROR: no PGS z-score column (pgs_z / '
                             'blup_PGS_residuals_z) in input.')

    # PGS group labels (reuse if present; otherwise derive from pgs_z)
    if 'pgs_group' not in df.columns:
        df['pgs_group'] = derive_pgs_group(df['blup_PGS_residuals_z'])

    # Male (or requested) subsample
    if args.sex != 'all':
        if 'Gender' not in df.columns:
            raise SystemExit('ERROR: --sex requested but no Gender column.')
        df = df[df['Gender'] == args.sex].copy()
    n_sub = len(df)

    rng = np.random.default_rng(args.seed)
    fig, panel_stats = build_figure(df, rng, args.n_bootstrap, args.sample_size)

    sex_tag = args.sex.lower() if args.sex != 'all' else 'all'
    pipeline_pdf = (Path(args.output) if args.output
                    else figures_dir / f'D3_variability_figure_{sex_tag}.pdf')
    pipeline_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pipeline_pdf, bbox_inches='tight')

    manuscript_pdf = (Path(args.manuscript_output) if args.manuscript_output
                      else project / 'manuscript' / 'figures'
                      / 'Figure_Variability.pdf')
    manuscript_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pipeline_pdf, manuscript_pdf)
    plt.close(fig)

    # Report
    summary = variability_summary(df)
    lines = [
        '=' * 80,
        'D3: MODULARITY-VARIABILITY FIGURE (pooled by PGS group)',
        '=' * 80,
        '',
        f'Input           : {input_file}',
        f'Sex subsample   : {args.sex}',
        f'N total (input) : {n_total}',
        f'N in figure     : {n_sub}',
        f'Bootstrap iters : {args.n_bootstrap}',
        f'Bootstrap size  : {"full group" if args.sample_size is None else args.sample_size}',
        f'Random seed     : {args.seed}',
        '',
        f'Pipeline figure : {pipeline_pdf}',
        f'Manuscript figure: {manuscript_pdf}',
        '',
        'PGS group Ns (panel a shading; full subsample): '
        f'low={panel_stats["dist_counts"]["n_low"]}, '
        f'middle={panel_stats["dist_counts"]["n_middle"]}, '
        f'high={panel_stats["dist_counts"]["n_high"]}',
        '',
        '(c) SDS ~ PGS association (Pearson): '
        f'r = {panel_stats["scatter_r"]:+.3f}, p = {panel_stats["scatter_p"]:.3f}',
        '',
        '(d/e) Variability by PGS group (Low/Middle/High):',
    ]
    for metric, label in [('modularity', 'Q (modularity)'),
                          ('global_efficiency', 'E (global efficiency)')]:
        s = summary[metric]
        lines.append(f'  {label}:')
        lines.append(f'    N per group   : ' + ', '.join(
            f'{g}={s["n"][g]}' for g in GROUP_ORDER))
        lines.append(f'    SD per group  : ' + ', '.join(
            f'{g}={s["stds"][g]:.4f}' for g in GROUP_ORDER))
        lines.append(f'    Var ratio High/Low : {s["var_ratio_high_low"]:.2f}x')
        lines.append(f'    Levene High vs Low : p = {s["levene_p_high_vs_low"]:.3f} '
                     f'{get_significance_symbol(s["levene_p_high_vs_low"])}')
    lines += ['', '=' * 80, 'END OF REPORT', '=' * 80]
    report_text = '\n'.join(lines)

    report_path = reports_dir / 'D3_variability_figure_report.txt'
    report_path.write_text(report_text)
    print(report_text)


# %%
if __name__ == '__main__':
    main()
