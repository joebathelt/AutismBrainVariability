# %%
"""
D2_publication_figures.py
=========================

Publication-quality figure showing the per-quintile coefficient of
variation (CV = SD / mean) of brain network metrics (modularity, global
efficiency) within men and within women, plotted against PGS quintile.

CV is computed on the raw (pre-residualisation) metric columns
(`modularity_raw`, `global_efficiency_raw`) because the residualised
columns have mean ~0, which makes CV undefined.

Data source:
    results/C3_heteroscedasticity_results.csv  (one row per subject,
        with raw and covariate-residualised modularity & global_efficiency,
        pgs_z, and Gender)

Outputs:
    figures/D2_quintile_cv_by_sex.png
    figures/D2_atypicality_pgs.png
    results/D2_quintile_cv_by_sex.csv
    results/D2_atypicality_scores.csv
    results/D2_atypicality_pgs_correlations.csv
    reports/D2_publication_figures_report.txt
"""

# %%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr


# %%
plt.style.use('default')
# Helvetica preferred; Nimbus Sans / Liberation Sans / Arial are
# metrically-compatible fallbacks if Helvetica isn't installed.
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Nimbus Sans',
                               'Liberation Sans', 'Arial']
rcParams['axes.labelsize'] = 7
rcParams['axes.titlesize'] = 8
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 7

mm2inches = 0.0393701
FIGURE_DPI = 300

METRICS = [
    ('modularity', 'Modularity'),
    ('global_efficiency', 'Global efficiency'),
]
SEX_COLORS = {'F': '#E64B35', 'M': '#4DBBD5'}
SEX_LABELS = {'F': 'Women', 'M': 'Men'}
SEX_X_OFFSET = {'F': -0.08, 'M': +0.08}
N_QUINTILES = 5


# %%
def compute_quintile_cv_by_sex(df, metric, n_bootstrap=1000,
                               n_quintiles=N_QUINTILES, seed=42):
    """Per-quintile CV (= SD/mean) within each sex, with bootstrap 95% CIs.

    Computed on the raw (pre-residualisation) metric column
    `<metric>_raw`. Returns a list of dicts, one per (sex, quintile).
    """
    raw_col = f'{metric}_raw'
    rng = np.random.default_rng(seed)
    rows = []
    for sex in sorted(df['Gender'].dropna().unique()):
        sub = df[df['Gender'] == sex].copy()
        sub['pgs_quintile'] = pd.qcut(sub['pgs_z'], q=n_quintiles,
                                      labels=False) + 1
        for q in range(1, n_quintiles + 1):
            mask = sub['pgs_quintile'] == q
            values = sub.loc[mask, raw_col].to_numpy()
            n = len(values)
            sd = np.std(values, ddof=1)
            mean = np.mean(values)
            cv = sd / mean
            boot_cvs = np.array([
                (lambda s: np.std(s, ddof=1) / np.mean(s))(
                    rng.choice(values, size=n, replace=True))
                for _ in range(n_bootstrap)
            ])
            ci_lo, ci_hi = np.percentile(boot_cvs, [2.5, 97.5])
            rows.append({
                'metric': metric,
                'sex': sex,
                'quintile': q,
                'n': n,
                'pgs_center': sub.loc[mask, 'pgs_z'].mean(),
                'mean': mean,
                'sd': sd,
                'cv': cv,
                'cv_ci_lower': ci_lo,
                'cv_ci_upper': ci_hi,
            })
    return rows


def plot_quintile_cv_by_sex(quintile_df, output_path, interaction_stats=None):
    """Two-panel figure: modularity CV | global efficiency CV, by sex.

    interaction_stats: optional dict
        {metric: {'beta': float, 'pvalue': float}} for the sex x PGS
        interaction on residual variance. Annotated on each panel.
    """
    fig, axes = plt.subplots(1, 2,
                             figsize=(80 * mm2inches, 60 * mm2inches),
                             dpi=FIGURE_DPI,
                             constrained_layout=True)

    legend_handles = None
    legend_labels = None
    for ax, (metric, label) in zip(axes, METRICS):
        sub = quintile_df[quintile_df['metric'] == metric]
        for sex_key, sex_df in sub.groupby('sex'):
            sex = str(sex_key)
            sex_df = sex_df.sort_values('quintile')
            x = sex_df['quintile'].to_numpy()
            y = sex_df['cv'].to_numpy()
            yerr_lo = y - sex_df['cv_ci_lower'].to_numpy()
            yerr_hi = sex_df['cv_ci_upper'].to_numpy() - y
            color = SEX_COLORS.get(sex, 'gray')
            x_plot = x + SEX_X_OFFSET.get(sex, 0.0)

            _, caplines, barlinecols = ax.errorbar(
                x_plot, y,
                yerr=[yerr_lo, yerr_hi],
                fmt='o',
                color=color,
                label=SEX_LABELS.get(sex, sex),
                capsize=2, elinewidth=1, markersize=4,
            )
            for artist in (*caplines, *barlinecols):
                artist.set_alpha(0.35)

            slope, intercept = np.polyfit(x, y, 1)
            xfit = np.array([x.min(), x.max()])
            ax.plot(xfit + SEX_X_OFFSET.get(sex, 0.0),
                    slope * xfit + intercept,
                    color=color, linewidth=0.9, linestyle='-',
                    zorder=1)
        ax.set_xlabel('PGS quintile')
        ax.set_xticks(range(1, N_QUINTILES + 1))
        ax.set_ylabel(f'{label} CV')
        ax.set_title(label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if interaction_stats is not None and metric in interaction_stats:
            stats = interaction_stats[metric]
            ax.text(
                0.97, 0.03,
                f'$p_\\mathrm{{int}}$ = {stats["pvalue"]:.3f}',
                transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6,
            )

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='outside lower center',
                   ncol=len(legend_handles), frameon=False,
                   handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.0)

    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _covariance_ellipse(x, y, n_std=1.0):
    """Return (xy, width, height, angle_deg) of the n_std covariance
    ellipse of the 2D point cloud (x, y), or None if degenerate."""
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    return (float(np.mean(x)), float(np.mean(y))), float(width), \
        float(height), angle


def plot_modularity_efficiency_kde_by_pgs(df, output_path,
                                          n_std=2.0,
                                          n_bootstrap=200,
                                          seed=0):
    """Two-panel figure (Men | Women): modularity vs global efficiency,
    with one covariance ellipse per PGS tertile (low / middle / high)
    drawn over a faint background scatter — a Venn-style overlap that
    makes any shift in centre or spread between tertiles legible at a
    glance.

    Each tertile's outline is wrapped by a bootstrap halo: ``n_bootstrap``
    ellipses fitted on resampled subjects, each rendered with low alpha
    so they stack into a fuzzy band representing the sampling
    uncertainty of the centre/shape estimate.

    Plotted on the *covariate-residualised* metrics (`modularity`,
    `global_efficiency`) — the same scale on which the C3
    heteroscedasticity tests are computed — so any visible spread
    differences correspond to the variance signal those tests target.

    Tertile edges are computed on the pooled sample so the three
    categories are directly comparable across panels.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2,
                             figsize=(150 * mm2inches, 75 * mm2inches),
                             dpi=FIGURE_DPI,
                             constrained_layout=True)

    panels = [('M', 'Men'), ('F', 'Women')]

    x_all = df['modularity'].to_numpy()
    y_all = df['global_efficiency'].to_numpy()
    x_lo, x_hi = np.percentile(x_all, [1, 99])
    y_lo, y_hi = np.percentile(y_all, [1, 99])
    x_pad = (x_hi - x_lo) * 0.08
    y_pad = (y_hi - y_lo) * 0.08
    x_lim = (x_lo - x_pad, x_hi + x_pad)
    y_lim = (y_lo - y_pad, y_hi + y_pad)

    tertile_edges = np.quantile(df['pgs_z'].to_numpy(),
                                np.linspace(0, 1, 4))
    df = df.assign(
        _pgs_t=pd.cut(df['pgs_z'], bins=tertile_edges,
                      labels=False, include_lowest=True)
    )

    cmap = plt.get_cmap('RdBu_r')
    tertile_colors = [mcolors.to_hex(cmap(t))
                      for t in (0.1, 0.5, 0.9)]
    tertile_labels = ['Low PGS (T1)', 'Middle PGS (T2)',
                      'High PGS (T3)']

    for ax, (sex, label) in zip(axes, panels):
        sub = df[df['Gender'] == sex]
        if sub.empty:
            ax.set_title(label)
            continue

        ax.scatter(sub['modularity'], sub['global_efficiency'],
                   s=4, color='0.7', alpha=0.18, linewidths=0,
                   zorder=1)

        for t, color in enumerate(tertile_colors):
            tsub = sub[sub['_pgs_t'] == t]
            x = tsub['modularity'].to_numpy()
            y = tsub['global_efficiency'].to_numpy()
            params = _covariance_ellipse(x, y, n_std=n_std)
            if params is None:
                continue
            xy, width, height, angle = params

            n = x.size
            if n_bootstrap > 0 and n >= 3:
                boot_alpha = float(np.clip(8.0 / n_bootstrap, 0.01, 0.12))
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, n, size=n)
                    bp = _covariance_ellipse(x[idx], y[idx], n_std=n_std)
                    if bp is None:
                        continue
                    bxy, bw, bh, bang = bp
                    ax.add_patch(Ellipse(xy=bxy, width=bw, height=bh,
                                         angle=bang,
                                         facecolor='none',
                                         edgecolor=color,
                                         linewidth=0.5,
                                         alpha=boot_alpha, zorder=2))

            ax.add_patch(Ellipse(xy=xy, width=width, height=height,
                                 angle=angle,
                                 facecolor='none', edgecolor=color,
                                 linewidth=1.6, alpha=0.95, zorder=3))
            ax.plot(xy[0], xy[1], marker='o',
                    markersize=4, markerfacecolor=color,
                    markeredgecolor='white', markeredgewidth=0.6,
                    linestyle='None', zorder=4)

        ax.set_xlabel('Modularity (residualised)')
        ax.set_ylabel('Global efficiency (residualised)')
        ax.set_title(label)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    legend_handles = [
        Patch(facecolor=c, edgecolor=c, alpha=0.6,
              linewidth=1.2, label=tlabel)
        for c, tlabel in zip(tertile_colors, tertile_labels)
    ]
    fig.legend(handles=legend_handles,
               loc='center right', frameon=False,
               fontsize=7,
               title=f'PGS tertile ({n_std:g} SD ellipse)',
               title_fontsize=7,
               bbox_to_anchor=(1.0, 0.5))

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_social_metric_association(df, output_path, n_bootstrap=1000, seed=0):
    """Two-panel figure: (covariate-residualised) graph metric ~ social
    difficulty score, with separate regression lines per sex.

    Uses the residualised metric columns (`modularity`, `global_efficiency`)
    from the C3 output, which have already been adjusted for Age, ICV and
    head motion. Sex is shown as a moderator (separate lines), not
    residualised away.

    A bootstrap 95% CI band is drawn around each per-sex regression line,
    and per-sex Pearson r and p are annotated on the panel.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2,
                             figsize=(110 * mm2inches, 65 * mm2inches),
                             dpi=FIGURE_DPI,
                             constrained_layout=True)

    legend_handles = None
    legend_labels = None
    for ax, (metric, label) in zip(axes, METRICS):
        x_grid = np.linspace(df['Social_Score'].min(),
                             df['Social_Score'].max(), 100)

        text_lines = []
        for sex in sorted(df['Gender'].dropna().unique()):
            sex_str = str(sex)
            sub = df[df['Gender'] == sex_str]
            x = sub['Social_Score'].to_numpy()
            y = sub[metric].to_numpy()
            color = SEX_COLORS.get(sex_str, 'gray')

            ax.scatter(x, y, s=4, color=color, alpha=0.18,
                       linewidths=0, zorder=1,
                       label=SEX_LABELS.get(sex_str, sex_str))

            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x_grid, slope * x_grid + intercept,
                    color=color, linewidth=1.4, zorder=3)

            n = x.size
            boot_lines = np.empty((n_bootstrap, x_grid.size))
            for b in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                bs, bi = np.polyfit(x[idx], y[idx], 1)
                boot_lines[b] = bs * x_grid + bi
            lo = np.percentile(boot_lines, 2.5, axis=0)
            hi = np.percentile(boot_lines, 97.5, axis=0)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18,
                            linewidth=0, zorder=2)

            r, p = pearsonr(x, y)
            text_lines.append(
                f'{SEX_LABELS.get(sex_str, sex_str)}: '
                f'r = {r:+.2f}, p = {p:.3f}'
            )

        ax.text(0.03, 0.97, '\n'.join(text_lines),
                transform=ax.transAxes, ha='left', va='top',
                fontsize=6)

        ax.set_xlabel('Social difficulty score')
        ax.set_ylabel(f'{label} (residualised)')
        ax.set_title(label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='outside lower center',
                   ncol=len(legend_handles), frameon=False,
                   handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.0)

    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def compute_atypicality_scores(df):
    """Per-subject atypicality scores from the residualised metrics.

    For each metric (modularity, global_efficiency), atypicality is the
    absolute deviation of the (already covariate-residualised) value from
    the sample mean:

        atyp_m = | modularity − mean(modularity) |

    A joint atypicality is computed as the Mahalanobis distance from the
    bivariate centroid in (modularity, global_efficiency) space, using
    the pooled sample covariance — capturing correlated deviations that
    the per-metric scores miss.

    Constructively, |residual| ~ pgs_z is the same signal the C3
    Breusch–Pagan tests target via residual² ~ pgs_z; the Mahalanobis
    score is the genuinely new piece (joint, not per-metric).
    """
    out = df.copy()
    for metric, _ in METRICS:
        x = out[metric].to_numpy()
        out[f'atypicality_{metric}'] = np.abs(x - float(np.mean(x)))

    xy = out[['modularity', 'global_efficiency']].to_numpy()
    mu = xy.mean(axis=0)
    cov = np.cov(xy.T)
    inv_cov = np.linalg.inv(cov)
    diff = xy - mu
    out['atypicality_joint'] = np.sqrt(
        np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    )
    return out


def summarise_atypicality_pgs(df):
    """Spearman + Pearson correlations of each atypicality score with
    pgs_z, pooled and within each sex. Returns a tidy DataFrame.
    """
    measures = [
        'atypicality_modularity',
        'atypicality_global_efficiency',
        'atypicality_joint',
    ]
    rows = []
    groups = [('all', df)]
    for sex in sorted(df['Gender'].dropna().unique()):
        groups.append((str(sex), df[df['Gender'] == sex]))
    for col in measures:
        for label, sub in groups:
            x = sub['pgs_z'].to_numpy()
            y = sub[col].to_numpy()
            rho, p_rho = spearmanr(x, y)
            r, p_r = pearsonr(x, y)
            rows.append({
                'measure': col,
                'group': label,
                'n': int(len(sub)),
                'spearman_rho': float(rho),
                'spearman_p': float(p_rho),
                'pearson_r': float(r),
                'pearson_p': float(p_r),
            })
    return pd.DataFrame(rows)


def plot_atypicality_pgs(df, output_path, n_bootstrap=1000, seed=0):
    """Three-panel figure: |modularity residual|, |global-efficiency
    residual|, and Mahalanobis joint atypicality, each plotted against
    pgs_z with per-sex regression lines, bootstrap 95% bands, and
    Spearman ρ annotations (per sex + pooled).

    Atypicality scores are computed on the C3 covariate-residualised
    metrics (Age, ICV, motion already removed); sex is shown as a
    moderator (separate lines), not residualised away — matching the
    sex × PGS framing in C3.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3,
                             figsize=(165 * mm2inches, 65 * mm2inches),
                             dpi=FIGURE_DPI,
                             constrained_layout=True)

    panels = [
        ('atypicality_modularity', '|Modularity − mean|'),
        ('atypicality_global_efficiency', '|Global efficiency − mean|'),
        ('atypicality_joint', 'Joint atypicality (Mahalanobis)'),
    ]

    x_grid = np.linspace(df['pgs_z'].min(), df['pgs_z'].max(), 100)

    legend_handles = None
    legend_labels = None
    for ax, (col, label) in zip(axes, panels):
        text_lines = []
        for sex in sorted(df['Gender'].dropna().unique()):
            sex_str = str(sex)
            sub = df[df['Gender'] == sex_str]
            x = sub['pgs_z'].to_numpy()
            y = sub[col].to_numpy()
            color = SEX_COLORS.get(sex_str, 'gray')

            ax.scatter(x, y, s=4, color=color, alpha=0.18,
                       linewidths=0, zorder=1,
                       label=SEX_LABELS.get(sex_str, sex_str))

            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x_grid, slope * x_grid + intercept,
                    color=color, linewidth=1.4, zorder=3)

            n = x.size
            boot_lines = np.empty((n_bootstrap, x_grid.size))
            for b in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                bs, bi = np.polyfit(x[idx], y[idx], 1)
                boot_lines[b] = bs * x_grid + bi
            lo = np.percentile(boot_lines, 2.5, axis=0)
            hi = np.percentile(boot_lines, 97.5, axis=0)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18,
                            linewidth=0, zorder=2)

            rho, p = spearmanr(x, y)
            text_lines.append(
                f'{SEX_LABELS.get(sex_str, sex_str)}: '
                f'ρ = {rho:+.2f}, p = {p:.3f}'
            )

        rho_all, p_all = spearmanr(df['pgs_z'], df[col])
        text_lines.append(f'All: ρ = {rho_all:+.2f}, p = {p_all:.3f}')

        ax.text(0.03, 0.97, '\n'.join(text_lines),
                transform=ax.transAxes, ha='left', va='top',
                fontsize=6)

        ax.set_xlabel('PGS (z)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='outside lower center',
                   ncol=len(legend_handles), frameon=False,
                   handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.0)

    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_modularity_efficiency_snake_by_pgs(df, output_path,
                                            n_std=1.0,
                                            n_anchors=11,
                                            window_frac=0.25):
    """Two-panel figure (Men | Women) showing how the joint distribution
    of (modularity, global efficiency) shifts continuously with PGS.

    For each sex, subjects are sorted by pgs_z and a sliding window of
    ``window_frac * n`` subjects is stepped across PGS at ``n_anchors``
    evenly-spaced positions. A 1-SD covariance ellipse is fitted per
    window and coloured by the window's median pgs_z, producing a
    "snake" of ellipses through bivariate space. Centroids are joined
    by a trajectory line so the eye can follow the drift.

    Plotted on the *covariate-residualised* metrics so the spread
    visible in each ellipse corresponds to what the C3 heteroscedasticity
    test reasons about.
    """
    fig, axes = plt.subplots(1, 2,
                             figsize=(150 * mm2inches, 75 * mm2inches),
                             dpi=FIGURE_DPI,
                             constrained_layout=True)

    panels = [('M', 'Men'), ('F', 'Women')]

    x_all = df['modularity'].to_numpy()
    y_all = df['global_efficiency'].to_numpy()
    x_lo, x_hi = np.percentile(x_all, [1, 99])
    y_lo, y_hi = np.percentile(y_all, [1, 99])
    x_pad = (x_hi - x_lo) * 0.08
    y_pad = (y_hi - y_lo) * 0.08
    x_lim = (x_lo - x_pad, x_hi + x_pad)
    y_lim = (y_lo - y_pad, y_hi + y_pad)

    pgs_abs_99 = float(np.percentile(np.abs(df['pgs_z'].to_numpy()), 99))
    vlim = pgs_abs_99 if pgs_abs_99 > 0 else 1.0
    cmap = plt.get_cmap('RdBu_r')  # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=-vlim, vmax=vlim)

    sm_handle = None
    for ax, (sex, label) in zip(axes, panels):
        sub = df[df['Gender'] == sex].sort_values('pgs_z')
        if sub.empty:
            ax.set_title(label)
            continue

        ax.scatter(sub['modularity'], sub['global_efficiency'],
                   s=4, color='0.7', alpha=0.18, linewidths=0,
                   zorder=1)

        x = sub['modularity'].to_numpy()
        y = sub['global_efficiency'].to_numpy()
        pgs = sub['pgs_z'].to_numpy()

        n = x.size
        window = max(int(round(window_frac * n)), 5)
        half = window // 2
        anchor_centers = np.linspace(half, n - half - 1, n_anchors,
                                     dtype=int)

        records = []
        for c in anchor_centers:
            i0 = max(0, c - half)
            i1 = min(n, i0 + window)
            i0 = max(0, i1 - window)
            params = _covariance_ellipse(x[i0:i1], y[i0:i1], n_std=n_std)
            if params is None:
                continue
            local_pgs = float(np.median(pgs[i0:i1]))
            records.append((params, local_pgs))

        if records:
            cx = [p[0][0] for p, _ in records]
            cy = [p[0][1] for p, _ in records]
            ax.plot(cx, cy, color='0.25', linewidth=0.8, alpha=0.55,
                    zorder=3)

        for params, local_pgs in records:
            xy, w, h, ang = params
            color = mcolors.to_hex(cmap(norm(local_pgs)))
            ax.add_patch(Ellipse(xy=xy, width=w, height=h, angle=ang,
                                 facecolor='none', edgecolor=color,
                                 linewidth=1.0, alpha=0.8, zorder=2))
            ax.plot(xy[0], xy[1], marker='o', markersize=3,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.5, linestyle='None', zorder=4)

        ax.set_xlabel('Modularity (residualised)')
        ax.set_ylabel('Global efficiency (residualised)')
        ax.set_title(label)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sm_handle = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore[attr-defined]

    if sm_handle is not None:
        cbar = fig.colorbar(sm_handle, ax=axes,
                            orientation='vertical',
                            fraction=0.04, pad=0.02, shrink=0.85)
        cbar.set_label(f'PGS (z), window = {int(window_frac * 100)}% '
                       f'of n, {n_std:g} SD ellipse', fontsize=6)
        cbar.ax.tick_params(labelsize=6)

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


# %%
def main():
    parser = argparse.ArgumentParser(
        description='D2 publication figures: per-quintile CV by sex.'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--input', default=None,
                        help='Path to C3 heteroscedasticity results CSV '
                             '(default: results/C3_heteroscedasticity_results.csv)')
    parser.add_argument('--interaction-stats', default=None,
                        help='Path to C3 variance-regression-by-sex CSV '
                             '(default: results/C3_variance_regression_sex.csv)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Bootstrap iterations for CV CIs (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for bootstrap (default: 42)')
    args = parser.parse_args()

    project_folder = Path(args.project)
    figures_dir = project_folder / 'figures'
    results_dir = project_folder / 'results'
    reports_dir = project_folder / 'reports'
    for d in (figures_dir, results_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    input_file = (Path(args.input) if args.input
                  else results_dir / 'C3_heteroscedasticity_results.csv')

    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Gender', 'pgs_z', 'Social_Score',
                           'modularity_raw', 'global_efficiency_raw',
                           'modularity', 'global_efficiency'])

    all_rows = []
    for metric, _ in METRICS:
        all_rows.extend(compute_quintile_cv_by_sex(
            df, metric, n_bootstrap=args.n_bootstrap, seed=args.seed))
    quintile_df = pd.DataFrame(all_rows)

    csv_path = results_dir / 'D2_quintile_cv_by_sex.csv'
    quintile_df.to_csv(csv_path, index=False)

    interaction_path = (Path(args.interaction_stats) if args.interaction_stats
                        else results_dir / 'C3_variance_regression_sex.csv')
    interaction_stats = None
    if interaction_path.exists():
        vr_df = pd.read_csv(interaction_path)
        interaction_rows = vr_df[vr_df['term'] == 'pgs_z:Gender']
        interaction_stats = {
            row['metric']: {'beta': row['beta'], 'pvalue': row['pvalue']}
            for _, row in interaction_rows.iterrows()
        }

    fig_path = figures_dir / 'D2_quintile_cv_by_sex.png'
    plot_quintile_cv_by_sex(quintile_df, fig_path,
                            interaction_stats=interaction_stats)

    kde_fig_path = figures_dir / 'D2_modularity_efficiency_kde_by_pgs.png'
    plot_modularity_efficiency_kde_by_pgs(df, kde_fig_path,
                                          seed=args.seed)

    snake_fig_path = (figures_dir
                      / 'D2_modularity_efficiency_snake_by_pgs.png')
    plot_modularity_efficiency_snake_by_pgs(df, snake_fig_path)

    social_fig_path = figures_dir / 'D2_social_metric_association.png'
    plot_social_metric_association(df, social_fig_path,
                                   n_bootstrap=args.n_bootstrap,
                                   seed=args.seed)

    atyp_df = compute_atypicality_scores(df)
    atyp_summary_df = summarise_atypicality_pgs(atyp_df)

    atyp_subject_cols = ['Subject', 'Gender', 'pgs_z',
                         'modularity', 'global_efficiency',
                         'atypicality_modularity',
                         'atypicality_global_efficiency',
                         'atypicality_joint']
    atyp_subject_cols = [c for c in atyp_subject_cols
                         if c in atyp_df.columns]
    atyp_csv = results_dir / 'D2_atypicality_scores.csv'
    atyp_df[atyp_subject_cols].to_csv(atyp_csv, index=False)

    atyp_summary_csv = results_dir / 'D2_atypicality_pgs_correlations.csv'
    atyp_summary_df.to_csv(atyp_summary_csv, index=False)

    atyp_fig_path = figures_dir / 'D2_atypicality_pgs.png'
    plot_atypicality_pgs(atyp_df, atyp_fig_path,
                         n_bootstrap=args.n_bootstrap,
                         seed=args.seed)

    report_lines = [
        '=' * 80,
        'D2: PUBLICATION FIGURE - PER-QUINTILE CV BY SEX',
        '=' * 80,
        '',
        f'Input            : {input_file}',
        f'N subjects       : {len(df)}',
        f'N bootstrap      : {args.n_bootstrap}',
        f'Random seed      : {args.seed}',
        '',
        f'Figure           : {fig_path}',
        f'KDE figure       : {kde_fig_path}',
        f'Social figure    : {social_fig_path}',
        f'Atypicality fig  : {atyp_fig_path}',
        f'Per-quintile CSV : {csv_path}',
        f'Atypicality CSV  : {atyp_csv}',
        f'Atypicality corr : {atyp_summary_csv}',
        '',
        'Per-sex quintile-CV trend (Pearson r between quintile rank and CV):',
    ]
    for metric, label in METRICS:
        report_lines.append(f'  {label}:')
        sub = quintile_df[quintile_df['metric'] == metric]
        for sex_key, sex_df in sub.groupby('sex'):
            sex = str(sex_key)
            r, p = pearsonr(sex_df['quintile'], sex_df['cv'])
            report_lines.append(
                f'    {SEX_LABELS.get(sex, sex):<6} (n={int(sex_df["n"].sum())}): '
                f'r = {r:+.3f}, p = {p:.4f}')

    report_lines += [
        '',
        'Per-sex social-difficulty x graph-metric association '
        '(residualised metrics):',
    ]
    for metric, label in METRICS:
        report_lines.append(f'  {label}:')
        for sex_key in sorted(df['Gender'].dropna().unique()):
            sex = str(sex_key)
            sub = df[df['Gender'] == sex]
            r, p = pearsonr(sub['Social_Score'], sub[metric])
            report_lines.append(
                f'    {SEX_LABELS.get(sex, sex):<6} (n={len(sub)}): '
                f'r = {r:+.3f}, p = {p:.4f}')

    report_lines += [
        '',
        'Atypicality (|residualised metric − mean|, joint Mahalanobis) '
        '~ PGS (Spearman):',
    ]
    atyp_panel_labels = {
        'atypicality_modularity': '|Modularity − mean|',
        'atypicality_global_efficiency': '|Global efficiency − mean|',
        'atypicality_joint': 'Joint Mahalanobis',
    }
    group_labels = {'all': 'All', 'F': 'Women', 'M': 'Men'}
    for col, label in atyp_panel_labels.items():
        report_lines.append(f'  {label}:')
        for group_key in ['all', 'F', 'M']:
            row = atyp_summary_df[
                (atyp_summary_df['measure'] == col)
                & (atyp_summary_df['group'] == group_key)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            report_lines.append(
                f'    {group_labels[group_key]:<6} (n={int(r["n"])}): '
                f'ρ = {r["spearman_rho"]:+.3f}, p = {r["spearman_p"]:.4f}  '
                f'(Pearson r = {r["pearson_r"]:+.3f}, '
                f'p = {r["pearson_p"]:.4f})'
            )

    report_lines += ['', '=' * 80, 'END OF REPORT', '=' * 80]
    report_text = '\n'.join(report_lines)

    report_path = reports_dir / 'D2_publication_figures_report.txt'
    with open(report_path, 'w') as fh:
        fh.write(report_text)

    print(report_text)


# %%
if __name__ == '__main__':
    main()

# %%
