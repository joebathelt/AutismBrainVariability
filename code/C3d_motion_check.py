#!/usr/bin/env python
"""
C3d: Does head-motion variance scale with the PGS?
==================================================

The network metrics are residualised for mean head motion (relative RMS)
before the variance analyses, which removes a linear mean effect of motion.
If the dispersion of motion itself increased with polygenic load (in a
sex-dependent way), residual metric variance could inherit that pattern.
This script therefore applies the C3 variance tests to head motion as the
outcome and re-estimates the modularity / global-efficiency variance model
with additional motion adjustment.

  1. Mean association of motion with the PGS (Spearman), by sex.
  2. Variance regression (resid^2 ~ pgs_z + sex + pgs_z:sex), decile-SD and
     PGS-balanced quintile-bin variance trends by sex, and the top-vs-bottom
     quintile SD ratio, with motion (and log motion) as the outcome;
     within-sex permutation p-values.
  3. Metric variance regression with motion and motion^2 added to the mean
     and variance models, and after excluding the 10% highest-motion
     participants.

Inputs:  results/C3_heteroscedasticity_results.csv
Outputs: results/C3d_motion_check.csv, figures/C3d_motion_check.png,
         reports/C3d_motion_check_report.txt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C3_continuous_heteroscedasticity_analysis import (  # noqa: E402
    _build_main_design, mm2inches, FIGURE_DPI)
from C3b_family_structure_sensitivity import (  # noqa: E402
    sample_statistics, _within_sex_participant_source, _sd_ratio_by_sex,
    _quantile_bins, METRICS, METRIC_LABELS, SEX_LEVELS)

MOTION = 'Movement_RelativeRMS_mean'
SEX_COLORS = {'F': '#2a78d6', 'M': '#eb6834'}
STAT_KEYS = [('vr_pgs_beta', 'vr_pgs_t', 'variance: pgs_z beta', 'two'),
             ('vr_interaction_beta', 'vr_interaction_t', 'variance: pgs_z:sex beta', 'two'),
             ('decile_r_F', 'decile_r_F', 'decile-SD trend r, F', 'one'),
             ('decile_r_M', 'decile_r_M', 'decile-SD trend r, M', 'one'),
             ('decile_diff', 'decile_diff', 'decile-SD trend M - F', 'two'),
             ('bin_r_F', 'bin_r_F', 'balanced-bin trend r, F', 'one'),
             ('bin_r_M', 'bin_r_M', 'balanced-bin trend r, M', 'one'),
             ('bin_diff', 'bin_diff', 'balanced-bin trend M - F', 'two')]


def motion_variance_tests(df, outcome, label, n_perm, rng, report):
    y = df[outcome].values.astype(float)
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    obs = sample_statistics(y, pgs, male)
    X = _build_main_design(df)
    mean_fit = sm.OLS(y, X).fit()
    aux = sm.OLS(mean_fit.resid ** 2, X).fit()
    null = pd.DataFrame.from_records(
        [sample_statistics(y, pgs[_within_sex_participant_source(male, rng)], male)
         for _ in range(n_perm)])
    rows = []
    report.append(f"  Outcome: {label}")
    report.append(f"    mean model pgs_z p = {mean_fit.pvalues[1]:.3f}, pgs_z:sex p = "
                  f"{mean_fit.pvalues[3]:.3f}")
    for est_key, stat_key, name, sided in STAT_KEYS:
        nd = null[stat_key].dropna().values
        o = obs[stat_key]
        p_perm = ((1 + np.sum(np.abs(nd) >= abs(o))) / (1 + len(nd)) if sided == 'two'
                  else (1 + np.sum(nd >= o)) / (1 + len(nd)))
        asym = {'vr_pgs_beta': aux.pvalues[1],
                'vr_interaction_beta': aux.pvalues[3]}.get(est_key, np.nan)
        rows.append({'analysis': 'motion_variance', 'outcome': label, 'statistic': name,
                     'estimate': obs[est_key], 'asymptotic_p': asym,
                     'perm_within_sex_p': float(p_perm), 'n': len(df), 'B': n_perm})
        report.append(f"    {name:<28} = {obs[est_key]:10.4g}"
                      + (f", asymptotic p = {asym:.3f}" if np.isfinite(asym) else "")
                      + f", permutation p = {p_perm:.3f}")
    ratio = _sd_ratio_by_sex(y, pgs, male)
    for sex in SEX_LEVELS:
        r = ratio[sex]
        rows.append({'analysis': 'motion_quintile_sd_ratio', 'outcome': label,
                     'statistic': f'SD top / bottom PGS quintile, {sex}',
                     'estimate': r['ratio'], 'asymptotic_p': np.nan,
                     'perm_within_sex_p': np.nan, 'n': len(df), 'B': np.nan})
        report.append(f"    top/bottom quintile SD ratio, {sex}: {r['ratio']:.2f} "
                      f"(SD {r['sd_top']:.4f} vs {r['sd_bottom']:.4f})")
    return rows


def metric_models_with_motion(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("3. METRIC VARIANCE REGRESSION WITH ADDITIONAL MOTION ADJUSTMENT")
    report.append("=" * 60)
    rows = []
    base = _build_main_design(df)
    m = df[MOTION].values.astype(float)
    m_c = (m - m.mean()) / m.std(ddof=1)
    cut = np.quantile(m, 0.9)
    specs = {
        'primary': (base, np.ones(len(df), bool)),
        '+ motion + motion^2': (np.column_stack([base, m_c, m_c ** 2]), np.ones(len(df), bool)),
        'excluding top 10% motion': (base, m <= cut),
    }
    for metric in METRICS:
        y = df[metric].values.astype(float)
        for label, (X, keep) in specs.items():
            fit = sm.OLS(y[keep], X[keep]).fit()
            aux = sm.OLS(fit.resid ** 2, X[keep]).fit()
            rows.append({'analysis': 'metric_variance_model', 'outcome': metric,
                         'statistic': f'pgs_z:sex beta | {label}',
                         'estimate': float(aux.params[3]), 'asymptotic_p': float(aux.pvalues[3]),
                         'perm_within_sex_p': np.nan, 'n': int(keep.sum()), 'B': np.nan})
            extra = ""
            if X.shape[1] > base.shape[1]:
                extra = (f"; motion terms in variance model p = {aux.pvalues[4]:.3f}, "
                         f"{aux.pvalues[5]:.3f}")
            report.append(f"  {METRIC_LABELS[metric]:<18} {label:<26} n = {int(keep.sum())}: "
                          f"pgs_z:sex beta = {aux.params[3]:.4e}, p = {aux.pvalues[3]:.4f}{extra}")
    # does |metric residual| relate to motion at all after residualisation?
    for metric in METRICS:
        e = sm.OLS(df[metric].values, base).fit().resid
        rho, p = stats.spearmanr(np.abs(e), m)
        report.append(f"  Spearman(|{metric} residual|, motion) = {rho:.3f}, p = {p:.3f}")
        rows.append({'analysis': 'abs_resid_vs_motion', 'outcome': metric,
                     'statistic': 'Spearman rho', 'estimate': rho, 'asymptotic_p': p,
                     'perm_within_sex_p': np.nan, 'n': len(df), 'B': np.nan})
    return rows


def make_figure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(180 * mm2inches, 70 * mm2inches), dpi=FIGURE_DPI)
    ax = axes[0]
    for sex, code in zip(SEX_LEVELS, (0.0, 1.0)):
        s = df[df.male == code]
        ax.scatter(s.pgs_z, s[MOTION], s=5, color=SEX_COLORS[sex], alpha=0.5,
                   edgecolors='none', label=f"{'female' if sex == 'F' else 'male'} (n = {len(s)})")
    ax.set_xlabel('PGS (z)'); ax.set_ylabel('head motion (relative RMS, mm)')
    ax.set_title('Head motion vs PGS', fontsize=8)
    ax.legend(fontsize=6, frameon=True, framealpha=1, edgecolor='none')
    ax = axes[1]
    for sex, code in zip(SEX_LEVELS, (0.0, 1.0)):
        s = df[df.male == code]
        bins = _quantile_bins(s.pgs_z.values, 5)
        centres = [s.pgs_z.values[bins == b].mean() for b in range(5)]
        sds = [s[MOTION].values[bins == b].std(ddof=1) for b in range(5)]
        ax.plot(centres, sds, '-o', color=SEX_COLORS[sex], markersize=4, linewidth=1,
                label='female' if sex == 'F' else 'male')
    ax.set_xlabel('PGS quintile centre (z, within sex)'); ax.set_ylabel('SD of head motion')
    ax.set_title('Motion dispersion across PGS quintiles', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, frameon=True, framealpha=1, edgecolor='none')
    for ax in axes:
        ax.tick_params(labelsize=6.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='C3d: head-motion variance vs PGS')
    ap.add_argument('--project', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    project = Path(args.project)
    for d in ('results', 'figures', 'reports'):
        (project / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(args.results, dtype={'Subject': str})
    df['male'] = (df['Gender'] == 'M').astype(float)
    df['log_motion'] = np.log(df[MOTION])
    rng = np.random.default_rng(np.random.SeedSequence(args.seed))
    report = ["=" * 80, "C3d: HEAD-MOTION VARIANCE AGAINST THE PGS", "=" * 80, "",
              f"N = {len(df)}; motion = {MOTION} (all < 0.2 mm after QC); "
              f"permutations: {args.n_perm}; seed: {args.seed}", "",
              "=" * 60, "1. MEAN ASSOCIATION OF MOTION WITH THE PGS", "=" * 60]
    rows = []
    for sex, code in (('all', None), ('F', 0.0), ('M', 1.0)):
        s = df if code is None else df[df.male == code]
        rho, p = stats.spearmanr(s.pgs_z, s[MOTION])
        report.append(f"  {sex:<3} n = {len(s)}: Spearman rho = {rho:.3f}, p = {p:.3f}; "
                      f"motion mean = {s[MOTION].mean():.4f}, SD = {s[MOTION].std(ddof=1):.4f}")
        rows.append({'analysis': 'motion_mean_association', 'outcome': MOTION,
                     'statistic': f'Spearman rho, {sex}', 'estimate': rho, 'asymptotic_p': p,
                     'perm_within_sex_p': np.nan, 'n': len(s), 'B': np.nan})
    report += ["", "=" * 60, "2. C3 VARIANCE TESTS APPLIED TO HEAD MOTION", "=" * 60,
               "  Same tests as for the network metrics; within-sex permutation of the PGS"]
    rows += motion_variance_tests(df, MOTION, 'head motion (relative RMS)', args.n_perm, rng, report)
    rows += motion_variance_tests(df, 'log_motion', 'log head motion', args.n_perm, rng, report)
    rows += metric_models_with_motion(df, report)
    pd.DataFrame(rows).to_csv(project / 'results' / 'C3d_motion_check.csv', index=False,
                              float_format='%.10g')
    make_figure(df, project / 'figures' / 'C3d_motion_check.png')
    report += ["", f"Runtime: {time.time() - t0:.1f} s", "=" * 80, "END OF REPORT", "=" * 80]
    (project / 'reports' / 'C3d_motion_check_report.txt').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
