#!/usr/bin/env python
"""
C3c: Ancestry checks for the PGS and the network-variance analyses
==================================================================

Reviewer request (A6): (1) does the PGS distribution differ by ancestry;
(2) are the PGS extremes ancestrally enriched; (3) does network-metric
variance differ by ancestry group; (4) add ancestry PCs to the variance
model. A fifth check repeats the primary variance tests within the
European-ancestry genetic cluster.

Ancestry is characterised two ways: self-reported HCP race/ethnicity
(collapsed to White / Black / Asian / Other, plus a Hispanic indicator) and
the within-sample genotype principal components (PC1-PC10 from B2). The
PGS enters the primary analyses residualised for age, sex and PC1-PC5
(B5); both the raw BLUP score and the residualised score are examined.

Inputs
------
  results/C3_heteroscedasticity_results.csv   per-subject metrics (from C3)
  results/pgs_residuals.csv                   raw blup_PGS + residuals
  data/PLINK_anonymised/Neuro_Chip_full_sample_pca.eigenvec
  data/raw_anonymised/phenotypic_data_anonymised.csv (Race, Ethnicity, Family_ID)

Outputs
-------
  results/C3c_ancestry_pgs_by_group.csv
  results/C3c_ancestry_extremes.csv
  results/C3c_ancestry_metric_variance.csv
  results/C3c_ancestry_variance_models.csv
  figures/C3c_ancestry_checks.png
  reports/C3c_ancestry_checks_report.txt
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
    _build_main_design, _MAIN_COEF_NAMES, mm2inches, FIGURE_DPI)
from C3b_family_structure_sensitivity import (  # noqa: E402
    sample_statistics, _within_sex_participant_source, METRICS,
    METRIC_LABELS, SEX_LEVELS)

GROUP_ORDER = ['White', 'Black', 'Asian', 'Other']
GROUP_COLORS = {'White': '#2a78d6', 'Black': '#eb6834', 'Asian': '#1baf7a',
                'Other': '#eda100'}
RACE_MAP = {
    'White': 'White',
    'Black or African Am.': 'Black',
    'Asian/Nat. Hawaiian/Othr Pacific Is.': 'Asian',
    'More than one': 'Other',
    'Unknown or Not Reported': 'Other',
    'Am. Indian/Alaskan Nat.': 'Other',
}
EXTREME_DEFS = {          # label: (low cut, high cut) on pgs_z
    'reference groups (z < -1 / z > 1)': (-1.0, 1.0),
}


# %%
# =============================================================================
# 1. DATA
# =============================================================================

def load_data(results_csv, pgs_csv, eigenvec, phenotypic_csv, report):
    res = pd.read_csv(results_csv, dtype={'Subject': str})
    pgs = pd.read_csv(pgs_csv, dtype={'Subject': str})[['Subject', 'blup_PGS']]
    pcs = pd.read_csv(eigenvec, sep=r'\s+', header=None, dtype={1: str})
    pcs.columns = ['FID', 'Subject'] + [f'PC{i}' for i in range(1, pcs.shape[1] - 1)]
    ph = pd.read_csv(phenotypic_csv, dtype=str,
                     usecols=['Individual_ID', 'Race', 'Ethnicity', 'Family_ID'])
    ph = ph.rename(columns={'Individual_ID': 'Subject'})
    df = (res.merge(pgs, on='Subject', validate='one_to_one')
             .merge(pcs.drop(columns='FID'), on='Subject', validate='one_to_one')
             .merge(ph, on='Subject', validate='one_to_one'))
    if len(df) != len(res):
        raise ValueError("PCs or phenotypic rows missing for some participants")
    df['group'] = df['Race'].map(RACE_MAP)
    if df['group'].isna().any():
        raise ValueError(f"unmapped Race values: {df.loc[df.group.isna(), 'Race'].unique()}")
    df['hispanic'] = (df['Ethnicity'] == 'Hispanic/Latino').astype(int)
    df['male'] = (df['Gender'] == 'M').astype(float)
    df['family'] = pd.factorize(df['Family_ID'])[0]

    # European-ancestry genetic cluster: within 3 SD of the self-reported
    # White, non-Hispanic centroid on PC1 and PC2 (the two ancestry axes).
    ref = df[(df['group'] == 'White') & (df['hispanic'] == 0)]
    z1 = (df['PC1'] - ref['PC1'].mean()) / ref['PC1'].std(ddof=1)
    z2 = (df['PC2'] - ref['PC2'].mean()) / ref['PC2'].std(ddof=1)
    df['eur_cluster'] = ((z1.abs() <= 3) & (z2.abs() <= 3)).astype(int)

    report.append(f"Participants: {len(df)}")
    report.append("Self-reported race (collapsed): " + ", ".join(
        f"{g}: {int((df.group == g).sum())}" for g in GROUP_ORDER))
    report.append(f"Hispanic/Latino ethnicity: {int(df.hispanic.sum())}")
    report.append(f"European-ancestry genetic cluster (PC1/PC2 within 3 SD of the "
                  f"White non-Hispanic centroid): {int(df.eur_cluster.sum())}; "
                  "composition by self-report: " + ", ".join(
                      f"{g}: {int(((df.group == g) & (df.eur_cluster == 1)).sum())}"
                      for g in GROUP_ORDER))
    return df


# %%
# =============================================================================
# 2. PGS DISTRIBUTION BY ANCESTRY
# =============================================================================

def pgs_by_group(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("1. PGS DISTRIBUTION BY ANCESTRY GROUP")
    report.append("=" * 60)
    rows = []
    for score, label in (('blup_PGS', 'raw BLUP PGS'),
                         ('pgs_z', 'PGS residualised for age, sex, PC1-5 (z)')):
        groups = [df.loc[df.group == g, score].values for g in GROUP_ORDER]
        welch = stats.f_oneway(*groups)              # classical ANOVA
        kw = stats.kruskal(*groups)
        lev = stats.levene(*groups, center='median')
        ref = df.loc[df.group == 'White', score]
        report.append(f"  {label}: ANOVA p = {welch.pvalue:.3g}, Kruskal-Wallis p = "
                      f"{kw.pvalue:.3g}, Brown-Forsythe (variance) p = {lev.pvalue:.3g}")
        for g in GROUP_ORDER:
            v = df.loc[df.group == g, score]
            smd = (v.mean() - ref.mean()) / np.sqrt((v.var(ddof=1) + ref.var(ddof=1)) / 2)
            rows.append({'score': score, 'group': g, 'n': len(v), 'mean': v.mean(),
                         'sd': v.std(ddof=1), 'median': v.median(),
                         'smd_vs_white': smd, 'anova_p': welch.pvalue,
                         'kruskal_p': kw.pvalue, 'brown_forsythe_p': lev.pvalue})
            report.append(f"    {g:<6} n = {len(v):3d}, mean = {v.mean():7.3f}, "
                          f"SD = {v.std(ddof=1):.3f}, SMD vs White = {smd:6.2f}")
        # PCs 1-5 explain how much of the raw score?
        X = sm.add_constant(df[[f'PC{i}' for i in range(1, 6)]].values)
        r2 = sm.OLS(df[score].values, X).fit().rsquared
        report.append(f"    R^2 of {label} on PC1-5: {r2:.3f}")
        rows.append({'score': score, 'group': 'ALL', 'n': len(df), 'mean': df[score].mean(),
                     'sd': df[score].std(ddof=1), 'median': df[score].median(),
                     'smd_vs_white': np.nan, 'anova_p': welch.pvalue,
                     'kruskal_p': kw.pvalue, 'brown_forsythe_p': lev.pvalue,
                     'r2_on_pc1_5': r2})
    for score in ('blup_PGS', 'pgs_z'):
        a = df.loc[df.hispanic == 1, score]
        b = df.loc[df.hispanic == 0, score]
        t = stats.ttest_ind(a, b, equal_var=False)
        report.append(f"  {score}: Hispanic (n = {len(a)}) vs not: mean "
                      f"{a.mean():.3f} vs {b.mean():.3f}, Welch p = {t.pvalue:.3g}")
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 3. ANCESTRAL ENRICHMENT OF THE PGS EXTREMES
# =============================================================================

def _wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def extremes_by_group(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("2. ANCESTRAL COMPOSITION OF THE PGS EXTREMES (residualised PGS)")
    report.append("=" * 60)
    rows = []
    defs = dict(EXTREME_DEFS)
    q = df['pgs_z'].quantile([0.1, 0.9])
    defs['deciles (bottom / top 10%)'] = (q.iloc[0], q.iloc[1])
    for label, (lo, hi) in defs.items():
        df['_tail'] = np.where(df.pgs_z < lo, 'low', np.where(df.pgs_z > hi, 'high', 'mid'))
        ct = pd.crosstab(df['group'], df['_tail']).reindex(GROUP_ORDER).fillna(0)
        chi2, p_chi, _, _ = stats.chi2_contingency(ct.values)
        report.append(f"  {label}: chi-square(group x low/mid/high) p = {p_chi:.3g}")
        for tail in ('low', 'high'):
            n_tail = int((df._tail == tail).sum())
            for g in GROUP_ORDER:
                k = int(((df._tail == tail) & (df.group == g)).sum())
                n_g = int((df.group == g).sum())
                share_tail = k / n_tail
                share_all = n_g / len(df)
                lo_ci, hi_ci = _wilson(k, n_tail)
                rows.append({'definition': label, 'tail': tail, 'group': g,
                             'n_in_tail': k, 'n_tail': n_tail, 'share_in_tail': share_tail,
                             'share_in_sample': share_all,
                             'enrichment_ratio': share_tail / share_all if share_all else np.nan,
                             'share_ci_low': lo_ci, 'share_ci_high': hi_ci,
                             'chi2_p': p_chi})
            comp = ", ".join(
                f"{g} {int(((df._tail == tail) & (df.group == g)).sum())}"
                f" ({100 * ((df._tail == tail) & (df.group == g)).sum() / n_tail:.0f}%)"
                for g in GROUP_ORDER)
            report.append(f"    {tail:<4} tail (n = {n_tail}): {comp}")
        report.append("    sample composition: " + ", ".join(
            f"{g} {100 * (df.group == g).mean():.0f}%" for g in GROUP_ORDER))
    df.drop(columns='_tail', inplace=True)
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 4. NETWORK-METRIC VARIANCE BY ANCESTRY GROUP
# =============================================================================

def metric_variance_by_group(df, rng, n_boot, report):
    report.append("")
    report.append("=" * 60)
    report.append("3. NETWORK-METRIC MEAN AND VARIANCE BY ANCESTRY GROUP")
    report.append("=" * 60)
    report.append("  Residualised metrics (age, ICV, motion); Brown-Forsythe test of "
                  "equal variances; bootstrap 95% CI of the SD")
    rows = []
    for metric in METRICS:
        groups = [df.loc[df.group == g, metric].values for g in GROUP_ORDER]
        lev = stats.levene(*groups, center='median')
        anova = stats.f_oneway(*groups)
        report.append(f"  {METRIC_LABELS[metric]}: Brown-Forsythe p = {lev.pvalue:.3g}; "
                      f"ANOVA (means) p = {anova.pvalue:.3g}")
        for g, vals in zip(GROUP_ORDER, groups):
            boots = np.array([rng.choice(vals, len(vals), replace=True).std(ddof=1)
                              for _ in range(n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({'metric': metric, 'group': g, 'n': len(vals),
                         'mean': vals.mean(), 'sd': vals.std(ddof=1),
                         'sd_ci_low': lo, 'sd_ci_high': hi,
                         'sd_ratio_vs_white': vals.std(ddof=1) / groups[0].std(ddof=1),
                         'brown_forsythe_p': lev.pvalue, 'anova_p': anova.pvalue})
            report.append(f"    {g:<6} n = {len(vals):3d}, mean = {vals.mean():8.5f}, "
                          f"SD = {vals.std(ddof=1):.5f} [{lo:.5f}, {hi:.5f}], "
                          f"SD ratio vs White = {vals.std(ddof=1) / groups[0].std(ddof=1):.2f}")
        # same within sex (the primary tests are sex-specific)
        for sex, code in zip(SEX_LEVELS, (0.0, 1.0)):
            sub = df[df.male == code]
            g_sex = [sub.loc[sub.group == g, metric].values for g in GROUP_ORDER]
            g_sex = [v for v in g_sex if len(v) >= 5]
            lev_s = stats.levene(*g_sex, center='median')
            report.append(f"    within {sex}: Brown-Forsythe p = {lev_s.pvalue:.3g} "
                          f"(groups with n >= 5)")
            rows.append({'metric': metric, 'group': f'within_{sex}', 'n': len(sub),
                         'mean': np.nan, 'sd': np.nan, 'sd_ci_low': np.nan,
                         'sd_ci_high': np.nan, 'sd_ratio_vs_white': np.nan,
                         'brown_forsythe_p': lev_s.pvalue, 'anova_p': np.nan})
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 5. VARIANCE MODEL WITH ANCESTRY PCs
# =============================================================================

def _ols_rows(res, names, metric, model, n_families, note):
    ci = np.asarray(res.conf_int())
    out = []
    for i, name in enumerate(names):
        out.append({'metric': metric, 'model': model, 'term': name,
                    'estimate': float(res.params[i]), 'se': float(res.bse[i]),
                    'ci_low': float(ci[i, 0]), 'ci_high': float(ci[i, 1]),
                    'pvalue': float(res.pvalues[i]), 'n': int(res.nobs),
                    'n_families': n_families, 'note': note})
    return out


def variance_models_with_pcs(df, report):
    report.append("")
    report.append("=" * 60)
    report.append("4. VARIANCE REGRESSION WITH ANCESTRY PCs")
    report.append("=" * 60)
    report.append("  Mean: metric ~ pgs_z + sex + pgs_z:sex [+ PCs]; "
                  "variance: resid^2 ~ same design; OLS and family cluster-robust p")
    fam = df['family'].values
    n_fam = int(fam.max() + 1)
    rows = []
    base = _build_main_design(df)
    specs = {'no PCs (primary)': [], 'PC1-2': [1, 2], 'PC1-5': [1, 2, 3, 4, 5],
             'PC1-10': list(range(1, 11)),
             'PC1-5 + Hispanic + group dummies': [1, 2, 3, 4, 5]}
    for metric in METRICS:
        y = df[metric].values.astype(float)
        for label, pcs in specs.items():
            extra = df[[f'PC{i}' for i in pcs]].values if pcs else np.empty((len(df), 0))
            names = list(_MAIN_COEF_NAMES) + [f'PC{i}' for i in pcs]
            if 'group dummies' in label:
                dummies = pd.get_dummies(df['group'], drop_first=False)[
                    [g for g in GROUP_ORDER if g != 'White']].astype(float).values
                extra = np.column_stack([extra, df[['hispanic']].values.astype(float), dummies])
                names += ['hispanic'] + [f'group_{g}' for g in GROUP_ORDER if g != 'White']
            X = np.column_stack([base, extra])
            mean_fit = sm.OLS(y, X).fit()
            e2 = mean_fit.resid ** 2
            aux = sm.OLS(e2, X).fit()
            aux_cr = sm.OLS(e2, X).fit(cov_type='cluster', cov_kwds={'groups': fam},
                                       use_t=True)
            # joint test of the added ancestry terms in the variance model
            k_extra = X.shape[1] - base.shape[1]
            if k_extra:
                R = np.zeros((k_extra, X.shape[1]))
                R[np.arange(k_extra), base.shape[1] + np.arange(k_extra)] = 1
                joint_p = float(aux.f_test(R).pvalue)
            else:
                joint_p = np.nan
            rows += _ols_rows(aux, names, metric, f'variance | {label}', n_fam,
                              f'OLS; joint F test of ancestry terms p = {joint_p:.3g}')
            rows.append({'metric': metric, 'model': f'variance | {label}',
                         'term': 'pgs_z:Gender (family cluster-robust)',
                         'estimate': float(aux_cr.params[3]), 'se': float(aux_cr.bse[3]),
                         'ci_low': float(aux_cr.conf_int()[3][0]),
                         'ci_high': float(aux_cr.conf_int()[3][1]),
                         'pvalue': float(aux_cr.pvalues[3]), 'n': int(len(y)),
                         'n_families': n_fam, 'note': 'CR1, t on G-1 df'})
            rows += _ols_rows(mean_fit, names, metric, f'mean | {label}', n_fam, 'OLS')
            report.append(f"  {METRIC_LABELS[metric]:<18} {label:<34} pgs_z:sex on variance: "
                          f"beta = {aux.params[3]:.4e}, p = {aux.pvalues[3]:.4f} "
                          f"(cluster-robust p = {aux_cr.pvalues[3]:.4f}); "
                          f"ancestry terms joint p = {joint_p:.3g}")
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 6. PRIMARY VARIANCE TESTS WITHIN THE EUROPEAN-ANCESTRY CLUSTER
# =============================================================================

def eur_cluster_sensitivity(df, n_perm, rng, report):
    report.append("")
    report.append("=" * 60)
    report.append("5. PRIMARY VARIANCE TESTS WITHIN THE EUROPEAN-ANCESTRY CLUSTER")
    report.append("=" * 60)
    sub = df[df.eur_cluster == 1].reset_index(drop=True)
    report.append(f"  n = {len(sub)} ({int(sub.male.sum())} male, "
                  f"{int((1 - sub.male).sum())} female); PGS re-standardised within "
                  "the cluster; within-sex permutation p (PGS permuted within sex)")
    sub['pgs_z'] = (sub['pgs_z'] - sub['pgs_z'].mean()) / sub['pgs_z'].std(ddof=1)
    rows = []
    male = sub['male'].values.astype(float)
    pgs = sub['pgs_z'].values.astype(float)
    keys = [('vr_interaction_beta', 'vr_interaction_t', 'variance pgs_z:sex beta', 'two'),
            ('decile_r_F', 'decile_r_F', 'decile-SD trend r, F', 'one'),
            ('decile_r_M', 'decile_r_M', 'decile-SD trend r, M', 'one'),
            ('decile_diff', 'decile_diff', 'decile-SD trend M - F', 'two'),
            ('bin_r_F', 'bin_r_F', 'balanced-bin trend r, F', 'one'),
            ('bin_r_M', 'bin_r_M', 'balanced-bin trend r, M', 'one'),
            ('bin_diff', 'bin_diff', 'balanced-bin trend M - F', 'two')]
    for metric in METRICS:
        y = sub[metric].values.astype(float)
        obs = sample_statistics(y, pgs, male)
        X = _build_main_design(sub)
        e2 = sm.OLS(y, X).fit().resid ** 2
        aux = sm.OLS(e2, X).fit()
        null = pd.DataFrame.from_records(
            [sample_statistics(y, pgs[_within_sex_participant_source(male, rng)], male)
             for _ in range(n_perm)])
        for est_key, stat_key, label, sided in keys:
            nd = null[stat_key].dropna().values
            o = obs[stat_key]
            if sided == 'two':
                p = (1 + np.sum(np.abs(nd) >= abs(o))) / (1 + len(nd))
            else:
                p = (1 + np.sum(nd >= o)) / (1 + len(nd))
            asym = aux.pvalues[3] if est_key == 'vr_interaction_beta' else np.nan
            rows.append({'metric': metric, 'statistic': label, 'estimate': obs[est_key],
                         'asymptotic_p': asym, 'perm_within_sex_p': float(p),
                         'n': len(sub), 'B': n_perm})
            report.append(f"  {METRIC_LABELS[metric]:<18} {label:<28} = {obs[est_key]:9.4g}"
                          + (f", asymptotic p = {asym:.4f}" if np.isfinite(asym) else "")
                          + f", permutation p = {p:.4f}")
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 7. FIGURE
# =============================================================================

def make_figure(df, var_df, path):
    fig, axes = plt.subplots(2, 2, figsize=(180 * mm2inches, 140 * mm2inches),
                             dpi=FIGURE_DPI)
    # (a) PC1 vs PC2 by self-reported group, EUR cluster outlined
    ax = axes[0, 0]
    for g in GROUP_ORDER:
        s = df[df.group == g]
        ax.scatter(s.PC1, s.PC2, s=6, color=GROUP_COLORS[g], alpha=0.7,
                   label=f"{g} (n = {len(s)})", edgecolors='none')
    eur = df[df.eur_cluster == 1]
    ax.scatter(eur.PC1, eur.PC2, s=14, facecolors='none', edgecolors='#444',
               linewidths=0.3, label=f"European-ancestry cluster (n = {len(eur)})")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('Genotype PCs by self-reported race', fontsize=8)
    ax.legend(fontsize=5.5, frameon=True, framealpha=1, edgecolor='none')
    # (b) raw and residualised PGS by group
    ax = axes[0, 1]
    pos = np.arange(len(GROUP_ORDER))
    for k, (score, lab, off) in enumerate((('blup_PGS', 'raw BLUP PGS (z within sample)', -0.18),
                                           ('pgs_z', 'PGS residualised for PC1-5', 0.18))):
        vals = df[score]
        z = (vals - vals.mean()) / vals.std(ddof=1)
        means = [z[df.group == g].mean() for g in GROUP_ORDER]
        sds = [z[df.group == g].std(ddof=1) for g in GROUP_ORDER]
        ax.errorbar(pos + off, means, yerr=sds, fmt='o', markersize=4, capsize=2,
                    color=['#2a78d6', '#eb6834'][k], label=lab, linewidth=1)
    ax.axhline(0, color='#999', linewidth=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(GROUP_ORDER, fontsize=7)
    ax.set_ylabel('mean +/- SD (z units)'); ax.set_title('PGS by ancestry group', fontsize=8)
    ax.legend(fontsize=6, frameon=True, framealpha=1, edgecolor='none')
    # (c, d) metric SD by group with bootstrap CI
    for ax, metric in zip(axes[1], METRICS):
        v = var_df[(var_df.metric == metric) & (var_df.group.isin(GROUP_ORDER))]
        ax.errorbar(pos, v.sd, yerr=[v.sd - v.sd_ci_low, v.sd_ci_high - v.sd], fmt='o',
                    markersize=4, capsize=2, color='#2a78d6', linewidth=1)
        ax.set_xticks(pos); ax.set_xticklabels(GROUP_ORDER, fontsize=7)
        ax.set_ylabel('SD of residualised metric')
        ax.set_title(f"{METRIC_LABELS[metric]}: SD by ancestry group", fontsize=8)
        ax.text(0.98, 0.04, f"Brown-Forsythe p = {v.brown_forsythe_p.iloc[0]:.2f}",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
                color='#444')
        ax.set_ylim(bottom=0)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


# %%
# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='C3c: ancestry checks')
    ap.add_argument('--project', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--pgs', required=True)
    ap.add_argument('--pca', required=True)
    ap.add_argument('--phenotypic', required=True)
    ap.add_argument('--n-bootstrap', type=int, default=5000)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    project = Path(args.project)
    results_dir, figures_dir, reports_dir = (project / 'results', project / 'figures',
                                             project / 'reports')
    for d in (results_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = ["=" * 80, "C3c: ANCESTRY CHECKS", "=" * 80, "",
              f"Bootstrap replicates: {args.n_bootstrap}; permutations: {args.n_perm}; "
              f"seed: {args.seed}", ""]
    ss = np.random.SeedSequence(args.seed)
    rng_boot, rng_perm = [np.random.default_rng(s) for s in ss.spawn(2)]

    df = load_data(Path(args.results), Path(args.pgs), Path(args.pca),
                   Path(args.phenotypic), report)
    pgs_df = pgs_by_group(df, report)
    ext_df = extremes_by_group(df, report)
    var_df = metric_variance_by_group(df, rng_boot, args.n_bootstrap, report)
    mod_df = variance_models_with_pcs(df, report)
    eur_df = eur_cluster_sensitivity(df, args.n_perm, rng_perm, report)

    pgs_df.to_csv(results_dir / 'C3c_ancestry_pgs_by_group.csv', index=False, float_format='%.10g')
    ext_df.to_csv(results_dir / 'C3c_ancestry_extremes.csv', index=False, float_format='%.10g')
    var_df.to_csv(results_dir / 'C3c_ancestry_metric_variance.csv', index=False, float_format='%.10g')
    mod_df.to_csv(results_dir / 'C3c_ancestry_variance_models.csv', index=False, float_format='%.10g')
    eur_df.to_csv(results_dir / 'C3c_ancestry_eur_cluster_sensitivity.csv', index=False, float_format='%.10g')
    make_figure(df, var_df, figures_dir / 'C3c_ancestry_checks.png')

    report += ["", f"Runtime: {time.time() - t0:.1f} s", "=" * 80, "END OF REPORT", "=" * 80]
    (reports_dir / 'C3c_ancestry_checks_report.txt').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
