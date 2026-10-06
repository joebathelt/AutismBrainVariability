#!/usr/bin/env python
"""
C3e: Pre-specified heteroscedasticity tests and density sensitivity
===================================================================

The Methods pre-specify four approaches to continuous heteroscedasticity
(Breusch-Pagan, White, quantile regression, PGS-balanced bootstrap) and a
density sensitivity analysis (15%, 20%, 25%). C3 reports the variance
regression with a sex x PGS interaction (whose omnibus LM is the
Breusch-Pagan statistic for that design) and the binned trends at 20%.
This script reports the remaining pre-specified tests in a form that
respects the sex x PGS model, and repeats every test at each density:

  1. Breusch-Pagan and White tests
       pooled       metric ~ pgs_z (the original, sex-free specification)
       interaction  metric ~ pgs_z + sex + pgs_z:sex; BP auxiliary on the
                    same design, White auxiliary adds pgs_z^2 and
                    pgs_z^2:sex; the sex-modulation terms of the White
                    auxiliary (pgs_z:sex, pgs_z^2:sex) are also tested jointly
       by sex       metric ~ pgs_z within each sex; BP auxiliary pgs_z,
                    White auxiliary pgs_z + pgs_z^2
  2. Quantile regression within each sex (tau = 0.10, 0.25, 0.50, 0.75,
     0.90): slope per quantile and the divergence slope(0.90) - slope(0.10)
     (fan shape > 0), per sex and M - F.
  3. Variance-regression sex x PGS interaction, decile-SD and balanced-bin
     trends, and top-vs-bottom quintile SD ratio (C3 / C3b statistics).

Inference: asymptotic p-values, PGS permuted within sex (C3b scheme), and
family cluster-robust (CR1) p-values for the regression coefficients.
Network metrics are recomputed with C3's own functions; the 20% metrics
are checked against the C3 per-subject table.

Inputs:  as C3, plus results/C3_heteroscedasticity_results.csv
Outputs: results/C3e_pre_specified_tests.csv, results/C3e_quantile_slopes.csv,
         figures/C3e_pre_specified_tests.png,
         reports/C3e_pre_specified_tests_report.txt
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C3_continuous_heteroscedasticity_analysis import (  # noqa: E402
    load_and_prepare_data, calculate_network_metrics, _build_main_design,
    mm2inches, FIGURE_DPI)
from C3b_family_structure_sensitivity import (  # noqa: E402
    sample_statistics, _within_sex_participant_source, _sd_ratio_by_sex,
    METRICS, METRIC_LABELS, SEX_LEVELS)

DENSITIES = (0.15, 0.20, 0.25)
PRIMARY_DENSITY = 0.20
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SEX_COLORS = {'F': '#2a78d6', 'M': '#eb6834'}
COLUMNS = ['density', 'metric', 'test', 'model', 'term', 'estimate', 'statistic',
           'df', 'asymptotic_p', 'perm_within_sex_p', 'cluster_robust_p', 'n', 'B',
           'note']


# %%
# =============================================================================
# 1. TEST STATISTICS (closed form, shared by observed data and permutations)
# =============================================================================

def _resid(X, y):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _lm(X_aux, e2):
    """Koenker LM = n * R^2 of the auxiliary regression of e^2 on X_aux."""
    r = _resid(X_aux, e2)
    tss = np.sum((e2 - e2.mean()) ** 2)
    return len(e2) * (1.0 - (r @ r) / tss)


def _designs(pgs, male):
    one = np.ones(len(pgs))
    base = np.column_stack([one, pgs, male, pgs * male])
    white = np.column_stack([base, pgs ** 2, pgs ** 2 * male])
    return base, white


def het_statistics(y, pgs, male):
    """BP / White LM statistics for every specification, plus the
    sex-modulation F of the White auxiliary. Asymptotic df in DF."""
    out = {}
    one = np.ones(len(y))
    # pooled, sex-free (original pre-specified form)
    Xp = np.column_stack([one, pgs])
    e2 = _resid(Xp, y) ** 2
    out['bp_pooled'] = _lm(Xp, e2)
    out['white_pooled'] = _lm(np.column_stack([Xp, pgs ** 2]), e2)
    # sex x PGS design
    base, white = _designs(pgs, male)
    e2 = _resid(base, y) ** 2
    out['bp_interaction'] = _lm(base, e2)
    out['white_interaction'] = _lm(white, e2)
    # joint F of the sex-modulation terms (pgs:sex, pgs^2:sex) in the White
    # auxiliary: full white vs reduced [1, pgs, sex, pgs^2]
    red = white[:, [0, 1, 2, 4]]
    rss_f = np.sum(_resid(white, e2) ** 2)
    rss_r = np.sum(_resid(red, e2) ** 2)
    out['white_sexmod_F'] = ((rss_r - rss_f) / 2) / (rss_f / (len(y) - white.shape[1]))
    # within each sex
    for label, code in zip(SEX_LEVELS, (0.0, 1.0)):
        m = male == code
        Xs = np.column_stack([np.ones(m.sum()), pgs[m]])
        e2s = _resid(Xs, y[m]) ** 2
        out[f'bp_{label}'] = _lm(Xs, e2s)
        out[f'white_{label}'] = _lm(np.column_stack([Xs, pgs[m] ** 2]), e2s)
    return out


DF = {'bp_pooled': 1, 'white_pooled': 2, 'bp_interaction': 3,
      'white_interaction': 5, 'bp_F': 1, 'bp_M': 1, 'white_F': 2, 'white_M': 2}
HET_META = [  # key, test, model, term
    ('bp_pooled', 'breusch_pagan', 'pooled: metric ~ pgs_z', 'LM'),
    ('white_pooled', 'white', 'pooled: metric ~ pgs_z', 'LM'),
    ('bp_interaction', 'breusch_pagan', 'metric ~ pgs_z * sex', 'LM (omnibus)'),
    ('white_interaction', 'white', 'metric ~ pgs_z * sex', 'LM (omnibus)'),
    ('white_sexmod_F', 'white', 'metric ~ pgs_z * sex',
     'F: pgs_z:sex, pgs_z^2:sex'),
    ('bp_F', 'breusch_pagan', 'within F: metric ~ pgs_z', 'LM'),
    ('bp_M', 'breusch_pagan', 'within M: metric ~ pgs_z', 'LM'),
    ('white_F', 'white', 'within F: metric ~ pgs_z', 'LM'),
    ('white_M', 'white', 'within M: metric ~ pgs_z', 'LM'),
]

TREND_META = [  # sample_statistics key, test, term, sidedness, statistic key
    ('vr_interaction_beta', 'variance_regression', 'pgs_z:sex', 'two', 'vr_interaction_t'),
    ('decile_r_F', 'decile_sd_trend', 'trend r, F', 'one', 'decile_r_F'),
    ('decile_r_M', 'decile_sd_trend', 'trend r, M', 'one', 'decile_r_M'),
    ('decile_diff', 'decile_sd_trend', 'M - F', 'two', 'decile_diff'),
    ('bin_r_F', 'balanced_bin_trend', 'trend r, F', 'one', 'bin_r_F'),
    ('bin_r_M', 'balanced_bin_trend', 'trend r, M', 'one', 'bin_r_M'),
    ('bin_diff', 'balanced_bin_trend', 'M - F', 'two', 'bin_diff'),
]


def _quantile_slopes(y, pgs, quantiles):
    X = np.column_stack([np.ones(len(y)), pgs])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [QuantReg(y, X).fit(q=q, max_iter=5000) for q in quantiles]


def qr_divergence(y, pgs, male):
    """slope(0.90) - slope(0.10) within each sex, and M - F."""
    out = {}
    for label, code in zip(SEX_LEVELS, (0.0, 1.0)):
        m = male == code
        lo, hi = _quantile_slopes(y[m], pgs[m], (QUANTILES[0], QUANTILES[-1]))
        out[f'qr_div_{label}'] = hi.params[1] - lo.params[1]
    out['qr_div_diff'] = out['qr_div_M'] - out['qr_div_F']
    return out


def _perm_p(obs, null, sided):
    null = null[~np.isnan(null)]
    if sided == 'two':
        return (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)), len(null)
    return (1 + np.sum(null >= obs)) / (1 + len(null)), len(null)


# %%
# =============================================================================
# 2. ONE DENSITY x METRIC
# =============================================================================

def analyse(df, metric, density, n_perm, n_perm_qr, n_boot, rng, report):
    y = df[metric].values.astype(float)
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    fam = df['family'].values
    n = len(y)
    rows, qrows = [], []

    def row(**kw):
        r = dict.fromkeys(COLUMNS, np.nan)
        r.update(density=density, metric=metric, n=n, **kw)
        rows.append(r)

    t0 = time.time()
    obs_het = het_statistics(y, pgs, male)
    obs_tr = sample_statistics(y, pgs, male)
    null_het, null_tr = [], []
    for _ in range(n_perm):
        src = _within_sex_participant_source(male, rng)
        null_het.append(het_statistics(y, pgs[src], male))
        null_tr.append(sample_statistics(y, pgs[src], male))
    null_het = pd.DataFrame.from_records(null_het)
    null_tr = pd.DataFrame.from_records(null_tr)

    report.append(f"  {METRIC_LABELS[metric]} ({n_perm} within-sex permutations, "
                  f"{time.time() - t0:.1f} s)")
    # --- BP / White -------------------------------------------------------
    for key, test, model, term in HET_META:
        o = obs_het[key]
        p_perm, B = _perm_p(o, null_het[key].values, 'one')  # large LM / F = evidence
        if key == 'white_sexmod_F':
            df_txt = '2, %d' % (n - 6)
            p_asym = stats.f.sf(o, 2, n - 6)
        else:
            df_txt = str(DF[key])
            p_asym = stats.chi2.sf(o, DF[key])
        note = ('permutation null: PGS permuted within sex' if 'within' in model
                or 'sex' in model else
                'permutation null: PGS permuted within sex (sex not in model)')
        row(test=test, model=model, term=term, estimate=o, statistic=o, df=df_txt,
            asymptotic_p=p_asym, perm_within_sex_p=p_perm, B=B, note=note)
        report.append(f"    {test:<14} {model:<28} {term:<26} = {o:7.3f} (df {df_txt}), "
                      f"asymptotic p = {p_asym:.4f}, permutation p = {p_perm:.4f}")

    # --- variance regression and binned trends ----------------------------
    X = _build_main_design(df)
    e2 = sm.OLS(y, X).fit().resid ** 2
    aux = sm.OLS(e2, X).fit()
    aux_cr = sm.OLS(e2, X).fit(cov_type='cluster', cov_kwds={'groups': fam}, use_t=True)
    for key, test, term, sided, skey in TREND_META:
        p_perm, B = _perm_p(obs_tr[skey], null_tr[skey].values, sided)
        extra = {}
        if key == 'vr_interaction_beta':
            extra = dict(statistic=obs_tr[skey], asymptotic_p=float(aux.pvalues[3]),
                         cluster_robust_p=float(aux_cr.pvalues[3]))
        row(test=test, model='metric ~ pgs_z * sex' if key.startswith('vr') else 'within sex',
            term=term, estimate=obs_tr[key], perm_within_sex_p=p_perm, B=B,
            note=('two-sided' if sided == 'two' else 'one-sided (increasing)'), **extra)
        report.append(f"    {test:<20} {term:<12} = {obs_tr[key]:10.4g}"
                      + (f", asymptotic p = {aux.pvalues[3]:.4f}, CR1 p = "
                         f"{aux_cr.pvalues[3]:.4f}" if extra else "")
                      + f", permutation p = {p_perm:.4f}")

    # --- top vs bottom quintile SD ratio ----------------------------------
    ratio = _sd_ratio_by_sex(y, pgs, male)
    sex_idx = {c: np.flatnonzero(male == c) for c in (0.0, 1.0)}
    boot = {lab: [] for lab in SEX_LEVELS}
    for _ in range(n_boot):
        idx = np.concatenate([sex_idx[c][rng.integers(0, len(sex_idx[c]), len(sex_idx[c]))]
                              for c in (0.0, 1.0)])
        r = _sd_ratio_by_sex(y[idx], pgs[idx], male[idx])
        for lab in SEX_LEVELS:
            boot[lab].append(r[lab]['ratio'])
    for lab in SEX_LEVELS:
        lo, hi = np.percentile(boot[lab], [2.5, 97.5])
        row(test='quintile_sd_ratio', model='within sex', term=f'SD top/bottom, {lab}',
            estimate=ratio[lab]['ratio'], B=n_boot,
            note=f'participant bootstrap within sex, 95% CI [{lo:.3f}, {hi:.3f}]')
        report.append(f"    quintile SD ratio, {lab}: {ratio[lab]['ratio']:.3f} "
                      f"(95% CI {lo:.3f}-{hi:.3f})")

    # --- quantile regression ---------------------------------------------
    t0 = time.time()
    for lab, code in zip(SEX_LEVELS, (0.0, 1.0)):
        m = male == code
        for q, fit in zip(QUANTILES, _quantile_slopes(y[m], pgs[m], QUANTILES)):
            qrows.append({'density': density, 'metric': metric, 'sex': lab, 'quantile': q,
                          'slope': fit.params[1], 'se': fit.bse[1], 'pvalue': fit.pvalues[1],
                          'ci_low': fit.conf_int()[1, 0], 'ci_high': fit.conf_int()[1, 1],
                          'n': int(m.sum())})
    obs_qr = qr_divergence(y, pgs, male)
    null_qr = pd.DataFrame.from_records(
        [qr_divergence(y, pgs[_within_sex_participant_source(male, rng)], male)
         for _ in range(n_perm_qr)])
    slopes = pd.DataFrame(qrows)
    for lab in SEX_LEVELS:
        s = slopes[slopes.sex == lab].set_index('quantile')['slope']
        report.append(f"    quantile slopes, {lab}: " + ", ".join(
            f"tau {q:.2f} = {s[q]:.2e}" for q in QUANTILES))
    for key, term, sided in (('qr_div_F', 'slope(0.90) - slope(0.10), F', 'one'),
                             ('qr_div_M', 'slope(0.90) - slope(0.10), M', 'one'),
                             ('qr_div_diff', 'divergence M - F', 'two')):
        p_perm, B = _perm_p(obs_qr[key], null_qr[key].values, sided)
        row(test='quantile_regression', model='within sex: metric ~ pgs_z', term=term,
            estimate=obs_qr[key], perm_within_sex_p=p_perm, B=B,
            note=('two-sided' if sided == 'two' else 'one-sided (fan shape > 0)'))
        report.append(f"    quantile divergence {term:<30} = {obs_qr[key]:10.3e}, "
                      f"permutation p = {p_perm:.4f}")
    report.append(f"    (quantile regression permutations: {n_perm_qr}, "
                  f"{time.time() - t0:.1f} s)")
    return rows, qrows


# %%
# =============================================================================
# 3. CHECKS, FIGURE, MAIN
# =============================================================================

def self_checks(df, report):
    """Closed-form LM statistics against statsmodels on the observed data."""
    y = df['modularity'].values
    pgs = df['pgs_z'].values
    male = df['male'].values
    ours = het_statistics(y, pgs, male)
    Xp = sm.add_constant(pgs)
    e = sm.OLS(y, Xp).fit().resid
    sm_bp = het_breuschpagan(e, Xp)[0]
    sm_white = het_white(e, Xp)[0]
    base, _ = _designs(pgs, male)
    e_int = sm.OLS(y, base).fit().resid
    sm_bp_int = het_breuschpagan(e_int, base)[0]
    checks = {'BP pooled': (ours['bp_pooled'], sm_bp),
              'White pooled': (ours['white_pooled'], sm_white),
              'BP interaction (= C3 omnibus LM)': (ours['bp_interaction'], sm_bp_int)}
    report.append("Self-checks against statsmodels (modularity, 20%):")
    for name, (a, b) in checks.items():
        ok = np.isclose(a, b, rtol=1e-8)
        report.append(f"  {name}: ours = {a:.6f}, statsmodels = {b:.6f} "
                      f"[{'OK' if ok else 'MISMATCH'}]")
        if not ok:
            raise AssertionError(f"Self-check failed: {name}")


def make_figure(summary, slopes, path):
    fig, axes = plt.subplots(1, 3, figsize=(200 * mm2inches, 65 * mm2inches),
                             dpi=FIGURE_DPI)
    # (a) quantile slopes by sex, modularity, primary density
    ax = axes[0]
    s = slopes[(slopes.metric == 'modularity') & np.isclose(slopes.density, PRIMARY_DENSITY)]
    for lab in SEX_LEVELS:
        d = s[s.sex == lab]
        ax.errorbar(d['quantile'], d['slope'],
                    yerr=[d['slope'] - d['ci_low'], d['ci_high'] - d['slope']],
                    fmt='-o', color=SEX_COLORS[lab], markersize=3, linewidth=1,
                    capsize=2, label='female' if lab == 'F' else 'male')
    ax.axhline(0, color='0.6', linewidth=0.6)
    ax.set_xlabel('quantile of modularity'); ax.set_ylabel('PGS slope (per SD)')
    ax.set_title('Quantile regression, modularity (20%)', fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    # (b) variance-regression interaction across densities
    ax = axes[1]
    for i, metric in enumerate(METRICS):
        d = summary[(summary.metric == metric) & (summary.test == 'variance_regression')]
        ax.plot(d['density'] * 100, d['perm_within_sex_p'], '-o', markersize=3,
                linewidth=1, color=('#333333', '#999999')[i], label=METRIC_LABELS[metric])
    ax.axhline(0.05, color='#c0392b', linewidth=0.6, linestyle='--')
    ax.set_xlabel('network density (%)'); ax.set_ylabel('permutation p')
    ax.set_title('Sex x PGS variance interaction', fontsize=8)
    ax.set_ylim(0, 1); ax.legend(fontsize=6, frameon=False)
    # (c) quintile SD ratio by sex across densities, modularity
    ax = axes[2]
    for lab in SEX_LEVELS:
        d = summary[(summary.metric == 'modularity') & (summary.test == 'quintile_sd_ratio')
                    & (summary.term == f'SD top/bottom, {lab}')]
        ci = d['note'].str.extract(r'\[([-\d.]+), ([-\d.]+)\]').astype(float)
        ax.errorbar(d['density'] * 100 + (-0.4 if lab == 'F' else 0.4), d['estimate'],
                    yerr=[d['estimate'].values - ci[0].values, ci[1].values - d['estimate'].values],
                    fmt='o', color=SEX_COLORS[lab], markersize=3, capsize=2,
                    label='female' if lab == 'F' else 'male')
    ax.axhline(1, color='0.6', linewidth=0.6)
    ax.set_xlabel('network density (%)'); ax.set_ylabel('SD ratio, top / bottom quintile')
    ax.set_title('Modularity dispersion ratio', fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='C3e: pre-specified heteroscedasticity '
                                             'tests and density sensitivity')
    ap.add_argument('--project', required=True)
    ap.add_argument('--pgs', required=True)
    ap.add_argument('--social', required=True)
    ap.add_argument('--behavioural', required=True)
    ap.add_argument('--phenotypic', required=True)
    ap.add_argument('--movement', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--matrices-dir', required=True)
    ap.add_argument('--partition', required=True)
    ap.add_argument('--results', required=True,
                    help='C3 per-subject table, used to verify the 20%% metrics')
    ap.add_argument('--motion-threshold', type=float, default=0.2)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-perm-qr', type=int, default=1000)
    ap.add_argument('--n-bootstrap', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    project = Path(args.project)
    for d in ('results', 'figures', 'reports'):
        (project / d).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(np.random.SeedSequence(args.seed))
    partition = Path(args.partition)
    n_nodes = len(pd.read_csv(partition))

    report = ["=" * 80, "C3e: PRE-SPECIFIED HETEROSCEDASTICITY TESTS AND DENSITY SENSITIVITY",
              "=" * 80, "",
              f"Parcellation: {n_nodes} nodes; densities: "
              + ", ".join(f"{d:.0%}" for d in DENSITIES),
              f"Permutations: {args.n_perm} (BP / White / variance regression / trends), "
              f"{args.n_perm_qr} (quantile regression); bootstrap: {args.n_bootstrap}; "
              f"seed: {args.seed}", ""]
    scratch = []  # C3's loaders write progress lines we do not need here
    merged = load_and_prepare_data(project, Path(args.pgs), Path(args.social),
                                   Path(args.behavioural), Path(args.phenotypic),
                                   Path(args.movement), Path(args.ids),
                                   Path(args.matrices_dir), n_nodes,
                                   args.motion_threshold, scratch)
    pheno = pd.read_csv(args.phenotypic, usecols=['Individual_ID', 'Family_ID'], dtype=str)
    pheno = pheno.rename(columns={'Individual_ID': 'Subject'})

    c3 = pd.read_csv(args.results, dtype={'Subject': str})
    rows, qrows = [], []
    for density in DENSITIES:
        df = calculate_network_metrics(merged, partition, n_nodes, density, scratch)
        df['Subject'] = df['Subject'].astype(str)
        df = df.merge(pheno, on='Subject', how='left', validate='one_to_one')
        if df['Family_ID'].isna().any():
            raise ValueError('Missing Family_ID after merge')
        df['male'] = (df['Gender'] == 'M').astype(float)
        df['family'] = pd.factorize(df['Family_ID'])[0]
        report += ["=" * 60, f"DENSITY {density:.0%} (N = {len(df)}; "
                   f"{int(df.male.sum())} M, {int((1 - df.male).sum())} F)", "=" * 60]
        if np.isclose(density, PRIMARY_DENSITY):
            if not (df['Subject'].values == c3['Subject'].values).all():
                raise AssertionError('Subject order differs from the C3 table')
            dev = max(np.max(np.abs(df[m].values - c3[m].values)) for m in METRICS)
            report.append(f"Recomputed 20% metrics vs C3 table: max |difference| = {dev:.2e}")
            if dev > 1e-10:
                raise AssertionError('20% metrics do not reproduce the C3 table')
            self_checks(df, report)
        for metric in METRICS:
            r, q = analyse(df, metric, density, args.n_perm, args.n_perm_qr,
                           args.n_bootstrap, rng, report)
            rows += r
            qrows += q
        report.append("")

    summary = pd.DataFrame(rows, columns=COLUMNS)
    slopes = pd.DataFrame(qrows)
    summary.to_csv(project / 'results' / 'C3e_pre_specified_tests.csv', index=False,
                   float_format='%.10g')
    slopes.to_csv(project / 'results' / 'C3e_quantile_slopes.csv', index=False,
                  float_format='%.10g')
    make_figure(summary, slopes, project / 'figures' / 'C3e_pre_specified_tests.png')

    # compact overview of the headline modularity tests across densities
    report += ["=" * 60, "SUMMARY: MODULARITY ACROSS DENSITIES", "=" * 60]
    key_tests = [('variance_regression', 'pgs_z:sex'),
                 ('breusch_pagan', 'LM (omnibus)'), ('white', 'LM (omnibus)'),
                 ('white', 'F: pgs_z:sex, pgs_z^2:sex'),
                 ('decile_sd_trend', 'M - F'), ('balanced_bin_trend', 'M - F'),
                 ('quantile_regression', 'divergence M - F')]
    mod = summary[summary.metric == 'modularity']
    for test, term in key_tests:
        cells = []
        for density in DENSITIES:
            r = mod[(mod.test == test) & (mod.term == term)
                    & np.isclose(mod.density, density)].iloc[0]
            cells.append(f"{density:.0%}: p_perm = {r.perm_within_sex_p:.3f}")
        report.append(f"  {test:<20} {term:<26} " + "; ".join(cells))
    for test, model in (('breusch_pagan', 'within M: metric ~ pgs_z'),
                        ('white', 'within M: metric ~ pgs_z'),
                        ('breusch_pagan', 'within F: metric ~ pgs_z'),
                        ('white', 'within F: metric ~ pgs_z')):
        cells = []
        for density in DENSITIES:
            r = mod[(mod.test == test) & (mod.model == model)
                    & np.isclose(mod.density, density)].iloc[0]
            cells.append(f"{density:.0%}: p = {r.asymptotic_p:.3f}")
        report.append(f"  {test:<20} {model:<26} " + "; ".join(cells))
    report += ["", f"Runtime: {time.time() - t_start:.1f} s", "=" * 80, "END OF REPORT",
               "=" * 80]
    (project / 'reports' / 'C3e_pre_specified_tests_report.txt').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
