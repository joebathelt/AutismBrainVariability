#!/usr/bin/env python
"""
C3b: Family-structure sensitivity analysis for the C3 heteroscedasticity tests
==============================================================================

The primary C3 analysis treats the N = 910 HCP participants as independent,
although the sample contains MZ/DZ twins and non-twin siblings (417 families,
sizes 1-5). This companion script re-examines every C3 inference while
respecting the family clustering, using two families of remedies:

  1. Family-clustered inference
       - cluster-robust (CR1) standard errors with Family_ID as the cluster
       - family cluster bootstrap (families resampled with replacement)
       - family-block permutation (whole families exchanged; Winkler et al.
         2015 exchangeability blocks), including a Freedman-Lane permutation
         for the headline sex x PGS variance interaction
  2. Family as a random effect
       - linear mixed models with a family random intercept (statsmodels
         MixedLM), for the mean-level association model and for the two-stage
         variance regression

The primary C3 outputs are left untouched; this script reads
results/C3_heteroscedasticity_results.csv, re-attaches Family_ID from the
phenotypic table and writes its own C3b_family_* outputs.

Inputs
------
  results/C3_heteroscedasticity_results.csv   per-subject metrics (from C3)
  data/raw_anonymised/phenotypic_data_anonymised.csv   Family_ID, zygosity
  results/C3_variance_regression_sex.csv      primary numbers (startup guard)
  results/C3_pgs_metric_association.csv       primary numbers (startup guard)

Outputs
-------
  results/C3b_family_sensitivity_summary.csv
  results/C3b_family_icc.csv
  reports/C3b_family_structure_sensitivity_report.txt
  figures/C3b_family_sensitivity_forest.png

Usage
-----
  python C3b_family_structure_sensitivity.py --project /path/to/project \
      --results results/C3_heteroscedasticity_results.csv \
      --phenotypic data/raw_anonymised/phenotypic_data_anonymised.csv \
      --n-bootstrap 2000 --n-perm 5000 --seed 42 [--self-check]
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
import statsmodels.formula.api as smf

# Import the primary C3 model definitions so the primary numbers are
# reproduced by the very same code, not re-implemented.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from C3_continuous_heteroscedasticity_analysis import (  # noqa: E402
    _build_main_design, _MAIN_COEF_NAMES, _ASSOC_COEF_NAMES,
    test_variance_regression_by_sex, test_pgs_graph_metric_association,
    mm2inches, FIGURE_DPI,
)

METRICS = ('modularity', 'global_efficiency')
METRIC_LABELS = {'modularity': 'Modularity',
                 'global_efficiency': 'Global efficiency'}
SEX_LEVELS = ('F', 'M')          # sorted, as in C3 (difference = M - F)
N_DECILES = 10
N_BINS = 5
SUMMARY_COLUMNS = ['metric', 'test', 'method', 'term', 'estimate', 'se',
                   'ci_low', 'ci_high', 'pvalue', 'statistic', 'df', 'n',
                   'n_families', 'B', 'note']


# %%
# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_family_data(results_csv, phenotypic_csv, report):
    """Join Family_ID / zygosity onto the C3 per-subject results.

    Row order of the C3 results is preserved so the primary C3 numbers are
    reproduced exactly by the imported C3 functions.
    """
    res = pd.read_csv(results_csv)
    pheno = pd.read_csv(phenotypic_csv,
                        usecols=['Individual_ID', 'Family_ID', 'ZygosityGT',
                                 'ZygositySR'],
                        dtype=str)
    pheno = pheno.rename(columns={'Individual_ID': 'Subject'})
    for col in ('ZygosityGT', 'ZygositySR'):
        pheno[col] = pheno[col].fillna('').str.strip().replace('', 'NA')

    if res['Subject'].duplicated().any():
        raise ValueError("C3 results contain duplicated Subject IDs")
    if not res['Subject'].astype(str).isin(pheno['Subject']).all():
        missing = (~res['Subject'].astype(str).isin(pheno['Subject'])).sum()
        raise ValueError(
            f"{missing} C3 Subject values are not phenotypic Individual_IDs. "
            "C3 must write real participant IDs (rerun C3 after the C3b fix).")

    res['Subject'] = res['Subject'].astype(str)
    df = res.merge(pheno, on='Subject', how='left', validate='one_to_one')
    if df['Family_ID'].isna().any():
        raise ValueError("Missing Family_ID after merge")

    # Sex dummy: C3 uses pd.get_dummies(drop_first=True), which keeps the
    # second sorted level ('M'); replicate that coding explicitly.
    if sorted(df['Gender'].unique()) != list(SEX_LEVELS):
        raise ValueError(f"Expected sexes {SEX_LEVELS}, got "
                         f"{sorted(df['Gender'].unique())}")
    df['male'] = (df['Gender'] == 'M').astype(float)
    df['family'] = pd.factorize(df['Family_ID'])[0]
    df['zygosity'] = df['ZygosityGT']

    sizes = df.groupby('family').size()
    size_counts = sizes.value_counts().sort_index()
    mixed_sex = (df.groupby('family')['male'].nunique() > 1).sum()

    report.append(f"Loaded {len(df)} participants from {results_csv}")
    report.append(f"Families (Family_ID): {sizes.size}")
    report.append("Family sizes: " + ", ".join(
        f"{int(k)}: {int(v)}" for k, v in size_counts.items()))
    report.append(f"Mixed-sex families: {int(mixed_sex)}")
    report.append("Genotyped zygosity: " + ", ".join(
        f"{k}: {v}" for k, v in df['zygosity'].value_counts().items()))
    report.append(f"Mean family size (m-bar = sum n_g^2 / N): "
                  f"{(sizes ** 2).sum() / len(df):.3f}")
    return df


def family_index(family_codes, order_key=None):
    """Member row indices per family (list of int arrays).

    Members within a family are listed in `order_key` order (Subject ID),
    which gives the canonical within-family position used by the
    whole-family permutation scheme.
    """
    family_codes = np.asarray(family_codes)
    n_fam = family_codes.max() + 1
    if order_key is None:
        order = np.arange(len(family_codes))
    else:
        order = np.lexsort((np.asarray(order_key), family_codes))
    fam_sorted = family_codes[order]
    boundaries = np.flatnonzero(np.diff(fam_sorted)) + 1
    groups = np.split(order, boundaries)
    assert len(groups) == n_fam
    return groups


# %%
# =============================================================================
# 2. DESCRIPTIVES: HOW MUCH NON-INDEPENDENCE IS THERE?
# =============================================================================

def icc_oneway(y, groups):
    """One-way random-effects ICC(1) for unbalanced groups (ANOVA estimator)
    plus the variance-components ICC from a random-intercept null model.

    Returns a dict with the ICC estimates, the design effect
    1 + (m_bar - 1) * ICC with m_bar = sum n_g^2 / N, and the effective N.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n = len(y)
    g_sizes = np.bincount(groups)
    g_sum = np.bincount(groups, weights=y)
    g_mean = g_sum / g_sizes
    grand = y.mean()
    n_groups = len(g_sizes)
    ss_between = float((g_sizes * (g_mean - grand) ** 2).sum())
    ss_within = float(((y - g_mean[groups]) ** 2).sum())
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n - n_groups)
    n0 = (n - (g_sizes ** 2).sum() / n) / (n_groups - 1)
    icc_anova = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)

    # ICC is scale-invariant; standardise so the optimiser sees O(1) values
    # (squared residuals are ~1e-3 and make the profile likelihood singular).
    data = pd.DataFrame({'y': (y - y.mean()) / y.std(ddof=1), 'g': groups})
    fit = _fit_mixedlm('y ~ 1', data, data['g'])
    icc_lmm = fit['icc']

    m_bar = float((g_sizes ** 2).sum() / n)
    icc_for_deff = max(icc_anova, 0.0)
    deff = 1.0 + (m_bar - 1.0) * icc_for_deff
    return {
        'icc_anova': float(icc_anova),
        'icc_mixedlm': float(icc_lmm),
        'mixedlm_converged': fit['converged'],
        'm_bar': m_bar,
        'design_effect': float(deff),
        'n_effective': float(n / deff),
        'n': int(n),
        'n_groups': int(n_groups),
    }


def twin_pair_correlation(df, col, zygosity_label):
    """Double-entry Pearson correlation across complete twin pairs."""
    sub = df[df['zygosity'] == zygosity_label].sort_values('Subject')
    pairs = []
    for _, fam in sub.groupby('Family_ID'):
        if len(fam) >= 2:
            vals = fam[col].values[:2]
            pairs.append(vals)
    if len(pairs) < 3:
        return np.nan, len(pairs)
    pairs = np.asarray(pairs, dtype=float)
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    r = float(np.corrcoef(a, b)[0, 1])
    return r, len(pairs)


def compute_descriptives(df, report):
    """ICC / design effect / twin correlations for the analysis variables."""
    report.append("")
    report.append("=" * 60)
    report.append("FAMILY CLUSTERING OF THE ANALYSIS VARIABLES")
    report.append("=" * 60)
    report.append("  ICC(1): one-way ANOVA estimator and random-intercept LMM;")
    report.append("  design effect = 1 + (m_bar - 1) * ICC; N_eff = N / DEFF")

    X = _build_main_design(df)
    variables = {'pgs_z': df['pgs_z'].values,
                 'Social_Score': df['Social_Score'].values}
    for metric in METRICS:
        variables[metric] = df[metric].values
        mean_model = sm.OLS(df[metric].values, X).fit()
        variables[f'e2_{metric}'] = mean_model.resid ** 2

    rows = []
    for name, values in variables.items():
        icc = icc_oneway(values, df['family'].values)
        tmp = df.assign(_v=values)
        r_mz, n_mz = twin_pair_correlation(tmp, '_v', 'MZ')
        r_dz, n_dz = twin_pair_correlation(tmp, '_v', 'DZ')
        row = {'variable': name, **icc, 'r_MZ': r_mz, 'n_MZ_pairs': n_mz,
               'r_DZ': r_dz, 'n_DZ_pairs': n_dz}
        rows.append(row)
        report.append(f"  {name:<22} ICC(ANOVA) = {icc['icc_anova']:6.3f}, "
                      f"ICC(LMM) = {icc['icc_mixedlm']:6.3f}, "
                      f"DEFF = {icc['design_effect']:5.3f}, "
                      f"N_eff = {icc['n_effective']:6.1f}, "
                      f"r_MZ = {r_mz:6.3f} ({n_mz} pairs), "
                      f"r_DZ = {r_dz:6.3f} ({n_dz} pairs)")
    return pd.DataFrame(rows)


# %%
# =============================================================================
# 3. CLUSTER-ROBUST AND MIXED-MODEL INFERENCE
# =============================================================================

def _ols_rows(res, names, metric, test, method, n_families, note=''):
    """Summary rows for every coefficient of a fitted OLS results object."""
    ci = np.asarray(res.conf_int())
    df_inf = getattr(res, 'df_resid_inference', None)
    if df_inf is None:
        df_inf = res.df_resid
    rows = []
    for i, name in enumerate(names):
        rows.append({
            'metric': metric, 'test': test, 'method': method, 'term': name,
            'estimate': float(res.params[i]), 'se': float(res.bse[i]),
            'ci_low': float(ci[i, 0]), 'ci_high': float(ci[i, 1]),
            'pvalue': float(res.pvalues[i]), 'statistic': float(res.tvalues[i]),
            'df': float(df_inf), 'n': int(res.nobs),
            'n_families': int(n_families), 'B': np.nan, 'note': note,
        })
    return rows


def fit_cluster_robust(df, report):
    """Primary OLS, HC1 and family cluster-robust (CR1) fits of the C3 models.

    Association:  pgs_z ~ metric + male + metric:male
    Mean model:   metric ~ pgs_z + male + pgs_z:male
    Variance:     resid^2 ~ pgs_z + male + pgs_z:male
    """
    report.append("")
    report.append("=" * 60)
    report.append("CLUSTER-ROBUST STANDARD ERRORS (cluster = Family_ID)")
    report.append("=" * 60)
    report.append("  CR1 small-sample correction G/(G-1)*(N-1)/(N-k); "
                  "t reference distribution with G-1 df")

    fam = df['family'].values
    n_families = fam.max() + 1
    X_main = _build_main_design(df)
    male = df['male'].values
    fits = {
        'primary_ols': dict(),
        'hc1': dict(cov_type='HC1'),
        'cluster_robust': dict(cov_type='cluster',
                               cov_kwds={'groups': fam}, use_t=True),
    }
    rows = []
    for metric in METRICS:
        m = df[metric].values.astype(float)
        X_assoc = sm.add_constant(np.column_stack([m, male, m * male]))
        y_pgs = df['pgs_z'].values.astype(float)
        e2 = sm.OLS(m, X_main).fit().resid ** 2
        for method, kw in fits.items():
            assoc = sm.OLS(y_pgs, X_assoc).fit(**kw)
            mean_fit = sm.OLS(m, X_main).fit(**kw)
            aux = sm.OLS(e2, X_main).fit(**kw)
            rows += _ols_rows(assoc, _ASSOC_COEF_NAMES, metric, 'association',
                              method, n_families)
            rows += _ols_rows(mean_fit, _MAIN_COEF_NAMES, metric, 'mean_model',
                              method, n_families)
            rows += _ols_rows(aux, _MAIN_COEF_NAMES, metric,
                              'variance_regression', method, n_families)
            report.append(
                f"  {metric:<18} {method:<15} "
                f"assoc metric p = {assoc.pvalues[1]:.4f}, "
                f"metric:Gender p = {assoc.pvalues[3]:.4f} | "
                f"variance pgs_z:Gender beta = {aux.params[3]:.4e}, "
                f"SE = {aux.bse[3]:.4e}, p = {aux.pvalues[3]:.4f}")
    return rows


def _fit_mixedlm(formula, data, groups, maxiter=2000):
    """Fit a random-intercept LMM with optimiser fallback and diagnostics."""
    # Optimisers are tried in turn: some raise (e.g. lbfgs hitting a
    # singular profile-likelihood Hessian) rather than reporting
    # non-convergence, so statsmodels' own list fallback is not enough.
    res, used, errors = None, None, []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = smf.mixedlm(formula, data, groups=groups)
        for method in ('bfgs', 'lbfgs', 'cg', 'powell'):
            try:
                cand = model.fit(reml=True, method=method, maxiter=maxiter)
            except (np.linalg.LinAlgError, ValueError) as err:
                errors.append(f"{method}: {type(err).__name__}")
                continue
            res, used = cand, method
            if cand.converged:
                break
    if res is None:
        raise RuntimeError(f"MixedLM failed for '{formula}': {errors}")
    var_re = float(res.cov_re.iloc[0, 0])
    scale = float(res.scale)
    boundary = var_re < 1e-8 * scale
    warn_msgs = sorted({str(w.message).splitlines()[0] for w in caught})
    return {
        'result': res,
        'converged': bool(res.converged),
        'boundary': bool(boundary),
        'var_family': var_re,
        'var_residual': scale,
        'icc': var_re / (var_re + scale) if (var_re + scale) > 0 else np.nan,
        'warnings': warn_msgs,
        'optimizer': used,
        'optimizer_errors': errors,
    }


def _mixed_rows(fit, names, scale_factors, metric, test, n_families, note):
    """Summary rows from a MixedLM fit, back-scaled to the original units."""
    res = fit['result']
    rows = []
    fe = res.fe_params
    bse = res.bse_fe
    for name, sf in zip(names, scale_factors):
        est = float(fe[name]) * sf
        se = float(bse[name]) * sf
        rows.append({
            'metric': metric, 'test': test, 'method': 'mixedlm', 'term': name,
            'estimate': est, 'se': se,
            'ci_low': est - stats.norm.ppf(0.975) * se,
            'ci_high': est + stats.norm.ppf(0.975) * se,
            'pvalue': float(res.pvalues[name]),
            'statistic': float(res.tvalues[name]), 'df': np.nan,
            'n': int(res.nobs), 'n_families': int(n_families), 'B': np.nan,
            'note': note,
        })
    return rows


def fit_mixed_models(df, report):
    """Family random-intercept LMMs for the association and variance models.

    Outcomes and metric predictors are scaled to unit SD for the optimiser
    and the coefficients are back-transformed, so estimates are directly
    comparable with the OLS rows. Scaling only (no centring) keeps the
    intercept and sex terms interpretable in C3's parameterisation.
    """
    report.append("")
    report.append("=" * 60)
    report.append("FAMILY RANDOM-INTERCEPT MIXED MODELS (statsmodels MixedLM)")
    report.append("=" * 60)
    report.append("  REML; Wald z tests; family variance reported as ICC")

    fam = df['family'].values
    n_families = fam.max() + 1
    data = pd.DataFrame({'pgs_z': df['pgs_z'].values, 'male': df['male'].values,
                         'family': fam})
    rows = []
    for metric in METRICS:
        m = df[metric].values.astype(float)
        sd_m = m.std(ddof=1)
        data['metric_s'] = m / sd_m

        # --- Association: pgs_z ~ metric + male + metric:male + (1 | family)
        fit = _fit_mixedlm('pgs_z ~ metric_s + male + metric_s:male', data,
                           data['family'])
        note = (f"converged={fit['converged']}; boundary={fit['boundary']}; "
                f"ICC_family={fit['icc']:.4f}")
        rows += _mixed_rows(fit,
                            ['Intercept', 'metric_s', 'male', 'metric_s:male'],
                            [1.0, 1.0 / sd_m, 1.0, 1.0 / sd_m],
                            metric, 'association', n_families, note)
        for r in rows[-4:]:
            r['term'] = dict(zip(['Intercept', 'metric_s', 'male',
                                  'metric_s:male'], _ASSOC_COEF_NAMES))[r['term']]
        res = fit['result']
        report.append(f"  {metric:<18} association: metric beta = "
                      f"{res.fe_params['metric_s'] / sd_m:.4e}, "
                      f"p = {res.pvalues['metric_s']:.4f}; metric:Gender p = "
                      f"{res.pvalues['metric_s:male']:.4f}; {note}")

        # --- Stage 1: metric ~ pgs_z + male + pgs_z:male + (1 | family)
        fit1 = _fit_mixedlm('metric_s ~ pgs_z + male + pgs_z:male', data,
                            data['family'])
        note1 = (f"converged={fit1['converged']}; boundary={fit1['boundary']}; "
                 f"ICC_family={fit1['icc']:.4f}")
        rows += _mixed_rows(fit1, ['Intercept', 'pgs_z', 'male', 'pgs_z:male'],
                            [sd_m] * 4, metric, 'mean_model', n_families, note1)
        for r in rows[-4:]:
            r['term'] = dict(zip(['Intercept', 'pgs_z', 'male', 'pgs_z:male'],
                                 _MAIN_COEF_NAMES))[r['term']]
        res1 = fit1['result']
        # Marginal residuals (fixed effects only), the analogue of C3's OLS
        # residuals: the dispersion of the metric around its population mean
        # given PGS and sex. statsmodels' `resid` is conditional on the
        # predicted family effects and would remove the between-family part
        # of that dispersion, so it is deliberately not used here; the
        # family clustering of e^2 is handled by the stage-2 random intercept.
        X1 = res1.model.exog
        e_marg = data['metric_s'].values - X1 @ res1.fe_params.values
        data['e2_s'] = e_marg ** 2
        report.append(f"  {metric:<18} mean model: pgs_z:Gender p = "
                      f"{res1.pvalues['pgs_z:male']:.4f}; {note1}")

        # --- Stage 2: e^2 ~ pgs_z + male + pgs_z:male + (1 | family)
        fit2 = _fit_mixedlm('e2_s ~ pgs_z + male + pgs_z:male', data,
                            data['family'])
        note2 = (f"converged={fit2['converged']}; boundary={fit2['boundary']}; "
                 f"ICC_family={fit2['icc']:.4f}; two-stage: marginal residuals "
                 f"from the stage-1 LMM (fixed effects only)")
        rows += _mixed_rows(fit2, ['Intercept', 'pgs_z', 'male', 'pgs_z:male'],
                            [sd_m ** 2] * 4, metric, 'variance_regression',
                            n_families, note2)
        for r in rows[-4:]:
            r['term'] = dict(zip(['Intercept', 'pgs_z', 'male', 'pgs_z:male'],
                                 _MAIN_COEF_NAMES))[r['term']]
        res2 = fit2['result']
        report.append(f"  {metric:<18} variance model: pgs_z:Gender beta = "
                      f"{res2.fe_params['pgs_z:male'] * sd_m ** 2:.4e}, "
                      f"p = {res2.pvalues['pgs_z:male']:.4f}; {note2}")
        for f_ in (fit, fit1, fit2):
            for w in f_['warnings']:
                report.append(f"    warning: {w}")
    return rows


# %%
# =============================================================================
# 4. RESAMPLING STATISTICS (shared by observed, bootstrap and permutation)
# =============================================================================

def _quantile_bins(values, q):
    """Equal-count quantile bins, as in C3 (pd.qcut on values).

    Value-based qcut reproduces C3 on the observed data. In resamples that
    duplicate whole families, tied values can produce duplicate bin edges;
    fall back to rank-based binning in that case.
    """
    s = pd.Series(values)
    try:
        return pd.qcut(s, q=q, labels=False).values
    except ValueError:
        return pd.qcut(s.rank(method='first'), q=q, labels=False).values


def _trend(centers, dispersions):
    centers = np.asarray(centers, dtype=float)
    dispersions = np.asarray(dispersions, dtype=float)
    if np.isnan(dispersions).any() or dispersions.std() == 0:
        return np.nan
    return float(np.corrcoef(centers, dispersions)[0, 1])


def _binned_dispersion_trend(pgs, y, q, use_var):
    bins = _quantile_bins(pgs, q)
    centers, disp = [], []
    for b in range(q):
        mask = bins == b
        vals = y[mask]
        centers.append(pgs[mask].mean())
        if len(vals) < 2:
            disp.append(np.nan)
        else:
            disp.append(vals.var(ddof=1) if use_var else vals.std(ddof=1))
    return _trend(centers, disp)


def sample_statistics(y, pgs, male, n_deciles=N_DECILES, n_bins=N_BINS):
    """Deterministic statistic set for one dataset (observed or resampled).

    Returns
    -------
    dict with
      vr_pgs_beta, vr_pgs_t, vr_interaction_beta, vr_interaction_t :
          auxiliary (resid^2) regression coefficients and classical t
      assoc_metric_beta/_t, assoc_interaction_beta/_t :
          mean-level association model pgs_z ~ metric + male + metric:male
      decile_r_F, decile_r_M, decile_diff : decile-SD trend per sex, M - F
      bin_r_F, bin_r_M, bin_diff : quantile-bin variance trend per sex, M - F
    """
    y = np.asarray(y, dtype=float)
    pgs = np.asarray(pgs, dtype=float)
    male = np.asarray(male, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), pgs, male, pgs * male])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e2 = (y - X @ beta) ** 2
    XtX_inv = np.linalg.inv(X.T @ X)
    b2 = XtX_inv @ X.T @ e2
    resid2 = e2 - X @ b2
    sigma2 = resid2 @ resid2 / (n - X.shape[1])
    se2 = np.sqrt(sigma2 * np.diag(XtX_inv))

    # Mean-level association model: pgs_z ~ metric + male + metric:male
    Xa = np.column_stack([np.ones(n), y, male, y * male])
    XtXa_inv = np.linalg.inv(Xa.T @ Xa)
    ba = XtXa_inv @ Xa.T @ pgs
    ra = pgs - Xa @ ba
    s2a = ra @ ra / (n - Xa.shape[1])
    sea = np.sqrt(s2a * np.diag(XtXa_inv))

    out = {'vr_pgs_beta': float(b2[1]),
           'vr_pgs_t': float(b2[1] / se2[1]),
           'vr_interaction_beta': float(b2[3]),
           'vr_interaction_t': float(b2[3] / se2[3]),
           'assoc_metric_beta': float(ba[1]),
           'assoc_metric_t': float(ba[1] / sea[1]),
           'assoc_interaction_beta': float(ba[3]),
           'assoc_interaction_t': float(ba[3] / sea[3])}
    for label, code in zip(SEX_LEVELS, (0.0, 1.0)):
        mask = male == code
        out[f'decile_r_{label}'] = _binned_dispersion_trend(
            pgs[mask], y[mask], n_deciles, use_var=False)
        out[f'bin_r_{label}'] = _binned_dispersion_trend(
            pgs[mask], y[mask], n_bins, use_var=True)
    out['decile_diff'] = out['decile_r_M'] - out['decile_r_F']
    out['bin_diff'] = out['bin_r_M'] - out['bin_r_F']
    return out


STAT_META = {
    # key: (test, term, sidedness)
    'vr_interaction_beta': ('variance_regression', 'pgs_z:Gender', 'two'),
    'vr_pgs_beta': ('variance_regression', 'pgs_z', 'two'),
    'decile_r_F': ('decile_trend', 'trend_r_F', 'one'),
    'decile_r_M': ('decile_trend', 'trend_r_M', 'one'),
    'decile_diff': ('decile_trend', 'M - F', 'two'),
    'bin_r_F': ('balanced_bin_trend', 'trend_r_F', 'one'),
    'bin_r_M': ('balanced_bin_trend', 'trend_r_M', 'one'),
    'bin_diff': ('balanced_bin_trend', 'M - F', 'two'),
}


# Permutation tests use a pivotal statistic (classical t) for regression
# coefficients and the trend statistics themselves for the binned tests.
# key of the reported estimate: (test, term, sidedness, key of the statistic)
PERM_META = {
    'vr_interaction_beta': ('variance_regression', 'pgs_z:Gender', 'two',
                            'vr_interaction_t'),
    'vr_pgs_beta': ('variance_regression', 'pgs_z', 'two', 'vr_pgs_t'),
    'assoc_metric_beta': ('association', 'metric', 'two', 'assoc_metric_t'),
    'assoc_interaction_beta': ('association', 'metric:Gender', 'two',
                               'assoc_interaction_t'),
    'decile_r_F': ('decile_trend', 'trend_r_F', 'one', 'decile_r_F'),
    'decile_r_M': ('decile_trend', 'trend_r_M', 'one', 'decile_r_M'),
    'decile_diff': ('decile_trend', 'M - F', 'two', 'decile_diff'),
    'bin_r_F': ('balanced_bin_trend', 'trend_r_F', 'one', 'bin_r_F'),
    'bin_r_M': ('balanced_bin_trend', 'trend_r_M', 'one', 'bin_r_M'),
    'bin_diff': ('balanced_bin_trend', 'M - F', 'two', 'bin_diff'),
}


def _stats_frame(records):
    return pd.DataFrame.from_records(records)


def _perm_rows(observed, null, metric, method, note, n, n_fam, report):
    """Summary rows for one permutation scheme, all PERM_META statistics."""
    rows = []
    for key, (test, term, sided, stat_key) in PERM_META.items():
        obs_stat = observed[stat_key]
        nd = null[stat_key].values
        nd = nd[~np.isnan(nd)]
        if sided == 'two':
            p = (1 + np.sum(np.abs(nd) >= abs(obs_stat))) / (1 + len(nd))
            side = 'two-sided'
        else:
            p = (1 + np.sum(nd >= obs_stat)) / (1 + len(nd))
            side = 'one-sided (increasing)'
        rows.append({
            'metric': metric, 'test': test, 'method': method, 'term': term,
            'estimate': observed[key], 'se': np.nan,
            'ci_low': float(np.percentile(nd, 2.5)),
            'ci_high': float(np.percentile(nd, 97.5)),
            'pvalue': float(p), 'statistic': float(obs_stat), 'df': np.nan,
            'n': int(n), 'n_families': int(n_fam), 'B': int(len(nd)),
            'note': f"{side}; statistic = {stat_key}; {note}; "
                    "ci = null 2.5-97.5 percentiles of the statistic",
        })
        report.append(f"    {test:<20} {term:<12} obs = {observed[key]:9.4e} "
                      f"(stat {obs_stat:7.3f}), perm p = {p:.4f}")
    return rows


# %%
# =============================================================================
# 5. FAMILY CLUSTER BOOTSTRAP
# =============================================================================

def family_bootstrap(df, metric, fam_idx, n_boot, rng, report):
    """Resample whole families with replacement and recompute all statistics.

    p-values follow C3's construction (2 * min tail mass of the bootstrap
    distribution around the observed value; one-tailed 'increasing' mass for
    per-sex trends). They are CI-inversion p-values, not null-referenced.
    """
    y = df[metric].values.astype(float)
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    n_fam = len(fam_idx)
    observed = sample_statistics(y, pgs, male)

    t0 = time.time()
    records = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_fam, n_fam)
        idx = np.concatenate([fam_idx[g] for g in draw])
        records.append(sample_statistics(y[idx], pgs[idx], male[idx]))
    boot = _stats_frame(records)
    elapsed = time.time() - t0

    rows = []
    report.append(f"  {metric}: {n_boot} family-bootstrap replicates "
                  f"({elapsed:.1f} s)")
    for key, (test, term, sided) in STAT_META.items():
        b = boot[key].values
        b = b[~np.isnan(b)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        if sided == 'two':
            p = 2.0 * min(np.mean(b >= 0), np.mean(b <= 0))
            note = 'CI-inversion p (C3 construction): 2*min(tail mass)'
        else:
            p = float(np.mean(b <= 0))
            note = 'one-tailed p (increasing trend): bootstrap mass <= 0'
        rows.append({
            'metric': metric, 'test': test, 'method': 'family_bootstrap',
            'term': term, 'estimate': observed[key], 'se': float(b.std(ddof=1)),
            'ci_low': float(lo), 'ci_high': float(hi), 'pvalue': float(p),
            'statistic': np.nan, 'df': np.nan, 'n': int(len(y)),
            'n_families': int(n_fam), 'B': int(len(b)), 'note': note,
        })
        report.append(f"    {test:<20} {term:<12} obs = {observed[key]:9.4e}, "
                      f"95% CI [{lo:9.4e}, {hi:9.4e}], p = {p:.4f}")
    return rows, observed


def _sd_ratio_by_sex(y, pgs, male, n_bins=N_BINS):
    """Ratio of the metric SD in the top vs bottom within-sex PGS quantile
    bin (quintiles by default, as in the PGS-balanced analysis)."""
    out = {}
    for label, code in zip(SEX_LEVELS, (0.0, 1.0)):
        m = male == code
        bins = _quantile_bins(pgs[m], n_bins)
        ys = y[m]
        top = ys[bins == n_bins - 1]
        bot = ys[bins == 0]
        sd_top = top.std(ddof=1) if len(top) > 1 else np.nan
        sd_bot = bot.std(ddof=1) if len(bot) > 1 else np.nan
        out[label] = {'ratio': sd_top / sd_bot, 'sd_top': sd_top,
                      'sd_bottom': sd_bot, 'n_top': len(top), 'n_bottom': len(bot)}
    return out


def quintile_sd_ratio(df, metric, fam_idx, n_boot, rng, report):
    """Top-vs-bottom within-sex PGS-quintile SD ratio with bootstrap CIs.

    Two bootstrap schemes: participants resampled within sex (the sampling
    model of the primary analysis) and whole families resampled (family
    cluster bootstrap). Percentile 95% intervals; the p-value is the
    CI-inversion p for ratio != 1.
    """
    y = df[metric].values.astype(float)
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    n = len(y)
    n_fam = len(fam_idx)
    observed = _sd_ratio_by_sex(y, pgs, male)
    sex_idx = {code: np.flatnonzero(male == code) for code in (0.0, 1.0)}

    t0 = time.time()
    boots = {'participant_bootstrap': {lab: [] for lab in SEX_LEVELS},
             'family_bootstrap': {lab: [] for lab in SEX_LEVELS}}
    for _ in range(n_boot):
        idx = np.concatenate([sex_idx[c][rng.integers(0, len(sex_idx[c]),
                                                       len(sex_idx[c]))]
                              for c in (0.0, 1.0)])
        r = _sd_ratio_by_sex(y[idx], pgs[idx], male[idx])
        for lab in SEX_LEVELS:
            boots['participant_bootstrap'][lab].append(r[lab]['ratio'])
        draw = rng.integers(0, n_fam, n_fam)
        idx = np.concatenate([fam_idx[g] for g in draw])
        r = _sd_ratio_by_sex(y[idx], pgs[idx], male[idx])
        for lab in SEX_LEVELS:
            boots['family_bootstrap'][lab].append(r[lab]['ratio'])
    elapsed = time.time() - t0

    rows, wide = [], []
    report.append(f"  {metric}: top vs bottom within-sex PGS-quintile SD ratio "
                  f"({n_boot} bootstrap replicates per scheme, {elapsed:.1f} s)")
    for lab in SEX_LEVELS:
        obs = observed[lab]
        line = (f"    {lab}: SD top = {obs['sd_top']:.5f} (n = {obs['n_top']}), "
                f"SD bottom = {obs['sd_bottom']:.5f} (n = {obs['n_bottom']}), "
                f"ratio = {obs['ratio']:.3f}")
        rec = {'metric': metric, 'sex': lab, 'ratio': obs['ratio'],
               'sd_top': obs['sd_top'], 'sd_bottom': obs['sd_bottom'],
               'n_top': obs['n_top'], 'n_bottom': obs['n_bottom']}
        for method in ('participant_bootstrap', 'family_bootstrap'):
            b = np.asarray(boots[method][lab])
            b = b[np.isfinite(b)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            pval = 2.0 * min(np.mean(b >= 1.0), np.mean(b <= 1.0))
            rows.append({
                'metric': metric, 'test': 'quintile_sd_ratio', 'method': method,
                'term': lab, 'estimate': obs['ratio'], 'se': float(b.std(ddof=1)),
                'ci_low': float(lo), 'ci_high': float(hi), 'pvalue': float(pval),
                'statistic': np.nan, 'df': np.nan, 'n': int(n),
                'n_families': int(n_fam), 'B': int(len(b)),
                'note': 'SD(top quintile) / SD(bottom quintile) of the '
                        'covariate-residualised metric, within-sex PGS '
                        'quintiles; percentile CI; CI-inversion p for ratio != 1',
            })
            tag = 'participant' if method == 'participant_bootstrap' else 'family'
            rec[f'ci_low_{tag}'] = float(lo)
            rec[f'ci_high_{tag}'] = float(hi)
            rec[f'p_{tag}'] = float(pval)
            line += f"; {tag} bootstrap 95% CI [{lo:.3f}, {hi:.3f}] (p = {pval:.3f})"
        report.append(line)
        wide.append(rec)
    return rows, wide


# %%
# =============================================================================
# 6. FAMILY-BLOCK PERMUTATION
# =============================================================================

def _block_permutation_source(blocks, rng):
    """Row-index map for one exchange of whole families within blocks.

    `blocks` is a list of exchangeability blocks; each block is a list of
    member-index arrays of identical length (and, for sex-preserving
    blocks, identical within-family sex pattern). Within a block the family
    order is permuted and members are mapped position-wise. Blocks with a
    single family stay fixed. Returns `source` such that
    permuted[i] = original[source[i]].
    """
    n = sum(len(f) for block in blocks for f in block)
    source = np.arange(n)
    for fams in blocks:
        if len(fams) < 2:
            continue
        perm = rng.permutation(len(fams))
        for j, fam in enumerate(fams):
            source[fam] = fams[perm[j]]
    return source


def family_size_blocks(fam_idx):
    """Exchangeability blocks = families of equal size (sex ignored)."""
    by_size = {}
    for f in fam_idx:
        by_size.setdefault(len(f), []).append(f)
    return [by_size[k] for k in sorted(by_size)]


def _within_sex_participant_source(male, rng):
    """Row-index map for a participant-level permutation of the PGS within
    each sex (family structure ignored)."""
    source = np.arange(len(male))
    for code in (0.0, 1.0):
        idx = np.flatnonzero(male == code)
        source[idx] = idx[rng.permutation(len(idx))]
    return source


def _whole_family_permutation(fam_idx, rng):
    """Whole-family exchange of a per-subject column within size classes."""
    return _block_permutation_source(family_size_blocks(fam_idx), rng)


def _describe_blocks(blocks, label, report):
    sizes = sorted(len(b) for b in blocks)
    fixed = sum(1 for b in blocks if len(b) < 2)
    report.append(f"  {label}: {len(blocks)} blocks "
                  f"(families per block: min {sizes[0]}, max {sizes[-1]}); "
                  f"{fixed} single-family block(s) kept fixed")


def family_block_permutation(df, metric, fam_idx, n_perm, rng, report):
    """Exchangeability-block permutation tests.

    (a) perm_whole_family: PGS exchanged between whole families of equal
        size; sex and the brain metric stay with the participant. Null:
        PGS is unrelated to the metric distribution (in either sex).
    (b) perm_within_sex: participant-level permutation of the PGS within
        each sex, ignoring family structure (the reviewer-requested test;
        exact under the null that PGS carries no information within sex,
        conditional on independence of participants).
    (c) perm_freedman_lane: Freedman-Lane test of the pgs_z:Gender
        variance-regression coefficient with the PGS and sex main effects as
        nuisance terms; residuals of the reduced auxiliary model are
        exchanged within the equal-size family blocks.
    Regression coefficients are tested with the classical t statistic.
    """
    y = df[metric].values.astype(float)
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    n = len(y)
    n_fam = len(fam_idx)
    observed = sample_statistics(y, pgs, male)
    rows = []

    schemes = (
        ('perm_whole_family',
         lambda: _block_permutation_source(family_size_blocks(fam_idx), rng),
         'null = PGS exchangeable across equal-size families'),
        ('perm_within_sex',
         lambda: _within_sex_participant_source(male, rng),
         'null = PGS exchangeable across participants within sex '
         '(family structure ignored)'),
    )
    for method, make_source, note in schemes:
        t0 = time.time()
        records = []
        for _ in range(n_perm):
            src = make_source()
            records.append(sample_statistics(y, pgs[src], male))
        null = _stats_frame(records)
        report.append(f"  {metric}: {method} ({n_perm} permutations, "
                      f"{time.time() - t0:.1f} s)")
        rows += _perm_rows(observed, null, metric, method, note, n, n_fam,
                           report)

    # ---- Freedman-Lane for the interaction coefficient
    t0 = time.time()
    X_full = np.column_stack([np.ones(n), pgs, male, pgs * male])
    X_red = X_full[:, :3]
    beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
    e2 = (y - X_full @ beta) ** 2
    b_red = np.linalg.lstsq(X_red, e2, rcond=None)[0]
    yhat_red = X_red @ b_red
    r_red = e2 - yhat_red
    XtX_inv = np.linalg.inv(X_full.T @ X_full)
    proj = XtX_inv @ X_full.T

    def _t_interaction(target):
        b = proj @ target
        resid = target - X_full @ b
        sigma2 = resid @ resid / (n - 4)
        return b[3] / np.sqrt(sigma2 * XtX_inv[3, 3])

    t_obs = _t_interaction(e2)
    assert np.isclose(t_obs, observed['vr_interaction_t'])
    t_null = np.empty(n_perm)
    for i in range(n_perm):
        src = _whole_family_permutation(fam_idx, rng)
        t_null[i] = _t_interaction(yhat_red + r_red[src])
    p_fl = (1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (1 + n_perm)
    rows.append({
        'metric': metric, 'test': 'variance_regression',
        'method': 'perm_freedman_lane', 'term': 'pgs_z:Gender',
        'estimate': observed['vr_interaction_beta'], 'se': np.nan,
        'ci_low': np.nan, 'ci_high': np.nan, 'pvalue': float(p_fl),
        'statistic': float(t_obs), 'df': np.nan, 'n': int(n),
        'n_families': int(n_fam), 'B': int(n_perm),
        'note': 'two-sided; Freedman-Lane, reduced model resid^2 ~ pgs_z + '
                'male, residuals exchanged within equal-size family blocks',
    })
    report.append(f"  {metric}: Freedman-Lane permutation of pgs_z:Gender "
                  f"(t_obs = {t_obs:.3f}): p = {p_fl:.4f} "
                  f"({n_perm} permutations, {time.time() - t0:.1f} s)")
    return rows


def sexcheck_sensitivity(df, sexcheck_csv, report):
    """Refit the primary C3 models without the individuals flagged by the
    B1b chrX sex check (discordant or ambiguous F)."""
    report.append("")
    report.append("=" * 60)
    report.append("SENSITIVITY: EXCLUDING chrX SEX-CHECK FLAGGED INDIVIDUALS")
    report.append("=" * 60)
    sc = pd.read_csv(sexcheck_csv, dtype={'IID': str})
    flagged = set(sc.loc[sc['flagged'].astype(bool), 'IID'])
    keep = ~df['Subject'].isin(flagged)
    n_drop = int((~keep).sum())
    report.append(f"  Flagged in B1b: {len(flagged)}; present in the analytic "
                  f"sample: {n_drop}; refitting on n = {int(keep.sum())}")
    sub = df[keep].reset_index(drop=True)
    n_fam = sub['family'].nunique()
    rows = []
    scratch = []
    vr = test_variance_regression_by_sex(sub, scratch)
    assoc = test_pgs_graph_metric_association(sub, scratch)
    note = f"primary OLS refitted without {n_drop} chrX sex-check flagged participants"
    for metric in METRICS:
        for term in _MAIN_COEF_NAMES:
            rows.append({'metric': metric, 'test': 'variance_regression',
                         'method': 'exclude_sexcheck_flagged', 'term': term,
                         'estimate': vr[metric]['coefs'][term], 'se': np.nan,
                         'ci_low': np.nan, 'ci_high': np.nan,
                         'pvalue': vr[metric]['pvalues'][term],
                         'statistic': np.nan, 'df': np.nan, 'n': int(len(sub)),
                         'n_families': int(n_fam), 'B': np.nan, 'note': note})
        for term in _ASSOC_COEF_NAMES:
            rows.append({'metric': metric, 'test': 'association',
                         'method': 'exclude_sexcheck_flagged', 'term': term,
                         'estimate': assoc[metric]['coefs'][term], 'se': np.nan,
                         'ci_low': np.nan, 'ci_high': np.nan,
                         'pvalue': assoc[metric]['pvalues'][term],
                         'statistic': np.nan, 'df': np.nan, 'n': int(len(sub)),
                         'n_families': int(n_fam), 'B': np.nan, 'note': note})
        report.append(f"  {metric:<18} variance pgs_z:Gender beta = "
                      f"{vr[metric]['interaction_beta']:.4e}, "
                      f"p = {vr[metric]['interaction_p']:.4f}; association "
                      f"metric p = {assoc[metric]['pvalues']['metric']:.4f}, "
                      f"metric:Gender p = {assoc[metric]['pvalues']['metric:Gender']:.4f}")
    return rows


# %%
# =============================================================================
# 7. PRIMARY-NUMBER GUARD
# =============================================================================

def reproduce_primary(df, results_dir, report):
    """Reproduce the C3 numbers with C3's own functions and the resampling
    statistic; abort if they disagree with the C3 result files."""
    report.append("")
    report.append("=" * 60)
    report.append("PRIMARY C3 NUMBERS (reproduced from the C3 results table)")
    report.append("=" * 60)
    scratch = []
    vr = test_variance_regression_by_sex(df, scratch)
    assoc = test_pgs_graph_metric_association(df, scratch)

    vr_file = results_dir / 'C3_variance_regression_sex.csv'
    assoc_file = results_dir / 'C3_pgs_metric_association.csv'
    if vr_file.exists() and assoc_file.exists():
        vr_ref = pd.read_csv(vr_file)
        assoc_ref = pd.read_csv(assoc_file)
        for metric in METRICS:
            for term in _MAIN_COEF_NAMES:
                ref = vr_ref[(vr_ref.metric == metric) & (vr_ref.term == term)]
                if not np.isclose(ref['beta'].iloc[0],
                                  vr[metric]['coefs'][term], rtol=1e-6):
                    raise RuntimeError(
                        f"Primary variance-regression beta for {metric}/{term}"
                        " does not match C3_variance_regression_sex.csv; "
                        "rerun C3 before C3b.")
            for term in _ASSOC_COEF_NAMES:
                ref = assoc_ref[(assoc_ref.metric == metric)
                                & (assoc_ref.term == term)]
                if not np.isclose(ref['beta'].iloc[0],
                                  assoc[metric]['coefs'][term], rtol=1e-6):
                    raise RuntimeError(
                        f"Primary association beta for {metric}/{term} does "
                        "not match C3_pgs_metric_association.csv; rerun C3.")
        report.append("  Reproduced C3_variance_regression_sex.csv and "
                      "C3_pgs_metric_association.csv exactly.")
    else:
        report.append("  (C3 coefficient CSVs not found; guard skipped)")

    for metric in METRICS:
        obs = sample_statistics(df[metric].values, df['pgs_z'].values,
                                df['male'].values)
        if not np.isclose(obs['vr_interaction_beta'],
                          vr[metric]['interaction_beta'], rtol=1e-8):
            raise RuntimeError("sample_statistics disagrees with C3's "
                               "variance regression")
        report.append(f"  {metric}: pgs_z:Gender beta = "
                      f"{vr[metric]['interaction_beta']:.6e} "
                      f"(p = {vr[metric]['interaction_p']:.4f}); "
                      f"decile trend r F = {obs['decile_r_F']:.3f}, "
                      f"M = {obs['decile_r_M']:.3f}, M-F = {obs['decile_diff']:.3f}; "
                      f"bin trend r F = {obs['bin_r_F']:.3f}, "
                      f"M = {obs['bin_r_M']:.3f}, M-F = {obs['bin_diff']:.3f}")
    return vr, assoc


# %%
# =============================================================================
# 8. FIGURE
# =============================================================================

METHOD_ORDER = [('primary_ols', 'Primary OLS (i.i.d.)'),
                ('hc1', 'HC1 robust'),
                ('cluster_robust', 'Family cluster-robust (CR1)'),
                ('mixedlm', 'Family random intercept'),
                ('family_bootstrap', 'Family bootstrap (percentile)')]
PANELS = [('variance_regression', 'pgs_z:Gender',
           'Variance regression: PGS x sex'),
          ('decile_trend', 'M - F', 'Decile-SD trend: M - F'),
          ('balanced_bin_trend', 'M - F', 'Quantile-bin variance trend: M - F')]


def plot_forest(summary, figure_path):
    """Forest plot: estimate and 95% CI per inference method; permutation
    p-values annotated. One series per panel, so identity lives on the axis
    labels and no legend is needed."""
    mark = '#2a78d6'
    fig, axes = plt.subplots(len(PANELS), len(METRICS),
                             figsize=(180 * mm2inches, 120 * mm2inches),
                             dpi=FIGURE_DPI,
                             gridspec_kw={'height_ratios': [3.2, 1.4, 1.4]})
    for col, metric in enumerate(METRICS):
        for row, (test, term, title) in enumerate(PANELS):
            ax = axes[row, col]
            sub = summary[(summary.metric == metric) & (summary.test == test)
                          & (summary.term == term)]
            # Variance-regression betas are ~1e-4 in resid^2 units; plot
            # them x1e4 so the axis reads without an offset label.
            scale = 1e4 if test == 'variance_regression' else 1.0
            present = [(m, lab) for m, lab in METHOD_ORDER
                       if not sub[sub.method == m].empty]
            k = len(present)
            labels, ys = [], []
            for i, (method, label) in enumerate(present):
                r = sub[sub.method == method].iloc[0]
                ypos = k - i
                if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                    ax.plot([r.ci_low * scale, r.ci_high * scale],
                            [ypos, ypos], '-', color=mark, linewidth=1.2,
                            solid_capstyle='butt')
                ax.plot(r.estimate * scale, ypos, 'o', color=mark,
                        markersize=4, markeredgecolor='white',
                        markeredgewidth=0.8)
                ax.annotate(f"p = {r.pvalue:.3f}", xy=(1.02, ypos),
                            xycoords=('axes fraction', 'data'), fontsize=6,
                            va='center', ha='left', color='#444')
                labels.append(label)
                ys.append(ypos)
            perm_labels = [('perm_within_sex', 'Permutation within sex'),
                           ('perm_whole_family', 'Permutation, whole-family'),
                           ('perm_freedman_lane', 'Permutation, Freedman-Lane')]
            txt = []
            for method, lab in perm_labels:
                r = sub[sub.method == method]
                if not r.empty:
                    txt.append(f"{lab}: p = {r.iloc[0].pvalue:.3f}")
            n_txt = len(txt)
            if txt:
                ax.text(0.0, k + 0.55 + 0.45 * n_txt, "\n".join(txt),
                        transform=ax.get_yaxis_transform(), fontsize=6,
                        va='top', ha='left', color='#444')
            ax.axvline(0, color='#999', linewidth=0.8, zorder=0)
            ax.set_yticks(ys)
            ax.set_yticklabels(labels, fontsize=6.5)
            ax.set_ylim(0.4, k + 0.6 + 0.5 * n_txt)
            ax.set_title(f"{METRIC_LABELS[metric]}: {title}", fontsize=7.5)
            ax.tick_params(axis='x', labelsize=6.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.grid(axis='x', color='#eee', linewidth=0.6)
            ax.set_axisbelow(True)
            if test == 'variance_regression':
                ax.set_xlabel('beta for pgs_z:sex (x 1e-4, resid^2 units)',
                              fontsize=6.5)
            else:
                ax.set_xlabel('M - F difference in trend r', fontsize=6.5)
    fig.suptitle('Family-structure sensitivity: estimate and 95% CI by '
                 'inference method', fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 0.97, 0.97), h_pad=1.2, w_pad=2.5)
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


# %%
# =============================================================================
# 9. SELF-CHECKS (manual; not part of the Snakemake DAG)
# =============================================================================

def run_self_checks(df, fam_idx, seed, report):
    report.append("")
    report.append("=" * 60)
    report.append("SELF-CHECKS")
    report.append("=" * 60)
    rng = np.random.default_rng(seed + 1)
    X = _build_main_design(df)
    y = df['modularity'].values.astype(float)
    e2 = sm.OLS(y, X).fit().resid ** 2

    # (a) cluster-robust with singleton clusters == HC1
    single = np.arange(len(y))
    hc1 = sm.OLS(e2, X).fit(cov_type='HC1')
    cr = sm.OLS(e2, X).fit(cov_type='cluster', cov_kwds={'groups': single})
    ok_a = np.allclose(hc1.bse, cr.bse, rtol=1e-8)
    report.append(f"  (a) singleton-cluster CR1 == HC1 SEs: {'PASS' if ok_a else 'FAIL'}")

    # (b) shuffled families -> ICC ~ 0, mixed == OLS, boundary flagged
    shuffled = rng.permutation(df['family'].values)
    icc = icc_oneway(y, shuffled)
    data = pd.DataFrame({'y': y, 'pgs_z': df['pgs_z'].values,
                         'male': df['male'].values, 'g': shuffled})
    data['y'] = data['y'] / data['y'].std()
    fit = _fit_mixedlm('y ~ pgs_z + male + pgs_z:male', data, data['g'])
    ols = sm.OLS(data['y'].values, X).fit()
    rel = np.max(np.abs(fit['result'].fe_params.values - ols.params)
                 / np.abs(ols.params))
    ok_b = abs(icc['icc_anova']) < 0.05 and rel < 1e-2
    report.append(f"  (b) shuffled families: ICC(ANOVA) = {icc['icc_anova']:.4f}, "
                  f"ICC(LMM) = {icc['icc_mixedlm']:.4f}, boundary = "
                  f"{fit['boundary']}, max rel. diff mixed vs OLS = {rel:.2e}: "
                  f"{'PASS' if ok_b else 'FAIL'}")

    # (c) Freedman-Lane permutation p uniform under the null
    n_null, n_perm = 100, 200
    pgs = df['pgs_z'].values.astype(float)
    male = df['male'].values.astype(float)
    pvals = []
    scratch = []
    for _ in range(n_null):
        src = _whole_family_permutation(fam_idx, rng)
        null_df = df.assign(pgs_z=pgs[src])
        rows = family_block_permutation(null_df, 'modularity', fam_idx,
                                        n_perm, rng, scratch)
        fl = [r for r in rows if r['method'] == 'perm_freedman_lane'][0]
        pvals.append(fl['pvalue'])
    pvals = np.asarray(pvals)
    ks_p = stats.kstest(pvals, 'uniform').pvalue
    rej = float(np.mean(pvals <= 0.05))
    ok_c = ks_p > 0.01 and rej < 0.12
    report.append(f"  (c) Freedman-Lane p under a family-permuted null "
                  f"({n_null} datasets x {n_perm} perms): KS p = {ks_p:.3f}, "
                  f"rejection rate at 0.05 = {rej:.3f}: {'PASS' if ok_c else 'FAIL'}")
    _ = male
    return ok_a and ok_b and ok_c


# %%
# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='C3b: family-structure sensitivity analysis for C3')
    parser.add_argument('--project', required=True)
    parser.add_argument('--results', required=False,
                        help='C3_heteroscedasticity_results.csv')
    parser.add_argument('--phenotypic', required=False,
                        help='phenotypic_data_anonymised.csv (Family_ID)')
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    parser.add_argument('--n-perm', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sexcheck', required=False,
                        help='B1b_chrx_sexcheck.csv; if given, the primary '
                             'models are refitted without the individuals '
                             'flagged by the chrX sex check')
    parser.add_argument('--self-check', action='store_true',
                        help='run internal validity checks (slow; manual)')
    args = parser.parse_args()

    project = Path(args.project)
    results_dir = project / 'results'
    reports_dir = project / 'reports'
    figures_dir = project / 'figures'
    for d in (results_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    results_csv = (Path(args.results) if args.results
                   else results_dir / 'C3_heteroscedasticity_results.csv')
    phenotypic_csv = (Path(args.phenotypic) if args.phenotypic
                      else project / 'data/raw_anonymised/'
                                     'phenotypic_data_anonymised.csv')

    t_start = time.time()
    report = ["=" * 80,
              "C3b: FAMILY-STRUCTURE SENSITIVITY ANALYSIS FOR C3",
              "=" * 80, "",
              f"Project folder: {project}",
              f"Bootstrap replicates: {args.n_bootstrap}",
              f"Permutations: {args.n_perm}",
              f"Seed: {args.seed}", ""]

    df = load_family_data(results_csv, phenotypic_csv, report)
    fam_idx = family_index(df['family'].values, order_key=df['Subject'].values)
    n_fam = len(fam_idx)

    reproduce_primary(df, results_dir, report)
    icc_df = compute_descriptives(df, report)

    rows = []
    rows += fit_cluster_robust(df, report)
    rows += fit_mixed_models(df, report)

    ss = np.random.SeedSequence(args.seed)
    rng_boot, rng_perm, rng_ratio = [np.random.default_rng(s)
                                     for s in ss.spawn(3)]

    report.append("")
    report.append("=" * 60)
    report.append("FAMILY CLUSTER BOOTSTRAP (families resampled with replacement)")
    report.append("=" * 60)
    for metric in METRICS:
        r, _ = family_bootstrap(df, metric, fam_idx, args.n_bootstrap,
                                rng_boot, report)
        rows += r

    report.append("")
    report.append("=" * 60)
    report.append("TOP vs BOTTOM PGS-QUINTILE SD RATIO BY SEX")
    report.append("=" * 60)
    report.append("  Within-sex PGS quintiles of the covariate-residualised "
                  "metric; ratio = SD(top) / SD(bottom)")
    ratio_wide = []
    for metric in METRICS:
        r, w = quintile_sd_ratio(df, metric, fam_idx, args.n_bootstrap,
                                 rng_ratio, report)
        rows += r
        ratio_wide += w

    report.append("")
    report.append("=" * 60)
    report.append("FAMILY-BLOCK PERMUTATION (exchangeability blocks = "
                  "equal-size families)")
    report.append("=" * 60)
    _describe_blocks(family_size_blocks(fam_idx), "size blocks", report)
    report.append("  perm_within_sex: participant-level permutation within "
                  "sex (family structure ignored)")
    for metric in METRICS:
        rows += family_block_permutation(df, metric, fam_idx, args.n_perm,
                                         rng_perm, report)

    if args.sexcheck:
        rows += sexcheck_sensitivity(df, Path(args.sexcheck), report)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    # ---- Headline comparison
    report.append("")
    report.append("=" * 80)
    report.append("FINAL SUMMARY: headline sex x PGS variance interaction "
                  "(pgs_z:Gender) by method")
    report.append("=" * 80)
    for metric in METRICS:
        report.append(f"{metric.upper()}:")
        sub = summary[(summary.metric == metric)
                      & (summary.test == 'variance_regression')
                      & (summary.term == 'pgs_z:Gender')]
        for _, r in sub.iterrows():
            ci = (f"95% CI [{r.ci_low:.3e}, {r.ci_high:.3e}]"
                  if np.isfinite(r.ci_low)
                  and not r.method.startswith('perm') else "")
            report.append(f"  {r.method:<20} beta = {r.estimate:.4e}  {ci}  "
                          f"p = {r.pvalue:.4f} [{'SIG' if r.pvalue < 0.05 else 'NS'}]")
        for test, label in (('decile_trend', 'Decile-SD trend M - F'),
                            ('balanced_bin_trend', 'Quantile-bin variance trend M - F')):
            sub = summary[(summary.metric == metric) & (summary.test == test)
                          & (summary.term == 'M - F')]
            for _, r in sub.iterrows():
                report.append(f"  {label:<36} {r.method:<20} diff = {r.estimate:.3f}, "
                              f"p = {r.pvalue:.4f} [{'SIG' if r.pvalue < 0.05 else 'NS'}]")

    if args.self_check:
        ok = run_self_checks(df, fam_idx, args.seed, report)
        report.append(f"  Self-checks overall: {'PASS' if ok else 'FAIL'}")

    # ---- Save
    summary_file = results_dir / 'C3b_family_sensitivity_summary.csv'
    icc_file = results_dir / 'C3b_family_icc.csv'
    figure_file = figures_dir / 'C3b_family_sensitivity_forest.png'
    report_file = reports_dir / 'C3b_family_structure_sensitivity_report.txt'
    summary.to_csv(summary_file, index=False, float_format='%.10g')
    icc_df.to_csv(icc_file, index=False, float_format='%.10g')
    ratio_file = results_dir / 'C3b_quintile_sd_ratio.csv'
    pd.DataFrame(ratio_wide).to_csv(ratio_file, index=False,
                                    float_format='%.10g')
    plot_forest(summary, figure_file)

    report.append("")
    report.append(f"Runtime: {time.time() - t_start:.1f} s")
    report.append(f"Summary saved to: {summary_file}")
    report.append(f"ICC table saved to: {icc_file}")
    report.append(f"Quintile SD ratios saved to: {ratio_file}")
    report.append(f"Figure saved to: {figure_file}")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
