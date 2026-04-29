"""
D1_sensitivity_analyses.py

Sensitivity analyses S1-S6 from the heteroscedasticity audit
(see /home/jmbathe/.claude/plans/the-main-findings-that-serialized-dahl.md).

Runs the full C3b heteroscedasticity test battery under multiple analytic
configurations to determine whether the disappearance of the modularity-PGS
heteroscedasticity finding after sex/ancestry filtering reflects a real
correction or a methodological artefact.

Cells:
  C0  ancestry=on, sex=M-only, PGS resid = age + 5 PCs   (current state, baseline)
  C1  ancestry=on, sex=M+F,    PGS resid = age + 5 PCs   (S2: drop sex filter)
  C2  ancestry=off, sex=M-only, PGS resid = age + 10 PCs (S6: drop ancestry)
  C3  ancestry=off, sex=M+F,    PGS resid = age + 10 PCs (S6: drop both)
  C4  ancestry=on, sex=M-only, PGS resid = 10 PCs (no age) (S3)
  C5  ancestry=on, sex=M-only, PGS resid = age + 10 PCs   (more PCs)
  C6  ancestry=on, sex=M-only, PGS resid = none           (raw PGS, z-scored)

For each cell we run BP, White, quantile-IQ, decile-trend, balanced-bootstrap,
plus a joint mean-variance DGLM (S4), and apply Bonferroni + BH-FDR (S5).
"""

import argparse
import json
from pathlib import Path

import bct
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.multitest import multipletests

PROJECT = Path('/home/jmbathe/Documents/1_Projects/BrainCompensation')
N_NODES = 100
NETWORK_THRESHOLD = 0.2
MOTION_THRESHOLD = 0.2
RNG_SEED = 1729


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_raw_inputs():
    """Load all inputs needed across cells. Returns a dict of DataFrames."""
    print("Loading raw inputs ...")
    conn = pd.read_csv(
        PROJECT / 'data/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d100_ts2/netmats1.txt',
        header=None, sep=r'\s+',
    )
    conn.columns = [f'conn_{i+1}' for i in range(conn.shape[1])]
    ids = pd.read_csv(PROJECT / 'data/hcp_subids_raw.txt', header=None)[0].tolist()
    conn.index = ids
    conn.index.name = 'Subject'
    print(f"  connectivity: {conn.shape[0]} subjects, {conn.shape[1]} entries")

    behaviour = pd.read_csv(PROJECT / 'data/hcp_behavioural_raw.csv')
    phenotypic = pd.read_csv(PROJECT / 'data/hcp_phenotypic_raw.csv').rename(
        columns={'Individual_ID': 'Subject'}
    )
    movement = pd.read_csv(PROJECT / 'data/hcp_movement_raw.csv')
    print(f"  behaviour: {len(behaviour)}, phenotypic: {len(phenotypic)}, "
          f"movement: {len(movement)}")

    # PGS variants
    blup_anc = pd.read_csv(
        PROJECT / 'data/PLINK_anonymised/full_pgs_scores.snp.blp.profile',
        sep=r'\s+',
    )[['IID', 'SCORESUM']].rename(columns={'IID': 'Subject', 'SCORESUM': 'pgs_anc'})
    blup_full = pd.read_csv(
        PROJECT / 'data/PLINK_anonymised/D1_sensitivity/full_cohort_pgs.profile',
        sep=r'\s+',
    )[['IID', 'SCORESUM']].rename(columns={'IID': 'Subject', 'SCORESUM': 'pgs_full'})
    print(f"  PGS (ancestry-filtered, BLUP): {len(blup_anc)}")
    print(f"  PGS (full pre-ancestry cohort): {len(blup_full)}")

    # PCA variants
    pca_anc = pd.read_csv(
        PROJECT / 'data/PLINK_anonymised/Neuro_Chip_full_sample_pca.eigenvec',
        sep=' ', header=None,
    )
    pca_anc.columns = ['FID', 'Subject'] + [f'PCanc{i}' for i in range(1, 11)]
    pca_anc = pca_anc.drop(columns='FID')

    pca_full = pd.read_csv(
        PROJECT / 'data/PLINK_anonymised/D1_sensitivity/full_cohort_pca.eigenvec',
        sep=' ', header=None,
    )
    pca_full.columns = ['FID', 'Subject'] + [f'PCfull{i}' for i in range(1, 11)]
    pca_full = pca_full.drop(columns='FID')
    print(f"  PCA (ancestry-filtered): {len(pca_anc)}, full cohort: {len(pca_full)}")

    return dict(
        conn=conn, behaviour=behaviour, phenotypic=phenotypic,
        movement=movement,
        blup_anc=blup_anc, blup_full=blup_full,
        pca_anc=pca_anc, pca_full=pca_full,
    )


# --------------------------------------------------------------------------- #
# Network metrics — computed once on all subjects with connectivity
# --------------------------------------------------------------------------- #

def compute_network_metrics_all(conn, partition_path):
    """Compute raw modularity and global_efficiency for every connectivity
    subject. Returns DataFrame indexed by Subject."""
    print(f"Computing network metrics for {len(conn)} subjects ...")
    partition = pd.read_csv(partition_path)['community_id'].values

    out = []
    for i, (subject_id, row) in enumerate(conn.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(conn)}")
        mat = row.values.reshape(N_NODES, N_NODES).astype(np.float64) / 100.0
        mat = bct.threshold_proportional(mat, NETWORK_THRESHOLD)
        mat = np.nan_to_num(mat, nan=0.0)
        _, modularity = bct.modularity_und_sign(mat, partition)
        mat_pos = mat.copy()
        mat_pos[mat_pos < 0] = 0
        ge = nx.global_efficiency(nx.from_numpy_array(mat_pos))
        out.append({'Subject': subject_id, 'modularity': modularity,
                    'global_efficiency': ge})
    return pd.DataFrame(out).set_index('Subject')


# --------------------------------------------------------------------------- #
# Per-cell helpers
# --------------------------------------------------------------------------- #

def residualise_pgs(df, pgs_col, pc_cols, include_age, include_gender=False):
    """OLS-residualise PGS on selected covariates. Returns z-scored residuals.
    include_gender adds a 0/1 male dummy when the cohort is mixed-sex."""
    cols = list(pc_cols) + (['Age_in_Yrs'] if include_age else [])
    X_parts = []
    if include_gender:
        X_parts.append((df['Gender'] == 'M').astype(float).values.reshape(-1, 1))
    if cols:
        X_parts.append(df[cols].values.astype(float))
    if not X_parts:
        return stats.zscore(df[pgs_col].values)
    X = sm.add_constant(np.concatenate(X_parts, axis=1))
    y = df[pgs_col].values.astype(float)
    resid = y - sm.OLS(y, X).fit().predict(X)
    return stats.zscore(resid)


def regress_out(y, X_df):
    """OLS-residualise y on X_df (with dummy encoding)."""
    X = pd.get_dummies(X_df, drop_first=True).astype(float).values
    if X.shape[1] == 0:
        return y - y.mean()
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)


# --------------------------------------------------------------------------- #
# Heteroscedasticity test battery
# --------------------------------------------------------------------------- #

def bp_white(df, metric):
    y = df[metric].values
    X = sm.add_constant(df['pgs_z'].values)
    model = sm.OLS(y, X).fit()
    bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, X)
    w_lm, w_p, _, _ = het_white(model.resid, X)
    return bp_lm, bp_p, w_lm, w_p


def quantile_iq(df, metric, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    y = df[metric].values
    X = sm.add_constant(df['pgs_z'].values)
    slopes = []
    for q in quantiles:
        try:
            res = QuantReg(y, X).fit(q=q, max_iter=2000)
            slopes.append((q, res.params[1], res.bse[1]))
        except Exception:
            slopes.append((q, np.nan, np.nan))
    slope_arr = np.array([s[1] for s in slopes])
    se_arr = np.array([s[2] for s in slopes])
    slope_range = float(np.nanmax(slope_arr) - np.nanmin(slope_arr))
    diff = slopes[-1][1] - slopes[0][1]
    se_diff = np.sqrt(slopes[-1][2]**2 + slopes[0][2]**2)
    t = diff / se_diff if se_diff > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    return slope_range, diff, t, p, slopes


def decile_trend(df, metric, n_bins=10, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    df_sorted = df.sort_values('pgs_z').reset_index(drop=True)
    df_sorted['bin'] = pd.qcut(df_sorted['pgs_z'], n_bins,
                               labels=False, duplicates='drop')
    centres, sds = [], []
    for b in sorted(df_sorted['bin'].dropna().unique()):
        sub = df_sorted[df_sorted['bin'] == b]
        if len(sub) >= 2:
            centres.append(sub['pgs_z'].mean())
            sds.append(sub[metric].std(ddof=1))
    if len(centres) < 3:
        return np.nan, np.nan, [], []
    r, p = stats.pearsonr(centres, sds)
    return r, p, centres, sds


def balanced_bootstrap(df, metric, n_bins=5, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    df_sorted = df.sort_values('pgs_z').reset_index(drop=True)
    df_sorted['bin'] = pd.qcut(df_sorted['pgs_z'], n_bins,
                               labels=False, duplicates='drop')
    bin_groups = [df_sorted[df_sorted['bin'] == b][metric].values
                  for b in sorted(df_sorted['bin'].dropna().unique())]
    if len(bin_groups) < 3:
        return np.nan, np.nan, np.nan
    n_per = min(len(g) for g in bin_groups)
    bin_centres = np.arange(len(bin_groups))
    obs_var = np.array([np.var(g, ddof=1) for g in bin_groups])
    obs_r = stats.pearsonr(bin_centres, obs_var)[0]

    boot_r = np.empty(n_boot)
    boot_ratio = np.empty(n_boot)
    for b in range(n_boot):
        var_b = np.array([
            np.var(rng.choice(g, size=n_per, replace=True), ddof=1)
            for g in bin_groups
        ])
        boot_r[b] = stats.pearsonr(bin_centres, var_b)[0]
        boot_ratio[b] = var_b[-1] / var_b[0] if var_b[0] > 0 else np.nan
    p = float(np.mean(boot_r <= 0))  # H1: increasing
    median_ratio = float(np.nanmedian(boot_ratio))
    return obs_r, p, median_ratio


def fit_dglm(y_in, X_mu, X_sigma):
    """Joint mean-variance Gaussian DGLM. Returns variance-PGS coefficient,
    likelihood-ratio statistic, and LRT p-value (1 df).

    y is z-scored internally so the optimiser sees gradients on a sane scale;
    the variance coefficient is invariant to rescaling y by a constant.
    """
    y_std = float(np.std(y_in, ddof=1))
    if y_std == 0:
        return np.nan, np.nan, np.nan
    y = (y_in - np.mean(y_in)) / y_std

    p_mu = X_mu.shape[1]

    def neg_ll(params, X_s):
        bm = params[:p_mu]
        bs = params[p_mu:p_mu + X_s.shape[1]]
        mu = X_mu @ bm
        log_var = X_s @ bs
        log_var = np.clip(log_var, -30, 30)
        var = np.exp(log_var)
        return 0.5 * np.sum(log_var + (y - mu) ** 2 / var)

    def fit(X_s):
        bm0 = np.linalg.lstsq(X_mu, y, rcond=None)[0]
        resid = y - X_mu @ bm0
        bs0 = np.zeros(X_s.shape[1])
        bs0[0] = float(np.log(max(np.var(resid), 1e-8)))
        x0 = np.concatenate([bm0, bs0])
        # First pass with Nelder-Mead to escape flat regions, then BFGS for
        # accurate convergence
        r1 = minimize(lambda p: neg_ll(p, X_s), x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        r2 = minimize(lambda p: neg_ll(p, X_s), r1.x, method='BFGS',
                      options={'maxiter': 1000, 'gtol': 1e-7})
        return r2

    full = fit(X_sigma)
    X_red = X_sigma[:, :1]  # intercept-only variance
    red = fit(X_red)
    lr_stat = float(2 * (red.fun - full.fun))
    df_diff = X_sigma.shape[1] - 1
    p_lrt = float(stats.chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
    alpha_pgs = float(full.x[p_mu + 1]) if X_sigma.shape[1] > 1 else np.nan
    return alpha_pgs, lr_stat, p_lrt


# --------------------------------------------------------------------------- #
# Cell runner
# --------------------------------------------------------------------------- #

def assemble_cell_df(spec, raw, network_df):
    """Build the analysis dataframe for one cell."""
    pgs_col = 'pgs_anc' if spec['ancestry'] == 'on' else 'pgs_full'
    pgs_df = raw['blup_anc'] if spec['ancestry'] == 'on' else raw['blup_full']
    pca_df = raw['pca_anc'] if spec['ancestry'] == 'on' else raw['pca_full']

    pca_cols = [c for c in pca_df.columns if c.startswith('PC')]
    df = network_df.reset_index().merge(pgs_df, on='Subject')
    df = df.merge(pca_df, on='Subject')
    df = df.merge(raw['behaviour'][['Subject', 'Gender', 'FS_IntraCranial_Vol']],
                  on='Subject')
    df = df.merge(raw['phenotypic'][['Subject', 'Age_in_Yrs']], on='Subject')
    df = df.merge(raw['movement'][['Subject', 'Movement_RelativeRMS_mean']],
                  on='Subject')

    if spec['sex_filter'] == 'M':
        df = df[df['Gender'] == 'M']
    df = df[df['Movement_RelativeRMS_mean'] < MOTION_THRESHOLD]
    df = df.dropna(subset=['Age_in_Yrs', 'FS_IntraCranial_Vol',
                           'Movement_RelativeRMS_mean', pgs_col,
                           'modularity', 'global_efficiency'])
    df = df.copy()

    # PGS residualisation per cell spec
    n_pcs = spec['n_pcs']
    chosen_pcs = pca_cols[:n_pcs] if n_pcs > 0 else []
    if spec['pgs_residualise']:
        df['pgs_z'] = residualise_pgs(df, pgs_col, chosen_pcs,
                                       include_gender=(spec['sex_filter'] == 'both'),
                                      include_age=spec['include_age_in_pgs'])
    else:
        df['pgs_z'] = stats.zscore(df[pgs_col].values)

    return df, chosen_pcs


def run_cell(spec, raw, network_df, report):
    df, chosen_pcs = assemble_cell_df(spec, raw, network_df)
    n = len(df)
    report.append("")
    report.append("=" * 78)
    report.append(f"CELL {spec['name']}: {spec['description']}")
    report.append("=" * 78)
    report.append(f"  ancestry filter        : {spec['ancestry']}")
    report.append(f"  sex filter             : {spec['sex_filter']}")
    pgs_resid_terms = []
    if spec['n_pcs']:
        pgs_resid_terms.append(f"PC1..PC{spec['n_pcs']}")
    if spec['include_age_in_pgs']:
        pgs_resid_terms.append('Age_in_Yrs')
    pgs_resid_str = '+'.join(pgs_resid_terms) if pgs_resid_terms else '(none, raw)'
    report.append(f"  PGS residualised on    : {pgs_resid_str}")
    report.append(f"  N subjects             : {n}")
    if n < 50:
        report.append(f"  *** WARNING: very small cell ({n} subjects)")
        return None

    # Build covariate set for brain metrics
    covariates = ['Age_in_Yrs', 'FS_IntraCranial_Vol', 'Movement_RelativeRMS_mean']
    if spec['sex_filter'] == 'both':
        covariates.append('Gender')
    if spec['ancestry'] == 'off':
        # Adjust outcome for population structure since PGS residualisation alone
        # cannot eliminate ancestry-driven variance in the brain phenotype
        covariates.extend(chosen_pcs)
    report.append(f"  brain metric covariates: {', '.join(covariates)}")

    cov_df = df[covariates].copy()
    df['modularity_resid'] = regress_out(df['modularity'].values, cov_df)
    df['global_efficiency_resid'] = regress_out(
        df['global_efficiency'].values, cov_df
    )

    # PGS-bin sizes
    z = df['pgs_z'].values
    n_low = int((z <= -1).sum())
    n_high = int((z >= 1).sum())
    report.append(f"  PGS tail sizes (|z|>1) : low={n_low}, high={n_high}")

    metrics = ('modularity_resid', 'global_efficiency_resid')
    cell_results = {'name': spec['name'], 'description': spec['description'],
                    'n': n, 'n_low': n_low, 'n_high': n_high}

    p_pool = []  # for multiple comparison correction
    p_labels = []

    for metric in metrics:
        m_label = metric.replace('_resid', '')
        bp_lm, bp_p, w_lm, w_p = bp_white(df, metric)
        sl_range, sl_diff, qt_t, qt_p, _ = quantile_iq(df, metric)
        d_r, d_p, _, _ = decile_trend(df, metric)
        b_r, b_p, b_ratio = balanced_bootstrap(df, metric, n_boot=1000)

        # DGLM with covariates in mean, PGS in variance
        Xmu_cols = ['pgs_z'] + [c for c in covariates if c != 'Gender']
        if 'Gender' in covariates:
            df['Gender_M'] = (df['Gender'] == 'M').astype(int)
            Xmu_cols.append('Gender_M')
        Xmu = sm.add_constant(df[Xmu_cols].values.astype(float))
        Xsig = sm.add_constant(df[['pgs_z']].values.astype(float))
        try:
            alpha_pgs, lr_stat, lr_p = fit_dglm(df[metric].values, Xmu, Xsig)
        except Exception as e:
            alpha_pgs, lr_stat, lr_p = np.nan, np.nan, np.nan

        report.append("")
        report.append(f"  ---- {m_label} ----")
        report.append(f"    BP   : LM={bp_lm:7.3f}  p={bp_p:.4f}")
        report.append(f"    White: LM={w_lm:7.3f}  p={w_p:.4f}")
        report.append(f"    Quant: range={sl_range:8.5f}  diff={sl_diff:8.5f}  t={qt_t:6.3f}  p={qt_p:.4f}")
        report.append(f"    Decile trend: r={d_r:+.3f}  p={d_p:.4f}")
        report.append(f"    Bootstrap: r_obs={b_r:+.3f}  p_inc={b_p:.4f}  var_ratio_high/low={b_ratio:.3f}")
        report.append(f"    DGLM  : alpha_PGS(log var)={alpha_pgs:+.4f}  LR={lr_stat:7.3f}  p={lr_p:.4f}")

        cell_results.update({
            f'{m_label}_bp_p': bp_p, f'{m_label}_white_p': w_p,
            f'{m_label}_quant_p': qt_p, f'{m_label}_quant_range': sl_range,
            f'{m_label}_decile_r': d_r, f'{m_label}_decile_p': d_p,
            f'{m_label}_boot_r': b_r, f'{m_label}_boot_p_inc': b_p,
            f'{m_label}_boot_ratio': b_ratio,
            f'{m_label}_dglm_alpha': alpha_pgs,
            f'{m_label}_dglm_lr': lr_stat, f'{m_label}_dglm_p': lr_p,
        })
        for nm, p in [('bp', bp_p), ('white', w_p), ('quant', qt_p),
                      ('decile', d_p), ('boot', b_p), ('dglm', lr_p)]:
            if np.isfinite(p):
                p_pool.append(p)
                p_labels.append(f'{m_label}_{nm}')

    # Multiple comparison correction across the cell's 12 tests
    if p_pool:
        bonf_reject, bonf_p, _, _ = multipletests(p_pool, alpha=0.05,
                                                   method='bonferroni')
        fdr_reject, fdr_p, _, _ = multipletests(p_pool, alpha=0.05,
                                                 method='fdr_bh')
        n_sig_unc = int(np.sum(np.array(p_pool) < 0.05))
        n_sig_fdr = int(np.sum(fdr_reject))
        n_sig_bonf = int(np.sum(bonf_reject))
        report.append("")
        report.append(f"  Multiple-comparison correction across {len(p_pool)} tests:")
        report.append(f"    uncorrected p<.05 : {n_sig_unc}")
        report.append(f"    FDR-BH    p<.05   : {n_sig_fdr}")
        report.append(f"    Bonferroni p<.05  : {n_sig_bonf}")
        if n_sig_fdr:
            sig_idx = np.where(fdr_reject)[0]
            report.append(
                "    FDR-significant: " +
                "; ".join(f"{p_labels[i]}(p_raw={p_pool[i]:.4f},p_fdr={fdr_p[i]:.4f})"
                          for i in sig_idx)
            )
        cell_results.update({'n_sig_unc': n_sig_unc, 'n_sig_fdr': n_sig_fdr,
                             'n_sig_bonf': n_sig_bonf})
    return cell_results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    out_dir = PROJECT / 'results/D1_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_inputs()
    network_df = compute_network_metrics_all(
        raw['conn'], PROJECT / 'results/C2b_selected_partition.csv'
    )
    network_df.to_csv(out_dir / 'network_metrics_all_subjects.csv')
    print(f"  metrics computed for {len(network_df)} subjects")

    cells = [
        dict(name='C0', description='current state (anc=on, mixed-sex, age+5PCs)',
             ancestry='on', sex_filter='both', n_pcs=5, include_age_in_pgs=True,
             pgs_residualise=True),
        dict(name='C1', description='S2: restrict to males-only',
             ancestry='on', sex_filter='M', n_pcs=5, include_age_in_pgs=True,
             pgs_residualise=True),
        dict(name='C2', description='S6: drop ancestry filter',
             ancestry='off', sex_filter='both', n_pcs=10, include_age_in_pgs=True,
             pgs_residualise=True),
        dict(name='C3', description='S6: drop ancestry, restrict to males-only',
             ancestry='off', sex_filter='M', n_pcs=10, include_age_in_pgs=True,
             pgs_residualise=True),
        dict(name='C4', description='S3: drop age from PGS resid; 10 PCs',
             ancestry='on', sex_filter='both', n_pcs=10, include_age_in_pgs=False,
             pgs_residualise=True),
        dict(name='C5', description='S3 partial: keep age, use 10 PCs',
             ancestry='on', sex_filter='both', n_pcs=10, include_age_in_pgs=True,
             pgs_residualise=True),
        dict(name='C6', description='no PGS residualisation (raw, z-scored)',
             ancestry='on', sex_filter='both', n_pcs=0, include_age_in_pgs=False,
             pgs_residualise=False),
    ]

    report = []
    report.append("=" * 78)
    report.append("D1 SENSITIVITY ANALYSES — heteroscedasticity audit")
    report.append("=" * 78)
    report.append(f"Project: {PROJECT}")
    report.append(f"N nodes: {N_NODES}, network threshold: {NETWORK_THRESHOLD}, "
                  f"motion threshold: {MOTION_THRESHOLD}")

    cell_results = []
    for spec in cells:
        try:
            res = run_cell(spec, raw, network_df, report)
            if res is not None:
                cell_results.append(res)
        except Exception as e:
            report.append(f"  *** CELL {spec['name']} FAILED: {e}")
            print(f"  cell {spec['name']} failed: {e}")
            raise

    # Master table
    table_df = pd.DataFrame(cell_results)
    table_df.to_csv(out_dir / 'D1_master_sensitivity_table.csv', index=False)

    # Compact summary table for the report
    summary_cols = ['name', 'description', 'n', 'n_low', 'n_high',
                    'modularity_decile_r', 'modularity_decile_p',
                    'modularity_boot_ratio', 'modularity_boot_p_inc',
                    'modularity_dglm_alpha', 'modularity_dglm_p',
                    'global_efficiency_decile_r', 'global_efficiency_decile_p',
                    'global_efficiency_dglm_alpha', 'global_efficiency_dglm_p',
                    'n_sig_unc', 'n_sig_fdr', 'n_sig_bonf']
    summary = table_df[summary_cols]
    summary.to_csv(out_dir / 'D1_summary_table.csv', index=False)

    report.append("")
    report.append("=" * 78)
    report.append("MASTER SUMMARY TABLE (per-cell headlines)")
    report.append("=" * 78)
    with pd.option_context('display.width', 200, 'display.max_columns', 30,
                           'display.float_format', lambda x: f'{x:7.4f}'):
        report.append(summary.to_string(index=False))

    report_path = PROJECT / 'reports/D1_sensitivity_analyses_report.txt'
    report_path.write_text('\n'.join(report) + '\n')
    print(f"\nReport: {report_path}")
    print(f"Tables: {out_dir}")


if __name__ == '__main__':
    main()
