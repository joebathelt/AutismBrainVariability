"""
D2_grid_sensitivity.py

Grid sweep across parcellation x matrix-type x edge-density threshold to find
configurations that change the SDS-driven heteroscedasticity finding. Holds
the cohort fixed at the C0 baseline (M+F, motion<0.2, brain metrics
residualised on age+ICV+motion+Gender). PGS is not loaded.

Grid:
  parcellation : 15, 25, 50, 100, 200, 300 nodes
  matrix type  : netmats1 (full correlation) | netmats2 (partial correlation)
  threshold    : 0.15, 0.20, 0.25 proportional density
  -> 6 x 2 x 3 = 36 cells

Predictor: sds_z = z(Social_Score). For each cell we run BP, White,
decile-trend, balanced-bootstrap, and DGLM, then apply Bonferroni + BH-FDR
within the cell.
"""

import argparse
import itertools
from pathlib import Path

import bct
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.multitest import multipletests

PROJECT = Path('/home/jmbathe/Documents/1_Projects/BrainCompensation')
MOTION_THRESHOLD = 0.2
RNG_SEED = 1729

PARCELLATIONS = [15, 25, 50, 100, 200, 300]
MATRIX_TYPES = ['netmats1', 'netmats2']
THRESHOLDS = [0.15, 0.20, 0.25]


# --------------------------------------------------------------------------- #
# Cohort assembly (M-only, motion<0.2, no PGS join)
# --------------------------------------------------------------------------- #

def load_cohort_inputs():
    behaviour = pd.read_csv(PROJECT / 'data/hcp_behavioural_raw.csv')
    # No sex filter: Gender is residualised out of brain metrics below.
    phenotypic = pd.read_csv(PROJECT / 'data/hcp_phenotypic_raw.csv').rename(
        columns={'Individual_ID': 'Subject'}
    )
    movement = pd.read_csv(PROJECT / 'data/hcp_movement_raw.csv')
    social = pd.read_csv(PROJECT / 'results/cfa_factor_scores_full_sample.csv')
    return dict(behaviour=behaviour, phenotypic=phenotypic,
                movement=movement, social=social)


def get_partition(n_nodes):
    if n_nodes == 100:
        path = PROJECT / 'results/C2b_selected_partition.csv'
    else:
        path = PROJECT / f'results/C2_final_partition_{n_nodes}Nodes.csv'
    return pd.read_csv(path)['community_id'].values


def compute_network_metrics(n_nodes, matrix_type, threshold, ids):
    matrix_file = (
        PROJECT
        / f'data/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d{n_nodes}_ts2/{matrix_type}.txt'
    )
    conn = pd.read_csv(matrix_file, header=None, sep=r'\s+')
    if conn.shape[1] != n_nodes ** 2:
        raise ValueError(f"matrix {matrix_file} has {conn.shape[1]} cols, "
                         f"expected {n_nodes**2}")
    conn.index = ids
    partition = get_partition(n_nodes)
    out = []
    for subject_id, row in conn.iterrows():
        mat = row.values.reshape(n_nodes, n_nodes).astype(np.float64) / 100.0
        mat = bct.threshold_proportional(mat, threshold)
        mat = np.nan_to_num(mat, nan=0.0)
        _, modularity = bct.modularity_und_sign(mat, partition)
        mat_pos = mat.copy(); mat_pos[mat_pos < 0] = 0
        ge = nx.global_efficiency(nx.from_numpy_array(mat_pos))
        out.append({'Subject': subject_id, 'modularity': modularity,
                    'global_efficiency': ge})
    return pd.DataFrame(out).set_index('Subject')


# --------------------------------------------------------------------------- #
# Stat helpers
# --------------------------------------------------------------------------- #

def regress_out(y, X_df):
    X = pd.get_dummies(X_df, drop_first=True).astype(float).values
    if X.shape[1] == 0:
        return y - y.mean()
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)


def bp_white_test(df, metric):
    y = df[metric].values
    X = sm.add_constant(df['sds_z'].values)
    resid = sm.OLS(y, X).fit().resid
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X)
    w_lm, w_p, _, _ = het_white(resid, X)
    return bp_lm, bp_p, w_lm, w_p


def decile_trend(df, metric, n_bins=10):
    s = df.sort_values('sds_z').reset_index(drop=True)
    s['bin'] = pd.qcut(s['sds_z'], n_bins, labels=False, duplicates='drop')
    centres, sds = [], []
    for b in sorted(s['bin'].dropna().unique()):
        sub = s[s['bin'] == b]
        if len(sub) >= 2:
            centres.append(sub['sds_z'].mean())
            sds.append(sub[metric].std(ddof=1))
    if len(centres) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(centres, sds)
    return r, p


def balanced_bootstrap(df, metric, n_bins=5, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    s = df.sort_values('sds_z').reset_index(drop=True)
    s['bin'] = pd.qcut(s['sds_z'], n_bins, labels=False, duplicates='drop')
    groups = [s[s['bin'] == b][metric].values
              for b in sorted(s['bin'].dropna().unique())]
    if len(groups) < 3:
        return np.nan, np.nan, np.nan
    n_per = min(len(g) for g in groups)
    centres = np.arange(len(groups))
    obs_var = np.array([np.var(g, ddof=1) for g in groups])
    obs_r = stats.pearsonr(centres, obs_var)[0]
    boot_r = np.empty(n_boot); boot_ratio = np.empty(n_boot)
    for k in range(n_boot):
        var_b = np.array([
            np.var(rng.choice(g, size=n_per, replace=True), ddof=1)
            for g in groups
        ])
        boot_r[k] = stats.pearsonr(centres, var_b)[0]
        boot_ratio[k] = var_b[-1] / var_b[0] if var_b[0] > 0 else np.nan
    return obs_r, float(np.mean(boot_r <= 0)), float(np.nanmedian(boot_ratio))


def fit_dglm(y_in, X_mu, X_sigma):
    y_std = float(np.std(y_in, ddof=1))
    if y_std == 0:
        return np.nan, np.nan, np.nan
    y = (y_in - np.mean(y_in)) / y_std
    p_mu = X_mu.shape[1]

    def neg_ll(params, X_s):
        bm = params[:p_mu]
        bs = params[p_mu:p_mu + X_s.shape[1]]
        log_var = np.clip(X_s @ bs, -30, 30)
        return 0.5 * np.sum(log_var + (y - X_mu @ bm) ** 2 / np.exp(log_var))

    def fit(X_s):
        bm0 = np.linalg.lstsq(X_mu, y, rcond=None)[0]
        resid = y - X_mu @ bm0
        bs0 = np.zeros(X_s.shape[1])
        bs0[0] = float(np.log(max(np.var(resid), 1e-8)))
        x0 = np.concatenate([bm0, bs0])
        r1 = minimize(lambda p: neg_ll(p, X_s), x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        r2 = minimize(lambda p: neg_ll(p, X_s), r1.x, method='BFGS',
                      options={'maxiter': 1000, 'gtol': 1e-7})
        return r2

    full = fit(X_sigma)
    red = fit(X_sigma[:, :1])
    lr = float(2 * (red.fun - full.fun))
    p = float(stats.chi2.sf(lr, df=X_sigma.shape[1] - 1))
    alpha = float(full.x[p_mu + 1])
    return alpha, lr, p


# --------------------------------------------------------------------------- #
# One grid cell
# --------------------------------------------------------------------------- #

def run_grid_cell(n_nodes, matrix_type, threshold, ids, raw):
    print(f"  d={n_nodes}  type={matrix_type}  thr={threshold}")
    metrics = compute_network_metrics(n_nodes, matrix_type, threshold, ids)

    df = metrics.reset_index().merge(
        raw['social'][['Subject', 'Social_Score']], on='Subject'
    )
    df = df.merge(raw['behaviour'][['Subject', 'Gender', 'FS_IntraCranial_Vol']],
                  on='Subject')
    df = df.merge(raw['phenotypic'][['Subject', 'Age_in_Yrs']], on='Subject')
    df = df.merge(raw['movement'][['Subject', 'Movement_RelativeRMS_mean']],
                  on='Subject')
    df = df[df['Movement_RelativeRMS_mean'] < MOTION_THRESHOLD]
    df = df.dropna(subset=['Age_in_Yrs', 'FS_IntraCranial_Vol',
                           'Movement_RelativeRMS_mean', 'Social_Score',
                           'modularity', 'global_efficiency'])
    df = df.copy()
    df['sds_z'] = stats.zscore(df['Social_Score'].values)

    cov_df = df[['Age_in_Yrs', 'FS_IntraCranial_Vol',
                 'Movement_RelativeRMS_mean', 'Gender']].copy()
    df['modularity_resid'] = regress_out(df['modularity'].values, cov_df)
    df['global_efficiency_resid'] = regress_out(
        df['global_efficiency'].values, cov_df
    )

    n = len(df)
    z = df['sds_z'].values
    n_low = int((z <= -1).sum()); n_high = int((z >= 1).sum())

    out = {
        'n_nodes': n_nodes, 'matrix_type': matrix_type, 'threshold': threshold,
        'n': n, 'n_low': n_low, 'n_high': n_high,
    }
    p_pool, p_labels = [], []
    for metric in ('modularity_resid', 'global_efficiency_resid'):
        m_label = metric.replace('_resid', '')
        bp_lm, bp_p, w_lm, w_p = bp_white_test(df, metric)
        d_r, d_p = decile_trend(df, metric)
        b_r, b_p, b_ratio = balanced_bootstrap(df, metric, n_boot=1000)

        Xmu_df = df[['sds_z', 'Age_in_Yrs', 'FS_IntraCranial_Vol',
                     'Movement_RelativeRMS_mean']].copy()
        Xmu_df['Gender_M'] = (df['Gender'] == 'M').astype(int)
        Xmu = sm.add_constant(Xmu_df.values.astype(float))
        Xsig = sm.add_constant(df[['sds_z']].values.astype(float))
        alpha, lr, lr_p = fit_dglm(df[metric].values, Xmu, Xsig)

        out.update({
            f'{m_label}_bp_p': bp_p, f'{m_label}_white_p': w_p,
            f'{m_label}_decile_r': d_r, f'{m_label}_decile_p': d_p,
            f'{m_label}_boot_r': b_r, f'{m_label}_boot_p_inc': b_p,
            f'{m_label}_boot_ratio': b_ratio,
            f'{m_label}_dglm_alpha': alpha,
            f'{m_label}_dglm_lr': lr, f'{m_label}_dglm_p': lr_p,
        })
        for nm, p in [('bp', bp_p), ('white', w_p), ('decile', d_p),
                      ('boot', b_p), ('dglm', lr_p)]:
            if np.isfinite(p):
                p_pool.append(p)
                p_labels.append(f'{m_label}_{nm}')

    if p_pool:
        bonf, _, _, _ = multipletests(p_pool, alpha=0.05, method='bonferroni')
        fdr, fdr_p, _, _ = multipletests(p_pool, alpha=0.05, method='fdr_bh')
        out['n_sig_unc'] = int(np.sum(np.array(p_pool) < 0.05))
        out['n_sig_fdr'] = int(np.sum(fdr))
        out['n_sig_bonf'] = int(np.sum(bonf))
        mod_dec_p = out.get('modularity_decile_p', np.nan)
        mod_dec_r = out.get('modularity_decile_r', np.nan)
        out['modularity_increasing_var_sig'] = (
            np.isfinite(mod_dec_p) and mod_dec_p < 0.05 and mod_dec_r > 0
        )
        out['modularity_boot_increasing_var_sig'] = (
            np.isfinite(out['modularity_boot_p_inc'])
            and out['modularity_boot_p_inc'] < 0.05
            and out.get('modularity_boot_ratio', np.nan) > 1
        )
        out['modularity_dglm_increasing_var_sig'] = (
            np.isfinite(out['modularity_dglm_p'])
            and out['modularity_dglm_p'] < 0.05
            and out.get('modularity_dglm_alpha', np.nan) > 0
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=str(PROJECT))
    args = parser.parse_args()
    project = Path(args.project)

    out_dir = project / 'results/D1_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_cohort_inputs()
    ids = pd.read_csv(project / 'data/hcp_subids_raw.txt', header=None)[0].tolist()
    print(f"Cohort: {len(raw['behaviour'])} M+F behavioural; "
          f"{len(raw['social'])} social factor; {len(ids)} connectivity ids")

    results = []
    combos = list(itertools.product(PARCELLATIONS, MATRIX_TYPES, THRESHOLDS))
    print(f"Running {len(combos)} grid cells ...")
    for n_nodes, mt, thr in combos:
        try:
            res = run_grid_cell(n_nodes, mt, thr, ids, raw)
            results.append(res)
        except Exception as e:
            print(f"    FAILED ({n_nodes}, {mt}, {thr}): {e}")
            results.append({'n_nodes': n_nodes, 'matrix_type': mt,
                            'threshold': thr, 'error': str(e)})

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'D2_grid_results.csv', index=False)

    cols = ['n_nodes', 'matrix_type', 'threshold', 'n', 'n_low', 'n_high',
            'modularity_decile_r', 'modularity_decile_p',
            'modularity_boot_ratio', 'modularity_boot_p_inc',
            'modularity_dglm_alpha', 'modularity_dglm_p',
            'global_efficiency_decile_r', 'global_efficiency_decile_p',
            'modularity_increasing_var_sig',
            'modularity_boot_increasing_var_sig',
            'modularity_dglm_increasing_var_sig',
            'n_sig_fdr']
    summary = df[[c for c in cols if c in df.columns]]
    summary.to_csv(out_dir / 'D2_grid_summary.csv', index=False)

    report = []
    report.append("=" * 88)
    report.append("D2 GRID SENSITIVITY — parcellation x matrix-type x threshold (SDS predictor)")
    report.append("=" * 88)
    report.append("Cohort: M-only, motion<0.2, brain metrics residualised on age+ICV+motion")
    report.append("Predictor: sds_z = z(Social_Score). PGS not loaded.")
    report.append(f"Grid: {PARCELLATIONS} x {MATRIX_TYPES} x {THRESHOLDS} = {len(combos)} cells")
    report.append("")
    report.append("Master grid (modularity-focused columns):")
    with pd.option_context('display.width', 200, 'display.max_columns', 30,
                           'display.float_format', lambda x: f'{x:7.4f}'):
        report.append(summary.to_string(index=False))
    report.append("")

    n_dec = int(df.get('modularity_increasing_var_sig', pd.Series([])).sum())
    n_boot = int(df.get('modularity_boot_increasing_var_sig', pd.Series([])).sum())
    n_dgl = int(df.get('modularity_dglm_increasing_var_sig', pd.Series([])).sum())
    report.append("Cells where modularity variance INCREASES with SDS (p < .05, uncorrected):")
    report.append(f"  Decile trend (r > 0)             : {n_dec} / {len(combos)}")
    report.append(f"  Bootstrap (var_ratio > 1)        : {n_boot} / {len(combos)}")
    report.append(f"  DGLM (alpha > 0)                 : {n_dgl} / {len(combos)}")
    report.append("")
    if n_dec or n_boot or n_dgl:
        flagged = df[(df.get('modularity_increasing_var_sig', False)) |
                     (df.get('modularity_boot_increasing_var_sig', False)) |
                     (df.get('modularity_dglm_increasing_var_sig', False))]
        report.append("Flagged cells (any sig increasing-variance signal):")
        report.append(flagged[['n_nodes', 'matrix_type', 'threshold',
                                'modularity_decile_r', 'modularity_decile_p',
                                'modularity_boot_ratio', 'modularity_boot_p_inc',
                                'modularity_dglm_alpha', 'modularity_dglm_p']].to_string(index=False))
    else:
        report.append("No cell across all 36 configurations shows significant (p < .05)")
        report.append("INCREASING modularity variance with SDS by any test.")

    rpath = project / 'reports/D2_grid_sensitivity_report.txt'
    rpath.write_text('\n'.join(report) + '\n')
    print(f"\nReport: {rpath}")
    print(f"Tables: {out_dir / 'D2_grid_results.csv'}")
    print(f"Summary: {out_dir / 'D2_grid_summary.csv'}")


if __name__ == '__main__':
    main()
