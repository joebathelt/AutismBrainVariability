"""
D1_sensitivity_analyses.py

SDS-stratified heteroscedasticity sensitivity sweep.

Runs the full C3b heteroscedasticity test battery (BP, White, quantile-IQ,
decile trend, balanced-bootstrap, DGLM) under multiple analytic specifications
to determine whether the SDS-driven variability finding is robust to choices
about cohort and covariate handling.

Cells (all use Social_Score / SDS as the continuous predictor; PGS is not
loaded):
  C0  baseline                  : M+F, motion<0.2, brain metrics residualised on age+ICV+motion+Gender
  C1  motion strict             : motion<0.15, otherwise C0
  C2  motion lax                : motion<0.30, otherwise C0
  C3  males only                : M-only sanity check
  C4  females only              : F-only sanity check
  C5  no brain residualisation  : raw modularity / global_efficiency
  C6  age-restricted (<= 30)    : drop subjects > 30 years, otherwise C0

For each cell we run BP, White, quantile-IQ (Q90 vs Q10), decile-trend,
balanced-bootstrap, and DGLM, then apply Bonferroni + BH-FDR within the cell.
"""

import argparse
from pathlib import Path

import bct
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.multitest import multipletests

PROJECT = Path('/home/jmbathe/Documents/1_Projects/BrainCompensation')
N_NODES = 100
NETWORK_THRESHOLD = 0.2
RNG_SEED = 1729


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_raw_inputs():
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
    social = pd.read_csv(PROJECT / 'results/cfa_factor_scores_full_sample.csv')
    print(f"  behaviour: {len(behaviour)}, phenotypic: {len(phenotypic)}, "
          f"movement: {len(movement)}, social: {len(social)}")

    return dict(conn=conn, behaviour=behaviour, phenotypic=phenotypic,
                movement=movement, social=social)


def compute_network_metrics_all(conn, partition_path):
    """Compute raw modularity & global_efficiency for every connectivity subject."""
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
        mat_pos = mat.copy(); mat_pos[mat_pos < 0] = 0
        ge = nx.global_efficiency(nx.from_numpy_array(mat_pos))
        out.append({'Subject': subject_id, 'modularity': modularity,
                    'global_efficiency': ge})
    return pd.DataFrame(out).set_index('Subject')


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def regress_out(y, X_df):
    X = pd.get_dummies(X_df, drop_first=True).astype(float).values
    if X.shape[1] == 0:
        return y - y.mean()
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)


# --------------------------------------------------------------------------- #
# Heteroscedasticity tests (predictor = sds_z)
# --------------------------------------------------------------------------- #

def bp_white(df, metric):
    y = df[metric].values
    X = sm.add_constant(df['sds_z'].values)
    model = sm.OLS(y, X).fit()
    bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, X)
    w_lm, w_p, _, _ = het_white(model.resid, X)
    return bp_lm, bp_p, w_lm, w_p


def quantile_iq(df, metric, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    y = df[metric].values
    X = sm.add_constant(df['sds_z'].values)
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
        return np.nan, np.nan, [], []
    r, p = stats.pearsonr(centres, sds)
    return r, p, centres, sds


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
    boot_r = np.empty(n_boot)
    boot_ratio = np.empty(n_boot)
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
        mu = X_mu @ bm
        log_var = np.clip(X_s @ bs, -30, 30)
        return 0.5 * np.sum(log_var + (y - mu) ** 2 / np.exp(log_var))

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
    lr_stat = float(2 * (red.fun - full.fun))
    df_diff = X_sigma.shape[1] - 1
    p_lrt = float(stats.chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
    alpha = float(full.x[p_mu + 1]) if X_sigma.shape[1] > 1 else np.nan
    return alpha, lr_stat, p_lrt


# --------------------------------------------------------------------------- #
# Cell runner
# --------------------------------------------------------------------------- #

def assemble_cell_df(spec, raw, network_df):
    df = network_df.reset_index().merge(
        raw['social'][['Subject', 'Social_Score']], on='Subject'
    )
    df = df.merge(raw['behaviour'][['Subject', 'Gender', 'FS_IntraCranial_Vol']],
                  on='Subject')
    df = df.merge(raw['phenotypic'][['Subject', 'Age_in_Yrs']], on='Subject')
    df = df.merge(raw['movement'][['Subject', 'Movement_RelativeRMS_mean']],
                  on='Subject')

    # Sex filter
    if spec['sex_filter'] == 'M':
        df = df[df['Gender'] == 'M']
    elif spec['sex_filter'] == 'F':
        df = df[df['Gender'] == 'F']
    # 'both' = no sex filter

    # Motion threshold
    df = df[df['Movement_RelativeRMS_mean'] < spec['motion_threshold']]

    # Optional age restriction
    if spec.get('max_age'):
        df = df[df['Age_in_Yrs'] <= spec['max_age']]

    # Optional ICV trim (drop top/bottom q tails)
    if spec.get('icv_trim'):
        q = spec['icv_trim']
        lo, hi = df['FS_IntraCranial_Vol'].quantile([q, 1 - q])
        df = df[(df['FS_IntraCranial_Vol'] >= lo) & (df['FS_IntraCranial_Vol'] <= hi)]

    df = df.dropna(subset=['Age_in_Yrs', 'FS_IntraCranial_Vol',
                           'Movement_RelativeRMS_mean', 'Social_Score',
                           'modularity', 'global_efficiency'])
    df = df.copy()
    df['sds_z'] = stats.zscore(df['Social_Score'].values)
    return df


def run_cell(spec, raw, network_df, report):
    df = assemble_cell_df(spec, raw, network_df)
    n = len(df)
    report.append("")
    report.append("=" * 78)
    report.append(f"CELL {spec['name']}: {spec['description']}")
    report.append("=" * 78)
    report.append(f"  sex filter         : {spec['sex_filter']}")
    report.append(f"  motion threshold   : {spec['motion_threshold']}")
    if spec.get('max_age'):
        report.append(f"  age cap            : <= {spec['max_age']}")
    if spec.get('icv_trim'):
        report.append(f"  ICV trim           : drop tails at q={spec['icv_trim']}")
    report.append(f"  brain residualised : {spec['residualise_brain']}")
    report.append(f"  N subjects         : {n}")
    if n < 50:
        report.append(f"  *** WARNING: very small cell ({n} subjects)")
        return None

    # Build covariate set for brain metrics (when residualising)
    covariates = ['Age_in_Yrs', 'FS_IntraCranial_Vol', 'Movement_RelativeRMS_mean']
    if spec['sex_filter'] == 'both':
        covariates.append('Gender')
    if spec['residualise_brain']:
        report.append(f"  brain covariates   : {', '.join(covariates)}")
        cov_df = df[covariates].copy()
        df['modularity_test'] = regress_out(df['modularity'].values, cov_df)
        df['global_efficiency_test'] = regress_out(df['global_efficiency'].values, cov_df)
    else:
        df['modularity_test'] = df['modularity'].values
        df['global_efficiency_test'] = df['global_efficiency'].values

    z = df['sds_z'].values
    n_low = int((z <= -1).sum()); n_high = int((z >= 1).sum())
    report.append(f"  SDS tail sizes     : low={n_low}, high={n_high}")

    metrics = ('modularity_test', 'global_efficiency_test')
    cell_results = {'name': spec['name'], 'description': spec['description'],
                    'n': n, 'n_low': n_low, 'n_high': n_high}

    p_pool, p_labels = [], []
    for metric in metrics:
        m_label = metric.replace('_test', '')
        bp_lm, bp_p, w_lm, w_p = bp_white(df, metric)
        sl_range, sl_diff, qt_t, qt_p, _ = quantile_iq(df, metric)
        d_r, d_p, _, _ = decile_trend(df, metric)
        b_r, b_p, b_ratio = balanced_bootstrap(df, metric, n_boot=1000)

        Xmu_cols = ['sds_z'] + [c for c in covariates if c != 'Gender']
        if 'Gender' in covariates:
            df['Gender_M'] = (df['Gender'] == 'M').astype(int)
            Xmu_cols.append('Gender_M')
        Xmu = sm.add_constant(df[Xmu_cols].values.astype(float))
        Xsig = sm.add_constant(df[['sds_z']].values.astype(float))
        try:
            alpha, lr_stat, lr_p = fit_dglm(df[metric].values, Xmu, Xsig)
        except Exception:
            alpha, lr_stat, lr_p = np.nan, np.nan, np.nan

        report.append("")
        report.append(f"  ---- {m_label} ----")
        report.append(f"    BP   : LM={bp_lm:7.3f}  p={bp_p:.4f}")
        report.append(f"    White: LM={w_lm:7.3f}  p={w_p:.4f}")
        report.append(f"    Quant: range={sl_range:8.5f}  diff={sl_diff:8.5f}  "
                      f"t={qt_t:6.3f}  p={qt_p:.4f}")
        report.append(f"    Decile trend: r={d_r:+.3f}  p={d_p:.4f}")
        report.append(f"    Bootstrap: r_obs={b_r:+.3f}  p_inc={b_p:.4f}  "
                      f"var_ratio_high/low={b_ratio:.3f}")
        report.append(f"    DGLM : alpha_SDS={alpha:+.4f}  LR={lr_stat:7.3f}  p={lr_p:.4f}")

        cell_results.update({
            f'{m_label}_bp_p': bp_p, f'{m_label}_white_p': w_p,
            f'{m_label}_quant_p': qt_p, f'{m_label}_quant_range': sl_range,
            f'{m_label}_decile_r': d_r, f'{m_label}_decile_p': d_p,
            f'{m_label}_boot_r': b_r, f'{m_label}_boot_p_inc': b_p,
            f'{m_label}_boot_ratio': b_ratio,
            f'{m_label}_dglm_alpha': alpha,
            f'{m_label}_dglm_lr': lr_stat, f'{m_label}_dglm_p': lr_p,
        })
        for nm, p in [('bp', bp_p), ('white', w_p), ('quant', qt_p),
                      ('decile', d_p), ('boot', b_p), ('dglm', lr_p)]:
            if np.isfinite(p):
                p_pool.append(p)
                p_labels.append(f'{m_label}_{nm}')

    if p_pool:
        bonf, _, _, _ = multipletests(p_pool, alpha=0.05, method='bonferroni')
        fdr, fdr_p, _, _ = multipletests(p_pool, alpha=0.05, method='fdr_bh')
        cell_results.update({
            'n_sig_unc': int(np.sum(np.array(p_pool) < 0.05)),
            'n_sig_fdr': int(np.sum(fdr)),
            'n_sig_bonf': int(np.sum(bonf)),
        })
        report.append("")
        report.append(f"  Multiple-comparison correction across {len(p_pool)} tests:")
        report.append(f"    uncorrected p<.05 : {cell_results['n_sig_unc']}")
        report.append(f"    FDR-BH    p<.05   : {cell_results['n_sig_fdr']}")
        report.append(f"    Bonferroni p<.05  : {cell_results['n_sig_bonf']}")
        if cell_results['n_sig_fdr']:
            sig_idx = np.where(fdr)[0]
            report.append("    FDR-significant: " + "; ".join(
                f"{p_labels[i]}(p_raw={p_pool[i]:.4f},p_fdr={fdr_p[i]:.4f})"
                for i in sig_idx
            ))
    return cell_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=str(PROJECT))
    args = parser.parse_args()
    project = Path(args.project)

    out_dir = project / 'results/D1_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_inputs()
    network_df = compute_network_metrics_all(
        raw['conn'], project / 'results/C2b_selected_partition.csv'
    )
    network_df.to_csv(out_dir / 'network_metrics_all_subjects.csv')
    print(f"  metrics computed for {len(network_df)} subjects")

    cells = [
        dict(name='C0', description='baseline (M+F, motion<0.2, brain resid incl. Gender)',
             sex_filter='both', motion_threshold=0.2, residualise_brain=True),
        dict(name='C1', description='motion-strict (motion<0.15)',
             sex_filter='both', motion_threshold=0.15, residualise_brain=True),
        dict(name='C2', description='motion-lax (motion<0.30)',
             sex_filter='both', motion_threshold=0.30, residualise_brain=True),
        dict(name='C3', description='males only',
             sex_filter='M', motion_threshold=0.2, residualise_brain=True),
        dict(name='C4', description='females only',
             sex_filter='F', motion_threshold=0.2, residualise_brain=True),
        dict(name='C5', description='no brain-metric residualisation',
             sex_filter='both', motion_threshold=0.2, residualise_brain=False),
        dict(name='C6', description='age-restricted (<= 30 yrs)',
             sex_filter='both', motion_threshold=0.2, residualise_brain=True,
             max_age=30),
        dict(name='C7', description='ICV trim (drop top/bottom 5%)',
             sex_filter='both', motion_threshold=0.2, residualise_brain=True,
             icv_trim=0.05),
    ]

    report = []
    report.append("=" * 78)
    report.append("D1 SENSITIVITY — SDS heteroscedasticity sweep")
    report.append("=" * 78)
    report.append(f"Project: {project}")
    report.append(f"N nodes: {N_NODES}, network threshold: {NETWORK_THRESHOLD}")
    report.append("Predictor: sds_z = z(Social_Score). PGS not loaded.")

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

    table_df = pd.DataFrame(cell_results)
    table_df.to_csv(out_dir / 'D1_master_sensitivity_table.csv', index=False)

    summary_cols = ['name', 'description', 'n', 'n_low', 'n_high',
                    'modularity_decile_r', 'modularity_decile_p',
                    'modularity_boot_ratio', 'modularity_boot_p_inc',
                    'modularity_dglm_alpha', 'modularity_dglm_p',
                    'global_efficiency_decile_r', 'global_efficiency_decile_p',
                    'global_efficiency_dglm_alpha', 'global_efficiency_dglm_p',
                    'n_sig_unc', 'n_sig_fdr', 'n_sig_bonf']
    summary_cols = [c for c in summary_cols if c in table_df.columns]
    summary = table_df[summary_cols]
    summary.to_csv(out_dir / 'D1_summary_table.csv', index=False)

    report.append(""); report.append("=" * 78)
    report.append("MASTER SUMMARY TABLE (per-cell headlines)")
    report.append("=" * 78)
    with pd.option_context('display.width', 200, 'display.max_columns', 30,
                           'display.float_format', lambda x: f'{x:7.4f}'):
        report.append(summary.to_string(index=False))

    report_path = project / 'reports/D1_sensitivity_analyses_report.txt'
    report_path.write_text('\n'.join(report) + '\n')
    print(f"\nReport: {report_path}")
    print(f"Tables: {out_dir}")


if __name__ == '__main__':
    main()
