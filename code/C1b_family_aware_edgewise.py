#!/usr/bin/env python
"""
C1b: Family-aware sensitivity analysis for the C1 edge-wise associations
========================================================================

C1 tests every connectivity edge with a Pearson correlation and FDR-BH
correction, treating the N = 910 participants as independent. The sample
contains twins and siblings, so this companion script re-tests the same
per-edge slopes with family cluster-robust (CR1) standard errors, using
Family_ID as the cluster, and (optionally) confirms the primary-significant
edges with a family random-intercept mixed model.

The C1 data preparation is reused verbatim by importing C1's
ConnectivityAnalysis class, so the residualised edges, outcomes and the
primary Pearson/FDR results are exactly those of the primary analysis.

Per parcellation (50/100/200 nodes) and per test
  social:       edge ~ social difficulty (z)
  pgs:          edge ~ PGS (z)
  interaction:  (edge x PGS) ~ social difficulty (z)
the script reports the number of FDR-significant edges under the i.i.d.
(primary) and cluster-robust tests, their overlap, and writes the full
per-edge statistics (which C1 itself does not persist).

Outputs
-------
  results/C1b_family_edgewise_counts.csv
  results/C1b_edge_stats_{n}nodes_{test}.csv   (one per parcellation x test)
  reports/C1b_family_aware_edgewise_report.txt
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import C1_run_univariate_fMRI_prediction as C1  # noqa: E402
from C3b_family_structure_sensitivity import _fit_mixedlm  # noqa: E402

# C1 silences all warnings at import time; restore the default filter so
# convergence problems in the mixed models are visible in the log.
warnings.simplefilter('default')

TESTS = (
    # name, X key, y key, description
    ('social', 'X_conn_residuals', 'y_social_scaled',
     'edge ~ social difficulty'),
    ('pgs', 'X_conn_residuals', 'pgs_scaled', 'edge ~ PGS'),
    ('interaction', 'X_interaction', 'y_social_scaled',
     '(edge x PGS) ~ social difficulty'),
)
PRIMARY_EXPECTED = {  # counts in the primary C1 report (guard)
    (50, 'social'): 32, (100, 'social'): 30, (200, 'social'): 18,
    (50, 'pgs'): 0, (100, 'pgs'): 1, (200, 'pgs'): 0,
    (50, 'interaction'): 0, (100, 'interaction'): 0, (200, 'interaction'): 0,
}


# %%
# =============================================================================
# VECTORISED SLOPE TESTS
# =============================================================================

def cluster_robust_slope_tests(X, y, groups):
    """Simple-regression slope tests for every column of X at once.

    For each edge e the model is y = a + b * X[:, e] + error. Returns the
    i.i.d. OLS test (identical to the Pearson correlation test) and the
    family cluster-robust test with the CR1 small-sample correction
    G/(G-1) * (N-1)/(N-2) and a t reference distribution on G-1 df.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n, n_edges = X.shape
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    Sxx = (Xc ** 2).sum(axis=0)
    b = (Xc.T @ yc) / Sxx
    resid = yc[:, None] - Xc * b
    ssr = (resid ** 2).sum(axis=0)
    se_iid = np.sqrt(ssr / (n - 2) / Sxx)
    t_iid = b / se_iid
    p_iid = 2.0 * stats.t.sf(np.abs(t_iid), n - 2)

    order = np.argsort(groups, kind='stable')
    g_sorted = groups[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    n_groups = len(starts)
    scores = Xc[order] * resid[order]
    score_by_family = np.add.reduceat(scores, starts, axis=0)
    correction = n_groups / (n_groups - 1) * (n - 1) / (n - 2)
    var_cr1 = correction * (score_by_family ** 2).sum(axis=0) / Sxx ** 2
    se_cr1 = np.sqrt(var_cr1)
    t_cr1 = b / se_cr1
    p_cr1 = 2.0 * stats.t.sf(np.abs(t_cr1), n_groups - 1)
    r = b * X.std(axis=0, ddof=1) / y.std(ddof=1)
    return {
        'slope': b, 'r': r, 'se_iid': se_iid, 't_iid': t_iid, 'p_iid': p_iid,
        'se_cr1': se_cr1, 't_cr1': t_cr1, 'p_cr1': p_cr1,
        'n': n, 'n_families': n_groups,
    }


def mixedlm_for_edges(X, y, groups, edge_idx):
    """Family random-intercept LMM y ~ x for a small set of edges."""
    out = {}
    for e in edge_idx:
        data = pd.DataFrame({'y': y, 'x': X[:, e], 'family': groups})
        fit = _fit_mixedlm('y ~ x', data, data['family'])
        res = fit['result']
        out[e] = {
            'mixedlm_slope': float(res.fe_params['x']),
            'mixedlm_se': float(res.bse_fe['x']),
            'mixedlm_p': float(res.pvalues['x']),
            'mixedlm_icc_family': fit['icc'],
            'mixedlm_converged': fit['converged'],
        }
    return out


# %%
# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='C1b: family cluster-robust re-test of the C1 edge-wise '
                    'associations')
    parser.add_argument('--project', required=True)
    parser.add_argument('--social', required=False)
    parser.add_argument('--pgs', required=False)
    parser.add_argument('--behavioural', required=False)
    parser.add_argument('--phenotypic', required=False)
    parser.add_argument('--movement', required=False)
    parser.add_argument('--ids', required=False)
    parser.add_argument('--matrices-dir', required=False)
    parser.add_argument('--motion-threshold', type=float, default=0.2)
    parser.add_argument('--parcellations', nargs='+', type=int,
                        default=[50, 100, 200])
    parser.add_argument('--mixedlm', dest='mixedlm', action='store_true',
                        default=True,
                        help='fit family random-intercept LMMs for the '
                             'primary- or CR1-significant edges (default)')
    parser.add_argument('--no-mixedlm', dest='mixedlm', action='store_false')
    parser.add_argument('--max-mixedlm-edges', type=int, default=500)
    args = parser.parse_args()

    project = Path(args.project)
    results_dir = project / 'results'
    reports_dir = project / 'reports'
    for d in (results_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    report = ["=" * 80,
              "C1b: FAMILY-AWARE SENSITIVITY ANALYSIS FOR THE C1 EDGE-WISE TESTS",
              "=" * 80, "",
              f"Motion threshold: {args.motion_threshold}",
              f"Parcellation sizes: {args.parcellations}",
              "Cluster: Family_ID; CR1 correction G/(G-1)*(N-1)/(N-2); "
              "t on G-1 df; FDR-BH at alpha = 0.05 within each test",
              ""]

    analysis = C1.ConnectivityAnalysis(
        project_folder=project, parcellation_sizes=args.parcellations,
        behavioural_file=args.behavioural, phenotypic_file=args.phenotypic,
        pgs_file=args.pgs, social_file=args.social,
        movement_file=args.movement, id_file=args.ids,
        matrices_dir=args.matrices_dir, report=[])
    pheno = C1._read_csv_retry(analysis.phenotypic_file, dtype=str)
    pheno = pheno.rename(columns={'Individual_ID': 'Subject'})
    family_map = pheno.set_index('Subject')['Family_ID']

    count_rows = []
    for n_nodes in args.parcellations:
        data = analysis.prepare_data(n_nodes, args.motion_threshold)
        merged = data['merged_df']
        fam_ids = merged['Subject'].astype(str).map(family_map)
        if fam_ids.isna().any():
            raise ValueError("Family_ID missing for some participants")
        groups = pd.factorize(fam_ids)[0]
        n = len(merged)
        n_fam = groups.max() + 1
        iu = np.triu_indices(n_nodes, k=1)   # C1's extract_upper_triangle order
        report.append(f"{n_nodes}-node parcellation: N = {n}, "
                      f"families = {n_fam}, edges = {len(iu[0])}")

        for test, x_key, y_key, desc in TESTS:
            X = data[x_key]
            y = data[y_key]
            primary = C1.perform_univariate_analysis(X, y)
            cr = cluster_robust_slope_tests(X, y, groups)
            if not np.allclose(cr['p_iid'], primary['p_values'], atol=1e-10):
                raise RuntimeError("i.i.d. slope test does not reproduce the "
                                   "primary Pearson p-values")
            sig_iid = primary['significant']
            q_iid = primary['p_corrected']
            sig_cr1, q_cr1, _, _ = multipletests(cr['p_cr1'], method='fdr_bh',
                                                 alpha=0.05)
            n_primary = int(sig_iid.sum())
            expected = PRIMARY_EXPECTED.get((n_nodes, test))
            if expected is not None and n_primary != expected:
                raise RuntimeError(
                    f"Primary count for {n_nodes}/{test} is {n_primary}, "
                    f"expected {expected} (C1 report); inputs changed?")
            n_cr1 = int(sig_cr1.sum())
            overlap = int((sig_iid & sig_cr1).sum())
            union = int((sig_iid | sig_cr1).sum())
            jaccard = overlap / union if union else np.nan

            edge_df = pd.DataFrame({
                'edge': np.arange(len(iu[0])) + 1,
                'node_i': iu[0] + 1, 'node_j': iu[1] + 1,
                'r': cr['r'], 'slope': cr['slope'],
                'se_iid': cr['se_iid'], 't_iid': cr['t_iid'],
                'p_iid': cr['p_iid'], 'q_iid': q_iid, 'sig_iid': sig_iid,
                'se_cr1': cr['se_cr1'], 't_cr1': cr['t_cr1'],
                'p_cr1': cr['p_cr1'], 'q_cr1': q_cr1, 'sig_cr1': sig_cr1,
            })
            se_ratio = float(np.median(cr['se_cr1'] / cr['se_iid']))

            n_mixed_conf = np.nan
            n_mixed_fit = 0
            if args.mixedlm:
                cand = np.flatnonzero(sig_iid | sig_cr1)
                if len(cand) > args.max_mixedlm_edges:
                    report.append(f"  {test}: {len(cand)} candidate edges "
                                  f"exceed --max-mixedlm-edges; LMMs skipped")
                    cand = np.array([], dtype=int)
                if len(cand):
                    t0 = time.time()
                    lmm = mixedlm_for_edges(X, y, groups, cand)
                    lmm_df = pd.DataFrame.from_dict(lmm, orient='index')
                    edge_df = edge_df.join(lmm_df)   # NaN for unfitted edges
                    prim_idx = np.flatnonzero(sig_iid)
                    n_mixed_fit = len(cand)
                    n_mixed_conf = int(sum(lmm[e]['mixedlm_p'] < 0.05
                                           for e in prim_idx))
                    n_nonconv = int(sum(not lmm[e]['mixedlm_converged']
                                        for e in cand))
                    report.append(f"  {test}: family LMM fitted for "
                                  f"{len(cand)} edges in {time.time() - t0:.1f} s"
                                  f" ({n_nonconv} non-converged)")

            edge_file = results_dir / f'C1b_edge_stats_{n_nodes}nodes_{test}.csv'
            edge_df.to_csv(edge_file, index=False, float_format='%.10g')

            count_rows.append({
                'n_nodes': n_nodes, 'test': test, 'model': desc,
                'n_edges': len(iu[0]), 'n': n, 'n_families': int(n_fam),
                'n_sig_primary': n_primary, 'n_sig_cr1': n_cr1,
                'n_overlap': overlap, 'jaccard': jaccard,
                'median_se_ratio_cr1_over_iid': se_ratio,
                'n_mixedlm_fitted': n_mixed_fit,
                'n_primary_sig_confirmed_mixedlm_p05': n_mixed_conf,
            })
            report.append(
                f"  {test:<12} ({desc}): FDR-significant edges primary = "
                f"{n_primary}, cluster-robust = {n_cr1}, overlap = {overlap}, "
                f"Jaccard = {jaccard if np.isfinite(jaccard) else float('nan'):.2f}; "
                f"median SE ratio (CR1 / iid) = {se_ratio:.3f}; "
                f"primary-significant edges with family-LMM p < 0.05: "
                f"{n_mixed_conf}")
        report.append("")

    counts = pd.DataFrame(count_rows)
    counts_file = results_dir / 'C1b_family_edgewise_counts.csv'
    counts.to_csv(counts_file, index=False, float_format='%.10g')

    report.append("=" * 60)
    report.append("SUMMARY OF FDR-SIGNIFICANT EDGE COUNTS (primary -> "
                  "family cluster-robust)")
    report.append("=" * 60)
    for _, r in counts.iterrows():
        report.append(f"  {int(r.n_nodes):>3} nodes  {r.test:<12} "
                      f"{int(r.n_sig_primary):>3} -> {int(r.n_sig_cr1):>3} "
                      f"(overlap {int(r.n_overlap)})")
    report.append("")
    report.append(f"Runtime: {time.time() - t_start:.1f} s")
    report.append(f"Counts saved to: {counts_file}")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    report_file = reports_dir / 'C1b_family_aware_edgewise_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
