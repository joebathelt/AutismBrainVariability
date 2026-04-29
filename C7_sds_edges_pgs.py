# %%
"""
C7_sds_edges_pgs.py
Exploratory: do C1's SDS-significant edges express autism PGS?

For each individual, compute the average connectivity across the edges
that C1 flagged as FDR-significant for Social_Score (full sample edge-wise
test). Then correlate that per-subject summary connectivity with autism
PGS on the genotyped subset of the C3 post-QC sample.

Three summary measures are reported:
  - mean_signed     : mean across all FDR-significant edges (signed)
  - mean_positive   : mean across edges with positive r (Social_Score)
  - mean_negative   : mean across edges with negative r (Social_Score)

For each summary, two models on the C3 inner-join PGS subset:
  M1 (primary):  blup_PGS_residuals_z ~ summary_z
  M2 (sanity):   blup_PGS_residuals_z ~ summary_z + Age_in_Yrs
                 + FS_IntraCranial_Vol + Movement_RelativeRMS_mean + Gender

PGS residuals were already adjusted for age + PC1..PC5 in B5, so M1's single
predictor model is the natural primary spec; M2 adds the C3 brain-metric
covariates as a sensitivity check.

Sample is necessarily smaller than C3 (PGS is restricted to genotyped
subjects); both Ns and the shrinkage are reported.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, zscore

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701
FIGURE_DPI = 300


def compute_per_subject_summary(matrix_file, id_file, n_nodes, edges_df):
    """For each subject, compute mean connectivity across significant edges
    (overall, positive-r, negative-r). edges_df must contain columns
    edge_idx (1-based) and r and significant."""
    mats = pd.read_csv(matrix_file, header=None, sep=r'\s+')
    if mats.shape[1] != n_nodes ** 2:
        raise ValueError(f"matrix has {mats.shape[1]} cols, expected {n_nodes**2}")
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    linear_idx = upper_i * n_nodes + upper_j
    mats_upper = mats.iloc[:, linear_idx]
    mats_upper.columns = [f'conn_{i+1}' for i in range(len(linear_idx))]
    ids = pd.read_csv(id_file, header=None)[0].tolist()
    mats_upper.index = ids
    mats_upper.index.name = 'Subject'

    sig = edges_df[edges_df['significant'].astype(bool)]
    if len(sig) == 0:
        raise ValueError("No significant edges in input edges CSV.")
    sig_idx = sig['edge_idx'].astype(int).values - 1
    pos_idx = sig.loc[sig['r'] > 0, 'edge_idx'].astype(int).values - 1
    neg_idx = sig.loc[sig['r'] < 0, 'edge_idx'].astype(int).values - 1

    sig_cols = [f'conn_{i+1}' for i in sig_idx]
    pos_cols = [f'conn_{i+1}' for i in pos_idx]
    neg_cols = [f'conn_{i+1}' for i in neg_idx]

    out = pd.DataFrame(index=mats_upper.index)
    out['mean_signed'] = mats_upper[sig_cols].mean(axis=1)
    out['mean_positive'] = mats_upper[pos_cols].mean(axis=1) if pos_cols else np.nan
    out['mean_negative'] = mats_upper[neg_cols].mean(axis=1) if neg_cols else np.nan
    out = out.reset_index()
    return out, len(sig), len(pos_idx), len(neg_idx)


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory: PGS ~ avg connectivity of SDS-significant edges'
    )
    parser.add_argument('--project', required=True)
    parser.add_argument('--c3-results', required=True,
                        help='C3 main results CSV (post-QC sample with covariates)')
    parser.add_argument('--pgs-residuals', required=True,
                        help='B5 output: pgs_residuals.csv (Subject, blup_PGS_residuals)')
    parser.add_argument('--edges', required=True,
                        help='C1 edge stats CSV (e.g. C1_social_univariate_edges_100.csv)')
    parser.add_argument('--matrices-dir', required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--n-nodes', type=int, default=100,
                        help='Parcellation size (default 100; must match --edges file)')
    parser.add_argument('--matrix-type', default='netmats2',
                        help='Connectivity matrix file basename (default netmats2, '
                             'matching C1)')
    args = parser.parse_args()

    project = Path(args.project)
    figures_dir = project / 'figures'
    reports_dir = project / 'reports'
    results_dir = project / 'results'
    for d in (figures_dir, reports_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 80,
        "C7: PGS ~ AVG CONNECTIVITY OF C1's SDS-SIGNIFICANT EDGES",
        "=" * 80,
        "",
    ]

    edges_df = pd.read_csv(args.edges)
    matrix_file = (Path(args.matrices_dir)
                   / f'3T_HCP1200_MSMAll_d{args.n_nodes}_ts2/{args.matrix_type}.txt')
    summary_df, n_sig, n_pos, n_neg = compute_per_subject_summary(
        matrix_file, args.ids, args.n_nodes, edges_df
    )
    report.append(f"Edges file: {args.edges}")
    report.append(f"Connectivity: {matrix_file.name} ({args.n_nodes} nodes)")
    report.append(f"Significant edges: {n_sig} "
                  f"(positive r: {n_pos}, negative r: {n_neg})")
    report.append("")

    c3 = pd.read_csv(args.c3_results)
    pgs = pd.read_csv(args.pgs_residuals)
    n_c3 = len(c3); n_pgs = len(pgs)
    report.append(f"C3 post-QC sample: {n_c3} subjects")
    report.append(f"PGS sample: {n_pgs} subjects")

    # Coerce Subject to common type before merging.
    for df in (c3, pgs, summary_df):
        df['Subject'] = df['Subject'].astype(str)

    df = pd.merge(c3[['Subject', 'Social_Score', 'Age_in_Yrs',
                      'FS_IntraCranial_Vol', 'Movement_RelativeRMS_mean',
                      'Gender']],
                  summary_df, on='Subject')
    df = pd.merge(df,
                  pgs[['Subject', 'blup_PGS_residuals']],
                  on='Subject').dropna(subset=['blup_PGS_residuals',
                                               'mean_signed'])
    n_joint = len(df)
    shrinkage = 1 - n_joint / n_c3 if n_c3 else float('nan')
    report.append(f"Joint (C3 inner-join PGS, with summary_signed): "
                  f"{n_joint} subjects (shrinkage from C3: {shrinkage:.1%})")
    report.append("")

    df['blup_PGS_residuals_z'] = zscore(df['blup_PGS_residuals'])

    rows = []
    summaries = ['mean_signed', 'mean_positive', 'mean_negative']
    for summary in summaries:
        if df[summary].isna().all():
            report.append(f"--- {summary}: no edges available, skipping ---")
            continue
        sub = df.dropna(subset=[summary]).copy()
        sub[f'{summary}_z'] = zscore(sub[summary])

        report.append("=" * 80)
        report.append(f"Summary: {summary}  (n = {len(sub)})")
        report.append("=" * 80)

        # Pearson r — direct correlation
        r, p = pearsonr(sub[f'{summary}_z'], sub['blup_PGS_residuals_z'])
        report.append(f"Pearson r({summary}_z, PGS_z) = {r:+.4f}, p = {p:.4e}")

        # M1: simple OLS
        m1 = smf.ols(f'blup_PGS_residuals_z ~ {summary}_z', data=sub).fit()
        beta1 = m1.params[f'{summary}_z']
        se1 = m1.bse[f'{summary}_z']
        p1 = m1.pvalues[f'{summary}_z']
        ci1_lo, ci1_hi = m1.conf_int().loc[f'{summary}_z']
        report.append(f"M1: PGS_z ~ {summary}_z")
        report.append(f"  beta = {beta1:+.4f} (SE = {se1:.4f}), "
                      f"p = {p1:.4e}, 95% CI = [{ci1_lo:+.4f}, {ci1_hi:+.4f}]")
        report.append(f"  R^2 = {m1.rsquared:.4f}, n = {int(m1.nobs)}")

        # M2: with C3 covariates
        m2 = smf.ols(
            f'blup_PGS_residuals_z ~ {summary}_z + Age_in_Yrs '
            f'+ FS_IntraCranial_Vol + Movement_RelativeRMS_mean + C(Gender)',
            data=sub,
        ).fit()
        beta2 = m2.params[f'{summary}_z']
        se2 = m2.bse[f'{summary}_z']
        p2 = m2.pvalues[f'{summary}_z']
        ci2_lo, ci2_hi = m2.conf_int().loc[f'{summary}_z']
        report.append(f"M2: PGS_z ~ {summary}_z + age + ICV + motion + Gender")
        report.append(f"  beta = {beta2:+.4f} (SE = {se2:.4f}), "
                      f"p = {p2:.4e}, 95% CI = [{ci2_lo:+.4f}, {ci2_hi:+.4f}]")
        report.append(f"  R^2 = {m2.rsquared:.4f}, n = {int(m2.nobs)}")
        report.append("")

        rows.append({
            'summary': summary, 'n_edges': n_sig if summary == 'mean_signed'
                                          else (n_pos if summary == 'mean_positive'
                                                else n_neg),
            'n_subjects': len(sub),
            'pearson_r': r, 'pearson_p': p,
            'm1_beta': beta1, 'm1_se': se1, 'm1_p': p1,
            'm1_ci_lower': ci1_lo, 'm1_ci_upper': ci1_hi,
            'm1_r2': m1.rsquared,
            'm2_beta': beta2, 'm2_se': se2, 'm2_p': p2,
            'm2_ci_lower': ci2_lo, 'm2_ci_upper': ci2_hi,
            'm2_r2': m2.rsquared,
            'shrinkage_from_c3': shrinkage,
        })

    out_csv = results_dir / 'C7_sds_edges_pgs.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    report.append(f"Saved: {out_csv}")

    # 3-panel figure: scatter for each summary
    fig, axes = plt.subplots(1, 3, figsize=(180 * mm2inches, 60 * mm2inches),
                             dpi=FIGURE_DPI)
    titles = {
        'mean_signed': f'All sig ({n_sig})',
        'mean_positive': f'r>0 sig ({n_pos})',
        'mean_negative': f'r<0 sig ({n_neg})',
    }
    for ax, summary in zip(axes, summaries):
        if df[summary].isna().all():
            ax.text(0.5, 0.5, 'no edges',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(titles[summary]); ax.axis('off'); continue
        sub = df.dropna(subset=[summary]).copy()
        sub[f'{summary}_z'] = zscore(sub[summary])
        ax.scatter(sub[f'{summary}_z'], sub['blup_PGS_residuals_z'],
                   s=10, alpha=0.5, color='#3a3a3a', edgecolors='none')
        z = np.polyfit(sub[f'{summary}_z'], sub['blup_PGS_residuals_z'], 1)
        xs = np.linspace(sub[f'{summary}_z'].min(), sub[f'{summary}_z'].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color='#c84e3b', linewidth=1.2)
        r, p = pearsonr(sub[f'{summary}_z'], sub['blup_PGS_residuals_z'])
        ax.text(0.05, 0.95, f'r = {r:+.3f}\np = {p:.3g}\nn = {len(sub)}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_xlabel(f'{summary} [z]')
        ax.set_ylabel('Autism PGS residuals [z]')
        ax.set_title(titles[summary])
    sns.despine()
    plt.tight_layout()
    out_fig = figures_dir / 'C7_sds_edges_pgs_scatter.png'
    fig.savefig(out_fig, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    report.append(f"Saved: {out_fig}")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    out_report = reports_dir / 'C7_sds_edges_pgs_report.txt'
    out_report.write_text('\n'.join(report) + '\n')
    print(f"Report: {out_report}")
    print(f"Results: {out_csv}")
    print(f"Figure: {out_fig}")


if __name__ == '__main__':
    main()
