# %%
"""
C6_exploratory_sds_pgs.py
Exploratory: does the autism PGS predict SDS (Social_Score)?

Secondary test for the behaviour-stratified analysis. SDS is the primary
stratifier in C3/C3b/C5; here we ask whether higher autism PGS goes with
higher SDS (worse social performance).

The sample is the inner-join of:
  - C3's post-QC sample (results/C3_graph_theory_landscape_results.csv)
  - The genotyped subjects with BLUP-extended PGS
    (results/pgs_residuals.csv, produced by B5)

This will be smaller than the C3 sample because not all participants are
genotyped. Both Ns are reported. Treat the test as exploratory.

Models (mirror B5_evalute_BLUP_prediction.py: see lines 178-188):

  M1 (primary): Social_Score ~ blup_PGS_residuals_z
       The PGS residuals have already been adjusted for age + PC1..PC5
       upstream in B5, so this single-predictor regression carries the
       same partialling as B5's adjusted_model on the C3-filtered cohort.

  M2 (sanity, plus C3 covariates):
       Social_Score ~ blup_PGS_residuals_z + Age_in_Yrs + FS_IntraCranial_Vol
                    + Movement_RelativeRMS_mean
       Age is residualised twice (once in PGS, once in mean), but it lets
       us check whether the C3-specific brain covariates change the PGS
       coefficient at all.

Outputs:
  - results/C6_sds_pgs_regression.csv  (β, SE, p, 95% CI per model)
  - reports/C6_exploratory_sds_pgs_report.txt
  - figures/C6_sds_vs_pgs_scatter.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import zscore

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701
FIGURE_DPI = 300


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory SDS~PGS regression on the C3 post-QC sample.'
    )
    parser.add_argument('--project', required=True)
    parser.add_argument('--c3-results', required=True,
                        help='C3 main results CSV '
                             '(C3_graph_theory_landscape_results.csv)')
    parser.add_argument('--pgs-residuals', required=True,
                        help='PGS residuals CSV from B5 (Subject, '
                             'blup_PGS_residuals, blup_PGS)')
    args = parser.parse_args()

    project = Path(args.project)
    figures_dir = project / 'figures'
    reports_dir = project / 'reports'
    results_dir = project / 'results'
    for d in (figures_dir, reports_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 80,
        "C6: EXPLORATORY SDS~PGS REGRESSION",
        "=" * 80,
        "",
    ]

    c3 = pd.read_csv(args.c3_results)
    pgs = pd.read_csv(args.pgs_residuals)
    # Coerce Subject to a common type so the merge below doesn't fail when
    # one CSV stores IDs as int and the other as string.
    c3['Subject'] = c3['Subject'].astype(str)
    pgs['Subject'] = pgs['Subject'].astype(str)

    n_c3 = len(c3)
    n_pgs = len(pgs)
    report.append(f"C3 post-QC sample: {n_c3} subjects")
    report.append(f"PGS sample: {n_pgs} subjects")

    if 'Social_Score' not in c3.columns:
        raise ValueError(f"{args.c3_results} missing 'Social_Score' column.")
    if 'blup_PGS_residuals' not in pgs.columns:
        raise ValueError(f"{args.pgs_residuals} missing 'blup_PGS_residuals' column.")

    df = pd.merge(
        c3[['Subject', 'Social_Score', 'Age_in_Yrs',
            'FS_IntraCranial_Vol', 'Movement_RelativeRMS_mean']],
        pgs[['Subject', 'blup_PGS_residuals'] +
            (['blup_PGS'] if 'blup_PGS' in pgs.columns else [])],
        on='Subject',
    ).dropna()
    n_joint = len(df)
    shrinkage = 1 - n_joint / n_c3 if n_c3 else float('nan')
    report.append(f"Joint (C3 inner-join PGS): {n_joint} subjects "
                  f"(shrinkage from C3: {shrinkage:.1%})")

    df['blup_PGS_residuals_z'] = zscore(df['blup_PGS_residuals'])

    # ---------------------------------------------------------------- M1
    report.append("")
    report.append("=" * 80)
    report.append("Model 1: Social_Score ~ blup_PGS_residuals_z")
    report.append("(PGS residuals already adjusted for age + PC1..PC5 in B5)")
    report.append("=" * 80)
    m1 = smf.ols('Social_Score ~ blup_PGS_residuals_z', data=df).fit()
    rows = []
    for label, model in [('M1', m1)]:
        beta = model.params['blup_PGS_residuals_z']
        se = model.bse['blup_PGS_residuals_z']
        p = model.pvalues['blup_PGS_residuals_z']
        ci_lo, ci_hi = model.conf_int().loc['blup_PGS_residuals_z']
        report.append(f"  beta = {beta:+.4f} (SE = {se:.4f})")
        report.append(f"  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        report.append(f"  p = {p:.4e}")
        report.append(f"  R^2 = {model.rsquared:.4f}, n = {int(model.nobs)}")
        rows.append({
            'model': label, 'predictor': 'blup_PGS_residuals_z',
            'beta': beta, 'se': se, 'p': p,
            'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'r_squared': model.rsquared, 'n': int(model.nobs),
            'shrinkage_from_c3': shrinkage,
        })
    if m1.pvalues['blup_PGS_residuals_z'] < 0.05 and m1.params['blup_PGS_residuals_z'] > 0:
        report.append("  *** Higher autism PGS predicts higher SDS (worse social performance)")
    elif m1.pvalues['blup_PGS_residuals_z'] < 0.05 and m1.params['blup_PGS_residuals_z'] < 0:
        report.append("  *** Higher autism PGS predicts LOWER SDS (better social performance)")
    else:
        report.append("  No significant SDS~PGS association (exploratory test).")

    # ---------------------------------------------------------------- M2
    report.append("")
    report.append("=" * 80)
    report.append("Model 2 (sanity): Social_Score ~ blup_PGS_residuals_z "
                  "+ Age_in_Yrs + FS_IntraCranial_Vol + Movement_RelativeRMS_mean")
    report.append("=" * 80)
    m2 = smf.ols(
        'Social_Score ~ blup_PGS_residuals_z + Age_in_Yrs + FS_IntraCranial_Vol '
        '+ Movement_RelativeRMS_mean',
        data=df,
    ).fit()
    beta = m2.params['blup_PGS_residuals_z']
    se = m2.bse['blup_PGS_residuals_z']
    p = m2.pvalues['blup_PGS_residuals_z']
    ci_lo, ci_hi = m2.conf_int().loc['blup_PGS_residuals_z']
    report.append(f"  beta = {beta:+.4f} (SE = {se:.4f})")
    report.append(f"  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    report.append(f"  p = {p:.4e}")
    report.append(f"  R^2 = {m2.rsquared:.4f}, n = {int(m2.nobs)}")
    rows.append({
        'model': 'M2', 'predictor': 'blup_PGS_residuals_z',
        'beta': beta, 'se': se, 'p': p,
        'ci_lower': ci_lo, 'ci_upper': ci_hi,
        'r_squared': m2.rsquared, 'n': int(m2.nobs),
        'shrinkage_from_c3': shrinkage,
    })

    out_csv = results_dir / 'C6_sds_pgs_regression.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    report.append("")
    report.append(f"Saved: {out_csv}")

    # Scatter plot of SDS vs PGS-z with M1 line + 95% CI band
    fig, ax = plt.subplots(figsize=(95 * mm2inches, 75 * mm2inches), dpi=FIGURE_DPI)
    ax.scatter(df['blup_PGS_residuals_z'], df['Social_Score'],
               alpha=0.4, s=18, color='#3a3a3a', edgecolors='none')

    xs = np.linspace(df['blup_PGS_residuals_z'].min(),
                     df['blup_PGS_residuals_z'].max(), 100)
    pred = m1.get_prediction(pd.DataFrame({'blup_PGS_residuals_z': xs}))
    pred_sf = pred.summary_frame(alpha=0.05)
    ax.plot(xs, pred_sf['mean'], color='#c84e3b', linewidth=1.5)
    ax.fill_between(xs, pred_sf['mean_ci_lower'], pred_sf['mean_ci_upper'],
                    color='#c84e3b', alpha=0.2, linewidth=0)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Autism PGS residuals [z]')
    ax.set_ylabel('SDS (Social_Score) [z]')
    ax.set_title('C6: SDS vs autism PGS (M1 fit + 95% CI)')

    beta1 = m1.params['blup_PGS_residuals_z']
    p1 = m1.pvalues['blup_PGS_residuals_z']
    ax.text(0.04, 0.96, f'beta = {beta1:+.3f}\np = {p1:.3g}\nn = {n_joint}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    sns.despine()
    plt.tight_layout()
    out_fig = figures_dir / 'C6_sds_vs_pgs_scatter.png'
    fig.savefig(out_fig, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    report.append(f"Saved: {out_fig}")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    out_report = reports_dir / 'C6_exploratory_sds_pgs_report.txt'
    out_report.write_text('\n'.join(report) + '\n')
    print(f"Report: {out_report}")
    print(f"Results: {out_csv}")
    print(f"Figure: {out_fig}")


if __name__ == '__main__':
    main()
