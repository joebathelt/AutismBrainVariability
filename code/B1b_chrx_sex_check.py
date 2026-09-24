#!/usr/bin/env python
"""
B1b: X-chromosome heterozygosity (F statistic) against phenotypic sex
=====================================================================

Companion to the B1 genotype QC. Runs PLINK's --check-sex on the raw
(prefiltered) genotypes and reports the distribution of the X-chromosome
inbreeding coefficient F against the sex recorded in the HCP demographics
table, the number of individuals whose genotype-inferred sex is discordant
or ambiguous, and how they were handled by the pipeline.

Important pitfall documented here: every HCP participant carries parental
IDs in the FAM file, so PLINK sees no founders. PLINK estimates allele
frequencies from founders only unless --make-founders / --nonfounders is
given; without it the female X heterozygosity is compared against a
mis-estimated expectation and every female receives F ~ 0.5 ("ambiguous").
That artefact is what removed almost all females in an earlier version of
the pipeline. The corrected check (--make-founders, as used for every other
frequency-based step in B1) is the primary result; the founders-only run is
reported alongside to document the artefact.

Outputs
-------
  results/B1b_chrx_sexcheck.csv          per-individual F (corrected and
                                          founders-only), FAM sex,
                                          demographic sex, inferred sex, flag
  figures/B1b_chrx_F_by_sex.png          F distributions by phenotypic sex
  reports/B1b_chrx_sex_check_report.txt

Usage
-----
  python B1b_chrx_sex_check.py --project /path --bfile data/.../Neuro_Chip_anonymised \
      --demographics data/raw_anonymised/behavioural_data_anonymised.csv \
      --phenotypic data/raw_anonymised/phenotypic_data_anonymised.csv \
      --sex-check-method demographics
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MALE_TH = 0.8      # PLINK --check-sex defaults
FEMALE_TH = 0.2
SEX_COLORS = {'F': '#2a78d6', 'M': '#eb6834'}
mm2inches = 0.0393701
FIGURE_DPI = 300


def run_check_sex(plink, bfile, out_prefix, make_founders, log):
    cmd = [plink, '--bfile', str(bfile), '--check-sex',
           '--out', str(out_prefix)]
    if make_founders:
        cmd.insert(3, '--make-founders')
    res = subprocess.run(cmd, capture_output=True, text=True)
    log.append(f"$ {' '.join(cmd)}")
    if res.returncode != 0:
        raise RuntimeError(f"plink --check-sex failed:\n{res.stdout}\n{res.stderr}")
    sexcheck = pd.read_csv(f"{out_prefix}.sexcheck", sep=r'\s+', dtype={'FID': str, 'IID': str})
    scanned = [l for l in res.stdout.splitlines() if '--check-sex:' in l]
    log += [f"  {l.strip()}" for l in scanned]
    return sexcheck


def infer_sex(f):
    return np.where(f > MALE_TH, 'M', np.where(f < FEMALE_TH, 'F', 'ambiguous'))


def main():
    ap = argparse.ArgumentParser(description='B1b: chrX F statistic vs phenotypic sex')
    ap.add_argument('--project', required=True)
    ap.add_argument('--bfile', required=True, help='PLINK prefix of the genotypes')
    ap.add_argument('--demographics', required=True,
                    help='CSV with Subject and Gender (phenotypic sex)')
    ap.add_argument('--phenotypic', required=False,
                    help='phenotypic CSV with Individual_ID, ZygosityGT (optional)')
    ap.add_argument('--sex-check-method', default='demographics',
                    choices=['demographics', 'chrx'],
                    help='the B1 rule actually applied (for the handling statement)')
    ap.add_argument('--plink', default='plink')
    ap.add_argument('--workdir', default=None,
                    help='where PLINK writes its files (default results/B1b_plink)')
    args = ap.parse_args()

    project = Path(args.project)
    results_dir = project / 'results'
    figures_dir = project / 'figures'
    reports_dir = project / 'reports'
    workdir = Path(args.workdir) if args.workdir else results_dir / 'B1b_plink'
    for d in (results_dir, figures_dir, reports_dir, workdir):
        d.mkdir(parents=True, exist_ok=True)
    if shutil.which(args.plink) is None:
        raise SystemExit(f"plink executable not found: {args.plink}")

    t0 = time.time()
    report = ["=" * 80, "B1b: X-CHROMOSOME HETEROZYGOSITY (F) AGAINST PHENOTYPIC SEX",
              "=" * 80, "", f"Genotypes: {args.bfile}",
              f"Phenotypic sex: {args.demographics} (column Gender)",
              f"PLINK --check-sex thresholds: F > {MALE_TH} male, F < {FEMALE_TH} female",
              f"B1 sex-check rule applied in the pipeline: {args.sex_check_method}", ""]

    log = []
    corrected = run_check_sex(args.plink, args.bfile, workdir / 'corrected',
                              make_founders=True, log=log)
    naive = run_check_sex(args.plink, args.bfile, workdir / 'founders_only',
                          make_founders=False, log=log)
    report += ["PLINK calls:"] + log + [""]

    demo = pd.read_csv(args.demographics, usecols=['Subject', 'Gender'], dtype=str)
    df = corrected[['FID', 'IID', 'PEDSEX', 'SNPSEX', 'STATUS', 'F']].merge(
        naive[['IID', 'F']].rename(columns={'F': 'F_founders_only'}), on='IID', how='left')
    df = df.merge(demo.rename(columns={'Subject': 'IID', 'Gender': 'phenotypic_sex'}),
                  on='IID', how='left')
    df['fam_sex'] = df['PEDSEX'].map({1: 'M', 2: 'F'})
    df['inferred_sex'] = infer_sex(df['F'].values)
    df['inferred_sex_founders_only'] = infer_sex(df['F_founders_only'].values)
    df['fam_vs_phenotypic_mismatch'] = df['fam_sex'] != df['phenotypic_sex']
    df['discordant'] = ((df['inferred_sex'] != 'ambiguous')
                        & (df['inferred_sex'] != df['phenotypic_sex']))
    df['ambiguous'] = df['inferred_sex'] == 'ambiguous'
    df['flagged'] = df['discordant'] | df['ambiguous']
    if args.sex_check_method == 'chrx':
        df['excluded_by_b1'] = df['flagged']
        handling = ("B1 (chrX mode) removes every flagged individual "
                    "(discordant or ambiguous) before downstream analyses.")
    else:
        df['excluded_by_b1'] = df['fam_vs_phenotypic_mismatch']
        handling = ("B1 (demographics mode) removes individuals whose FAM sex "
                    "disagrees with the demographic sex; the chrX F statistic "
                    "is reported here but not used for exclusion. Flagged "
                    "individuals are therefore retained, and their influence "
                    "is assessed in C3b by refitting the primary models "
                    "without them.")
    if args.phenotypic:
        ph = pd.read_csv(args.phenotypic, usecols=['Individual_ID', 'Family_ID', 'ZygosityGT'],
                         dtype=str).rename(columns={'Individual_ID': 'IID'})
        ph['ZygosityGT'] = ph['ZygosityGT'].fillna('').str.strip()
        df = df.merge(ph, on='IID', how='left')

    # ---- report
    n_missing = int(df['phenotypic_sex'].isna().sum())
    report += [f"Genotyped individuals: {len(df)}; without phenotypic sex: {n_missing}",
               f"FAM sex vs phenotypic sex mismatches: {int(df['fam_vs_phenotypic_mismatch'].sum())}",
               "", "chrX F statistic by phenotypic sex (corrected: --make-founders):"]
    for label, col in (('corrected', 'F'), ('founders-only artefact', 'F_founders_only')):
        if label != 'corrected':
            report += ["", f"chrX F statistic by phenotypic sex ({label}):"]
        for sex, sub in df.groupby('phenotypic_sex'):
            f = sub[col]
            inf = infer_sex(f.values)
            report.append(
                f"  {sex}: n = {len(f)}, mean = {f.mean():.3f}, SD = {f.std(ddof=1):.3f}, "
                f"median = {f.median():.3f}, range = [{f.min():.3f}, {f.max():.3f}]; "
                f"inferred F/M/ambiguous = {int((inf == 'F').sum())}/"
                f"{int((inf == 'M').sum())}/{int((inf == 'ambiguous').sum())}")
    report += ["",
               f"Discordant (inferred sex opposite to phenotypic sex): {int(df['discordant'].sum())}",
               f"Ambiguous ({FEMALE_TH} <= F <= {MALE_TH}): {int(df['ambiguous'].sum())}",
               f"Flagged in total: {int(df['flagged'].sum())}",
               f"Founders-only artefact would flag: {int((df['inferred_sex_founders_only'] != df['phenotypic_sex']).sum())}",
               "", "Flagged individuals (corrected F):"]
    cols = ['IID', 'phenotypic_sex', 'fam_sex', 'inferred_sex', 'F', 'F_founders_only']
    if 'ZygosityGT' in df:
        cols.append('ZygosityGT')
    for _, r in df[df['flagged']].sort_values('F').iterrows():
        report.append("  " + ", ".join(f"{c} = {r[c]:.3f}" if isinstance(r[c], float)
                                       else f"{c} = {r[c]}" for c in cols))
    report += ["", "Handling:", f"  {handling}",
               f"  Individuals excluded by B1 on sex grounds: {int(df['excluded_by_b1'].sum())}"]

    # ---- figure: F by phenotypic sex, corrected vs founders-only
    fig, axes = plt.subplots(1, 2, figsize=(180 * mm2inches, 70 * mm2inches),
                             dpi=FIGURE_DPI, sharey=False)
    for ax, (col, title) in zip(axes, (('F', 'Allele frequencies from all samples (--make-founders)'),
                                        ('F_founders_only', 'Founders-only frequencies (PLINK default; no founders in HCP FAM)'))):
        bins = np.linspace(-0.3, 1.05, 55)
        for sex in ('F', 'M'):
            vals = df.loc[df['phenotypic_sex'] == sex, col].dropna()
            ax.hist(vals, bins=bins, color=SEX_COLORS[sex], alpha=0.85,
                    label=f"phenotypic {'female' if sex == 'F' else 'male'} (n = {len(vals)})",
                    edgecolor='white', linewidth=0.3)
        for th in (FEMALE_TH, MALE_TH):
            ax.axvline(th, color='#666', linewidth=0.8, linestyle='--')
        ax.set_xlabel('chrX inbreeding coefficient F', fontsize=7)
        ax.set_ylabel('participants', fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6, frameon=True, framealpha=1.0, edgecolor='none')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig_file = figures_dir / 'B1b_chrx_F_by_sex.png'
    fig.savefig(fig_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    csv_file = results_dir / 'B1b_chrx_sexcheck.csv'
    df.to_csv(csv_file, index=False, float_format='%.6g')
    report += ["", f"Runtime: {time.time() - t0:.1f} s", f"Per-individual table: {csv_file}",
               f"Figure: {fig_file}", "=" * 80, "END OF REPORT", "=" * 80]
    report_file = reports_dir / 'B1b_chrx_sex_check_report.txt'
    report_file.write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
