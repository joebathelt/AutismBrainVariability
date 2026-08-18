# %%
"""
CONSORT-style Participant Flow Diagram
=======================================

Produces a LaTeX/TikZ flow diagram tracking how many participants survive
each filtering step of the pipeline:

    HCP cohort (n=1134)
        |-- excluded: missing rs-fMRI metadata
        v
    After screening (n=1025)
        |-- excluded: failed plinkQC heterozygosity, missing connectivity
        |             matrix, or excessive head motion
        v
    rs-fMRI analysis sample (n=910)

Note: A1's `HasGT` flag indicates HCP genotyped a subject, not that the
genotypes passed QC. The plinkQC heterozygosity failures (B1) that
survive A1 only get filtered out when the downstream BLUP/PCA pipeline
joins on the post-QC clean .fam, hence the second exclusion box.

Outputs:
    figures/D1_consort_diagram.tex      -- standalone TikZ figure
    results/D1_consort_counts.csv        -- one row per stage
    reports/D1_create_consort_diagram_report.txt
"""

# %%
import argparse
import os
import time
from pathlib import Path

import pandas as pd


# %%
def _read_csv_retry(path, max_retries=5, delay=1.0, **kwargs):
    """Read CSV with retry logic for filesystem locking issues (mirrors C1)."""
    for attempt in range(max_retries):
        try:
            return pd.read_csv(path, **kwargs)
        except OSError as e:
            if attempt < max_retries - 1 and e.errno == 35:
                print(f"  Filesystem lock on {path}, retrying "
                      f"({attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise


def count_lines(path):
    """Count non-empty lines in a text file; returns 0 if file is missing."""
    if not Path(path).exists():
        return 0
    with open(path, 'r') as fh:
        return sum(1 for line in fh if line.strip())


def count_qc_failures(qc_dir, prefix='Neuro_Chip_anonymised'):
    """Line-count each per-individual plinkQC fail file."""
    qc_dir = Path(qc_dir)
    return {
        'sex':         count_lines(qc_dir / f'{prefix}.fail-sex.IDs'),
        'missingness': count_lines(qc_dir / f'{prefix}.fail-imiss.IDs'),
        'heterozygosity': count_lines(qc_dir / f'{prefix}.fail-het.IDs'),
        'relatedness': count_lines(qc_dir / f'{prefix}.fail-IBD.IDs'),
    }


# %%
def compute_consort_counts(project_folder, motion_threshold=0.2):
    """
    Compute every count needed for the CONSORT diagram from raw inputs.

    Returns a dict keyed by template placeholder name.
    """
    project_folder = Path(project_folder)

    # --- Inputs ---
    behavioural_file = (project_folder
                        / 'data/raw_anonymised/behavioural_data_anonymised.csv')
    phenotypic_file = (project_folder
                       / 'data/raw_anonymised/phenotypic_data_anonymised.csv')
    movement_file = (project_folder
                     / 'data/raw_anonymised/movement_data_anonymised.csv')
    subject_ids_file = (project_folder
                        / 'data/raw_anonymised/subject_ids_anonymised.txt')
    pgs_file = project_folder / 'results/pgs_residuals.csv'
    qc_dir = project_folder / 'data/plinkQC_output'

    # --- Stage 0: original cohort ---
    behavioural_df = _read_csv_retry(behavioural_file)
    behavioural_df['Subject'] = behavioural_df['Subject'].astype(str)
    n_initial = len(behavioural_df)

    # --- Stage 1: A1 screening (fMRI metadata + HasGT) ---
    phenotypic_df = _read_csv_retry(phenotypic_file)
    phenotypic_df = phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
    phenotypic_df['Subject'] = phenotypic_df['Subject'].astype(str)

    merged_a1 = pd.merge(
        behavioural_df[['Subject', '3T_RS-fMRI_Count', '3T_RS-fMRI_PctCompl']],
        phenotypic_df[['Subject', 'HasGT']],
        on='Subject', how='inner',
    )

    has_fmri = (merged_a1['3T_RS-fMRI_Count'] != 0) & \
               (merged_a1['3T_RS-fMRI_PctCompl'] != 0)
    has_gt = merged_a1['HasGT'] == True  # noqa: E712 -- pandas boolean compare

    n_after_screening = int((has_fmri & has_gt).sum())
    n_drop_screening = n_initial - n_after_screening

    # All A1 dropouts are missing rs-fMRI metadata: the HasGT flag indicates
    # HCP attempted genotyping, not that genotypes passed QC, so A1 cannot
    # exclude plinkQC failures. Confirmed empirically (no_gt_only == 0).
    n_no_fmri = int((~has_fmri).sum())

    # --- Stage 2: post-screening exclusions (heterozygosity QC + fMRI) ---
    pgs_df = _read_csv_retry(pgs_file)
    pgs_df['Subject'] = pgs_df['Subject'].astype(str)
    n_after_pgs = len(pgs_df)
    n_drop_qc_het = n_after_screening - n_after_pgs  # plinkQC het outliers

    with open(subject_ids_file, 'r') as fh:
        connectivity_ids = {line.strip() for line in fh if line.strip()}
    pgs_with_conn = pgs_df[pgs_df['Subject'].isin(connectivity_ids)].copy()
    n_after_connectivity = len(pgs_with_conn)
    n_drop_connectivity = n_after_pgs - n_after_connectivity

    movement_df = _read_csv_retry(movement_file)
    movement_df['Subject'] = movement_df['Subject'].astype(str)

    # Mirror C1's merge chain exactly so the final count matches the C1 report.
    merged_c1 = pd.merge(
        pgs_with_conn,
        movement_df[['Subject', 'Movement_RelativeRMS_mean']],
        on='Subject', how='inner',
    )
    merged_c1 = pd.merge(
        merged_c1,
        behavioural_df[['Subject', 'Gender', 'FS_IntraCranial_Vol']],
        on='Subject', how='inner',
    )
    merged_c1 = pd.merge(
        merged_c1,
        phenotypic_df[['Subject', 'Age_in_Yrs']],
        on='Subject', how='inner',
    )
    merged_c1 = merged_c1.loc[
        merged_c1['Movement_RelativeRMS_mean'] < motion_threshold]
    merged_c1 = merged_c1.dropna()
    n_after_motion = len(merged_c1)
    n_drop_motion = n_after_connectivity - n_after_motion

    n_drop_post_screening = (n_drop_qc_het
                             + n_drop_connectivity
                             + n_drop_motion)

    counts = {
        'n_initial': n_initial,
        'n_after_screening': n_after_screening,
        'n_drop_screening': n_drop_screening,
        'n_no_fmri': n_no_fmri,
        'n_drop_post_screening': n_drop_post_screening,
        'n_drop_qc_het': n_drop_qc_het,
        'n_drop_connectivity': n_drop_connectivity,
        'n_drop_motion': n_drop_motion,
        'n_after_motion': n_after_motion,
        'motion_threshold': motion_threshold,
    }
    return counts


# %%
TIKZ_TEMPLATE = r"""% AUTOGENERATED BY D1_create_consort_diagram.py -- DO NOT EDIT
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{center}
\setlength\PreviewBorder{10pt}%
\usepackage{caption}

\newcommand*{\h}{\hspace{5pt}}% indentation
\newcommand*{\hh}{\h\h}% double indentation

\begin{document}
\begin{center}
  \sffamily
  \footnotesize
  \begin{tikzpicture}[
    auto,
    block_center/.style ={rectangle, draw=black, thick, fill=white,
      text width=12em, align=center, minimum height=4em},
    block_left/.style ={rectangle, draw=black, thick, fill=white,
      text width=18em, align=left, minimum height=4em, inner sep=6pt},
    line/.style ={draw, -Latex, thick, shorten >=0pt},
  ]
    \matrix [column sep=5mm,row sep=3mm] {
      \node [block_center] (start) {Original HCP cohort: n=@n_initial@}; &
      \node [block_left] (excluded1) {Excluded: n=@n_drop_screening@ \\
        \h Missing rs-fMRI metadata: n=@n_no_fmri@
      }; \\
      \node [block_center] (step1) {After screening: n=@n_after_screening@}; &
      \node [block_left] (excluded2) {Excluded: n=@n_drop_post_screening@ \\
        \h Failed genotype QC (heterozygosity): n=@n_drop_qc_het@ \\
        \h Missing connectivity matrix: n=@n_drop_connectivity@ \\
        \h Excessive motion (RMS $\geq$ @motion_threshold@) or NA covariate: n=@n_drop_motion@
      }; \\
      \node [block_center] (step2) {rs-fMRI analysis sample: n=@n_after_motion@}; \\
    };
    % Paths
    \begin{scope}[every path/.style=line]
      \path (start) -- (excluded1);
      \path (start) -- (step1);
      \path (step1) -- (excluded2);
      \path (step1) -- (step2);
    \end{scope}
  \end{tikzpicture}
\end{center}
\end{document}
"""


def build_tikz_document(counts):
    """Render the TikZ template by replacing @key@ placeholders.

    Using simple string replacement (not str.format) avoids the
    brace-escaping minefield with LaTeX's many literal { and } characters.
    """
    out = TIKZ_TEMPLATE
    for key, value in counts.items():
        out = out.replace(f'@{key}@', str(value))
    return out


# %%
def write_counts_csv(counts, out_path):
    """Persist the per-stage counts table for traceability."""
    rows = [
        {'step': '0_original',
         'description': 'Original HCP cohort',
         'n_retained': counts['n_initial'],
         'n_excluded': '',
         'notes': ''},
        {'step': '1_screening',
         'description': 'A1: fMRI metadata + HasGT filter',
         'n_retained': counts['n_after_screening'],
         'n_excluded': counts['n_drop_screening'],
         'notes': f"missing rs-fMRI metadata={counts['n_no_fmri']}"},
        {'step': '2_analysis',
         'description': (f'B5+C1: plinkQC clean + connectivity + '
                         f'motion < {counts["motion_threshold"]}'),
         'n_retained': counts['n_after_motion'],
         'n_excluded': counts['n_drop_post_screening'],
         'notes': (f"plinkQC heterozygosity={counts['n_drop_qc_het']}, "
                   f"missing connectivity={counts['n_drop_connectivity']}, "
                   f"excessive motion or NA={counts['n_drop_motion']}")},
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_report(counts):
    """Build the human-readable text report; surfaces sanity warnings."""
    lines = [
        '=' * 80,
        'D1: CONSORT FLOW DIAGRAM REPORT',
        '=' * 80,
        '',
        'PER-STAGE COUNTS',
        '-' * 80,
        f"  Original HCP cohort:                       n={counts['n_initial']}",
        f"  After A1 screening (rs-fMRI metadata):     n={counts['n_after_screening']}  (excluded: {counts['n_drop_screening']})",
        f"    Missing rs-fMRI metadata:                n={counts['n_no_fmri']}",
        '',
        f"  rs-fMRI analysis sample:                   n={counts['n_after_motion']}  (excluded: {counts['n_drop_post_screening']})",
        f"    Failed plinkQC heterozygosity:           n={counts['n_drop_qc_het']}",
        f"    Missing connectivity matrix:             n={counts['n_drop_connectivity']}",
        f"    Excessive motion (RMS >= {counts['motion_threshold']}) or NA:    n={counts['n_drop_motion']}",
        '',
        'SANITY CHECKS',
        '-' * 80,
    ]

    expected = {
        'After screening matches A1 report (1025)':
            counts['n_after_screening'] == 1025,
        'After motion matches C1 report (910)':
            counts['n_after_motion'] == 910,
        'Pipeline arithmetic balances (initial - all_dropouts == final)':
            counts['n_initial']
            - counts['n_drop_screening']
            - counts['n_drop_post_screening']
            == counts['n_after_motion'],
    }
    for label, ok in expected.items():
        status = 'OK' if ok else 'WARNING'
        lines.append(f"  [{status}] {label}")

    lines += ['', '=' * 80, 'END OF REPORT', '=' * 80]
    return '\n'.join(lines)


# %%
def main():
    parser = argparse.ArgumentParser(
        description='Create a CONSORT-style participant flow diagram (.tex).'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--motion-threshold', type=float, default=0.2,
                        help='Motion threshold used by C1 (default: 0.2)')
    args = parser.parse_args()

    project_folder = Path(args.project)
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'
    for d in (figures_dir, reports_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    counts = compute_consort_counts(project_folder,
                                    motion_threshold=args.motion_threshold)

    tex_path = figures_dir / 'D1_consort_diagram.tex'
    with open(tex_path, 'w') as fh:
        fh.write(build_tikz_document(counts))

    csv_path = results_dir / 'D1_consort_counts.csv'
    write_counts_csv(counts, csv_path)

    report = build_report(counts)
    report_path = reports_dir / 'D1_create_consort_diagram_report.txt'
    with open(report_path, 'w') as fh:
        fh.write(report)

    print(report)
    print()
    print(f"TeX figure : {tex_path}")
    print(f"Counts CSV : {csv_path}")
    print(f"Report     : {report_path}")
    print()
    print(f"Render with: pdflatex -output-directory {figures_dir} {tex_path}")


if __name__ == '__main__':
    main()

# %%
