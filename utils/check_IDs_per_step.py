#!/usr/bin/env python3
"""
Check Subject ID Retention Across Analysis Steps
=================================================

This script tracks how many subjects are retained at each step of the
Brain Compensation analysis pipeline.

Usage:
    python check_IDs_per_step.py \
        --project /path/to/project \
        --behavioural data/behavioural_data_preprocessed.csv \
        --cfa data/cfa_factor_scores_full_sample.csv \
        --prs data/PLINK_anonymised/full_prs_scores.snp.blp.profile \
        --prs-residuals data/prs_residuals.csv \
        --ids data/subjectIDs_anonymised.txt \
        --output data/DataRetention_Overview.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check subject ID retention across analysis steps"
    )
    parser.add_argument("--project", required=True, help="Project directory path")
    parser.add_argument("--behavioural", required=True,
                        help="Preprocessed behavioural data CSV")
    parser.add_argument("--cfa", required=True,
                        help="CFA factor scores CSV")
    parser.add_argument("--prs", required=True,
                        help="PRS scores profile file")
    parser.add_argument("--prs-residuals", required=True,
                        help="PRS residuals CSV (from B4)")
    parser.add_argument("--ids", required=True,
                        help="Subject IDs file for fMRI data")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for retention overview")
    return parser.parse_args()


def main():
    """Check subject ID retention across analysis steps."""
    args = parse_args()

    print("=" * 60)
    print("SUBJECT ID RETENTION CHECK")
    print("=" * 60)

    # Load behavioural data (baseline for comparison)
    behavioural_df = pd.read_csv(args.behavioural)
    behavioural_ids = set(behavioural_df['Subject'].values)
    print(f"\nBehavioural data: {len(behavioural_ids)} subjects")

    # Load CFA data
    cfa_df = pd.read_csv(args.cfa)
    cfa_ids = set(cfa_df['Subject'].values)
    print(f"CFA factor scores: {len(cfa_ids)} subjects")

    # Load PRS data
    prs_df = pd.read_csv(args.prs, sep=r'\s+')
    prs_df = prs_df[['IID', 'SCORESUM']].copy()
    prs_df.columns = pd.Index(['Subject', 'blup_PRS'])
    prs_ids = set(prs_df['Subject'].values)
    print(f"PRS scores: {len(prs_ids)} subjects")

    # Load PRS residuals (final genetic data)
    prs_residuals_df = pd.read_csv(args.prs_residuals)
    prs_residuals_ids = set(prs_residuals_df['Subject'].values)
    print(f"PRS residuals: {len(prs_residuals_ids)} subjects")

    # Load fMRI subject IDs
    fmri_ids = set(pd.read_csv(args.ids, header=None)[0].values)
    print(f"fMRI data: {len(fmri_ids)} subjects")

    # Create retention overview using behavioural as baseline
    results_df = pd.DataFrame({'Subject': list(behavioural_ids)})
    results_df['in_behavioural'] = True
    results_df['in_cfa'] = results_df['Subject'].isin(cfa_ids)
    results_df['in_prs'] = results_df['Subject'].isin(prs_ids)
    results_df['in_prs_residuals'] = results_df['Subject'].isin(prs_residuals_ids)
    results_df['in_fmri'] = results_df['Subject'].isin(fmri_ids)

    # Calculate subjects present in all datasets
    results_df['in_all'] = (
        results_df['in_behavioural'] &
        results_df['in_cfa'] &
        results_df['in_prs_residuals'] &
        results_df['in_fmri']
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("RETENTION SUMMARY")
    print("=" * 60)
    print(f"Behavioural data: {results_df['in_behavioural'].sum()} subjects")
    print(f"CFA data: {results_df['in_cfa'].sum()} subjects")
    print(f"PRS data: {results_df['in_prs'].sum()} subjects")
    print(f"PRS residuals: {results_df['in_prs_residuals'].sum()} subjects")
    print(f"fMRI data: {results_df['in_fmri'].sum()} subjects")
    print(f"\nSubjects in ALL datasets: {results_df['in_all'].sum()} subjects")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nRetention overview saved to: {output_path}")


if __name__ == "__main__":
    main()
