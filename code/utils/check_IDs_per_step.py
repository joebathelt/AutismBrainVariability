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
        --pgs data/PLINK_anonymised/full_pgs_scores.snp.blp.profile \
        --pgs-residuals data/pgs_residuals.csv \
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
    parser.add_argument("--pgs", required=True,
                        help="PGS scores profile file")
    parser.add_argument("--pgs-residuals", required=True,
                        help="PGS residuals CSV (from B4)")
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

    # Load PGS data
    pgs_df = pd.read_csv(args.pgs, sep=r'\s+')
    pgs_df = pgs_df[['IID', 'SCORESUM']].copy()
    pgs_df.columns = pd.Index(['Subject', 'blup_PGS'])
    pgs_ids = set(pgs_df['Subject'].values)
    print(f"PGS scores: {len(pgs_ids)} subjects")

    # Load PGS residuals (final genetic data)
    pgs_residuals_df = pd.read_csv(args.pgs_residuals)
    pgs_residuals_ids = set(pgs_residuals_df['Subject'].values)
    print(f"PGS residuals: {len(pgs_residuals_ids)} subjects")

    # Load fMRI subject IDs
    fmri_ids = set(pd.read_csv(args.ids, header=None)[0].values)
    print(f"fMRI data: {len(fmri_ids)} subjects")

    # Create retention overview using behavioural as baseline
    # sorted(), not list(): iterating a set gives an order that varies with
    # PYTHONHASHSEED, which made this file differ byte-wise between runs.
    results_df = pd.DataFrame({'Subject': sorted(behavioural_ids)})
    results_df['in_behavioural'] = True
    results_df['in_cfa'] = results_df['Subject'].isin(cfa_ids)
    results_df['in_pgs'] = results_df['Subject'].isin(pgs_ids)
    results_df['in_pgs_residuals'] = results_df['Subject'].isin(pgs_residuals_ids)
    results_df['in_fmri'] = results_df['Subject'].isin(fmri_ids)

    # Calculate subjects present in all datasets
    results_df['in_all'] = (
        results_df['in_behavioural'] &
        results_df['in_cfa'] &
        results_df['in_pgs_residuals'] &
        results_df['in_fmri']
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("RETENTION SUMMARY")
    print("=" * 60)
    print(f"Behavioural data: {results_df['in_behavioural'].sum()} subjects")
    print(f"CFA data: {results_df['in_cfa'].sum()} subjects")
    print(f"PGS data: {results_df['in_pgs'].sum()} subjects")
    print(f"PGS residuals: {results_df['in_pgs_residuals'].sum()} subjects")
    print(f"fMRI data: {results_df['in_fmri'].sum()} subjects")
    print(f"\nSubjects in ALL datasets: {results_df['in_all'].sum()} subjects")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nRetention overview saved to: {output_path}")


if __name__ == "__main__":
    main()
