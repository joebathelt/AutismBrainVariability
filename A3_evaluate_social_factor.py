import sys
import argparse

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--behavioural', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--factor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    if not os.path.isdir(project_folder / 'reports/'):
        os.mkdir(project_folder / 'reports/')
    if not os.path.isdir(project_folder / 'results/'):
        os.mkdir(project_folder / 'results/')

    # Initialize report
    report = []
    report.append("=" * 80)
    report.append("A3: SOCIAL FACTOR EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Load data
    factor_df = pd.read_csv(args.factor)
    behaviour_df = pd.read_csv(args.behavioural)

    report.append("DATA LOADING:")
    report.append("-" * 80)
    report.append(f"Factor scores shape: {factor_df.shape}")
    report.append(f"Behavioural data shape: {behaviour_df.shape}")
    report.append("")

    merged_df = pd.merge(factor_df, behaviour_df, on='Subject', how='left')
    report.append(f"Merged dataset shape: {merged_df.shape}")
    report.append("")

    # Calculate correlations
    report.append("CORRELATIONS WITH COGNITIVE MEASURES:")
    report.append("-" * 80)

    correlations = {}
    for var in ['CogFluidComp_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj', 'CardSort_AgeAdj']:
        temp_df = merged_df.dropna(subset=['Social_Score', var]).copy().dropna()
        corr, p_value = pearsonr(temp_df['Social_Score'], temp_df[var])
        correlations[var] = {'correlation': corr, 'p_value': p_value, 'n': len(temp_df)}

        # Add to report with significance markers
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        report.append(f"{var}: r = {corr:.3f}, p = {p_value:.3g}, n = {len(temp_df)} {sig}")

    report.append("")

    correlations_df = pd.DataFrame(correlations).transpose()
    correlations_df['correlation'] = correlations_df['correlation'].round(3)
    correlations_df['p_value_numeric'] = correlations_df['p_value']  # Keep numeric for plotting
    correlations_df['p_value'] = correlations_df['p_value'].apply(lambda x: f'{x:.3g}')
    correlations_df['n'] = correlations_df['n'].astype(int)

    report.append("SUMMARY TABLE:")
    report.append("-" * 80)
    report.append(correlations_df[['correlation', 'p_value', 'n']].to_string())
    report.append("")

    # Save CSV to results folder
    csv_file = project_folder / 'results/A3_social_factor_correlations.csv'
    correlations_df.to_csv(csv_file, index=True, header=True)
    report.append(f"Correlation table saved to: {csv_file}")
    report.append("")

    # Create visualization
    fig, ax = plt.subplots(figsize=(120*mm2inches, 80*mm2inches), dpi=300)

    vars_sorted = correlations_df.sort_values('correlation', ascending=False).index
    colors = ['steelblue' if p < 0.05 else 'lightgray'
              for p in correlations_df.loc[vars_sorted, 'p_value_numeric']]

    ax.barh(range(len(vars_sorted)),
            correlations_df.loc[vars_sorted, 'correlation'],
            color=colors)
    ax.set_yticks(range(len(vars_sorted)))
    ax.set_yticklabels([v.replace('_AgeAdj', '') for v in vars_sorted])
    ax.set_xlabel('Correlation with Social Score')
    ax.set_title('Cognitive Measures vs Social Factor')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='p < 0.05'),
        Patch(facecolor='lightgray', label='p $\\geq$ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()

    report.append(f"Visualization saved to: {args.output}")
    report.append("")

    # Write text report
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_file = project_folder / 'reports/A3_evaluate_social_factor_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"Analysis complete. Report saved to: {report_file}")

if __name__ == "__main__":
    main()



