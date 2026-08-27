import sys
import argparse

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701

# %%
def process_hcp_data(project_folder, behavioural_file, phenotypic_file, output_file, report_file):
    """
    Process HCP behavioural and phenotypic data.

    Steps:
    1. Load datasets
    2. Filter columns of interest
    3. Merge datasets on 'Subject'
    4. Filter participants with resting-state fMRI data
    5. Filter participants with genetic data
    6. Replace NaN values with np.nan
    7. Generate missingness report
    8. Adjust outliers in RT data
    9. Impute missing values with MICE
    10. Save the imputed dataset to a CSV file

    Parameters:
        project_folder (Path): Path to the project folder.
        behavioural_file (str): Path to the behavioural data CSV file.
        phenotypic_file (str): Path to the phenotypic data CSV file.
        output_file (str): Path to save the processed data CSV file.
        report_file (str): Path to save the processing report.

    Returns:
        pd.DataFrame: The processed and imputed dataset.
    """
    # Initialize report content
    report = []
    report.append("=" * 80)
    report.append("A1: PHENOTYPIC DATA PREPROCESSING REPORT")
    report.append("=" * 80)
    report.append("")

    # Load datasets
    behavioural_data = pd.read_csv(behavioural_file, delimiter=',')
    phenotypic_data = pd.read_csv(phenotypic_file,delimiter=',')
    report.append(f"Behavioural data shape: {behavioural_data.shape}")
    report.append(f"Phenotypic data shape: {phenotypic_data.shape}")
    report.append("")

    # Filter columns of interest
    behaviour_columns = [
        "Subject",
        "Gender",
        "Friendship_Unadj",
        "Loneliness_Unadj",
        "PercHostil_Unadj",
        "PercReject_Unadj",
        "EmotSupp_Unadj",
        "InstruSupp_Unadj",
        "Emotion_Task_Face_Median_RT",
        "Language_Task_Story_Median_RT",
        "Social_Task_TOM_Median_RT_TOM",
        "WM_Task_0bk_Median_RT",
        "ER40_CRT",
        "Language_Task_Story_Acc",
        "ER40_CR",
        "Emotion_Task_Face_Acc",
        "Social_Task_TOM_Perc_TOM",
        "3T_RS-fMRI_Count",
        "3T_RS-fMRI_PctCompl",
        "FS_IntraCranial_Vol",
        "FS_BrainSeg_Vol",
    ]
    behavioural_data = behavioural_data[behaviour_columns]

    phenotype_columns = [
        "Individual_ID",
        'Family_ID',
        "Age_in_Yrs",
        "HasGT",
        'Height',
        'Weight',
        'BPSystolic',
        'BPDiastolic',
    ]
    phenotypic_data = phenotypic_data[phenotype_columns]
    phenotypic_data = phenotypic_data.rename(columns={'Individual_ID': 'Subject'})

    # Replace empty strings with NA if they exist
    phenotypic_data[phenotypic_data == ''] = np.nan

    # Ensure 'Subject' is consistent in type
    behavioural_data['Subject'] = behavioural_data['Subject'].astype(str)
    phenotypic_data['Subject'] = phenotypic_data['Subject'].astype(str)

    # Merge datasets on 'Subject'
    merged_df = pd.merge(behavioural_data, phenotypic_data, on='Subject', how='inner')

    # Filter participants with resting-state fMRI data
    columns_of_interest = ['3T_RS-fMRI_Count', '3T_RS-fMRI_PctCompl']
    merged_df = merged_df[(merged_df[columns_of_interest] != 0.0).all(axis=1)]
    report.append(f"Data shape after fMRI filtering: {merged_df.shape}")

    # Filter participants with genetic data
    merged_df = merged_df[merged_df['HasGT'] == True]
    report.append(f"Data shape after genetic data filtering: {merged_df.shape}")
    report.append("")

    # Replace NaN values with np.nan
    final_df = merged_df.replace({pd.NA: np.nan})

    # Generate missingness report
    missing_report = pd.DataFrame({
        'Missing Values': final_df.isna().sum(),
        'Percentage Missing': final_df.isna().mean() * 100
    }).sort_values(by='Percentage Missing', ascending=False)
    report.append("MISSINGNESS REPORT:")
    report.append("-" * 80)
    report.append(missing_report.to_string())
    report.append("")

    # Adjust outliers in RT data
    RT_variables = [
        "Emotion_Task_Face_Median_RT",
        "Language_Task_Story_Median_RT",
        "Social_Task_TOM_Median_RT_TOM",
        "ER40_CRT",
    ]

    report.append("OUTLIER DETECTION AND REMOVAL:")
    report.append("-" * 80)

    for variable in RT_variables:
        q1 = final_df[variable].quantile(0.25)
        q3 = final_df[variable].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Plot the distribution of the variable
        final_df[variable].plot.hist(bins=100, title=variable)
        plt.plot([lower_bound, lower_bound], [0, 50], 'r--')
        plt.plot([upper_bound, upper_bound], [0, 50], 'r--')
        plt.close()  # Close plot instead of showing

        # Replace outliers with NaN
        final_df.loc[final_df[variable] < lower_bound, variable] = np.nan
        final_df.loc[final_df[variable] > upper_bound, variable] = np.nan

        # Record the number of outliers
        num_outliers = final_df[variable].isna().sum()
        report.append(f"Variable: {variable}, Number of outliers: {num_outliers}")

    report.append("")

    # Impute missing values with MICE
    behavioural_variables = [
        "Friendship_Unadj",
        "Loneliness_Unadj",
        "PercHostil_Unadj",
        "PercReject_Unadj",
        "EmotSupp_Unadj",
        "InstruSupp_Unadj",
        "Emotion_Task_Face_Median_RT",
        "Language_Task_Story_Median_RT",
        "Social_Task_TOM_Median_RT_TOM",
        "WM_Task_0bk_Median_RT",
        "ER40_CRT",
        "Language_Task_Story_Acc",
        "ER40_CR",
        "Emotion_Task_Face_Acc",
        "Social_Task_TOM_Perc_TOM",
    ]

    # Select only behavioural variables
    df = final_df[behavioural_variables].reset_index()

    # Create a kernel (stores multiple imputations)
    imputer_rf = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        random_state=42,
        max_iter=5
    )
    imputed_data = imputer_rf.fit_transform(df.values[:, 1:])

    df_imputed = pd.DataFrame(imputed_data, columns=df.columns[1:])
    df_imputed = df_imputed.set_index(final_df['Subject'])

    # Create a table with the mean, median, skewness, and kurtosis of each variable
    summary_table = pd.DataFrame(columns=["Mean", "Median", "Skewness", "Kurtosis"])
    for variable in behavioural_variables:
        data = df_imputed[variable].dropna()
        summary_table.loc[variable] = [
            data.mean(),
            data.median(),
            stats.skew(data),
            stats.kurtosis(data)
        ]

    report.append("DESCRIPTIVE STATISTICS (after imputation):")
    report.append("-" * 80)
    report.append(summary_table.to_string())
    report.append("")

    # Merge imputed data with phenotypic data
    final_df = pd.merge(df_imputed, phenotypic_data, on='Subject')
    report.append(f"Final data shape after imputation: {final_df.shape}")
    report.append("")

    # Save the imputed dataset to a CSV file
    final_df.to_csv(output_file, index=True)
    report.append(f"Output saved to: {output_file}")
    report.append("")

    # Write report to file
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavioural', required=True)
    parser.add_argument('--phenotypic', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--figure', required=False,
                        help='Path to output figure file')
    args = parser.parse_args()

    project_folder = Path(args.project)
    behavioural_file = project_folder / args.behavioural
    phenotypic_file = project_folder / args.phenotypic
    output_file = project_folder / args.output

    # Create necessary directories
    if not os.path.isdir(project_folder / 'figures/'):
        os.mkdir(project_folder / 'figures/')

    if not os.path.isdir(project_folder / 'reports/'):
        os.mkdir(project_folder / 'reports/')

    report_file = project_folder / 'reports/A1_preprocess_phenotypic_data_report.txt'

    final_df = process_hcp_data(project_folder, behavioural_file, phenotypic_file, output_file, report_file)

    selected_df = final_df[['Emotion_Task_Face_Median_RT', 'Language_Task_Story_Median_RT', 'Social_Task_TOM_Median_RT_TOM', 'ER40_CRT']]
    selected_df = selected_df.apply(stats.zscore)
    selected_df = selected_df.rename(columns={
        'Emotion_Task_Face_Median_RT': 'Emotion',
        'Language_Task_Story_Median_RT': 'Language',
        'Social_Task_TOM_Median_RT_TOM': 'Social',
        'ER40_CRT': 'ER40'
    })
    corr = selected_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    _, ax = plt.subplots(figsize=(60*mm2inches, 60*mm2inches), dpi=300)
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                vmin=0,
                vmax=0.3,
                cbar=False,
                cmap=cmr.ember,
                ax=ax)
    plt.tight_layout()
    
    # Determine output figure path
    if args.figure:
        figure_file = Path(args.figure)
    else:
        figure_file = project_folder / "figures/A1_Behaviour_correlations.png"
    
    plt.savefig(figure_file, dpi=300)

if __name__ == "__main__":
    main()