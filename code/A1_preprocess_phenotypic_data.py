# ==============================================================================
# Title: A1_preprocess_phenotypic_data.py
# ==============================================================================
# Description: This script preprocesses phenotypic data for further analysis.
# It includes steps such as data cleaning, handling missing values, and generating
# derived variables. The processed data is saved for use in subsequent analysis scripts.
# ==============================================================================

# %%
import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
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
rcParams['text.usetex'] = False
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701

# %%
def process_hcp_data(behavioural_file, phenotypic_file, output_file):
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
        behavioural_file (str): Path to the behavioural data CSV file.
        phenotypic_file (str): Path to the phenotypic data CSV file.
        output_file (str): Path to save the processed data CSV file.

    Returns:
        pd.DataFrame: The processed and imputed dataset.
    """
    # Load datasets
    behavioural_data = pd.read_csv(behavioural_file, delimiter=',')
    phenotypic_data = pd.read_csv(phenotypic_file,delimiter=',')
    print(f"Behavioural data shape: {behavioural_data.shape}")
    print(f"Phenotypic data shape: {phenotypic_data.shape}")

    # Filter columns of interest
    behaviour_colummns = [
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
    behavioural_data = behavioural_data[behaviour_colummns]

    phenotype_columns = [
        "Subject",
        'Family_ID',
        "Age_in_Yrs",
        "HasGT",
        'Height',
        'Weight',
        'BPSystolic',
        'BPDiastolic',
    ]
    phenotypic_data = phenotypic_data[phenotype_columns]

    # Replace empty strings with NA if they exist
    phenotypic_data[phenotypic_data == ''] = np.nan

    # Ensure 'Subject' is consistent in type
    behavioural_data['Subject'] = behavioural_data['Subject'].astype(str)
    phenotypic_data['Subject'] = phenotypic_data['Subject'].astype(str)

    # Merge datasets on 'Subject'
    merged_df = pd.merge(behavioural_data, phenotypic_data, on='Subject', how='inner')
    merged_df.to_csv(project_folder / "data/merged_data_initial.csv", index=False)

    # Filter participants with resting-state fMRI data
    columns_of_interest = ['3T_RS-fMRI_Count', '3T_RS-fMRI_PctCompl']
    merged_df = merged_df[(merged_df[columns_of_interest] != 0.0).all(axis=1)]
    print(f"Data shape after fMRI filtering: {merged_df.shape}")

    # Filter participants with genetic data
    merged_df = merged_df[merged_df['HasGT'] == True]
    print(f"Data shape after genetic data filtering: {merged_df.shape}")

    # Replace NaN values with np.nan
    final_df = merged_df.replace({pd.NA: np.nan})

    # Generate missingness report
    missing_report = pd.DataFrame({
        'Missing Values': final_df.isna().sum(),
        'Percentage Missing': final_df.isna().mean() * 100
    }).sort_values(by='Percentage Missing', ascending=False)
    missing_report.to_csv(project_folder / "data/missingness_report.csv")
    print(missing_report)

    # Adjust outliers in RT data
    RT_variables = [
        "Emotion_Task_Face_Median_RT",
        "Language_Task_Story_Median_RT",
        "Social_Task_TOM_Median_RT_TOM",
        "ER40_CRT",
    ]

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
        plt.savefig(project_folder / f"figures/{variable}_outliers.png", dpi=300)
        plt.clf()

        # Replace outliers with NaN
        final_df.loc[final_df[variable] < lower_bound, variable] = np.nan
        final_df.loc[final_df[variable] > upper_bound, variable] = np.nan

        # Print the number of outliers
        num_outliers = final_df[variable].isna().sum()
        print(f"Variable: {variable}, Number of outliers: {num_outliers}")

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
    print("Starting MICE imputation...")
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

    summary_table.to_csv(project_folder / "data/descriptive_statistics_table.csv")
    print(summary_table)

    # Merge imputed data with phenotypic data
    final_df = pd.merge(df_imputed, phenotypic_data, on='Subject')
    print(f"Final data shape after imputation: {final_df.shape}")

    # Save the imputed dataset to a CSV file
    final_df.to_csv(output_file, index=True)

    return final_df

# %%
project_folder = Path(__file__).resolve().parents[1]
behavioural_file = project_folder / "data/behavioural_data_anonymised.csv"
phenotypic_file = project_folder / "data/phenotypic_data_anonymised.csv"
output_file = project_folder / "data/behavioural_data_preprocessed.csv"

final_df = process_hcp_data(behavioural_file, phenotypic_file, output_file)
# %%
final_df = pd.read_csv(project_folder / "data/behavioural_data_preprocessed.csv")

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

fig, ax = plt.subplots(figsize=(60*mm2inches, 60*mm2inches), dpi=300)
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
plt.savefig(project_folder / "figures/Behaviour_correlations.png", dpi=300)

# %%
