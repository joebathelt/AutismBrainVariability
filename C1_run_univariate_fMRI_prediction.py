# %%
"""
Connectivity Analysis Pipeline for Social Difficulty (SDS-primary)
==================================================================

Edge-wise statistical analyses on brain connectivity to examine associations
with the Social Difficulty Score (SDS = Social_Score CFA factor; higher z =
worse social performance):

  1. Univariate connectivity ~ Social_Score (full sample, FDR-corrected).
  2. Three-group SDS comparisons (low/middle/high SDS) on residualised
     connectivity, using the same z-cuts with buffer as C3
     (low z<-1 / middle |z|<=0.5 / high z>1; buffer rows excluded).

PGS is intentionally not loaded here. PGS-related analyses are exploratory
and live in C6_exploratory_sds_pgs.py on the genotyped subset.
"""

# %%

import argparse
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn import plotting
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from neuromaps.datasets import fetch_fslr
import seaborn as sns
from scipy.stats import zscore, pearsonr, ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from surfplot import Plot
from utils import connectome_viz
import time
import warnings
warnings.filterwarnings('ignore')


def _read_csv_retry(path, max_retries=5, delay=1.0, **kwargs):
    """Read CSV with retry logic for Docker bind-mount filesystem locking issues."""
    for attempt in range(max_retries):
        try:
            return pd.read_csv(path, **kwargs)
        except OSError as e:
            if attempt < max_retries - 1 and e.errno == 35:  # Resource deadlock avoided
                print(f"  Filesystem lock on {path}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise

# Set up plotting parameters
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

mm2inches = 0.0393701

# %%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def regress_out(X, covariates):
    """
    Regress out covariates from each column of X using linear models.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix where covariates will be regressed out from each column
    covariates : array-like, shape (n_samples, n_covariates)
        Covariate matrix

    Returns:
    --------
    X_residuals : array-like, shape (n_samples, n_features)
        Residualised data matrix
    """
    X_residuals = np.zeros_like(X)
    for i in range(X.shape[1]):
        model = LinearRegression().fit(covariates, X[:, i])
        X_residuals[:, i] = X[:, i] - model.predict(covariates)
    return X_residuals


def extract_upper_triangle(n_nodes):
    """
    Extract indices for upper triangle of connectivity matrix.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the connectivity matrix

    Returns:
    --------
    linear_indices : array
        Linear indices for upper triangle elements
    """
    upper_indices = np.triu_indices(n_nodes, k=1)
    linear_indices = upper_indices[0] * n_nodes + upper_indices[1]
    return linear_indices


def load_connectivity_data(matrix_file, id_file, n_nodes):
    """
    Load and process connectivity matrices.

    Parameters:
    -----------
    matrix_file : str or Path
        Path to connectivity matrix file
    id_file : str or Path
        Path to subject ID file
    n_nodes : int
        Number of nodes in parcellation

    Returns:
    --------
    mats_df : DataFrame
        Connectivity data with subject IDs as index
    """
    # Load connectivity matrices
    mats_df = _read_csv_retry(matrix_file, header=None, sep=r'\s+')

    # Extract upper triangle for non-redundancy
    linear_indices = extract_upper_triangle(n_nodes)
    mats_df = mats_df.iloc[:, linear_indices]

    # Rename features and add participant IDs
    n_connections = len(linear_indices)
    mats_df.columns = [f'conn_{i+1}' for i in range(n_connections)]

    # Load subject IDs
    ids = _read_csv_retry(id_file, header=None)[0].tolist()
    mats_df.index = ids

    print(f'Loaded connectivity data: {mats_df.shape[0]} subjects, {n_connections} connections')
    return mats_df


def perform_univariate_analysis(X_conn_residuals, y, alpha=0.05):
    """
    Perform univariate correlation analysis with FDR correction.

    Parameters:
    -----------
    X_conn_residuals : array-like, shape (n_samples, n_features)
        Residualised connectivity features
    y : array-like, shape (n_samples,)
        Outcome variable (e.g., social difficulty scores)
    alpha : float, default=0.05
        Significance threshold

    Returns:
    --------
    results : dict
        Dictionary containing correlation results
    """
    n_features = X_conn_residuals.shape[1]
    rs = np.zeros(n_features)
    ps = np.zeros(n_features)

    # Compute correlations for each feature
    for i in range(n_features):
        r, p = pearsonr(X_conn_residuals[:, i], y)
        rs[i] = r
        ps[i] = p

    # Apply FDR correction
    reject, p_corr, _, _ = multipletests(ps, method='fdr_bh', alpha=alpha)

    results = {
        'correlations': rs,
        'p_values': ps,
        'p_corrected': p_corr,
        'significant': reject,
        'n_significant': np.sum(reject),
        'significant_indices': np.where(reject)[0]
    }

    return results


def perform_group_comparison(X_conn_residuals, group1_mask, group2_mask, alpha=0.05):
    """
    Perform group comparison analysis using t-tests with FDR correction.

    Parameters:
    -----------
    X_conn_residuals : array-like, shape (n_samples, n_features)
        Residualised connectivity features
    group1_mask : array-like, shape (n_samples,)
        Boolean mask for group 1 subjects
    group2_mask : array-like, shape (n_samples,)
        Boolean mask for group 2 subjects
    alpha : float, default=0.05
        Significance threshold

    Returns:
    --------
    results : dict
        Dictionary containing group comparison results
    """
    n_features = X_conn_residuals.shape[1]
    t_stats = np.zeros(n_features)
    ps = np.zeros(n_features)
    effect_sizes = np.zeros(n_features)

    # Perform t-tests for each feature
    for i in range(n_features):
        group1_data = X_conn_residuals[group1_mask, i]
        group2_data = X_conn_residuals[group2_mask, i]
        
        # Two-sample t-test
        t_stat, p_val = ttest_ind(group1_data, group2_data)
        t_stats[i] = t_stat
        ps[i] = p_val
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                             (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                             (len(group1_data) + len(group2_data) - 2))
        effect_sizes[i] = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std

    # Apply FDR correction
    reject, p_corr, _, _ = multipletests(ps, method='fdr_bh', alpha=alpha)

    results = {
        't_statistics': t_stats,
        'p_values': ps,
        'p_corrected': p_corr,
        'effect_sizes': effect_sizes,
        'significant': reject,
        'n_significant': np.sum(reject),
        'significant_indices': np.where(reject)[0],
        'group1_n': np.sum(group1_mask),
        'group2_n': np.sum(group2_mask)
    }

    return results


# %%
# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_connectivity_matrix(indices, values, n_nodes):
    """
    Create a symmetric connectivity matrix from edge indices and values.
    
    Parameters:
    -----------
    indices : array-like
        Linear indices of significant connections
    values : array-like
        Values to populate the matrix
    n_nodes : int
        Number of nodes in the parcellation
        
    Returns:
    --------
    matrix : array, shape (n_nodes, n_nodes)
        Symmetric connectivity matrix
    """
    matrix = np.zeros((n_nodes, n_nodes))
    sig_idx = connectome_viz.connectivity_indices_to_matrix_indices(indices, n_nodes)
    
    for idx in range(len(indices)):
        i, j = sig_idx[0][idx], sig_idx[1][idx]
        matrix[i, j] = values[idx]
        matrix[j, i] = values[idx]
    
    return matrix


def create_bezier_connectome_plot(matrix, n_nodes, project_folder, output_filename,
                                  edge_color, sign=None):
    """
    Create a Bezier connectome plot.
    
    Parameters:
    -----------
    matrix : array, shape (n_nodes, n_nodes)
        Connectivity matrix
    n_nodes : int
        Number of nodes
    project_folder : Path
        Project directory path
    output_filename : str or Path
        Output filename
    edge_color : tuple
        RGB color for edges
    sign : str, optional
        'positive' or 'negative' for filtering connections
    """
    # Apply sign filtering if specified
    if sign == 'positive':
        matrix = matrix.copy()
        matrix[matrix < 0] = 0
    elif sign == 'negative':
        matrix = np.abs(matrix.copy())
        matrix[matrix == 0] = 0  # Keep only originally negative values
    
    # Remove disconnected nodes
    connected_mask = np.mean(matrix, axis=0) != 0
    if np.sum(connected_mask) == 0:
        print(f"No connected nodes for {sign} connections")
        return None, None

    matrix_filtered = matrix[np.ix_(connected_mask, connected_mask)]

    # Create labels for connected nodes
    label_names = [str(i+1) for i, connected in enumerate(connected_mask)
                   if connected]

    # Load coordinates and create thumbnail dictionary
    coordinates_df = _read_csv_retry(
        project_folder / f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_node_coords.csv'
    )
    coordinates_df['label'] = [str(int(comp[2:]) + 1)
                               for comp in coordinates_df['component']]

    thumbnail_folder = Path(
        project_folder / f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_sum.sum'
    )
    thumbnail_files = sorted([img for img in os.listdir(thumbnail_folder)
                              if img.endswith('.png')])
    thumbnail_dict = {
        str(int(thumbnail.split('.')[0]) + 1): str(thumbnail_folder / thumbnail)
        for thumbnail in thumbnail_files
    }
    
    # Create plot
    try:
        fig, ax = connectome_viz.create_bezier_connectome(
            matrix=matrix_filtered,
            labels=label_names,
            image_paths=thumbnail_dict,
            coordinates_df=coordinates_df,
            output_path=output_filename,
            figsize=(75*mm2inches, 75*mm2inches),
            edge_color=edge_color,
            image_scale=0.15,
            curve_height=0.3,
            min_line_width=1,
            connection_threshold=0,
            label_fontsize=8,
            node_size=0.1,
            dpi=300,
        )
        return fig, ax
    except Exception as e:
        print(f"Error creating connectome plot: {e}")
        return None, None


def create_surface_plot(node_vector, n_nodes, project_folder, output_filename,
                        cmap='coolwarm', color_range=None, cbar_label=None):
    """
    Create a surface plot visualization.
    
    Parameters:
    -----------
    node_vector : array
        Values for each node
    n_nodes : int
        Number of nodes
    project_folder : Path
        Project directory path
    output_filename : str or Path
        Output filename
    cmap : str, default='coolwarm'
        Colormap name
    color_range : tuple, optional
        Color range (min, max)
    cbar_label : str, optional
        Colorbar label
    """
    # Load surface data
    dlabel_filename = (project_folder /
                       f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_ftb.dlabel.nii')
    dscalar_filename = (project_folder /
                        f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC.dscalar.nii')

    left_full, right_full = connectome_viz.mask_dlabel_data(dscalar_filename,
                                                            dlabel_filename)
    lh_val, rh_val = left_full.copy(), right_full.copy()
    
    # Map node values to surface
    for i in range(len(node_vector)):
        lh_val[left_full == i+1] = node_vector[i]
        rh_val[right_full == i+1] = node_vector[i]
    
    # Create surface plot
    width, height = int(120 * 300 / 25.4), int(90 * 300 / 25.4)
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    
    p = Plot(lh, rh, zoom=1.5, size=(width, height))
    
    # Set color range if not provided
    if color_range is None:
        max_val = np.max(np.abs(node_vector))
        color_range = (-max_val, max_val) if max_val > 0 else (-1, 1)
    
    p.add_layer({'left': lh_val, 'right': rh_val},
                cmap=cmap,
                color_range=color_range,
                cbar=True,
                cbar_label=cbar_label
                )
    p.add_layer({'left': left_full, 'right': right_full},
                cmap='gray', as_outline=True, cbar=False)
    
    cbar_kws = {'fontsize': 16, 'pad': 0.05}
    fig = p.build(cbar_kws=cbar_kws)
    for attempt in range(5):
        try:
            fig.savefig(output_filename, dpi=300, bbox_inches='tight',
                        pad_inches=0.1)
            break
        except OSError as e:
            if attempt < 4 and e.errno == 35:
                print(f"  Filesystem lock saving {output_filename}, "
                      f"retrying ({attempt+1}/5)...")
                time.sleep(1.0)
            else:
                raise

    return fig


def create_social_difficulty_visualizations(results_nodes, n_nodes, project_folder):
    """
    Create visualizations for social difficulty associations.
    
    Parameters:
    -----------
    results_nodes : dict
        Results for specific parcellation
    n_nodes : int
        Number of nodes
    project_folder : Path
        Project directory path
    """
    indices = results_nodes['social_univariate']['significant_indices']
    if len(indices) == 0:
        print(f"No significant social difficulty associations for "
              f"{n_nodes}-node parcellation")
        return

    correlations = results_nodes['social_univariate']['correlations'][indices]

    # Create separate plots for positive and negative correlations
    colors = {'positive': (0.902, 0.294, 0.208),
              'negative': (0.302, 0.733, 0.835)}
    
    for sign in ['positive', 'negative']:
        # Filter correlations by sign
        if sign == 'positive':
            sign_mask = correlations > 0
        else:
            sign_mask = correlations < 0
            
        if not np.any(sign_mask):
            continue

        sign_indices = indices[sign_mask]
        sign_correlations = correlations[sign_mask]

        # Create connectivity matrix
        matrix = create_connectivity_matrix(sign_indices, sign_correlations,
                                            n_nodes)

        # Bezier connectome plot
        filename = (project_folder /
                    f'figures/Connectome_sign_nodes-{n_nodes}_stat-corr_{sign}.png')
        create_bezier_connectome_plot(matrix, n_nodes, project_folder,
                                      filename, colors[sign], sign)
    
    # Create surface plot (all correlations)
    matrix_all = create_connectivity_matrix(indices, correlations, n_nodes)
    node_vector = matrix_all.sum(axis=0)

    filename = (project_folder /
                f'figures/Connectome_sign_nodes-{n_nodes}_stat-corr_surf.png')
    try:
        create_surface_plot(node_vector, n_nodes, project_folder, filename,
                            color_range=(-0.5, 0.5), cbar_label='Correlation Sum')
    except Exception as e:
        print(f"Error creating surface plot for social difficulty "
              f"({n_nodes}-node): {e}")


def create_group_split_visualizations(results_nodes, n_nodes, project_folder):
    """
    Create visualizations for three-group split analyses.

    Parameters:
    -----------
    results_nodes : dict
        Results for specific parcellation
    n_nodes : int
        Number of nodes
    project_folder : Path
        Project directory path
    """
    group_splits = results_nodes['group_splits']

    # Visualization parameters for different metrics
    viz_params = {
        't_statistics': {'cmap': 'coolwarm', 'suffix': 'tstat',
                         'label': 't-statistic'},
        'effect_sizes': {'cmap': 'RdBu_r', 'suffix': 'cohend',
                         'color_range': (-1, 1), 'label': "Cohen's d"}
    }

    comparison_names = ['low_vs_middle', 'high_vs_middle', 'low_vs_high']

    for comparison_name in comparison_names:
        if comparison_name not in group_splits:
            continue

        comparison_data = group_splits[comparison_name]
        results = comparison_data['results']

        if results['n_significant'] == 0:
            print(f"No significant connections for {comparison_name} "
                  f"comparison ({n_nodes}-node)")
            continue

        indices = results['significant_indices']

        for metric, params in viz_params.items():
            values = results[metric][indices]

            # Create connectivity matrix and sum for node vector
            matrix = create_connectivity_matrix(indices, values, n_nodes)
            node_vector = matrix.sum(axis=0)

            # Set color range
            color_range = params.get('color_range')
            if color_range is None:
                max_val = np.max(np.abs(values))
                color_range = ((-max_val, max_val) if max_val > 0
                               else (-1, 1))

            # Create surface plot
            filename = (project_folder /
                        f'figures/GroupSplit_{comparison_name}_nodes-{n_nodes}_{params["suffix"]}_surf.png')
            try:
                create_surface_plot(node_vector, n_nodes, project_folder,
                                    filename, cmap=params['cmap'],
                                    color_range=color_range,
                                    cbar_label=params['label'])
            except Exception as e:
                print(f"Error creating surface plot for "
                      f"{comparison_name} ({n_nodes}-node, "
                      f"{params['suffix']}): {e}")


# %%
# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class ConnectivityAnalysis:
    """
    Class for conducting edge-wise connectivity analyses with three-group
    comparisons.
    """

    def __init__(self, project_folder, parcellation_sizes=[50, 100, 200],
                 behavioural_file=None, phenotypic_file=None,
                 social_file=None, movement_file=None, id_file=None,
                 matrices_dir=None, report=None):
        """
        Initialize the analysis pipeline.

        Parameters mirror C3: behavioural, phenotypic, social, movement, ids,
        matrices. PGS is not loaded.
        """
        self.project_folder = Path(project_folder)
        self.parcellation_sizes = parcellation_sizes
        self.results = {}
        self.report = report if report is not None else []

        self.behavioural_file = (Path(behavioural_file) if behavioural_file
                                 else self.project_folder / "data/behavioural_data_anonymised.csv")
        self.phenotypic_file = (Path(phenotypic_file) if phenotypic_file
                                else self.project_folder / "data/phenotypic_data_anonymised.csv")
        self.social_file = (Path(social_file) if social_file
                            else self.project_folder / 'data/cfa_factor_scores_full_sample.csv')
        self.movement_file = (Path(movement_file) if movement_file
                              else self.project_folder / 'data/movement_data_anonymised.csv')
        self.id_file = (Path(id_file) if id_file
                        else self.project_folder / 'data/subjectIDs_anonymised.txt')
        self.matrices_dir = (Path(matrices_dir) if matrices_dir
                             else self.project_folder / 'data/raw_anonymised/')

        self._load_phenotypic_data()

    def _load_phenotypic_data(self):
        """Load all non-connectivity data (no PGS)."""
        print("Loading phenotypic data...")
        self.social_df = _read_csv_retry(self.social_file)
        self.behavioural_df = _read_csv_retry(self.behavioural_file)
        self.phenotypic_df = _read_csv_retry(self.phenotypic_file)
        self.movement_df = _read_csv_retry(self.movement_file)
        print(f"Loaded social factor scores for {len(self.social_df)} subjects")

    def load_connectivity_data(self, n_nodes):
        """
        Load connectivity data for specified parcellation size.

        Parameters:
        -----------
        n_nodes : int
            Number of nodes in parcellation

        Returns:
        --------
        mats_df : DataFrame
            Connectivity data
        """
        matrix_file = (self.matrices_dir /
                       f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats2.txt')
        return load_connectivity_data(matrix_file, self.id_file, n_nodes)

    def prepare_data(self, n_nodes, motion_threshold=0.2):
        """
        Prepare and preprocess data for analysis.

        Parameters:
        -----------
        n_nodes : int
            Number of nodes in parcellation
        motion_threshold : float, default=0.2
            Threshold for excluding high-motion subjects

        Returns:
        --------
        analysis_data : dict
            Dictionary containing prepared data arrays
        """
        msg = f"\nPreparing data for {n_nodes}-node parcellation..."
        print(msg)
        self.report.append(msg)

        # Load connectivity data
        mats_df = self.load_connectivity_data(n_nodes)

        # Merge from social (no PGS join — preserves N).
        merged_df = pd.merge(self.social_df[['Subject', 'Social_Score']],
                             mats_df, left_on='Subject', right_index=True)
        merged_df = pd.merge(merged_df,
                             self.behavioural_df[['Subject', 'Gender',
                                                  'FS_IntraCranial_Vol']],
                             on='Subject')
        self.phenotypic_df = self.phenotypic_df.rename(
            columns={'Individual_ID': 'Subject'})
        merged_df = pd.merge(merged_df,
                             self.phenotypic_df[['Subject', 'Age_in_Yrs']],
                             on='Subject')
        merged_df = pd.merge(merged_df,
                             self.movement_df[['Subject',
                                               'Movement_RelativeRMS_mean']],
                             on='Subject')

        # Quality control filters
        merged_df = merged_df.loc[
            merged_df['Movement_RelativeRMS_mean'] < motion_threshold]
        merged_df = merged_df.dropna()

        msg = f'Final sample size: {merged_df.shape[0]} subjects'
        print(msg)
        self.report.append(msg)

        # Extract data arrays
        conn_features = [col for col in merged_df.columns
                         if col.startswith('conn')]
        X_conn = merged_df[conn_features].values

        # Covariates: age, ICV, motion, plus Gender (mixed-sex sample).
        cov_df = merged_df[['Age_in_Yrs', 'FS_IntraCranial_Vol',
                            'Movement_RelativeRMS_mean', 'Gender']]
        X_cov = pd.get_dummies(cov_df, drop_first=True)

        y_social = merged_df['Social_Score'].values

        scaler = RobustScaler()
        X_conn_scaled = scaler.fit_transform(X_conn)
        X_cov_scaled = scaler.fit_transform(X_cov)

        y_social_scaled = zscore(y_social)
        sds_z = y_social_scaled  # SDS = z-scored Social_Score

        X_conn_residuals = regress_out(X_conn_scaled, X_cov_scaled)

        # SDS group masks (mirror C3): low z<-1, middle |z|<=0.5, high z>1
        low_sds_mask = sds_z < -1.0
        middle_mask = (sds_z >= -0.5) & (sds_z <= 0.5)
        high_sds_mask = sds_z > 1.0

        analysis_data = {
            'X_conn_residuals': X_conn_residuals,
            'y_social_scaled': y_social_scaled,
            'sds_z': sds_z,
            'merged_df': merged_df,
            'n_connections': len(conn_features),
            'n_subjects': len(merged_df),
            'low_sds_mask': low_sds_mask,
            'middle_mask': middle_mask,
            'high_sds_mask': high_sds_mask,
        }

        return analysis_data

    def run_univariate_analysis(self, analysis_data):
        """Univariate connectivity ~ Social_Score (FDR-corrected)."""
        print("\nRunning univariate analysis (target=Social_Score)...")
        X_conn_residuals = analysis_data['X_conn_residuals']
        y = analysis_data['y_social_scaled']
        results = perform_univariate_analysis(X_conn_residuals, y)
        print(f"Significant features after FDR correction: "
              f"{results['n_significant']}")
        return results

    def run_three_group_analysis(self, analysis_data):
        """Three-group SDS comparisons on residualised connectivity:
        Low vs Middle, High vs Middle, Low vs High."""
        print("\nRunning three-group SDS analyses...")

        X_conn_residuals = analysis_data['X_conn_residuals']
        low_mask = analysis_data['low_sds_mask']
        middle_mask = analysis_data['middle_mask']
        high_mask = analysis_data['high_sds_mask']

        print(f"Group sizes - Low SDS (z<-1): {np.sum(low_mask)}")
        print(f"Group sizes - Middle SDS (|z|<=0.5): {np.sum(middle_mask)}")
        print(f"Group sizes - High SDS (z>1): {np.sum(high_mask)}")

        print("\nLow SDS vs Middle SDS comparison...")
        low_vs_middle_results = perform_group_comparison(
            X_conn_residuals, low_mask, middle_mask)
        print(f"Significant differences after FDR correction: "
              f"{low_vs_middle_results['n_significant']}")

        print("\nHigh SDS vs Middle SDS comparison...")
        high_vs_middle_results = perform_group_comparison(
            X_conn_residuals, high_mask, middle_mask)
        print(f"Significant differences after FDR correction: "
              f"{high_vs_middle_results['n_significant']}")

        print("\nLow SDS vs High SDS comparison...")
        low_vs_high_results = perform_group_comparison(
            X_conn_residuals, low_mask, high_mask)
        print(f"Significant differences after FDR correction: "
              f"{low_vs_high_results['n_significant']}")

        return {
            'low_vs_middle': {
                'results': low_vs_middle_results,
                'description': 'Low SDS (z<-1) vs Middle SDS (|z|<=0.5)',
            },
            'high_vs_middle': {
                'results': high_vs_middle_results,
                'description': 'High SDS (z>1) vs Middle SDS (|z|<=0.5)',
            },
            'low_vs_high': {
                'results': low_vs_high_results,
                'description': 'Low SDS (z<-1) vs High SDS (z>1)',
            },
        }

    def run_full_analysis(self, motion_threshold=0.2):
        """
        Run complete analysis pipeline for all parcellation sizes.

        Parameters:
        -----------
        motion_threshold : float, default=0.2
            Motion threshold for subject exclusion
        """
        header = "=" * 60
        title = "CONNECTIVITY ANALYSIS PIPELINE - THREE GROUP COMPARISON"
        print(header)
        print(title)
        print(header)
        self.report.append(header)
        self.report.append(title)
        self.report.append(header)
        self.report.append("")
        self.report.append(f"Motion threshold: {motion_threshold}")
        self.report.append(f"Parcellation sizes: {self.parcellation_sizes}")
        self.report.append("")

        results_dir = self.project_folder / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        for n_nodes in self.parcellation_sizes:
            print(f"\n{'='*20} {n_nodes}-NODE ANALYSIS {'='*20}")
            analysis_data = self.prepare_data(n_nodes, motion_threshold)
            social_results = self.run_univariate_analysis(analysis_data)
            group_split_results = self.run_three_group_analysis(analysis_data)
            self.results[n_nodes] = {
                'analysis_data': analysis_data,
                'social_univariate': social_results,
                'group_splits': group_split_results,
            }

            # Persist edge-wise social univariate stats so downstream scripts
            # (e.g. C7) can pick up the significant edges without re-running C1.
            upper_i, upper_j = np.triu_indices(n_nodes, k=1)
            edges_df = pd.DataFrame({
                'edge_idx': np.arange(1, len(upper_i) + 1),
                'node_i': upper_i,
                'node_j': upper_j,
                'r': social_results['correlations'],
                'p': social_results['p_values'],
                'p_corrected': social_results['p_corrected'],
                'significant': social_results['significant'],
            })
            edges_path = results_dir / f'C1_social_univariate_edges_{n_nodes}.csv'
            edges_df.to_csv(edges_path, index=False)
            self.report.append(
                f"  Saved edge stats: {edges_path.name} "
                f"({int(edges_df['significant'].sum())} significant edges)"
            )

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

        # Summary across parcellations
        self._print_summary()

    def _print_summary(self):
        """Print summary of results across all parcellations."""
        print("\nSUMMARY OF RESULTS:")
        print("-" * 40)
        self.report.append("")
        self.report.append("SUMMARY OF RESULTS:")
        self.report.append("-" * 40)

        for n_nodes in self.parcellation_sizes:
            if n_nodes not in self.results:
                continue
            results = self.results[n_nodes]
            n_conn = results['analysis_data']['n_connections']
            n_subj = results['analysis_data']['n_subjects']
            social_sig = results['social_univariate']['n_significant']
            low_vs_middle_sig = (results['group_splits']['low_vs_middle']
                                 ['results']['n_significant'])
            high_vs_middle_sig = (results['group_splits']['high_vs_middle']
                                  ['results']['n_significant'])
            low_vs_high_sig = (results['group_splits']['low_vs_high']
                               ['results']['n_significant'])

            msg_header = f"\n{n_nodes}-node parcellation:"
            print(msg_header)
            self.report.append(msg_header)
            details = [
                f"  N subjects: {n_subj}",
                f"  N connections: {n_conn}",
                f"  Social difficulty associations: {social_sig}",
                f"  Low SDS vs Middle: {low_vs_middle_sig}",
                f"  High SDS vs Middle: {high_vs_middle_sig}",
                f"  Low SDS vs High SDS: {low_vs_high_sig}",
            ]
            for d in details:
                print(d)
                self.report.append(d)
            print()
            self.report.append("")


# %%
# =============================================================================
# MAIN ANALYSIS EXECUTION
# =============================================================================

def main():
    """Main function to run univariate fMRI prediction analysis."""
    parser = argparse.ArgumentParser(
        description='Run univariate fMRI connectivity prediction analysis'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--social', required=False,
                        help='Path to social factor scores CSV')
    parser.add_argument('--behavioural', required=False,
                        help='Path to behavioural data CSV')
    parser.add_argument('--phenotypic', required=False,
                        help='Path to phenotypic data CSV')
    parser.add_argument('--movement', required=False,
                        help='Path to movement data CSV')
    parser.add_argument('--ids', required=False,
                        help='Path to subject IDs file')
    parser.add_argument('--matrices-dir', required=False,
                        help='Path to connectivity matrices directory')
    parser.add_argument('--motion-threshold', type=float, default=0.2,
                        help='Motion threshold for subject exclusion (default: 0.2)')
    parser.add_argument('--parcellations', nargs='+', type=int,
                        default=[50, 100, 200],
                        help='Parcellation sizes to analyze (default: 50 100 200)')
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True, exist_ok=True)
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = [
        "=" * 80,
        "C1: UNIVARIATE fMRI CONNECTIVITY PREDICTION REPORT",
        "=" * 80,
        ""
    ]

    # Create analysis object with different parcellation sizes
    # Primary analysis: 100-node parcellation
    # Sensitivity analyses: 50-node and 200-node parcellations
    analysis = ConnectivityAnalysis(
        project_folder=project_folder,
        parcellation_sizes=args.parcellations,
        behavioural_file=args.behavioural,
        phenotypic_file=args.phenotypic,
        social_file=args.social,
        movement_file=args.movement,
        id_file=args.ids,
        matrices_dir=args.matrices_dir,
        report=report
    )

    # Run complete analysis pipeline
    analysis.run_full_analysis(motion_threshold=args.motion_threshold)

    # Access results for specific parcellation and create visualizations
    for n_nodes in args.parcellations:
        results_nodes = analysis.results[n_nodes]

        print(f"\nCreating visualizations for {n_nodes}-node parcellation...")

        try:
            # Create social difficulty visualizations
            create_social_difficulty_visualizations(results_nodes, n_nodes,
                                                     project_folder)
        except Exception as e:
            print(f"Error in social difficulty visualizations "
                  f"({n_nodes}-node): {e}")

        try:
            # Create group-split visualizations
            create_group_split_visualizations(results_nodes, n_nodes,
                                              project_folder)
        except Exception as e:
            print(f"Error in group-split visualizations "
                  f"({n_nodes}-node): {e}")

    # Detailed SDS-group split summary
    header = "\n" + "="*60
    title = "DETAILED THREE-GROUP SDS RESULTS SUMMARY"
    separator = "="*60
    print(header); print(title); print(separator)
    report.append(header); report.append(title); report.append(separator)
    report.append("")

    for n_nodes in args.parcellations:
        if n_nodes not in analysis.results:
            continue
        results_nodes = analysis.results[n_nodes]
        group_splits = results_nodes['group_splits']
        analysis_data = results_nodes['analysis_data']

        msg = f"\n{n_nodes}-node parcellation:"
        print(msg); report.append(msg)
        report.append("-" * 30); print("-" * 30)

        n_low = int(np.sum(analysis_data['low_sds_mask']))
        n_middle = int(np.sum(analysis_data['middle_mask']))
        n_high = int(np.sum(analysis_data['high_sds_mask']))
        sample_info = [
            f"Final sample size: {analysis_data['n_subjects']} subjects",
            "",
            "SDS group sizes:",
            f"  Low SDS  (z<-1):       {n_low}",
            f"  Middle SDS (|z|<=0.5): {n_middle}",
            f"  High SDS (z>1):        {n_high}",
        ]
        for info in sample_info:
            print(info); report.append(info)

        for comparison_name, key in [('Low SDS vs Middle SDS', 'low_vs_middle'),
                                     ('High SDS vs Middle SDS', 'high_vs_middle'),
                                     ('Low SDS vs High SDS', 'low_vs_high')]:
            data = group_splits[key]
            n_sig = data['results']['n_significant']
            lines = [
                f"\n{comparison_name}:",
                f"  Significant connections (FDR): {n_sig}",
            ]
            if n_sig > 0:
                effects = (data['results']['effect_sizes']
                           [data['results']['significant_indices']])
                lines.append(
                    f"  Effect sizes: range = [{np.min(effects):.3f}, "
                    f"{np.max(effects):.3f}]"
                )
            for line in lines:
                print(line); report.append(line)

    # Write report to file
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_file = reports_dir / 'C1_run_univariate_fMRI_prediction_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()

# %%