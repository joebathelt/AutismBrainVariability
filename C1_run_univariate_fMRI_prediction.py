# %%
"""
Connectivity Analysis Pipeline for Social Difficulty and Polygenic Risk
========================================================================

This script performs edge-wise statistical analyses on brain connectivity data
to examine associations with social difficulty scores and polygenic scores.
Includes three-group comparisons based on polygenic score thresholds:
- Low PGS: <-1 SD
- Middle PGS: >-0.5 SD & <0.5 SD
- High PGS: >1 SD
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


def compute_interaction_terms(X_conn_residuals, pgs_scaled):
    """
    Compute interaction terms between connectivity features and polygenic scores.
    
    Parameters:
    -----------
    X_conn_residuals : array-like, shape (n_samples, n_features)
        Residualised connectivity features
    pgs_scaled : array-like, shape (n_samples,)
        Standardised polygenic scores
        
    Returns:
    --------
    X_interaction : array-like, shape (n_samples, n_features)
        Interaction terms (connectivity × polygenic score)
    """
    X_interaction = X_conn_residuals * pgs_scaled[:, np.newaxis]
    return X_interaction


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
                 behavioural_file=None, phenotypic_file=None, pgs_file=None,
                 social_file=None, movement_file=None, id_file=None,
                 matrices_dir=None, report=None):
        """
        Initialize the analysis pipeline.

        Parameters:
        -----------
        project_folder : str or Path
            Path to project directory
        parcellation_sizes : list
            List of parcellation sizes to analyze
        behavioural_file : str or Path, optional
            Path to behavioural data file
        phenotypic_file : str or Path, optional
            Path to phenotypic data file
        pgs_file : str or Path, optional
            Path to PGS/PGS file
        social_file : str or Path, optional
            Path to social factor scores file
        movement_file : str or Path, optional
            Path to movement data file
        id_file : str or Path, optional
            Path to subject IDs file
        matrices_dir : str or Path, optional
            Path to connectivity matrices directory
        report : list, optional
            Report list for logging
        """
        self.project_folder = Path(project_folder)
        self.parcellation_sizes = parcellation_sizes
        self.results = {}
        self.report = report if report is not None else []

        # Define file paths with defaults
        self.behavioural_file = (Path(behavioural_file) if behavioural_file 
                                 else self.project_folder / "data/behavioural_data_anonymised.csv")
        self.phenotypic_file = (Path(phenotypic_file) if phenotypic_file 
                                else self.project_folder / "data/phenotypic_data_anonymised.csv")
        self.pgs_file = (Path(pgs_file) if pgs_file 
                        else self.project_folder / "data/pgs_residuals.csv")
        self.social_file = (Path(social_file) if social_file 
                           else self.project_folder / 'data/cfa_factor_scores_full_sample.csv')
        self.movement_file = (Path(movement_file) if movement_file 
                             else self.project_folder / 'data/movement_data_anonymised.csv')
        self.id_file = (Path(id_file) if id_file 
                       else self.project_folder / 'data/subjectIDs_anonymised.txt')
        self.matrices_dir = (Path(matrices_dir) if matrices_dir 
                            else self.project_folder / 'data/raw_anonymised/')

        # Load non-connectivity data
        self._load_phenotypic_data()

    def _load_phenotypic_data(self):
        """Load and merge all non-connectivity data."""
        print("Loading phenotypic data...")

        # Load individual datasets
        self.pgs_df = _read_csv_retry(self.pgs_file)
        self.social_df = _read_csv_retry(self.social_file)
        self.behavioural_df = _read_csv_retry(self.behavioural_file)
        self.behavioural_df = self.behavioural_df[self.behavioural_df['Gender'] == 'M']
        self.phenotypic_df = _read_csv_retry(self.phenotypic_file)
        self.movement_df = _read_csv_retry(self.movement_file)

        print(f"Loaded data for {len(self.pgs_df)} subjects")

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

        # Merge all datasets
        merged_df = pd.merge(self.pgs_df, mats_df, left_on='Subject',
                             right_index=True)
        merged_df = pd.merge(merged_df,
                             self.social_df[['Subject', 'Social_Score']],
                             on='Subject')
        merged_df = pd.merge(merged_df,
                             self.behavioural_df[['Subject', 'Gender',
                                                  'FS_IntraCranial_Vol']],
                             on='Subject')
        self.phenotypic_df = self.phenotypic_df.rename(columns={'Individual_ID': 'Subject'})
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

        # Covariates to regress out (age, intracranial volume, head motion)
        covariates = ['Age_in_Yrs', 'FS_IntraCranial_Vol',
                      'Movement_RelativeRMS_mean']
        X_cov = pd.get_dummies(merged_df[covariates], drop_first=True)

        # Outcome variables
        y_social = merged_df['Social_Score'].values
        pgs = merged_df['blup_PGS_residuals'].values

        # Standardization and preprocessing
        scaler = RobustScaler()
        X_conn_scaled = scaler.fit_transform(X_conn)
        X_cov_scaled = scaler.fit_transform(X_cov)

        # Z-score social difficulty scores and polygenic scores
        y_social_scaled = zscore(y_social)
        pgs_scaled = zscore(pgs)

        # Regress out covariates from connectivity data
        X_conn_residuals = regress_out(X_conn_scaled, X_cov_scaled)

        # Compute interaction terms
        X_interaction = compute_interaction_terms(X_conn_residuals, pgs_scaled)

        # Create three group masks based on polygenic scores
        pgs_mean = np.mean(pgs_scaled)
        pgs_std = np.std(pgs_scaled)

        # Three groups:
        # Low PGS: < -1 SD
        # Middle PGS: > -0.5 SD & < 0.5 SD
        # High PGS: > +1 SD
        low_pgs_mask = pgs_scaled < (pgs_mean - 1 * pgs_std)
        middle_pgs_mask = ((pgs_scaled > (pgs_mean - 0.5 * pgs_std)) &
                           (pgs_scaled < (pgs_mean + 0.5 * pgs_std)))
        high_pgs_mask = pgs_scaled > (pgs_mean + 1 * pgs_std)

        analysis_data = {
            'X_conn_residuals': X_conn_residuals,
            'X_interaction': X_interaction,
            'y_social_scaled': y_social_scaled,
            'pgs_scaled': pgs_scaled,
            'merged_df': merged_df,
            'n_connections': len(conn_features),
            'n_subjects': len(merged_df),
            'low_pgs_mask': low_pgs_mask,
            'middle_pgs_mask': middle_pgs_mask,
            'high_pgs_mask': high_pgs_mask
        }

        return analysis_data

    def run_univariate_analysis(self, analysis_data, target='social'):
        """
        Run univariate correlation analysis.

        Parameters:
        -----------
        analysis_data : dict
            Prepared data dictionary
        target : str, default='social'
            Target variable ('social' or 'pgs')

        Returns:
        --------
        results : dict
            Univariate analysis results
        """
        print(f"\nRunning univariate analysis for {target}...")

        X_conn_residuals = analysis_data['X_conn_residuals']

        if target == 'social':
            y = analysis_data['y_social_scaled']
        elif target == 'pgs':
            y = analysis_data['pgs_scaled']
        else:
            raise ValueError("Target must be 'social' or 'pgs'")

        results = perform_univariate_analysis(X_conn_residuals, y)

        print(f"Significant features after FDR correction: "
              f"{results['n_significant']}")

        return results

    def run_interaction_analysis(self, analysis_data):
        """
        Run interaction analysis (connectivity × polygenic score).

        Parameters:
        -----------
        analysis_data : dict
            Prepared data dictionary

        Returns:
        --------
        results : dict
            Interaction analysis results
        """
        print("\nRunning interaction analysis...")

        X_interaction = analysis_data['X_interaction']
        y_social_scaled = analysis_data['y_social_scaled']

        results = perform_univariate_analysis(X_interaction, y_social_scaled)

        print(f"Significant interaction terms after FDR correction: "
              f"{results['n_significant']}")

        return results

    def run_three_group_analysis(self, analysis_data):
        """
        Run three-group comparisons: Low vs Middle, High vs Middle, Low vs High.

        Parameters:
        -----------
        analysis_data : dict
            Prepared data dictionary

        Returns:
        --------
        results : dict
            Three-group comparison results
        """
        print("\nRunning three-group analyses...")

        X_conn_residuals = analysis_data['X_conn_residuals']
        low_pgs_mask = analysis_data['low_pgs_mask']
        middle_pgs_mask = analysis_data['middle_pgs_mask']
        high_pgs_mask = analysis_data['high_pgs_mask']
        y_social_scaled = analysis_data['y_social_scaled']

        print(f"Group sizes - Low PGS (<-1 SD): {np.sum(low_pgs_mask)}")
        print(f"Group sizes - Middle PGS (>-0.5 SD & <0.5 SD): "
              f"{np.sum(middle_pgs_mask)}")
        print(f"Group sizes - High PGS (>+1 SD): {np.sum(high_pgs_mask)}")

        # Low vs Middle comparison
        print("\nLow PGS vs Middle PGS comparison...")
        low_vs_middle_results = perform_group_comparison(
            X_conn_residuals,
            low_pgs_mask,
            middle_pgs_mask
        )
        print(f"Significant differences after FDR correction: "
              f"{low_vs_middle_results['n_significant']}")

        # High vs Middle comparison
        print("\nHigh PGS vs Middle PGS comparison...")
        high_vs_middle_results = perform_group_comparison(
            X_conn_residuals,
            high_pgs_mask,
            middle_pgs_mask
        )
        print(f"Significant differences after FDR correction: "
              f"{high_vs_middle_results['n_significant']}")

        # Low vs High comparison
        print("\nLow PGS vs High PGS comparison...")
        low_vs_high_results = perform_group_comparison(
            X_conn_residuals,
            low_pgs_mask,
            high_pgs_mask
        )
        print(f"Significant differences after FDR correction: "
              f"{low_vs_high_results['n_significant']}")

        # Social difficulty comparisons between groups
        social_low = y_social_scaled[low_pgs_mask]
        social_middle = y_social_scaled[middle_pgs_mask]
        social_high = y_social_scaled[high_pgs_mask]

        # Social difficulty: Low vs Middle
        social_t_low_mid, social_p_low_mid = ttest_ind(social_low, social_middle)

        # Social difficulty: High vs Middle
        social_t_high_mid, social_p_high_mid = ttest_ind(social_high, social_middle)

        # Social difficulty: Low vs High
        social_t_low_high, social_p_low_high = ttest_ind(social_low, social_high)

        results = {
            'low_vs_middle': {
                'results': low_vs_middle_results,
                'description': 'Low PGS (<-1 SD) vs. Middle PGS (>-0.5 SD & <0.5 SD)',
                'social_difficulty_t': social_t_low_mid,
                'social_difficulty_p': social_p_low_mid,
                'social_difficulty_group1_mean': np.mean(social_low),
                'social_difficulty_group2_mean': np.mean(social_middle)
            },
            'high_vs_middle': {
                'results': high_vs_middle_results,
                'description': 'High PGS (>+1 SD) vs. Middle PGS (>-0.5 SD & <0.5 SD)',
                'social_difficulty_t': social_t_high_mid,
                'social_difficulty_p': social_p_high_mid,
                'social_difficulty_group1_mean': np.mean(social_high),
                'social_difficulty_group2_mean': np.mean(social_middle)
            },
            'low_vs_high': {
                'results': low_vs_high_results,
                'description': 'Low PGS (<-1 SD) vs. High PGS (>+1 SD)',
                'social_difficulty_t': social_t_low_high,
                'social_difficulty_p': social_p_low_high,
                'social_difficulty_group1_mean': np.mean(social_low),
                'social_difficulty_group2_mean': np.mean(social_high)
            }
        }

        return results

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

        for n_nodes in self.parcellation_sizes:
            print(f"\n{'='*20} {n_nodes}-NODE ANALYSIS {'='*20}")

            # Prepare data
            analysis_data = self.prepare_data(n_nodes, motion_threshold)

            # Run univariate analyses
            social_results = self.run_univariate_analysis(analysis_data,
                                                          target='social')
            pgs_results = self.run_univariate_analysis(analysis_data,
                                                       target='pgs')

            # Run interaction analysis
            interaction_results = self.run_interaction_analysis(analysis_data)

            # Run three-group analyses
            group_split_results = self.run_three_group_analysis(analysis_data)

            # Store results
            self.results[n_nodes] = {
                'analysis_data': analysis_data,
                'social_univariate': social_results,
                'pgs_univariate': pgs_results,
                'interaction': interaction_results,
                'group_splits': group_split_results
            }

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
            if n_nodes in self.results:
                results = self.results[n_nodes]
                n_conn = results['analysis_data']['n_connections']
                n_subj = results['analysis_data']['n_subjects']

                social_sig = results['social_univariate']['n_significant']
                pgs_sig = results['pgs_univariate']['n_significant']
                interact_sig = results['interaction']['n_significant']

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
                    f"  PGS associations: {pgs_sig}",
                    f"  Interaction terms: {interact_sig}",
                    f"  Low vs Middle: {low_vs_middle_sig}",
                    f"  High vs Middle: {high_vs_middle_sig}",
                    f"  Low vs High: {low_vs_high_sig}"
                ]
                for detail in details:
                    print(detail)
                    self.report.append(detail)

                # Social difficulty group differences
                low_mid_p = (results['group_splits']['low_vs_middle']
                             ['social_difficulty_p'])
                high_mid_p = (results['group_splits']['high_vs_middle']
                              ['social_difficulty_p'])
                low_high_p = (results['group_splits']['low_vs_high']
                              ['social_difficulty_p'])
                
                p_details = [
                    f"  Social difficulty (Low vs Middle) p-value: {low_mid_p:.4f}",
                    f"  Social difficulty (High vs Middle) p-value: {high_mid_p:.4f}",
                    f"  Social difficulty (Low vs High) p-value: {low_high_p:.4f}"
                ]
                for detail in p_details:
                    print(detail)
                    self.report.append(detail)
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
    parser.add_argument('--pgs', required=False,
                        help='Path to PGS/PGS residuals CSV')
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
        pgs_file=args.pgs,
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

    # Print detailed group-split results summary
    header = "\n" + "="*60
    title = "DETAILED THREE-GROUP RESULTS SUMMARY"
    separator = "="*60
    print(header)
    print(title)
    print(separator)
    report.append(header)
    report.append(title)
    report.append(separator)
    report.append("")

    for n_nodes in args.parcellations:
        if n_nodes in analysis.results:
            results_nodes = analysis.results[n_nodes]
            group_splits = results_nodes['group_splits']
            analysis_data = results_nodes['analysis_data']

            msg = f"\n{n_nodes}-node parcellation:"
            print(msg)
            report.append(msg)
            report.append("-" * 30)
            print("-" * 30)

            # Group sizes
            n_low = np.sum(analysis_data['low_pgs_mask'])
            n_middle = np.sum(analysis_data['middle_pgs_mask'])
            n_high = np.sum(analysis_data['high_pgs_mask'])
            
            sample_info = [
                f"Final sample size: {analysis_data['n_subjects']} subjects",
                "",
                "Group sizes:",
                f"  Low PGS (<-1 SD): {n_low}",
                f"  Middle PGS (>-0.5 SD & <0.5 SD): {n_middle}",
                f"  High PGS (>+1 SD): {n_high}"
            ]
            for info in sample_info:
                print(info)
                report.append(info)

            # Low vs Middle comparison details
            low_mid = group_splits['low_vs_middle']
            low_mid_results = [
                "\nLow vs Middle comparison:",
                f"  Significant connections: {low_mid['results']['n_significant']}",
                f"  Social difficulty difference: t = {low_mid['social_difficulty_t']:.3f}, p = {low_mid['social_difficulty_p']:.4f}",
                f"  Social difficulty means: Low = {low_mid['social_difficulty_group1_mean']:.3f}, Middle = {low_mid['social_difficulty_group2_mean']:.3f}"
            ]
            for line in low_mid_results:
                print(line)
                report.append(line)

            # High vs Middle comparison details
            high_mid = group_splits['high_vs_middle']
            high_mid_results = [
                "\nHigh vs Middle comparison:",
                f"  Significant connections: {high_mid['results']['n_significant']}",
                f"  Social difficulty difference: t = {high_mid['social_difficulty_t']:.3f}, p = {high_mid['social_difficulty_p']:.4f}",
                f"  Social difficulty means: High = {high_mid['social_difficulty_group1_mean']:.3f}, Middle = {high_mid['social_difficulty_group2_mean']:.3f}"
            ]
            for line in high_mid_results:
                print(line)
                report.append(line)

            # Low vs High comparison details
            low_high = group_splits['low_vs_high']
            low_high_results = [
                "\nLow vs High comparison:",
                f"  Significant connections: {low_high['results']['n_significant']}",
                f"  Social difficulty difference: t = {low_high['social_difficulty_t']:.3f}, p = {low_high['social_difficulty_p']:.4f}",
                f"  Social difficulty means: Low = {low_high['social_difficulty_group1_mean']:.3f}, High = {low_high['social_difficulty_group2_mean']:.3f}"
            ]
            for line in low_high_results:
                print(line)
                report.append(line)

            # Effect size ranges for significant connections
            for comparison_name, comparison_data in [('Low vs Middle', low_mid),
                                                     ('High vs Middle', high_mid),
                                                     ('Low vs High', low_high)]:
                if comparison_data['results']['n_significant'] > 0:
                    effects = (comparison_data['results']['effect_sizes']
                               [comparison_data['results']['significant_indices']])
                    effect_msg = (f"  {comparison_name} effect sizes: "
                                 f"range = [{np.min(effects):.3f}, "
                                 f"{np.max(effects):.3f}]")
                    print(effect_msg)
                    report.append(effect_msg)

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