# %%
# ==============================================================================
# Title: C2_run_multivariate_fMRI_prediction.py
# ==============================================================================
# Description: This script performs multivariate statistical analyses on brain
# connectivity data to examine associations with social difficulty scores and
# polygenic risk. It includes three-group comparisons based on polygenic score
# thresholds: Low PGS: <-1 SD, Middle PGS: >-0.5 SD & <0.5 SD, High PGS: >1 SD
# ==============================================================================

import connectome_viz
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn import plotting
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_predict, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample, shuffle
import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
import time

# Set up plotting parameters
import matplotlib.pyplot as plt
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
def load_connectivity_data(n_nodes=100):
    """Load connectivity matrices for specified number of nodes"""
    matrix_file = (project_folder /
                   f'data/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats2.txt')
    id_file = project_folder / 'data/subjectIDs_anonymised.txt'

    mats_df = pd.read_csv(matrix_file, header=None, sep='\s+')

    # Extract upper triangle for non-redundancy
    upper_indices = np.triu_indices(n_nodes, k=1)
    linear_indices = upper_indices[0] * n_nodes + upper_indices[1]
    mats_df = mats_df.iloc[:, linear_indices]

    # Rename features and add participant IDs
    mats_df.columns = ['conn_' + str(i+1) for i in range(mats_df.shape[1])]
    ids = pd.read_csv(id_file, header=None)[0].values
    mats_df.set_index(ids, inplace=True)

    print(f'Loaded connectivity data for {mats_df.shape[0]} subjects with '
          f'{n_nodes} nodes ({mats_df.shape[1]} connections)')
    return mats_df

def bootstrap_pls_coefficients(X, y, n_components, B=1000, random_state=42):
    """
    Returns lower and upper 95% confidence intervals for PLS regression
    coefficients via bootstrapping.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    coefs_array = np.zeros((B, n_features))

    for i in range(B):
        # Draw a bootstrap sample (with replacement)
        indices = rng.randint(0, n_samples, size=n_samples)
        X_resampled = X[indices]
        y_resampled = y[indices]

        # Fit PLS to the resampled data
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_resampled, y_resampled)

        # Extract the regression coefficients
        coefs_array[i, :] = pls.coef_.ravel()

    # Compute the 2.5th and 97.5th percentiles for each feature's coefficients
    ci_lower = np.percentile(coefs_array, 2.5, axis=0)
    ci_upper = np.percentile(coefs_array, 97.5, axis=0)

    return ci_lower, ci_upper

def regress_out(X, covariates):
    """Regress out covariates from each column of X"""
    X_residuals = np.zeros_like(X)
    for i in range(X.shape[1]):
        model = LinearRegression().fit(covariates, X[:, i])
        X_residuals[:, i] = X[:, i] - model.predict(covariates)
    return X_residuals

def bootstrap_group_comparison(X, y, group_masks, group_names, pls_model,
                               n_bootstrap=1000, subset_size=None,
                               random_state=42):
    """
    Compare PLS prediction performance across polygenic risk groups using
    bootstrap sampling

    Parameters:
    -----------
    X : array-like
        Feature matrix (connectivity data)
    y : array-like
        Target variable (social difficulty scores)
    group_masks : list of arrays
        Boolean masks for each group
    group_names : list of str
        Names for each group
    pls_model : PLSRegression
        Pre-trained PLS model to apply to subsets
    n_bootstrap : int
        Number of bootstrap iterations
    subset_size : int or None
        Size of equal subsets to sample from each group. If None, use minimum
        group size
    random_state : int
        Random seed
    """
    print(f"\nBootstrap group comparison: {' vs '.join(group_names)}")

    # Determine subset size
    group_sizes = [np.sum(mask) for mask in group_masks]
    if subset_size is None:
        subset_size = min(group_sizes)

    print(f"Group sizes: {dict(zip(group_names, group_sizes))}")
    print(f"Using subset size: {subset_size}")

    # Storage for bootstrap results
    bootstrap_results = {name: [] for name in group_names}
    bootstrap_differences = []

    rng = np.random.RandomState(random_state)

    print("Running bootstrap iterations...")
    for i in range(n_bootstrap):
        iteration_results = {}

        # Sample equal-sized subsets from each group
        for group_name, group_mask in zip(group_names, group_masks):
            group_indices = np.where(group_mask)[0]

            # Sample subset_size indices from this group
            if len(group_indices) >= subset_size:
                sampled_indices = rng.choice(group_indices, size=subset_size,
                                             replace=True)
            else:
                # If group is smaller than subset_size, use all and sample
                # with replacement
                sampled_indices = rng.choice(group_indices, size=subset_size,
                                             replace=True)

            # Get data for this subset
            X_subset = X[sampled_indices]
            y_subset = y[sampled_indices]

            # Apply pre-trained model directly (no re-training)
            y_pred = pls_model.predict(X_subset).ravel()

            # Calculate R²
            r2 = r2_score(y_subset, y_pred)
            iteration_results[group_name] = r2
            bootstrap_results[group_name].append(r2)

        # Calculate difference (assuming two groups)
        if len(group_names) == 2:
            diff = (iteration_results[group_names[0]] -
                    iteration_results[group_names[1]])
            bootstrap_differences.append(diff)

        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} iterations...")

    # Calculate summary statistics
    results = {}
    for group_name in group_names:
        group_r2 = np.array(bootstrap_results[group_name])
        results[group_name] = {
            'mean_r2': np.mean(group_r2),
            'std_r2': np.std(group_r2),
            'ci_lower': np.percentile(group_r2, 2.5),
            'ci_upper': np.percentile(group_r2, 97.5),
            'bootstrap_values': group_r2
        }

    # Calculate difference statistics (for two groups)
    if len(group_names) == 2:
        diff_array = np.array(bootstrap_differences)
        results['difference'] = {
            'mean_diff': np.mean(diff_array),
            'ci_lower': np.percentile(diff_array, 2.5),
            'ci_upper': np.percentile(diff_array, 97.5),
            'p_value': (np.mean(diff_array >= 0) if np.mean(diff_array) < 0
                        else np.mean(diff_array <= 0)),
            'significant': not (np.percentile(diff_array, 2.5) <= 0 <=
                                np.percentile(diff_array, 97.5)),
            'bootstrap_values': diff_array
        }

        print(f"\nResults:")
        for group_name in group_names:
            print(f"{group_name}: R² = "
                  f"{results[group_name]['mean_r2']:.3f} ± "
                  f"{results[group_name]['std_r2']:.3f}")
            print(f"  95% CI: [{results[group_name]['ci_lower']:.3f}, "
                  f"{results[group_name]['ci_upper']:.3f}]")

        print(f"\nDifference ({group_names[0]} - {group_names[1]}):")
        print(f"  Mean: {results['difference']['mean_diff']:.3f}")
        print(f"  95% CI: [{results['difference']['ci_lower']:.3f}, "
              f"{results['difference']['ci_upper']:.3f}]")
        print(f"  p-value: {results['difference']['p_value']:.4f}")
        print(f"  Significant: {results['difference']['significant']}")

    return results

def run_single_pls_model(X, y, model_name, cv, n_components_range=range(1, 11),
                         permutation_data=None, n_permutations=1000):
    """Run PLS analysis for a single model type"""
    print(f"\n{'-'*40}")
    print(f"Model: {model_name}")
    print(f"Features: {X.shape[1]}")
    print(f"{'-'*40}")

    # Find optimal number of components
    component_scores = []
    print("Finding optimal number of components...")
    for n_comp in n_components_range:
        pls = PLSRegression(n_components=n_comp, scale=False)
        cv_scores = cross_val_score(pls, X, y, cv=cv,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        component_scores.append(cv_rmse)
        print(f"  Components: {n_comp:2d}, CV RMSE: {cv_rmse:.4f} ± "
              f"{np.sqrt(-cv_scores).std():.4f}")

    # Find best number of components
    best_components = n_components_range[np.argmin(component_scores)]
    best_rmse = np.min(component_scores)
    print(f"Optimal: {best_components} components with CV RMSE: "
          f"{best_rmse:.4f}")

    # Final cross-validation with optimal components
    pls = PLSRegression(n_components=best_components, scale=False)
    y_pred_cv = cross_val_predict(pls, X, y, cv=cv)

    # Calculate metrics
    r2_cv = r2_score(y, y_pred_cv)
    mae_cv = mean_absolute_error(y, y_pred_cv)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    corr_cv, p_val_cv = pearsonr(y, y_pred_cv)

    print(f"Cross-validated results:")
    print(f"R² = {r2_cv:.3f}")
    print(f"Correlation = {corr_cv:.3f}, p = {p_val_cv:.3e}")
    print(f"RMSE = {rmse_cv:.3f}")

    # Train final model
    final_pls = PLSRegression(n_components=best_components, scale=False)
    final_pls.fit(X, y)

    # Permutation test
    permuted_r2 = []
    if permutation_data is not None:
        print("Running permutation test...")
        start_time = time.time()

        for i in range(n_permutations):
            # Create permuted features based on model type
            if model_name == "Connectivity Only":
                # For connectivity-only model, just shuffle the outcome
                y_perm = shuffle(y, random_state=i)
                X_perm = X
            elif model_name == "Interaction Only":
                # For interaction model, shuffle PRS to break interaction
                prs_perm = shuffle(permutation_data['prs'], random_state=i)
                X_perm = np.zeros_like(X)
                for j in range(X.shape[1]):
                    X_perm[:, j] = (permutation_data['conn_residuals'][:, j] *
                                    prs_perm)
                y_perm = y

            # Get cross-validated predictions for permuted data
            pls_perm = PLSRegression(n_components=best_components, scale=False)
            y_pred_perm = cross_val_predict(pls_perm, X_perm, y_perm, cv=cv)
            permuted_r2.append(r2_score(y_perm, y_pred_perm))

            if (i+1) % 200 == 0:
                print(f"  Completed {i+1}/{n_permutations} permutations...")

        p_value = np.mean([1 if perm_r2 >= r2_cv else 0
                           for perm_r2 in permuted_r2])
        print(f"Permutation test completed in "
              f"{time.time() - start_time:.1f} seconds")
        print(f"Permutation p-value = {p_value:.4f}")
    else:
        p_value = None

    # Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    ci_lower, ci_upper = bootstrap_pls_coefficients(X, y, best_components,
                                                    B=1000)

    # Count significant coefficients
    positive_coefs = np.sum((ci_lower > 0) & (ci_upper > 0))
    negative_coefs = np.sum((ci_lower < 0) & (ci_upper < 0))
    total_coefs = len(ci_lower)
    
    # Identify significant coefficients and their indices
    significant_mask = (ci_lower > 0) & (ci_upper > 0) | (ci_lower < 0) & (ci_upper < 0)
    significant_indices = np.where(significant_mask)[0]
    significant_weights = final_pls.coef_.ravel()[significant_mask]
    positive_indices = np.where((ci_lower > 0) & (ci_upper > 0))[0]
    negative_indices = np.where((ci_lower < 0) & (ci_upper < 0))[0]

    print(f"Significant coefficients:")
    print(f"  Positive: {positive_coefs} "
          f"({100*positive_coefs/total_coefs:.1f}%)")
    print(f"  Negative: {negative_coefs} "
          f"({100*negative_coefs/total_coefs:.1f}%)")

    return {
        'model_name': model_name,
        'n_features': X.shape[1],
        'best_components': best_components,
        'r2_cv': r2_cv,
        'corr_cv': corr_cv,
        'p_val_cv': p_val_cv,
        'rmse_cv': rmse_cv,
        'mae_cv': mae_cv,
        'permutation_p': p_value,
        'positive_coefs': positive_coefs,
        'negative_coefs': negative_coefs,
        'total_coefs': total_coefs,
        'final_pls': final_pls,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'component_scores': component_scores,
        'permuted_r2': permuted_r2,
        'y_pred_cv': y_pred_cv,
        'significant_mask': significant_mask,
        'significant_indices': significant_indices,
        'significant_weights': significant_weights,
        'positive_indices': positive_indices,
        'negative_indices': negative_indices,
        'all_weights': final_pls.coef_.ravel()
    }

def run_pls_analysis(merged_df, n_nodes=100,
                     n_components_range=range(1, 11)):
    """Run complete PLS analysis for given number of nodes with separate models"""
    print(f"\n{'='*60}")
    print(f"Running PLS analysis with {n_nodes} nodes")
    print(f"{'='*60}")

    # Extract connectivity features
    conn_features = [col for col in merged_df.columns
                     if col.startswith('conn')]
    X_conn = merged_df[conn_features].values

    # Extract covariates to regress out
    covariates = ['Gender', 'Age_in_Yrs', 'FS_IntraCranial_Vol',
                  'Movement_RelativeRMS_mean']
    X_cov = pd.get_dummies(merged_df[covariates], drop_first=True)

    # Extract outcome and PRS
    y = merged_df['Social_Score'].values
    prs = merged_df['blup_PRS_residuals'].values

    # Standardize all data
    scaler = RobustScaler()
    X_conn_scaled = scaler.fit_transform(X_conn)
    X_cov_scaled = scaler.fit_transform(X_cov)
    y_scaled = zscore(y)
    prs_scaled = zscore(prs)

    # Regress out covariates from connectivity data
    X_conn_residuals = regress_out(X_conn_scaled, X_cov_scaled)

    # Create interaction terms between PRS and connectivity features
    X_interaction = np.zeros_like(X_conn_residuals)
    for i in range(X_conn_residuals.shape[1]):
        X_interaction[:, i] = X_conn_residuals[:, i] * prs_scaled

    # Set up cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Prepare permutation data
    perm_data = {
        'conn_residuals': X_conn_residuals,
        'prs': prs_scaled
    }

    # Run separate models
    models = {}

    # Model 1: Connectivity features only
    models['connectivity'] = run_single_pls_model(
        X_conn_residuals, y_scaled, "Connectivity Only", cv,
        n_components_range, permutation_data=None
    )

    # Model 2: Interaction features only
    models['interaction'] = run_single_pls_model(
        X_interaction, y_scaled, "Interaction Only", cv,
        n_components_range, permutation_data=perm_data
    )

    # Create three PGS group masks for bootstrap comparison
    pgs_mean = np.mean(prs_scaled)
    pgs_std = np.std(prs_scaled)

    # Three groups: Low (<-1 SD), Middle (>-0.5 & <0.5 SD), High (>+1 SD)
    low_pgs_mask = prs_scaled < (pgs_mean - 1 * pgs_std)
    middle_pgs_mask = ((prs_scaled > (pgs_mean - 0.5 * pgs_std)) &
                       (prs_scaled < (pgs_mean + 0.5 * pgs_std)))
    high_pgs_mask = prs_scaled > (pgs_mean + 1 * pgs_std)

    # Define the three pairwise group comparisons
    group_comparisons = {
        'low_vs_middle': {
            'masks': [low_pgs_mask, middle_pgs_mask],
            'names': ['Low PGS (<-1 SD)', 'Middle PGS (>-0.5 & <0.5 SD)']
        },
        'high_vs_middle': {
            'masks': [high_pgs_mask, middle_pgs_mask],
            'names': ['High PGS (>+1 SD)', 'Middle PGS (>-0.5 & <0.5 SD)']
        },
        'low_vs_high': {
            'masks': [low_pgs_mask, high_pgs_mask],
            'names': ['Low PGS (<-1 SD)', 'High PGS (>+1 SD)']
        }
    }

    # Print group sizes
    print(f"\nThree-group stratification:")
    print(f"Low PGS (<-1 SD): {np.sum(low_pgs_mask)} subjects")
    print(f"Middle PGS (>-0.5 & <0.5 SD): {np.sum(middle_pgs_mask)} subjects")
    print(f"High PGS (>+1 SD): {np.sum(high_pgs_mask)} subjects")

    # Run bootstrap group comparisons for connectivity model
    print(f"\n{'='*60}")
    print("BOOTSTRAP GROUP COMPARISONS - CONNECTIVITY MODEL")
    print(f"{'='*60}")

    bootstrap_results = {}
    connectivity_model = models['connectivity']['final_pls']

    for comparison_name, comparison_info in group_comparisons.items():
        bootstrap_results[comparison_name] = bootstrap_group_comparison(
            X_conn_residuals, y_scaled,
            comparison_info['masks'], comparison_info['names'],
            connectivity_model, n_bootstrap=1000
        )

    # Store combined results
    results = {
        'n_nodes': n_nodes,
        'conn_features': conn_features,
        'y_scaled': y_scaled,
        'models': models,
        'bootstrap_results': bootstrap_results,
        'X_conn_residuals': X_conn_residuals,
        'X_interaction': X_interaction,
        'prs_scaled': prs_scaled,
        'group_comparisons': group_comparisons,
        'group_sizes': {
            'low': np.sum(low_pgs_mask),
            'middle': np.sum(middle_pgs_mask),
            'high': np.sum(high_pgs_mask)
        }
    }

    return results


def create_pls_surface_visualization(results, n_nodes, model_type='connectivity', 
                                    project_folder=None, output_filename=None,
                                    cmap='RdBu_r', sign_filter=None, fixed_color_range=None):
    """
    Create surface visualization showing summed edge weights for each node from PLS analysis
    
    Parameters:
    -----------
    results : dict
        Results from run_pls_analysis containing models
    n_nodes : int
        Number of nodes in the parcellation
    model_type : str
        Which model to visualize ('connectivity' or 'interaction')
    project_folder : Path
        Path to project folder for surface data
    output_filename : str
        Output filename for the plot
    cmap : str
        Colormap for visualization ('RdBu_r', 'coolwarm', etc.)
    sign_filter : str or None
        Filter by sign: 'positive', 'negative', or None for both
    fixed_color_range : tuple or None
        Fixed color range (min, max) for standardized scaling across plots
    """
    from neuromaps.datasets import fetch_fslr
    from surfplot import Plot
    import connectome_viz
    
    # Get model results
    model_results = results['models'][model_type]
    
    # Get significant weights and indices
    significant_indices = model_results['significant_indices']
    significant_weights = model_results['significant_weights']
    
    if len(significant_indices) == 0:
        print(f"No significant weights found for {model_type} model")
        return None
    
    # Apply sign filtering if requested
    if sign_filter == 'positive':
        pos_mask = significant_weights > 0
        filtered_indices = significant_indices[pos_mask]
        filtered_weights = significant_weights[pos_mask]
    elif sign_filter == 'negative':
        neg_mask = significant_weights < 0
        filtered_indices = significant_indices[neg_mask]
        filtered_weights = significant_weights[neg_mask]
    else:
        filtered_indices = significant_indices
        filtered_weights = significant_weights
    
    if len(filtered_indices) == 0:
        print(f"No {sign_filter} significant weights found for {model_type} model")
        return None
    
    # Create connectivity matrix from significant weights
    matrix = np.zeros((n_nodes, n_nodes))
    
    # Convert connectivity indices to matrix indices
    row_indices, col_indices = connectome_viz.connectivity_indices_to_matrix_indices(
        filtered_indices, n_nodes)
    
    # Fill the matrix (make it symmetric)
    for idx in range(len(filtered_indices)):
        i, j = row_indices[idx], col_indices[idx]
        weight = filtered_weights[idx]
        matrix[i, j] = weight
        matrix[j, i] = weight  # Make symmetric
    
    # Calculate node-wise sums (sum of absolute weights for each node)
    if sign_filter == 'positive':
        node_sums = np.sum(matrix, axis=1)  # Sum positive weights
        cbar_label = f'Sum of Positive PLS Weights ({model_type.title()})'
    elif sign_filter == 'negative':
        node_sums = np.sum(matrix, axis=1)  # Sum negative weights (will be negative)
        cbar_label = f'Sum of Negative PLS Weights ({model_type.title()})'
    else:
        node_sums = np.sum(np.abs(matrix), axis=1)  # Sum absolute weights for all
        cbar_label = f'Sum of |PLS Weights| ({model_type.title()})'
    
    print(f"Node sums range: {np.min(node_sums):.3f} to {np.max(node_sums):.3f}")
    
    if project_folder is None:
        print("Error: project_folder is required for surface visualization")
        return None
    
    # Load surface data
    dlabel_filename = (project_folder /
                       f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_ftb.dlabel.nii')
    dscalar_filename = (project_folder /
                        f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC.dscalar.nii')

    try:
        left_full, right_full = connectome_viz.mask_dlabel_data(dscalar_filename,
                                                                dlabel_filename)
        lh_val, rh_val = left_full.copy(), right_full.copy()
        
        # Map node values to surface
        for i in range(len(node_sums)):
            lh_val[left_full == i+1] = node_sums[i]
            rh_val[right_full == i+1] = node_sums[i]
        
        # Create surface plot
        width, height = int(120 * 300 / 25.4), int(90 * 300 / 25.4)
        surfaces = fetch_fslr()
        lh, rh = surfaces['inflated']
        
        p = Plot(lh, rh, zoom=1.5, size=(width, height))
        
        # Set color range - use fixed range if provided, otherwise auto-scale
        if fixed_color_range is not None:
            color_range = fixed_color_range
            print(f"Using fixed color range: {color_range[0]:.3f} to {color_range[1]:.3f}")
        else:
            # Auto-scale based on data
            if sign_filter == 'positive':
                max_val = np.max(node_sums) if np.max(node_sums) > 0 else 1
                color_range = (0, max_val)
            elif sign_filter == 'negative':
                min_val = np.min(node_sums) if np.min(node_sums) < 0 else -1
                color_range = (min_val, 0)
            else:
                max_val = np.max(node_sums) if np.max(node_sums) > 0 else 1
                color_range = (0, max_val)
        
        # Set colormap based on sign filter
        if sign_filter == 'positive':
            cmap = 'Reds'
        elif sign_filter == 'negative':
            cmap = 'Blues_r'
        else:
            cmap = 'Reds'
        
        p.add_layer({'left': lh_val, 'right': rh_val},
                    cmap=cmap,
                    color_range=color_range,
                    cbar=True,
                    cbar_label=cbar_label
                    )
        p.add_layer({'left': left_full, 'right': right_full},
                    cmap='gray', as_outline=True, cbar=False)
        
        cbar_kws = {'fontsize': 12, 'pad': 0.05}
        fig = p.build(cbar_kws=cbar_kws)
        
        if output_filename:
            fig.savefig(output_filename, dpi=300, bbox_inches='tight',
                        pad_inches=0.1)
        
        print(f"Created surface visualization for {model_type} model")
        print(f"  - Significant connections: {len(filtered_indices)}")
        print(f"  - Node range: {np.min(node_sums):.3f} to {np.max(node_sums):.3f}")
        print(f"  - Color range: {color_range[0]:.3f} to {color_range[1]:.3f}")
        if sign_filter:
            print(f"  - Sign filter: {sign_filter}")
        
        return fig
        
    except Exception as e:
        print(f"Error creating surface plot: {e}")
        return None


def create_pls_component_scatter_plot(results, n_nodes, model_type='connectivity',
                                     project_folder=None, output_filename=None):
    """
    Create scatter plot showing association between PLS component scores and social difficulty
    
    Parameters:
    -----------
    results : dict
        Results from run_pls_analysis containing models and data
    n_nodes : int
        Number of nodes in the parcellation
    model_type : str
        Which model to visualize ('connectivity' or 'interaction')
    project_folder : Path
        Path to project folder
    output_filename : str
        Output filename for the plot
    """
    # Get model results
    model_results = results['models'][model_type]
    final_pls = model_results['final_pls']
    
    # Get the data used for training
    if model_type == 'connectivity':
        X_data = results['X_conn_residuals']
    else:  # interaction
        X_data = results['X_interaction']
    
    y_data = results['y_scaled']
    
    # Get PLS component scores using the trained model
    X_scores = final_pls.transform(X_data)
    pls_component_1 = X_scores[:, 0]  # First PLS component
    
    # Calculate correlation statistics
    correlation, p_value = pearsonr(pls_component_1, y_data)
    r_squared = correlation**2
    
    print(f"\nPLS Component Analysis ({model_type} model):")
    print(f"  Correlation: r = {correlation:.3f}")
    print(f"  R-squared: R² = {r_squared:.3f}")
    print(f"  P-value: p = {p_value:.3e}")
    print(f"  Sample size: n = {len(y_data)}")
    print(f"  Optimal components: {final_pls.n_components}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(60*mm2inches, 50*mm2inches))
    
    # Create scatter plot
    ax.scatter(pls_component_1, y_data, alpha=0.6, s=12, color='darkgrey',
               edgecolors='none', rasterized=True)
    
    # Add regression line
    z = np.polyfit(pls_component_1, y_data, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(np.min(pls_component_1), np.max(pls_component_1), 100)
    ax.plot(x_line, p_line(x_line), "k-", linewidth=1.5, alpha=0.8)
    
    # Add correlation info as text
    textstr = (f'$r = {correlation:.3f}$\n'
              f'$R^2 = {r_squared:.3f}$\n'
              f'$p = {p_value:.2e}$\n'
              f'$n = {len(y_data)}$')
    
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                 edgecolor='gray', linewidth=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    # Labels and formatting
    ax.set_xlabel(f'PLS Comp 1 Score', fontsize=9)
    ax.set_ylabel('SDS [$z$]', fontsize=9)
    
    # Make axes look nice
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                    pad_inches=0.05)
        print(f"  Saved scatter plot: {output_filename}")
    
    return fig, ax


def create_all_pls_scatter_plots(all_results, project_folder):
    """
    Create scatter plots for all PLS results
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all node configurations
    project_folder : Path
        Path to project folder
    """
    
    for n_nodes, results in all_results.items():
        print(f"\nCreating scatter plots for {n_nodes}-node parcellation...")
        
        # Create scatter plots for connectivity model
        model_type = 'connectivity'
        model_results = results['models'][model_type]
        
        filename = (project_folder /
                    f'figures/PLS_scatter_{model_type}_n{n_nodes}.png')
        create_pls_component_scatter_plot(
            results, n_nodes, model_type, project_folder, filename
        )
        
        # Create scatter plots for interaction model
        model_type = 'interaction'
        model_results = results['models'][model_type]
        
        if model_results['r2_cv'] > 0:  # Only create if model has some predictive power
            filename = (project_folder /
                       f'figures/PLS_scatter_{model_type}_n{n_nodes}.png')
            create_pls_component_scatter_plot(
                results, n_nodes, model_type, project_folder, filename
            )


def create_all_pls_visualizations(all_results, project_folder):
    """
    Create surface visualizations for all PLS results with standardized color scales
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all node configurations
    project_folder : Path
        Path to project folder
    """
    
    # First pass: Calculate global color ranges across all results
    print("Calculating global color ranges...")
    
    global_ranges = {
        'positive': {'min': float('inf'), 'max': -float('inf')},
        'negative': {'min': float('inf'), 'max': -float('inf')},
        'absolute': {'min': 0, 'max': -float('inf')}
    }
    
    for n_nodes, results in all_results.items():
        for model_type in ['connectivity', 'interaction']:
            model_results = results['models'][model_type]
            
            if len(model_results['significant_indices']) == 0:
                continue
                
            # Calculate node sums for this model
            significant_indices = model_results['significant_indices']
            significant_weights = model_results['significant_weights']
            
            # Create connectivity matrix from significant weights
            matrix = np.zeros((n_nodes, n_nodes))
            row_indices, col_indices = connectome_viz.connectivity_indices_to_matrix_indices(
                significant_indices, n_nodes)
            
            for idx in range(len(significant_indices)):
                i, j = row_indices[idx], col_indices[idx]
                weight = significant_weights[idx]
                matrix[i, j] = weight
                matrix[j, i] = weight
            
            # Calculate different node sum types
            positive_mask = significant_weights > 0
            negative_mask = significant_weights < 0
            
            if np.any(positive_mask):
                pos_matrix = np.zeros((n_nodes, n_nodes))
                pos_indices = significant_indices[positive_mask]
                pos_weights = significant_weights[positive_mask]
                pos_rows, pos_cols = connectome_viz.connectivity_indices_to_matrix_indices(
                    pos_indices, n_nodes)
                for idx in range(len(pos_indices)):
                    i, j = pos_rows[idx], pos_cols[idx]
                    pos_matrix[i, j] = pos_weights[idx]
                    pos_matrix[j, i] = pos_weights[idx]
                
                pos_sums = np.sum(pos_matrix, axis=1)
                global_ranges['positive']['min'] = min(global_ranges['positive']['min'], np.min(pos_sums))
                global_ranges['positive']['max'] = max(global_ranges['positive']['max'], np.max(pos_sums))
            
            if np.any(negative_mask):
                neg_matrix = np.zeros((n_nodes, n_nodes))
                neg_indices = significant_indices[negative_mask]
                neg_weights = significant_weights[negative_mask]
                neg_rows, neg_cols = connectome_viz.connectivity_indices_to_matrix_indices(
                    neg_indices, n_nodes)
                for idx in range(len(neg_indices)):
                    i, j = neg_rows[idx], neg_cols[idx]
                    neg_matrix[i, j] = neg_weights[idx]
                    neg_matrix[j, i] = neg_weights[idx]
                
                neg_sums = np.sum(neg_matrix, axis=1)
                global_ranges['negative']['min'] = min(global_ranges['negative']['min'], np.min(neg_sums))
                global_ranges['negative']['max'] = max(global_ranges['negative']['max'], np.max(neg_sums))
            
            # Absolute sums
            abs_sums = np.sum(np.abs(matrix), axis=1)
            global_ranges['absolute']['max'] = max(global_ranges['absolute']['max'], np.max(abs_sums))
    
    # Handle cases where no significant weights exist for a category
    if global_ranges['positive']['min'] == float('inf'):
        global_ranges['positive'] = {'min': 0, 'max': 1}
    if global_ranges['negative']['min'] == float('inf'):
        global_ranges['negative'] = {'min': -1, 'max': 0}
    if global_ranges['absolute']['max'] == -float('inf'):
        global_ranges['absolute']['max'] = 1
    
    # Set negative range to be symmetric around zero
    global_ranges['negative']['max'] = 0
    
    print(f"Global ranges calculated:")
    print(f"  Positive: {global_ranges['positive']['min']:.3f} to {global_ranges['positive']['max']:.3f}")
    print(f"  Negative: {global_ranges['negative']['min']:.3f} to {global_ranges['negative']['max']:.3f}")
    print(f"  Absolute: {global_ranges['absolute']['min']:.3f} to {global_ranges['absolute']['max']:.3f}")
    
    # Second pass: Create visualizations with standardized scales
    for n_nodes, results in all_results.items():
        print(f"\nCreating visualizations for {n_nodes}-node parcellation...")
        
        # Create visualizations for connectivity model
        model_type = 'connectivity'
        model_results = results['models'][model_type]
        
        if model_results['positive_coefs'] > 0:
            filename = (project_folder / 
                       f'figures/PLS_surface_{model_type}_positive_n{n_nodes}.png')
            try:
                create_pls_surface_visualization(
                    results, n_nodes, model_type, project_folder, filename,
                    sign_filter='positive', 
                    fixed_color_range=(global_ranges['positive']['min'], global_ranges['positive']['max'])
                )
            except Exception as e:
                print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")
        
        if model_results['negative_coefs'] > 0:
            filename = (project_folder / 
                       f'figures/PLS_surface_{model_type}_negative_n{n_nodes}.png')
            try:
                create_pls_surface_visualization(
                    results, n_nodes, model_type, project_folder, filename,
                    sign_filter='negative',
                    fixed_color_range=(global_ranges['negative']['min'], global_ranges['negative']['max'])
                )
            except Exception as e:
                print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")
        
        # All significant weights combined
        if model_results['positive_coefs'] + model_results['negative_coefs'] > 0:
            filename = (project_folder / 
                       f'figures/PLS_surface_{model_type}_all_n{n_nodes}.png')
            try:
                create_pls_surface_visualization(
                    results, n_nodes, model_type, project_folder, filename,
                    sign_filter=None,
                    fixed_color_range=(global_ranges['absolute']['min'], global_ranges['absolute']['max'])
                )
            except Exception as e:
                print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")
        
        # Create visualizations for interaction model if it has significant weights
        model_type = 'interaction'
        model_results = results['models'][model_type]
        
        if model_results['positive_coefs'] + model_results['negative_coefs'] > 0:
            if model_results['positive_coefs'] > 0:
                filename = (project_folder / 
                           f'figures/PLS_surface_{model_type}_positive_n{n_nodes}.png')
                try:
                    create_pls_surface_visualization(
                        results, n_nodes, model_type, project_folder, filename,
                        sign_filter='positive',
                        fixed_color_range=(global_ranges['positive']['min'], global_ranges['positive']['max'])
                    )
                except Exception as e:
                    print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")
            
            if model_results['negative_coefs'] > 0:
                filename = (project_folder / 
                           f'figures/PLS_surface_{model_type}_negative_n{n_nodes}.png')
                try:
                    create_pls_surface_visualization(
                        results, n_nodes, model_type, project_folder, filename,
                        sign_filter='negative',
                        fixed_color_range=(global_ranges['negative']['min'], global_ranges['negative']['max'])
                    )
                except Exception as e:
                    print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")

            # All significant weights combined
            filename = (project_folder /
                       f'figures/PLS_surface_{model_type}_all_n{n_nodes}.png')
            try:
                create_pls_surface_visualization(
                    results, n_nodes, model_type, project_folder, filename,
                    sign_filter=None,
                    fixed_color_range=(global_ranges['absolute']['min'], global_ranges['absolute']['max'])
                )
            except Exception as e:
                print(f"Error creating surface plot for {comparison_name}: {e}. Are you running this in a headless environment?")

def main():
    # Define paths
    project_folder = Path(__file__).resolve().parents[1]
    behavioural_file = (project_folder /
                        "data/behavioural_data_anonymised.csv")
    phenotypic_file = (project_folder /
                    "data/phenotypic_data_anonymised.csv")
    PRS_file = project_folder / "data/prs_residuals.csv"
    social_file = (project_folder /
                'data/cfa_factor_scores_full_sample.csv')

    # Load other data (same for all analyses)
    prs_df = pd.read_csv(PRS_file)
    social_df = pd.read_csv(social_file)
    behavioural_df = pd.read_csv(behavioural_file)
    phenotypic_df = pd.read_csv(phenotypic_file)
    movement_df = pd.read_csv(project_folder /
                            'data/movement_data_anonymised.csv')

    # Run analyses for different numbers of nodes
    node_configurations = [50, 100, 200]  # 100 is main analysis, 50 and 200 are sensitivity
    all_results = {}

    for n_nodes in node_configurations:
        # Load connectivity data for this configuration
        mats_df = load_connectivity_data(n_nodes)

        # Prepare merged dataset
        merged_df = pd.merge(prs_df, mats_df, left_on='Subject',
                            right_index=True)
        merged_df = pd.merge(merged_df,
                            social_df[['Subject', 'Social_Score']],
                            on='Subject')
        merged_df = pd.merge(merged_df,
                            behavioural_df[['Subject', 'Gender',
                                            'FS_IntraCranial_Vol']],
                            on='Subject')
        merged_df = pd.merge(merged_df,
                            phenotypic_df[['Subject', 'Age_in_Yrs']],
                            on='Subject')
        merged_df = pd.merge(merged_df,
                            movement_df[['Subject',
                                        'Movement_RelativeRMS_mean']],
                            on='Subject')

        # Filter and clean data
        merged_df = merged_df.loc[merged_df['Movement_RelativeRMS_mean'] < 0.2]
        merged_df = merged_df.dropna()
        merged_df.to_csv(project_folder / 'data/merged_fMRI_data.csv', index=False)

        print(f'Final sample size for {n_nodes} nodes: '
            f'{merged_df.shape[0]} subjects')

        # Run PLS analysis
        results = run_pls_analysis(merged_df, n_nodes)
        all_results[n_nodes] = results

    # Create comprehensive summary table including bootstrap results
    summary_data = []
    bootstrap_summary_data = []

    for n_nodes, results in all_results.items():
        # Model performance summary
        for model_type, model_results in results['models'].items():
            summary_data.append({
                'N_Nodes': n_nodes,
                'Model_Type': model_results['model_name'],
                'N_Features': model_results['n_features'],
                'Best_Components': model_results['best_components'],
                'CV_R2': model_results['r2_cv'],
                'CV_Correlation': model_results['corr_cv'],
                'CV_P_Value': model_results['p_val_cv'],
                'CV_RMSE': model_results['rmse_cv'],
                'Permutation_P': model_results['permutation_p'],
                'Positive_Coefs': model_results['positive_coefs'],
                'Negative_Coefs': model_results['negative_coefs'],
                'Percent_Significant': (100 *
                                        (model_results['positive_coefs'] +
                                        model_results['negative_coefs']) /
                                        model_results['total_coefs'])
            })

        # Bootstrap comparison summary
        for comparison_name, bootstrap_result in results['bootstrap_results'].items():
            for group_name in bootstrap_result.keys():
                if group_name != 'difference':
                    bootstrap_summary_data.append({
                        'N_Nodes': n_nodes,
                        'Comparison': comparison_name,
                        'Group': group_name,
                        'Mean_R2': bootstrap_result[group_name]['mean_r2'],
                        'Std_R2': bootstrap_result[group_name]['std_r2'],
                        'CI_Lower': bootstrap_result[group_name]['ci_lower'],
                        'CI_Upper': bootstrap_result[group_name]['ci_upper']
                    })

            # Add difference results if available
            if 'difference' in bootstrap_result:
                bootstrap_summary_data.append({
                    'N_Nodes': n_nodes,
                    'Comparison': comparison_name,
                    'Group': 'Difference',
                    'Mean_R2': bootstrap_result['difference']['mean_diff'],
                    'Std_R2': np.std(bootstrap_result['difference']['bootstrap_values']),
                    'CI_Lower': bootstrap_result['difference']['ci_lower'],
                    'CI_Upper': bootstrap_result['difference']['ci_upper'],
                    'P_Value': bootstrap_result['difference']['p_value'],
                    'Significant': bootstrap_result['difference']['significant']
                })

    summary_df = pd.DataFrame(summary_data)
    bootstrap_summary_df = pd.DataFrame(bootstrap_summary_data)

    print("\n" + "="*100)
    print("SUMMARY OF MODEL PERFORMANCE ACROSS NODE CONFIGURATIONS")
    print("="*100)
    print(summary_df.round(3))

    print("\n" + "="*100)
    print("SUMMARY OF BOOTSTRAP GROUP COMPARISONS")
    print("="*100)
    print(bootstrap_summary_df.round(3))

    # Save summaries
    summary_df.to_csv(project_folder / 'results/pls_analysis_summary_all_models.csv',
                    index=False)
    bootstrap_summary_df.to_csv(project_folder /
                                'results/pls_bootstrap_group_comparisons.csv',
                                index=False)

    # Create visualization for bootstrap group comparisons (100 nodes)
    main_results = all_results[100]
    bootstrap_results = main_results['bootstrap_results']

    # Create bootstrap comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(160*mm2inches, 120*mm2inches))
    axes = axes.flatten()

    comparison_titles = {
        'low_vs_middle': 'Low PGS (<-1 SD) vs Middle PGS (>-0.5 & <0.5 SD)',
        'high_vs_middle': 'High PGS (>+1 SD) vs Middle PGS (>-0.5 & <0.5 SD)',
        'low_vs_high': 'Low PGS (<-1 SD) vs High PGS (>+1 SD)'
    }

    for i, (comp_name, comp_title) in enumerate(comparison_titles.items()):
        if i < 3:  # Only plot the three comparisons
            ax = axes[i]
            result = bootstrap_results[comp_name]

            # Plot distributions for each group
            group_names = list(result.keys())
            if 'difference' in group_names:
                group_names.remove('difference')

            positions = [1, 2]
            colors = ['lightblue', 'lightcoral']

            for j, group_name in enumerate(group_names):
                group_data = result[group_name]['bootstrap_values']
                parts = ax.violinplot([group_data], positions=[positions[j]],
                                    widths=0.4, showmeans=True,
                                    showmedians=True)
                parts['bodies'][0].set_facecolor(colors[j])
                parts['bodies'][0].set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels([name.split(' (')[0] for name in group_names],
                            rotation=45)
            ax.set_ylabel('Bootstrap R² Distribution')
            ax.set_title(comp_title, fontsize=8)

            # Add significance annotation
            if 'difference' in result:
                sig_text = f"p = {result['difference']['p_value']:.3f}"
                if result['difference']['significant']:
                    sig_text += "*"
                ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                        ha='center', va='top', fontsize=8)

    # Plot difference distributions in the fourth subplot
    ax = axes[3]
    x_pos = 0
    colors = ['blue', 'red', 'green']

    for i, (comp_name, comp_title) in enumerate(comparison_titles.items()):
        if 'difference' in bootstrap_results[comp_name]:
            diff_data = (bootstrap_results[comp_name]['difference']
                        ['bootstrap_values'])
            parts = ax.violinplot([diff_data], positions=[x_pos], widths=0.8,
                                showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(colors[i])
            parts['bodies'][0].set_alpha(0.7)
            x_pos += 1

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(comparison_titles)))
    ax.set_xticklabels(['Low vs Middle', 'High vs Middle', 'Low vs High'],
                    rotation=45)
    ax.set_ylabel('Bootstrap R² Difference')
    ax.set_title('Group Differences in Prediction Performance')

    plt.tight_layout()
    plt.savefig(project_folder /
                'figures/pls_bootstrap_group_comparisons_100nodes.png', dpi=300)

    print(f"\nBootstrap analysis complete! Main findings (100 nodes):")
    print(f"Group sizes: Low PGS = {main_results['group_sizes']['low']}, "
        f"Middle PGS = {main_results['group_sizes']['middle']}, "
        f"High PGS = {main_results['group_sizes']['high']}")

    for comp_name, result in bootstrap_results.items():
        if 'difference' in result:
            print(f"{comparison_titles[comp_name]}:")
            print(f"  Difference: {result['difference']['mean_diff']:.3f} "
                f"[{result['difference']['ci_lower']:.3f}, "
                f"{result['difference']['ci_upper']:.3f}], "
                f"p = {result['difference']['p_value']:.3f}")

    # Additional summary across all node configurations
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL NODE CONFIGURATIONS")
    print(f"{'='*80}")

    for n_nodes in node_configurations:
        results = all_results[n_nodes]
        print(f"\n{n_nodes}-node parcellation:")
        print(f"  Sample size: {len(results['prs_scaled'])} subjects")
        print(f"  Group sizes: Low = {results['group_sizes']['low']}, "
            f"Middle = {results['group_sizes']['middle']}, "
            f"High = {results['group_sizes']['high']}")

        # Model performance
        connectivity_model = results['models']['connectivity']
        interaction_model = results['models']['interaction']

        print(f"  Connectivity model: R² = {connectivity_model['r2_cv']:.3f}, "
            f"p = {connectivity_model['p_val_cv']:.3e}")
        print(f"  Interaction model: R² = {interaction_model['r2_cv']:.3f}, "
            f"p = {interaction_model['p_val_cv']:.3e}")

        # Bootstrap comparison results
        print(f"  Bootstrap group comparisons (p-values):")
        for comp_name, comp_result in results['bootstrap_results'].items():
            if 'difference' in comp_result:
                significant = ('significant' if comp_result['difference']['significant']
                            else 'n.s.')
                print(f"    {comp_name}: p = "
                    f"{comp_result['difference']['p_value']:.3f} "
                    f"({significant})")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("CREATE VISUALIZATIONS")
    print(f"{'='*80}")

    # Create all visualizations automatically
    create_all_pls_visualizations(all_results, project_folder)

    # Create scatter plots showing PLS component associations
    create_all_pls_scatter_plots(all_results, project_folder)

if __name__ == "__main__":
    main()