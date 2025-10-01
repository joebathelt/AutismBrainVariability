# %%
# ==============================================================================
# Title: C6_visualize_networks.py
# ==============================================================================
# Description: This script visualizes brain connectivity networks and their
# associations with social difficulty scores and polygenic risk. It includes
# visual representations of network metrics and group comparisons based on
# polygenic score thresholds: Low PGS: <-1 SD, Middle PGS: >-0.5 SD & <0.5 SD,
# High PGS: >1 SD
# ==============================================================================

import bct
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import networkx as nx
from nilearn.connectome import vec_to_sym_matrix
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
import seaborn as sns


# Set up plotting parameters
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'CMU Serif',
    'mathtext.it': 'CMU Serif:italic',
    'mathtext.bf': 'CMU Serif:bold',
})

mm2inches = 0.0393701

n_nodes = 100

# Define paths
project_folder = Path(__file__).resolve().parents[1]
pgs_file = project_folder / "data/PGS_Groups_Data.csv"
social_file = project_folder / 'data/cfa_factor_scores_full_sample.csv'

# Load the PGS data
pgs_df = pd.read_csv(pgs_file)[['Subject', 'pgs_group']]

def get_significance_symbol(p_value):
    """Convert p-value to significance symbol"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def create_bootstrap_samples(df, n_bs_samples=1000, n_samples=90):
    """Create bootstrap samples for each PGS group."""
    bootstrap_samples = {}
    for group in df['pgs_group'].unique():
        group_data = df[df['pgs_group'] == group]['Subject'].values
        samples = []
        for _ in range(n_bs_samples):
            sample = np.random.choice(group_data, size=n_samples, replace=True)
            samples.append(sample)
        bootstrap_samples[group] = np.array(samples)

    return bootstrap_samples

# %%
bootstrap_samples = create_bootstrap_samples(pgs_df)

partition_df = pd.read_csv(project_folder / f'data/final_partition_{n_nodes}Nodes.csv')
partition_df['community_id'] = partition_df['community_id'].astype(int)

# Load connectivity matrices
matrix_file = project_folder / f'data/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
id_file = project_folder / 'data/subjectIDs_anonymised.txt'
ids = pd.read_csv(id_file, header=None).values.flatten()

# Load the full connectivity matrix data once
mats_df_full = pd.read_csv(matrix_file, header=None, sep='\s+')

# Remove lower triangle for non-redundancy
lower_indices = np.tril_indices(n_nodes, k=-1)
linear_indices = lower_indices[0] * n_nodes + lower_indices[1]
mats_df_full = mats_df_full.iloc[:, linear_indices]

for group in ['low', 'middle', 'high']:
    print(f'Processing group: {group}')

    # Get bootstrap samples for this group
    group_bootstrap_samples = bootstrap_samples[group]
    n_bootstrap_samples = len(group_bootstrap_samples)

    # Initialize array to store all bootstrap connectivity matrices
    bootstrap_matrices = []

    print(f'Processing {n_bootstrap_samples} bootstrap samples for group {group}')

    for i, bootstrap_sample in enumerate(group_bootstrap_samples):
        if i % 100 == 0:  # Progress indicator
            print(f'  Processing bootstrap sample {i+1}/{n_bootstrap_samples}')

        # Find indices for subjects in this bootstrap sample
        sample_indices = []
        for subject in bootstrap_sample:
            subject_idx = np.where(ids == subject)[0]
            if len(subject_idx) > 0:
                sample_indices.append(subject_idx[0])

        if len(sample_indices) == 0:
            print(f'  No valid subjects found in bootstrap sample {i+1}. Skipping...')
            continue

        # Extract connectivity matrices for this bootstrap sample
        sample_mats = mats_df_full.iloc[sample_indices, :]

        # Calculate average connectivity matrix for this bootstrap sample
        avg_mat = sample_mats.mean(axis=0).values
        avg_mat = avg_mat / 100  # Normalize to [-1, 1] range
        correlation_matrix = vec_to_sym_matrix(avg_mat, diagonal=np.zeros(n_nodes))

        bootstrap_matrices.append(correlation_matrix)

    if len(bootstrap_matrices) == 0:
        print(f'No valid bootstrap samples found for group {group}. Skipping...')
        continue

    # Calculate the average across all bootstrap samples
    bootstrap_matrices = np.array(bootstrap_matrices)
    final_avg_matrix = np.mean(bootstrap_matrices, axis=0)

    print(f'Computed average from {len(bootstrap_matrices)} valid bootstrap samples')

    # Save the bootstrap-averaged matrix for each PGS group
    np.save(project_folder / f'data/avg_connectivity_{group}_pgs_bootstrap.npy', final_avg_matrix)

    # Reorder matrix by community
    community_order = np.argsort(partition_df['community_id'].values)
    reordered_matrix = final_avg_matrix[community_order][:, community_order]

    # Plot the reordered connectivity matrix
    plt.figure(figsize=(55*mm2inches, 55*mm2inches))
    plt.imshow(
        reordered_matrix,
        cmap='RdBu_r',
        vmin=-0.3,
        vmax=0.3,
        )

    # Add community boundaries
    community_sizes = partition_df['community_id'].value_counts().sort_index()
    boundaries = np.cumsum([0] + list(community_sizes))[:-1]
    for boundary in boundaries[1:]:  # Skip first boundary at 0
        plt.axhline(boundary - 0.5, color='black', linewidth=0.5)
        plt.axvline(boundary - 0.5, color='black', linewidth=0.5)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(project_folder / f'figures/Connectivity_Matrix_{n_nodes}Nodes_{group}PGS_bootstrap.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Colour the nodes based on the partition
    node_colours = partition_df['community_id'].values
    node_colour_dict = {
        0:           '#7B2D8E',  # Visual Network (deeper purple)
        1:            '#C85450',  # DMN (warmer red)
        2:            '#A8B8C8',  # Other (softer blue-gray)
        3:            '#D17A47',  # FPN (warmer orange)
        4:            '#2DB574',  # VAN (balanced green)
        5:            '#4FB3D9'   # Other (warmer cyan)
    }
    node_colours = np.array([node_colour_dict[comm] for comm in node_colours])

    final_avg_matrix_thresholded = bct.threshold_proportional(final_avg_matrix.copy(), 0.05, copy=False)
    G = nx.from_numpy_array(final_avg_matrix_thresholded)
    pos = nx.spring_layout(G, iterations=100, seed=42)

    # Remove disconnected nodes
    largest_cc = max(nx.connected_components(G), key=len)
    G_filtered = G.subgraph(largest_cc)
    node_colours_filtered = node_colours[list(largest_cc)]

    # Make the node size proportional to the degree
    node_sizes = [2 * G_filtered.degree(n) for n in G_filtered.nodes()]

    plt.figure(figsize=(50*mm2inches, 50*mm2inches))
    nx.draw(
        G_filtered, pos,
        node_size=node_sizes,
        node_color=node_colours_filtered,
        with_labels=False,
        edge_color='gray',
        edgecolors='black'
        )
    plt.savefig(project_folder / f'figures/Network_Visualisation_{n_nodes}Nodes_{group}PGS_bootstrap.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# %%
# Bootstrap density and boxplot visualization with matched sample sizes
graph_metrics_df = pd.read_csv(project_folder / 'results/network_metrics_100nodes_0.20thresh.csv')

def bootstrap_density_boxplot(data, ax, position=0, width=0.4, color='lightblue', label=None, n_bootstrap=1000, sample_size=90):
    """
    Create bootstrap density plot with boxplot overlay
    
    Parameters:
    -----------
    data : array-like
        Original data to bootstrap from
    ax : matplotlib axis
        Axis to plot on
    position : float
        X position for the plot
    width : float
        Width of the density curve
    color : str
        Color for the plot
    label : str
        Label for legend
    n_bootstrap : int
        Number of bootstrap samples to generate
    sample_size : int
        Size of each bootstrap sample
    """
    
    # Generate bootstrap samples
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_samples.append(np.mean(bootstrap_sample))  # or use the raw samples if you want distribution of all points
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Kernel density estimation for the bootstrap distribution
    density = stats.gaussian_kde(bootstrap_samples)
    xs = np.linspace(bootstrap_samples.min(), bootstrap_samples.max(), 200)
    density_curve = density(xs)
    
    # Scale and position the density curve
    density_curve = density_curve / density_curve.max() * width
    
    # Plot the density curve
    ax.fill_betweenx(xs, position - density_curve, position,
                     alpha=0.6, color=color, label=label)
    
    # Add box plot for the bootstrap distribution
    box_data = ax.boxplot([bootstrap_samples], positions=[position + width/2],
                         widths=width/4, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         showfliers=False)

# Set parameters for bootstrap
n_bootstrap = 1000
matched_sample_size = 90  # Size to match across all groups

for measure, measure_label in zip(['modularity', 'global_efficiency'], ['Modularity', 'Global Efficiency']):
    
    fig, ax = plt.subplots(figsize=(60*mm2inches, 45*mm2inches))
    
    # Extract your data
    data_low = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'low', measure].values
    data_medium = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'middle', measure].values
    data_high = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'high', measure].values
    
    # Z-score all values across groups for more intuitive scale
    all_data = np.concatenate([data_low, data_medium, data_high])
    mean_all = np.mean(all_data)
    std_all = np.std(all_data, ddof=1)  # Using sample standard deviation
    
    # Apply z-scoring to each group
    data_low = (data_low - mean_all) / std_all
    data_medium = (data_medium - mean_all) / std_all
    data_high = (data_high - mean_all) / std_all
    
    # Define colors for each group
    colors = ['#8491B499', '#91D1C299', '#4DBBD599']
    labels = ['low', 'middle', 'high']
    
    # Create bootstrap density plots with boxplots
    positions = [0, 1.2, 2.4]  # Tighter spacing for small figure
    for i, (data, pos, color, label) in enumerate(zip([data_low, data_medium, data_high],
                                                    positions, colors, labels)):
        bootstrap_density_boxplot(data, ax, position=pos, width=0.4, color=color, label=label,
                                n_bootstrap=n_bootstrap, sample_size=matched_sample_size)
    
    # Customize the plot for small size
    ax.set_xlim(-0.5, 3.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Low', 'Middle', 'High'], fontsize=8)
    ax.set_ylabel(f'{measure_label} [$z$]', fontsize=8)
    ax.set_xlabel('PGS Group', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Set intuitive y-ticks for z-scored data
    ax.set_yticks([-0.5, 0, 0.5])
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    sns.despine(offset=8, trim=True)
    
    plt.tight_layout()
    plt.savefig(project_folder / f'figures/Bootstrap_Density_Plot_{measure}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# %%
def bootstrap_ellipse_parameters(data_mod, data_eff, n_bootstrap_iterations=1000, sample_size=90):
    """
    Bootstrap the ellipse parameters (size, shape, orientation) across many samples
    
    Returns:
    --------
    ellipse_params : list of dicts
        Each dict contains: {'center': (x, y), 'width': float, 'height': float, 'angle': float}
    """
    ellipse_params = []
    
    for iteration in range(n_bootstrap_iterations):
        # Create bootstrap sample
        bootstrap_indices = np.random.choice(len(data_mod), size=min(sample_size, len(data_mod)), replace=True)
        bootstrap_mod = data_mod[bootstrap_indices]
        bootstrap_eff = data_eff[bootstrap_indices]
        
        if len(bootstrap_mod) < 4:  # Need at least 4 points for meaningful ellipse
            continue
            
        # Calculate covariance matrix for this bootstrap sample
        data_2d = np.column_stack([bootstrap_mod, bootstrap_eff])
        cov = np.cov(data_2d.T)
        
        # Calculate ellipse parameters
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues (largest first)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        
        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * 1.96 * np.sqrt(eigenvals[0])   # 95% confidence
        height = 2 * 1.96 * np.sqrt(eigenvals[1])
        center = (np.mean(bootstrap_mod), np.mean(bootstrap_eff))
        
        ellipse_params.append({
            'center': center,
            'width': width,
            'height': height, 
            'angle': angle,
            'area': np.pi * width * height / 4,  # Area of ellipse
            'eccentricity': np.sqrt(1 - (min(eigenvals) / max(eigenvals)))  # Shape measure
        })
    
    return ellipse_params


def create_bootstrap_ellipse_extent_plot(graph_df, n_bootstrap_iterations=1000, sample_size=90,
                                       group_colors=['#9BB3C7', '#7FB069', '#4FB3D9'], 
                                        alpha_individual=0.05):
    """
    Create plot showing bootstrapped ellipse extents (no scatter points)

    Parameters:
    -----------
    alpha_individual : float
        Alpha for individual ellipses if show_all_ellipses=True
    """
    fig, ax = plt.subplots(1, 1, figsize=(60*mm2inches, 45*mm2inches))

    groups = ['low', 'middle', 'high']
    group_labels = ['Low PGS', 'Middle PGS', 'High PGS']

    bootstrap_results = {}

    for i, group in enumerate(groups):
        print(f"\nBootstrapping ellipse extent for {group} group...")

        # Get group data
        group_data = graph_df[graph_df['pgs_group'] == group]
        available_mod = group_data['modularity'].values
        available_eff = group_data['global_efficiency'].values

        if len(available_mod) == 0:
            print(f"No data found for {group} group. Skipping...")
            continue

        # Bootstrap ellipse parameters
        ellipse_params = bootstrap_ellipse_parameters(
            available_mod, available_eff, n_bootstrap_iterations, sample_size
        )

        if len(ellipse_params) == 0:
            print(f"No valid ellipses generated for {group} group. Skipping...")
            continue

        bootstrap_results[group] = ellipse_params

        # Calculate confidence bounds for ellipse parameters
        widths = [p['width'] for p in ellipse_params]
        heights = [p['height'] for p in ellipse_params]
        centers_x = [p['center'][0] for p in ellipse_params]
        centers_y = [p['center'][1] for p in ellipse_params]
        angles = [p['angle'] for p in ellipse_params]

        # Use percentiles to show confidence envelope
        width_low, width_med, width_high = np.percentile(widths, [2.5, 50, 97.5])
        height_low, height_med, height_high = np.percentile(heights, [2.5, 50, 97.5])
        center_x_med = np.median(centers_x)
        center_y_med = np.median(centers_y)
        angle_med = np.median(angles)

        # Draw confidence envelope ellipses
        # Outer boundary (97.5th percentile)
        ellipse_outer = Ellipse(
            xy=(center_x_med, center_y_med),
            width=width_high,
            height=height_high,
            angle=angle_med,
            facecolor='none',
            edgecolor=group_colors[i],
            linewidth=2,
            linestyle='-',
            alpha=0.8
        )
        ax.add_patch(ellipse_outer)

        # Add group centroid
        mean_center_x = np.mean(centers_x)
        mean_center_y = np.mean(centers_y)
        ax.scatter(mean_center_x, mean_center_y, marker='o', s=100, alpha=0.5,
                  color=group_colors[i], edgecolors='black', linewidth=0.5,
                  zorder=10, label=group_labels[i])

    ax.set_xlabel('Modularity')
    ax.set_ylabel('Global Efficiency')
    # Remove the legend
    ax.legend([], frameon=False)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='datalim')
    sns.despine(offset=6, trim=True)
    plt.tight_layout(pad=5)

    return fig, ax, bootstrap_results

# %%
fig, ax, results = create_bootstrap_ellipse_extent_plot(
    graph_metrics_df, n_bootstrap_iterations=1000, sample_size=90, alpha_individual=0.05
)
ax.set_yticks([0.45, 0.50, 0.55])
fig.savefig(project_folder / 'figures/Bootstrap_Ellipse_Extent_Plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# %%
