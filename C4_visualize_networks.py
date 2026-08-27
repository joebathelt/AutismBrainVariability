#!/usr/bin/env python3
"""
C4: Network Visualization for Brain Variability Analysis
==========================================================

This script creates network visualizations for the brain compensation analysis:

1. Bootstrap-averaged connectivity matrices by PGS group
2. Network graph visualizations by PGS group
3. Bootstrap density plots for modularity and global efficiency
4. Bootstrap ellipse extent plots showing network organization space

Parcellation resolution is fixed upstream by C2b_evaluate_communities.py;
n_nodes is derived from the number of rows in the partition file.

Usage:
    python C4_visualize_networks.py \
        --project /path/to/project \
        --pgs data/pgs_residuals.csv \
        --social data/cfa_factor_scores_full_sample.csv \
        --graph-metrics results/C4_main_network_metrics.csv \
        --partition results/C2b_selected_partition.csv \
        --ids data/subjectIDs_anonymised.txt \
        --matrices-dir data/HCP_PTN1200/netmats
"""

import argparse
import bct
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import networkx as nx
from nilearn.connectome import vec_to_sym_matrix
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
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
FIGURE_DPI = 300

# Node colors based on community
NODE_COLOUR_DICT = {
    0: '#7B2D8E',  # Visual Network (deeper purple)
    1: '#C85450',  # DMN (warmer red)
    2: '#A8B8C8',  # Other (softer blue-gray)
    3: '#D17A47',  # FPN (warmer orange)
    4: '#2DB574',  # VAN (balanced green)
    5: '#4FB3D9'   # Other (warmer cyan)
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="C6: Network visualization for brain compensation analysis"
    )
    parser.add_argument("--project", required=True, help="Project directory path")
    parser.add_argument("--pgs", required=True, help="PGS/PGS residuals file (from B4)")
    parser.add_argument("--social", required=True, help="Social scores file")
    parser.add_argument("--graph-metrics", required=True,
                        help="Main-threshold graph metrics CSV from C3 "
                             "(default name: C4_main_network_metrics.csv)")
    parser.add_argument("--partition", required=True,
                        help="Community partition from C2b "
                             "(default: C2b_selected_partition.csv). "
                             "n_nodes is derived from the number of rows.")
    parser.add_argument("--ids", required=True, help="Subject IDs file")
    parser.add_argument("--matrices-dir", required=True,
                        help="Directory containing connectivity matrices")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap samples (default: 1000)")
    parser.add_argument("--sample-size", type=int, default=90,
                        help="Bootstrap sample size (default: 90)")
    return parser.parse_args()


def get_significance_symbol(p_value):
    """Convert p-value to significance symbol."""
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
            sample = np.random.choice(group_data, size=min(n_samples, len(group_data)), replace=True)
            samples.append(sample)
        bootstrap_samples[group] = np.array(samples)

    return bootstrap_samples


def create_connectivity_matrices(pgs_df, mats_df_full, ids, partition_df, n_nodes,
                                  project_dir, n_bootstrap, sample_size, report):
    """Create bootstrap-averaged connectivity matrices for each PGS group."""
    report.append("\n" + "=" * 80)
    report.append("CONNECTIVITY MATRIX VISUALIZATION")
    report.append("=" * 80)

    bootstrap_samples = create_bootstrap_samples(pgs_df, n_bootstrap, sample_size)

    figures_dir = project_dir / 'figures'
    data_dir = project_dir / 'data'

    for group in ['low', 'middle', 'high']:
        report.append(f"\nProcessing group: {group}")
        print(f'Processing group: {group}')

        # Get bootstrap samples for this group
        if group not in bootstrap_samples:
            report.append(f"  No data for group {group}, skipping")
            continue

        group_bootstrap_samples = bootstrap_samples[group]
        n_bootstrap_samples = len(group_bootstrap_samples)

        # Initialize array to store all bootstrap connectivity matrices
        bootstrap_matrices = []

        print(f'Processing {n_bootstrap_samples} bootstrap samples for group {group}')

        for i, bootstrap_sample in enumerate(group_bootstrap_samples):
            if i % 200 == 0:
                print(f'  Processing bootstrap sample {i+1}/{n_bootstrap_samples}')

            # Find indices for subjects in this bootstrap sample
            sample_indices = []
            for subject in bootstrap_sample:
                subject_idx = np.where(ids == subject)[0]
                if len(subject_idx) > 0:
                    sample_indices.append(subject_idx[0])

            if len(sample_indices) == 0:
                continue

            # Extract connectivity matrices for this bootstrap sample
            sample_mats = mats_df_full.iloc[sample_indices, :]

            # Calculate average connectivity matrix for this bootstrap sample
            avg_mat = sample_mats.mean(axis=0).values
            avg_mat = avg_mat / 100  # Normalize to [-1, 1] range
            correlation_matrix = vec_to_sym_matrix(avg_mat, diagonal=np.zeros(n_nodes))

            bootstrap_matrices.append(correlation_matrix)

        if len(bootstrap_matrices) == 0:
            report.append(f'  No valid bootstrap samples found for group {group}. Skipping...')
            print(f'No valid bootstrap samples found for group {group}. Skipping...')
            continue

        # Calculate the average across all bootstrap samples
        bootstrap_matrices = np.array(bootstrap_matrices)
        final_avg_matrix = np.mean(bootstrap_matrices, axis=0)

        report.append(f'  Computed average from {len(bootstrap_matrices)} valid bootstrap samples')
        print(f'Computed average from {len(bootstrap_matrices)} valid bootstrap samples')

        # Save the bootstrap-averaged matrix for each PGS group
        matrix_path = data_dir / f'C5_avg_connectivity_{group}_pgs_bootstrap.npy'
        np.save(matrix_path, final_avg_matrix)
        report.append(f'  Saved matrix to: {matrix_path.name}')

        # Reorder matrix by community
        community_order = np.argsort(partition_df['community_id'].values)
        reordered_matrix = final_avg_matrix[community_order][:, community_order]

        # Plot the reordered connectivity matrix
        fig = plt.figure(figsize=(55*mm2inches, 55*mm2inches))
        plt.imshow(
            reordered_matrix,
            cmap='RdBu_r',
            vmin=-0.3,
            vmax=0.3,
        )

        # Add community boundaries
        community_sizes = partition_df['community_id'].value_counts().sort_index()
        boundaries = np.cumsum([0] + list(community_sizes))[:-1]
        for boundary in boundaries[1:]:
            plt.axhline(boundary - 0.5, color='black', linewidth=0.5)
            plt.axvline(boundary - 0.5, color='black', linewidth=0.5)

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        matrix_fig_path = figures_dir / f'C5_Connectivity_Matrix_{n_nodes}Nodes_{group}PGS_bootstrap.png'
        plt.savefig(matrix_fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f'  Saved matrix figure: {matrix_fig_path.name}')

        # Create network visualization
        node_colours = partition_df['community_id'].values
        node_colours = np.array([NODE_COLOUR_DICT.get(comm, '#888888') for comm in node_colours])

        final_avg_matrix_thresholded = bct.threshold_proportional(final_avg_matrix.copy(), 0.05, copy=False)
        G = nx.from_numpy_array(final_avg_matrix_thresholded)
        pos = nx.spring_layout(G, iterations=100, seed=42)

        # Remove disconnected nodes
        largest_cc = max(nx.connected_components(G), key=len)
        G_filtered = G.subgraph(largest_cc)
        node_colours_filtered = node_colours[list(largest_cc)]

        # Make the node size proportional to the degree
        node_sizes = [2 * G_filtered.degree(n) for n in G_filtered.nodes()]

        fig = plt.figure(figsize=(50*mm2inches, 50*mm2inches))
        nx.draw(
            G_filtered, pos,
            node_size=node_sizes,
            node_color=node_colours_filtered,
            with_labels=False,
            edge_color='gray',
            edgecolors='black'
        )

        network_fig_path = figures_dir / f'C5_Network_Visualisation_{n_nodes}Nodes_{group}PGS_bootstrap.png'
        plt.savefig(network_fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f'  Saved network figure: {network_fig_path.name}')


def create_exemplar_networks(pgs_df, mats_df_full, ids, partition_df, n_nodes,
                              project_dir, report,
                              n_exemplars=4, n_quartiles=4,
                              modularity_threshold=0.2, viz_threshold=0.05,
                              seed=42):
    """Per PGS group, sample n_exemplars subjects from each within-group modularity
    quartile and draw an individual spring-layout network for each.

    Modularity is computed per subject from the raw correlation matrix at
    `modularity_threshold` (matches C3's main-threshold pipeline). The spring-layout
    visualisation uses `viz_threshold` (matches the bootstrap network panels).
    Saves one PNG per exemplar plus an index CSV.
    """
    report.append("\n" + "=" * 80)
    report.append("EXEMPLAR NETWORK VISUALIZATIONS (per PGS group x modularity quartile)")
    report.append("=" * 80)

    figures_dir = project_dir / 'figures'
    results_dir = project_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    node_colours = partition_df['community_id'].values
    node_colours = np.array([NODE_COLOUR_DICT.get(comm, '#888888') for comm in node_colours])
    community_ids = partition_df['community_id'].values

    # Compute per-subject modularity at the analysis threshold so quartile
    # assignment matches the rest of the pipeline.
    mod_records = []
    for subject in pgs_df['Subject'].values:
        subj_idx = np.where(ids == subject)[0]
        if len(subj_idx) == 0:
            continue
        subj_idx = subj_idx[0]
        row = mats_df_full.iloc[subj_idx, :].values / 100
        mat = vec_to_sym_matrix(row, diagonal=np.zeros(n_nodes))
        mat_t = bct.threshold_proportional(mat, modularity_threshold)
        mat_t = np.nan_to_num(mat_t, nan=0.0)
        try:
            _, modularity = bct.modularity_und_sign(mat_t, community_ids)
        except Exception:
            modularity = np.nan
        if np.isnan(modularity):
            continue
        mod_records.append({'Subject': subject, 'modularity': modularity, 'subj_idx': subj_idx})

    mod_df = pd.DataFrame(mod_records).merge(pgs_df, on='Subject')
    report.append(f"\n  Modularity computed for {len(mod_df)} subjects at threshold {modularity_threshold}")

    index_rows = []

    group_order = ['low', 'middle', 'high']
    for g_idx, group in enumerate(group_order):
        group_df = mod_df[mod_df['pgs_group'] == group].copy()
        if len(group_df) < n_quartiles:
            report.append(f"\n  {group}: only {len(group_df)} subjects, skipping")
            continue

        group_df['mod_quartile'] = pd.qcut(
            group_df['modularity'], q=n_quartiles, labels=False, duplicates='drop'
        )
        # Deterministic order before sampling so rng state is reproducible
        group_df = group_df.sort_values('Subject').reset_index(drop=True)

        report.append(f"\n  Group: {group} (n={len(group_df)})")
        print(f'Creating exemplars for group: {group}')

        for q in range(n_quartiles):
            pool = group_df[group_df['mod_quartile'] == q]
            if len(pool) == 0:
                report.append(f"    Q{q+1}: empty quartile, skipping")
                continue

            rng = np.random.default_rng(seed + g_idx * 100 + q)
            n_pick = min(n_exemplars, len(pool))
            chosen = rng.choice(pool['Subject'].values, size=n_pick, replace=False)

            report.append(f"    Q{q+1}: picked {n_pick} of {len(pool)} subjects")

            for i, subject in enumerate(chosen):
                subj_row = pool[pool['Subject'] == subject].iloc[0]
                subj_idx = int(subj_row['subj_idx'])

                row = mats_df_full.iloc[subj_idx, :].values / 100
                subj_mat = vec_to_sym_matrix(row, diagonal=np.zeros(n_nodes))

                thresholded = bct.threshold_proportional(subj_mat.copy(), viz_threshold, copy=False)
                G = nx.from_numpy_array(thresholded)
                pos = nx.spring_layout(G, iterations=100, seed=42)

                largest_cc = max(nx.connected_components(G), key=len)
                G_filtered = G.subgraph(largest_cc)
                node_colours_filtered = node_colours[list(largest_cc)]
                node_sizes = [2 * G_filtered.degree(n) for n in G_filtered.nodes()]

                fig = plt.figure(figsize=(50*mm2inches, 50*mm2inches))
                nx.draw(
                    G_filtered, pos,
                    node_size=node_sizes,
                    node_color=node_colours_filtered,
                    with_labels=False,
                    edge_color='gray',
                    edgecolors='black'
                )

                fname = (f'C5_Exemplar_Network_{n_nodes}Nodes_{group}PGS_'
                         f'modQ{q+1}_e{i+1}_subj{subject}.png')
                fig_path = figures_dir / fname
                plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                index_rows.append({
                    'pgs_group': group,
                    'mod_quartile': q + 1,
                    'exemplar_idx': i + 1,
                    'Subject': subject,
                    'modularity': subj_row['modularity'],
                    'filename': fname,
                })

    index_df = pd.DataFrame(index_rows)
    index_path = results_dir / 'C5_exemplar_subjects.csv'
    index_df.to_csv(index_path, index=False)
    report.append(f"\n  Saved exemplar index: {index_path.name} ({len(index_df)} rows)")


def bootstrap_density_boxplot(data, ax, position=0, width=0.4, color='lightblue',
                               label=None, n_bootstrap=1000, sample_size=90):
    """
    Create bootstrap density plot with boxplot overlay.

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
        bootstrap_sample = np.random.choice(data, size=min(sample_size, len(data)), replace=True)
        bootstrap_samples.append(np.mean(bootstrap_sample))

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
    ax.boxplot([bootstrap_samples], positions=[position + width/2],
               widths=width/4, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color='black', linewidth=2),
               showfliers=False)


def create_density_plots(graph_metrics_df, project_dir, n_bootstrap, sample_size, report):
    """Create bootstrap density plots for modularity and global efficiency."""
    report.append("\n" + "=" * 80)
    report.append("BOOTSTRAP DENSITY PLOTS")
    report.append("=" * 80)

    figures_dir = project_dir / 'figures'

    for measure, measure_label in zip(['modularity', 'global_efficiency'],
                                       ['Modularity (residualised)',
                                        'Global Efficiency (residualised)']):
        report.append(f"\nCreating density plot for {measure_label}...")

        fig, ax = plt.subplots(figsize=(60*mm2inches, 45*mm2inches))

        # Extract data for each group
        data_low = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'low', measure].values
        data_medium = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'middle', measure].values
        data_high = graph_metrics_df.loc[graph_metrics_df['pgs_group'] == 'high', measure].values

        report.append(f"  Group sizes: Low={len(data_low)}, Middle={len(data_medium)}, High={len(data_high)}")

        # Z-score all values across groups for more intuitive scale
        all_data = np.concatenate([data_low, data_medium, data_high])
        mean_all = np.mean(all_data)
        std_all = np.std(all_data, ddof=1)

        # Apply z-scoring to each group
        data_low = (data_low - mean_all) / std_all
        data_medium = (data_medium - mean_all) / std_all
        data_high = (data_high - mean_all) / std_all

        # Define colors for each group
        colors = ['#8491B499', '#91D1C299', '#4DBBD599']
        labels = ['low', 'middle', 'high']

        # Create bootstrap density plots with boxplots
        positions = [0, 1.2, 2.4]
        for i, (data, pos, color, label) in enumerate(zip([data_low, data_medium, data_high],
                                                          positions, colors, labels)):
            bootstrap_density_boxplot(data, ax, position=pos, width=0.4, color=color,
                                      label=label, n_bootstrap=n_bootstrap,
                                      sample_size=sample_size)

        # Customize the plot
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

        fig_path = figures_dir / f'C5_Bootstrap_Density_Plot_{measure}.png'
        plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f"  Saved: {fig_path.name}")


def bootstrap_ellipse_parameters(data_mod, data_eff, n_bootstrap_iterations=1000, sample_size=90):
    """
    Bootstrap the ellipse parameters (size, shape, orientation) across many samples.

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

        if len(bootstrap_mod) < 4:
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
        width = 2 * 1.96 * np.sqrt(eigenvals[0])
        height = 2 * 1.96 * np.sqrt(eigenvals[1])
        center = (np.mean(bootstrap_mod), np.mean(bootstrap_eff))

        ellipse_params.append({
            'center': center,
            'width': width,
            'height': height,
            'angle': angle,
            'area': np.pi * width * height / 4,
            'eccentricity': np.sqrt(1 - (min(eigenvals) / max(eigenvals)))
        })

    return ellipse_params


def create_ellipse_extent_plot(graph_metrics_df, project_dir, n_bootstrap, sample_size, report):
    """Create bootstrap ellipse extent plot showing network organization space."""
    report.append("\n" + "=" * 80)
    report.append("BOOTSTRAP ELLIPSE EXTENT PLOT")
    report.append("=" * 80)

    figures_dir = project_dir / 'figures'

    fig, ax = plt.subplots(1, 1, figsize=(60*mm2inches, 45*mm2inches))

    groups = ['low', 'middle', 'high']
    group_labels = ['Low PGS', 'Middle PGS', 'High PGS']
    group_colors = ['#9BB3C7', '#7FB069', '#4FB3D9']

    bootstrap_results = {}

    for i, group in enumerate(groups):
        report.append(f"\nBootstrapping ellipse extent for {group} group...")
        print(f"Bootstrapping ellipse extent for {group} group...")

        # Get group data
        group_data = graph_metrics_df[graph_metrics_df['pgs_group'] == group]
        available_mod = group_data['modularity'].values
        available_eff = group_data['global_efficiency'].values

        if len(available_mod) == 0:
            report.append(f"  No data found for {group} group. Skipping...")
            continue

        report.append(f"  N subjects: {len(available_mod)}")

        # Bootstrap ellipse parameters
        ellipse_params = bootstrap_ellipse_parameters(
            available_mod, available_eff, n_bootstrap, sample_size
        )

        if len(ellipse_params) == 0:
            report.append(f"  No valid ellipses generated for {group} group. Skipping...")
            continue

        bootstrap_results[group] = ellipse_params

        # Calculate confidence bounds for ellipse parameters
        widths = [p['width'] for p in ellipse_params]
        heights = [p['height'] for p in ellipse_params]
        centers_x = [p['center'][0] for p in ellipse_params]
        centers_y = [p['center'][1] for p in ellipse_params]
        angles = [p['angle'] for p in ellipse_params]
        areas = [p['area'] for p in ellipse_params]

        # Use percentiles to show confidence envelope
        width_low, width_med, width_high = np.percentile(widths, [2.5, 50, 97.5])
        height_low, height_med, height_high = np.percentile(heights, [2.5, 50, 97.5])
        center_x_med = np.median(centers_x)
        center_y_med = np.median(centers_y)
        angle_med = np.median(angles)
        area_med = np.median(areas)

        report.append(f"  Median ellipse area: {area_med:.4f}")
        report.append(f"  Width (95% CI): {width_med:.4f} [{width_low:.4f}, {width_high:.4f}]")
        report.append(f"  Height (95% CI): {height_med:.4f} [{height_low:.4f}, {height_high:.4f}]")

        # Draw confidence envelope ellipse (97.5th percentile)
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

    ax.set_xlabel('Modularity (residualised)')
    ax.set_ylabel('Global Efficiency (residualised)')
    ax.legend([], frameon=False)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_yticks([0.45, 0.50, 0.55])
    sns.despine(offset=6, trim=True)
    plt.tight_layout(pad=5)

    fig_path = figures_dir / 'C5_Bootstrap_Ellipse_Extent_Plot.png'
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    report.append(f"\nSaved ellipse plot: {fig_path.name}")

    return bootstrap_results


def main():
    """Run the complete network visualization pipeline."""
    args = parse_args()
    project_dir = Path(args.project)

    # Parcellation size is inherited from the C2b-selected partition
    n_nodes = len(pd.read_csv(args.partition))

    # Initialize report
    report = []
    report.append("=" * 80)
    report.append("C5: NETWORK VISUALIZATION REPORT")
    report.append("=" * 80)
    report.append(f"\nProject directory: {project_dir}")
    report.append(f"Number of nodes (from C2b partition): {n_nodes}")
    report.append(f"Number of bootstrap samples: {args.n_bootstrap}")
    report.append(f"Bootstrap sample size: {args.sample_size}")

    print("=" * 80)
    print("C5: NETWORK VISUALIZATION")
    print("=" * 80)

    # Create directories
    figures_dir = project_dir / 'figures'
    reports_dir = project_dir / 'reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    report.append("\n" + "=" * 80)
    report.append("LOADING DATA")
    report.append("=" * 80)

    # Load PGS residuals and compute group assignments
    # Groups: low (<-1 SD), middle (-0.5 to 0.5 SD), high (>+1 SD)
    # Subjects between -1 and -0.5, and between 0.5 and 1 are excluded
    pgs_raw = pd.read_csv(args.pgs)
    pgs_raw['pgs_z'] = zscore(pgs_raw['blup_PGS_residuals'])
    pgs_raw['pgs_group'] = pd.cut(
        pgs_raw['pgs_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['low', 'exclude_low', 'middle', 'exclude_high', 'high']
    )
    # Keep main three groups
    pgs_df = pgs_raw[pgs_raw['pgs_group'].isin(['low', 'middle', 'high'])][['Subject', 'pgs_group']].copy()
    report.append(f"PGS data loaded: {len(pgs_raw)} subjects")
    report.append("PGS group sizes:")
    for group in ['low', 'middle', 'high']:
        n = len(pgs_df[pgs_df['pgs_group'] == group])
        report.append(f"  {group.capitalize()}: {n}")

    partition_df = pd.read_csv(args.partition)
    partition_df['community_id'] = partition_df['community_id'].astype(int)
    report.append(f"Partition data: {len(partition_df)} nodes")

    ids = pd.read_csv(args.ids, header=None).values.flatten()
    report.append(f"Subject IDs: {len(ids)}")

    # Load connectivity matrices (full correlation)
    matrices_dir = Path(args.matrices_dir)
    matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
    mats_df_full = pd.read_csv(matrix_file, header=None, sep=r'\s+')

    # Remove lower triangle for non-redundancy
    lower_indices = np.tril_indices(n_nodes, k=-1)
    linear_indices = lower_indices[0] * n_nodes + lower_indices[1]
    mats_df_full = mats_df_full.iloc[:, linear_indices]
    report.append(f"Connectivity matrices loaded: {mats_df_full.shape}")

    # Load graph metrics
    graph_metrics_df = pd.read_csv(args.graph_metrics)
    report.append(f"Graph metrics data: {len(graph_metrics_df)} subjects")

    # C3_heteroscedasticity_results.csv ships pgs_z but no pgs_group; derive the
    # group labels with the same cutoffs used for pgs_df above so the density and
    # ellipse panels are grouped identically to the connectivity panels.
    if 'pgs_group' not in graph_metrics_df.columns:
        if 'pgs_z' not in graph_metrics_df.columns:
            raise ValueError(
                f"{args.graph_metrics} has neither 'pgs_group' nor 'pgs_z'; "
                "cannot assign PGS groups."
            )
        graph_metrics_df['pgs_group'] = pd.cut(
            graph_metrics_df['pgs_z'],
            bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
            labels=['low', 'exclude_low', 'middle', 'exclude_high', 'high']
        )
        report.append("Derived pgs_group from pgs_z (low z<-1, middle -0.5..0.5, high z>1)")
        for group in ['low', 'middle', 'high']:
            n = int((graph_metrics_df['pgs_group'] == group).sum())
            report.append(f"  {group.capitalize()}: {n}")

    # Create connectivity matrix and network visualizations
    create_connectivity_matrices(pgs_df, mats_df_full, ids, partition_df, n_nodes,
                                  project_dir, args.n_bootstrap, args.sample_size, report)

    # Create per-subject exemplar networks by PGS group x modularity quartile
    create_exemplar_networks(pgs_df, mats_df_full, ids, partition_df,
                             n_nodes, project_dir, report)

    # Create density plots
    create_density_plots(graph_metrics_df, project_dir, args.n_bootstrap,
                         args.sample_size, report)

    # Create ellipse extent plot
    bootstrap_results = create_ellipse_extent_plot(graph_metrics_df, project_dir,
                                                    args.n_bootstrap, args.sample_size, report)

    # Final summary
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append("\nGenerated figures:")
    report.append("  - C5_Connectivity_Matrix_*Nodes_*PGS_bootstrap.png (per group)")
    report.append("  - C5_Network_Visualisation_*Nodes_*PGS_bootstrap.png (per group)")
    report.append("  - C5_Exemplar_Network_*Nodes_*PGS_modQ*_e*_subj*.png "
                  "(4 exemplars x 4 quartiles x 3 groups = 48)")
    report.append("  - C5_exemplar_subjects.csv (index of exemplar selections)")
    report.append("  - C5_Bootstrap_Density_Plot_modularity.png")
    report.append("  - C5_Bootstrap_Density_Plot_global_efficiency.png")
    report.append("  - C5_Bootstrap_Ellipse_Extent_Plot.png")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_path = reports_dir / 'C5_visualize_networks_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}/C5_*.png")


if __name__ == "__main__":
    main()
