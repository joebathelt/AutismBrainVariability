# %%
import argparse
import bct
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib.colors import ListedColormap
from nilearn.connectome import vec_to_sym_matrix
from utils import connectome_viz
import nibabel as nib
from neuromaps.datasets import fetch_fslr
from surfplot import Plot

# %%
def get_adaptive_gamma_range(n_nodes, target_communities=(5, 15)):
    """
    Generate adaptive gamma range based on network size and target community count.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    target_communities : tuple
        Target range of communities (min, max)
    
    Returns:
    --------
    numpy.ndarray
        Gamma values to test
    """
    # More aggressive scaling for larger networks
    # Empirical observation: 200-node networks need much higher gamma
    if n_nodes >= 200:
        # For 200+ nodes, use much higher gamma range
        base_gamma = 2.0 + 0.5 * np.log10(n_nodes / 100)
        min_gamma = 1.0
        max_gamma = base_gamma * 2.0
        
        # Denser sampling in the high gamma region
        gamma_range = np.concatenate([
            np.linspace(min_gamma, base_gamma, 10),
            np.linspace(base_gamma, max_gamma, 15)[1:],
            #np.linspace(max_gamma, max_gamma * 1.5, 8)[1:]  # Even higher values
        ])
        
    elif n_nodes >= 100:
        # For 100-199 nodes, moderate scaling
        base_gamma = 1.2 + 0.4 * np.log10(n_nodes / 50)
        min_gamma = 0.8
        max_gamma = base_gamma * 2.0
        
        gamma_range = np.concatenate([
            np.linspace(min_gamma, base_gamma, 8),
            np.linspace(base_gamma, max_gamma, 12)[1:]
        ])
        
    else:
        # For smaller networks, use original scaling
        base_gamma = 0.5 + 0.3 * np.log10(max(n_nodes / 10, 1))
        min_gamma = base_gamma * 0.5
        max_gamma = base_gamma * 2.5
        
        gamma_range = np.concatenate([
            np.linspace(min_gamma, base_gamma, 6),
            np.linspace(base_gamma, max_gamma, 10)[1:]
        ])
    
    print(f"Network size: {n_nodes} nodes")
    print(f"Base gamma: {base_gamma:.2f}")
    print(f"Gamma range: [{min_gamma:.2f}, {max_gamma:.2f}] ({len(gamma_range)} values)")
    
    return gamma_range

def evaluate_partition_quality(correlation_matrix, partition, target_communities=(5, 15)):
    """
    Evaluate the quality of a partition using multiple criteria.
    
    Parameters:
    -----------
    correlation_matrix : numpy.ndarray
        The correlation matrix
    partition : dict
        Community partition
    target_communities : tuple
        Target range of communities
        
    Returns:
    --------
    dict
        Quality metrics
    """
    n_communities = len(set(partition.values()))
    
    # Calculate modularity
    _, modularity = calculate_modularity_signed(correlation_matrix, partition)
    
    # Community size balance (prefer more balanced communities)
    community_sizes = []
    for comm_id in set(partition.values()):
        size = sum(1 for v in partition.values() if v == comm_id)
        community_sizes.append(size)
    
    size_std = np.std(community_sizes)
    size_balance = 1.0 / (1.0 + size_std / np.mean(community_sizes))
    
    # Target community count score (penalize being outside target range)
    min_target, max_target = target_communities
    if min_target <= n_communities <= max_target:
        community_score = 1.0
    else:
        # Penalize being outside the range
        if n_communities < min_target:
            community_score = n_communities / min_target
        else:
            community_score = max_target / n_communities
    
    # Composite score
    composite_score = modularity * community_score * (0.5 + 0.5 * size_balance)
    
    return {
        'modularity': modularity,
        'n_communities': n_communities,
        'size_balance': size_balance,
        'community_score': community_score,
        'composite_score': composite_score,
        'community_sizes': community_sizes
    }

def consensus_louvain_adaptive(correlation_matrix, n_iterations=50, seed=None, 
                                 tau=0.5, target_communities=(5, 15)):
    """
    Improved adaptive consensus with network-size aware gamma selection.
    """
    n_nodes = correlation_matrix.shape[0]
    
    # Get adaptive gamma range
    gamma_range = get_adaptive_gamma_range(n_nodes, target_communities)
    
    print(f"\nRunning adaptive consensus clustering for {n_nodes} nodes...")
    print(f"Target communities: {target_communities[0]}-{target_communities[1]}")
    
    all_results = []
    
    for gamma in gamma_range:
        print(f"\nTrying gamma = {gamma:.3f}")
        
        try:
            # Run consensus with this gamma
            partition, consensus_matrix, all_partitions, modularity_scores = consensus_louvain_signed(
                correlation_matrix, n_iterations, tau=tau, seed=seed, gamma=gamma
            )
            
            # Evaluate quality
            quality = evaluate_partition_quality(correlation_matrix, partition, target_communities)
            
            print(f"  Communities: {quality['n_communities']}")
            print(f"  Modularity: {quality['modularity']:.3f}")
            print(f"  Composite score: {quality['composite_score']:.3f}")
            
            # Store result
            result = {
                'gamma': gamma,
                'partition': partition,
                'consensus_matrix': consensus_matrix,
                'all_partitions': all_partitions,
                'modularity_scores': modularity_scores,
                **quality
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"  Gamma {gamma:.3f} failed: {e}")
            continue
    
    if not all_results:
        raise RuntimeError("All gamma values failed!")
    
    # Select best result based on composite score
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)
    best_result = all_results[0]
    
    print(f"\nBest result:")
    print(f"  Gamma: {best_result['gamma']:.3f}")
    print(f"  Communities: {best_result['n_communities']}")
    print(f"  Modularity: {best_result['modularity']:.3f}")
    print(f"  Composite score: {best_result['composite_score']:.3f}")
    
    # Also show top 3 alternatives
    print(f"\nTop alternatives:")
    for i, result in enumerate(all_results[1:4], 1):
        print(f"  {i}. Gamma={result['gamma']:.3f}, "
              f"Communities={result['n_communities']}, "
              f"Q={result['modularity']:.3f}, "
              f"Score={result['composite_score']:.3f}")
    
    return (best_result['partition'], best_result['consensus_matrix'], 
            best_result['all_partitions'], all_results)

def consensus_louvain_multi_tau(correlation_matrix, n_iterations=50, seed=None, 
                               gamma_range=None, tau_range=None, target_communities=(5, 15)):
    """
    Try multiple tau values in addition to multiple gamma values.
    """
    n_nodes = correlation_matrix.shape[0]
    
    if gamma_range is None:
        gamma_range = get_adaptive_gamma_range(n_nodes, target_communities)
    
    if tau_range is None:
        tau_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"\nRunning multi-parameter consensus clustering...")
    print(f"Gamma range: {len(gamma_range)} values")
    print(f"Tau range: {tau_range}")
    
    all_results = []
    
    for gamma in gamma_range:
        for tau in tau_range:
            print(f"\nTrying gamma={gamma:.3f}, tau={tau:.2f}")
            
            try:
                # Run consensus
                partition, consensus_matrix, all_partitions, modularity_scores = consensus_louvain_signed(
                    correlation_matrix, n_iterations, tau=tau, seed=seed, gamma=gamma
                )
                
                # Evaluate quality
                quality = evaluate_partition_quality(correlation_matrix, partition, target_communities)
                
                print(f"  Communities: {quality['n_communities']}, Q: {quality['modularity']:.3f}")
                
                # Store result
                result = {
                    'gamma': gamma,
                    'tau': tau,
                    'partition': partition,
                    'consensus_matrix': consensus_matrix,
                    'all_partitions': all_partitions,
                    'modularity_scores': modularity_scores,
                    **quality
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
    
    if not all_results:
        raise RuntimeError("All parameter combinations failed!")
    
    # Select best result
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)
    best_result = all_results[0]
    
    print(f"\nBest result:")
    print(f"  Gamma: {best_result['gamma']:.3f}, Tau: {best_result['tau']:.2f}")
    print(f"  Communities: {best_result['n_communities']}")
    print(f"  Modularity: {best_result['modularity']:.3f}")
    
    return (best_result['partition'], best_result['consensus_matrix'], 
            best_result['all_partitions'], all_results)

def plot_parameter_landscape(all_results, n_nodes):
    """
    Plot the parameter landscape showing modularity and community count.
    """
    if not all_results:
        print("No results to plot")
        return
    
    # Extract data
    gammas = [r['gamma'] for r in all_results]
    modularities = [r['modularity'] for r in all_results]
    n_communities = [r['n_communities'] for r in all_results]
    composite_scores = [r['composite_score'] for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Parameter Landscape ({n_nodes} nodes)', fontsize=14)
    
    # Modularity vs gamma
    axes[0, 0].scatter(gammas, modularities, c=n_communities, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Gamma')
    axes[0, 0].set_ylabel('Modularity')
    axes[0, 0].set_title('Modularity vs Gamma')
    cbar1 = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar1.set_label('N Communities')
    
    # Communities vs gamma
    axes[0, 1].scatter(gammas, n_communities, c=modularities, cmap='plasma', alpha=0.7)
    axes[0, 1].set_xlabel('Gamma')
    axes[0, 1].set_ylabel('Number of Communities')
    axes[0, 1].set_title('Communities vs Gamma')
    axes[0, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Target range')
    axes[0, 1].axhline(y=15, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    cbar2 = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar2.set_label('Modularity')
    
    # Composite score vs gamma
    axes[1, 0].scatter(gammas, composite_scores, c=n_communities, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Gamma')
    axes[1, 0].set_ylabel('Composite Score')
    axes[1, 0].set_title('Composite Score vs Gamma')
    
    # Best results histogram
    top_results = sorted(all_results, key=lambda x: x['composite_score'], reverse=True)[:10]
    top_communities = [r['n_communities'] for r in top_results]
    axes[1, 1].hist(top_communities, bins=range(1, max(top_communities)+2), alpha=0.7)
    axes[1, 1].set_xlabel('Number of Communities')
    axes[1, 1].set_ylabel('Frequency (Top 10 Results)')
    axes[1, 1].set_title('Community Count Distribution (Best Results)')
    
    plt.tight_layout()
    return fig

def consensus_louvain_signed(correlation_matrix, n_iterations=100, tau=0.5, seed=None, gamma=1.0):
    """
    Consensus community detection using BCT's modularity_louvain_und_sign.
    Designed for signed correlation matrices (e.g., rsfMRI).

    Parameters:
    -----------
    correlation_matrix : numpy.ndarray
        N x N correlation matrix with positive and negative weights
    n_iterations : int
        Number of Louvain iterations to run
    tau : float
        Consensus threshold (0.5 = majority rule)
    seed : int, optional
        Random seed for reproducibility
    gamma : float
        Resolution parameter for modularity (higher = more communities)

    Returns:
    --------
    consensus_partition : dict
        Final consensus partition {node: community_id}
    consensus_matrix : numpy.ndarray
        N x N matrix with consensus values
    all_partitions : list
        All individual partitions from iterations
    modularity_scores : list
        Modularity score for each iteration
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Preprocess the matrix
    matrix = preprocess_correlation_matrix(correlation_matrix)
    n_nodes = matrix.shape[0]

    # Store all partitions and modularity scores
    all_partitions = []
    modularity_scores = []
    
    print(f"Running {n_iterations} iterations of BCT signed Louvain...")
    print(f"Gamma (resolution): {gamma}")
    
    # Run BCT signed Louvain multiple times
    for i in range(n_iterations):
        try:
            # BCT modularity_louvain_und_sign returns (partition, modularity)
            partition, modularity = bct.modularity_louvain_und_sign(
                matrix, gamma=gamma, seed=seed+i if seed is not None else None
            )
            
            # Debug: Check what BCT returned
            if partition is None:
                print(f"Warning: Iteration {i} returned None partition")
                continue
                
            if not isinstance(partition, np.ndarray):
                print(f"Warning: Iteration {i} returned non-array: {type(partition)}")
                continue
                
            if len(partition) != n_nodes:
                print(f"Warning: Iteration {i} returned partition of wrong size: {len(partition)} vs {n_nodes}")
                continue
                
            # Check for valid partition values
            if np.any(np.isnan(partition)) or np.any(partition < 1):
                print(f"Warning: Iteration {i} returned invalid partition values")
                continue
            
            # BCT returns 1-indexed communities, convert to 0-indexed
            partition_0indexed = partition - 1
            
            # Convert to dictionary format
            partition_dict = {node: int(partition_0indexed[node]) for node in range(n_nodes)}
            
            all_partitions.append(partition_dict)
            modularity_scores.append(modularity)
            
        except Exception as e:
            print(f"Warning: Iteration {i} failed with error: {e}")
            continue
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{n_iterations} iterations")
            if len(modularity_scores) > 0:
                print(f"Mean modularity so far: {np.mean(modularity_scores):.3f} ± {np.std(modularity_scores):.3f}")
    
    # Check if we got any successful iterations
    if len(all_partitions) == 0:
        raise RuntimeError("All iterations failed! This might be due to:\n"
                          "1. Matrix has NaN or infinite values\n"
                          "2. Matrix is not symmetric\n"
                          "3. Matrix diagonal is not zero\n"
                          "4. BCT version incompatibility\n"
                          "Please check your correlation matrix.")
    
    print(f"Successfully completed {len(all_partitions)}/{n_iterations} iterations")
    if len(modularity_scores) > 0:
        print(f"Final modularity: {np.mean(modularity_scores):.3f} ± {np.std(modularity_scores):.3f}")
    else:
        raise RuntimeError("No successful iterations completed!")
    
    # Build consensus matrix
    consensus_matrix = np.zeros((n_nodes, n_nodes))
    
    for partition in all_partitions:
        for i in range(n_nodes):
            for j in range(n_nodes):
                if partition[i] == partition[j]:
                    consensus_matrix[i, j] += 1
    
    # Normalize by number of successful iterations (not total iterations)
    consensus_matrix = consensus_matrix / len(all_partitions)
    
    # Apply consensus threshold
    consensus_adjacency = (consensus_matrix >= tau).astype(int)
    
    # Extract final communities from consensus adjacency matrix
    G_consensus = nx.from_numpy_array(consensus_adjacency)
    final_communities = list(nx.connected_components(G_consensus))
    
    # Convert to partition dictionary
    consensus_partition = {}
    for comm_id, community in enumerate(final_communities):
        for node in community:
            consensus_partition[node] = comm_id
    
    print(f"Consensus clustering complete!")
    print(f"Final number of communities: {len(final_communities)}")
    
    return consensus_partition, consensus_matrix, all_partitions, modularity_scores

def calculate_modularity_signed(correlation_matrix, partition):
    """
    Calculate signed modularity using BCT.
    """
    # Convert partition dict to array (1-indexed for BCT)
    partition_array = np.array([partition[i] for i in range(len(partition))]) + 1
    return bct.modularity_und_sign(correlation_matrix, partition_array)

def preprocess_correlation_matrix(correlation_matrix):
    """
    Preprocess the correlation matrix for consensus clustering:
    - Handle NaN/inf values
    - Ensure symmetry
    - Set diagonal to zero
    - Clip values to [-1, 1] for signed networks

    Parameters:
    -----------
    correlation_matrix : numpy.ndarray
        N x N correlation matrix with positive and negative weights
    Returns:
    --------
    numpy.ndarray
        Preprocessed correlation matrix ready for consensus clustering
    -----------
    """
    matrix = correlation_matrix.copy()

    # Handle NaN/inf
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        print("Warning: Found NaN/inf values in matrix, replacing with 0")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure symmetry
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    # For signed networks, ensure reasonable range
    matrix = np.clip(matrix, -1, 1)

    return matrix

def analyze_consensus_stability(all_partitions):
    """
    Analyze stability of consensus using Adjusted Rand Index.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("Warning: sklearn not available, skipping stability analysis")
        return []

    n_partitions = len(all_partitions)
    if n_partitions < 2:
        print("Warning: Need at least 2 partitions for stability analysis")
        return []

    n_nodes = len(all_partitions[0])

    # Calculate pairwise ARI scores
    ari_scores = []
    for i in range(n_partitions):
        for j in range(i + 1, n_partitions):
            labels1 = [all_partitions[i][k] for k in range(n_nodes)]
            labels2 = [all_partitions[j][k] for k in range(n_nodes)]
            ari = adjusted_rand_score(labels1, labels2)
            ari_scores.append(ari)

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)

    print(f"\nStability Analysis:")
    print(f"Mean ARI: {mean_ari:.3f} ± {std_ari:.3f}")
    print(f"Range: [{np.min(ari_scores):.3f}, {np.max(ari_scores):.3f}]")

    return ari_scores


def plot_consensus_matrix(consensus_matrix, consensus_partition=None,
                         node_labels=None, figsize=(10, 8), output_file=None):
    """
    Plot the consensus matrix, optionally reordered by communities.

    Parameters:
    -----------
    consensus_matrix : numpy.ndarray
        N x N matrix with consensus values
    consensus_partition : dict, optional
        Partition mapping {node: community_id}
    node_labels : list, optional
        Labels for nodes
    figsize : tuple, optional
        Figure size (default: (10, 8))
    output_file : str or Path, optional
        Path to save the figure. If None, figure is not displayed or saved.
    """
    n_nodes = consensus_matrix.shape[0]

    if node_labels is None:
        node_labels = [f"Node_{i}" for i in range(n_nodes)]

    # Reorder by communities if partition provided
    if consensus_partition is not None:
        # Sort nodes by community assignment
        node_order = sorted(range(n_nodes), key=lambda x: consensus_partition[x])
        reordered_matrix = consensus_matrix[np.ix_(node_order, node_order)]
        reordered_labels = [node_labels[i] for i in node_order]

        # Get community boundaries for plotting
        communities = {}
        for node, comm in consensus_partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
    else:
        reordered_matrix = consensus_matrix
        reordered_labels = node_labels
        communities = None

    fig = plt.figure(figsize=figsize)

    # Create heatmap
    im = plt.imshow(reordered_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, label='Consensus Value')

    # Add community boundaries if partition provided
    if communities is not None:
        # Draw community boundaries
        cumulative = 0
        for comm_id in sorted(communities.keys()):
            comm_size = len(communities[comm_id])
            if cumulative > 0:
                plt.axhline(y=cumulative-0.5, color='red', linewidth=2)
                plt.axvline(x=cumulative-0.5, color='red', linewidth=2)
            cumulative += comm_size

    plt.title('Consensus Matrix')
    plt.xlabel('Nodes')
    plt.ylabel('Nodes')

    # Set ticks if not too many nodes
    if n_nodes <= 50:
        plt.xticks(range(n_nodes), reordered_labels, rotation=45, ha='right')
        plt.yticks(range(n_nodes), reordered_labels)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Consensus matrix plot saved to: {output_file}")

    plt.close(fig)


def print_community_summary(partition, node_labels=None):
    """
    Print summary of community assignments.
    """
    if node_labels is None:
        node_labels = [f"Node_{i}" for i in range(len(partition))]

    # Group nodes by community
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)

    print(f"\nCommunity assignments:")
    print(f"Total communities found: {len(communities)}")

    for comm_id in sorted(communities.keys()):
        nodes = communities[comm_id]
        node_names = [node_labels[node] for node in nodes]
        print(f"Community {comm_id}: {len(nodes)} nodes")
        if len(nodes) <= 10:  # Only show node names if community is small
            print(f"  Nodes: {', '.join(node_names)}")
        else:
            print(f"  Nodes: {', '.join(node_names[:5])} ... (and {len(nodes)-5} more)")

    return communities

def find_optimal_tau(consensus_matrix, tau_range=None):
    """
    Find optimal tau by looking for stable number of communities.
    If tau_range is None, use default range from 0.1 to 0.9 in steps of 0.1.
    """
    if tau_range is None:
        tau_range = np.arange(0.1, 1.0, 0.1)

    n_communities = []

    for tau in tau_range:
        consensus_adjacency = (consensus_matrix >= tau).astype(int)
        G_consensus = nx.from_numpy_array(consensus_adjacency)
        n_comms = len(list(nx.connected_components(G_consensus)))
        n_communities.append(n_comms)

    # Look for plateaus in number of communities
    differences = np.diff(n_communities)
    stable_regions = np.where(differences == 0)[0]

    if len(stable_regions) > 0:
        # Choose tau in the middle of the most stable region
        optimal_tau = tau_range[stable_regions[len(stable_regions)//2]]
    else:
        # Fallback to tau that gives reasonable number of communities
        target_range = (3, len(consensus_matrix) // 5)  # 3 to N/5 communities
        valid_indices = [i for i, n in enumerate(n_communities) 
                        if target_range[0] <= n <= target_range[1]]
        if valid_indices:
            optimal_tau = tau_range[valid_indices[0]]
        else:
            optimal_tau = 0.5

    return optimal_tau, tau_range, n_communities

def calculate_modularity(adjacency_matrix, partition):
    """
    Calculate modularity Q of a partition.
    """
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Convert partition to list of communities
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)
    
    community_list = [set(nodes) for nodes in communities.values()]
    
    # Calculate modularity
    modularity = nx.community.modularity(G, community_list)
    return modularity

def plot_communities_on_surface(partition, project_folder, cmap):
    """
    Plot the communities on the brain surface using connectome_viz.

    Parameters:
    -----------
    partition : dict
        Community partition {node: community_id}
    project_folder : Path
        Path to the project folder containing data files
    """

    # Convert partition to 1-indexed vector for CIFTI compatibility
    partition_vector = pd.Series(partition).sort_index()
    partition_vector = partition_vector.astype('int') + 1  # Convert to 1-indexed for CIFTI
    n_nodes = len(partition_vector)

    dlabel_filename = project_folder / f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_ftb.dlabel.nii'
    dscalar_filename = project_folder / f'data/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC.dscalar.nii'

    left_full, right_full = connectome_viz.mask_dlabel_data(dscalar_filename, dlabel_filename)
    lh_val, rh_val = left_full.copy(), right_full.copy()

    for i in range(1, len(partition_vector)+1):
        lh_val[left_full == i] = partition_vector[i-1]
        rh_val[right_full == i] = partition_vector[i-1]

    width, height = int(120 * 300 / 25.4), int(90 * 300 / 25.4)
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']

    p = Plot(lh, rh, zoom=1.5, size=(width, height))
    p.add_layer({'left': lh_val, 'right': rh_val},
                cmap=cmap,
                cbar=False,
    )
    p.add_layer({'left': left_full, 'right': right_full}, cmap='gray', as_outline=True, cbar=False)
    fig = p.build(cbar_kws={'fontsize': 16, 'pad': 0.05})

    return fig

def main():
    """Main function to find network communities in fMRI data."""
    parser = argparse.ArgumentParser(
        description='Find network communities in fMRI connectivity data'
    )
    parser.add_argument('--project', required=True,
                        help='Path to project directory')
    parser.add_argument('--matrices-dir', required=False,
                        help='Path to connectivity matrices directory')
    parser.add_argument('--ids', required=False,
                        help='Path to subject IDs file')
    parser.add_argument('--parcellations', nargs='+', type=int,
                        default=[50, 100, 200],
                        help='Parcellation sizes to analyze (default: 50 100 200)')
    parser.add_argument('--n-iterations', type=int, default=50,
                        help='Number of consensus clustering iterations (default: 50)')
    parser.add_argument('--target-communities', nargs=2, type=int,
                        default=[5, 15],
                        help='Target community count range (default: 5 15)')
    args = parser.parse_args()

    project_folder = Path(args.project)

    # Create necessary directories
    figures_dir = project_folder / 'figures'
    reports_dir = project_folder / 'reports'
    results_dir = project_folder / 'results'

    if not figures_dir.exists():
        figures_dir.mkdir(parents=True, exist_ok=True)
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = [
        "=" * 80,
        "C3: NETWORK COMMUNITY DETECTION REPORT",
        "=" * 80,
        "",
        "Analysis parameters:",
        f"  Parcellation sizes: {args.parcellations}",
        f"  Consensus iterations: {args.n_iterations}",
        f"  Target communities: {args.target_communities[0]}-{args.target_communities[1]}",
        ""
    ]

    # Set up file paths with defaults
    matrices_dir = (Path(args.matrices_dir) if args.matrices_dir 
                    else project_folder / 'data/HCP_PTN1200/netmats')
    id_file = (Path(args.ids) if args.ids 
               else project_folder / 'data/subjectIDs_anonymised.txt')

    for n_nodes in args.parcellations:
        msg = f"\n{'='*60}\nRunning analysis for {n_nodes} nodes\n{'='*60}"
        print(msg)
        report.append(msg)

        # Load connectivity matrices (full correlation)
        matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'

        mats_df = pd.read_csv(matrix_file, header=None, sep=r'\s+')

        # Remove lower triangle for non-redundancy
        lower_indices = np.tril_indices(n_nodes, k=-1)
        linear_indices = lower_indices[0] * n_nodes + lower_indices[1]
        mats_df = mats_df.iloc[:, linear_indices]

        # Calculate the average connectivity matrix
        avg_mat = mats_df.mean(axis=0).values
        avg_mat = avg_mat/100  # Normalize to [-1, 1] range
        correlation_matrix = vec_to_sym_matrix(avg_mat, diagonal=np.zeros(n_nodes))

        matrix_info = [
            f"Correlation matrix range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]",
            f"Negative correlations: {(correlation_matrix < 0).sum()} / {correlation_matrix.size}"
        ]
        for info in matrix_info:
            print(info)
            report.append(info)

        # Run consensus clustering with BCT signed modularity
        header = f"\n{'='*60}\nBCT Signed Modularity Consensus Clustering\n{'='*60}"
        print(header)
        report.append(header)

        # Method 1: Manual parameters
        print("\n--- Manual parameters ---")
        report.append("\n--- Manual parameters ---")
        manual_partition, manual_consensus, manual_partitions, manual_mod_scores = consensus_louvain_signed(
            correlation_matrix,
            n_iterations=args.n_iterations,
            tau=0.5,
            gamma=1.0,
            seed=42
        )

        manual_results = [
            f"Manual - Communities: {len(set(manual_partition.values()))}",
            f"Manual - Mean Q: {np.mean(manual_mod_scores):.3f} ± {np.std(manual_mod_scores):.3f}"
        ]
        for result in manual_results:
            print(result)
            report.append(result)

        # Method 2: Adaptive parameters
        print("\n--- Adaptive parameters ---")
        report.append("\n--- Adaptive parameters ---")
        adaptive_partition, adaptive_consensus, adaptive_partitions, all_results = consensus_louvain_adaptive(
            correlation_matrix,
            n_iterations=args.n_iterations,
            target_communities=tuple(args.target_communities),
            seed=42
        )

        # Calculate modularity for adaptive result
        _, adaptive_Q = calculate_modularity_signed(correlation_matrix, adaptive_partition)
        adaptive_results = [
            f"Adaptive - Communities: {len(set(adaptive_partition.values()))}",
            f"Adaptive - Final Q: {adaptive_Q:.3f}"
        ]
        for result in adaptive_results:
            print(result)
            report.append(result)

        # Use the adaptive result for final analysis
        final_header = f"\n{'='*60}\nFinal Analysis\n{'='*60}"
        print(final_header)
        report.append(final_header)

        final_partition = adaptive_partition
        final_consensus_matrix = adaptive_consensus
        final_partitions = adaptive_partitions

        # Calculate final modularity
        _, final_Q = calculate_modularity_signed(correlation_matrix, final_partition)
        final_info = f"Final signed modularity: {final_Q:.3f}"
        print(final_info)
        report.append(final_info)

        # Analyze final results
        ari_scores = analyze_consensus_stability(final_partitions)
        consensus_plot_file = figures_dir / f'C2_consensus_matrix_{n_nodes}Nodes.png'
        plot_consensus_matrix(final_consensus_matrix, final_partition,
                             output_file=consensus_plot_file)

        # Print community summary
        communities = print_community_summary(final_partition)

        # Plot communities on surface
        if n_nodes == 50:
            cmap = ListedColormap([
                '#A8B8C8',  # Other (softer blue-gray)
                '#C85450',  # DMN (warmer red)
                '#2DB574',  # VAN (balanced green)
            ])
        elif n_nodes == 100:
            cmap = ListedColormap([
                '#7B2D8E',  # Visual Network (deeper purple)
                '#C85450',  # DMN (warmer red)
                '#A8B8C8',  # Other (softer blue-gray)
                '#D17A47',  # FPN (warmer orange)
                '#2DB574',  # VAN (balanced green)
                '#4FB3D9'   # Other (warmer cyan)
            ])
        elif n_nodes == 200:
            cmap = ListedColormap([
                '#7B2D8E',  # Visual Network (deeper purple)
                '#A8B8C8',  # Other (softer blue-gray)
                '#C85450',  # DMN (warmer red)
                '#2DB574',  # VAN (balanced green)
                '#E69422'   # SomMot (warmer orange)
            ])
        fig = plot_communities_on_surface(final_partition, project_folder, cmap=cmap)
        fig.savefig(figures_dir / f'C2_community_plot_{n_nodes}Nodes.png', dpi=300)
        plt.close(fig)

        # Save the final partition to a file
        partition_df = pd.DataFrame.from_dict(final_partition, orient='index', columns=['community_id'])
        partition_df.index.name = 'node'
        partition_df.sort_index(inplace=True)
        partition_df.to_csv(results_dir / f'C2_final_partition_{n_nodes}Nodes.csv')
        report.append(f"Saved partition to: C2_final_partition_{n_nodes}Nodes.csv")

        # Plot parameter landscape
        print("\nPlotting parameter landscape...")
        report.append("Plotting parameter landscape...")
        landscape_fig = plot_parameter_landscape(all_results, n_nodes)
        if landscape_fig is not None:
            landscape_file = figures_dir / f'C2_parameter_landscape_{n_nodes}Nodes.png'
            landscape_fig.savefig(landscape_file, dpi=300, bbox_inches='tight')
            plt.close(landscape_fig)
            report.append(f"Parameter landscape saved to: {landscape_file}")

    # Write report to file
    report.append("")
    report.append("=" * 80)
    report.append("ANALYSIS COMPLETE")
    report.append("=" * 80)
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_file = reports_dir / 'C2_find_communities_fMRI_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()

# %%
