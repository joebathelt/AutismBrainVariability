"""
C2b: Parcellation Resolution Evaluation
========================================
Evaluate community detection stability and quality across all available
HCP ICA parcellation sizes to select optimal resolution for landscape analysis.

Selection criteria (defined a priori, independent of PGS hypothesis):
  1. Number of communities in expected range for resting-state networks (5-8)
  2. Highest joint score: geometric mean of mean pairwise ARI (partition
     stability across consensus iterations) and modularity Q (partition
     quality). Using the geometric mean requires both metrics to be high;
     a strong score on one cannot compensate for a weak score on the other.

Usage:
    python C2b_evaluate_parcellations.py --project /app
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from nilearn.connectome import vec_to_sym_matrix

# Import only what is needed to evaluate loaded partitions
from C2_find_communities_fMRI import (
    calculate_modularity_signed,
)

# ==============================================================================
# Configuration
# ==============================================================================

# A priori target: canonical resting-state networks (Yeo et al., 2011)
TARGET_COMMUNITIES = (6, 8)

# Consensus clustering parameters
N_ITERATIONS = 50
TAU = 0.5
SEED = 42

# Available HCP ICA dimensionalities
HCP_PARCELLATIONS = [15, 25, 50, 100, 200, 300]


def load_connectivity_matrix(matrices_dir, n_nodes):
    """Load and compute group-average connectivity matrix for a parcellation."""
    matrix_file = (
        matrices_dir
        / f"3T_HCP1200_MSMAll_d{n_nodes}_ts2"
        / "netmats1.txt"
    )

    if not matrix_file.exists():
        print(f"  Matrix file not found: {matrix_file}")
        return None

    mats_df = pd.read_csv(matrix_file, header=None, sep=r"\s+")

    # Extract lower triangle for non-redundancy
    lower_indices = np.tril_indices(n_nodes, k=-1)
    linear_indices = lower_indices[0] * n_nodes + lower_indices[1]
    mats_df = mats_df.iloc[:, linear_indices]

    # Compute group-average matrix
    avg_mat = mats_df.mean(axis=0).values / 100  # Normalize to [-1, 1]
    correlation_matrix = vec_to_sym_matrix(avg_mat, diagonal=np.zeros(n_nodes))

    return correlation_matrix


def compute_ari_scores(all_partitions):
    """Compute pairwise ARI scores across consensus iterations."""
    n_partitions = len(all_partitions)
    if n_partitions < 2:
        return np.array([])

    n_nodes = len(all_partitions[0])
    ari_scores = []

    for i in range(n_partitions):
        for j in range(i + 1, n_partitions):
            labels1 = [all_partitions[i][k] for k in range(n_nodes)]
            labels2 = [all_partitions[j][k] for k in range(n_nodes)]
            ari_scores.append(adjusted_rand_score(labels1, labels2))

    return np.array(ari_scores)


def evaluate_single_parcellation(correlation_matrix, n_nodes, results_dir, report):
    """Load C2 community solution and compute evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {n_nodes}-node parcellation")
    print(f"{'='*60}")

    report.append(f"\n{'='*60}")
    report.append(f"Evaluating {n_nodes}-node parcellation")
    report.append(f"{'='*60}")

    # --- Load C2's final partition ---
    partition_file = results_dir / f"C2_final_partition_{n_nodes}Nodes.csv"
    if not partition_file.exists():
        msg = f"  SKIPPED: {partition_file} not found. Run C2_find_communities_fMRI.py first."
        print(msg)
        report.append(msg)
        return None

    partition_df = pd.read_csv(partition_file, index_col="node")
    partition = {int(i): int(v) for i, v in partition_df["community_id"].items()}
    print(f"  Loaded partition from {partition_file.name}")

    # --- Load C2's iteration partitions for ARI ---
    iter_file = results_dir / f"C2_all_partitions_{n_nodes}Nodes.npy"
    if iter_file.exists():
        arr = np.load(iter_file)
        all_partitions = [
            {node: int(arr[i, node]) for node in range(n_nodes)}
            for i in range(arr.shape[0])
        ]
        print(f"  Loaded {len(all_partitions)} iteration partitions for ARI")
    else:
        all_partitions = []
        print(f"  Iteration partitions not found; ARI will be NaN")

    # --- Compute metrics ---
    n_communities = len(set(partition.values()))
    _, Q = calculate_modularity_signed(correlation_matrix, partition)

    ari_scores = compute_ari_scores(all_partitions)
    mean_ari = float(np.mean(ari_scores)) if len(ari_scores) > 0 else np.nan
    std_ari = float(np.std(ari_scores)) if len(ari_scores) > 0 else np.nan

    comm_sizes = {}
    for comm in partition.values():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
    sizes = sorted(comm_sizes.values(), reverse=True)
    size_cv = np.std(sizes) / np.mean(sizes)

    in_target = TARGET_COMMUNITIES[0] <= n_communities <= TARGET_COMMUNITIES[1]

    report.append(f"\n  Loaded from C2 (adaptive gamma):")
    report.append(f"    Communities: {n_communities}")
    report.append(f"    Q: {Q:.3f}")
    if not np.isnan(mean_ari):
        report.append(f"    Mean ARI: {mean_ari:.3f} ± {std_ari:.3f}")
        report.append(f"    ARI range: [{np.min(ari_scores):.3f}, {np.max(ari_scores):.3f}]")
    else:
        report.append(f"    Mean ARI: N/A")
    report.append(f"    Community sizes: {sizes}")
    report.append(f"    Size CV: {size_cv:.3f}")
    report.append(f"    In target range: {'YES' if in_target else 'NO'}")

    return {
        "n_nodes": n_nodes,
        "n_communities": n_communities,
        "Q": Q,
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "in_target_range": in_target,
        "community_sizes": sizes,
        "size_cv": size_cv,
        "partition": partition,
        "all_partitions": all_partitions,
    }


def plot_tuning_curves(results_df, target_range, output_file):
    """
    Create tuning curve figure: n_communities, Q, and ARI vs parcellation size.

    Three panels showing how C2 community solutions vary with parcellation resolution.
    Shaded band in the community count panel marks the a priori target range.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    nodes = results_df["n_nodes"].values
    x_pos = np.arange(len(nodes))

    # --- Panel A: Number of communities ---
    ax = axes[0]
    ax.axhspan(
        target_range[0], target_range[1],
        alpha=0.15, color="tab:green",
        label=f"Target ({target_range[0]}–{target_range[1]})",
    )
    ax.plot(x_pos, results_df["n_communities"], "s-", color="tab:blue", markersize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nodes)
    ax.set_xlabel("Parcellation size (ICA dimensionality)")
    ax.set_ylabel("Number of communities")
    ax.set_title("A. Community count")
    ax.legend(fontsize=8, loc="upper left")

    # --- Panel B: Modularity Q ---
    ax = axes[1]
    ax.plot(x_pos, results_df["Q"], "s-", color="tab:blue", markersize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nodes)
    ax.set_xlabel("Parcellation size (ICA dimensionality)")
    ax.set_ylabel("Modularity (Q)")
    ax.set_title("B. Partition quality")

    # --- Panel C: Mean ARI ---
    ax = axes[2]
    ax.errorbar(
        x_pos, results_df["mean_ari"], yerr=results_df["std_ari"],
        fmt="s-", color="tab:blue", markersize=7, capsize=3,
    )
    ax.axhline(0.8, color="tab:red", linestyle=":", alpha=0.5, label="ARI = 0.8")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nodes)
    ax.set_xlabel("Parcellation size (ICA dimensionality)")
    ax.set_ylabel("Mean pairwise ARI")
    ax.set_title("C. Partition stability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTuning curve saved to: {output_file}")


def compute_joint_score(mean_ari, Q):
    """Geometric mean of ARI and Q, guarded for non-positive values.

    Returns 0 when either metric is non-positive, so a parcellation cannot
    be rewarded for strength in one dimension while failing on the other.
    NaN inputs (e.g. ARI unavailable) propagate as NaN.
    """
    if np.isnan(mean_ari) or np.isnan(Q):
        return np.nan
    if mean_ari <= 0 or Q <= 0:
        return 0.0
    return float(np.sqrt(mean_ari * Q))


def select_optimal_parcellation(results_df, target_range):
    """
    Select parcellation based on a priori criteria.

    Priority:
      1. Communities within target range
      2. Highest joint score = sqrt(mean_ari * Q) among those meeting (1)
    """
    # Filter to those within target range
    in_range = results_df[results_df["in_target_range"]].copy()

    if len(in_range) == 0:
        print("\nWARNING: No parcellation produced communities in target range.")
        print("Selecting parcellation with community count closest to target.")
        mid_target = (target_range[0] + target_range[1]) / 2
        results_df["dist_to_target"] = abs(results_df["n_communities"] - mid_target)
        best = results_df.sort_values(
            ["dist_to_target", "joint_score"], ascending=[True, False]
        ).iloc[0]
    else:
        best = in_range.sort_values("joint_score", ascending=False).iloc[0]

    return int(best["n_nodes"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate parcellation resolution for community detection"
    )
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument(
        "--matrices-dir", required=False,
        help="Path to connectivity matrices directory",
    )
    parser.add_argument(
        "--parcellations", nargs="+", type=int, default=HCP_PARCELLATIONS,
        help=f"Parcellation sizes to test (default: {HCP_PARCELLATIONS})",
    )
    parser.add_argument(
        "--target-communities", nargs=2, type=int, default=list(TARGET_COMMUNITIES),
        help=f"Target community count range (default: {TARGET_COMMUNITIES})",
    )
    args = parser.parse_args()

    project_folder = Path(args.project)
    figures_dir = project_folder / "figures"
    reports_dir = project_folder / "reports"
    results_dir = project_folder / "results"

    for d in [figures_dir, reports_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    matrices_dir = (
        Path(args.matrices_dir)
        if args.matrices_dir
        else project_folder / "data/HCP_PTN1200/netmats"
    )

    target_range = tuple(args.target_communities)

    # Initialize report
    report = [
        "=" * 80,
        "C2b: PARCELLATION RESOLUTION EVALUATION",
        "=" * 80,
        "",
        "A priori selection criteria:",
        f"  Target community count: {target_range[0]}-{target_range[1]}",
        f"  (Based on canonical resting-state networks; Yeo et al., 2011)",
        f"  Selection rule: highest joint score = sqrt(mean_ARI * Q)",
        f"                  among parcellations meeting the community-count target",
        "",
        f"Consensus parameters:",
        f"  Iterations: {N_ITERATIONS}",
        f"  Tau: {TAU}",
        f"  Seed: {SEED}",
        "",
        f"Parcellations to evaluate: {args.parcellations}",
    ]

    # =========================================================================
    # Evaluate each parcellation
    # =========================================================================
    all_eval_results = []

    for n_nodes in args.parcellations:
        corr_mat = load_connectivity_matrix(matrices_dir, n_nodes)
        if corr_mat is None:
            report.append(f"\n  SKIPPED {n_nodes} nodes: matrix not found")
            continue

        result = evaluate_single_parcellation(corr_mat, n_nodes, results_dir, report)
        if result is not None:
            all_eval_results.append(result)

    if not all_eval_results:
        raise RuntimeError("No parcellations could be evaluated!")

    # =========================================================================
    # Compile results table
    # =========================================================================
    summary_cols = [
        "n_nodes", "n_communities", "Q", "mean_ari", "std_ari",
        "in_target_range", "size_cv",
    ]
    results_df = pd.DataFrame(
        [{k: r[k] for k in summary_cols} for r in all_eval_results]
    )
    results_df["joint_score"] = [
        compute_joint_score(row["mean_ari"], row["Q"])
        for _, row in results_df.iterrows()
    ]

    # Summary table for report
    report.append(f"\n\n{'='*80}")
    report.append("SUMMARY TABLE")
    report.append(f"{'='*80}")
    report.append(
        f"\n{'Nodes':>6} | {'N comm':>6} | {'Q':>6} | {'ARI':>10} | "
        f"{'Joint':>6} | {'Size CV':>7} | {'In target':>9}"
    )
    report.append("-" * 75)
    for _, row in results_df.iterrows():
        ari_str = (
            f"{row['mean_ari']:.3f}±{row['std_ari']:.3f}"
            if not np.isnan(row["mean_ari"])
            else "       N/A"
        )
        joint_str = (
            f"{row['joint_score']:.3f}"
            if not np.isnan(row["joint_score"])
            else "  N/A"
        )
        report.append(
            f"{int(row['n_nodes']):>6} | "
            f"{int(row['n_communities']):>6} | "
            f"{row['Q']:>6.3f} | "
            f"{ari_str:>10} | "
            f"{joint_str:>6} | "
            f"{row['size_cv']:>7.3f} | "
            f"{'YES' if row['in_target_range'] else 'NO':>9}"
        )

    # =========================================================================
    # Select optimal parcellation
    # =========================================================================
    optimal = select_optimal_parcellation(results_df, target_range)

    report.append(f"\n\n{'='*80}")
    report.append("PARCELLATION SELECTION")
    report.append(f"{'='*80}")
    report.append(f"\n  Selected parcellation: {optimal} nodes")

    optimal_row = results_df[results_df["n_nodes"] == optimal].iloc[0]
    report.append(f"  Communities: {int(optimal_row['n_communities'])}")
    report.append(f"  Q: {optimal_row['Q']:.3f}")
    report.append(f"  Mean ARI: {optimal_row['mean_ari']:.3f}" if not np.isnan(optimal_row['mean_ari']) else "  Mean ARI: N/A")
    report.append(f"  Joint score (sqrt(ARI*Q)): {optimal_row['joint_score']:.3f}" if not np.isnan(optimal_row['joint_score']) else "  Joint score: N/A")
    report.append(f"  In target range: {optimal_row['in_target_range']}")

    # =========================================================================
    # Save outputs
    # =========================================================================

    # Tuning curve figure
    plot_tuning_curves(
        results_df, target_range, figures_dir / "C2b_parcellation_tuning_curves.png"
    )

    # Results CSV
    results_df.to_csv(
        results_dir / "C2b_parcellation_evaluation.csv", index=False
    )
    report.append(f"\nSaved: C2b_parcellation_evaluation.csv")

    # Save selected partition
    optimal_result = next(r for r in all_eval_results if r["n_nodes"] == optimal)
    partition_df = pd.DataFrame.from_dict(
        optimal_result["partition"], orient="index", columns=["community_id"]
    )
    partition_df.index.name = "node"
    partition_df.sort_index(inplace=True)
    partition_df.to_csv(results_dir / f"C2b_selected_partition_{optimal}Nodes.csv")
    report.append(f"Saved: C2b_selected_partition_{optimal}Nodes.csv")

    # Fixed-name copy so downstream rules can consume the selection without
    # knowing the chosen parcellation size in advance. Downstream scripts
    # derive n_nodes from len(partition).
    partition_df.to_csv(results_dir / "C2b_selected_partition.csv")
    report.append(f"Saved: C2b_selected_partition.csv (n_nodes = {optimal})")

    # Write report
    report.append(f"\n\n{'='*80}")
    report.append("END OF REPORT")
    report.append(f"{'='*80}")

    report_file = reports_dir / "C2b_evaluate_parcellations_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to: {report_file}")

    print(f"\n*** SELECTED PARCELLATION: {optimal} nodes ***")


if __name__ == "__main__":
    main()