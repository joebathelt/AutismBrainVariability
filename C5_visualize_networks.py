#!/usr/bin/env python3
"""
C5: Network Visualisation (SDS-stratified)
==========================================

Visualisations for the SDS-stratified landscape analysis. Groups participants
by Social_Score (SDS) z-score using the same buffered cuts as C3
(low_sds / middle / high_sds; buffer rows dropped):

  1. Bootstrap-averaged connectivity matrices per SDS group
  2. Spring-layout networks per SDS group
  3. Per-subject exemplar networks (4 exemplars x 4 modularity quartiles
     per SDS group)
  4. Bootstrap density plots for modularity and global efficiency
  5. Bootstrap ellipse extent plot in modularity x efficiency space

PGS is intentionally not loaded here.

Usage:
    python C5_visualize_networks.py \
        --project /path/to/project \
        --social results/cfa_factor_scores_full_sample.csv \
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

NODE_COLOUR_DICT = {
    0: '#7B2D8E', 1: '#C85450', 2: '#A8B8C8',
    3: '#D17A47', 4: '#2DB574', 5: '#4FB3D9',
}

SDS_GROUPS = ['low_sds', 'middle', 'high_sds']
SDS_GROUP_LABELS = {'low_sds': 'Low SDS', 'middle': 'Middle', 'high_sds': 'High SDS'}
SDS_ELLIPSE_COLOURS = {'low_sds': '#9BB3C7', 'middle': '#7FB069', 'high_sds': '#4FB3D9'}
SDS_DENSITY_COLOURS = {'low_sds': '#8491B499', 'middle': '#91D1C299', 'high_sds': '#4DBBD599'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="C5: SDS-stratified network visualisation"
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--social", required=True,
                        help="Social factor scores CSV (Social_Score column)")
    parser.add_argument("--graph-metrics", required=True,
                        help="Main-threshold graph metrics CSV from C3 "
                             "(C4_main_network_metrics.csv)")
    parser.add_argument("--partition", required=True,
                        help="Community partition from C2b "
                             "(C2b_selected_partition.csv)")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--matrices-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=90)
    return parser.parse_args()


def assign_sds_groups(social_df):
    """Z-score Social_Score and assign low_sds / middle / high_sds with buffer
    rows dropped (matches C3)."""
    df = social_df.copy()
    df['sds_z'] = zscore(df['Social_Score'])
    df['sds_group'] = pd.cut(
        df['sds_z'],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=['low_sds', 'buffer_low', 'middle', 'buffer_high', 'high_sds']
    )
    return df[df['sds_group'].isin(SDS_GROUPS)][['Subject', 'sds_group']].copy()


def create_bootstrap_samples(df, n_bs_samples=1000, n_samples=90):
    out = {}
    for group in df['sds_group'].unique():
        subjects = df[df['sds_group'] == group]['Subject'].values
        samples = [np.random.choice(subjects, size=min(n_samples, len(subjects)),
                                    replace=True) for _ in range(n_bs_samples)]
        out[group] = np.array(samples)
    return out


def create_connectivity_matrices(sds_df, mats_df_full, ids, partition_df, n_nodes,
                                  project_dir, n_bootstrap, sample_size, report):
    report.append("\n" + "=" * 80)
    report.append("CONNECTIVITY MATRIX VISUALISATION (per SDS group)")
    report.append("=" * 80)

    bootstrap_samples = create_bootstrap_samples(sds_df, n_bootstrap, sample_size)
    figures_dir = project_dir / 'figures'
    data_dir = project_dir / 'data'

    for group in SDS_GROUPS:
        report.append(f"\nProcessing group: {SDS_GROUP_LABELS[group]}")
        print(f'Processing group: {SDS_GROUP_LABELS[group]}')

        if group not in bootstrap_samples:
            report.append(f"  No data for group {group}, skipping")
            continue

        group_samples = bootstrap_samples[group]
        bootstrap_matrices = []
        for i, sample in enumerate(group_samples):
            if i % 200 == 0:
                print(f'  Processing bootstrap sample {i+1}/{len(group_samples)}')
            sample_indices = []
            for subject in sample:
                idx = np.where(ids == subject)[0]
                if len(idx) > 0:
                    sample_indices.append(idx[0])
            if len(sample_indices) == 0:
                continue
            sample_mats = mats_df_full.iloc[sample_indices, :]
            avg_mat = sample_mats.mean(axis=0).values / 100
            corr = vec_to_sym_matrix(avg_mat, diagonal=np.zeros(n_nodes))
            bootstrap_matrices.append(corr)

        if not bootstrap_matrices:
            report.append(f"  No valid bootstrap samples for {group}, skipping")
            continue

        bootstrap_matrices = np.array(bootstrap_matrices)
        final = np.mean(bootstrap_matrices, axis=0)

        report.append(f'  Computed average from {len(bootstrap_matrices)} bootstrap samples')

        matrix_path = data_dir / f'C5_avg_connectivity_{group}_bootstrap.npy'
        np.save(matrix_path, final)
        report.append(f'  Saved matrix: {matrix_path.name}')

        community_order = np.argsort(partition_df['community_id'].values)
        reordered = final[community_order][:, community_order]

        fig = plt.figure(figsize=(55*mm2inches, 55*mm2inches))
        plt.imshow(reordered, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        community_sizes = partition_df['community_id'].value_counts().sort_index()
        boundaries = np.cumsum([0] + list(community_sizes))[:-1]
        for boundary in boundaries[1:]:
            plt.axhline(boundary - 0.5, color='black', linewidth=0.5)
            plt.axvline(boundary - 0.5, color='black', linewidth=0.5)
        plt.gca().set_xticks([]); plt.gca().set_yticks([])
        out_fig = figures_dir / f'C5_Connectivity_Matrix_{n_nodes}Nodes_{group}_bootstrap.png'
        plt.savefig(out_fig, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f'  Saved matrix figure: {out_fig.name}')

        # Spring-layout network
        node_colours = partition_df['community_id'].values
        node_colours = np.array([NODE_COLOUR_DICT.get(c, '#888888') for c in node_colours])
        thresh = bct.threshold_proportional(final.copy(), 0.05, copy=False)
        G = nx.from_numpy_array(thresh)
        pos = nx.spring_layout(G, iterations=100, seed=42)
        largest_cc = max(nx.connected_components(G), key=len)
        G_filt = G.subgraph(largest_cc)
        node_colours_filt = node_colours[list(largest_cc)]
        node_sizes = [2 * G_filt.degree(n) for n in G_filt.nodes()]
        fig = plt.figure(figsize=(50*mm2inches, 50*mm2inches))
        nx.draw(G_filt, pos, node_size=node_sizes,
                node_color=node_colours_filt, with_labels=False,
                edge_color='gray', edgecolors='black')
        out_fig = figures_dir / f'C5_Network_Visualisation_{n_nodes}Nodes_{group}_bootstrap.png'
        plt.savefig(out_fig, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f'  Saved network figure: {out_fig.name}')


def create_exemplar_networks(sds_df, mats_df_full, ids, partition_df, n_nodes,
                              project_dir, report,
                              n_exemplars=4, n_quartiles=4,
                              modularity_threshold=0.2, viz_threshold=0.05,
                              seed=42):
    report.append("\n" + "=" * 80)
    report.append("EXEMPLAR NETWORK VISUALISATIONS (SDS group x modularity quartile)")
    report.append("=" * 80)

    figures_dir = project_dir / 'figures'
    results_dir = project_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    node_colours = partition_df['community_id'].values
    node_colours = np.array([NODE_COLOUR_DICT.get(c, '#888888') for c in node_colours])
    community_ids = partition_df['community_id'].values

    mod_records = []
    for subject in sds_df['Subject'].values:
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

    mod_df = pd.DataFrame(mod_records).merge(sds_df, on='Subject')
    report.append(f"\n  Modularity computed for {len(mod_df)} subjects "
                  f"at threshold {modularity_threshold}")

    index_rows = []
    for g_idx, group in enumerate(SDS_GROUPS):
        group_df = mod_df[mod_df['sds_group'] == group].copy()
        if len(group_df) < n_quartiles:
            report.append(f"\n  {group}: only {len(group_df)} subjects, skipping")
            continue

        group_df['mod_quartile'] = pd.qcut(
            group_df['modularity'], q=n_quartiles, labels=False, duplicates='drop'
        )
        group_df = group_df.sort_values('Subject').reset_index(drop=True)
        report.append(f"\n  Group: {SDS_GROUP_LABELS[group]} (n={len(group_df)})")
        print(f'Creating exemplars for group: {SDS_GROUP_LABELS[group]}')

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
                thresholded = bct.threshold_proportional(subj_mat.copy(),
                                                         viz_threshold, copy=False)
                G = nx.from_numpy_array(thresholded)
                pos = nx.spring_layout(G, iterations=100, seed=42)
                largest_cc = max(nx.connected_components(G), key=len)
                G_filt = G.subgraph(largest_cc)
                node_colours_filt = node_colours[list(largest_cc)]
                node_sizes = [2 * G_filt.degree(n) for n in G_filt.nodes()]
                fig = plt.figure(figsize=(50*mm2inches, 50*mm2inches))
                nx.draw(G_filt, pos, node_size=node_sizes,
                        node_color=node_colours_filt, with_labels=False,
                        edge_color='gray', edgecolors='black')
                fname = (f'C5_Exemplar_Network_{n_nodes}Nodes_{group}_'
                         f'modQ{q+1}_e{i+1}_subj{subject}.png')
                fig_path = figures_dir / fname
                plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                index_rows.append({
                    'sds_group': group, 'mod_quartile': q + 1,
                    'exemplar_idx': i + 1, 'Subject': subject,
                    'modularity': subj_row['modularity'], 'filename': fname,
                })

    index_df = pd.DataFrame(index_rows)
    index_path = results_dir / 'C5_exemplar_subjects.csv'
    index_df.to_csv(index_path, index=False)
    report.append(f"\n  Saved exemplar index: {index_path.name} ({len(index_df)} rows)")


def bootstrap_density_boxplot(data, ax, position=0, width=0.4, color='lightblue',
                               label=None, n_bootstrap=1000, sample_size=90):
    bootstrap_samples = [
        np.mean(np.random.choice(data, size=min(sample_size, len(data)), replace=True))
        for _ in range(n_bootstrap)
    ]
    bootstrap_samples = np.array(bootstrap_samples)
    density = stats.gaussian_kde(bootstrap_samples)
    xs = np.linspace(bootstrap_samples.min(), bootstrap_samples.max(), 200)
    density_curve = density(xs) / density(xs).max() * width
    ax.fill_betweenx(xs, position - density_curve, position,
                     alpha=0.6, color=color, label=label)
    ax.boxplot([bootstrap_samples], positions=[position + width/2],
               widths=width/4, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color='black', linewidth=2),
               showfliers=False)


def create_density_plots(graph_metrics_df, project_dir, n_bootstrap, sample_size, report):
    report.append("\n" + "=" * 80)
    report.append("BOOTSTRAP DENSITY PLOTS")
    report.append("=" * 80)
    figures_dir = project_dir / 'figures'

    for measure, measure_label in zip(['modularity', 'global_efficiency'],
                                       ['Modularity (residualised)',
                                        'Global Efficiency (residualised)']):
        report.append(f"\nCreating density plot for {measure_label}...")
        fig, ax = plt.subplots(figsize=(60*mm2inches, 45*mm2inches))

        data_low = graph_metrics_df.loc[
            graph_metrics_df['sds_group'] == 'low_sds', measure].values
        data_med = graph_metrics_df.loc[
            graph_metrics_df['sds_group'] == 'middle', measure].values
        data_high = graph_metrics_df.loc[
            graph_metrics_df['sds_group'] == 'high_sds', measure].values

        report.append(f"  Group sizes: Low={len(data_low)}, "
                      f"Middle={len(data_med)}, High={len(data_high)}")

        all_data = np.concatenate([data_low, data_med, data_high])
        mean_all = np.mean(all_data)
        std_all = np.std(all_data, ddof=1)
        data_low = (data_low - mean_all) / std_all
        data_med = (data_med - mean_all) / std_all
        data_high = (data_high - mean_all) / std_all

        positions = [0, 1.2, 2.4]
        for data, pos, group in zip(
            [data_low, data_med, data_high], positions, SDS_GROUPS
        ):
            bootstrap_density_boxplot(data, ax, position=pos, width=0.4,
                                       color=SDS_DENSITY_COLOURS[group],
                                       label=SDS_GROUP_LABELS[group],
                                       n_bootstrap=n_bootstrap,
                                       sample_size=sample_size)

        ax.set_xlim(-0.5, 3.2)
        ax.set_xticks(positions)
        ax.set_xticklabels([SDS_GROUP_LABELS[g] for g in SDS_GROUPS], fontsize=8)
        ax.set_ylabel(f'{measure_label} [$z$]', fontsize=8)
        ax.set_xlabel('SDS Group', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
        sns.despine(offset=8, trim=True)
        plt.tight_layout()
        fig_path = figures_dir / f'C5_Bootstrap_Density_Plot_{measure}.png'
        plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        report.append(f"  Saved: {fig_path.name}")


def bootstrap_ellipse_parameters(data_mod, data_eff,
                                  n_bootstrap_iterations=1000, sample_size=90):
    params = []
    for _ in range(n_bootstrap_iterations):
        idx = np.random.choice(len(data_mod), size=min(sample_size, len(data_mod)),
                               replace=True)
        b_mod = data_mod[idx]; b_eff = data_eff[idx]
        if len(b_mod) < 4:
            continue
        cov = np.cov(np.column_stack([b_mod, b_eff]).T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * 1.96 * np.sqrt(eigenvals[0])
        height = 2 * 1.96 * np.sqrt(eigenvals[1])
        params.append({
            'center': (np.mean(b_mod), np.mean(b_eff)),
            'width': width, 'height': height, 'angle': angle,
            'area': np.pi * width * height / 4,
            'eccentricity': np.sqrt(1 - (min(eigenvals) / max(eigenvals)))
        })
    return params


def create_ellipse_extent_plot(graph_metrics_df, project_dir, n_bootstrap,
                                sample_size, report):
    report.append("\n" + "=" * 80)
    report.append("BOOTSTRAP ELLIPSE EXTENT PLOT")
    report.append("=" * 80)

    figures_dir = project_dir / 'figures'
    fig, ax = plt.subplots(1, 1, figsize=(60*mm2inches, 45*mm2inches))

    bootstrap_results = {}
    for group in SDS_GROUPS:
        report.append(f"\nBootstrapping ellipse extent for {SDS_GROUP_LABELS[group]}...")
        print(f"Bootstrapping ellipse for {SDS_GROUP_LABELS[group]}...")
        gd = graph_metrics_df[graph_metrics_df['sds_group'] == group]
        mod = gd['modularity'].values
        eff = gd['global_efficiency'].values
        if len(mod) == 0:
            report.append(f"  No data for {group}, skipping")
            continue
        report.append(f"  N subjects: {len(mod)}")
        params = bootstrap_ellipse_parameters(mod, eff, n_bootstrap, sample_size)
        if not params:
            report.append(f"  No valid ellipses for {group}, skipping")
            continue
        bootstrap_results[group] = params

        widths = [p['width'] for p in params]
        heights = [p['height'] for p in params]
        centers_x = [p['center'][0] for p in params]
        centers_y = [p['center'][1] for p in params]
        angles = [p['angle'] for p in params]
        areas = [p['area'] for p in params]

        wlo, wmd, whi = np.percentile(widths, [2.5, 50, 97.5])
        hlo, hmd, hhi = np.percentile(heights, [2.5, 50, 97.5])
        cx_md = np.median(centers_x); cy_md = np.median(centers_y)
        a_md = np.median(angles); area_md = np.median(areas)

        report.append(f"  Median ellipse area: {area_md:.4f}")
        report.append(f"  Width  (95% CI): {wmd:.4f} [{wlo:.4f}, {whi:.4f}]")
        report.append(f"  Height (95% CI): {hmd:.4f} [{hlo:.4f}, {hhi:.4f}]")

        ellipse_outer = Ellipse(
            xy=(cx_md, cy_md), width=whi, height=hhi, angle=a_md,
            facecolor='none', edgecolor=SDS_ELLIPSE_COLOURS[group],
            linewidth=2, linestyle='-', alpha=0.8
        )
        ax.add_patch(ellipse_outer)
        ax.scatter(np.mean(centers_x), np.mean(centers_y), marker='o', s=100,
                   alpha=0.5, color=SDS_ELLIPSE_COLOURS[group], edgecolors='black',
                   linewidth=0.5, zorder=10, label=SDS_GROUP_LABELS[group])

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
    args = parse_args()
    project_dir = Path(args.project)
    n_nodes = len(pd.read_csv(args.partition))

    report = [
        "=" * 80,
        "C5: NETWORK VISUALISATION REPORT (SDS-stratified)",
        "=" * 80,
        f"\nProject directory: {project_dir}",
        f"Number of nodes (from C2b partition): {n_nodes}",
        f"Number of bootstrap samples: {args.n_bootstrap}",
        f"Bootstrap sample size: {args.sample_size}",
    ]

    print("=" * 80)
    print("C5: NETWORK VISUALISATION (SDS-stratified)")
    print("=" * 80)

    figures_dir = project_dir / 'figures'
    reports_dir = project_dir / 'reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report.append("\n" + "=" * 80)
    report.append("LOADING DATA")
    report.append("=" * 80)

    social_df = pd.read_csv(args.social)
    sds_df = assign_sds_groups(social_df)
    report.append(f"Social scores loaded: {len(social_df)} subjects")
    report.append("SDS group sizes (after dropping buffer rows):")
    for group in SDS_GROUPS:
        n = (sds_df['sds_group'] == group).sum()
        report.append(f"  {SDS_GROUP_LABELS[group]}: {n}")

    partition_df = pd.read_csv(args.partition)
    partition_df['community_id'] = partition_df['community_id'].astype(int)
    report.append(f"Partition data: {len(partition_df)} nodes")

    ids = pd.read_csv(args.ids, header=None).values.flatten()
    report.append(f"Subject IDs: {len(ids)}")

    matrices_dir = Path(args.matrices_dir)
    matrix_file = matrices_dir / f'3T_HCP1200_MSMAll_d{n_nodes}_ts2/netmats1.txt'
    mats_df_full = pd.read_csv(matrix_file, header=None, sep=r'\s+')
    lower_indices = np.tril_indices(n_nodes, k=-1)
    linear_indices = lower_indices[0] * n_nodes + lower_indices[1]
    mats_df_full = mats_df_full.iloc[:, linear_indices]
    report.append(f"Connectivity matrices loaded: {mats_df_full.shape}")

    graph_metrics_df = pd.read_csv(args.graph_metrics)
    report.append(f"Graph metrics data: {len(graph_metrics_df)} subjects")
    if 'sds_group' not in graph_metrics_df.columns:
        raise ValueError(
            f"{args.graph_metrics} does not contain 'sds_group' column. "
            "Re-run C3 to produce SDS-stratified graph metrics first."
        )

    create_connectivity_matrices(sds_df, mats_df_full, ids, partition_df, n_nodes,
                                  project_dir, args.n_bootstrap, args.sample_size, report)
    create_exemplar_networks(sds_df, mats_df_full, ids, partition_df,
                             n_nodes, project_dir, report)
    create_density_plots(graph_metrics_df, project_dir, args.n_bootstrap,
                         args.sample_size, report)
    create_ellipse_extent_plot(graph_metrics_df, project_dir,
                                args.n_bootstrap, args.sample_size, report)

    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append("\nGenerated figures:")
    report.append("  - C5_Connectivity_Matrix_*Nodes_*_bootstrap.png (per SDS group)")
    report.append("  - C5_Network_Visualisation_*Nodes_*_bootstrap.png (per SDS group)")
    report.append("  - C5_Exemplar_Network_*Nodes_*_modQ*_e*_subj*.png "
                  "(4 exemplars x 4 quartiles x 3 SDS groups = 48)")
    report.append("  - C5_exemplar_subjects.csv")
    report.append("  - C5_Bootstrap_Density_Plot_modularity.png")
    report.append("  - C5_Bootstrap_Density_Plot_global_efficiency.png")
    report.append("  - C5_Bootstrap_Ellipse_Extent_Plot.png")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_path = reports_dir / 'C5_visualize_networks_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\nReport saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}/C5_*.png")


if __name__ == "__main__":
    main()
