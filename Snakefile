"""
Snakemake workflow for Brain Compensation project
Analysis pipeline for integrating phenotypic, genetic (PGS), and fMRI data
"""

configfile: "config.yaml"

# Define project paths
PROJECT_DIR = config["project_dir"]
DATA_DIR = f"{PROJECT_DIR}/data"
CODE_DIR = f"{PROJECT_DIR}/code"
GENETICS_INPUT_DIR = f"{DATA_DIR}/raw_anonymised"          # Original genetics data, hg19 (read-only)
PLINK_DIR = f"{DATA_DIR}/PLINK_anonymised"                # PLINK working directory (outputs)
QCDIR = f"{DATA_DIR}/plinkQC_output"                      # B1 plinkQC output directory
RESULTS_DIR = f"{PROJECT_DIR}/results"
LOGS_DIR = f"{PROJECT_DIR}/logs"
GCTA_PATH = config.get("gcta_path", "/opt/gcta")

# PLINK file prefix (must match defaults in B1 R script)
GENETICS_NAME = "Neuro_Chip_anonymised"
GENETICS_BUILD = "hg19"

# Final target outputs
rule all:
    input:
        # Phase A: Phenotypic data preprocessing
        f"{RESULTS_DIR}/behavioural_data_preprocessed.csv",
        f"{PROJECT_DIR}/reports/A1_preprocess_phenotypic_data_report.txt",
        f"{PROJECT_DIR}/figures/A1_Behaviour_correlations.png",
        f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        f"{PROJECT_DIR}/reports/A2_factor_analysis_report.txt",
        f"{PROJECT_DIR}/figures/A2_factor_loadings.png",
        f"{PROJECT_DIR}/figures/A2_factor_scores_distribution.png",
        f"{PROJECT_DIR}/figures/A3_social_factor_evaluation.png",
        f"{PROJECT_DIR}/reports/A3_evaluate_social_factor_report.txt",

        # Phase B: Genetic/PGS analysis (hg19 throughout)
        f"{QCDIR}/{GENETICS_NAME}.clean.bed",
        f"{PROJECT_DIR}/reports/B1_plinkQC_genotype_qc_report.txt",
        f"{PLINK_DIR}/B1b_within_sample_pca.eigenvec",
        f"{PROJECT_DIR}/reports/B1b_ancestry_PCA_mahalanobis_report.txt",
        f"{PROJECT_DIR}/figures/B1b_PCA_scatter.png",
        f"{PROJECT_DIR}/figures/B1b_mahalanobis_distribution.png",
        f"{PROJECT_DIR}/figures/B1b_within_sample_scree.png",
        f"{PLINK_DIR}/full_pgs_scores.snp.blp.profile",
        f"{PROJECT_DIR}/figures/B3_pgs_threshold_evaluation.png",
        f"{PROJECT_DIR}/figures/B5_blup_evaluation.png",
        f"{RESULTS_DIR}/pgs_residuals.csv",

        # Phase C: fMRI analysis
        f"{PROJECT_DIR}/reports/C1_run_univariate_fMRI_prediction_report.txt",
        f"{PROJECT_DIR}/reports/C2_find_communities_fMRI_report.txt",
        f"{PROJECT_DIR}/reports/C2b_evaluate_parcellations_report.txt",
        f"{RESULTS_DIR}/C2b_parcellation_evaluation.csv",
        f"{RESULTS_DIR}/C2b_selected_partition.csv",
        f"{PROJECT_DIR}/figures/C2b_parcellation_tuning_curves.png",
        f"{PROJECT_DIR}/reports/C3_perform_main_landscape_analysis_report.txt",
        f"{RESULTS_DIR}/C3_graph_theory_landscape_results.csv",
        f"{RESULTS_DIR}/C4_main_network_metrics.csv",
        f"{PROJECT_DIR}/reports/C3b_continuous_heteroscedasticity_report.txt",
        f"{RESULTS_DIR}/C3b_heteroscedasticity_results.csv",
        f"{RESULTS_DIR}/C4_sensitivity_summary.csv",
        f"{PROJECT_DIR}/reports/C5_visualize_networks_report.txt",
        f"{PROJECT_DIR}/figures/C5_Bootstrap_Ellipse_Extent_Plot.png",

        # Data quality check
        f"{RESULTS_DIR}/DataRetention_Overview.csv",

        # Publication figures
        f"{PROJECT_DIR}/figures/publication/fig_pgs_social_scatter.svg"


# ============================================================================
# Phase A: Phenotypic Data Preprocessing
# ============================================================================

rule preprocess_phenotypic:
    """Preprocess HCP behavioural and phenotypic data"""
    input:
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}"
    output:
        data=f"{RESULTS_DIR}/behavioural_data_preprocessed.csv",
        report=f"{PROJECT_DIR}/reports/A1_preprocess_phenotypic_data_report.txt",
        figure=f"{PROJECT_DIR}/figures/A1_Behaviour_correlations.png"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/A1_preprocess_phenotypic.log"
    shell:
        """
        python {CODE_DIR}/A1_preprocess_phenotypic_data.py \
            --behavioural {input.behavioural} \
            --phenotypic {input.phenotypic} \
            --output {output.data} \
            --project {PROJECT_DIR} \
            --figure {output.figure} > {log} 2>&1
        """


rule factor_analysis:
    """Perform confirmatory factor analysis on behavioural data"""
    input:
        f"{RESULTS_DIR}/behavioural_data_preprocessed.csv"
    output:
        data=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        report=f"{PROJECT_DIR}/reports/A2_factor_analysis_report.txt",
        fig1=f"{PROJECT_DIR}/figures/A2_factor_loadings.png",
        fig2=f"{PROJECT_DIR}/figures/A2_factor_scores_distribution.png"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/A2_factor_analysis.log"
    shell:
        """
        Rscript {CODE_DIR}/A2_factor_analysis.R \
            --input {input} \
            --output {output.data} \
            --project {PROJECT_DIR} > {log} 2>&1
        """


rule evaluate_social_factor:
    """Evaluate social factor from CFA results"""
    input:
        factor=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}"
    output:
        fig=f"{PROJECT_DIR}/figures/A3_social_factor_evaluation.png",
        report=f"{PROJECT_DIR}/reports/A3_evaluate_social_factor_report.txt"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/A3_evaluate_social_factor.log"
    shell:
        """
        python {CODE_DIR}/A3_evaluate_social_factor.py \
            --factor {input.factor} \
            --behavioural {input.behavioural} \
            --output {output.fig} \
            --project {PROJECT_DIR} > {log} 2>&1
        """


# ============================================================================
# Phase B: Genetic/PGS Analysis
# ============================================================================

rule plinkqc_genotype_qc:
    """Genotype quality control using plinkQC on hg19 NeuroChip data (B1).

    Sex filtering is via FAM PEDSEX vs phenotype Gender cross-check (chrX
    F-statistic disabled — see B1 docstring). Ancestry filtering is
    delegated to B1b. cleanData() therefore drops only on heterozygosity,
    sample missingness, and per-marker filters; the sex cross-check is
    applied as a post-cleanData PLINK --remove pass inside B1."""
    input:
        bed=f"{GENETICS_INPUT_DIR}/{GENETICS_NAME}.bed",
        bim=f"{GENETICS_INPUT_DIR}/{GENETICS_NAME}.bim",
        fam=f"{GENETICS_INPUT_DIR}/{GENETICS_NAME}.fam"
    output:
        clean_bed=f"{QCDIR}/{GENETICS_NAME}.clean.bed",
        clean_bim=f"{QCDIR}/{GENETICS_NAME}.clean.bim",
        clean_fam=f"{QCDIR}/{GENETICS_NAME}.clean.fam",
        report=f"{PROJECT_DIR}/reports/B1_plinkQC_genotype_qc_report.txt"
    params:
        sex_filter_flag=lambda wildcards: "" if config.get("apply_sex_filter", True) else "--no-apply-sex-filter"
    log:
        f"{LOGS_DIR}/B1_plinkqc_genotype_qc.log"
    shell:
        """
        Rscript {CODE_DIR}/B1_plinkQC_genotype_qc.R \
            --project {PROJECT_DIR} \
            --indir data/raw_anonymised \
            --name {GENETICS_NAME} \
            --genomebuild {GENETICS_BUILD} \
            {params.sex_filter_flag} > {log} 2>&1
        """


rule ancestry_pca_mahalanobis:
    """Within-sample ancestry PCA + 1KG-projected diagnostic (B1b).

    Two PCAs run side-by-side. (1) Within-HCP-sample PCA on the B1 .clean
    triplet — its eigenvec is the canonical PC source consumed by C1/C3/C3b
    as ancestry nuisance covariates. (2) 1KG-merged reference PCA + projection
    + Mahalanobis distance to a CEU+GBR+IBS+TSI centroid — diagnostic only,
    no participants are dropped. The hg19 1KG reference must be pre-staged
    at data/reference/1000Genomes/phase3_hg19/ (no baked-in download URLs);
    see B1b docstring."""
    input:
        clean_bed=f"{QCDIR}/{GENETICS_NAME}.clean.bed",
        clean_bim=f"{QCDIR}/{GENETICS_NAME}.clean.bim",
        clean_fam=f"{QCDIR}/{GENETICS_NAME}.clean.fam"
    output:
        within_eigenvec=f"{PLINK_DIR}/B1b_within_sample_pca.eigenvec",
        within_eigenval=f"{PLINK_DIR}/B1b_within_sample_pca.eigenval",
        per_sample=f"{PLINK_DIR}/B1b_per_sample_distance.csv",
        report=f"{PROJECT_DIR}/reports/B1b_ancestry_PCA_mahalanobis_report.txt",
        scatter=f"{PROJECT_DIR}/figures/B1b_PCA_scatter.png",
        dist=f"{PROJECT_DIR}/figures/B1b_mahalanobis_distribution.png",
        scree=f"{PROJECT_DIR}/figures/B1b_within_sample_scree.png"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/B1b_ancestry_PCA_mahalanobis.log"
    shell:
        """
        python {CODE_DIR}/B1b_ancestry_PCA_mahalanobis.py \
            --project {PROJECT_DIR} \
            --build {GENETICS_BUILD} \
            --qc-prefix {GENETICS_NAME}.clean > {log} 2>&1
        """


rule translate_pgs_to_hcp:
    """SNP harmonization, PCA, relatedness filtering, and PGS calculation (B2).

    Consumes B1's .clean triplet directly — B1b no longer filters by
    ancestry; ancestry information enters the pipeline as PC covariates in
    C1/C3/C3b."""
    input:
        clean_bed=f"{QCDIR}/{GENETICS_NAME}.clean.bed",
        gwas=f"{GENETICS_INPUT_DIR}/iPSYCH_PGC_ASD_Nov_2017.gz",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}"
    output:
        pgs_scores=f"{PLINK_DIR}/hcp_pgs_scores.profile",
        pca=f"{PLINK_DIR}/Neuro_Chip_full_sample_pca.eigenvec",
        qc_bed=f"{PLINK_DIR}/Neuro_Chip_qc_nodup_sexfiltered.bed",
        unrelated_pgs=f"{PLINK_DIR}/unrelated_pgs_scores.txt"
    log:
        f"{LOGS_DIR}/B2_translate_pgs.log"
    shell:
        """
        bash {CODE_DIR}/B2_translate_PGS_to_HCP.sh \
            --original-data {GENETICS_INPUT_DIR} \
            --plink-dir {PLINK_DIR} \
            --data-dir {DATA_DIR} \
            --code-dir {CODE_DIR} \
            --phenotypic {input.phenotypic} \
            --qcdir {QCDIR} \
            --clean-name {GENETICS_NAME}.clean \
            --output {output.pgs_scores} > {log} 2>&1
        """


rule select_pgs_threshold:
    """Select optimal PGS threshold based on prediction performance"""
    input:
        pgs=f"{PLINK_DIR}/hcp_pgs_scores.profile",
        phenotype=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        pca=f"{PLINK_DIR}/Neuro_Chip_full_sample_pca.eigenvec"
    output:
        plot=f"{PROJECT_DIR}/figures/B3_pgs_threshold_evaluation.png",
        selected=f"{RESULTS_DIR}/pgs_selected_threshold.txt"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/B3_select_pgs_threshold.log"
    shell:
        """
        python {CODE_DIR}/B3_select_PGS_threshold.py \
            --pgs {input.pgs} \
            --phenotype {input.phenotype} \
            --pca {input.pca} \
            --output-plot {output.plot} \
            --output-threshold {output.selected} \
            --project {PROJECT_DIR} > {log} 2>&1
        """


rule extend_pgs_with_blup:
    """Extend PGS with BLUP predictions"""
    input:
        pgs=f"{PLINK_DIR}/hcp_pgs_scores.profile",
        threshold=f"{RESULTS_DIR}/pgs_selected_threshold.txt",
        bfile=f"{PLINK_DIR}/Neuro_Chip_qc_nodup_sexfiltered.bed"
    output:
        f"{PLINK_DIR}/full_pgs_scores.snp.blp.profile"
    log:
        f"{LOGS_DIR}/B4_extend_pgs_blup.log"
    shell:
        """
        bash {CODE_DIR}/B4_extend_PGS_with_BLUP.sh \
            --plink-dir {PLINK_DIR} \
            --pgs-file {input.pgs} \
            --threshold-file {input.threshold} \
            --gcta-path {GCTA_PATH} \
            --output {output} > {log} 2>&1
        """


rule evaluate_blup:
    """Evaluate BLUP prediction accuracy"""
    input:
        blup_pgs=f"{PLINK_DIR}/full_pgs_scores.snp.blp.profile",
        original_pgs=f"{PLINK_DIR}/unrelated_pgs_scores.txt",
        social_scores=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        pca=f"{PLINK_DIR}/Neuro_Chip_full_sample_pca.eigenvec"
    output:
        plot=f"{PROJECT_DIR}/figures/B5_blup_evaluation.png",
        residuals=f"{RESULTS_DIR}/pgs_residuals.csv"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/B5_evaluate_blup.log"
    shell:
        """
        python {CODE_DIR}/B5_evalute_BLUP_prediction.py \
            --blup-pgs {input.blup_pgs} \
            --original-pgs {input.original_pgs} \
            --social-scores {input.social_scores} \
            --phenotypic {input.phenotypic} \
            --behavioural {input.behavioural} \
            --pca {input.pca} \
            --output-residuals {output.residuals} \
            --output-plot {output.plot} \
            --project {PROJECT_DIR} > {log} 2>&1
        """


# ============================================================================
# Phase C: fMRI Analysis
# ============================================================================

rule univariate_fmri_prediction:
    """Run univariate fMRI prediction analysis"""
    input:
        pgs=f"{RESULTS_DIR}/pgs_residuals.csv",
        social=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}",
        movement=lambda wildcards: f"{DATA_DIR}/{config['input_movement']}",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}",
        ancestry_pcs=f"{PLINK_DIR}/B1b_within_sample_pca.eigenvec"
    output:
        report=f"{PROJECT_DIR}/reports/C1_run_univariate_fMRI_prediction_report.txt"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        motion_threshold=config.get("motion_threshold", 0.2),
        parcellations=config.get("parcellations", "50 100 200"),
        n_ancestry_pcs=config.get("n_ancestry_pcs", 5)
    conda:
        "environment.yml"
    threads: config.get("threads", 4)
    resources:
        mem_mb=config.get("mem_mb", 8000)
    log:
        f"{LOGS_DIR}/C1_univariate_fmri.log"
    shell:
        """
        xvfb-run -a python {CODE_DIR}/C1_run_univariate_fMRI_prediction.py \
            --project {PROJECT_DIR} \
            --social {input.social} \
            --pgs {input.pgs} \
            --behavioural {input.behavioural} \
            --phenotypic {input.phenotypic} \
            --movement {input.movement} \
            --ids {input.ids} \
            --ancestry-pcs {input.ancestry_pcs} \
            --n-ancestry-pcs {params.n_ancestry_pcs} \
            --matrices-dir {params.matrices_dir} \
            --motion-threshold {params.motion_threshold} \
            --parcellations {params.parcellations} > {log} 2>&1
        """


rule find_fmri_communities:
    """Identify network communities in fMRI data"""
    input:
        report=f"{PROJECT_DIR}/reports/C1_run_univariate_fMRI_prediction_report.txt",
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}"
    output:
        report=f"{PROJECT_DIR}/reports/C2_find_communities_fMRI_report.txt",
        partition_15=f"{RESULTS_DIR}/C2_final_partition_15Nodes.csv",
        partition_25=f"{RESULTS_DIR}/C2_final_partition_25Nodes.csv",
        partition_50=f"{RESULTS_DIR}/C2_final_partition_50Nodes.csv",
        partition_100=f"{RESULTS_DIR}/C2_final_partition_100Nodes.csv",
        partition_200=f"{RESULTS_DIR}/C2_final_partition_200Nodes.csv",
        partition_300=f"{RESULTS_DIR}/C2_final_partition_300Nodes.csv"
    params:
        parcellations="15 25 50 100 200 300",
        n_iterations=config.get("n_iterations", 50),
        target_communities=config.get("target_communities", "5 15")
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/C2_find_communities.log"
    shell:
        """
        xvfb-run -a python {CODE_DIR}/C2_find_communities_fMRI.py \
            --project {PROJECT_DIR} \
            --matrices-dir {input.matrices_dir} \
            --ids {input.ids} \
            --parcellations {params.parcellations} \
            --n-iterations {params.n_iterations} \
            --target-communities {params.target_communities} > {log} 2>&1
        """


rule evaluate_parcellations:
    """Evaluate community detection across all HCP ICA parcellation sizes
    and select the most consistent resolution for downstream landscape
    analysis. The selected partition is written to a stable filename so
    downstream rules do not need to know the chosen size in advance."""
    input:
        report=f"{PROJECT_DIR}/reports/C2_find_communities_fMRI_report.txt",
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}"
    output:
        report=f"{PROJECT_DIR}/reports/C2b_evaluate_parcellations_report.txt",
        results=f"{RESULTS_DIR}/C2b_parcellation_evaluation.csv",
        selected_partition=f"{RESULTS_DIR}/C2b_selected_partition.csv",
        figure=f"{PROJECT_DIR}/figures/C2b_parcellation_tuning_curves.png"
    params:
        parcellations="15 25 50 100 200 300",
        target_communities=config.get("target_communities", "5 8")
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/C2b_evaluate_parcellations.log"
    shell:
        """
        python {CODE_DIR}/C2b_evaluate_communities.py \
            --project {PROJECT_DIR} \
            --matrices-dir {input.matrices_dir} \
            --parcellations {params.parcellations} \
            --target-communities {params.target_communities} > {log} 2>&1
        """


rule main_landscape_analysis:
    """Main landscape analysis at the C2b-selected parcellation, with
    threshold sensitivity."""
    input:
        pgs=f"{RESULTS_DIR}/pgs_residuals.csv",
        social=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}",
        movement=lambda wildcards: f"{DATA_DIR}/{config['input_movement']}",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}",
        partition=f"{RESULTS_DIR}/C2b_selected_partition.csv",
        ancestry_pcs=f"{PLINK_DIR}/B1b_within_sample_pca.eigenvec"
    output:
        report=f"{PROJECT_DIR}/reports/C3_perform_main_landscape_analysis_report.txt",
        results=f"{RESULTS_DIR}/C3_graph_theory_landscape_results.csv",
        summary=f"{RESULTS_DIR}/C4_sensitivity_summary.csv",
        main_metrics=f"{RESULTS_DIR}/C4_main_network_metrics.csv"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        motion_threshold=config.get("motion_threshold", 0.2),
        thresholds="0.15 0.20 0.25",
        n_ancestry_pcs=config.get("n_ancestry_pcs", 5)
    conda:
        "environment.yml"
    threads: config.get("threads", 4)
    resources:
        mem_mb=config.get("mem_mb", 16000)
    log:
        f"{LOGS_DIR}/C3_main_landscape.log"
    shell:
        """
        python {CODE_DIR}/C3_perform_main_landscape_analysis.py \
            --project {PROJECT_DIR} \
            --pgs {input.pgs} \
            --social {input.social} \
            --behavioural {input.behavioural} \
            --phenotypic {input.phenotypic} \
            --movement {input.movement} \
            --ids {input.ids} \
            --ancestry-pcs {input.ancestry_pcs} \
            --n-ancestry-pcs {params.n_ancestry_pcs} \
            --matrices-dir {params.matrices_dir} \
            --partition {input.partition} \
            --thresholds {params.thresholds} \
            --motion-threshold {params.motion_threshold} > {log} 2>&1
        """


rule continuous_heteroscedasticity_analysis:
    """Continuous heteroscedasticity analysis at the C2b-selected parcellation."""
    input:
        pgs=f"{RESULTS_DIR}/pgs_residuals.csv",
        social=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}",
        movement=lambda wildcards: f"{DATA_DIR}/{config['input_movement']}",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}",
        partition=f"{RESULTS_DIR}/C2b_selected_partition.csv",
        main_report=f"{PROJECT_DIR}/reports/C3_perform_main_landscape_analysis_report.txt",
        ancestry_pcs=f"{PLINK_DIR}/B1b_within_sample_pca.eigenvec"
    output:
        report=f"{PROJECT_DIR}/reports/C3b_continuous_heteroscedasticity_report.txt",
        results=f"{RESULTS_DIR}/C3b_heteroscedasticity_results.csv"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        motion_threshold=config.get("motion_threshold", 0.2),
        threshold=0.2,
        n_ancestry_pcs=config.get("n_ancestry_pcs", 5)
    conda:
        "environment.yml"
    threads: config.get("threads", 4)
    resources:
        mem_mb=config.get("mem_mb", 16000)
    log:
        f"{LOGS_DIR}/C3b_continuous_heteroscedasticity.log"
    shell:
        """
        python {CODE_DIR}/C3b_continuous_heteroscedasticity_analysis.py \
            --project {PROJECT_DIR} \
            --pgs {input.pgs} \
            --social {input.social} \
            --behavioural {input.behavioural} \
            --phenotypic {input.phenotypic} \
            --movement {input.movement} \
            --ids {input.ids} \
            --ancestry-pcs {input.ancestry_pcs} \
            --n-ancestry-pcs {params.n_ancestry_pcs} \
            --matrices-dir {params.matrices_dir} \
            --partition {input.partition} \
            --threshold {params.threshold} \
            --motion-threshold {params.motion_threshold} > {log} 2>&1
        """


rule visualize_networks:
    """Visualize network results at the C2b-selected parcellation."""
    input:
        pgs=f"{RESULTS_DIR}/pgs_residuals.csv",
        social=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        graph_metrics=f"{RESULTS_DIR}/C4_main_network_metrics.csv",
        partition=f"{RESULTS_DIR}/C2b_selected_partition.csv",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}"
    output:
        report=f"{PROJECT_DIR}/reports/C5_visualize_networks_report.txt",
        ellipse_plot=f"{PROJECT_DIR}/figures/C5_Bootstrap_Ellipse_Extent_Plot.png"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        n_bootstrap=config.get("n_bootstrap", 1000),
        sample_size=config.get("bootstrap_sample_size", 90)
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/C5_visualize_networks.log"
    shell:
        """
        python {CODE_DIR}/C5_visualize_networks.py \
            --project {PROJECT_DIR} \
            --pgs {input.pgs} \
            --social {input.social} \
            --graph-metrics {input.graph_metrics} \
            --partition {input.partition} \
            --ids {input.ids} \
            --matrices-dir {params.matrices_dir} \
            --n-bootstrap {params.n_bootstrap} \
            --sample-size {params.sample_size} > {log} 2>&1
        """


rule generate_publication_figures:
    """Generate standalone publication-ready SVG figures"""
    input:
        graph_metrics=f"{RESULTS_DIR}/C4_main_network_metrics.csv",
        ellipse_plot=f"{PROJECT_DIR}/figures/C5_Bootstrap_Ellipse_Extent_Plot.png"
    output:
        fig1=f"{PROJECT_DIR}/figures/publication/fig_pgs_social_scatter.svg",
        fig2=f"{PROJECT_DIR}/figures/publication/fig_pgs_distribution.svg",
        fig3=f"{PROJECT_DIR}/figures/publication/fig_pgs_group_boxplot.svg",
        fig4=f"{PROJECT_DIR}/figures/publication/fig_modularity_social_scatter.svg",
        fig5=f"{PROJECT_DIR}/figures/publication/fig_modularity_variability_bar.svg",
        fig6=f"{PROJECT_DIR}/figures/publication/fig_efficiency_variability_bar.svg",
        fig7=f"{PROJECT_DIR}/figures/publication/fig_network_organization_space.svg",
        fig8=f"{PROJECT_DIR}/figures/publication/fig_compensation_strategies.svg",
        fig9=f"{PROJECT_DIR}/figures/publication/fig_bootstrap_density_modularity.svg",
        fig10=f"{PROJECT_DIR}/figures/publication/fig_bootstrap_density_global_efficiency.svg",
        fig11=f"{PROJECT_DIR}/figures/publication/fig_bootstrap_ellipse_extent.svg"
    params:
        results_file=f"{RESULTS_DIR}/C4_main_network_metrics.csv"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/generate_publication_figures.log"
    shell:
        """
        python {CODE_DIR}/generate_publication_figures.py \
            --project {PROJECT_DIR} \
            --results-file {params.results_file} > {log} 2>&1
        """


# ============================================================================
# Quality Control & Reporting
# ============================================================================

rule check_data_retention:
    """Check subject ID retention across analysis steps"""
    input:
        behavioural=f"{RESULTS_DIR}/behavioural_data_preprocessed.csv",
        cfa=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        pgs=f"{PLINK_DIR}/full_pgs_scores.snp.blp.profile",
        pgs_residuals=f"{RESULTS_DIR}/pgs_residuals.csv",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}"
    output:
        f"{RESULTS_DIR}/DataRetention_Overview.csv"
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/check_data_retention.log"
    shell:
        """
        python {CODE_DIR}/utils/check_IDs_per_step.py \
            --project {PROJECT_DIR} \
            --behavioural {input.behavioural} \
            --cfa {input.cfa} \
            --pgs {input.pgs} \
            --pgs-residuals {input.pgs_residuals} \
            --ids {input.ids} \
            --output {output} > {log} 2>&1
        """


# ============================================================================
# Utility Rules
# ============================================================================

rule clean:
    """Remove all generated files"""
    shell:
        """
        # Results directory
        rm -rf {RESULTS_DIR}/*

        # Phase A outputs
        rm -f {DATA_DIR}/behavioural_data_preprocessed.csv
        rm -f {DATA_DIR}/cfa_factor_scores_full_sample.csv

        # Phase B outputs
        rm -f {DATA_DIR}/pgs_selected_threshold.txt
        rm -f {DATA_DIR}/pgs_residuals.csv
        # Clean B1 plinkQC output directory
        rm -rf {QCDIR}/*
        # Clean all PLINK working directory outputs (inputs are in genetics_data/)
        rm -rf {PLINK_DIR}/*

        # Phase C outputs
        rm -f {DATA_DIR}/merged_fMRI_data.csv
        rm -f {DATA_DIR}/C2_final_partition_*.csv

        # Quality control outputs
        rm -f {DATA_DIR}/DataRetention_Overview.csv

        # Reports and figures
        rm -f {PROJECT_DIR}/reports/A1_*.txt
        rm -f {PROJECT_DIR}/reports/A2_*.txt
        rm -f {PROJECT_DIR}/reports/A3_*.txt
        rm -f {PROJECT_DIR}/reports/B0_*.txt
        rm -f {PROJECT_DIR}/reports/B1_*.txt
        rm -f {PROJECT_DIR}/reports/B1b_*.txt
        rm -f {PROJECT_DIR}/reports/B3_*.txt
        rm -f {PROJECT_DIR}/reports/B5_*.txt
        rm -f {PROJECT_DIR}/reports/C1_*.txt
        rm -f {PROJECT_DIR}/reports/C2_*.txt
        rm -f {PROJECT_DIR}/reports/C3_*.txt
        rm -f {PROJECT_DIR}/reports/C3b_*.txt
        rm -f {PROJECT_DIR}/reports/C5_*.txt
        rm -f {PROJECT_DIR}/figures/A1_*.png
        rm -f {PROJECT_DIR}/figures/A2_*.png
        rm -f {PROJECT_DIR}/figures/A3_*.png
        rm -f {PROJECT_DIR}/figures/B1_*.png
        rm -f {PROJECT_DIR}/figures/B1b_*.png
        rm -f {PROJECT_DIR}/figures/B3_*.png
        rm -f {PROJECT_DIR}/figures/B5_*.png
        rm -f {PROJECT_DIR}/figures/C1_*.png
        rm -f {PROJECT_DIR}/figures/C2_*.png
        rm -f {PROJECT_DIR}/figures/C3_*.png
        rm -f {PROJECT_DIR}/figures/C3b_*.png
        rm -f {PROJECT_DIR}/figures/C4_*.png
        rm -f {PROJECT_DIR}/figures/C5_*.png
        rm -f {DATA_DIR}/C5_*.npy

        # Bezier connectome plots from C1 (not prefixed with C1_)
        rm -f {PROJECT_DIR}/figures/*_positive.png
        rm -f {PROJECT_DIR}/figures/*_negative.png
        rm -f {PROJECT_DIR}/figures/*_surf.png

        # Stale pre-rename outputs (B2->B3, B4->B5)
        rm -f {PROJECT_DIR}/reports/B2_*.txt
        rm -f {PROJECT_DIR}/reports/B4_*.txt
        rm -f {PROJECT_DIR}/figures/B2_*.png
        rm -f {PROJECT_DIR}/figures/B4_*.png

        # Publication figures
        rm -rf {PROJECT_DIR}/figures/publication/

        # Logs
        rm -f {LOGS_DIR}/*
        """


rule clean_genetics:
    """Remove genetics outputs (Phase B) and the Phase C results that
    consume pgs_residuals, so a rerun redoes B1 -> B5 and the PGS-dependent
    C-stage steps. Keeps Phase A factor scores and the C2 community
    partitions (the slow consensus-clustering output) intact."""
    shell:
        """
        # Phase B working directories and per-stage results
        rm -rf {QCDIR}/*
        rm -rf {PLINK_DIR}/*
        rm -f {RESULTS_DIR}/pgs_residuals.csv
        rm -f {RESULTS_DIR}/pgs_selected_threshold.txt

        # Phase C results that consume pgs_residuals
        rm -f {RESULTS_DIR}/C3_graph_theory_landscape_results.csv
        rm -f {RESULTS_DIR}/C3b_*.csv
        rm -f {RESULTS_DIR}/C4_*.csv
        rm -f {RESULTS_DIR}/C5_exemplar_subjects.csv
        rm -f {RESULTS_DIR}/DataRetention_Overview.csv

        # Reports
        rm -f {PROJECT_DIR}/reports/B1_*.txt
        rm -f {PROJECT_DIR}/reports/B1b_*.txt
        rm -f {PROJECT_DIR}/reports/B3_*.txt
        rm -f {PROJECT_DIR}/reports/B5_*.txt
        rm -f {PROJECT_DIR}/reports/C1_*.txt
        rm -f {PROJECT_DIR}/reports/C3_*.txt
        rm -f {PROJECT_DIR}/reports/C3b_*.txt
        rm -f {PROJECT_DIR}/reports/C5_*.txt

        # Figures
        rm -f {PROJECT_DIR}/figures/B1_*.png
        rm -f {PROJECT_DIR}/figures/B1b_*.png
        rm -f {PROJECT_DIR}/figures/B3_*.png
        rm -f {PROJECT_DIR}/figures/B5_*.png
        rm -f {PROJECT_DIR}/figures/C1_*.png
        rm -f {PROJECT_DIR}/figures/C3_*.png
        rm -f {PROJECT_DIR}/figures/C3b_*.png
        rm -f {PROJECT_DIR}/figures/C4_*.png
        rm -f {PROJECT_DIR}/figures/C5_*.png

        # Bezier connectome plots from C1 (not prefixed with C1_)
        rm -f {PROJECT_DIR}/figures/*_positive.png
        rm -f {PROJECT_DIR}/figures/*_negative.png
        rm -f {PROJECT_DIR}/figures/*_surf.png

        # Publication figures (depend on C4 metrics)
        rm -rf {PROJECT_DIR}/figures/publication/

        # Logs for the rules being cleared
        rm -f {LOGS_DIR}/B1_*.log
        rm -f {LOGS_DIR}/B1b_*.log
        rm -f {LOGS_DIR}/B2_*.log
        rm -f {LOGS_DIR}/B3_*.log
        rm -f {LOGS_DIR}/B4_*.log
        rm -f {LOGS_DIR}/B5_*.log
        rm -f {LOGS_DIR}/C1_*.log
        rm -f {LOGS_DIR}/C3_*.log
        rm -f {LOGS_DIR}/C3b_*.log
        rm -f {LOGS_DIR}/C5_*.log
        rm -f {LOGS_DIR}/check_data_retention.log
        rm -f {LOGS_DIR}/generate_publication_figures.log
        """
