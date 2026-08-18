"""
Snakemake workflow for Brain Compensation project
Analysis pipeline for integrating phenotypic, genetic (PGS), and fMRI data
"""

configfile: "config.yaml"

# Define project paths
PROJECT_DIR = config["project_dir"]
DATA_DIR = f"{PROJECT_DIR}/data"
CODE_DIR = f"{PROJECT_DIR}/code"
GENETICS_INPUT_DIR = f"{DATA_DIR}/raw_anonymised"  # Original genetics data (read-only)
PLINK_DIR = f"{DATA_DIR}/PLINK_anonymised"        # PLINK working directory (outputs)
PREFILTER_INDIR = f"{PLINK_DIR}/prefilter"        # Sex-prefiltered genotypes (B1 input)
QCDIR = f"{DATA_DIR}/plinkQC_output"              # B1 plinkQC output directory
RESULTS_DIR = f"{PROJECT_DIR}/results"
LOGS_DIR = f"{PROJECT_DIR}/logs"
GCTA_PATH = config.get("gcta_path", "/opt/gcta")
PREFILTER_DROP_SEX = config.get("prefilter_drop_sex", "")

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

        # Phase B: Genetic/PGS analysis
        f"{QCDIR}/Neuro_Chip_anonymised.clean.bed",
        f"{PROJECT_DIR}/reports/B1_plinkQC_genotype_qc_report.txt",
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
        f"{PROJECT_DIR}/reports/C3_continuous_heteroscedasticity_report.txt",
        f"{RESULTS_DIR}/C3_heteroscedasticity_results.csv",

        # Phase D: Publication figures
        f"{PROJECT_DIR}/figures/D2_quintile_cv_by_sex.png",
        f"{PROJECT_DIR}/figures/D2_modularity_efficiency_kde_by_pgs.png",
        f"{PROJECT_DIR}/figures/D2_social_metric_association.png",
        f"{RESULTS_DIR}/D2_quintile_cv_by_sex.csv",
        f"{PROJECT_DIR}/reports/D2_publication_figures_report.txt",
        f"{PROJECT_DIR}/manuscript/figures/Figure_Variability.pdf",
        f"{PROJECT_DIR}/reports/D3_variability_figure_report.txt",

        # Data quality check
        f"{RESULTS_DIR}/DataRetention_Overview.csv"


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

rule prefilter_genotypes_by_sex:
    """Optional sex prefilter applied at the start of the genotype pipeline.
    Configured via prefilter_drop_sex: "F" drops females (reproduces the
    broken chrX filter behaviour), "M" drops males, "" passes the file through
    unchanged. NOTE: when toggling this flag, also clear the QCDIR cache
    (B1 reuses cached PLINK files) — e.g. `rm -rf data/plinkQC_output/*`."""
    input:
        bed=f"{GENETICS_INPUT_DIR}/Neuro_Chip_anonymised.bed",
        bim=f"{GENETICS_INPUT_DIR}/Neuro_Chip_anonymised.bim",
        fam=f"{GENETICS_INPUT_DIR}/Neuro_Chip_anonymised.fam"
    output:
        bed=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.bed",
        bim=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.bim",
        fam=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.fam"
    params:
        drop_sex=PREFILTER_DROP_SEX,
        in_prefix=f"{GENETICS_INPUT_DIR}/Neuro_Chip_anonymised",
        out_prefix=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised"
    log:
        f"{LOGS_DIR}/B0_prefilter_genotypes_by_sex.log"
    shell:
        """
        mkdir -p {PREFILTER_INDIR}
        case "{params.drop_sex}" in
            F)  echo "Dropping females (--filter-males)" > {log}
                plink --bfile {params.in_prefix} --filter-males \
                      --make-bed --out {params.out_prefix} >> {log} 2>&1 ;;
            M)  echo "Dropping males (--filter-females)" > {log}
                plink --bfile {params.in_prefix} --filter-females \
                      --make-bed --out {params.out_prefix} >> {log} 2>&1 ;;
            "") echo "prefilter_drop_sex empty - passing raw genotypes through" > {log}
                cp {input.bed} {output.bed}
                cp {input.bim} {output.bim}
                cp {input.fam} {output.fam} ;;
            *)  echo "ERROR: prefilter_drop_sex must be 'F', 'M', or '' (got '{params.drop_sex}')" >&2
                exit 1 ;;
        esac
        """


rule plinkqc_genotype_qc:
    """Genotype quality control using plinkQC (B1)"""
    input:
        bed=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.bed",
        bim=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.bim",
        fam=f"{PREFILTER_INDIR}/Neuro_Chip_anonymised.fam"
    output:
        clean_bed=f"{QCDIR}/Neuro_Chip_anonymised.clean.bed",
        report=f"{PROJECT_DIR}/reports/B1_plinkQC_genotype_qc_report.txt"
    params:
        sex_check_method=config.get("sex_check_method", "demographics"),
        indir_rel="data/PLINK_anonymised/prefilter"
    log:
        f"{LOGS_DIR}/B1_plinkqc_genotype_qc.log"
    shell:
        """
        Rscript {CODE_DIR}/B1_plinkQC_genotype_qc.R \
            --project {PROJECT_DIR} \
            --indir {params.indir_rel} \
            --sex-check-method {params.sex_check_method} > {log} 2>&1
        """


rule translate_pgs_to_hcp:
    """SNP harmonization, PCA, relatedness filtering, and PGS calculation (B2)"""
    input:
        clean_bed=f"{QCDIR}/Neuro_Chip_anonymised.clean.bed",
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
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}"
    output:
        report=f"{PROJECT_DIR}/reports/C1_run_univariate_fMRI_prediction_report.txt"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        motion_threshold=config.get("motion_threshold", 0.2),
        parcellations=config.get("parcellations", "50 100 200")
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

rule continuous_heteroscedasticity_analysis:
    """Continuous heteroscedasticity analysis at the C2b-selected parcellation."""
    input:
        pgs=f"{RESULTS_DIR}/pgs_residuals.csv",
        social=f"{RESULTS_DIR}/cfa_factor_scores_full_sample.csv",
        behavioural=lambda wildcards: f"{DATA_DIR}/{config['input_behavioural']}",
        phenotypic=lambda wildcards: f"{DATA_DIR}/{config['input_phenotypic']}",
        movement=lambda wildcards: f"{DATA_DIR}/{config['input_movement']}",
        ids=lambda wildcards: f"{DATA_DIR}/{config['input_subject_ids']}",
        partition=f"{RESULTS_DIR}/C2b_selected_partition.csv"
    output:
        report=f"{PROJECT_DIR}/reports/C3_continuous_heteroscedasticity_report.txt",
        results=f"{RESULTS_DIR}/C3_heteroscedasticity_results.csv"
    params:
        matrices_dir=lambda wildcards: f"{DATA_DIR}/{config.get('matrices_dir', 'HCP_PTN1200/netmats')}",
        motion_threshold=config.get("motion_threshold", 0.2),
        threshold=0.2
    conda:
        "environment.yml"
    threads: config.get("threads", 4)
    resources:
        mem_mb=config.get("mem_mb", 16000)
    log:
        f"{LOGS_DIR}/C3_continuous_heteroscedasticity.log"
    shell:
        """
        python {CODE_DIR}/C3_continuous_heteroscedasticity_analysis.py \
            --project {PROJECT_DIR} \
            --pgs {input.pgs} \
            --social {input.social} \
            --behavioural {input.behavioural} \
            --phenotypic {input.phenotypic} \
            --movement {input.movement} \
            --ids {input.ids} \
            --matrices-dir {params.matrices_dir} \
            --partition {input.partition} \
            --threshold {params.threshold} \
            --motion-threshold {params.motion_threshold} > {log} 2>&1
        """


# ============================================================================
# Phase D: Publication Figures
# ============================================================================

rule publication_figures_d2:
    """Per-quintile CV by sex and modularity x global-efficiency KDE
    contours per PGS quintile (D2)."""
    input:
        results=f"{RESULTS_DIR}/C3_heteroscedasticity_results.csv",
        variance_regression=f"{RESULTS_DIR}/C3_variance_regression_sex.csv"
    output:
        cv_fig=f"{PROJECT_DIR}/figures/D2_quintile_cv_by_sex.png",
        kde_fig=f"{PROJECT_DIR}/figures/D2_modularity_efficiency_kde_by_pgs.png",
        social_fig=f"{PROJECT_DIR}/figures/D2_social_metric_association.png",
        cv_csv=f"{RESULTS_DIR}/D2_quintile_cv_by_sex.csv",
        report=f"{PROJECT_DIR}/reports/D2_publication_figures_report.txt"
    params:
        n_bootstrap=config.get("d2_n_bootstrap", 1000),
        seed=config.get("d2_seed", 42)
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/D2_publication_figures.log"
    shell:
        """
        python {CODE_DIR}/D2_publication_figures.py \
            --project {PROJECT_DIR} \
            --input {input.results} \
            --interaction-stats {input.variance_regression} \
            --n-bootstrap {params.n_bootstrap} \
            --seed {params.seed} > {log} 2>&1
        """


rule variability_figure_d3:
    """Pooled modularity-variability figure (panels a-f) on the male subsample.
    Recreates the presentation of the original Figure_2.pdf, correctly
    restricted to males (Gender == 'M'), and writes it to the manuscript's
    variability-figure slot (figures/Figure_Variability.pdf). Pooled by PGS
    group (Low/Middle/High) rather than sex-stratified like D2."""
    input:
        results=f"{RESULTS_DIR}/C3_heteroscedasticity_results.csv"
    output:
        fig=f"{PROJECT_DIR}/figures/D3_variability_figure_m.pdf",
        manuscript_fig=f"{PROJECT_DIR}/manuscript/figures/Figure_Variability.pdf",
        report=f"{PROJECT_DIR}/reports/D3_variability_figure_report.txt"
    params:
        sex=config.get("variability_figure_sex", "M"),
        n_bootstrap=config.get("d3_n_bootstrap", 1000),
        seed=config.get("d3_seed", 42)
    conda:
        "environment.yml"
    log:
        f"{LOGS_DIR}/D3_variability_figure.log"
    shell:
        """
        python {CODE_DIR}/D3_variability_figure.py \
            --project {PROJECT_DIR} \
            --input {input.results} \
            --sex {params.sex} \
            --output {output.fig} \
            --manuscript-output {output.manuscript_fig} \
            --n-bootstrap {params.n_bootstrap} \
            --seed {params.seed} > {log} 2>&1
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
        rm -f {PROJECT_DIR}/reports/B1_*.txt
        rm -f {PROJECT_DIR}/reports/B3_*.txt
        rm -f {PROJECT_DIR}/reports/B5_*.txt
        rm -f {PROJECT_DIR}/reports/C1_*.txt
        rm -f {PROJECT_DIR}/reports/C2_*.txt
        rm -f {PROJECT_DIR}/reports/C3_*.txt
        rm -f {PROJECT_DIR}/reports/D2_*.txt
        rm -f {PROJECT_DIR}/figures/D2_*.png
        rm -f {PROJECT_DIR}/figures/A1_*.png
        rm -f {PROJECT_DIR}/figures/A2_*.png
        rm -f {PROJECT_DIR}/figures/A3_*.png
        rm -f {PROJECT_DIR}/figures/B1_*.png
        rm -f {PROJECT_DIR}/figures/B3_*.png
        rm -f {PROJECT_DIR}/figures/B5_*.png
        rm -f {PROJECT_DIR}/figures/C1_*.png
        rm -f {PROJECT_DIR}/figures/C2_*.png
        rm -f {PROJECT_DIR}/figures/C3_*.png

        # Bezier connectome plots from C1 (not prefixed with C1_)
        rm -f {PROJECT_DIR}/figures/*_positive.png
        rm -f {PROJECT_DIR}/figures/*_negative.png
        rm -f {PROJECT_DIR}/figures/*_surf.png

        # Stale pre-rename / retired-rule outputs
        # (B0->B1, B2->B3, B4->B5; retired: visualize_networks (C4/C5),
        # main_landscape_analysis (C3 perform), generate_publication_figures)
        rm -f {PROJECT_DIR}/reports/B0_*.txt
        rm -f {PROJECT_DIR}/reports/B2_*.txt
        rm -f {PROJECT_DIR}/reports/B4_*.txt
        rm -f {PROJECT_DIR}/reports/C4_*.txt
        rm -f {PROJECT_DIR}/reports/C5_*.txt
        rm -f {PROJECT_DIR}/figures/B0_*.png
        rm -f {PROJECT_DIR}/figures/B2_*.png
        rm -f {PROJECT_DIR}/figures/B4_*.png
        rm -f {PROJECT_DIR}/figures/C4_*.png
        rm -f {PROJECT_DIR}/figures/C5_*.png
        rm -f {DATA_DIR}/C4_*.npy
        rm -f {DATA_DIR}/C5_*.npy
        rm -f {RESULTS_DIR}/C4_main_network_metrics.csv
        rm -f {RESULTS_DIR}/C4_sensitivity_summary.csv
        rm -f {RESULTS_DIR}/C3_graph_theory_landscape_results.csv
        rm -rf {PROJECT_DIR}/figures/publication/

        # Logs
        rm -f {LOGS_DIR}/*
        """
