# Autism Brain Variability — behaviour-stratified

Analysis code for a study of brain–behaviour variability in the Human
Connectome Project (HCP). Participants are stratified by **behaviour**
(Social Difficulty Score, SDS = the CFA `Social_Score` factor where higher z
= worse social performance) and we test whether worse social performance
goes with **greater variability** in graph-theoretical brain measures.
PGS is reserved for an exploratory secondary test of the SDS↔PGS link.

The pipeline integrates three data modalities:

1. **Phenotypic** — HCP behavioural measures, reduced via confirmatory factor
   analysis to a social-cognition factor (`Social_Score`, the SDS).
2. **fMRI** — resting-state connectivity, consensus community detection, and
   graph-theoretical landscape analysis (modularity, global efficiency, and
   their variance as a function of SDS).
3. **Genetic (exploratory only)** — genotype QC, PGS calculation from the
   Grove et al. (2019) iPSYCH-PGC autism GWAS, and BLUP extension to the full
   (related) sample. PGS is intentionally NOT joined into the primary SDS
   analyses to preserve N; only C6 inner-joins on the genotyped subset.

## Core hypotheses

- **Primary.** Participants with higher SDS (worse social performance) show
  greater variability of modularity and (to a lesser extent) global
  efficiency, with means relatively preserved.
- **Secondary (exploratory).** Higher autism PGS is associated with higher
  SDS.

The primary hypothesis is tested by SDS-tertile group comparison (low / middle
/ high SDS, z-cuts with buffer) and by continuous heteroscedasticity tests
(Breusch–Pagan, White, quantile regression, decile-trend, balanced-bootstrap,
DGLM). The secondary is tested by an OLS regression in C6.

The previous PGS-primary scripts (when participants were grouped by autism
PGS) are quarantined under `archive/pgs_primary/` and are not part of the
active workflow. The pre-flip state is also tagged `pgs-primary-v1`.

## Repository layout

```
code/
  A1_preprocess_phenotypic_data.py    # HCP behavioural + phenotypic cleanup, imputation
  A2_factor_analysis.R                # CFA for the social factor
  A3_evaluate_social_factor.py        # Sanity-check the CFA factor

  B1_plinkQC_genotype_qc.R            # Genotype QC (plinkQC)
  B2_translate_PGS_to_HCP.sh          # SNP harmonization, PCA, PRSice PGS
  B3_select_PGS_threshold.py          # p-value threshold selection (p=0.1, per Grove 2019)
  B4_extend_PGS_with_BLUP.sh          # GCTA BLUP extension to related individuals
  B5_evalute_BLUP_prediction.py       # BLUP evaluation + residualized PGS

  C1_run_univariate_fMRI_prediction.py      # Edge-wise connectivity ~ social / PGS
  C2_find_communities_fMRI.py               # Consensus community detection
  C2b_evaluate_communities.py               # Parcellation resolution tuning
  C3_perform_main_landscape_analysis.py     # SDS-stratified group + variability tests
  C3b_continuous_heteroscedasticity_analysis.py  # Continuous BP / White / quantile / DGLM on SDS
  C5_visualize_networks.py                  # SDS-grouped bootstrap ellipses & network figures
  C6_exploratory_sds_pgs.py                 # Exploratory: Social_Score ~ blup_PGS_residuals
  D1_sensitivity_analyses.py                # SDS-stratified sensitivity sweep (cohort/covariate cells)
  D2_grid_sensitivity.py                    # SDS-stratified grid (parcellation × matrix-type × threshold)
  generate_publication_figures.py           # Standalone publication SVGs (SDS-themed)

  archive/pgs_primary/                # Frozen PGS-primary versions of C3/C3b/C5/generate_publication_figures
  utils/                              # Shared helpers (ID tracking, connectome viz, covariates, …)

  Snakefile                           # Workflow definition (see README_SNAKEMAKE.md)
  config.yaml                         # Paths, thresholds, parcellation sizes
  environment.yml                     # Conda environment (BrainComp)
  README_SNAKEMAKE.md                 # Detailed Snakemake usage guide
```

Scripts are numbered by phase (A / B / C) and execution order; letter suffixes
(`C2b`, `C3b`) mark companion analyses rather than replacements.

## Requirements

### Software

- Python 3.11 (see [environment.yml](environment.yml))
- R ≥ 4.0 with `lavaan`, `psych`, `dplyr`, `plinkQC`
- [PLINK 1.9 / 2.0](https://www.cog-genomics.org/plink/)
- [PRSice-2](https://choishingwan.github.io/PRSice/) (invoked by `B2`)
- [GCTA](https://yanglab.westlake.edu.cn/software/gcta/) (invoked by `B4`, path set in `config.yaml`)
- [Snakemake](https://snakemake.readthedocs.io/) for workflow orchestration

Install the conda environment once:

```bash
conda env create -f environment.yml
conda activate BrainComp
```

### Data

The pipeline expects HCP data and an autism GWAS summary file, organised under
`project_dir/data/` as configured in [config.yaml](config.yaml):

- `raw_anonymised/behavioural_data_anonymised.csv` — HCP behavioural
- `raw_anonymised/phenotypic_data_anonymised.csv` — HCP phenotypic (demographics, QC)
- `raw_anonymised/movement_data_anonymised.csv` — motion summary per subject
- `raw_anonymised/subject_ids_anonymised.txt` — subject ID list
- `raw_anonymised/Neuro_Chip_anonymised.{bed,bim,fam}` — HCP Neuro-Chip genotypes
- `raw_anonymised/iPSYCH_PGC_ASD_Nov_2017.gz` — Grove et al. (2019) GWAS summary stats
- `HCP_PTN1200/netmats/` — HCP group-ICA netmats (parcellations 15–300)

**The raw HCP data is not included in this repository** — access is governed
by the HCP data use terms. See
[HCP data access](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms).

## Quick start

Edit [config.yaml](config.yaml) to point at your `project_dir` and GCTA
install, then:

```bash
# Dry-run to inspect the DAG
snakemake --use-conda -n

# Run the full pipeline on 4 cores
snakemake --use-conda --cores 4

# Stop after community-detection tuning (Phase C, C2b)
snakemake --use-conda --cores 4 --until evaluate_parcellations
```

See [README_SNAKEMAKE.md](README_SNAKEMAKE.md) for per-phase targets,
cluster execution, logging, and troubleshooting.

## Outputs

Outputs are written to `project_dir/{results,reports,figures,logs}/`:

- `results/` — numeric results (CSV): factor scores, PGS residuals, partitions,
  graph-theory metrics, heteroscedasticity tests, data-retention tracking.
- `reports/` — per-step text reports summarising what ran and key statistics.
- `figures/` — per-step PNGs. `figures/publication/` contains standalone SVGs
  produced by `generate_publication_figures.py`.
- `logs/` — one log file per Snakemake rule.

## Reproducibility notes

- PGS threshold is **fixed at p = 0.1** (the discovery-GWAS optimum from Grove
  et al. 2019) rather than re-tuned in-sample, to avoid overfitting. See the
  header of [B3_select_PGS_threshold.py](B3_select_PGS_threshold.py).
- Community detection uses consensus clustering across `n_iterations` runs
  (default 50); the resolution is selected a priori (5–8 communities for
  resting-state networks) in [C2b_evaluate_communities.py](C2b_evaluate_communities.py).
  The selected partition is written to a stable filename
  (`C2b_selected_partition.csv`) so downstream steps inherit it without
  repeating the parameter search.
- Landscape analysis (C3, C3b, C5) runs at the single parcellation chosen
  by C2b — `n_nodes` is derived from the selected partition. Robustness is
  assessed across edge thresholds only (0.15 / 0.20 / 0.25) in
  [C3_perform_main_landscape_analysis.py](C3_perform_main_landscape_analysis.py).

## Citation

TODO — add paper / preprint citation once available.

## License

This project is licensed under the terms of the MIT license.
