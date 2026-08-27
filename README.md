# Autism Brain Variability

Analysis code for a study testing **landscape theory** of brain organization
in autism using the Human Connectome Project (HCP): does higher autism
polygenic score (PGS) predict greater *variability* in brain network
organisation, while preserving global efficiency?

The pipeline integrates three data modalities:

1. **Phenotypic** — HCP behavioural measures, reduced via confirmatory factor
   analysis to a social-cognition factor.
2. **Genetic** — genotype QC, PGS calculation from the Grove et al. (2019)
   iPSYCH-PGC autism GWAS, and BLUP extension to the full (related) sample.
3. **fMRI** — resting-state connectivity, consensus community detection, and
   graph-theoretical landscape analysis (modularity, global efficiency, and
   their variance as a function of PGS).

## Core hypotheses

- **Brain–behaviour link.** Graph measures of integration and segregation predict social functioning.
- **Compensation diversity.** High-PGS individuals show greater *variability*
  in graph measures of integration and segregation.


These are tested via formal heteroscedasticity analyses on the continuous
PGS (Breusch–Pagan, White, quantile regression, decile-SD trends, and a
balanced bootstrap variance test). The primary test is a generalised
Breusch–Pagan with a `pgs_z × sex` interaction on the squared residuals
(does the PGS→variance slope differ by sex?), supported by sex-stratified
versions of the decile-SD and balanced-variance tests.

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
  C3_continuous_heteroscedasticity_analysis.py   # Continuous BP / White / quantile + sex × PGS variance tests

  utils/                              # Shared helpers (ID tracking, connectome viz, …)

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

### Option A — Docker (recommended, reproducible)

Everything the pipeline needs is pinned in a container image: Python, R, PLINK,
PRSice-2, GCTA, LaTeX and offscreen OpenGL. From the repository root:

```bash
./docker/run.sh            # build the image, then run the whole pipeline
./docker/run.sh -n         # dry run
./docker/run.sh --selftest # verify every dependency, touch no data
```

The image contains **no data** — supply `data/` yourself (see [Data](#data)),
and `docker/run.sh` wires up the mounts, including resolving the symlinks under
`data/`. Outputs are written back into the repository owned by your user.

See [../docker/README.md](../docker/README.md) for details, and note that a
genuine end-to-end reproduction needs a clean tree (`./docker/run.sh clean`
first) — otherwise Snakemake correctly does nothing, because the outputs are
already present.

### Option B — native install

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
- `figures/` — per-step PNGs.
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
- Heteroscedasticity analysis (C3) runs at the single parcellation chosen
  by C2b — `n_nodes` is derived from the selected partition. See
  [C3_continuous_heteroscedasticity_analysis.py](C3_continuous_heteroscedasticity_analysis.py).

## Citation

TODO — add paper / preprint citation once available.

## License

This project is licensed under the terms of the MIT license.
