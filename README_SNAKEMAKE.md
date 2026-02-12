# Brain Compensation Project - Snakemake Workflow

This document explains how to use the Snakemake workflow for the Brain Compensation analysis pipeline.

## Overview

The workflow orchestrates three main analysis phases:
- **Phase A**: Phenotypic data preprocessing (Python + R)
- **Phase B**: Genetic/PRS analysis (PLINK + Python)
- **Phase C**: fMRI analysis (Python)

## Prerequisites

1. **Install Snakemake** (if not already installed):
```bash
conda install -c conda-forge -c bioconda snakemake
```

2. **Verify your conda environment** exists:
```bash
conda env list | grep BrainComp
```

3. **Install R packages** (for factor analysis step):
```R
install.packages(c("lavaan", "psych", "dplyr"))
```

## Setup

1. **Edit the config file** ([config.yaml](config.yaml)) to match your data:
   - Update `input_behavioural` with your behavioural data filename
   - Update `input_phenotypic` with your phenotypic data filename
   - Update `genetic_data_prefix` with your genetic data PLINK prefix
   - Adjust `threads` and `mem_mb` based on your system resources

2. **Create required directories**:
```bash
mkdir -p ../data/results
mkdir -p logs
```

## Running the Workflow

### Dry-run (recommended first step)
Check what will be executed without actually running anything:
```bash
snakemake --dry-run --cores 1
```

### Visualize the workflow
Generate a DAG (directed acyclic graph) of the workflow:
```bash
snakemake --dag | dot -Tpng > workflow_dag.png
```

### Run the full pipeline
Execute all rules using 4 cores:
```bash
snakemake --cores 4 --use-conda
```

### Run specific phases

**Only phenotypic preprocessing (Phase A)**:
```bash
snakemake --cores 4 --use-conda \
    ../data/behavioural_data_preprocessed.csv \
    ../data/cfa_factor_scores_full_sample.csv \
    ../data/social_factor_evaluation.png
```

**Only genetic analysis (Phase B)**:
```bash
snakemake --cores 4 --use-conda \
    ../data/PLINK_anonymised/full_prs_scores.snp.blp.profile \
    ../data/blup_evaluation_results.png
```

**Only fMRI analysis (Phase C)**:
```bash
snakemake --cores 4 --use-conda \
    ../data/results/network_visualizations.png
```

### Run specific rules
Execute a single rule:
```bash
snakemake --cores 1 --use-conda preprocess_phenotypic
```

## Workflow Structure

### Phase A: Phenotypic Data
1. `preprocess_phenotypic` - Clean and impute behavioural/phenotypic data
2. `factor_analysis` - Run CFA on behavioural measures (R)
3. `evaluate_social_factor` - Evaluate social factor results

### Phase B: Genetic/PRS
1. `translate_prs_to_hcp` - Map PRS to HCP subjects
2. `select_prs_threshold` - Optimize PRS threshold
3. `extend_prs_with_blup` - Extend with BLUP predictions
4. `evaluate_blup` - Evaluate BLUP accuracy

### Phase C: fMRI Analysis
1. `univariate_fmri_prediction` - Univariate prediction models
2. `multivariate_fmri_prediction` - Multivariate prediction models
3. `find_fmri_communities` - Network community detection
4. `main_landscape_analysis` - Main landscape analysis
5. `sensitivity_landscape_analysis` - Sensitivity analysis
6. `visualize_networks` - Generate network visualizations

### Quality Control
- `check_data_retention` - Track subject IDs across pipeline steps

## Monitoring and Logs

All logs are saved in the `logs/` directory with filenames matching the rule names.

View a specific log:
```bash
tail -f logs/C1_univariate_fmri.log
```

## Troubleshooting

### Missing input files
If you get errors about missing input files, check:
1. The filenames in [config.yaml](config.yaml) match your actual data files
2. All required input files exist in the `../data/` directory

### Script path errors
The workflow assumes all scripts are in the `code/` directory. If you've reorganized files, update the paths in the [Snakefile](Snakefile).

### Memory errors
If jobs fail due to memory issues, increase `mem_mb` in [config.yaml](config.yaml).

### Conda environment issues
Make sure the environment is activated when running Snakemake:
```bash
conda activate BrainComp
snakemake --cores 4 --use-conda
```

## Rerunning Failed Jobs

If a job fails, fix the issue and rerun:
```bash
snakemake --cores 4 --use-conda --rerun-incomplete
```

## Cleaning Up

Remove all generated files (be careful!):
```bash
snakemake clean
```

## Advanced Options

### Run on a cluster (SLURM example)
```bash
snakemake --cluster "sbatch -t 2:00:00 -c {threads} --mem={resources.mem_mb}" \
    --jobs 10 --use-conda
```

### Force rerun specific rules
```bash
snakemake --cores 4 --use-conda --forcerun preprocess_phenotypic
```

### Generate a workflow report
```bash
snakemake --report report.html
```

## Customization

To modify the workflow:
1. Edit [Snakefile](Snakefile) to add/remove rules
2. Update [config.yaml](config.yaml) to add new parameters
3. Adjust resource requirements (threads, memory) for specific rules

## Dependencies

The workflow uses the conda environment defined in [environment.yml](environment.yml). Snakemake will automatically activate this environment when using the `--use-conda` flag.
