# Autism Brain Variability

Analysis code for a study testing **landscape theory** of brain organisation in
autism using the Human Connectome Project (HCP): does a higher autism polygenic
score (PGS) predict greater *variability* in brain network organisation, while
global efficiency is preserved?

The pipeline integrates three data modalities:

1. **Phenotypic** — HCP behavioural measures, reduced by confirmatory factor
   analysis to a social-cognition factor.
2. **Genetic** — genotype QC, PGS from the Grove et al. (2019) iPSYCH-PGC autism
   GWAS, and BLUP extension to the full (related) sample.
3. **fMRI** — resting-state connectivity, consensus community detection, and
   graph-theoretical landscape analysis.

Everything is orchestrated by a single [Snakemake](code/Snakefile) workflow of
20 rules across four phases (A: phenotypic, B: genetics, C: fMRI, D: figures).

## Reproducing the analysis

The whole pipeline runs in a pinned container — Python, R, PLINK, PRSice-2,
GCTA, LaTeX and offscreen OpenGL, all at fixed versions:

```bash
./docker/run.sh              # build the image, then run the full pipeline
./docker/run.sh -n           # dry run: show the DAG
./docker/run.sh --selftest   # verify every dependency, touch no data
```

`make help` lists the shortcuts. See [docker/README.md](docker/README.md) for
mounts, provenance and caveats, and [code/README.md](code/README.md) for the
scientific detail and a native (non-container) install.

## Repository layout

```
code/         analysis scripts (Python, R, shell) + the Snakefile
docker/       container definition, pinned environment, run wrapper
Makefile      convenience targets over docker/run.sh
```

Not tracked here, and created or supplied locally:

```
data/         inputs - see "Data access" below (never committed)
results/      numeric outputs (CSV)
reports/      per-step text reports
figures/      per-step figures
logs/         one log per Snakemake rule
manuscript/   paper sources
```

## Data access

**No data is included in this repository, and none can be.** To run the
pipeline you must obtain, under your own agreements:

| Path | Source |
|---|---|
| `data/HCP_PTN1200/` | HCP1200 PTN release (netmats + group-ICA) via [ConnectomeDB](https://db.humanconnectome.org/), under the [HCP Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms) |
| `data/raw_anonymised/*.csv`, `*.txt` | HCP behavioural, phenotypic and motion tables |
| `data/raw_anonymised/Neuro_Chip_anonymised.{bed,bim,fam}` | HCP genotypes (restricted access) |
| `data/raw_anonymised/iPSYCH_PGC_ASD_Nov_2017.gz` | Grove et al. (2019) GWAS summary statistics, under PGC terms |

Paths are configured in [code/config.yaml](code/config.yaml).

## Citation

TODO — add the paper / preprint citation once available.

## License

MIT — see [LICENSE](LICENSE).
