# Reproducing the pipeline with Docker

The image contains **every software dependency** the Snakemake workflow needs —
Python, R, PLINK, PRSice-2, GCTA, LaTeX, and offscreen OpenGL — pinned to the
versions the published analysis ran against. It contains **no data**.

## Quick start

```bash
cd /path/to/BrainCompensation-repro
./docker/run.sh
```

That builds the image (first run: ~25–40 min) and then runs the whole pipeline.
Every argument is forwarded to `snakemake`:

```bash
./docker/run.sh -n            # dry run: show the DAG, change nothing
./docker/run.sh --selftest    # verify the environment, touch no data
./docker/run.sh clean         # delete outputs via the workflow's `clean` rule
./docker/run.sh bash          # interactive shell
./docker/run.sh /project/results/pgs_residuals.csv    # one target

CORES=16 MEM_MB=64000 ./docker/run.sh
SKIP_BUILD=1 ./docker/run.sh  # reuse the existing image
```

`make help` lists equivalent shortcuts.

> **Incremental by default.** Snakemake sees the outputs already in `results/`,
> `figures/` and `reports/` and does nothing. That is correct behaviour, but it
> is *not* a reproduction. For a genuine end-to-end run, start from a clean
> tree: `./docker/run.sh clean` first, or clone the repo fresh.

## What you must supply: the data

The image ships zero data, and the data cannot be redistributed. Provide, under
your own agreements:

| Path | Size | Source |
|---|---|---|
| `data/HCP_PTN1200/{netmats,groupICA}` | ~15 GB | HCP1200 PTN release, ConnectomeDB (HCP Data Use Agreement) |
| `data/raw_anonymised/*.csv`, `*.txt` | small | Anonymised HCP behavioural / phenotypic / movement tables |
| `data/raw_anonymised/Neuro_Chip_anonymised.{bed,bim,fam}` | ~180 MB | Anonymised HCP genotypes (restricted) |
| `data/raw_anonymised/iPSYCH_PGC_ASD_Nov_2017.gz` | ~134 MB | PGC ASD 2017 summary statistics (PGC access terms) |

Never `COPY` these into an image or push an image built with them.

### Symlinked data

In this checkout the three entries below are symlinks into a sibling directory:

```
data/raw_anonymised        -> .../BrainCompensation/data/raw_anonymised
data/HCP_PTN1200           -> .../BrainCompensation/data/HCP_PTN1200
data/hcp_behavioural_raw.csv -> .../BrainCompensation/data/hcp_behavioural_raw.csv
```

Docker does not follow host symlinks across a bind mount, so a naive
`-v $PWD:/project` leaves them dangling and every rule fails. `docker/run.sh`
handles this: it `readlink -f`s each symlink under `data/` and mounts the
resolved target **at its own absolute path** (read-only), so the symlink
resolves naturally inside the container. Real directories need no special
handling.

`data/PLINK_anonymised` and `data/plinkQC_output` are pipeline *outputs* that
live inside `data/`; they stay writable through the top-level mount.

## How it works

- **Base:** `rocker/r-ver:4.5.3`, pinned by digest. It fixes R at 4.5.3 and
  ships a date-pinned Posit binary CRAN snapshot
  (`p3m.dev/cran/__linux__/noble/2026-04-23`) that resolves to exactly the
  package versions the pipeline last ran against — including **lavaan 0.6-21**,
  the version recorded in `logs/A2_factor_analysis.log`, and **plinkQC 1.1.0**.
- **Python:** micromamba creates the env from
  [`environment.linux.yml`](environment.linux.yml), transcribed from the working
  linux-64 environment (`~/anaconda3/envs/BrainComp`) rather than from
  `code/environment.yml`, which is an Apple-Silicon export that cannot solve on
  linux-64 and is missing `networkx`.
- **Binaries:** PLINK 1.9 from bioconda — it self-reports `v1.9.0-b.8
  (22 Oct 2024)`, matching `logs/B2_translate_pgs.log` from the original run.
  GCTA 1.95.1 and PRSice-2 2.3.5 are downloaded at pinned versions and verified
  against SHA-256 checksums recorded in `/opt/provenance/binaries.sha256`.
  GCTA ships as an AppImage that self-mounts via FUSE, which is unavailable in a
  container, so the Dockerfile unpacks it at build time and installs the inner
  binary together with its bundled Intel MKL under `/opt/gcta`.
- **Graphics:** `xvfb` plus Mesa, because C1 and C2 render surfaces through
  surfplot/brainspace/VTK under `xvfb-run -a`.
- **LaTeX:** a full TeX Live set, because A1, A3 and C1 set
  `matplotlib.rcParams['text.usetex'] = True`.
- **Prefetched:** the neuromaps fsLR surfaces, so runs need no network.

### Paths and config

`code/config.yaml` is **not** modified. Its `project_dir` is host-specific, so
the entrypoint overrides just that key and `gcta_path` on the command line:

```
snakemake --snakefile /project/code/Snakefile \
          --directory /work \
          --configfile /project/code/config.yaml \
          --config project_dir=/project gcta_path=/usr/local/bin \
          --cores $CORES --rerun-incomplete --printshellcmds --latency-wait 30
```

`--directory /work` keeps Snakemake's `.snakemake/` state in a container-local
mount, so the host's `code/.snakemake` — whose metadata filenames encode
absolute host paths — is never read or written. The two states are independent
and will disagree about what is up to date; `make clean-state` resets the
container's.

`--use-conda` is rejected by the entrypoint: the environment is baked in, and
the rules' `conda:` directives are inert without that flag.

## Verifying the image

```bash
make build
make selftest    # versions, imports, R libraries, usetex, xvfb+surfplot
make offline     # the same with --network none: proves nothing fetches at runtime
make dry         # snakemake dry run
make phaseA      # ~minutes: Python + R/lavaan + LaTeX + mounts + file ownership
```

`make offline` is the meaningful check on the R layer. The three R scripts call
`install.packages()` at runtime if a package is missing; with networking off,
that fails loudly instead of hanging on a CRAN mirror.

After `make phaseA`, confirm outputs are yours, not root's:

```bash
stat -c '%U:%G' figures/A1_Behaviour_correlations.png
```

`docker/run.sh` picks the right ownership strategy automatically:

- **Rootful Docker** — passes `--user $(id -u):$(id -g)`, so outputs are written
  as you rather than as root.
- **Rootless Docker** — container-root *already* maps to your host user, so
  `--user` is omitted; passing it would map to an unrelated subuid and every
  write would fail with `Permission denied`.

Override with `DOCKER_USER=1000:1000 ./docker/run.sh` if you need something else.

## Provenance

The image records exactly what it contains in `/opt/provenance/`:
`conda-env.full.yml`, `conda-env.explicit.txt` (an `@EXPLICIT` URL list that
rebuilds the env bit-for-bit), `pip-freeze.txt`, `r-packages.csv`,
`binaries.sha256`, `gcta-MIT_License.txt`. Extract them with:

```bash
docker run --rm --entrypoint tar braincomp:latest -cC /opt/provenance . \
  | tar -x -C docker/provenance/
```

## Expectations and caveats

- **Runtime:** measured on this machine (64 cores), the 15-job path from Phase A
  through Phase D — including C1 (xvfb/surfplot), C2 (consensus Louvain over six
  parcellations), C3, C4's 1000 bootstraps and B4's GCTA GRM — completed in
  **~8 minutes**. B1 (plinkQC) and B2 (PRSice) were already up to date and did
  not re-run; budget additional time for those on a genuinely clean tree. Peak
  RSS stayed well within the configured `mem_mb: 16000`.
- **Image size:** ~3.6 GB — TeX Live ~1.2 GB, the conda env ~1.5 GB (VTK alone
  ~340 MB), and the unpacked GCTA ~280 MB.
- **linux/amd64 only.** PLINK, PRSice and GCTA are x86-64 binaries and
  `vtk==9.3.1` is a `manylinux_2_17_x86_64` wheel. On Apple Silicon this
  emulates and will be very slow.
- **Not bit-identical to the original macOS runs, but numerically equivalent.**
  Measured by re-running Phase A in the container and diffing against the
  pre-existing outputs:
  - `results/behavioural_data_preprocessed.csv` — byte-identical.
  - `results/cfa_factor_scores_full_sample.csv` — max |difference| **8.9e-08**,
    Pearson r = 1.0000000000 over all 1025 subjects.
  - `reports/A{1,2,3}_*.txt` — every statistic identical; the only diffs are the
    recorded paths (`/home/jmbathe/...` vs `/project/...`).

  The residual 1e-8 drift is the expected consequence of a different BLAS
  (macOS Accelerate vs Linux OpenBLAS). Seeds are fixed (`42`) throughout, so
  run-to-run determinism *within* the image holds exactly.
- **Three build-time network dependencies are single-sourced**: the GCTA 1.95.1
  zip (Yang Lab only — GitHub releases stop at 1.94.1), the neuromaps fsLR
  fetch from OSF, and the P3M CRAN snapshot. Checksums make substitution
  detectable, but `docker save` of a built image is the real archival artefact.
- **Licences:** PLINK and PRSice-2 are GPLv3, GCTA is MIT — the image is
  redistributable. Prefer publishing the Dockerfile over a registry image so
  the downloads are re-fetched from upstream.
- `code/environment.yml` is left untouched as the record of the author's macOS
  environment. It is not used by the image.
- `D1_create_consort_diagram.py` has no Snakemake rule and is not in `rule all`.
  The image has everything it needs (TikZ + `preview.sty`); run it manually via
  `./docker/run.sh bash`.
- `data/reference/1000Genomes` is empty and that is fine —
  `B1_plinkQC_genotype_qc.R` defaults `skip_ancestry <- TRUE`, so neither the
  1000 Genomes loadings nor `plink2` are exercised.
