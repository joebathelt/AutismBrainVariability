# syntax=docker/dockerfile:1.7
# ===========================================================================
# BrainCompensation analysis pipeline - reproducible environment
#
# Contains every dependency the Snakemake workflow needs; contains NO DATA.
# Mount the project at /project (see docker/run.sh) and the container's
# default command runs `snakemake` over the whole pipeline.
#
# linux/amd64 only: PLINK / PRSice / GCTA are x86-64 binaries and vtk==9.3.1
# is a manylinux_2_17_x86_64 wheel.
# ===========================================================================

# rocker/r-ver pins R to an exact version AND ships a date-pinned Posit binary
# CRAN snapshot (CRAN=https://p3m.dev/cran/__linux__/noble/2026-04-23). That
# snapshot resolves to exactly the nine package versions in the library the
# pipeline last ran against (~/R/x86_64-pc-linux-gnu-library/4.5), including
# lavaan 0.6-21 -- the version recorded in logs/A2_factor_analysis.log.
FROM rocker/r-ver:4.5.3@sha256:c3f39b365d1077fe24f8e9ab2742e352b6d3950897f51af1624a5bb5550c21c0

LABEL org.opencontainers.image.title="BrainCompensation pipeline" \
      org.opencontainers.image.description="Snakemake pipeline: phenotypic (R/lavaan), genetics (PLINK/PRSice/GCTA/plinkQC), fMRI (nilearn/bctpy/surfplot)" \
      org.opencontainers.image.licenses="PLINK GPLv3, PRSice-2 GPLv3, GCTA MIT; see docker/README.md"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    TZ=Etc/UTC
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /opt/provenance

# ---------------------------------------------------------------------------
# Layer 1: system packages (changes ~never -> first, for cache)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget unzip bzip2 gzip xz-utils gawk procps file \
      \
      # Offscreen OpenGL for vtk / brainspace / surfplot. Snakefile:339 and
      # :377 wrap C1 and C2 in `xvfb-run -a python`.
      xvfb libgl1 libglx-mesa0 libgl1-mesa-dri libglu1-mesa libegl1 libopengl0 \
      libx11-6 libxext6 libxrender1 libxt6 libsm6 libice6 libxcursor1 \
      libxinerama1 libxi6 libxrandr2 libxfixes3 libxtst6 libxcb1 \
      libfontconfig1 libfreetype6 \
      \
      # LaTeX for matplotlib rcParams['text.usetex']=True, set in
      # A1:20, A3:14, C1:53 and utils/plot_standalone_colourbar.py:8.
      # texlive-latex-extra also supplies preview.sty for D1's TikZ diagram;
      # fonts-urw-base35 supplies Nimbus Sans for the 'Helvetica' requests.
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      texlive-pictures texlive-fonts-recommended texlive-science \
      cm-super dvipng ghostscript fonts-urw-base35 fonts-cmu \
      \
      # Headers in case a CRAN package needs a source build.
      libxml2-dev libcurl4-openssl-dev libssl-dev libglpk-dev libgit2-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Layer 2: R packages, from the base image's date-pinned P3M snapshot
#
# Pre-installing these means the install_if_missing() helpers in
# A2_factor_analysis.R:4-12, B1_plinkQC_genotype_qc.R:20-28 and
# utils/harmonize_snps.R:28-37 never fire. In an offline container a runtime
# install.packages() would hang rather than fail.
# ---------------------------------------------------------------------------
RUN install2.r --error --skipinstalled --ncpus -1 \
      plinkQC ggplot2 data.table dplyr lavaan psych tibble magrittr R.utils \
 && Rscript -e 'for (p in c("plinkQC","ggplot2","data.table","dplyr","lavaan","psych","tibble","magrittr","R.utils")) suppressPackageStartupMessages(library(p, character.only=TRUE)); cat("R layer OK\n")' \
 && Rscript -e 'ip <- installed.packages()[, c("Package","Version")]; write.csv(ip, "/opt/provenance/r-packages.csv", row.names=FALSE)' \
 && rm -rf /tmp/downloaded_packages /tmp/Rtmp*

# ---------------------------------------------------------------------------
# Layer 3: micromamba + the Python / genetics conda environment
# ---------------------------------------------------------------------------
ARG MICROMAMBA_VERSION=2.9.0-0
ARG MICROMAMBA_SHA256=8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd
RUN set -eux; \
    curl -fsSL -o /tmp/mm.tar.bz2 \
      "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64.tar.bz2"; \
    echo "${MICROMAMBA_SHA256}  /tmp/mm.tar.bz2" | sha256sum -c -; \
    echo "micromamba ${MICROMAMBA_VERSION} sha256=${MICROMAMBA_SHA256}" >> /opt/provenance/binaries.sha256; \
    tar -xjf /tmp/mm.tar.bz2 -C /tmp bin/micromamba; \
    install -m0755 /tmp/bin/micromamba /usr/local/bin/micromamba; \
    rm -rf /tmp/mm.tar.bz2 /tmp/bin

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_ENV=/opt/conda/envs/braincomp

COPY docker/environment.linux.yml /opt/environment.linux.yml
RUN micromamba create -y -p "${CONDA_ENV}" -f /opt/environment.linux.yml \
 && micromamba clean -a -y -f

# Put the env first on PATH. Note R stays at /usr/local/bin/Rscript (rocker's),
# which is what the Snakefile's `Rscript ...` calls resolve to.
ENV PATH=/opt/conda/envs/braincomp/bin:${PATH}

RUN micromamba env export -p "${CONDA_ENV}"            > /opt/provenance/conda-env.full.yml \
 && micromamba env export -p "${CONDA_ENV}" --explicit > /opt/provenance/conda-env.explicit.txt \
 && pip freeze                                         > /opt/provenance/pip-freeze.txt

# ---------------------------------------------------------------------------
# Layer 4: external genetics binaries
#
# PLINK 1.9 comes from bioconda (layer 3) as the same build as the host.
# GCTA goes to /usr/local/bin because config.yaml sets gcta_path: /usr/local/bin
# and B4_extend_PGS_with_BLUP.sh calls ${GCTA_FOLDER}/gcta64.
# PRSice MUST be named PRSice_linux: hardcoded at B2_translate_PGS_to_HCP.sh:242.
# ---------------------------------------------------------------------------
ARG PRSICE_VERSION=2.3.5
ARG PRSICE_SHA256=25a143edcf966225abf7b716aeff04b9386317973a981e3f1cc533ae48d6cae7
ARG GCTA_VERSION=1.95.1
ARG GCTA_SHA256=78f255041825ae83747021954bb3544332367f65af6b6fb7b80e5e887f861505
RUN set -eux; \
    curl -fsSL -o /tmp/prsice.zip \
      "https://github.com/choishingwan/PRSice/releases/download/${PRSICE_VERSION}/PRSice_linux.zip"; \
    echo "${PRSICE_SHA256}  /tmp/prsice.zip" | sha256sum -c -; \
    unzip -j -o /tmp/prsice.zip PRSice_linux PRSice.R -d /tmp/prsice; \
    install -m0755 /tmp/prsice/PRSice_linux /usr/local/bin/PRSice_linux; \
    install -m0644 /tmp/prsice/PRSice.R     /usr/local/share/PRSice.R; \
    echo "PRSice-2 ${PRSICE_VERSION} sha256=${PRSICE_SHA256}" >> /opt/provenance/binaries.sha256; \
    \
    curl -fsSL -o /tmp/gcta.zip \
      "https://yanglab.westlake.edu.cn/software/gcta/bin/gcta-${GCTA_VERSION}-linux-x86_64.zip"; \
    echo "${GCTA_SHA256}  /tmp/gcta.zip" | sha256sum -c -; \
    unzip -o /tmp/gcta.zip -d /tmp/gcta; \
    install -m0644 "/tmp/gcta/gcta-${GCTA_VERSION}-linux-x86_64/MIT_License.txt" \
                   /opt/provenance/gcta-MIT_License.txt; \
    echo "GCTA ${GCTA_VERSION} sha256=${GCTA_SHA256}" >> /opt/provenance/binaries.sha256; \
    \
    # The shipped gcta64 is an AppImage: it self-mounts via FUSE, which is not
    # available in a container (`No suitable fusermount binary found`), so B4
    # would fail at runtime. Unpack it once at build time instead. The inner
    # binary is only 4.7 MB but links its bundled Intel MKL through an
    # RPATH of $ORIGIN/../lib, so the whole usr/ tree has to come along.
    cd /tmp/gcta && "/tmp/gcta/gcta-${GCTA_VERSION}-linux-x86_64/gcta64" --appimage-extract >/dev/null; \
    mkdir -p /opt/gcta; \
    cp -a /tmp/gcta/squashfs-root/usr/bin /tmp/gcta/squashfs-root/usr/lib /opt/gcta/; \
    ln -sf /opt/gcta/bin/gcta64 /usr/local/bin/gcta64; \
    cd /; \
    # Fail the build here rather than hours into a pipeline run. gcta64 exits
    # 1 when given no options, so capture its banner before grepping rather
    # than piping into grep under `set -o pipefail`.
    gcta_banner="$(/usr/local/bin/gcta64 2>&1 || true)"; \
    printf '%s' "$gcta_banner" | grep -q "version v${GCTA_VERSION}"; \
    rm -rf /tmp/prsice* /tmp/gcta*

# ---------------------------------------------------------------------------
# Layer 5: prefetch runtime downloads so runs are hermetic
#
# C1:362 and C2:732 call neuromaps fetch_fslr(), which otherwise downloads
# fsLR surfaces from OSF on first use.
# ---------------------------------------------------------------------------
ENV NEUROMAPS_DATA=/opt/neuromaps-data \
    MPLCONFIGDIR=/opt/mplconfig \
    MPLBACKEND=Agg
RUN mkdir -p "${NEUROMAPS_DATA}" "${MPLCONFIGDIR}" \
 && python -c "from neuromaps.datasets import fetch_fslr; fetch_fslr()" \
 && python -c "import matplotlib.pyplot" \
 && chmod -R a+rwX "${NEUROMAPS_DATA}" "${MPLCONFIGDIR}" /opt/provenance

# ---------------------------------------------------------------------------
# Layer 6: entrypoint + code snapshot (changes most often -> last)
# ---------------------------------------------------------------------------
# Snakemake handles parallelism via --cores; pinning BLAS threads avoids
# oversubscription and removes float reduction-order nondeterminism.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PROJECT_DIR=/project \
    SNAKEMAKE_WORKDIR=/work

COPY docker/entrypoint.sh /usr/local/bin/pipeline-entrypoint
COPY docker/selftest.sh   /usr/local/bin/pipeline-selftest
RUN chmod 0755 /usr/local/bin/pipeline-entrypoint /usr/local/bin/pipeline-selftest

# Read-only provenance snapshot. The code that actually EXECUTES is the
# bind-mounted /project/code, because CODE_DIR derives from project_dir.
COPY code/ /opt/pipeline-src/code/

WORKDIR /project
ENTRYPOINT ["/usr/local/bin/pipeline-entrypoint"]
CMD []
