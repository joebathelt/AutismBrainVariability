#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entrypoint: prepare the environment, then exec snakemake.
# Every argument passed to `docker run` is forwarded to snakemake verbatim,
# so `-n`, `--forcerun X`, `clean`, or an explicit target all work.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project}"
WORKDIR="${SNAKEMAKE_WORKDIR:-/work}"
CORES="${CORES:-$(nproc)}"
MEM_MB="${MEM_MB:-}"
GCTA_PATH="${GCTA_PATH:-/usr/local/bin}"

# --- escape hatches --------------------------------------------------------
case "${1:-}" in
  bash|sh)     exec "$@" ;;
  --selftest)  exec /usr/local/bin/pipeline-selftest ;;
esac

# --- guardrail -------------------------------------------------------------
# The environment is baked into the image. code/environment.yml is an
# Apple-Silicon export that cannot solve on linux-64, so --use-conda would
# fail. Without that flag snakemake never even reads the file, which is why
# the rules' `conda:` directives are inert and needed no edits.
for a in "$@"; do
  case "$a" in
    --use-conda|--use-singularity|--use-apptainer|--software-deployment-method)
      cat >&2 <<MSG
ERROR: '$a' is not supported in this image.

The full environment is already baked in. code/environment.yml is a macOS-ARM
conda export that will not solve on linux-64; the rules' 'conda:' directives
are inert because --use-conda is never passed.
MSG
      exit 2 ;;
  esac
done

# --- sanity: is the project mounted, with reachable data? ------------------
if [ ! -d "$PROJECT_DIR/code" ]; then
  echo "ERROR: $PROJECT_DIR/code not found - bind-mount the project at $PROJECT_DIR." >&2
  echo "       Use docker/run.sh, which wires up all the mounts for you." >&2
  exit 2
fi

missing=0
for f in "$PROJECT_DIR/data/raw_anonymised/behavioural_data_anonymised.csv" \
         "$PROJECT_DIR/data/raw_anonymised/phenotypic_data_anonymised.csv" \
         "$PROJECT_DIR/data/raw_anonymised/Neuro_Chip_anonymised.bed" \
         "$PROJECT_DIR/data/raw_anonymised/iPSYCH_PGC_ASD_Nov_2017.gz" \
         "$PROJECT_DIR/data/HCP_PTN1200/netmats"; do
  if [ ! -e "$f" ]; then echo "MISSING INPUT: $f" >&2; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo "" >&2
  echo "  data/ contains symlinks pointing outside the repo. Docker does not" >&2
  echo "  follow those across a bind mount - docker/run.sh resolves them and" >&2
  echo "  mounts each target at its own absolute path. See docker/README.md." >&2
fi

# --- writable scratch ------------------------------------------------------
# We run as the caller's UID, which has no /etc/passwd entry, so anything that
# resolves ~ needs an explicit HOME.
export HOME="${HOME:-$WORKDIR/home}"
export MPLCONFIGDIR="$WORKDIR/.mplconfig"
export XDG_CACHE_HOME="$WORKDIR/.cache"
export TMPDIR="${TMPDIR:-/tmp}"
export NEUROMAPS_DATA="${NEUROMAPS_DATA:-/opt/neuromaps-data}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
# Reuse the font cache built at image time rather than rebuilding it per run.
cp -rn /opt/mplconfig/. "$MPLCONFIGDIR"/ 2>/dev/null || true

# --- directories the scripts assume exist ----------------------------------
mkdir -p "$PROJECT_DIR"/results \
         "$PROJECT_DIR"/reports \
         "$PROJECT_DIR"/figures \
         "$PROJECT_DIR"/logs \
         "$PROJECT_DIR"/manuscript/figures \
         "$PROJECT_DIR"/data/PLINK_anonymised \
         "$PROJECT_DIR"/data/plinkQC_output

# --- build the snakemake invocation ----------------------------------------
#  --directory  : keeps .snakemake/ in the container-local /work mount, so the
#                 host's code/.snakemake state (whose metadata filenames are
#                 base64 of /home/jmbathe/... absolute paths) is never touched.
#                 Safe because every rule path is absolute, derived from
#                 PROJECT_DIR.
#  --configfile : required alongside --directory. The Snakefile's bare
#                 `configfile: "config.yaml"` resolves against the CWD, which
#                 --directory changes; passing --configfile takes the branch
#                 that skips the missing-file error.
#  --config     : overrides only the two host-specific keys, leaving
#                 code/config.yaml untouched for the host workflow.
SM=( snakemake
     --snakefile  "$PROJECT_DIR/code/Snakefile"
     --directory  "$WORKDIR"
     --configfile "$PROJECT_DIR/code/config.yaml"
     --config     "project_dir=$PROJECT_DIR" "gcta_path=$GCTA_PATH"
     --cores      "$CORES"
     --rerun-incomplete
     --printshellcmds
     --latency-wait 30 )
if [ -n "$MEM_MB" ]; then SM+=( --resources "mem_mb=$MEM_MB" ); fi

# --- release a lock left behind by a killed container ----------------------
if compgen -G "$WORKDIR/.snakemake/locks/*" >/dev/null 2>&1; then
  echo "[entrypoint] releasing stale snakemake lock in $WORKDIR"
  "${SM[@]}" --unlock >/dev/null 2>&1 || true
fi

echo "[entrypoint] snakemake $(snakemake --version) | cores=$CORES | project=$PROJECT_DIR | state=$WORKDIR/.snakemake"
exec "${SM[@]}" "$@"
