#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command reproduction of the BrainCompensation pipeline.
#
#   ./docker/run.sh                 build if needed, then run the whole pipeline
#   ./docker/run.sh -n              dry run (show what would execute)
#   ./docker/run.sh clean           snakemake's `clean` rule - wipe outputs
#   ./docker/run.sh --selftest      verify the environment, touch no data
#   ./docker/run.sh bash            interactive shell in the container
#   ./docker/run.sh <target>        build one target, e.g.
#                                   /project/results/pgs_residuals.csv
#
# Environment overrides:
#   CORES=16 MEM_MB=64000 ./docker/run.sh
#   IMAGE=braincomp:v1 ./docker/run.sh
#   SKIP_BUILD=1 ./docker/run.sh     reuse the existing image
#   DOCKER_RUN_EXTRA="--network none" ./docker/run.sh --selftest
# ---------------------------------------------------------------------------
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-braincomp:latest}"
CORES="${CORES:-$(nproc)}"

# --- build -----------------------------------------------------------------
if [ -z "${SKIP_BUILD:-}" ]; then
  echo "==> Building $IMAGE (context: $PROJECT_DIR, see .dockerignore)"
  docker build --platform linux/amd64 \
    -f "$HERE/Dockerfile" -t "$IMAGE" "$PROJECT_DIR"
fi

MOUNTS=( -v "$PROJECT_DIR:/project" )

# --- resolve symlinked data ------------------------------------------------
# data/raw_anonymised, data/HCP_PTN1200 and data/hcp_behavioural_raw.csv are
# absolute symlinks into a sibling checkout. Docker does not follow host
# symlinks out of a bind mount, so they would dangle inside the container.
# Mount each resolved target at its OWN absolute path and the symlink resolves
# naturally. Read-only: the pipeline only reads these (B2 and
# prefilter_genotypes_by_sex copy outward into data/PLINK_anonymised).
declare -A SEEN=()
if [ -d "$PROJECT_DIR/data" ]; then
  while IFS= read -r link; do
    tgt="$(readlink -f "$link" 2>/dev/null || true)"
    if [ -n "$tgt" ] && [ -e "$tgt" ] && [ -z "${SEEN[$tgt]:-}" ]; then
      SEEN[$tgt]=1
      MOUNTS+=( -v "$tgt:$tgt:ro" )
      echo "==> Data mount: $tgt (ro)  <- $(basename "$link")"
    elif [ -n "$tgt" ] && [ ! -e "$tgt" ]; then
      echo "WARNING: dangling symlink $link -> $tgt" >&2
    fi
  done < <(find "$PROJECT_DIR/data" -maxdepth 1 -type l 2>/dev/null)
fi

# --- container-local snakemake state ---------------------------------------
# Keeps the host's code/.snakemake (stale absolute-path metadata) untouched.
mkdir -p "$PROJECT_DIR/.docker/work/home"
MOUNTS+=( -v "$PROJECT_DIR/.docker/work:/work" )

TTY=()
if [ -t 0 ] && [ -t 1 ]; then TTY=( -it ); fi

# --- file ownership --------------------------------------------------------
# Rootful Docker: the container runs as root by default, so outputs would land
# on the host owned by root. Pass --user to write them as the caller instead.
# Rootless Docker: container-root ALREADY maps to the invoking host user, so
# --user would map to an unrelated subuid and every write fails with EACCES.
# Detect which daemon we are talking to rather than guessing.
USER_ARGS=()
if [ -n "${DOCKER_USER:-}" ]; then
  USER_ARGS=( --user "$DOCKER_USER" )
elif docker info --format '{{join .SecurityOptions ","}}' 2>/dev/null | grep -q 'name=rootless'; then
  echo "==> Rootless Docker detected: running as container-root (maps to $(id -un) on the host)"
else
  USER_ARGS=( --user "$(id -u):$(id -g)" )
fi

# shellcheck disable=SC2206
EXTRA=( ${DOCKER_RUN_EXTRA:-} )

echo "==> Running $IMAGE (cores=$CORES)"
exec docker run --rm "${TTY[@]}" --platform linux/amd64 \
  "${USER_ARGS[@]}" \
  --shm-size=2g \
  -e HOME=/work/home \
  -e CORES="$CORES" \
  -e MEM_MB="${MEM_MB:-}" \
  "${MOUNTS[@]}" "${EXTRA[@]}" \
  "$IMAGE" "$@"
