#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Environment smoke test. Verifies every dependency the pipeline needs is
# present and working, without touching any data.
#
#   ./docker/run.sh --selftest
#
# Run it with `--network none` added to the docker run line to prove the image
# is hermetic (no runtime CRAN or OSF fetches).
# ---------------------------------------------------------------------------
set -uo pipefail

FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

export HOME="${HOME:-/tmp/selftest-home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/selftest-mpl}"
mkdir -p "$HOME" "$MPLCONFIGDIR"

echo "=== workflow engine ==="
echo "  snakemake $(snakemake --version 2>&1)"
check "snakemake is 7.x"          "snakemake --version | grep -q '^7\.'"
# snakemake 7 crashes against pulp >= 2.8 (snakemake/snakemake#2606)
check "pulp < 2.8"                "python -c \"import pulp,sys; sys.exit(0 if tuple(int(x) for x in pulp.__version__.split('.')[:2]) < (2,8) else 1)\""

echo
echo "=== python packages ==="
for m in numpy pandas scipy matplotlib seaborn sklearn statsmodels \
         nibabel nilearn bct networkx pingouin cmasher \
         neuromaps surfplot brainspace vtk; do
  check "import $m" "python -c 'import $m'"
done

echo
echo "=== R packages ==="
for p in plinkQC lavaan psych dplyr tibble ggplot2 data.table magrittr R.utils; do
  check "library($p)" "Rscript -e 'suppressPackageStartupMessages(library($p))'"
done
echo "  lavaan $(Rscript -e 'cat(as.character(packageVersion("lavaan")))' 2>/dev/null) (logs/A2 recorded 0.6-21)"

echo
echo "=== external binaries ==="
# These must actually EXECUTE, not merely be on PATH. GCTA in particular ships
# as a FUSE-mounted AppImage that is present but unrunnable in a container
# unless unpacked at build time, and B4 would only fail hours into a run.
check "plink runs"        "plink --version"
# gcta64 and PRSice_linux both exit non-zero when invoked with no real job,
# so capture their banners instead of piping under `set -o pipefail`.
check "gcta64 runs"       "printf '%s' \"\$(gcta64 2>&1 || true)\" | grep -q 'Genome-wide Complex Trait Analysis'"
check "PRSice_linux runs" "printf '%s' \"\$(PRSice_linux --version 2>&1 || true)\" | grep -qE '[0-9]+\\.[0-9]+'"
check "xvfb-run on PATH"  "command -v xvfb-run"
check "pdflatex runs"     "pdflatex --version"
check "dvipng runs"       "dvipng --version"
check "gs runs"           "gs --version"
check "gawk runs"         "gawk --version"
echo "  $(plink --version 2>&1 | head -1)   (logs/B2_translate_pgs.log recorded v1.9.0-b.8)"
echo "  GCTA $(gcta64 2>&1 | grep -oE 'version v[0-9.]+ Linux' | head -1)"
echo "  PRSice $(PRSice_linux --version 2>&1 | head -1)"

echo
echo "=== matplotlib usetex (A1:20, A3:14, C1:53) ==="
if python - <<'PY' >/dev/null 2>&1
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams, pyplot as plt
rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
ax.set_xlabel(r"$r^2$")
fig.savefig("/tmp/selftest_usetex.png", dpi=72)
PY
then ok "usetex render"; else bad "usetex render (needs latex + dvipng + ghostscript)"; fi

echo
echo "=== offscreen surface rendering (C1:339, C2:377 use xvfb-run) ==="
if xvfb-run -a python - <<'PY' >/dev/null 2>&1
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
surfaces = fetch_fslr()
lh, rh = surfaces["inflated"]
p = Plot(lh, rh, size=(400, 300))
p.build().savefig("/tmp/selftest_surf.png", dpi=72)
PY
then ok "xvfb + surfplot + neuromaps fsLR"; else bad "xvfb/surfplot render (check OpenGL libs and the prefetched fsLR data)"; fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed."
else
  echo "Some checks FAILED (see above)." >&2
fi
exit "$FAIL"
