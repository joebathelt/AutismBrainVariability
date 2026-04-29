#!/bin/bash
# =============================================================================
# D3_ab_ancestry_test.sh
#
# A/B test: re-run B2-B5 + C3 + C3b without ancestry filtering, alongside
# the existing ancestry-filtered run, to disentangle whether the
# loss of strong findings at C3/C3b is driven by:
#   (a) the ancestry filter shrinking N (power loss), or
#   (b) other QC fixes (e.g. sex-filter) that removed a chopstick artefact
#
# The standard pipeline (with current SD-6 within-sample-PCA ancestry filter)
# must already have been run before invoking this script. This script does
# NOT touch any of the ancestry-filtered outputs; it produces a parallel set
# of outputs with "_noancestry" suffix on key files / a separate
# data/PLINK_anonymised_noancestry/ working directory.
#
# Inputs consumed (from the standard run):
#   data/plinkQC_output/{PREFIX}.clean.{bed,bim,fam}    (B1 output, no anc.)
#   results/cfa_factor_scores_full_sample.csv           (A2 output)
#   results/C2b_selected_partition.csv                  (C2b output)
#   data/raw_anonymised/{phenotypic,behavioural,
#                        movement,subject_ids}*         (raw inputs)
#   reports/C3_perform_main_landscape_analysis_report.txt        (with-anc.)
#   reports/C3b_continuous_heteroscedasticity_report.txt         (with-anc.)
#
# Outputs produced (all marked _noancestry to avoid collisions):
#   data/PLINK_anonymised_noancestry/                   (parallel PLINK dir)
#   results/pgs_residuals_noancestry.csv
#   results/pgs_selected_threshold_noancestry.txt
#   reports/C3_perform_main_landscape_analysis_report_noancestry.txt
#   reports/C3b_continuous_heteroscedasticity_report_noancestry.txt
#   results/C3_graph_theory_landscape_results_noancestry.csv
#   results/C4_*_noancestry.csv
#   results/C3b_*_noancestry.csv
#   figures/B3_pgs_threshold_evaluation_noancestry.png
#   figures/B5_blup_evaluation_noancestry.png
#   figures/C3_landscape_theory_graph_analysis_noancestry.png
#   figures/C4_sensitivity_analysis_noancestry.png
#   figures/C3b_continuous_heteroscedasticity_noancestry.png
#
# Usage:
#   bash D3_ab_ancestry_test.sh
#   bash D3_ab_ancestry_test.sh --skip-pgs   # if no-anc. PGS already produced
# =============================================================================

set -e

PROJECT_DIR="/home/jmbathe/Documents/1_Projects/BrainCompensation"
CODE_DIR="$PROJECT_DIR/code"
DATA_DIR="$PROJECT_DIR/data"
GENETICS_INPUT_DIR="$DATA_DIR/raw_anonymised"
PLINK_DIR_NOA="$DATA_DIR/PLINK_anonymised_noancestry"
QCDIR="$DATA_DIR/plinkQC_output"
RESULTS_DIR="$PROJECT_DIR/results"
REPORTS_DIR="$PROJECT_DIR/reports"
FIGURES_DIR="$PROJECT_DIR/figures"
LOGS_DIR="$PROJECT_DIR/logs"
GENETICS_NAME_HG38="Neuro_Chip_anonymised_hg38"
GCTA_PATH="/usr/local/bin"

INPUT_PHEN="$DATA_DIR/raw_anonymised/phenotypic_data_anonymised.csv"
INPUT_BEH="$DATA_DIR/raw_anonymised/behavioural_data_anonymised.csv"
INPUT_MOV="$DATA_DIR/raw_anonymised/movement_data_anonymised.csv"
INPUT_IDS="$DATA_DIR/raw_anonymised/subject_ids_anonymised.txt"
MATRICES_DIR="$DATA_DIR/HCP_PTN1200/netmats"
PARTITION="$RESULTS_DIR/C2b_selected_partition.csv"
SOCIAL_FILE="$RESULTS_DIR/cfa_factor_scores_full_sample.csv"

SKIP_PGS=0
for a in "$@"; do
    case "$a" in
        --skip-pgs) SKIP_PGS=1 ;;
        *) echo "Unknown arg: $a"; exit 1 ;;
    esac
done

mkdir -p "$PLINK_DIR_NOA" "$LOGS_DIR"

# --- Sanity checks ---
required=(
    "$QCDIR/${GENETICS_NAME_HG38}.clean.bed"
    "$QCDIR/${GENETICS_NAME_HG38}.clean.bim"
    "$QCDIR/${GENETICS_NAME_HG38}.clean.fam"
    "$RESULTS_DIR/pgs_residuals.csv"
    "$REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
    "$REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
    "$PARTITION"
    "$SOCIAL_FILE"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing prerequisite: $f"
        echo "Run the standard ancestry-filtered pipeline first."
        exit 1
    fi
done

echo "=========================================================================="
echo "A/B ancestry test"
echo "  WITH-ancestry baseline already exists at:"
echo "    $RESULTS_DIR/pgs_residuals.csv"
echo "    $REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
echo "    $REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
echo "  Producing NO-ancestry parallel set in:"
echo "    $PLINK_DIR_NOA"
echo "    *_noancestry.csv / *_noancestry.txt under results/ and reports/"
echo "=========================================================================="

# ---- Phase 1: re-run B2-B5 with .clean (no ancestry filter) ----
if [[ "$SKIP_PGS" -eq 0 ]]; then
    echo ""
    echo "[1/5] B2 (PGS calculation, no ancestry filter)..."
    bash "$CODE_DIR/B2_translate_PGS_to_HCP.sh" \
        --original-data "$GENETICS_INPUT_DIR" \
        --plink-dir "$PLINK_DIR_NOA" \
        --data-dir "$DATA_DIR" \
        --code-dir "$CODE_DIR" \
        --phenotypic "$INPUT_PHEN" \
        --qcdir "$QCDIR" \
        --clean-name "${GENETICS_NAME_HG38}.clean" \
        --output "$PLINK_DIR_NOA/hcp_pgs_scores.profile" \
        > "$LOGS_DIR/D3_B2_noancestry.log" 2>&1
    echo "    log: $LOGS_DIR/D3_B2_noancestry.log"

    echo "[2/5] B3 (PGS threshold selection)..."
    python "$CODE_DIR/B3_select_PGS_threshold.py" \
        --pgs "$PLINK_DIR_NOA/hcp_pgs_scores.profile" \
        --phenotype "$SOCIAL_FILE" \
        --pca "$PLINK_DIR_NOA/Neuro_Chip_full_sample_pca.eigenvec" \
        --output-plot "$FIGURES_DIR/B3_pgs_threshold_evaluation_noancestry.png" \
        --output-threshold "$RESULTS_DIR/pgs_selected_threshold_noancestry.txt" \
        --project "$PROJECT_DIR" \
        > "$LOGS_DIR/D3_B3_noancestry.log" 2>&1

    echo "[3/5] B4 (BLUP extension)..."
    bash "$CODE_DIR/B4_extend_PGS_with_BLUP.sh" \
        --plink-dir "$PLINK_DIR_NOA" \
        --pgs-file "$PLINK_DIR_NOA/hcp_pgs_scores.profile" \
        --threshold-file "$RESULTS_DIR/pgs_selected_threshold_noancestry.txt" \
        --gcta-path "$GCTA_PATH" \
        --output "$PLINK_DIR_NOA/full_pgs_scores.snp.blp.profile" \
        > "$LOGS_DIR/D3_B4_noancestry.log" 2>&1

    echo "[4/5] B5 (BLUP evaluation, residualised PGS)..."
    python "$CODE_DIR/B5_evalute_BLUP_prediction.py" \
        --blup-pgs "$PLINK_DIR_NOA/full_pgs_scores.snp.blp.profile" \
        --original-pgs "$PLINK_DIR_NOA/unrelated_pgs_scores.txt" \
        --social-scores "$SOCIAL_FILE" \
        --phenotypic "$INPUT_PHEN" \
        --behavioural "$INPUT_BEH" \
        --pca "$PLINK_DIR_NOA/Neuro_Chip_full_sample_pca.eigenvec" \
        --output-residuals "$RESULTS_DIR/pgs_residuals_noancestry.csv" \
        --output-plot "$FIGURES_DIR/B5_blup_evaluation_noancestry.png" \
        --project "$PROJECT_DIR" \
        > "$LOGS_DIR/D3_B5_noancestry.log" 2>&1
else
    echo "[1-4/5] PGS pipeline skipped (--skip-pgs)"
fi

n_with=$(($(wc -l < "$RESULTS_DIR/pgs_residuals.csv") - 1))
n_without=$(($(wc -l < "$RESULTS_DIR/pgs_residuals_noancestry.csv") - 1))
echo ""
echo "  PGS residuals N (with    ancestry filter): $n_with"
echo "  PGS residuals N (without ancestry filter): $n_without"

# ---- Phase 2: re-run C3 + C3b on no-ancestry residuals ----
echo ""
echo "[5/5] C3 + C3b on no-ancestry residuals..."
python "$CODE_DIR/C3_perform_main_landscape_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals_noancestry.csv" \
    --social "$SOCIAL_FILE" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --thresholds 0.15 0.20 0.25 \
    --motion-threshold 0.2 \
    --output-suffix "_noancestry" \
    > "$LOGS_DIR/D3_C3_noancestry.log" 2>&1

python "$CODE_DIR/C3b_continuous_heteroscedasticity_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals_noancestry.csv" \
    --social "$SOCIAL_FILE" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --threshold 0.2 \
    --motion-threshold 0.2 \
    --output-suffix "_noancestry" \
    > "$LOGS_DIR/D3_C3b_noancestry.log" 2>&1

# ---- Phase 3: side-by-side summary ----
echo ""
echo "=========================================================================="
echo "COMPARISON SUMMARY"
echo "  Files annotated 'WITH'  = current SD-6 ancestry-filtered pipeline."
echo "  Files annotated 'NO  '  = same code, B2-B5 fed .clean (no anc filter)."
echo "=========================================================================="

extract_block () {
    # Extract the variability/heteroscedasticity-relevant lines from a report.
    # Tolerant of missing patterns: just show what's present.
    grep -E "PGS group sizes|Final n =|n =|var ratio|Levene|Boot|MAIN|Pearson|coher|Bre|White|Quantile" "$1" \
        2>/dev/null | head -30
}

with_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
without_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report_noancestry.txt"
echo ""
echo "--- C3: main landscape analysis ---"
echo ""
echo "  WITH ancestry filter:"
extract_block "$with_c3" | sed 's/^/    /'
echo ""
echo "  NO ancestry filter:"
extract_block "$without_c3" | sed 's/^/    /'

with_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
without_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report_noancestry.txt"
echo ""
echo "--- C3b: continuous heteroscedasticity ---"
echo ""
echo "  WITH ancestry filter:"
extract_block "$with_c3b" | sed 's/^/    /'
echo ""
echo "  NO ancestry filter:"
extract_block "$without_c3b" | sed 's/^/    /'

echo ""
echo "Full reports:"
echo "  $with_c3"
echo "  $without_c3"
echo "  $with_c3b"
echo "  $without_c3b"
