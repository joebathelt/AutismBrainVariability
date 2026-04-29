#!/bin/bash
# =============================================================================
# D5_mixedsex_rerun.sh
#
# Re-run the main pipeline (A1 -> A2 -> B5 -> C3 -> C3b) on the full mixed-sex
# sample, after the male-only filters were removed from A1, B5, C3, C3b, C1
# (and Gender added back as a covariate to B5/C1).
#
# Backs up current (males-only) outputs with _malesonly suffix before
# overwriting with mixed-sex outputs. Prints a comparison block at the end.
#
# B2-B4 are NOT re-run: their outputs are sex-agnostic. The BLUP profile
# at PLINK_DIR/full_pgs_scores.snp.blp.profile already covers all 1123
# subjects post genotype QC; B5 was the first stage to filter males, and
# now no longer does.
#
# Prerequisites:
#   - Code edits in place (script verifies male filter is removed)
#   - Standard pipeline outputs from prior runs are on disk (for backup)
#   - data/PLINK_anonymised/full_pgs_scores.snp.blp.profile exists
#
# Usage:
#   bash D5_mixedsex_rerun.sh
# =============================================================================

set -e

PROJECT_DIR="/home/jmbathe/Documents/1_Projects/BrainCompensation"
CODE_DIR="$PROJECT_DIR/code"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results"
REPORTS_DIR="$PROJECT_DIR/reports"
FIGURES_DIR="$PROJECT_DIR/figures"
LOGS_DIR="$PROJECT_DIR/logs"
PLINK_DIR="$DATA_DIR/PLINK_anonymised"

INPUT_PHEN="$DATA_DIR/raw_anonymised/phenotypic_data_anonymised.csv"
INPUT_BEH="$DATA_DIR/raw_anonymised/behavioural_data_anonymised.csv"
INPUT_MOV="$DATA_DIR/raw_anonymised/movement_data_anonymised.csv"
INPUT_IDS="$DATA_DIR/raw_anonymised/subject_ids_anonymised.txt"
MATRICES_DIR="$DATA_DIR/HCP_PTN1200/netmats"
PARTITION="$RESULTS_DIR/C2b_selected_partition.csv"

mkdir -p "$LOGS_DIR" "$FIGURES_DIR" "$REPORTS_DIR" "$RESULTS_DIR"

# --- Sanity check: confirm male filter has been removed ---
echo "Verifying male filter has been removed from pipeline..."
for f in A1_preprocess_phenotypic_data.py B5_evalute_BLUP_prediction.py \
         C1_run_univariate_fMRI_prediction.py \
         C3_perform_main_landscape_analysis.py \
         C3b_continuous_heteroscedasticity_analysis.py; do
    if grep -qE "Gender.*== 'M'|Gender'\] == 'M'" "$CODE_DIR/$f" 2>/dev/null; then
        echo "  ERROR: male-only filter still in $f. Re-edit before re-running."
        exit 1
    fi
done
echo "  All clean."

# --- Sanity check: required upstream files exist ---
required=(
    "$PLINK_DIR/full_pgs_scores.snp.blp.profile"
    "$PLINK_DIR/unrelated_pgs_scores.txt"
    "$PLINK_DIR/Neuro_Chip_full_sample_pca.eigenvec"
    "$PARTITION"
    "$INPUT_BEH"
    "$INPUT_PHEN"
    "$INPUT_MOV"
    "$INPUT_IDS"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing prerequisite: $f"
        exit 1
    fi
done

# --- Backup current (males-only) outputs ---
echo ""
echo "Backing up current (males-only) outputs to *_malesonly.* ..."

backup_if_exists () {
    if [[ -f "$1" ]]; then
        # Insert _malesonly before the extension
        ext="${1##*.}"
        base="${1%.*}"
        backup="${base}_malesonly.${ext}"
        cp "$1" "$backup"
        echo "  $(basename "$1") -> $(basename "$backup")"
    fi
}

backup_if_exists "$RESULTS_DIR/behavioural_data_preprocessed.csv"
backup_if_exists "$RESULTS_DIR/cfa_factor_scores_full_sample.csv"
backup_if_exists "$RESULTS_DIR/pgs_residuals.csv"
backup_if_exists "$REPORTS_DIR/A1_preprocess_phenotypic_data_report.txt"
backup_if_exists "$REPORTS_DIR/A2_factor_analysis_report.txt"
backup_if_exists "$REPORTS_DIR/B5_evaluate_blup_prediction_report.txt"
backup_if_exists "$REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
backup_if_exists "$REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
backup_if_exists "$RESULTS_DIR/C3_graph_theory_landscape_results.csv"
backup_if_exists "$RESULTS_DIR/C4_main_network_metrics.csv"
backup_if_exists "$RESULTS_DIR/C4_sensitivity_summary.csv"
backup_if_exists "$RESULTS_DIR/C3b_heteroscedasticity_results.csv"
backup_if_exists "$RESULTS_DIR/C3b_quantile_regression_coefficients.csv"
backup_if_exists "$RESULTS_DIR/C3b_decile_variability.csv"

echo ""
echo "=========================================================================="
echo "Re-running pipeline on the full mixed-sex sample"
echo "=========================================================================="

# --- 1) A1: preprocess phenotypic data (mixed-sex) ---
echo ""
echo "[1/5] A1 (mixed-sex behaviour preprocessing)..."
python "$CODE_DIR/A1_preprocess_phenotypic_data.py" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --output "$RESULTS_DIR/behavioural_data_preprocessed.csv" \
    --project "$PROJECT_DIR" \
    --figure "$FIGURES_DIR/A1_Behaviour_correlations.png" \
    > "$LOGS_DIR/D5_A1_mixedsex.log" 2>&1
echo "  N preprocessed: $(($(wc -l < "$RESULTS_DIR/behavioural_data_preprocessed.csv") - 1))"

# --- 2) A2: factor analysis ---
echo "[2/5] A2 (factor analysis on mixed-sex)..."
Rscript "$CODE_DIR/A2_factor_analysis.R" \
    --input "$RESULTS_DIR/behavioural_data_preprocessed.csv" \
    --output "$RESULTS_DIR/cfa_factor_scores_full_sample.csv" \
    --project "$PROJECT_DIR" \
    > "$LOGS_DIR/D5_A2_mixedsex.log" 2>&1
echo "  N factor scores: $(($(wc -l < "$RESULTS_DIR/cfa_factor_scores_full_sample.csv") - 1))"

# --- 3) B5: BLUP residualisation (mixed-sex; uses existing B4 BLUP profile) ---
echo "[3/5] B5 (BLUP residuals, Gender added as covariate)..."
python "$CODE_DIR/B5_evalute_BLUP_prediction.py" \
    --blup-pgs "$PLINK_DIR/full_pgs_scores.snp.blp.profile" \
    --original-pgs "$PLINK_DIR/unrelated_pgs_scores.txt" \
    --social-scores "$RESULTS_DIR/cfa_factor_scores_full_sample.csv" \
    --phenotypic "$INPUT_PHEN" \
    --behavioural "$INPUT_BEH" \
    --pca "$PLINK_DIR/Neuro_Chip_full_sample_pca.eigenvec" \
    --output-residuals "$RESULTS_DIR/pgs_residuals.csv" \
    --output-plot "$FIGURES_DIR/B5_blup_evaluation.png" \
    --project "$PROJECT_DIR" \
    > "$LOGS_DIR/D5_B5_mixedsex.log" 2>&1
echo "  N PGS residuals: $(($(wc -l < "$RESULTS_DIR/pgs_residuals.csv") - 1))"

# --- 4) C3: main landscape analysis ---
echo "[4/5] C3 (mixed-sex landscape, Gender residualised in COVARIATES)..."
python "$CODE_DIR/C3_perform_main_landscape_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals.csv" \
    --social "$RESULTS_DIR/cfa_factor_scores_full_sample.csv" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --thresholds 0.15 0.20 0.25 \
    --motion-threshold 0.2 \
    > "$LOGS_DIR/D5_C3_mixedsex.log" 2>&1

# --- 5) C3b: continuous heteroscedasticity ---
echo "[5/5] C3b (mixed-sex heteroscedasticity)..."
python "$CODE_DIR/C3b_continuous_heteroscedasticity_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals.csv" \
    --social "$RESULTS_DIR/cfa_factor_scores_full_sample.csv" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --threshold 0.2 \
    --motion-threshold 0.2 \
    > "$LOGS_DIR/D5_C3b_mixedsex.log" 2>&1

# --- Comparison ---
echo ""
echo "=========================================================================="
echo "MALES-ONLY (backup, prior code) vs MIXED-SEX (new, current code)"
echo "=========================================================================="

n_old=$(if [[ -f "$RESULTS_DIR/pgs_residuals_malesonly.csv" ]]; then echo $(($(wc -l < "$RESULTS_DIR/pgs_residuals_malesonly.csv") - 1)); else echo "n/a"; fi)
n_new=$(($(wc -l < "$RESULTS_DIR/pgs_residuals.csv") - 1))
echo ""
echo "  PGS residuals N (males-only, backup):  $n_old"
echo "  PGS residuals N (mixed-sex, new):      $n_new"

extract_block () {
    grep -E "PGS group sizes|Final n =|n =|var ratio|Levene|Boot|MAIN|Pearson|coher|Bre|White|Quantile" "$1" \
        2>/dev/null | head -30
}

old_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report_malesonly.txt"
new_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
echo ""
echo "--- C3: main landscape analysis ---"
echo ""
echo "  MALES-ONLY (backup):"
extract_block "$old_c3" | sed 's/^/    /'
echo ""
echo "  MIXED-SEX (new):"
extract_block "$new_c3" | sed 's/^/    /'

old_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report_malesonly.txt"
new_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
echo ""
echo "--- C3b: continuous heteroscedasticity ---"
echo ""
echo "  MALES-ONLY (backup):"
extract_block "$old_c3b" | sed 's/^/    /'
echo ""
echo "  MIXED-SEX (new):"
extract_block "$new_c3b" | sed 's/^/    /'

echo ""
echo "Reports:"
echo "  $old_c3"
echo "  $new_c3"
echo "  $old_c3b"
echo "  $new_c3b"
