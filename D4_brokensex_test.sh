#!/bin/bash
# =============================================================================
# D4_brokensex_test.sh
#
# Condition C of the A/B/C ancestry-vs-sex-filter test: re-run the PGS
# pipeline + C3 + C3b with the pre-d9554de broken sex filter re-enabled,
# to test whether the original "strong findings" were driven by chrX-F
# sex-check dropping HCP samples (per the project_broken_sex_filter_history
# memory note: corrupted female chrX -> F ~ 0.49 -> all females flagged
# as PROBLEM -> dropped).
#
# Pre-d9554de behaviour we reproduce:
#   - plink --check-sex (chrX F-statistic) with maleTh=0.8 / femaleTh=0.2
#   - filterSex = TRUE -> drop everyone with STATUS=PROBLEM
#   - No phenotype-CSV cross-check
#
# Implementation: run --check-sex on the current B1 .clean output,
# drop the PROBLEM samples, write .clean.brokensex.{bed,bim,fam},
# then feed B2-B5 + C3 + C3b. Outputs go alongside existing files
# with a "_brokensex" suffix; nothing in current state is overwritten.
#
# Prerequisites (same as D3):
#   data/plinkQC_output/{PREFIX}.clean.{bed,bim,fam}
#   results/pgs_residuals.csv         (with-ancestry baseline)
#   results/pgs_residuals_noancestry.csv  (D3 output)  [optional, for 3-way comp]
#   reports/C3*_report.txt, reports/C3b*_report.txt
#
# Usage:
#   bash D4_brokensex_test.sh
#   bash D4_brokensex_test.sh --skip-pgs   # if .clean.brokensex.* + B2-B5 done
# =============================================================================

set -e

PROJECT_DIR="/home/jmbathe/Documents/1_Projects/BrainCompensation"
CODE_DIR="$PROJECT_DIR/code"
DATA_DIR="$PROJECT_DIR/data"
GENETICS_INPUT_DIR="$DATA_DIR/raw_anonymised"
QCDIR="$DATA_DIR/plinkQC_output"
PLINK_DIR_BS="$DATA_DIR/PLINK_anonymised_brokensex"
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

mkdir -p "$PLINK_DIR_BS" "$LOGS_DIR"

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
        exit 1
    fi
done

echo "=========================================================================="
echo "D4: re-run with PRE-d9554de broken sex filter (chrX-F based --check-sex)"
echo "=========================================================================="

# ---- Phase 0: produce .clean.brokensex via chrX-F sex check ----
clean_pre="$QCDIR/${GENETICS_NAME_HG38}.clean"
clean_bs="$QCDIR/${GENETICS_NAME_HG38}.clean.brokensex"

if [[ "$SKIP_PGS" -eq 0 ]]; then
    echo ""
    echo "[0a/5] Running plink --check-sex on current .clean ..."
    plink --bfile "$clean_pre" --check-sex \
          --out "$clean_pre" \
          > "$LOGS_DIR/D4_chrx_sexcheck.log" 2>&1

    if [[ ! -f "$clean_pre.sexcheck" ]]; then
        echo "  ERROR: --check-sex did not produce $clean_pre.sexcheck"
        echo "  See $LOGS_DIR/D4_chrx_sexcheck.log"
        exit 1
    fi

    n_total=$(awk 'NR>1' "$clean_pre.sexcheck" | wc -l)
    n_ok=$(awk 'NR>1 && $5 == "OK"      {print}' "$clean_pre.sexcheck" | wc -l)
    n_prob=$(awk 'NR>1 && $5 == "PROBLEM" {print}' "$clean_pre.sexcheck" | wc -l)
    echo "  --check-sex result: $n_total total, $n_ok OK, $n_prob PROBLEM (will be dropped)"
    echo ""
    echo "  PROBLEM breakdown by self-reported PEDSEX:"
    awk 'NR>1 && $5 == "PROBLEM" {print "    PEDSEX=" $3 " (1=M, 2=F, 0=missing)"}' \
        "$clean_pre.sexcheck" | sort | uniq -c

    awk 'NR>1 && $5 == "PROBLEM" {print $1, $2}' "$clean_pre.sexcheck" \
        > "$QCDIR/${GENETICS_NAME_HG38}.brokensex_remove.IDs"

    echo ""
    echo "[0b/5] Producing .clean.brokensex by dropping PROBLEM samples..."
    plink --bfile "$clean_pre" \
          --remove "$QCDIR/${GENETICS_NAME_HG38}.brokensex_remove.IDs" \
          --make-bed \
          --out "$clean_bs" \
          > "$LOGS_DIR/D4_make_brokensex.log" 2>&1

    n_kept=$(wc -l < "$clean_bs.fam")
    echo "  $clean_bs.fam: $n_kept samples kept (vs $n_total in .clean)"

    # ---- Phase 1: B2-B5 with .clean.brokensex ----
    echo ""
    echo "[1/5] B2 (PGS, broken sex filter)..."
    bash "$CODE_DIR/B2_translate_PGS_to_HCP.sh" \
        --original-data "$GENETICS_INPUT_DIR" \
        --plink-dir "$PLINK_DIR_BS" \
        --data-dir "$DATA_DIR" \
        --code-dir "$CODE_DIR" \
        --phenotypic "$INPUT_PHEN" \
        --qcdir "$QCDIR" \
        --clean-name "${GENETICS_NAME_HG38}.clean.brokensex" \
        --output "$PLINK_DIR_BS/hcp_pgs_scores.profile" \
        > "$LOGS_DIR/D4_B2_brokensex.log" 2>&1

    echo "[2/5] B3 (PGS threshold)..."
    python "$CODE_DIR/B3_select_PGS_threshold.py" \
        --pgs "$PLINK_DIR_BS/hcp_pgs_scores.profile" \
        --phenotype "$SOCIAL_FILE" \
        --pca "$PLINK_DIR_BS/Neuro_Chip_full_sample_pca.eigenvec" \
        --output-plot "$FIGURES_DIR/B3_pgs_threshold_evaluation_brokensex.png" \
        --output-threshold "$RESULTS_DIR/pgs_selected_threshold_brokensex.txt" \
        --project "$PROJECT_DIR" \
        > "$LOGS_DIR/D4_B3_brokensex.log" 2>&1

    echo "[3/5] B4 (BLUP)..."
    bash "$CODE_DIR/B4_extend_PGS_with_BLUP.sh" \
        --plink-dir "$PLINK_DIR_BS" \
        --pgs-file "$PLINK_DIR_BS/hcp_pgs_scores.profile" \
        --threshold-file "$RESULTS_DIR/pgs_selected_threshold_brokensex.txt" \
        --gcta-path "$GCTA_PATH" \
        --output "$PLINK_DIR_BS/full_pgs_scores.snp.blp.profile" \
        > "$LOGS_DIR/D4_B4_brokensex.log" 2>&1

    echo "[4/5] B5 (BLUP eval)..."
    python "$CODE_DIR/B5_evalute_BLUP_prediction.py" \
        --blup-pgs "$PLINK_DIR_BS/full_pgs_scores.snp.blp.profile" \
        --original-pgs "$PLINK_DIR_BS/unrelated_pgs_scores.txt" \
        --social-scores "$SOCIAL_FILE" \
        --phenotypic "$INPUT_PHEN" \
        --behavioural "$INPUT_BEH" \
        --pca "$PLINK_DIR_BS/Neuro_Chip_full_sample_pca.eigenvec" \
        --output-residuals "$RESULTS_DIR/pgs_residuals_brokensex.csv" \
        --output-plot "$FIGURES_DIR/B5_blup_evaluation_brokensex.png" \
        --project "$PROJECT_DIR" \
        > "$LOGS_DIR/D4_B5_brokensex.log" 2>&1
else
    echo "[0-4/5] PGS pipeline skipped (--skip-pgs)"
fi

n_with=$(($(wc -l < "$RESULTS_DIR/pgs_residuals.csv") - 1))
n_without=$(if [[ -f "$RESULTS_DIR/pgs_residuals_noancestry.csv" ]]; then echo $(($(wc -l < "$RESULTS_DIR/pgs_residuals_noancestry.csv") - 1)); else echo "n/a"; fi)
n_brokensex=$(($(wc -l < "$RESULTS_DIR/pgs_residuals_brokensex.csv") - 1))
echo ""
echo "  PGS residuals N (with    ancestry filter, current main): $n_with"
echo "  PGS residuals N (no      ancestry filter, D3):           $n_without"
echo "  PGS residuals N (broken  sex filter,      D4):           $n_brokensex"

# ---- Phase 2: C3 + C3b on broken-sex residuals ----
echo ""
echo "[5/5] C3 + C3b on broken-sex residuals..."
python "$CODE_DIR/C3_perform_main_landscape_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals_brokensex.csv" \
    --social "$SOCIAL_FILE" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --thresholds 0.15 0.20 0.25 \
    --motion-threshold 0.2 \
    --output-suffix "_brokensex" \
    > "$LOGS_DIR/D4_C3_brokensex.log" 2>&1

python "$CODE_DIR/C3b_continuous_heteroscedasticity_analysis.py" \
    --project "$PROJECT_DIR" \
    --pgs "$RESULTS_DIR/pgs_residuals_brokensex.csv" \
    --social "$SOCIAL_FILE" \
    --behavioural "$INPUT_BEH" \
    --phenotypic "$INPUT_PHEN" \
    --movement "$INPUT_MOV" \
    --ids "$INPUT_IDS" \
    --matrices-dir "$MATRICES_DIR" \
    --partition "$PARTITION" \
    --threshold 0.2 \
    --motion-threshold 0.2 \
    --output-suffix "_brokensex" \
    > "$LOGS_DIR/D4_C3b_brokensex.log" 2>&1

# ---- Phase 3: 3-way side-by-side summary ----
echo ""
echo "=========================================================================="
echo "3-WAY COMPARISON SUMMARY"
echo "  WITH        = current SD-6 ancestry filter, sex-fix on (main)"
echo "  NO-ANC      = no ancestry filter,           sex-fix on (D3)"
echo "  BROKEN-SEX  = broken chrX-F sex filter (drops PROBLEM), no anc filter"
echo "                replicating pre-d9554de behaviour"
echo "=========================================================================="

extract_block () {
    grep -E "PGS group sizes|Final n =|n =|var ratio|Levene|Boot|MAIN|Pearson|coher|Bre|White|Quantile" "$1" \
        2>/dev/null | head -30
}

with_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report.txt"
noa_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report_noancestry.txt"
bs_c3="$REPORTS_DIR/C3_perform_main_landscape_analysis_report_brokensex.txt"

echo ""
echo "--- C3: main landscape analysis ---"
echo ""
echo "  WITH ancestry:";       extract_block "$with_c3" | sed 's/^/    /'
echo ""
echo "  NO ancestry (D3):";    extract_block "$noa_c3"  | sed 's/^/    /'
echo ""
echo "  BROKEN sex (D4):";     extract_block "$bs_c3"   | sed 's/^/    /'

with_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report.txt"
noa_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report_noancestry.txt"
bs_c3b="$REPORTS_DIR/C3b_continuous_heteroscedasticity_report_brokensex.txt"

echo ""
echo "--- C3b: continuous heteroscedasticity ---"
echo ""
echo "  WITH ancestry:";       extract_block "$with_c3b" | sed 's/^/    /'
echo ""
echo "  NO ancestry (D3):";    extract_block "$noa_c3b"  | sed 's/^/    /'
echo ""
echo "  BROKEN sex (D4):";     extract_block "$bs_c3b"   | sed 's/^/    /'

echo ""
echo "Full reports:"
echo "  $with_c3"
echo "  $noa_c3"
echo "  $bs_c3"
echo "  $with_c3b"
echo "  $noa_c3b"
echo "  $bs_c3b"
