#!/bin/bash
set -e

# =============================================================================
# B4_extend_PGS_with_BLUP.sh
# Extend PGS to full sample using BLUP (Best Linear Unbiased Prediction)
# =============================================================================
#
# This script uses GCTA to:
# 1. Calculate a Genomic Relationship Matrix (GRM) for the full sample
# 2. Fit a Linear Mixed Model using PGS from unrelated individuals
# 3. Extract SNP effect sizes (BLUP solutions)
# 4. Calculate PGS for the full sample using BLUP-derived effects
#
# =============================================================================

usage() {
    echo "Usage: $0 --plink-dir <path> --pgs-file <path> --threshold-file <path> --gcta-path <path> --output <path>"
    echo ""
    echo "Arguments:"
    echo "  --plink-dir       Path to PLINK working directory"
    echo "  --pgs-file        Path to PGS scores file (all thresholds)"
    echo "  --threshold-file  Path to selected threshold file from B3"
    echo "  --gcta-path       Path to GCTA executable directory"
    echo "  --output          Path to output BLUP PGS profile file"
    echo "  --help            Show this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --plink-dir)
            PLINK_FOLDER="$2"
            shift 2
            ;;
        --pgs-file)
            PGS_FILE="$2"
            shift 2
            ;;
        --threshold-file)
            THRESHOLD_FILE="$2"
            shift 2
            ;;
        --gcta-path)
            GCTA_FOLDER="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PLINK_FOLDER" || -z "$PGS_FILE" || -z "$THRESHOLD_FILE" || -z "$GCTA_FOLDER" || -z "$OUTPUT_FILE" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

echo "============================================="
echo "B4: Extend PGS with BLUP"
echo "============================================="
echo "PLINK folder: $PLINK_FOLDER"
echo "PGS file: $PGS_FILE"
echo "Threshold file: $THRESHOLD_FILE"
echo "GCTA folder: $GCTA_FOLDER"
echo "Output file: $OUTPUT_FILE"
echo ""

# Read the selected threshold from B3
SELECTED_THRESHOLD=$(head -1 "$THRESHOLD_FILE")
echo "Selected threshold from B3: $SELECTED_THRESHOLD"

# Step 1: Check for z-scored PGS file from B3
echo "============================================="
echo "Step 1: Checking for z-scored PGS from B3..."

# B3 should have created unrelated_pgs_scores.txt with z-scored PGS values
# for individuals who have both PGS and social scores (the same sample used
# for threshold selection). This is critical for reproducibility.
if [[ -f "${PLINK_FOLDER}/unrelated_pgs_scores.txt" ]]; then
    echo "Found unrelated_pgs_scores.txt created by B2"
    echo "Using pre-computed z-scored PGS values (filtered to individuals with social scores)"
    echo "Individuals: $(wc -l < ${PLINK_FOLDER}/unrelated_pgs_scores.txt)"
else
    echo "WARNING: unrelated_pgs_scores.txt not found from B3"
    echo "Falling back to extracting from raw PGS file (may include individuals without social scores)"

    # Get the column number for the selected threshold
    HEADER=$(head -1 "$PGS_FILE")
    COL_NUM=$(echo "$HEADER" | tr ' ' '\n' | grep -n "^${SELECTED_THRESHOLD}$" | cut -d: -f1)

    if [[ -z "$COL_NUM" ]]; then
        echo "Error: Could not find threshold column $SELECTED_THRESHOLD in PGS file"
        exit 1
    fi

    echo "Found threshold at column $COL_NUM"

    # Extract FID, IID, and the selected PGS column
    awk -v col="$COL_NUM" 'NR==1 {next} {print $1, $2, $col}' "$PGS_FILE" > "${PLINK_FOLDER}/unrelated_pgs_scores_raw.txt"

    # Z-score normalize the PGS values
    echo "Z-score normalizing PGS values..."
    awk '{
        fid[NR] = $1
        iid[NR] = $2
        pgs[NR] = $3
        sum += $3
        n++
    }
    END {
        mean = sum / n
        for (i = 1; i <= n; i++) {
            sumsq += (pgs[i] - mean)^2
        }
        sd = sqrt(sumsq / (n - 1))
        for (i = 1; i <= n; i++) {
            zscore = (pgs[i] - mean) / sd
            print fid[i], iid[i], zscore
        }
    }' "${PLINK_FOLDER}/unrelated_pgs_scores_raw.txt" > "${PLINK_FOLDER}/unrelated_pgs_scores.txt"

    echo "Created phenotype file with $(wc -l < ${PLINK_FOLDER}/unrelated_pgs_scores.txt) individuals"
fi

# Step 2: Genomic Relationship Matrix (GRM) calculation for the full sample
echo "============================================="
echo "Step 2: Calculating Genomic Relationship Matrix..."

${GCTA_FOLDER}/gcta64 --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
    --autosome \
    --make-grm \
    --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_grm

# Step 3: Linear Mixed Model (LMM) using PGS from unrelated individuals
echo "============================================="
echo "Step 3: Fitting Linear Mixed Model..."

${GCTA_FOLDER}/gcta64 --grm ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_grm \
      --pheno ${PLINK_FOLDER}/unrelated_pgs_scores.txt \
      --reml \
      --reml-pred-rand \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_pgs_lmm

# Step 4: Extract SNP Effect Sizes from the Model (BLUP solutions for SNPs)
echo "============================================="
echo "Step 4: Extracting SNP effect sizes (BLUP solutions)..."

${GCTA_FOLDER}/gcta64 \
      --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --blup-snp ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_pgs_lmm.indi.blp \
      --autosome \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_snp_effects

# Step 5: Calculate the PGS for the full sample using BLUP-derived SNP effects
echo "============================================="
echo "Step 5: Calculating BLUP-extended PGS for full sample..."

# Determine output prefix from OUTPUT_FILE (remove .profile extension)
OUTPUT_PREFIX="${OUTPUT_FILE%.profile}"

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --score ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_snp_effects.snp.blp sum 1 2 3 \
      --out "$OUTPUT_PREFIX"

echo ""
echo "============================================="
echo "BLUP extension completed!"
echo "Output written to: $OUTPUT_FILE"
echo "============================================="
