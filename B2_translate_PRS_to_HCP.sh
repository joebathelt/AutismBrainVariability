#!/bin/bash
set -e

# =============================================================================
# B2_translate_PRS_to_HCP.sh
# SNP harmonization, PCA, relatedness filtering, and PRS calculation
#
# Starts from B1 plinkQC clean output (Neuro_Chip_anonymised.clean.bed/bim/fam).
# Genotype QC (MAF, HWE, missingness, sex check, heterozygosity, ancestry)
# is handled by B1_plinkQC_genotype_qc.R. This script handles PRS-specific
# steps: harmonizing SNP alleles with GWAS summary stats, computing PCA,
# filtering to unrelated individuals, and running PRSice.
# =============================================================================

usage() {
    echo "Usage: $0 --original-data <path> --plink-dir <path> --data-dir <path> --code-dir <path> --phenotypic <path> --output <path> --qcdir <path>"
    echo ""
    echo "Arguments:"
    echo "  --original-data   Path to original genetic data directory (for GWAS files)"
    echo "  --plink-dir       Path to PLINK working directory"
    echo "  --data-dir        Path to data directory"
    echo "  --code-dir        Path to code directory"
    echo "  --phenotypic      Path to phenotypic data CSV file"
    echo "  --output          Path to output PRS profile file"
    echo "  --qcdir           Path to B1 plinkQC output directory"
    echo "  --clean-name      B1 clean file prefix (default: Neuro_Chip_anonymised.clean)"
    echo "  --help            Show this help message"
    exit 1
}

# Parse command-line arguments
CLEAN_NAME="Neuro_Chip_anonymised.clean"

while [[ $# -gt 0 ]]; do
    case $1 in
        --original-data)
            ORIGINAL_DATA="$2"
            shift 2
            ;;
        --plink-dir)
            PLINK_FOLDER="$2"
            shift 2
            ;;
        --data-dir)
            DATA_FOLDER="$2"
            shift 2
            ;;
        --code-dir)
            CODE_FOLDER="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --phenotypic)
            PHENOTYPIC_FILE="$2"
            shift 2
            ;;
        --qcdir)
            QCDIR="$2"
            shift 2
            ;;
        --clean-name)
            CLEAN_NAME="$2"
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
if [[ -z "$ORIGINAL_DATA" || -z "$PLINK_FOLDER" || -z "$DATA_FOLDER" || -z "$CODE_FOLDER" || -z "$PHENOTYPIC_FILE" || -z "$OUTPUT_FILE" || -z "$QCDIR" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

# Validate B1 clean output exists
if [[ ! -f "${QCDIR}/${CLEAN_NAME}.bed" ]]; then
    echo "Error: B1 clean output not found: ${QCDIR}/${CLEAN_NAME}.bed"
    echo "Run B1_plinkQC_genotype_qc.R first."
    exit 1
fi

echo "============================================="
echo "B2: SNP Harmonization & PRS Calculation"
echo "============================================="
echo "B1 clean data: ${QCDIR}/${CLEAN_NAME}"
echo "Original data: $ORIGINAL_DATA"
echo "PLINK folder: $PLINK_FOLDER"
echo "Data folder: $DATA_FOLDER"
echo "Code folder: $CODE_FOLDER"
echo "Phenotypic file: $PHENOTYPIC_FILE"
echo "Output file: $OUTPUT_FILE"

# Derive results folder (sibling of data folder)
RESULTS_FOLDER="$(dirname "$DATA_FOLDER")/results"
mkdir -p "$RESULTS_FOLDER"
mkdir -p "$PLINK_FOLDER"

# Step 1: Copy iPSYCH GWAS summary statistics to working directory
echo "============================================="
echo "Step 1: Copying GWAS summary statistics..."

for file in ${ORIGINAL_DATA}/iPSYCH*; do
  dest_file="${PLINK_FOLDER}/$(basename "$file" | tr ' ' '_')"
  cp "$file" "$dest_file"
done

# Step 2: LD pruning on B1 clean data
echo "============================================="
echo "Step 2: Performing SNP pruning on QC'd data..."

plink --bfile ${QCDIR}/${CLEAN_NAME} \
      --make-founders \
      --indep-pairwise 100 25 0.5 \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_pruned

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in

plink --bfile ${QCDIR}/${CLEAN_NAME} \
      --make-founders \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_pruned

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.fam

# Step 3: Heterozygosity statistics (needed by harmonize_snps.R)
echo "============================================="
echo "Step 3: Computing heterozygosity statistics..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_pruned \
      --make-founders \
      --het \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_het

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_het.het

# Step 4: Run R-script to harmonize SNPs with GWAS base data
echo "============================================="
echo "Step 4: Running harmonisation Rscript..."

Rscript ${CODE_FOLDER}/utils/harmonize_snps.R --plink-dir "${PLINK_FOLDER}"

# Step 5: Rebuilding dataset with corrected SNPs
echo "============================================="
echo "Step 5: Rebuilding PLINK dataset with recoded SNPs..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_pruned \
      --bim "${PLINK_FOLDER}/Neuro_Chip_qc_recoded.bim" \
      --make-bed \
      --out "${PLINK_FOLDER}/Neuro_Chip_qc_recoded"

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.fam

# Step 6: Remove duplicate SNPs
# Output named Neuro_Chip_qc_nodup_sexfiltered for backward compatibility
# (sex filtering already done by B1 plinkQC)
echo "============================================="
echo "Step 6: Removing duplicate SNPs..."
awk '{if (!seen[$1]++) print $1}' ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.a1 > ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_recoded \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered.fam

# Step 7: Calculate PCs for the full sample (for population stratification control)
echo "============================================="
echo "Step 7: Calculating Principal Components..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --pca 10 \
      --out ${PLINK_FOLDER}/Neuro_Chip_full_sample_pca

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_full_sample_pca.eigenvec)
echo "Step 7 (full sample PCA): ${N} participants" >> ${RESULTS_FOLDER}/sample_counts.txt

# Step 8: Relatedness filtering (using phenotypic twin data)
echo "============================================="
echo "Step 8: Relatedness filtering..."
awk -F',' 'NR > 1 && $4 == "NotTwin" {print $6, $1}' "${PHENOTYPIC_FILE}" > ${PLINK_FOLDER}/unrelated_samples.txt

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --keep ${PLINK_FOLDER}/unrelated_samples.txt \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated.fam)
echo "Step 8 (unrelated): ${N} participants" >> ${RESULTS_FOLDER}/sample_counts.txt

# Continue with IBD-based relatedness filtering
plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --make-founders \
      --rel-cutoff 0.125 \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered.fam)
echo "Step 8 (filtered): ${N} participants" >> ${RESULTS_FOLDER}/sample_counts.txt

# Step 9: Final QC with proper allele alignment
echo "============================================="
echo "Step 9: Final allele alignment..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered \
      --make-bed \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup \
      --exclude ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.mismatch \
      --a1-allele ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.a1 \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated.fam)
echo "Step 9 (final unrelated): ${N} participants" >> ${RESULTS_FOLDER}/sample_counts.txt

# Step 10: Base data QC
echo "============================================="
echo "Step 10: Base data QC (iPSYCH GWAS cleaning)..."

gunzip -c ${PLINK_FOLDER}/iPSYCH_PGC_ASD_Nov_2017.gz | awk '!seen[$2]++' | gzip > ${PLINK_FOLDER}/iPSYCH_cleaned.gz
gunzip -c ${PLINK_FOLDER}/iPSYCH_cleaned.gz | awk '$5 != "NA" && $6 != "NA" && $7 >= 0.7' | gzip > ${PLINK_FOLDER}/iPSYCH_filtered.gz
gunzip -c ${PLINK_FOLDER}/iPSYCH_filtered.gz | awk '$8 >= 0.01' | gzip > ${PLINK_FOLDER}/iPSYCH_qc_final.gz

# Step 11: Calculate PRS at multiple thresholds for unrelated sample
echo "============================================="
echo "Step 11: Running PRSice for threshold selection..."

PRSice_linux \
      --base ${PLINK_FOLDER}/iPSYCH_qc_final.gz \
      --target ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated \
      --no-regress \
      --out ${PLINK_FOLDER}/Neuro_Chip_unrelated_prsice

# Copy the output to the expected location
# Note: With --no-regress, PRSice creates .all_score (multi-threshold) not .best
# Threshold selection happens in B3_select_PRS_threshold.py
cp ${PLINK_FOLDER}/Neuro_Chip_unrelated_prsice.all_score "$OUTPUT_FILE"

# Also create the unrelated_prs_scores.txt for downstream evaluation
cp ${PLINK_FOLDER}/Neuro_Chip_unrelated_prsice.all_score ${PLINK_FOLDER}/unrelated_prs_scores.txt

echo ""
echo "============================================="
echo "B2 completed successfully."
echo "Output written to: $OUTPUT_FILE"
echo "Unrelated PRS scores written to: ${PLINK_FOLDER}/unrelated_prs_scores.txt"
echo "============================================="
