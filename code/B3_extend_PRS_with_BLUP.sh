#!/bin/bash
# ==============================================================================
# Title: B3_extend_PRS_with_BLUP.sh (REVISED)
# ==============================================================================
# Description: This script extends polygenic risk scores (PRS) with BLUP
# (Best Linear Unbiased Prediction) estimates. It uses the full QC'd sample
# to calculate a genomic relationship matrix, then extends PGS from unrelated
# individuals to the full sample (including related individuals) while
# controlling for population stratification with pre-calculated PCs.
# ==============================================================================

set -e  # Stop execution if any command fails

# Target data directories
CODE_FOLDER="../code"
DATA_FOLDER="../data"
ORIGINAL_DATA="../data/genetics_data"
PLINK_FOLDER="../data/plink"

# Genomic Relationship Matrix (GRM) calculation for the full sample
gcta64 --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
    --autosome \
    --make-grm \
    --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_grm

# Linear Mixed Model (LMM) using the polygenic risk scores (PRS) from unrelated individuals
gcta64 --grm ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_grm \
      --pheno ${PLINK_FOLDER}/unrelated_prs_scores.txt \
      --reml \
      --reml-pred-rand \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_prs_lmm

# Extract SNP Effect Sizes from the Model (BLUP solutions for SNPs)
gcta64 \
      --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --blup-snp ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_prs_lmm.indi.blp \
      --autosome \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_snp_effects

# Calculate the PRS for the full sample
plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --score ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered_snp_effects.snp.blp sum 1 2 3 \
      --out ${PLINK_FOLDER}/full_prs_scores.snp.blp

# Calculate the PCs for the full sample
plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --pca 10 \
      --out ${PLINK_FOLDER}/full_prs_scores.snp.blp.pca