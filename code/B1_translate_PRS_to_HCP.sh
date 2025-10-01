#!/bin/bash
# ==============================================================================
# Title: B1_translate_PRS_to_HCP.sh (REVISED)
# ==============================================================================
# Description: This script translates polygenic risk scores (PRS) from the
# original genetic sample to the HCP (Human Connectome Project) sample. It
# includes steps such as quality control, PRS computation, and output
# generation. PCs are calculated on the full QC'd sample for later BLUP
# extension to related individuals.
# ==============================================================================

set -e
echo "Starting PLINK QC process..."

ORIGINAL_DATA="../data/genetics_data"
DATA_FOLDER="../data"
PLINK_FOLDER="../data/plink"
CODE_FOLDER="../code"

mkdir -p ${PLINK_FOLDER}

# Step 1: Initial Quality Control (QC)
echo "============================================="
echo "Running initial SNP QC..."

plink --bfile ${ORIGINAL_DATA}/Neuro_Chip_anonymised \
      --maf 0.01 --hwe 1e-6 --geno 0.01 --mind 0.01 \
      --write-snplist --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc

# Report SNPs & individuals
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc.fam

cp ${ORIGINAL_DATA}/Neuro_Chip* ${PLINK_FOLDER}/
cp ${ORIGINAL_DATA}/iPSYCH* ${PLINK_FOLDER}/

# Step 2: Force individuals to be founders
echo "============================================="
echo "Forcing individuals to be founders..."

awk '{$3=0; $4=0; print $0}' ${PLINK_FOLDER}/Neuro_Chip_qc.fam > ${PLINK_FOLDER}/Neuro_Chip_founders.fam

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc \
      --fam ${PLINK_FOLDER}/Neuro_Chip_founders.fam \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_founders

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_founders.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_founders.fam

# Step 3: Pruning for high linkage disequilibrium (LD)
echo "============================================="
echo "Performing SNP pruning..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_founders \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc.snplist \
      --indep-pairwise 100 25 0.5 \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_pruned

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_founders \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_pruned

# Step 4: Heterozygosity check
echo "============================================="
echo "Checking heterozygosity..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_pruned \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --keep ${PLINK_FOLDER}/Neuro_Chip_founders.fam \
      --het \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_het

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_het.het

# Step 5: Run R-script to match SNPs with base data
echo "============================================="
echo "Running harmonisation Rscript..."

Rscript ${CODE_FOLDER}/snp_mismatched.R "${PLINK_FOLDER}"

# Step 6: Rebuilding dataset with corrected SNPs
echo "============================================="
echo "Rebuilding PLINK dataset with recoded SNPs..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_pruned \
      --bim "${PLINK_FOLDER}/Neuro_Chip_qc_recoded.bim" \
      --make-bed \
      --out "${PLINK_FOLDER}/Neuro_Chip_qc_recoded"

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.fam

# Step 7: Remove duplicate SNPs
echo "============================================="
echo "Removing duplicate SNPs..."
awk '{if (!seen[$1]++) print $1}' ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.a1 > ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_recoded \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup.fam

# Step 8: Sex check
echo "============================================="
echo "Performing sex check..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --keep ${PLINK_FOLDER}/Neuro_Chip_qc.valid.sample \
      --check-sex \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_sexcheck

awk '$5 == "OK"' ${PLINK_FOLDER}/Neuro_Chip_qc_sexcheck.sexcheck > ${PLINK_FOLDER}/sex_consistent_samples.txt

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup \
      --keep ${PLINK_FOLDER}/sex_consistent_samples.txt \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered.fam

# Step 9: Calculate PCs EARLY for the full sample (after sex filtering)
echo "============================================="
echo "Calculating Principal Components for population stratification control..."

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --pca 10 \
      --out ${PLINK_FOLDER}/Neuro_Chip_full_sample_pca

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_full_sample_pca.fam)
echo "Step 9 (full sample): ${N} participants" >> ${DATA_FOLDER}/sample_counts.txt

# Step 10: Relatedness filtering
echo "============================================="
echo "Relatedness filtering..."
awk -F',' 'NR > 1 && $4 == "NotTwin" {print $6, $1}' ${DATA_FOLDER}/phenotypic_data_anonymised.csv > ${PLINK_FOLDER}/unrelated_samples.txt

plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_nodup_sexfiltered \
      --keep ${PLINK_FOLDER}/unrelated_samples.txt \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated.fam)
echo "Step 10 (unrelated): ${N} participants" >> ${DATA_FOLDER}/sample_counts.txt


# Continue with relatedness filtering
plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_pruned.prune.in \
      --rel-cutoff 0.125 \
      --make-bed \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered.fam)
echo "Step 10 (filtered): ${N} participants" >> ${DATA_FOLDER}/sample_counts.txt

# Write the IDs of the related participants who were randomly removed
cut -d ' ' -f1,2 ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered.fam | sort > ${PLINK_FOLDER}/post_relatedness_ids.txt
comm -23 ${PLINK_FOLDER}/pre_relatedness_ids.txt ${PLINK_FOLDER}/post_relatedness_ids.txt > ${PLINK_FOLDER}/removed_relatedness.txt

wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered.bim
wc -l ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered.fam

# Step 11: Final QC with proper allele alignment
plink --bfile ${PLINK_FOLDER}/Neuro_Chip_qc_unrelated_filtered \
      --make-bed \
      --extract ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.snplist.nodup \
      --exclude ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.mismatch \
      --a1-allele ${PLINK_FOLDER}/Neuro_Chip_qc_recoded.a1 \
      --out ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated

N=$(wc -l < ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated.fam)
echo "Step 11 (recoded): ${N} participants" >> ${DATA_FOLDER}/sample_counts.txt

# Step 12: Base data QC
gunzip -c ${PLINK_FOLDER}/iPSYCH_PGC_ASD_Nov_2017.gz | awk '!seen[$2]++' | gzip > ${PLINK_FOLDER}/iPSYCH_cleaned.gz
gunzip -c ${PLINK_FOLDER}/iPSYCH_cleaned.gz | awk '$5 != "NA" && $6 != "NA" && $7 >= 0.7' | gzip > ${PLINK_FOLDER}/iPSYCH_filtered.gz
gunzip -c ${PLINK_FOLDER}/iPSYCH_filtered.gz | awk '$8 >= 0.01' | gzip > ${PLINK_FOLDER}/iPSYCH_qc_final.gz

# Step 13: Calculate PRS at multiple thresholds for unrelated sample
echo "============================================="
echo "Running PRSice for threshold selection..."

PRSice_linux \
      --base ${PLINK_FOLDER}/iPSYCH_qc_final.gz \
      --target ${PLINK_FOLDER}/Neuro_Chip_qc_final_unrelated \
      --no-regress \
      --out ${PLINK_FOLDER}/Neuro_Chip_unrelated_prsice
