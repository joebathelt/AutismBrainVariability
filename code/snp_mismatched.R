library(data.table)
library(magrittr)

# ==============================================================================
# Title: snp_mismatched.R
# ==============================================================================
# Description: This script processes PLINK genetic data to identify and
# correct mismatched SNP alleles based on GWAS summary statistics. It handles
# allele complementing and recoding, and outputs updated BIM files and lists of
# mismatched SNPs.
# ==============================================================================

# --- Read command-line argument ---
args <- commandArgs(trailingOnly = TRUE)

# Ensure a directory was provided
if (length(args) < 1) {
  stop("Error: No plink_data_dir provided. Usage: Rscript process_plink_data.R /path/to/data")
}

plink_data_dir <- args[1]
print(paste("Using PLINK data directory:", plink_data_dir))

# --- Check if required files exist ---
required_files <- c("Neuro_Chip_qc_het.het", "Neuro_Chip_qc_pruned.bim",
                    "iPSYCH_PGC_ASD_Nov_2017.gz", "Neuro_Chip_qc_pruned.prune.in")

missing_files <- required_files[!file.exists(file.path(plink_data_dir, required_files))]
if (length(missing_files) > 0) {
  stop("Error: The following required files are missing: ", paste(missing_files, collapse = ", "))
}

# --- Read in heterozygosity file ---
het_file <- file.path(plink_data_dir, "Neuro_Chip_qc_het.het")
dat <- fread(het_file)

# Get valid samples within 3 SD of the population mean
mean_F <- mean(dat$F, na.rm = TRUE)
sd_F <- sd(dat$F, na.rm = TRUE)
valid <- dat[F >= mean_F - 3 * sd_F & F <= mean_F + 3 * sd_F, .(FID, IID)]

# Write valid samples to file
valid_file <- file.path(plink_data_dir, "Neuro_Chip_qc.valid.sample")
fwrite(valid, valid_file, sep = "\t", col.names = FALSE)

# --- Read in BIM file and process allele columns ---
bim_file <- file.path(plink_data_dir, "Neuro_Chip_qc_pruned.bim")
bim <- fread(bim_file, col.names = c("CHR", "SNP", "CM", "BP", "B.A1", "B.A2"))
bim[, `:=`(B.A1 = toupper(B.A1), B.A2 = toupper(B.A2))]

# --- Read in GWAS summary statistics and process alleles ---
ipsych_file <- file.path(plink_data_dir, "iPSYCH_PGC_ASD_Nov_2017.gz")
autism <- fread(ipsych_file)
autism[, `:=`(A1 = toupper(A1), A2 = toupper(A2))]

# --- Read in QC SNPs ---
snplist_file <- file.path(plink_data_dir, "Neuro_Chip_qc_pruned.prune.in")
qc <- fread(snplist_file, header = FALSE)

# --- Merge BIM and summary statistics, filter QC SNPs ---
info <- merge(bim, autism, by = c("SNP", "CHR", "BP"), all.x = TRUE)
info <- info[SNP %in% qc$V1]

# --- Define a function for complementary alleles ---
complement <- function(x) {
  switch(x,
         "A" = "T", "T" = "A",
         "C" = "G", "G" = "C",
         NA)
}

# --- Identify SNPs with matching alleles ---
info.match <- info[A1 == B.A1 & A2 == B.A2, SNP]

# --- Identify complementary SNPs ---
com.snps <- info[mapply(complement, B.A1) == A1 & mapply(complement, B.A2) == A2, SNP]

# --- Update BIM file for complementary SNPs ---
bim[SNP %in% com.snps, `:=`(
  B.A1 = mapply(complement, B.A1),
  B.A2 = mapply(complement, B.A2)
)]

# Save the recoded BIM file
recoded_bim_file <- file.path(plink_data_dir, "Neuro_Chip_qc_recoded.bim")
fwrite(bim, recoded_bim_file, col.names = FALSE, sep = "\t", quote = FALSE)

# --- Identify SNPs requiring recoding ---
recode.snps <- info[B.A1 == A2 & B.A2 == A1, SNP]

# --- Update BIM file for recoded SNPs ---
bim[SNP %in% recode.snps, `:=`(B.A1 = B.A2, B.A2 = B.A1)]

# --- Identify SNPs requiring both recoding and complementing ---
com.recode <- info[mapply(complement, B.A1) == A2 & mapply(complement, B.A2) == A1, SNP]

# --- Update BIM file for recoded & complemented SNPs ---
bim[SNP %in% com.recode, `:=`(
  B.A1 = mapply(complement, B.A2),
  B.A2 = mapply(complement, B.A1)
)]

# --- Write updated allele information ---
a1_file <- file.path(plink_data_dir, "Neuro_Chip_qc_recoded.a1")
fwrite(bim[, .(SNP, B.A1)], a1_file, col.names = FALSE, sep = "\t")

# --- Identify mismatched SNPs ---
mismatch_file <- file.path(plink_data_dir, "Neuro_Chip_qc_recoded.mismatch")
mismatch <- bim[!(SNP %in% info.match | SNP %in% com.snps | SNP %in% recode.snps | SNP %in% com.recode), SNP]
fwrite(data.table(mismatch), mismatch_file, col.names = FALSE, quote = FALSE)

print("Processing complete!")