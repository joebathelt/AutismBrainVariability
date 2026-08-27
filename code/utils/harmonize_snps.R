#!/usr/bin/env Rscript
# =============================================================================
# harmonize_snps.R
# Harmonize SNP alleles between target genotype data and GWAS summary statistics
# =============================================================================
#
# This script performs allele harmonization between PLINK genotype data and
# GWAS summary statistics, handling strand flips and allele recoding.
#
# Usage:
#   Rscript harmonize_snps.R --plink-dir <path> [--output-dir <path>]
#
# Inputs (from plink-dir):
#   - Neuro_Chip_qc_het.het      : Heterozygosity estimates from PLINK
#   - Neuro_Chip_qc_pruned.bim   : BIM file with SNP information
#   - iPSYCH_PGC_ASD_Nov_2017.gz : GWAS summary statistics
#   - Neuro_Chip_qc_pruned.prune.in : List of QC-passed SNPs
#
# Outputs (to output-dir, defaults to plink-dir):
#   - Neuro_Chip_qc.valid.sample    : Samples passing heterozygosity QC
#   - Neuro_Chip_qc_recoded.bim     : BIM file with harmonized alleles
#   - Neuro_Chip_qc_recoded.a1      : Effect allele file for PLINK
#   - Neuro_Chip_qc_recoded.mismatch: SNPs that couldn't be harmonized
#
# =============================================================================

# Function to install and load required packages
install_if_missing <- function(packages) {
    for (pkg in packages) {
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
            cat(paste("Installing missing package:", pkg, "\n"))
            install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
            library(pkg, character.only = TRUE)
        }
    }
}

# List of required packages
required_packages <- c("data.table", "magrittr", "R.utils")
install_if_missing(required_packages)

# =============================================================================
# Helper Functions
# =============================================================================

#' Get complementary DNA base
#' @param x Single nucleotide character (A, T, C, or G)
#' @return Complementary base or NA if invalid
complement <- function(x) {
    switch(x,
           "A" = "T", "T" = "A",
           "C" = "G", "G" = "C",
           NA)
}

#' Parse command-line arguments
#' @return Named list with plink_dir and output_dir
parse_args <- function() {
    args <- commandArgs(trailingOnly = TRUE)

    # Initialize defaults
    plink_dir <- NULL
    output_dir <- NULL

    # Parse arguments
    i <- 1
    while (i <= length(args)) {
        if (args[i] == "--plink-dir" && i < length(args)) {
            plink_dir <- args[i + 1]
            i <- i + 2
        } else if (args[i] == "--output-dir" && i < length(args)) {
            output_dir <- args[i + 1]
            i <- i + 2
        } else if (args[i] == "--help" || args[i] == "-h") {
            cat("Usage: Rscript harmonize_snps.R --plink-dir <path> [--output-dir <path>]\n")
            cat("\nArguments:\n")
            cat("  --plink-dir   Path to PLINK data directory (required)\n")
            cat("  --output-dir  Path to output directory (defaults to plink-dir)\n")
            cat("  --help        Show this help message\n")
            quit(status = 0)
        } else {
            # Support legacy single-argument mode
            if (is.null(plink_dir)) {
                plink_dir <- args[i]
            }
            i <- i + 1
        }
    }

    if (is.null(plink_dir)) {
        stop("Error: --plink-dir is required. Use --help for usage information.")
    }

    if (is.null(output_dir)) {
        output_dir <- plink_dir
    }

    list(plink_dir = plink_dir, output_dir = output_dir)
}

#' Check that required input files exist
#' @param plink_dir Directory containing input files
#' @return Invisible NULL, stops on missing files
check_required_files <- function(plink_dir) {
    required_files <- c(
        "Neuro_Chip_qc_het.het",
        "Neuro_Chip_qc_pruned.bim",
        "iPSYCH_PGC_ASD_Nov_2017.gz",
        "Neuro_Chip_qc_pruned.prune.in"
    )

    missing_files <- required_files[!file.exists(file.path(plink_dir, required_files))]

    if (length(missing_files) > 0) {
        stop("Error: Missing required files: ", paste(missing_files, collapse = ", "))
    }

    invisible(NULL)
}

# =============================================================================
# QC Functions
# =============================================================================

#' Filter samples based on heterozygosity
#' @param plink_dir Directory containing het file
#' @param output_dir Directory for output file
#' @return data.table of valid samples
filter_heterozygosity <- function(plink_dir, output_dir) {
    message("Filtering samples by heterozygosity...")

    het_file <- file.path(plink_dir, "Neuro_Chip_qc_het.het")
    dat <- fread(het_file)

    # Filter to samples within 3 SD of mean F coefficient
    mean_F <- mean(dat$F, na.rm = TRUE)
    sd_F <- sd(dat$F, na.rm = TRUE)

    valid <- dat[F >= mean_F - 3 * sd_F & F <= mean_F + 3 * sd_F, .(FID, IID)]

    message(sprintf("  %d / %d samples passed heterozygosity filter", nrow(valid), nrow(dat)))

    # Write valid samples
    valid_file <- file.path(output_dir, "Neuro_Chip_qc.valid.sample")
    fwrite(valid, valid_file, sep = "\t", col.names = FALSE)
    message(sprintf("  Written to: %s", valid_file))

    valid
}

# =============================================================================
# Allele Harmonization Functions
# =============================================================================

#' Load and prepare BIM file
#' @param plink_dir Directory containing BIM file
#' @return data.table with standardized alleles
load_bim <- function(plink_dir) {
    message("Loading BIM file...")

    bim_file <- file.path(plink_dir, "Neuro_Chip_qc_pruned.bim")
    bim <- fread(bim_file, col.names = c("CHR", "SNP", "CM", "BP", "B.A1", "B.A2"))

    # Standardize alleles to uppercase
    bim[, `:=`(B.A1 = toupper(B.A1), B.A2 = toupper(B.A2))]

    message(sprintf("  Loaded %d SNPs", nrow(bim)))
    bim
}

#' Load and prepare GWAS summary statistics
#' @param plink_dir Directory containing GWAS file
#' @return data.table with standardized alleles
load_gwas <- function(plink_dir) {
    message("Loading GWAS summary statistics...")

    gwas_file <- file.path(plink_dir, "iPSYCH_PGC_ASD_Nov_2017.gz")
    gwas <- fread(gwas_file)

    # Standardize alleles to uppercase
    gwas[, `:=`(A1 = toupper(A1), A2 = toupper(A2))]

    message(sprintf("  Loaded %d variants", nrow(gwas)))
    gwas
}

#' Load QC SNP list
#' @param plink_dir Directory containing SNP list
#' @return Character vector of SNP IDs
load_qc_snps <- function(plink_dir) {
    message("Loading QC SNP list...")

    snplist_file <- file.path(plink_dir, "Neuro_Chip_qc_pruned.prune.in")
    qc <- fread(snplist_file, header = FALSE)

    message(sprintf("  %d QC-passed SNPs", nrow(qc)))
    qc$V1
}

#' Harmonize alleles between target and base data
#' @param bim BIM data.table
#' @param gwas GWAS data.table
#' @param qc_snps Character vector of QC-passed SNPs
#' @param output_dir Directory for output files
#' @return Updated BIM data.table
harmonize_alleles <- function(bim, gwas, qc_snps, output_dir) {
    message("Harmonizing alleles...")

    # Merge BIM and GWAS data, filter to QC SNPs
    info <- merge(bim, gwas, by = c("SNP", "CHR", "BP"), all.x = TRUE)
    info <- info[SNP %in% qc_snps]

    # 1. Identify exact matches (no action needed)
    match_snps <- info[A1 == B.A1 & A2 == B.A2, SNP]
    message(sprintf("  Exact matches: %d SNPs", length(match_snps)))

    # 2. Identify strand flips (complement needed)
    complement_snps <- info[
        mapply(complement, B.A1) == A1 & mapply(complement, B.A2) == A2,
        SNP
    ]
    message(sprintf("  Strand flips: %d SNPs", length(complement_snps)))

    # Apply complement to BIM
    bim[SNP %in% complement_snps, `:=`(
        B.A1 = mapply(complement, B.A1),
        B.A2 = mapply(complement, B.A2)
    )]

    # Save recoded BIM
    recoded_bim_file <- file.path(output_dir, "Neuro_Chip_qc_recoded.bim")
    fwrite(bim, recoded_bim_file, col.names = FALSE, sep = "\t", quote = FALSE)
    message(sprintf("  Written recoded BIM: %s", recoded_bim_file))

    # 3. Identify allele swaps (A1/A2 reversed)
    recode_snps <- info[B.A1 == A2 & B.A2 == A1, SNP]
    message(sprintf("  Allele swaps: %d SNPs", length(recode_snps)))

    # Apply swap to BIM
    bim[SNP %in% recode_snps, `:=`(B.A1 = B.A2, B.A2 = B.A1)]

    # 4. Identify SNPs needing both complement and swap
    complement_recode_snps <- info[
        mapply(complement, B.A1) == A2 & mapply(complement, B.A2) == A1,
        SNP
    ]
    message(sprintf("  Complement + swap: %d SNPs", length(complement_recode_snps)))

    # Apply complement + swap to BIM
    bim[SNP %in% complement_recode_snps, `:=`(
        B.A1 = mapply(complement, B.A2),
        B.A2 = mapply(complement, B.A1)
    )]

    # Write effect allele file
    a1_file <- file.path(output_dir, "Neuro_Chip_qc_recoded.a1")
    fwrite(bim[, .(SNP, B.A1)], a1_file, col.names = FALSE, sep = "\t")
    message(sprintf("  Written effect allele file: %s", a1_file))

    # 5. Identify mismatched SNPs (couldn't be harmonized)
    harmonized_snps <- c(match_snps, complement_snps, recode_snps, complement_recode_snps)
    mismatch_snps <- bim[!(SNP %in% harmonized_snps), SNP]
    message(sprintf("  Mismatched (excluded): %d SNPs", length(mismatch_snps)))

    mismatch_file <- file.path(output_dir, "Neuro_Chip_qc_recoded.mismatch")
    fwrite(data.table(mismatch_snps), mismatch_file, col.names = FALSE, quote = FALSE)
    message(sprintf("  Written mismatch file: %s", mismatch_file))

    bim
}

# =============================================================================
# Main
# =============================================================================

main <- function() {
    # Parse arguments
    args <- parse_args()

    message("==============================================")
    message("SNP Harmonization Script")
    message("==============================================")
    message(sprintf("PLINK directory: %s", args$plink_dir))
    message(sprintf("Output directory: %s", args$output_dir))
    message("")

    # Check inputs exist
    check_required_files(args$plink_dir)

    # Step 1: Heterozygosity filtering
    filter_heterozygosity(args$plink_dir, args$output_dir)

    # Step 2: Load data
    bim <- load_bim(args$plink_dir)
    gwas <- load_gwas(args$plink_dir)
    qc_snps <- load_qc_snps(args$plink_dir)

    # Step 3: Harmonize alleles
    harmonize_alleles(bim, gwas, qc_snps, args$output_dir)

    message("")
    message("==============================================")
    message("Processing complete!")
    message("==============================================")
}

# Run main if executed as script
if (!interactive()) {
    main()
}
