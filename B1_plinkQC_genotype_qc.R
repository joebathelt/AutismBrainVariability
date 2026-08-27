#!/usr/bin/env Rscript

# =============================================================================
# B1_plinkQC_genotype_qc.R
# Standalone genotype quality control pipeline using plinkQC
# BrainCompensation Project
#
# Performs per-individual QC, per-marker QC, and combined data cleanup.
# Produces diagnostic plots and a text QC report.
#
# Sex filtering is done via a phenotype cross-check (FAM PEDSEX vs the
# Gender column in the phenotype CSV) — the chrX F-statistic check is
# disabled because the source HCP data has corrupted female chrX
# heterozygosity. Samples whose FAM PEDSEX disagrees with the phenotype
# Gender are dropped via an explicit PLINK --remove pass after cleanData.
#
# Ancestry inference is delegated to B1b_ancestry_PCA_mahalanobis.py, which
# consumes B1's .clean triplet to produce a within-sample PCA (canonical PC
# source for C1/C3/C3b nuisance covariates) plus a 1KG-projected Mahalanobis
# diagnostic. B1b does not drop any participants; B2 consumes B1's .clean
# triplet directly.
#
# Default input is hg19 (the original NeuroChip genotypes); pass
# --genomebuild hg38 if running on lifted-over data.
#
# Usage:
#   Rscript B1_plinkQC_genotype_qc.R --project /path/to/BrainCompensation
#
# See --help for all options.
# =============================================================================

# --- Section 1: Package management -------------------------------------------

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(paste("Installing missing package:", pkg, "\n"))
      install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

required_packages <- c("plinkQC", "ggplot2", "data.table")
install_if_missing(required_packages)

library(plinkQC)
library(ggplot2)
library(data.table)

# Muffle known deprecation / non-finite warnings from upstream packages
# (plinkQC uses aes_string(); UpSetR uses size= for lines; some internal
# plinkQC histograms emit "Removed N rows containing non-finite"). All other
# warnings still surface normally.
suppress_upstream_warnings <- function(expr) {
  patterns <- c(
    "aes_string\\(\\) was deprecated",
    "`size` aesthetic for lines was deprecated",
    "`size` argument of `element_line\\(\\)` is deprecated",
    "Removed [0-9]+ rows containing non-finite"
  )
  withCallingHandlers(
    expr,
    warning = function(w) {
      msg <- conditionMessage(w)
      if (any(vapply(patterns, grepl, logical(1), x = msg))) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

# --- Section 2: Command-line argument parsing ---------------------------------

args <- commandArgs(trailingOnly = TRUE)

usage <- function() {
  cat("Usage: Rscript B1_plinkQC_genotype_qc.R --project PROJECT [OPTIONS]\n\n")
  cat("Required:\n")
  cat("  --project       Path to project root directory\n\n")
  cat("Optional:\n")
  cat("  --indir         Input PLINK directory (relative to project) [data/raw_anonymised]\n")
  cat("  --name          PLINK file prefix [Neuro_Chip_anonymised]\n")
  cat("  --qcdir         QC output directory (relative to project) [data/plinkQC_output]\n")
  cat("  --path2plink    Path to PLINK v1.9 executable [auto-detect]\n")
  cat("  --path2plink2   Path to PLINK 2.0 executable [auto-detect]\n")
  cat("  --genomebuild   Genome build of input data [hg19]\n")
  cat("  --maf           Minor allele frequency threshold [0.01]\n")
  cat("  --hwe           HWE p-value threshold [1e-6]\n")
  cat("  --geno          SNP missingness threshold [0.01]\n")
  cat("  --mind          Sample missingness threshold [0.03]\n")
  cat("  --het           Heterozygosity SD threshold [3]\n")
  cat("  --ibd           IBD relatedness threshold [0.1875]\n")
  cat("  --pheno-sex-csv Phenotype CSV with trusted sex (Subject,Gender) for sex cross-check\n")
  cat("                  [data/hcp_behavioural_raw.csv]\n")
  cat("  --no-apply-sex-filter  Sanity-check mode: still run the PEDSEX vs phenotype\n")
  cat("                         Gender cross-check and write the audit CSV /\n")
  cat("                         .fail-sexcheck.IDs (so the report shows what *would*\n")
  cat("                         have been dropped), but do NOT remove mismatching\n")
  cat("                         samples from the .clean triplet.\n")
  cat("  --help          Show this help message\n")
  quit(status = 1)
}

if (length(args) == 0) usage()

# Initialize with defaults
project_dir   <- NULL
indir_rel     <- "data/raw_anonymised"
name          <- "Neuro_Chip_anonymised"
qcdir_rel     <- "data/plinkQC_output"
path2plink    <- NULL
path2plink2   <- NULL
genomebuild   <- "hg19"
mafTh         <- 0.01
hweTh         <- 1e-6
lmissTh       <- 0.01
imissTh       <- 0.03
hetTh         <- 3
highIBDTh     <- 0.1875
pheno_sex_csv_rel <- "data/hcp_behavioural_raw.csv"
apply_sex_filter  <- TRUE

i <- 1
while (i <= length(args)) {
  if (args[i] == "--project") {
    project_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--indir") {
    indir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--name") {
    name <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--qcdir") {
    qcdir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2plink") {
    path2plink <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2plink2") {
    path2plink2 <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--genomebuild") {
    genomebuild <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--maf") {
    mafTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--hwe") {
    hweTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--geno") {
    lmissTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--mind") {
    imissTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--het") {
    hetTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--ibd") {
    highIBDTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--pheno-sex-csv") {
    pheno_sex_csv_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--no-apply-sex-filter") {
    apply_sex_filter <- FALSE; i <- i + 1
  } else if (args[i] == "--help") {
    usage()
  } else {
    cat("Unknown argument:", args[i], "\n")
    usage()
  }
}

if (is.null(project_dir)) {
  cat("Error: --project is required\n")
  usage()
}

# --- Section 3: Resolve paths and configuration -------------------------------

indir      <- file.path(project_dir, indir_rel)
qcdir      <- file.path(project_dir, qcdir_rel)
figures_dir <- file.path(project_dir, "figures")
reports_dir <- file.path(project_dir, "reports")

# Phenotype CSV can be absolute or relative-to-project
pheno_sex_csv <- if (startsWith(pheno_sex_csv_rel, "/")) {
  pheno_sex_csv_rel
} else {
  file.path(project_dir, pheno_sex_csv_rel)
}

# --- Section 4: Input validation and setup ------------------------------------

check_required_files <- function(dir, prefix) {
  extensions <- c(".bed", ".bim", ".fam")
  for (ext in extensions) {
    f <- file.path(dir, paste0(prefix, ext))
    if (!file.exists(f)) {
      stop("Required file not found: ", f)
    }
  }
}

detect_plink <- function() {
  # Try common locations on macOS
  candidates <- c(
    Sys.which("plink"),
    "/Applications/PLINK 1.9 Mac Oct 2024/plink",
    "/usr/local/bin/plink",
    "/opt/homebrew/bin/plink"
  )
  for (p in candidates) {
    if (nchar(p) > 0 && file.exists(p)) return(p)
  }
  stop("PLINK v1.9 not found. Please specify --path2plink.")
}

detect_plink2 <- function() {
  candidates <- c(
    Sys.which("plink2"),
    "/Applications/plink2_mac_arm64_20260110/plink2",
    "/Applications/PLINK 2.0 Mac Arm64 Jan 29 2025/plink2",
    "/usr/local/bin/plink2",
    "/opt/homebrew/bin/plink2"
  )
  for (p in candidates) {
    if (nchar(p) > 0 && file.exists(p)) return(p)
  }
  return(NULL)  # plink2 is optional
}

message("=============================================================")
message("B1: plinkQC Genotype Quality Control")
message("=============================================================")
message("Project:    ", project_dir)
message("Input dir:  ", indir)
message("File prefix:", name)
message("QC dir:     ", qcdir)
message("Build:      ", genomebuild)
message("")

# Validate input files
check_required_files(indir, name)

# Detect PLINK
if (is.null(path2plink)) path2plink <- detect_plink()
if (is.null(path2plink2)) path2plink2 <- detect_plink2()
message("PLINK v1.9: ", path2plink)
message("PLINK 2.0:  ", ifelse(is.null(path2plink2), "not found", path2plink2))

# Create output directories
for (d in c(qcdir, figures_dir, reports_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# Count initial samples and SNPs
bim <- fread(file.path(indir, paste0(name, ".bim")), header = FALSE)
fam <- fread(file.path(indir, paste0(name, ".fam")), header = FALSE)
initial_snp_count <- nrow(bim)
initial_ind_count <- nrow(fam)
rm(bim, fam)

message("Initial data: ", initial_snp_count, " SNPs, ", initial_ind_count, " individuals")
message("")

# Initialize report
report_lines <- c(
  paste(rep("=", 72), collapse = ""),
  "B1: GENOTYPE QUALITY CONTROL REPORT (plinkQC)",
  paste(rep("=", 72), collapse = ""),
  "",
  paste("Date:           ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste("Input data:     ", name),
  paste("Input directory: ", indir),
  paste("Output directory:", qcdir),
  paste("PLINK v1.9:     ", path2plink),
  "",
  "INITIAL DATA:",
  paste("  SNPs:        ", initial_snp_count),
  paste("  Individuals: ", initial_ind_count),
  ""
)

# --- Section 5: Pre-compute PLINK files needed by plinkQC --------------------
# plinkQC expects certain PLINK output files to exist in qcdir.
# We generate them explicitly here because plinkQC's internal calls can fail
# silently when the PLINK path contains spaces, and because the FAM file has
# non-zero paternal/maternal IDs requiring --make-founders.

precompute_plink_files <- function() {
  plink <- function(...) {
    system2(path2plink, args = c(...))
  }

  # 5a. Missingness (produces .imiss and .lmiss)
  if (!file.exists(file.path(qcdir, paste0(name, ".imiss")))) {
    message("Computing missingness...")
    plink("--bfile", file.path(indir, name), "--missing",
          "--out", file.path(qcdir, name))
  }

  # 5b. Heterozygosity (produces .het)
  if (!file.exists(file.path(qcdir, paste0(name, ".het")))) {
    message("Computing heterozygosity...")
    plink("--bfile", file.path(indir, name), "--het",
          "--out", file.path(qcdir, name))
  }

  # 5c. Sex check by chrX F-statistic is INTENTIONALLY DISABLED.
  # The source genotype data has corrupted female chrX calls (heterozygosity
  # ~26% instead of ~50%, observed both pre- and post-liftover), so every
  # female's F clusters around 0.49 and gets flagged as ambiguous. We instead
  # cross-check FAM PEDSEX against a trusted external phenotype CSV — see
  # cross_check_sex_with_phenotype().

  # 5d. LD pruning (produces .prune.in and .prune.out)
  if (!file.exists(file.path(qcdir, paste0(name, ".prune.in")))) {
    message("LD pruning...")
    plink("--bfile", file.path(indir, name),
          "--make-founders",
          "--indep-pairwise", "50", "5", "0.2",
          "--out", file.path(qcdir, name))
  }

  # 5e. IBD / relatedness (produces .genome) — uses pruned SNPs
  if (!file.exists(file.path(qcdir, paste0(name, ".genome")))) {
    message("Computing IBD (--genome) on pruned SNPs... this may take a few minutes")
    plink("--bfile", file.path(indir, name),
          "--make-founders",
          "--extract", file.path(qcdir, paste0(name, ".prune.in")),
          "--genome",
          "--out", file.path(qcdir, name))
  }

  # 5f. Allele frequencies (produces .frq) — needed by per-marker QC
  if (!file.exists(file.path(qcdir, paste0(name, ".frq")))) {
    message("Computing allele frequencies...")
    plink("--bfile", file.path(indir, name),
          "--make-founders",
          "--freq",
          "--out", file.path(qcdir, name))
  }

  # 5g. Hardy-Weinberg (produces .hwe) — needed by per-marker QC
  hwe_file <- file.path(qcdir, paste0(name, ".hwe"))
  if (!file.exists(hwe_file)) {
    message("Computing HWE statistics...")
    plink("--bfile", file.path(indir, name),
          "--make-founders",
          "--hardy",
          "--out", file.path(qcdir, name))
  }
  # Fix TEST column: plinkQC expects "ALL" but PLINK writes "ALL(NP)" when
  # all phenotypes are missing (-9). Patch in place.
  hwe_raw <- readLines(hwe_file)
  if (any(grepl("ALL\\(NP\\)", hwe_raw))) {
    message("Patching .hwe TEST column: ALL(NP) -> ALL")
    hwe_raw <- gsub("ALL\\(NP\\)", "ALL    ", hwe_raw)
    writeLines(hwe_raw, hwe_file)
  }

  message("All PLINK pre-computation files ready.")
}

# --- Section 5b: Sex cross-check from external phenotype ----------------------
# Replaces the chrX F-statistic sex check (disabled because the source data
# has corrupted female chrX genotypes — see precompute_plink_files()).
# Reads the phenotype CSV (expects columns "Subject" and "Gender" with M/F),
# joins to the FAM by IID == Subject, and flags samples where FAM PEDSEX
# disagrees with the trusted Gender. Mismatches are written to
# .fail-sexcheck.IDs and dropped after cleanData via PLINK --remove.
# Samples missing from the phenotype CSV are reported in the per-sample
# CSV but NOT dropped at this stage (they'll be excluded later when the
# downstream analyses join on Subject).

cross_check_sex_with_phenotype <- function() {
  message("")
  message("=============================================================")
  message("Sex cross-check vs phenotype CSV...")
  message("=============================================================")
  message("Phenotype source: ", pheno_sex_csv)

  if (!file.exists(pheno_sex_csv)) {
    stop("Phenotype CSV not found: ", pheno_sex_csv,
         "\nProvide a different path with --pheno-sex-csv.")
  }

  pheno <- fread(pheno_sex_csv)
  required_cols <- c("Subject", "Gender")
  missing_cols  <- setdiff(required_cols, names(pheno))
  if (length(missing_cols) > 0) {
    stop("Phenotype CSV missing required column(s): ",
         paste(missing_cols, collapse = ", "),
         "\nExpected columns: Subject, Gender (M/F).")
  }

  fam <- fread(file.path(indir, paste0(name, ".fam")), header = FALSE,
               col.names = c("FID", "IID", "PAT", "MAT", "PEDSEX", "PHENO"))

  # Coerce IIDs to character to avoid type mismatches on join
  pheno[, Subject := as.character(Subject)]
  fam[,   IID     := as.character(IID)]
  fam[,   FID     := as.character(FID)]

  pheno_sex_map <- pheno[, .(IID = Subject,
                             pheno_gender = toupper(trimws(as.character(Gender))))]

  per_sex <- merge(fam[, .(FID, IID, PEDSEX)], pheno_sex_map,
                   by = "IID", all.x = TRUE)

  # M -> 1, F -> 2; anything else -> NA
  per_sex[, pheno_pedsex := fifelse(pheno_gender == "M", 1L,
                            fifelse(pheno_gender == "F", 2L, NA_integer_))]

  per_sex[, missing_in_pheno := is.na(pheno_gender)]
  per_sex[, sex_mismatch     := !missing_in_pheno &
                                !is.na(PEDSEX) &
                                PEDSEX != pheno_pedsex]
  # Only mismatches drive the QC drop. missing_in_pheno is reported but
  # not enforced here (downstream analyses gate on phenotype availability).
  per_sex[, fail_sex         := sex_mismatch]

  n_total    <- nrow(per_sex)
  n_mismatch <- sum(per_sex$sex_mismatch)
  n_missing  <- sum(per_sex$missing_in_pheno)
  n_match    <- n_total - n_mismatch

  message("  Total FAM samples:           ", n_total)
  message("  Match phenotype sex:         ", n_match)
  message("  Mismatch (FAM != phenotype): ", n_mismatch, "  [DROPPED]")
  message("  Missing from phenotype CSV:  ", n_missing,  "  [reported only]")

  fail_path <- file.path(qcdir, paste0(name, ".fail-sexcheck.IDs"))
  if (n_mismatch > 0) {
    write.table(per_sex[fail_sex == TRUE, .(FID, IID)],
                fail_path,
                quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t")
  } else {
    file.create(fail_path)
  }

  mismatch_csv <- file.path(reports_dir, "B1_sex_mismatch.csv")
  fwrite(per_sex[sex_mismatch == TRUE | missing_in_pheno == TRUE,
                 .(FID, IID, PEDSEX, pheno_gender, pheno_pedsex,
                   sex_mismatch, missing_in_pheno)],
         mismatch_csv)
  message("  Cross-check audit CSV: ", mismatch_csv)

  list(per_sex = per_sex,
       n_total = n_total, n_match = n_match,
       n_mismatch = n_mismatch, n_missing = n_missing,
       fail_path = fail_path, mismatch_csv = mismatch_csv)
}

# --- Section 6: Per-individual QC ---------------------------------------------

run_per_individual_qc <- function() {
  message("")
  message("=============================================================")
  message("Running per-individual QC...")
  message("=============================================================")

  fail_individuals <- suppress_upstream_warnings(perIndividualQC(
    indir           = indir,
    name            = name,
    qcdir           = qcdir,
    path2plink      = path2plink,
    path2plink2     = path2plink2,

    # Sex check disabled — source chrX genotypes are corrupted for females
    # (see precompute_plink_files() and cross_check_sex_with_phenotype()).
    dont.check_sex  = TRUE,

    # Heterozygosity and missingness
    dont.check_het_and_miss = FALSE,
    imissTh         = imissTh,
    hetTh           = hetTh,

    # Relatedness
    dont.check_relatedness   = FALSE,
    highIBDTh                = highIBDTh,
    mafThRelatedness         = 0.1,
    filter_high_ldregion     = TRUE,
    genomebuild              = genomebuild,

    # Ancestry handled separately (after marker QC)
    dont.ancestry_prediction = TRUE,

    interactive     = FALSE,
    verbose         = TRUE,
    showPlinkOutput = TRUE
  ))

  return(fail_individuals)
}

# --- Section 7: Per-marker QC (manual) ----------------------------------------
# We bypass perMarkerQC() because it re-runs PLINK internally without
# --make-founders, overwriting our pre-computed files and causing errors.

run_per_marker_qc <- function() {
  message("")
  message("=============================================================")
  message("Running per-marker QC (manual)...")
  message("=============================================================")

  results <- list()

  # 7a. SNP missingness
  lmiss <- fread(file.path(qcdir, paste0(name, ".lmiss")))
  fail_snpmiss <- lmiss[lmiss$F_MISS > lmissTh, ]
  message("Failed SNP missingness (F_MISS > ", lmissTh, "): ", nrow(fail_snpmiss))
  writeLines(fail_snpmiss$SNP,
             file.path(qcdir, paste0(name, ".fail-lmiss.IDs")))

  results$fail_snpmissing <- fail_snpmiss
  results$p_snpmissing <- ggplot(lmiss, aes(x = F_MISS)) +
    geom_histogram(bins = 50, fill = "steelblue", colour = "black", linewidth = 0.2) +
    geom_vline(xintercept = lmissTh, colour = "red", linetype = "dashed") +
    labs(title = "SNP missingness", x = "Fraction missing", y = "Count") +
    theme_bw()

  # 7b. Hardy-Weinberg equilibrium
  hwe <- fread(file.path(qcdir, paste0(name, ".hwe")))
  hwe <- hwe[grepl("^ALL", hwe$TEST), ]
  hwe$P <- as.numeric(hwe$P)
  fail_hwe <- hwe[!is.na(hwe$P) & hwe$P < hweTh, ]
  message("Failed HWE (P < ", hweTh, "): ", nrow(fail_hwe))
  writeLines(fail_hwe$SNP,
             file.path(qcdir, paste0(name, ".fail-hwe.IDs")))

  results$fail_hwe <- fail_hwe
  results$p_hwe <- ggplot(hwe[!is.na(hwe$P) & hwe$P > 0, ], aes(x = -log10(P))) +
    geom_histogram(bins = 50, fill = "steelblue", colour = "black", linewidth = 0.2) +
    geom_vline(xintercept = -log10(hweTh), colour = "red", linetype = "dashed") +
    labs(title = "HWE p-values", x = "-log10(p)", y = "Count") +
    theme_bw()

  # 7c. Minor allele frequency
  frq <- fread(file.path(qcdir, paste0(name, ".frq")))
  frq$MAF <- as.numeric(frq$MAF)
  fail_maf <- frq[!is.na(frq$MAF) & frq$MAF < mafTh, ]
  message("Failed MAF (< ", mafTh, "): ", nrow(fail_maf))
  writeLines(fail_maf$SNP,
             file.path(qcdir, paste0(name, ".fail-maf.IDs")))

  results$fail_maf <- fail_maf
  results$p_maf <- ggplot(frq[!is.na(frq$MAF), ], aes(x = MAF)) +
    geom_histogram(bins = 50, fill = "steelblue", colour = "black", linewidth = 0.2) +
    geom_vline(xintercept = mafTh, colour = "red", linetype = "dashed") +
    labs(title = "Minor allele frequency", x = "MAF", y = "Count") +
    theme_bw()

  # Total unique failing markers
  all_fail_snps <- unique(c(fail_snpmiss$SNP, fail_hwe$SNP, fail_maf$SNP))
  results$nr_fail_markers <- length(all_fail_snps)
  message("Total unique markers failing: ", length(all_fail_snps))

  return(results)
}

# --- Section 8: Combined cleanup ---------------------------------------------
# Ancestry inference is delegated to B1b_ancestry_PCA_mahalanobis.py, which
# computes a within-sample PCA on B1's .clean triplet for use as nuisance
# covariates in C1/C3/C3b (no participant exclusion). cleanData() therefore
# runs with filterAncestry = FALSE.

run_clean_data <- function() {
  message("")
  message("=============================================================")
  message("Running combined data cleanup...")
  message("=============================================================")

  clean <- suppress_upstream_warnings(cleanData(
    indir       = indir,
    name        = name,
    qcdir       = qcdir,
    path2plink  = path2plink,

    # Sex check off (chrX corruption in source; phenotype cross-check is
    # applied separately via PLINK --remove after cleanData).
    # Ancestry off (delegated to B1b_ancestry_PCA_mahalanobis.py).
    filterSex               = FALSE,
    filterHeterozygosity    = TRUE,
    filterSampleMissingness = TRUE,
    filterRelated           = FALSE,
    filterAncestry          = FALSE,

    # Marker filters
    filterSNPMissingness    = TRUE,
    filterHWE               = TRUE,
    filterMAF               = TRUE,
    lmissTh                 = lmissTh,
    hweTh                   = hweTh,
    mafTh                   = mafTh,
    macTh                   = NULL,

    verbose         = TRUE,
    showPlinkOutput = TRUE
  ))

  return(clean)
}

# Apply the phenotype-derived sex filter to cleanData's output by running
# `plink --remove .fail-sexcheck.IDs --make-bed`, overwriting the .clean
# triplet in place. No-op when the fail file is empty.
apply_sex_remove <- function(sex_check) {
  fail_path <- sex_check$fail_path
  if (!file.exists(fail_path) || file.info(fail_path)$size == 0) {
    message("Sex cross-check produced no drops; .clean triplet unchanged.")
    return(invisible(NULL))
  }
  clean_prefix <- file.path(qcdir, paste0(name, ".clean"))
  tmp_prefix   <- file.path(qcdir, paste0(name, ".clean.tmp"))
  message("Applying sex cross-check filter to .clean triplet...")
  status <- system2(
    path2plink,
    args = c(
      "--bfile",   shQuote(clean_prefix),
      "--remove",  shQuote(fail_path),
      "--make-bed",
      "--out",     shQuote(tmp_prefix)
    ),
    stdout = TRUE, stderr = TRUE
  )
  for (line in status) message("  [plink] ", line)
  for (ext in c(".bed", ".bim", ".fam", ".log")) {
    src <- paste0(tmp_prefix, ext); dst <- paste0(clean_prefix, ext)
    if (file.exists(src)) file.rename(src, dst)
  }
  invisible(NULL)
}

# --- Section 10: Report and plot saving ---------------------------------------

save_qc_plot <- function(plot_obj, filename, width = 12, height = 10) {
  filepath <- file.path(figures_dir, filename)
  tryCatch({
    suppress_upstream_warnings(ggsave(
      filename = filepath,
      plot     = plot_obj,
      width    = width,
      height   = height,
      dpi      = 150
    ))
    message("Saved plot: ", filepath)
  }, error = function(e) {
    message("Warning: Could not save plot ", filename, ": ", e$message)
  })
}

write_qc_report <- function(lines, filename = "B1_plinkQC_genotype_qc_report.txt") {
  filepath <- file.path(reports_dir, filename)
  writeLines(lines, filepath)
  message("Saved report: ", filepath)
}

# --- Section 8b: Per-sample drop-reason breakdown ----------------------------
# Re-derives pass/fail for every sample directly from the raw PLINK outputs
# (.imiss, .het) plus the external sex cross-check, then writes one row per
# sample with the reason(s). Sex mismatches are first-class drop reasons
# (applied via the post-cleanData PLINK --remove pass). Ancestry filtering
# happens in B1b and is reported there.

summarize_drops <- function(sex_check) {
  imiss <- fread(file.path(qcdir, paste0(name, ".imiss")))
  het   <- fread(file.path(qcdir, paste0(name, ".het")))

  # Het threshold: mean(F) ± hetTh * sd(F), matching plinkQC's convention
  het[, F := as.numeric(F)]
  het_mean <- mean(het$F, na.rm = TRUE)
  het_sd   <- sd(het$F,   na.rm = TRUE)
  het_lo   <- het_mean - hetTh * het_sd
  het_hi   <- het_mean + hetTh * het_sd

  per <- merge(imiss[, .(FID, IID, F_MISS)],
               het[,   .(FID, IID, F_HET = F)],
               by = c("FID", "IID"), all = TRUE)
  per[, FID := as.character(FID)][, IID := as.character(IID)]
  per[, fail_imiss := !is.na(F_MISS) & F_MISS > imissTh]
  per[, fail_het   := !is.na(F_HET)  & (F_HET < het_lo | F_HET > het_hi)]

  if (!is.null(sex_check)) {
    sx <- as.data.table(sex_check$per_sex)[, .(FID, IID, PEDSEX,
                                               pheno_gender, pheno_pedsex,
                                               sex_mismatch, missing_in_pheno)]
    sx[, FID := as.character(FID)][, IID := as.character(IID)]
    per <- merge(per, sx, by = c("FID", "IID"), all.x = TRUE)
  }
  if (!"sex_mismatch"     %in% names(per)) per[, sex_mismatch := FALSE]
  if (!"missing_in_pheno" %in% names(per)) per[, missing_in_pheno := FALSE]
  per[is.na(sex_mismatch),     sex_mismatch     := FALSE]
  per[is.na(missing_in_pheno), missing_in_pheno := FALSE]

  per[, fail_sex := sex_mismatch]
  per[, dropped  := fail_imiss | fail_het | fail_sex]
  per[, reasons  := apply(.SD, 1, function(r) {
        paste(c("imiss", "het", "sex")[as.logical(r)], collapse = ";")
      }),
      .SDcols = c("fail_imiss", "fail_het", "fail_sex")]

  full_path    <- file.path(reports_dir, "B1_per_sample_qc.csv")
  dropped_path <- file.path(reports_dir, "B1_dropped_individuals.csv")
  fwrite(per, full_path)
  fwrite(per[dropped == TRUE], dropped_path)
  message("Saved per-sample QC: ", full_path)
  message("Saved dropped list:  ", dropped_path)

  n_imiss <- sum(per$fail_imiss)
  n_het   <- sum(per$fail_het)
  n_sex   <- sum(per$fail_sex)
  n_total <- sum(per$dropped)
  n_missing  <- sum(per$missing_in_pheno)

  pair <- function(a, b, lab) {
    paste0("    ", lab, ": ", sum(per[[a]] & per[[b]]))
  }

  lines <- c(
    "DROPPED INDIVIDUALS BREAKDOWN (authoritative — recomputed from raw PLINK files):",
    paste0("  Failed sample missingness:  ", n_imiss, "  (F_MISS > ", imissTh, ")"),
    paste0("  Failed heterozygosity:      ", n_het,
           "  (F outside [", round(het_lo, 4), ", ", round(het_hi, 4),
           "] = mean ± ", hetTh, " SD)"),
    paste0("    het only (not imiss):     ", sum(per$fail_het & !per$fail_imiss)),
    paste0("    imiss only (not het):     ", sum(per$fail_imiss & !per$fail_het)),
    paste0("    both het AND imiss:       ", sum(per$fail_het &  per$fail_imiss)),
    paste0("  Failed sex cross-check:     ", n_sex,
           "  (FAM PEDSEX != phenotype Gender)"),
    "",
    "  Samples missing from phenotype CSV (reported, not dropped):",
    paste0("    n = ", n_missing),
    "",
    "  Pairwise overlap (samples failing BOTH checks):",
    pair("fail_imiss", "fail_het", "imiss & het"),
    pair("fail_imiss", "fail_sex", "imiss & sex"),
    pair("fail_het",   "fail_sex", "het & sex"),
    "",
    paste0("  TOTAL UNIQUE DROPPED: ", n_total,
           "  (sum of categories = ", n_imiss + n_het + n_sex,
           "; samples failing >1 check = ",
           n_imiss + n_het + n_sex - n_total, ")"),
    "",
    "  Per-sample CSVs:",
    paste0("    All samples + flags:  ", full_path),
    paste0("    Dropped only:         ", dropped_path)
  )

  list(lines = lines, per_sample = per, n_total_dropped = n_total)
}

# --- Section 11: Main execution -----------------------------------------------

main <- function() {

  # --- Pre-compute PLINK files ---
  precompute_plink_files()

  # --- Sex cross-check (replaces broken chrX F-statistic check) ---
  sex_check <- cross_check_sex_with_phenotype()
  mismatch_status <- if (apply_sex_filter) "[DROPPED]" else "[would be dropped — filter bypassed]"
  apply_status <- if (apply_sex_filter) "YES" else "NO (sanity-check mode — mismatches retained)"
  report_lines <- c(report_lines,
    "SEX CROSS-CHECK (FAM PEDSEX vs phenotype CSV):",
    paste("  Phenotype source:           ", pheno_sex_csv),
    paste("  Total FAM samples:          ", sex_check$n_total),
    paste("  Match phenotype sex:        ", sex_check$n_match),
    paste("  Mismatch (FAM != phenotype):", sex_check$n_mismatch, mismatch_status),
    paste("  Missing from phenotype CSV: ", sex_check$n_missing,  " [reported only]"),
    paste("  Audit CSV:                  ", sex_check$mismatch_csv),
    paste("  Apply filter:               ", apply_status),
    "  Note: chrX F-statistic check is DISABLED (source data has corrupted",
    "        female chrX heterozygosity). When the filter is applied, sex",
    "        mismatches are dropped via a post-cleanData PLINK --remove pass.",
    ""
  )

  # --- Per-individual QC ---
  fail_individuals <- run_per_individual_qc()

  # Extract failure counts (sex check disabled; no fail_sex list)
  n_het_fail <- if (!is.null(fail_individuals$fail_list$fail_het_imiss))
                nrow(fail_individuals$fail_list$fail_het_imiss) else 0
  n_rel_fail <- if (!is.null(fail_individuals$fail_list$fail_relatedness))
                nrow(fail_individuals$fail_list$fail_relatedness) else 0
  message("")
  message("Per-individual QC results:")
  message("  Failed het/missingness:  ", n_het_fail)
  message("  Failed relatedness:      ", n_rel_fail)

  report_lines <- c(report_lines,
    "PER-INDIVIDUAL QC:",
    paste("  Failed het/missingness:", n_het_fail),
    paste("  Failed relatedness:    ", n_rel_fail)
  )

  # Overview plot
  tryCatch({
    overview_ind <- suppress_upstream_warnings(overviewPerIndividualQC(
      fail_individuals,
      interactive = FALSE
    ))
    save_qc_plot(overview_ind$p_overview, "B1_QC_overview.png", width = 10, height = 8)
    n_total_ind_fail <- overview_ind$nr_fail_samples
    report_lines <- c(report_lines,
      paste("  Total unique individuals failing:", n_total_ind_fail)
    )
  }, error = function(e) {
    message("Warning: Could not generate individual QC overview: ", e$message)
  })

  report_lines <- c(report_lines, "")

  # Save per-individual diagnostic plots (no sex plot — F-statistic check off)
  if (!is.null(fail_individuals$p_het_imiss)) {
    save_qc_plot(fail_individuals$p_het_imiss, "B1_het_vs_miss.png", width = 8, height = 6)
  }
  if (!is.null(fail_individuals$p_relatedness)) {
    save_qc_plot(fail_individuals$p_relatedness, "B1_relatedness.png", width = 8, height = 6)
  }

  # --- Per-marker QC ---
  fail_markers <- run_per_marker_qc()

  n_snpmiss_fail <- nrow(fail_markers$fail_snpmissing)
  n_hwe_fail     <- nrow(fail_markers$fail_hwe)
  n_maf_fail     <- nrow(fail_markers$fail_maf)
  n_total_mrk_fail <- fail_markers$nr_fail_markers

  message("")
  message("Per-marker QC results:")
  message("  Failed SNP missingness: ", n_snpmiss_fail)
  message("  Failed HWE:             ", n_hwe_fail)
  message("  Failed MAF:             ", n_maf_fail)
  message("  Total unique markers:   ", n_total_mrk_fail)

  report_lines <- c(report_lines,
    "PER-MARKER QC:",
    paste("  Failed SNP missingness:", n_snpmiss_fail),
    paste("  Failed HWE:           ", n_hwe_fail),
    paste("  Failed MAF:           ", n_maf_fail),
    paste("  Total unique markers failing:", n_total_mrk_fail),
    ""
  )

  # Save per-marker diagnostic plots
  if (!is.null(fail_markers$p_snpmissing)) {
    save_qc_plot(fail_markers$p_snpmissing, "B1_snp_missingness.png", width = 8, height = 6)
  }
  if (!is.null(fail_markers$p_hwe)) {
    save_qc_plot(fail_markers$p_hwe, "B1_hwe.png", width = 8, height = 6)
  }
  if (!is.null(fail_markers$p_maf)) {
    save_qc_plot(fail_markers$p_maf, "B1_maf.png", width = 8, height = 6)
  }

  report_lines <- c(report_lines,
    "ANCESTRY CHECK:",
    "  Delegated to B1b_ancestry_PCA_mahalanobis.py (1000G-merged PCA).",
    ""
  )

  # --- Per-individual drop-reason breakdown ---
  drops <- tryCatch(
    summarize_drops(sex_check),
    error = function(e) {
      message("Warning: drop-reason breakdown failed: ", e$message)
      list(lines = c("DROPPED INDIVIDUALS BREAKDOWN:",
                     paste("  FAILED to compute:", e$message), ""),
           n_total_dropped = NA_integer_)
    }
  )
  report_lines <- c(report_lines, drops$lines, "")

  # --- Combined cleanup ---
  clean <- run_clean_data()
  if (apply_sex_filter) {
    apply_sex_remove(sex_check)
  } else {
    message("Sex filter BYPASSED (--no-apply-sex-filter): .clean triplet ",
            "retains all QC-passing samples regardless of PEDSEX vs phenotype ",
            "Gender agreement. ", sex_check$n_mismatch,
            " mismatching sample(s) recorded in audit CSV but not removed.")
  }

  # Count final samples and SNPs
  clean_prefix <- file.path(qcdir, paste0(name, ".clean"))
  if (file.exists(paste0(clean_prefix, ".bim"))) {
    final_bim <- fread(paste0(clean_prefix, ".bim"), header = FALSE)
    final_fam <- fread(paste0(clean_prefix, ".fam"), header = FALSE)
    final_snp_count <- nrow(final_bim)
    final_ind_count <- nrow(final_fam)
    rm(final_bim, final_fam)
  } else {
    final_snp_count <- "N/A (clean files not found)"
    final_ind_count <- "N/A (clean files not found)"
  }

  # Reconcile: actual drop count from cleanData vs breakdown above.
  # A mismatch usually means cleanData applied an additional filter or that
  # the raw .imiss/.het/.sexcheck thresholds drift from plinkQC's internals.
  if (is.numeric(final_ind_count) && !is.na(drops$n_total_dropped)) {
    actual_dropped <- initial_ind_count - final_ind_count
    if (actual_dropped != drops$n_total_dropped) {
      msg <- paste0(
        "NOTE: cleanData() removed ", actual_dropped,
        " individuals; per-sample breakdown above totals ",
        drops$n_total_dropped, " (difference = ",
        actual_dropped - drops$n_total_dropped, ")."
      )
      message(msg)
      report_lines <- c(report_lines, msg, "")
    }
  }

  message("")
  message("=============================================================")
  message("QC COMPLETE")
  message("=============================================================")
  message("Final data: ", final_snp_count, " SNPs, ", final_ind_count, " individuals")

  # --- Write report ---
  report_lines <- c(report_lines,
    paste(rep("-", 72), collapse = ""),
    "FINAL CLEANED DATA:",
    paste("  SNPs:        ", final_snp_count),
    paste("  Individuals: ", final_ind_count),
    "",
    paste(rep("-", 72), collapse = ""),
    "THRESHOLDS USED:",
    paste("  Sample missingness (imissTh):", imissTh),
    paste("  Heterozygosity SD (hetTh):   ", hetTh),
    paste("  Sex check (chrX F):          ", "DISABLED (corrupted female chrX in source)"),
    paste("  Sex cross-check source:      ", pheno_sex_csv),
    paste("  IBD threshold (highIBDTh):   ", highIBDTh),
    paste("  MAF for relatedness:         ", 0.1),
    paste("  SNP missingness (lmissTh):   ", lmissTh),
    paste("  HWE p-value (hweTh):         ", hweTh),
    paste("  MAF threshold (mafTh):       ", mafTh),
    paste("  Genome build:                ", genomebuild),
    paste("  Ancestry filter:             ", "delegated to B1b"),
    "",
    paste(rep("-", 72), collapse = ""),
    "OUTPUT FILES:",
    paste("  Clean data:  ", clean_prefix, ".bed/bim/fam"),
    paste("  Figures:     ", figures_dir),
    paste("  This report: ", file.path(reports_dir, "B1_plinkQC_genotype_qc_report.txt")),
    "",
    paste(rep("=", 72), collapse = "")
  )

  write_qc_report(report_lines)

  message("")
  message("Report saved to: ", file.path(reports_dir, "B1_plinkQC_genotype_qc_report.txt"))
  message("Clean data at:   ", clean_prefix, ".bed/bim/fam")
  message("Figures saved to: ", figures_dir)
}

# --- Run main with error handling ---------------------------------------------

tryCatch({
  main()
}, error = function(e) {
  message("")
  message("FATAL ERROR: ", e$message)
  message("")
  message("Partial report written.")
  report_lines <- c(report_lines,
    "",
    paste("FATAL ERROR:", e$message),
    paste("Pipeline terminated at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  )
  write_qc_report(report_lines)
  quit(status = 1)
})
