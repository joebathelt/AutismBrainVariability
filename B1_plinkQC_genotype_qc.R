#!/usr/bin/env Rscript

# =============================================================================
# B1_plinkQC_genotype_qc.R
# Standalone genotype quality control pipeline using plinkQC
# BrainCompensation Project
#
# Performs per-individual QC, per-marker QC, ancestry checking (via
# plinkQC's pretrained RF classifier), and combined data cleanup.
# Produces diagnostic plots and a text QC report.
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
  cat("  --refdir        Reference data directory (relative to project) [data/reference/1000Genomes]\n")
  cat("  --path2plink    Path to PLINK v1.9 executable [auto-detect]\n")
  cat("  --path2plink2   Path to PLINK 2.0 executable [auto-detect]\n")
  cat("  --maf           Minor allele frequency threshold [0.01]\n")
  cat("  --hwe           HWE p-value threshold [1e-6]\n")
  cat("  --geno          SNP missingness threshold [0.01]\n")
  cat("  --mind          Sample missingness threshold [0.03]\n")
  cat("  --het           Heterozygosity SD threshold [3]\n")
  cat("  --ibd           IBD relatedness threshold [0.1875]\n")
  cat("  --ancestry-th   Ancestry SD threshold from EUR center [1.5]\n")
  cat("  --skip-ancestry Skip ancestry check [FALSE]\n")
  cat("  --help          Show this help message\n")
  quit(status = 1)
}

if (length(args) == 0) usage()

# Initialize with defaults
project_dir   <- NULL
indir_rel     <- "data/raw_anonymised"
name          <- "Neuro_Chip_anonymised"
qcdir_rel     <- "data/plinkQC_output"
refdir_rel    <- "data/reference/1000Genomes"
path2plink    <- NULL
path2plink2   <- NULL
mafTh         <- 0.01
hweTh         <- 1e-6
lmissTh       <- 0.01
imissTh       <- 0.03
hetTh         <- 3
highIBDTh     <- 0.1875
europeanTh    <- 1.5
skip_ancestry <- FALSE

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
  } else if (args[i] == "--refdir") {
    refdir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2plink") {
    path2plink <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2plink2") {
    path2plink2 <- args[i + 1]; i <- i + 2
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
  } else if (args[i] == "--ancestry-th") {
    europeanTh <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--skip-ancestry") {
    skip_ancestry <- TRUE; i <- i + 1
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
refdir     <- file.path(project_dir, refdir_rel)
figures_dir <- file.path(project_dir, "figures")
reports_dir <- file.path(project_dir, "reports")

# Reference populations for ancestry check
refPopulation <- c("CEU", "TSI")

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
message("Ref dir:    ", refdir)
message("")

# Validate input files
check_required_files(indir, name)

# Detect PLINK
if (is.null(path2plink)) path2plink <- detect_plink()
if (is.null(path2plink2)) path2plink2 <- detect_plink2()
message("PLINK v1.9: ", path2plink)
message("PLINK 2.0:  ", ifelse(is.null(path2plink2), "not found", path2plink2))

# Create output directories
for (d in c(qcdir, refdir, figures_dir, reports_dir)) {
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

  # 5c. Sex check (produces .sexcheck)
  if (!file.exists(file.path(qcdir, paste0(name, ".sexcheck")))) {
    message("Running sex check...")
    plink("--bfile", file.path(indir, name), "--check-sex",
          "--out", file.path(qcdir, name))
  }

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

# --- Section 6: Per-individual QC ---------------------------------------------

run_per_individual_qc <- function() {
  message("")
  message("=============================================================")
  message("Running per-individual QC...")
  message("=============================================================")

  fail_individuals <- perIndividualQC(
    indir           = indir,
    name            = name,
    qcdir           = qcdir,
    path2plink      = path2plink,
    path2plink2     = path2plink2,

    # Sex check
    dont.check_sex  = FALSE,
    maleTh          = 0.8,
    femaleTh        = 0.2,

    # Heterozygosity and missingness
    dont.check_het_and_miss = FALSE,
    imissTh         = imissTh,
    hetTh           = hetTh,

    # Relatedness
    dont.check_relatedness   = FALSE,
    highIBDTh                = highIBDTh,
    mafThRelatedness         = 0.1,
    filter_high_ldregion     = TRUE,
    genomebuild              = "hg19",

    # Ancestry handled separately (after marker QC)
    dont.ancestry_prediction = TRUE,

    interactive     = FALSE,
    verbose         = TRUE,
    showPlinkOutput = TRUE
  )

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

# --- Section 8: Ancestry check (RF prediction) --------------------------------
# Uses plinkQC's pretrained random forest classifier via ancestry_prediction().
# Requires plink2 and loading matrices from plinkQCAncestryData.

run_ancestry_check <- function() {
  message("")
  message("=============================================================")
  message("Running ancestry check (RF prediction)...")
  message("=============================================================")

  # Locate loading matrices: check package bundled data first, then local path
  load_mat <- system.file("extdata", "load_mat",
                           "all_hg38.pca",
                           package = "plinkQC")
  if (nchar(load_mat) == 0) {
    load_mat <- file.path(refdir, "loading_matrix",
                           "all_hg38.pca")
    if (!file.exists(paste0(load_mat, ".eigenvec.allele")) &&
        !file.exists(paste0(load_mat, ".acount"))) {
      stop("Loading matrices not found at: ", load_mat, "\n",
           "Download from: https://github.com/meyer-lab-cshl/plinkQCAncestryData\n",
           "Unzip into:    ", file.path(refdir, "loading_matrix/"))
    }
  }

  if (is.null(path2plink2)) {
    stop("PLINK 2.0 is required for ancestry_prediction(). ",
         "Please install plink2 or specify --path2plink2.")
  }

  ancestry <- ancestry_prediction(
    indir         = indir,
    name          = name,
    qcdir         = qcdir,
    path2plink2   = path2plink2,
    path2load_mat = load_mat,
    plink2format  = FALSE,    # input is plink1 .bed/.bim/.fam
    var_format    = FALSE,    # variant IDs are rsIDs, not chr:pos[hg38]
    excludeAncestry = c("AFR", "AMR", "EAS", "SAS"), # only keep EUR
    verbose       = TRUE,
    showPlinkOutput = TRUE
  )

  return(ancestry)
}

# --- Section 9: Combined cleanup ---------------------------------------------

run_clean_data <- function(do_ancestry) {
  message("")
  message("=============================================================")
  message("Running combined data cleanup...")
  message("=============================================================")

  clean <- cleanData(
    indir       = indir,
    name        = name,
    qcdir       = qcdir,
    path2plink  = path2plink,

    # Individual filters
    filterSex               = TRUE,
    filterHeterozygosity    = TRUE,
    filterSampleMissingness = TRUE,
    filterRelated           = FALSE,
    filterAncestry          = do_ancestry,

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
  )

  return(clean)
}

# --- Section 10: Report and plot saving ---------------------------------------

save_qc_plot <- function(plot_obj, filename, width = 12, height = 10) {
  filepath <- file.path(figures_dir, filename)
  tryCatch({
    ggsave(
      filename = filepath,
      plot     = plot_obj,
      width    = width,
      height   = height,
      dpi      = 150
    )
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

# --- Section 11: Main execution -----------------------------------------------

main <- function() {

  # --- Pre-compute PLINK files ---
  precompute_plink_files()

  # --- Per-individual QC ---
  fail_individuals <- run_per_individual_qc()

  # Extract failure counts
  n_sex_fail   <- length(fail_individuals$fail_sex$FID)
  n_het_fail   <- length(fail_individuals$fail_het_imiss$FID)
  n_rel_fail   <- length(fail_individuals$fail_relatedness$FID)

  message("")
  message("Per-individual QC results:")
  message("  Failed sex check:       ", n_sex_fail)
  message("  Failed het/missingness:  ", n_het_fail)
  message("  Failed relatedness:      ", n_rel_fail)

  report_lines <- c(report_lines,
    "PER-INDIVIDUAL QC:",
    paste("  Failed sex check:      ", n_sex_fail),
    paste("  Failed het/missingness:", n_het_fail),
    paste("  Failed relatedness:    ", n_rel_fail)
  )

  # Overview plot
  tryCatch({
    overview_ind <- overviewPerIndividualQC(
      fail_individuals,
      interactive = FALSE
    )
    save_qc_plot(overview_ind$p_overview, "B1_QC_overview.png", width = 10, height = 8)
    n_total_ind_fail <- overview_ind$nr_fail_samples
    report_lines <- c(report_lines,
      paste("  Total unique individuals failing:", n_total_ind_fail)
    )
  }, error = function(e) {
    message("Warning: Could not generate individual QC overview: ", e$message)
  })

  report_lines <- c(report_lines, "")

  # Save per-individual diagnostic plots
  if (!is.null(fail_individuals$p_sexcheck)) {
    save_qc_plot(fail_individuals$p_sexcheck, "B1_sexcheck.png", width = 8, height = 6)
  }
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

  # --- Ancestry check ---
  ancestry_done <- FALSE
  n_ancestry_fail <- 0

  if (!skip_ancestry) {
    tryCatch({
      ancestry <- run_ancestry_check()

      n_ancestry_fail <- length(ancestry$fail_ancestry$FID)
      ancestry_done <- TRUE

      message("")
      message("Ancestry check results:")
      message("  Failed ancestry:  ", n_ancestry_fail)

      report_lines <- c(report_lines,
        "ANCESTRY CHECK (RF prediction):",
        paste("  Failed ancestry:      ", n_ancestry_fail),
        ""
      )

      # Save ancestry plot
      if (!is.null(ancestry$p_ancestry)) {
        save_qc_plot(ancestry$p_ancestry, "B1_ancestry_check.png", width = 10, height = 8)
      }

    }, error = function(e) {
      message("")
      message("WARNING: Ancestry check failed: ", e$message)
      message("Continuing without ancestry filtering.")
      report_lines <<- c(report_lines,
        "ANCESTRY CHECK:",
        paste("  SKIPPED (error:", e$message, ")"),
        ""
      )
    })
  } else {
    message("")
    message("Ancestry check: SKIPPED (--skip-ancestry)")
    report_lines <- c(report_lines,
      "ANCESTRY CHECK:",
      "  SKIPPED (--skip-ancestry flag)",
      ""
    )
  }

  # --- Combined cleanup ---
  clean <- run_clean_data(do_ancestry = ancestry_done)

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
    paste("  Male F threshold:            ", 0.8),
    paste("  Female F threshold:          ", 0.2),
    paste("  IBD threshold (highIBDTh):   ", highIBDTh),
    paste("  MAF for relatedness:         ", 0.1),
    paste("  SNP missingness (lmissTh):   ", lmissTh),
    paste("  HWE p-value (hweTh):         ", hweTh),
    paste("  MAF threshold (mafTh):       ", mafTh),
    paste("  Ancestry SD (europeanTh):    ", europeanTh),
    paste("  Reference populations:       ", paste(refPopulation, collapse = ", ")),
    paste("  Genome build:                ", "hg19"),
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
