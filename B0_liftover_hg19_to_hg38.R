#!/usr/bin/env Rscript

# =============================================================================
# B0_liftover_hg19_to_hg38.R
# Standalone liftover from hg19 (GRCh37) to hg38 for the NeuroChip genotype data
# BrainCompensation Project
#
# Converts PLINK .bim coordinates from hg19 to hg38 using UCSC liftOver, drops
# variants that fail to lift or jump chromosomes, and rebuilds a PLINK triplet
# at the new coordinates so that B1's plinkQC ancestry projection (which expects
# hg38 variant IDs of the form chr:pos[hg38]) finds enough overlapping variants.
#
# Usage:
#   Rscript B0_liftover_hg19_to_hg38.R --project /path/to/BrainCompensation
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

required_packages <- c("data.table")
install_if_missing(required_packages)

library(data.table)

# --- Section 2: Command-line argument parsing ---------------------------------

args <- commandArgs(trailingOnly = TRUE)

usage <- function() {
  cat("Usage: Rscript B0_liftover_hg19_to_hg38.R --project PROJECT [OPTIONS]\n\n")
  cat("Required:\n")
  cat("  --project        Path to project root directory\n\n")
  cat("Optional:\n")
  cat("  --indir          Input PLINK directory (relative to project) [data/raw_anonymised]\n")
  cat("  --name           Input PLINK file prefix [Neuro_Chip_anonymised]\n")
  cat("  --outdir         Output directory (relative to project) [data/raw_anonymised_hg38]\n")
  cat("  --outname        Output PLINK file prefix [<name>_hg38]\n")
  cat("  --chain-dir      Chain file directory (relative to project) [data/reference/liftover]\n")
  cat("  --path2plink     Path to PLINK v1.9 executable [auto-detect]\n")
  cat("  --path2liftover  Path to UCSC liftOver executable [auto-detect]\n")
  cat("  --force          Re-run even if outputs exist\n")
  cat("  --help           Show this help message\n")
  quit(status = 1)
}

if (length(args) == 0) usage()

project_dir   <- NULL
indir_rel     <- "data/raw_anonymised"
name          <- "Neuro_Chip_anonymised"
outdir_rel    <- "data/raw_anonymised_hg38"
outname       <- NULL
chaindir_rel  <- "data/reference/liftover"
path2plink    <- NULL
path2liftover <- NULL
force_run     <- FALSE

i <- 1
while (i <= length(args)) {
  if (args[i] == "--project") {
    project_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--indir") {
    indir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--name") {
    name <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--outdir") {
    outdir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--outname") {
    outname <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--chain-dir") {
    chaindir_rel <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2plink") {
    path2plink <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--path2liftover") {
    path2liftover <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--force") {
    force_run <- TRUE; i <- i + 1
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

if (is.null(outname)) outname <- paste0(name, "_hg38")

# --- Section 3: Resolve paths and configuration -------------------------------

indir       <- file.path(project_dir, indir_rel)
outdir      <- file.path(project_dir, outdir_rel)
chaindir    <- file.path(project_dir, chaindir_rel)
reports_dir <- file.path(project_dir, "reports")

chain_url   <- "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
chain_file  <- file.path(chaindir, "hg19ToHg38.over.chain.gz")

# Sanity-check SNPs (rsID -> expected hg38 chr, pos). NeuroChip multi-probe
# variants such as seq-rs429358-T2 are matched by substring of the rsID.
sanity_snps <- list(
  list(rsid = "rs429358",  chr = "19", pos = 44908684L),
  list(rsid = "rs7412",    chr = "19", pos = 44908822L),
  list(rsid = "rs1801133", chr = "1",  pos = 11796321L)
)

# --- Section 4: Helper functions ---------------------------------------------

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

detect_liftover <- function() {
  candidates <- c(
    Sys.which("liftOver"),
    "/usr/local/bin/liftOver",
    "/opt/homebrew/bin/liftOver"
  )
  for (p in candidates) {
    if (nchar(p) > 0 && file.exists(p)) return(p)
  }
  stop(
    "UCSC liftOver binary not found.\n",
    "Download the platform-appropriate binary from:\n",
    "  https://hgdownload.soe.ucsc.edu/admin/exe/\n",
    "Make it executable (chmod +x liftOver) and put it on PATH or\n",
    "specify the path with --path2liftover."
  )
}

ensure_chain_file <- function(chain_path, url) {
  if (file.exists(chain_path) && file.info(chain_path)$size > 0) {
    message("Chain file present: ", chain_path)
    return(invisible(NULL))
  }
  message("Downloading chain file from ", url)
  if (!dir.exists(dirname(chain_path))) {
    dir.create(dirname(chain_path), recursive = TRUE)
  }
  download.file(url, destfile = chain_path, mode = "wb", quiet = FALSE)
  if (!file.exists(chain_path) || file.info(chain_path)$size == 0) {
    stop("Chain file download failed (empty or missing): ", chain_path)
  }
  message("Chain file downloaded (", file.info(chain_path)$size, " bytes)")
}

# Map PLINK numeric chromosome code -> UCSC name (NA if unsupported).
# 23/25 -> chrX (X and PAR/XY both live on chrX in UCSC builds).
plink_to_ucsc <- function(code) {
  code <- as.character(code)
  out <- character(length(code))
  out[code %in% as.character(1:22)] <- paste0("chr", code[code %in% as.character(1:22)])
  out[code == "23"] <- "chrX"
  out[code == "24"] <- "chrY"
  out[code == "25"] <- "chrX"
  out[code == "26"] <- "chrM"
  out[!(code %in% as.character(c(1:26)))] <- NA_character_
  out
}

# Map UCSC chromosome name -> PLINK numeric code (NA if not 1-22, X, Y, M).
ucsc_to_plink <- function(uc) {
  base <- sub("^chr", "", uc)
  out <- character(length(uc))
  std_auto <- base %in% as.character(1:22)
  out[std_auto] <- base[std_auto]
  out[base == "X"]  <- "23"
  out[base == "Y"]  <- "24"
  out[base == "M" | base == "MT"] <- "26"
  out[!(std_auto | base %in% c("X", "Y", "M", "MT"))] <- NA_character_
  out
}

# Canonical short form ("1".."22","X","Y","M") for chromosome equality checks.
# Treats PLINK 25 (PAR/XY) and 23 (X) as both "X".
canonical_chr <- function(code, source = c("plink", "ucsc")) {
  source <- match.arg(source)
  code <- as.character(code)
  out <- character(length(code))
  if (source == "plink") {
    out[code %in% as.character(1:22)] <- code[code %in% as.character(1:22)]
    out[code %in% c("23", "25")] <- "X"
    out[code == "24"] <- "Y"
    out[code == "26"] <- "M"
    out[!(code %in% as.character(1:26))] <- NA_character_
  } else {
    base <- sub("^chr", "", code)
    out[base %in% as.character(1:22)] <- base[base %in% as.character(1:22)]
    out[base == "X"]  <- "X"
    out[base == "Y"]  <- "Y"
    out[base %in% c("M", "MT")] <- "M"
    out[!(base %in% c(as.character(1:22), "X", "Y", "M", "MT"))] <- NA_character_
  }
  out
}

write_input_bed <- function(bim, bed_path) {
  ucsc_chr <- plink_to_ucsc(bim$V1)
  keep <- !is.na(ucsc_chr)
  dropped <- sum(!keep)
  if (dropped > 0) {
    message("Dropping ", dropped, " variants on non-standard chromosomes (codes ",
            paste(unique(bim$V1[!keep]), collapse = ","), ")")
  }
  bed <- data.table(
    chrom = ucsc_chr[keep],
    start = bim$V4[keep] - 1L,
    end   = bim$V4[keep],
    name  = bim$V2[keep]
  )
  fwrite(bed, bed_path, sep = "\t", col.names = FALSE, quote = FALSE)
  invisible(list(written = nrow(bed), dropped = dropped))
}

run_liftover <- function(liftover_bin, in_bed, chain, out_bed, unmapped_bed) {
  message("Running liftOver...")
  status <- system2(
    liftover_bin,
    args = c(shQuote(in_bed), shQuote(chain), shQuote(out_bed), shQuote(unmapped_bed)),
    stdout = TRUE, stderr = TRUE
  )
  for (line in status) message("  [liftOver] ", line)
  if (!file.exists(out_bed)) {
    stop("liftOver did not produce output BED: ", out_bed)
  }
}

# --- Section 5: Main ---------------------------------------------------------

main <- function() {

  message("=============================================================")
  message("B0: Liftover hg19 -> hg38")
  message("=============================================================")
  message("Project:    ", project_dir)
  message("Input dir:  ", indir)
  message("Input name: ", name)
  message("Output dir: ", outdir)
  message("Output name:", outname)
  message("Chain dir:  ", chaindir)
  message("")

  check_required_files(indir, name)

  if (is.null(path2plink))    path2plink    <<- detect_plink()
  if (is.null(path2liftover)) path2liftover <<- detect_liftover()
  message("PLINK v1.9: ", path2plink)
  message("liftOver:   ", path2liftover)

  for (d in c(outdir, chaindir, reports_dir)) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }

  out_prefix <- file.path(outdir, outname)
  out_files  <- paste0(out_prefix, c(".bed", ".bim", ".fam"))

  if (all(file.exists(out_files)) && !force_run) {
    message("")
    message("Outputs exist; skipping (use --force to re-run):")
    for (f in out_files) message("  ", f)
    return(invisible(NULL))
  }

  ensure_chain_file(chain_file, chain_url)

  # 5a. Read original .bim and write UCSC BED
  message("")
  message("=============================================================")
  message("Step 1: Convert .bim to UCSC BED")
  message("=============================================================")
  bim_path <- file.path(indir, paste0(name, ".bim"))
  fam_path <- file.path(indir, paste0(name, ".fam"))
  bim <- fread(bim_path, header = FALSE,
               col.names = c("V1", "V2", "V3", "V4", "V5", "V6"))
  initial_n_variants <- nrow(bim)
  fam <- fread(fam_path, header = FALSE)
  initial_n_samples  <- nrow(fam)
  rm(fam)

  message("Input variants: ", initial_n_variants)
  message("Input samples:  ", initial_n_samples)

  in_bed       <- file.path(outdir, paste0(outname, ".input.bed"))
  out_bed      <- file.path(outdir, paste0(outname, ".lifted.bed"))
  unmapped_bed <- file.path(outdir, paste0(outname, ".unmapped.bed"))
  write_res    <- write_input_bed(bim, in_bed)
  message("Wrote BED: ", in_bed, " (", write_res$written, " variants)")

  # 5b. Run liftOver
  message("")
  message("=============================================================")
  message("Step 2: liftOver hg19 -> hg38")
  message("=============================================================")
  run_liftover(path2liftover, in_bed, chain_file, out_bed, unmapped_bed)

  lifted   <- fread(out_bed, header = FALSE,
                    col.names = c("lchr", "lstart", "lend", "rsid"))
  unmapped_rsids <- character(0)
  if (file.exists(unmapped_bed) && file.info(unmapped_bed)$size > 0) {
    um <- tryCatch(
      fread(unmapped_bed, header = FALSE, fill = TRUE),
      error = function(e) NULL
    )
    if (!is.null(um) && ncol(um) >= 4) {
      unmapped_rsids <- um[[4]][!startsWith(as.character(um[[1]]), "#")]
      unmapped_rsids <- unmapped_rsids[nchar(unmapped_rsids) > 0]
    }
  }
  unmapped_count <- length(unmapped_rsids)
  silent_dropped <- write_res$written - nrow(lifted) - unmapped_count
  if (silent_dropped < 0) silent_dropped <- 0L
  message("Lifted variants:   ", nrow(lifted))
  message("Unmapped variants: ", unmapped_count)
  if (silent_dropped > 0) {
    message("Silently dropped by liftOver (chain-boundary/ambiguous): ", silent_dropped)
  }

  # Persist a clean unmapped list (rsIDs only) alongside outputs
  unmapped_txt <- paste0(out_prefix, ".unmapped.txt")
  writeLines(unmapped_rsids, unmapped_txt)

  # 5c. Detect chromosome-jumping variants
  message("")
  message("=============================================================")
  message("Step 3: Filter chromosome-jumping variants")
  message("=============================================================")
  setkey(bim, V2)
  setkey(lifted, rsid)
  joined <- bim[lifted, on = c(V2 = "rsid"), nomatch = NULL]
  joined[, orig_canon := canonical_chr(V1, "plink")]
  joined[, lift_canon := canonical_chr(lchr, "ucsc")]
  chr_jumpers <- joined[is.na(lift_canon) | orig_canon != lift_canon]
  stable      <- joined[!is.na(lift_canon) & orig_canon == lift_canon]

  chr_changed_txt <- paste0(out_prefix, ".chr_changed.txt")
  writeLines(chr_jumpers$V2, chr_changed_txt)
  message("Chromosome-stable variants: ", nrow(stable))
  message("Chromosome-jumpers dropped: ", nrow(chr_jumpers))

  # 5d. Build PLINK update files and extract list
  message("")
  message("=============================================================")
  message("Step 4: Build PLINK update files")
  message("=============================================================")
  # Preserve original PLINK chr code where canonical match holds (e.g. PAR=25
  # stays 25 if liftover keeps it on chrX).
  stable[, new_chr := V1]
  # Translate any chrM/Y/X coming from UCSC -> PLINK numeric where the original
  # encoding was non-numeric should not happen, but guard anyway.
  stable[, new_pos := lend]  # BED end is 1-based pos

  update_chr_path  <- file.path(outdir, "update_chr.txt")
  update_map_path  <- file.path(outdir, "update_map.txt")
  keep_path        <- file.path(outdir, "keep_variants.txt")

  fwrite(stable[, .(V2, new_chr)], update_chr_path,
         sep = "\t", col.names = FALSE, quote = FALSE)
  fwrite(stable[, .(V2, new_pos)], update_map_path,
         sep = "\t", col.names = FALSE, quote = FALSE)
  writeLines(stable$V2, keep_path)

  # 5e. Run PLINK to apply the lift
  message("")
  message("=============================================================")
  message("Step 5: Apply liftover with PLINK --make-bed")
  message("=============================================================")
  plink_args <- c(
    "--bfile",      shQuote(file.path(indir, name)),
    "--extract",    shQuote(keep_path),
    "--update-chr", shQuote(update_chr_path),
    "--update-map", shQuote(update_map_path),
    "--make-bed",
    "--allow-extra-chr",
    "--out",        shQuote(out_prefix)
  )
  plink_out <- system2(path2plink, args = plink_args,
                       stdout = TRUE, stderr = TRUE)
  for (line in plink_out) message("  [plink] ", line)
  status <- attr(plink_out, "status")
  if (!is.null(status) && status != 0) {
    stop("PLINK --make-bed failed (status ", status, ")")
  }
  if (!all(file.exists(out_files))) {
    stop("PLINK did not produce all output files: ",
         paste(out_files[!file.exists(out_files)], collapse = ", "))
  }

  # 5f. Verification
  message("")
  message("=============================================================")
  message("Step 6: Verification")
  message("=============================================================")
  out_bim <- fread(paste0(out_prefix, ".bim"), header = FALSE,
                   col.names = c("V1", "V2", "V3", "V4", "V5", "V6"))
  out_fam <- fread(paste0(out_prefix, ".fam"), header = FALSE)
  final_n_variants <- nrow(out_bim)
  final_n_samples  <- nrow(out_fam)

  if (final_n_samples != initial_n_samples) {
    stop("Sample count changed: ", initial_n_samples, " -> ", final_n_samples)
  }
  message("Sample count unchanged: ", final_n_samples)

  loss_frac <- 1 - final_n_variants / initial_n_variants
  message(sprintf("Variant retention: %d / %d (%.2f%% lost)",
                  final_n_variants, initial_n_variants, 100 * loss_frac))
  if (loss_frac > 0.10) {
    stop(sprintf("Variant loss exceeds 10%% (%.2f%%); aborting.", 100 * loss_frac))
  } else if (loss_frac > 0.05) {
    message(sprintf("WARNING: variant loss exceeds 5%% (%.2f%%).", 100 * loss_frac))
  }

  # Sanity-check SNPs: any variant whose rsID contains the canonical rsID and
  # sits at the expected hg38 (chr, pos) should be present in the new bim.
  sanity_results <- list()
  for (s in sanity_snps) {
    in_input  <- bim[grepl(s$rsid, V2, fixed = TRUE)]
    in_output <- out_bim[grepl(s$rsid, V2, fixed = TRUE)]
    if (nrow(in_input) == 0) {
      sanity_results[[s$rsid]] <- list(
        status = "ABSENT_IN_INPUT",
        msg = paste0(s$rsid, ": not in input .bim; nothing to verify")
      )
      message("  ", sanity_results[[s$rsid]]$msg)
      next
    }
    matches <- in_output[as.character(V1) == s$chr & V4 == s$pos]
    if (nrow(matches) == 0) {
      sanity_results[[s$rsid]] <- list(
        status = "FAIL",
        msg = paste0(s$rsid, ": expected ", s$chr, ":", s$pos,
                     " (hg38) not found in output .bim")
      )
      stop(sanity_results[[s$rsid]]$msg)
    }
    sanity_results[[s$rsid]] <- list(
      status = "OK",
      msg = paste0(s$rsid, ": ", nrow(matches),
                   " variant(s) at ", s$chr, ":", s$pos, " (hg38) OK")
    )
    message("  ", sanity_results[[s$rsid]]$msg)
  }

  # 5g. Tool versions for the report
  liftover_banner <- tryCatch({
    out <- suppressWarnings(system2(path2liftover, args = character(0),
                                     stdout = TRUE, stderr = TRUE))
    out[1]
  }, error = function(e) "unknown")
  plink_version <- tryCatch({
    out <- suppressWarnings(system2(path2plink, args = "--version",
                                     stdout = TRUE, stderr = TRUE))
    paste(out, collapse = " ")
  }, error = function(e) "unknown")

  # 5h. Write report
  message("")
  message("=============================================================")
  message("Step 7: Write report")
  message("=============================================================")
  report_lines <- c(
    paste(rep("=", 72), collapse = ""),
    "B0: LIFTOVER hg19 -> hg38 REPORT",
    paste(rep("=", 72), collapse = ""),
    "",
    paste("Date:             ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    paste("Input directory:  ", indir),
    paste("Input prefix:     ", name),
    paste("Output directory: ", outdir),
    paste("Output prefix:    ", outname),
    paste("Chain file:       ", chain_file),
    paste("liftOver:         ", path2liftover, " (", liftover_banner, ")"),
    paste("PLINK v1.9:       ", path2plink),
    paste("PLINK version:    ", plink_version),
    "",
    paste(rep("-", 72), collapse = ""),
    "VARIANT COUNTS:",
    paste("  Input variants:               ", initial_n_variants),
    paste("  Dropped (non-std chromosome): ", write_res$dropped),
    paste("  Unmapped by liftOver:         ", unmapped_count),
    paste("  Silently dropped by liftOver: ", silent_dropped),
    paste("  Chromosome-jumpers dropped:   ", nrow(chr_jumpers)),
    paste("  Final variants:               ", final_n_variants),
    sprintf("  Variant loss:                  %.2f%%", 100 * loss_frac),
    "",
    "SAMPLE COUNTS:",
    paste("  Input samples:  ", initial_n_samples),
    paste("  Output samples: ", final_n_samples),
    "",
    paste(rep("-", 72), collapse = ""),
    "SANITY CHECKS (hg38 expected positions):"
  )
  for (s in sanity_snps) {
    r <- sanity_results[[s$rsid]]
    report_lines <- c(report_lines, paste0("  [", r$status, "] ", r$msg))
  }
  report_lines <- c(report_lines,
    "",
    paste(rep("-", 72), collapse = ""),
    "OUTPUT FILES:",
    paste("  PLINK triplet:   ", out_prefix, ".{bed,bim,fam}"),
    paste("  Unmapped IDs:    ", unmapped_txt),
    paste("  Chr-changed IDs: ", chr_changed_txt),
    paste("  Update chr:      ", update_chr_path),
    paste("  Update map:      ", update_map_path),
    paste("  Keep list:       ", keep_path),
    "",
    paste(rep("-", 72), collapse = ""),
    "NOTES:",
    "  - NeuroChip carries multi-probe variants (e.g. seq-rs429358-B1/B2/B3/T2/T3).",
    "    These are preserved as separate entries through liftover; downstream",
    "    --rm-dup in B1 collapses them.",
    "  - Variant IDs (rsIDs) are unchanged here. The chr:pos[hg38] suffix is",
    "    added later by B1's --set-all-var-ids step.",
    "",
    paste(rep("=", 72), collapse = "")
  )

  report_path <- file.path(reports_dir, "B0_liftover_report.txt")
  writeLines(report_lines, report_path)
  message("Saved report: ", report_path)

  message("")
  message("=============================================================")
  message("B0 LIFTOVER COMPLETE")
  message("=============================================================")
  message("Output: ", out_prefix, ".{bed,bim,fam}")
  message(final_n_variants, " variants, ", final_n_samples, " samples")
}

# --- Section 6: Run main with error handling ---------------------------------

tryCatch({
  main()
}, error = function(e) {
  message("")
  message("FATAL ERROR: ", e$message)
  partial_path <- file.path(reports_dir, "B0_liftover_report.txt")
  if (dir.exists(reports_dir)) {
    writeLines(c(
      paste(rep("=", 72), collapse = ""),
      "B0: LIFTOVER REPORT (PARTIAL — ERRORED)",
      paste(rep("=", 72), collapse = ""),
      paste("Date:  ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
      paste("Error: ", e$message)
    ), partial_path)
    message("Partial report written: ", partial_path)
  }
  quit(status = 1)
})
