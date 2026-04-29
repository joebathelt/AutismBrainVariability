#!/usr/bin/env python3
"""
B1b_ancestry_PCA_mahalanobis.py
Ancestry inference via in-sample PCA on 1000G phase 3 hg38 + HCP merge,
with Mahalanobis distance to a CEU+GBR+IBS+TSI centroid.

Replaces B1's plinkQC RF ancestry step. B1 must be run with --skip-ancestry;
this script consumes the resulting .clean.bed/bim/fam, downloads the full
1000G phase 3 hg38 reference if not cached, merges on shared chr:pos
variants, LD-prunes, runs --pca 20 on the merged set, computes Mahalanobis
distance from each sample to the EUR-reference centroid (using the top K
PCs), and applies a chi-squared (default) or SD-based cutoff. Produces a
PC1-vs-PC2 scatter plot with subpopulation centroids, a Mahalanobis
distribution plot, a per-sample CSV, keep/fail ID lists, and the canonical
clean PLINK triplet for B2 to consume.

Usage:
    python B1b_ancestry_PCA_mahalanobis.py --project /path/to/BrainCompensation
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import chi2

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8

# 1000G phase 3 hg38 download URLs (plink2 reference resources).
# Source: https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3
# These Dropbox URLs change occasionally; if download fails, refresh from
# the page above and update here, or pre-stage the files manually and pass
# --skip-download.
KG_URLS = {
    "all_hg38.pgen.zst": "https://www.dropbox.com/s/e5n8yr4n7y91fyp/all_hg38.pgen.zst?dl=1",
    "all_hg38.pvar.zst": "https://www.dropbox.com/s/vx09262b4k1kszy/all_hg38.pvar.zst?dl=1",
    "all_hg38.psam":     "https://www.dropbox.com/s/qhtb5t3py3kyjeq/hg38_orig.psam?dl=1",
}

SUPERPOP_COLOURS = {
    "AFR": "#E69F00",
    "AMR": "#56B4E9",
    "EAS": "#009E73",
    "EUR": "#CC79A7",
    "SAS": "#0072B2",
}


# ============================================================================
# Helpers
# ============================================================================

def log(msg):
    """Print to stdout with flush so Snakemake log redirection sees it live."""
    print(msg, flush=True)


def detect_plink(override=None):
    if override:
        return override
    candidates = [
        shutil.which("plink"),
        "/Applications/PLINK 1.9 Mac Oct 2024/plink",
        "/usr/local/bin/plink",
        "/opt/homebrew/bin/plink",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise RuntimeError("PLINK v1.9 not found. Specify --path2plink.")


def detect_plink2(override=None):
    if override:
        return override
    candidates = [
        shutil.which("plink2"),
        "/Applications/plink2_mac_arm64_20260110/plink2",
        "/Applications/PLINK 2.0 Mac Arm64 Jan 29 2025/plink2",
        "/usr/local/bin/plink2",
        "/opt/homebrew/bin/plink2",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise RuntimeError("PLINK 2.0 not found. Specify --path2plink2.")


def run(cmd, label, allow_fail=False):
    """Run a subprocess command, streaming output to stdout. Returns
    (returncode, stdout_lines)."""
    log(f"  $ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.rstrip().splitlines():
        log(f"    [{label}] {line}")
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}); see log above.")
    return proc.returncode, out


def count_lines(path):
    return sum(1 for _ in open(path))


def download(url, dest):
    """Download with simple progress reporting. Removes partial file on failure."""
    log(f"  Downloading {url}")
    log(f"    -> {dest}")
    tmp = Path(str(dest) + ".part")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            block = 1024 * 1024
            while True:
                buf = r.read(block)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"    {done/1e9:.2f} / {total/1e9:.2f} GB ({pct:.1f}%)",
                          end="\r", flush=True)
        print()
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ============================================================================
# Pipeline steps
# ============================================================================

def ensure_reference(refdir, plink2, skip_download):
    """Download (if needed) and decompress the 1000G phase 3 hg38 reference.

    Produces decompressed all_hg38.pgen, all_hg38.pvar, all_hg38.psam in refdir.
    Returns the path prefix (refdir / 'all_hg38').
    """
    refdir = Path(refdir)
    refdir.mkdir(parents=True, exist_ok=True)
    prefix = refdir / "all_hg38"

    pgen = refdir / "all_hg38.pgen"
    pvar = refdir / "all_hg38.pvar"
    psam = refdir / "all_hg38.psam"

    if pgen.exists() and pvar.exists() and psam.exists():
        log(f"Reference already decompressed: {prefix}.{{pgen,pvar,psam}}")
        return prefix

    if skip_download:
        missing = [str(p) for p in (pgen, pvar, psam) if not p.exists()]
        raise RuntimeError(
            "--skip-download set but reference files missing:\n  "
            + "\n  ".join(missing)
            + f"\nPre-stage them in {refdir} and retry."
        )

    # Download .zst archives + .psam if not present
    for fname, url in KG_URLS.items():
        dest = refdir / fname
        if dest.exists() and dest.stat().st_size > 0:
            log(f"Cached: {dest}")
            continue
        download(url, dest)

    # Decompress .zst via plink2
    pgen_zst = refdir / "all_hg38.pgen.zst"
    pvar_zst = refdir / "all_hg38.pvar.zst"
    if not pgen.exists():
        log("Decompressing all_hg38.pgen.zst")
        run([plink2, "--zst-decompress", str(pgen_zst), str(pgen)], "plink2-zst")
    if not pvar.exists():
        log("Decompressing all_hg38.pvar.zst")
        run([plink2, "--zst-decompress", str(pvar_zst), str(pvar)], "plink2-zst")
    if not psam.exists():
        raise RuntimeError(f"psam download did not produce {psam}")

    return prefix


def convert_kg_to_bed(plink2, kg_prefix, workdir):
    """Convert 1000G pgen/pvar/psam to plink1 bed (biallelic SNPs only,
    de-duplicated). Returns the output prefix."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_prefix = workdir / "1kg_phase3_hg38"

    if all((Path(str(out_prefix) + ext)).exists() for ext in (".bed", ".bim", ".fam")):
        log(f"1KG bed already built: {out_prefix}")
        return out_prefix

    log("Converting 1000G pgen -> bed (biallelic SNPs, deduped)")
    run([
        plink2,
        "--pfile", str(kg_prefix),
        "--max-alleles", "2",
        "--snps-only", "just-acgt",
        "--rm-dup", "exclude-all",
        "--make-bed",
        "--out", str(out_prefix),
    ], "plink2")
    return out_prefix


def harmonise_ids(plink2, in_prefix, out_prefix, label):
    """Recode variant IDs to chr:pos[hg38], drop multi-allelics and indels,
    de-duplicate. Both HCP and 1KG go through this so chr:pos IDs match."""
    out_prefix = Path(out_prefix)
    if all(Path(f"{out_prefix}{ext}").exists() for ext in (".bed", ".bim", ".fam")):
        log(f"{label} chrpos bed already built: {out_prefix}")
        return out_prefix

    log(f"Harmonising {label} variant IDs to chr:pos[hg38]")
    run([
        plink2,
        "--bfile", str(in_prefix),
        "--chr", "1-22",
        "--rm-dup", "force-first",
        "--max-alleles", "2",
        "--snps-only", "just-acgt",
        "--set-all-var-ids", "@:#[hg38]",
        "--new-id-max-allele-len", "50", "missing",
        "--make-bed",
        "--out", str(out_prefix),
    ], f"plink2-{label}")
    return out_prefix


def shared_variants(hcp_bim, kg_bim, shared_path):
    """Write the chr:pos intersection of two bim files to shared_path."""
    log("Computing chr:pos intersection")
    hcp_ids = set()
    with open(hcp_bim) as f:
        for line in f:
            hcp_ids.add(line.split()[1])
    kg_ids = set()
    with open(kg_bim) as f:
        for line in f:
            kg_ids.add(line.split()[1])
    shared = sorted(hcp_ids & kg_ids)
    with open(shared_path, "w") as f:
        for vid in shared:
            f.write(vid + "\n")
    log(f"  HCP variants: {len(hcp_ids):,}")
    log(f"  1KG variants: {len(kg_ids):,}")
    log(f"  Intersection: {len(shared):,}")
    return len(hcp_ids), len(kg_ids), len(shared)


def extract(plink, in_prefix, snps_file, out_prefix, label):
    """plink --extract wrapper."""
    out_prefix = Path(out_prefix)
    if all(Path(f"{out_prefix}{ext}").exists() for ext in (".bed", ".bim", ".fam")):
        log(f"{label} extracted bed already built: {out_prefix}")
        return out_prefix
    run([
        plink,
        "--bfile", str(in_prefix),
        "--extract", str(snps_file),
        "--make-bed",
        "--out", str(out_prefix),
    ], f"plink-{label}")
    return out_prefix


def merge_with_flip_retry(plink, hcp_prefix, kg_prefix, workdir):
    """Merge HCP and 1KG with one strand-flip retry; if conflicts remain,
    drop them on both sides and merge a third time. Returns merged prefix and
    a dict of merge stats."""
    workdir = Path(workdir)
    merged = workdir / "merged"
    stats = {"first_missnp": 0, "after_flip_missnp": 0, "final_excluded": 0}

    def attempt(hcp_pre, suffix=""):
        out = Path(str(merged) + suffix)
        rc, _ = run([
            plink,
            "--bfile", str(hcp_pre),
            "--bmerge", str(kg_prefix),
            "--make-bed",
            "--out", str(out),
        ], f"plink-bmerge{suffix}", allow_fail=True)
        return rc, out

    # First attempt
    rc, out = attempt(hcp_prefix)
    if rc == 0:
        log("Merge succeeded on first attempt.")
        return out, stats

    # Inspect missnp
    missnp = Path(str(merged) + "-merge.missnp")
    if not missnp.exists():
        raise RuntimeError(
            f"plink --bmerge failed (exit {rc}) but no .missnp produced; "
            "see log above."
        )
    stats["first_missnp"] = count_lines(missnp)
    log(f"First merge produced {stats['first_missnp']:,} conflicting SNPs; flipping HCP and retrying.")

    # Flip HCP and retry
    flipped = workdir / "hcp_flipped"
    run([
        plink,
        "--bfile", str(hcp_prefix),
        "--flip", str(missnp),
        "--make-bed",
        "--out", str(flipped),
    ], "plink-flip")

    rc, out = attempt(flipped, suffix="_v2")
    if rc == 0:
        # Promote v2 -> merged
        for ext in (".bed", ".bim", ".fam", ".log"):
            src = Path(f"{merged}_v2{ext}")
            dst = Path(f"{merged}{ext}")
            if src.exists():
                shutil.copy(src, dst)
        log("Merge succeeded after one flip.")
        return merged, stats

    # Drop residual missnp on both sides and merge a third time
    missnp2 = Path(f"{merged}_v2-merge.missnp")
    if not missnp2.exists():
        raise RuntimeError(
            "Second merge attempt failed without producing a .missnp; "
            "see log above."
        )
    stats["after_flip_missnp"] = count_lines(missnp2)
    stats["final_excluded"] = stats["after_flip_missnp"]
    log(f"Second merge produced {stats['after_flip_missnp']:,} residual conflicts; "
        "excluding on both sides for final merge.")

    hcp_final = workdir / "hcp_final"
    kg_final = workdir / "kg_final"
    run([
        plink,
        "--bfile", str(flipped),
        "--exclude", str(missnp2),
        "--make-bed",
        "--out", str(hcp_final),
    ], "plink-hcp-exclude")
    run([
        plink,
        "--bfile", str(kg_prefix),
        "--exclude", str(missnp2),
        "--make-bed",
        "--out", str(kg_final),
    ], "plink-kg-exclude")
    rc, _ = run([
        plink,
        "--bfile", str(hcp_final),
        "--bmerge", str(kg_final),
        "--make-bed",
        "--out", str(merged),
    ], "plink-bmerge-final")
    if rc != 0:
        raise RuntimeError("Third merge attempt failed; see log above.")
    log("Merge succeeded after exclusion.")
    return merged, stats


def ld_prune(plink, merged_prefix, workdir):
    """LD prune the merged dataset and rebuild the bed restricted to prune.in."""
    workdir = Path(workdir)
    pruned_prefix = workdir / "merged_pruned"
    log("LD pruning (--indep-pairwise 50 5 0.2)")
    run([
        plink,
        "--bfile", str(merged_prefix),
        "--make-founders",
        "--indep-pairwise", "50", "5", "0.2",
        "--out", str(pruned_prefix),
    ], "plink-prune")
    n_in = count_lines(f"{pruned_prefix}.prune.in")

    ldpruned = workdir / "merged_ldpruned"
    run([
        plink,
        "--bfile", str(merged_prefix),
        "--make-founders",
        "--extract", f"{pruned_prefix}.prune.in",
        "--make-bed",
        "--out", str(ldpruned),
    ], "plink-extract-pruned")
    return ldpruned, n_in


def run_pca(plink2, ldpruned_prefix, n_pcs, kg_iids, workdir):
    """Reference-only PCA on 1KG samples, then project the full merged
    dataset onto those PCs.

    In-sample PCA on HCP+1KG produces axes contaminated by HCP-vs-1KG
    batch effects: the EUR-ref cluster ends up artificially tight on every
    PC, the inverse covariance blows up, and Mahalanobis D explodes for
    HCP samples. Projecting onto 1KG-derived PCs forces the axes to encode
    1KG genetic structure (real ancestry) and HCP gets *placed* on those
    axes rather than co-defining them.

    Returns (DataFrame of PCs for all merged samples, sscore_path).
    """
    workdir = Path(workdir)

    # 1. Reference PCA on 1KG only, with allele weights + counts for projection.
    #    1KG samples carry FID=0 in merged.fam; --keep matches by FID+IID.
    kg_keep = workdir / "kg_iids.keep"
    with open(kg_keep, "w") as f:
        for iid in kg_iids:
            f.write(f"0\t{iid}\n")

    kg_pca_prefix = workdir / "kg_pca"
    log(f"Reference PCA on 1KG (--pca {n_pcs} allele-wts)")
    # 1KG contains real trios; --ac-founders restricts allele counting to
    # founders, which is what we want for PCA reference frequencies. --pca
    # itself already uses founders-only by default.
    run([
        plink2,
        "--bfile", str(ldpruned_prefix),
        "--keep", str(kg_keep),
        "--freq", "counts", "--ac-founders",
        "--pca", str(n_pcs), "allele-wts",
        "--out", str(kg_pca_prefix),
    ], "plink2-pca-ref")

    # 2. Project all merged samples onto the reference PCs via --score.
    #    eigenvec.allele cols: #CHROM ID REF ALT PROVISIONAL_REF? A1 PC1...PCn
    #    (the PROVISIONAL_REF? column appears because the merged bed came
    #    from plink 1.9, so REF/ALT aren't authoritatively known.)
    #    --read-freq uses 1KG allele counts so variance-standardize scales
    #    target scores in the same units as the reference eigenvectors.
    proj_prefix = workdir / "merged_projected"
    log("Projecting all samples onto reference PCs (--score)")
    score_col_end = 6 + n_pcs
    run([
        plink2,
        "--bfile", str(ldpruned_prefix),
        "--read-freq", f"{kg_pca_prefix}.acount",
        "--score", f"{kg_pca_prefix}.eigenvec.allele",
        "2", "6", "header-read", "no-mean-imputation", "variance-standardize",
        "--score-col-nums", f"7-{score_col_end}",
        "--out", str(proj_prefix),
    ], "plink2-project")

    sscore = Path(f"{proj_prefix}.sscore")
    df = pd.read_csv(sscore, sep="\t", dtype={"#FID": str, "IID": str})
    df = df.rename(columns={"#FID": "FID"})
    rename = {f"PC{i}_AVG": f"PC{i}" for i in range(1, n_pcs + 1)}
    missing = [src for src in rename if src not in df.columns]
    if missing:
        raise RuntimeError(
            f"Projected sscore missing expected PC columns {missing}; "
            f"found {list(df.columns)}"
        )
    df = df.rename(columns=rename)
    cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, n_pcs + 1)]
    return df[cols].copy(), sscore


def load_psam(psam_path):
    """Read the 1000G psam, returning a DataFrame with IID, Population, SuperPop."""
    # The psam uses '#IID' as the first header; header lines start with '##'
    # are metadata. Read all non-## lines, keep the # header.
    with open(psam_path) as f:
        header = None
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split("\t")
                continue
            if not line.strip():
                continue
            rows.append(line.split("\t"))
    if header is None:
        raise RuntimeError(f"psam has no header line: {psam_path}")
    df = pd.DataFrame(rows, columns=header)
    # Normalise column names
    rename = {}
    for c in df.columns:
        if c.lower() == "iid":
            rename[c] = "IID"
        elif c.lower() in ("population", "pop"):
            rename[c] = "Population"
        elif c.lower() in ("superpop", "super_pop"):
            rename[c] = "SuperPop"
    df = df.rename(columns=rename)
    keep = ["IID", "Population", "SuperPop"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"psam missing required columns {missing}; "
            f"found {list(df.columns)}"
        )
    return df[keep].copy()


def label_samples(pcs, psam):
    """Left-join PC table with psam labels. Rows missing from psam are HCP."""
    merged = pcs.merge(psam, on="IID", how="left")
    merged["source"] = np.where(merged["Population"].isna(), "HCP", "1KG")
    return merged


def mahalanobis_distances(labelled, ref_subpops, k_pcs):
    """Compute Mahalanobis D and D^2 for every sample using the top K PCs.

    Centroid + covariance estimated from `ref_subpops` only — never from HCP.
    Returns the labelled DataFrame with new D, D2 columns plus the centroid mu.
    """
    pc_cols = [f"PC{i}" for i in range(1, k_pcs + 1)]
    ref_mask = labelled["Population"].isin(ref_subpops)
    if ref_mask.sum() < 50:
        raise RuntimeError(
            f"Too few reference samples ({ref_mask.sum()}) in subpops "
            f"{ref_subpops}; check --ref-subpops and that the psam loaded."
        )
    ref_pcs = labelled.loc[ref_mask, pc_cols].to_numpy(dtype=float)
    mu = ref_pcs.mean(axis=0)
    cov = np.cov(ref_pcs.T)
    inv_cov = np.linalg.pinv(cov)

    X = labelled[pc_cols].to_numpy(dtype=float) - mu
    D2 = np.einsum("ij,jk,ik->i", X, inv_cov, X)
    D = np.sqrt(np.clip(D2, 0, None))
    out = labelled.copy()
    out["D2"] = D2
    out["D"] = D
    return out, mu, ref_mask


def apply_threshold(labelled, ref_mask, k, method, chi2_quantile, sd_cutoff):
    """Compute the cutoff and label each row pass/fail. Returns (labelled, info)."""
    if method == "chi2":
        d2_thresh = float(chi2.ppf(chi2_quantile, df=k))
        d_thresh = float(np.sqrt(d2_thresh))
        info = {
            "method": "chi2",
            "quantile": chi2_quantile,
            "df": k,
            "D_thresh": d_thresh,
            "D2_thresh": d2_thresh,
        }
    elif method == "sd":
        ref_d = labelled.loc[ref_mask, "D"].to_numpy(dtype=float)
        mean_d = float(ref_d.mean())
        sd_d = float(ref_d.std(ddof=1))
        d_thresh = mean_d + sd_cutoff * sd_d
        d2_thresh = d_thresh ** 2
        info = {
            "method": "sd",
            "sd_cutoff": sd_cutoff,
            "ref_mean_D": mean_d,
            "ref_sd_D": sd_d,
            "D_thresh": d_thresh,
            "D2_thresh": d2_thresh,
        }
    else:
        raise ValueError(f"Unknown cutoff method: {method}")
    labelled = labelled.copy()
    labelled["pass"] = labelled["D2"] <= d2_thresh
    return labelled, info


def nearest_subpop(labelled, k_pcs):
    """For each HCP sample, find the closest 1000G subpopulation centroid
    by Euclidean distance on the K PCs. Returns a Series of subpop labels."""
    pc_cols = [f"PC{i}" for i in range(1, k_pcs + 1)]
    ref = labelled[labelled["source"] == "1KG"]
    centroids = ref.groupby("Population")[pc_cols].mean()
    ref_pops = centroids.index.to_numpy()
    ref_arr = centroids.to_numpy(dtype=float)

    hcp_mask = labelled["source"] == "HCP"
    hcp_pcs = labelled.loc[hcp_mask, pc_cols].to_numpy(dtype=float)
    diffs = hcp_pcs[:, None, :] - ref_arr[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    nearest_idx = d2.argmin(axis=1)
    nearest = pd.Series("", index=labelled.index, dtype=object)
    nearest.loc[hcp_mask] = ref_pops[nearest_idx]
    return nearest


def write_keep_fail(labelled, outdir):
    """Write keep/fail HCP ID lists in PLINK --keep format (FID IID, tab)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hcp = labelled[labelled["source"] == "HCP"].copy()
    keep_path = outdir / "B1b_keep_ancestry.IDs"
    fail_path = outdir / "B1b_fail_ancestry.IDs"
    hcp.loc[hcp["pass"], ["FID", "IID"]].to_csv(
        keep_path, sep="\t", header=False, index=False)
    hcp.loc[~hcp["pass"], ["FID", "IID"]].to_csv(
        fail_path, sep="\t", header=False, index=False)
    return keep_path, fail_path


def write_per_sample(labelled, k_pcs, n_pcs, outdir):
    outdir = Path(outdir)
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    out = labelled[["FID", "IID", "source", "Population", "SuperPop",
                    *pc_cols, "D2", "D", "pass"]].copy()
    out["nearest_subpop"] = nearest_subpop(labelled, k_pcs)
    path = outdir / "B1b_per_sample_distance.csv"
    out.to_csv(path, index=False)
    return path


def build_clean_triplet(plink, qc_prefix, keep_path, qcdir, qc_prefix_name):
    """Apply the keep list to B1's clean.bed/bim/fam to produce the final
    .clean.ancestry triplet that B2 consumes."""
    out_prefix = Path(qcdir) / f"{qc_prefix_name}.ancestry"
    log(f"Building canonical clean triplet: {out_prefix}.bed/bim/fam")
    run([
        plink,
        "--bfile", str(qc_prefix),
        "--keep", str(keep_path),
        "--make-bed",
        "--out", str(out_prefix),
    ], "plink-final")
    return out_prefix


# ============================================================================
# Plotting
# ============================================================================

def plot_pca_scatter(labelled, mu, ref_subpops, fig_path):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # 1KG samples by SuperPop
    kg = labelled[labelled["source"] == "1KG"]
    for sp, colour in SUPERPOP_COLOURS.items():
        sub = kg[kg["SuperPop"] == sp]
        if len(sub):
            ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.45,
                       color=colour, label=f"1KG {sp}")

    # HCP samples (pass green, fail red)
    hcp = labelled[labelled["source"] == "HCP"]
    hcp_pass = hcp[hcp["pass"]]
    hcp_fail = hcp[~hcp["pass"]]
    if len(hcp_pass):
        ax.scatter(hcp_pass["PC1"], hcp_pass["PC2"], s=12, marker="o",
                   facecolor="none", edgecolor="#2ca02c",
                   linewidth=0.7, label=f"HCP pass (n={len(hcp_pass)})")
    if len(hcp_fail):
        ax.scatter(hcp_fail["PC1"], hcp_fail["PC2"], s=20, marker="x",
                   color="#d62728", linewidth=1.0,
                   label=f"HCP fail (n={len(hcp_fail)})")

    # Subpopulation centroids
    sub_centroids = kg.groupby("Population")[["PC1", "PC2"]].mean()
    for pop, row in sub_centroids.iterrows():
        ax.scatter(row["PC1"], row["PC2"], marker="+", s=120, color="black",
                   linewidth=1.5)
        ax.annotate(pop, (row["PC1"], row["PC2"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, weight="bold")

    # SuperPop centroids
    sp_centroids = kg.groupby("SuperPop")[["PC1", "PC2"]].mean()
    for sp, row in sp_centroids.iterrows():
        ax.scatter(row["PC1"], row["PC2"], marker="P", s=200, color="black",
                   edgecolor="white", linewidth=1.0)
        ax.annotate(sp, (row["PC1"], row["PC2"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=9, weight="bold", color="black")

    # EUR-reference centroid (from supplied subpops)
    ax.scatter(mu[0], mu[1], marker="*", s=300, color="gold",
               edgecolor="black", linewidth=1.0,
               label=f"EUR ref centroid ({'+'.join(ref_subpops)})")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("1000G phase 3 hg38 + HCP merged PCA")
    ax.legend(loc="best", frameon=False, fontsize=7, markerscale=1.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distance_distribution(labelled, ref_mask, info, fig_path):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    hcp = labelled[labelled["source"] == "HCP"]["D"]
    ref = labelled.loc[ref_mask, "D"]
    other_kg = labelled[(labelled["source"] == "1KG") & ~ref_mask]["D"]

    bins = np.linspace(0, max(hcp.max(), ref.max(), other_kg.max(), info["D_thresh"]) * 1.05, 60)
    ax.hist(other_kg, bins=bins, alpha=0.5, color="#888888",
            label=f"1KG non-EUR-ref (n={len(other_kg)})")
    ax.hist(hcp, bins=bins, alpha=0.6, color="steelblue",
            label=f"HCP (n={len(hcp)})")
    ax.hist(ref, bins=bins, alpha=0.7, color="#CC79A7",
            label=f"EUR ref (n={len(ref)})")
    ax.axvline(info["D_thresh"], color="red", linestyle="--", linewidth=1.0,
               label=f"cutoff D={info['D_thresh']:.3f}")
    ax.set_xlabel("Mahalanobis distance D")
    ax.set_ylabel("Count")
    ax.set_title("Mahalanobis distance to EUR-ref centroid")
    ax.legend(loc="best", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# Report
# ============================================================================

def write_report(report_path, sections):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(sections))


def banner(title, char="="):
    return char * 72 + "\n" + title + "\n" + char * 72


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Ancestry inference via 1000G-merged PCA + Mahalanobis "
                    "(replaces B1's plinkQC RF step).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project", required=True, help="Project root directory")
    p.add_argument("--qcdir", default="data/plinkQC_output",
                   help="B1 plinkQC output directory (relative to --project)")
    p.add_argument("--qc-prefix", default="Neuro_Chip_anonymised_hg38.clean",
                   help="B1 clean PLINK file prefix")
    p.add_argument("--refdir", default="data/reference/1000Genomes",
                   help="1000G reference directory (relative to --project)")
    p.add_argument("--ref-subdir", default="phase3_hg38",
                   help="Subdir under --refdir for the full phase 3 download")
    p.add_argument("--workdir", default="data/PLINK_anonymised/B1b_ancestry",
                   help="Workdir for merge / PCA intermediates")
    p.add_argument("--outdir", default="data/PLINK_anonymised",
                   help="Output dir for keep/fail ID lists and per-sample CSV")
    p.add_argument("--n-pcs", type=int, default=20,
                   help="Number of PCs to compute via plink --pca")
    p.add_argument("--mahalanobis-pcs", type=int, default=4,
                   help="Top K PCs used for the Mahalanobis distance "
                        "(K<n-pcs; brittleness grows with K)")
    p.add_argument("--ref-subpops", default="CEU,GBR,IBS,TSI",
                   help="Comma-separated 1KG subpopulations defining the "
                        "EUR reference centroid (FIN deliberately excluded)")
    p.add_argument("--cutoff-method", choices=["chi2", "sd"], default="sd",
                   help="chi2: D^2 <= chi2.ppf(q, df=K); "
                        "sd: D <= mean(ref-D) + n*sd(ref-D)")
    p.add_argument("--chi2-quantile", type=float, default=0.9999,
                   help="Used when --cutoff-method=chi2")
    p.add_argument("--sd-cutoff", type=float, default=6.0,
                   help="Used when --cutoff-method=sd; applied to D, not D^2")
    p.add_argument("--path2plink", default=None, help="PLINK 1.9 path (auto-detect)")
    p.add_argument("--path2plink2", default=None, help="PLINK 2.0 path (auto-detect)")
    p.add_argument("--skip-download", action="store_true",
                   help="Require pre-staged 1000G reference files")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    project = Path(args.project).resolve()
    qcdir = project / args.qcdir
    refdir = project / args.refdir / args.ref_subdir
    workdir = project / args.workdir
    outdir = project / args.outdir
    figures_dir = project / "figures"
    reports_dir = project / "reports"
    for d in (workdir, outdir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    qc_prefix_path = qcdir / args.qc_prefix
    for ext in (".bed", ".bim", ".fam"):
        if not (qcdir / f"{args.qc_prefix}{ext}").exists():
            raise FileNotFoundError(
                f"Required B1 clean file missing: {qcdir / (args.qc_prefix + ext)}\n"
                "Run B1 with --skip-ancestry first."
            )

    ref_subpops = [s.strip().upper() for s in args.ref_subpops.split(",") if s.strip()]
    if args.mahalanobis_pcs > args.n_pcs:
        raise ValueError("--mahalanobis-pcs cannot exceed --n-pcs")

    plink = detect_plink(args.path2plink)
    plink2 = detect_plink2(args.path2plink2)

    log(banner("B1b: Ancestry inference (1000G-merged PCA + Mahalanobis)"))
    log(f"Project:         {project}")
    log(f"B1 clean prefix: {qc_prefix_path}")
    log(f"Reference dir:   {refdir}")
    log(f"Workdir:         {workdir}")
    log(f"PLINK 1.9:       {plink}")
    log(f"PLINK 2.0:       {plink2}")
    log(f"Reference subpops: {ref_subpops}")
    log(f"PCs computed:    {args.n_pcs};  K (Mahalanobis): {args.mahalanobis_pcs}")
    log(f"Cutoff:          {args.cutoff_method} "
        f"(chi2_q={args.chi2_quantile}, sd_cutoff={args.sd_cutoff})")
    log("")

    initial_hcp_n = count_lines(qcdir / f"{args.qc_prefix}.fam")
    initial_hcp_snps = count_lines(qcdir / f"{args.qc_prefix}.bim")

    # 1. Reference download / decompression
    log(banner("Step 1: 1000G phase 3 hg38 reference", "-"))
    kg_prefix = ensure_reference(refdir, plink2, args.skip_download)
    psam = load_psam(refdir / "all_hg38.psam")
    log(f"psam samples: {len(psam)};  populations: {psam['Population'].nunique()}")

    # 2. Convert 1KG to bed
    log(banner("Step 2: Convert 1KG pgen -> bed", "-"))
    kg_bed = convert_kg_to_bed(plink2, kg_prefix, workdir)

    # 3. Harmonise variant IDs to chr:pos[hg38]
    log(banner("Step 3: Harmonise variant IDs to chr:pos[hg38]", "-"))
    hcp_chrpos = harmonise_ids(plink2, qc_prefix_path, workdir / "hcp_chrpos", "hcp")
    kg_chrpos = harmonise_ids(plink2, kg_bed, workdir / "1kg_chrpos", "1kg")

    # 4. Restrict to chr:pos intersection
    log(banner("Step 4: Restrict to chr:pos intersection", "-"))
    shared = workdir / "shared.snps"
    n_hcp, n_kg, n_shared = shared_variants(
        f"{hcp_chrpos}.bim", f"{kg_chrpos}.bim", shared)
    if n_shared < 30000:
        log(f"WARNING: shared variant count is {n_shared:,}; "
            "expected 50k–200k on NeuroChip. Check harmonisation.")
    hcp_shared = extract(plink, hcp_chrpos, shared, workdir / "hcp_shared", "hcp")
    kg_shared = extract(plink, kg_chrpos, shared, workdir / "kg_shared", "1kg")

    # 5. Merge with strand-flip retry
    log(banner("Step 5: Merge HCP + 1KG", "-"))
    merged_prefix, merge_stats = merge_with_flip_retry(
        plink, hcp_shared, kg_shared, workdir)
    n_merged_snps = count_lines(f"{merged_prefix}.bim")
    n_merged_samples = count_lines(f"{merged_prefix}.fam")
    log(f"Merged dataset: {n_merged_snps:,} variants, {n_merged_samples:,} samples")

    # 6. LD prune
    log(banner("Step 6: LD prune (--indep-pairwise 50 5 0.2)", "-"))
    ldpruned_prefix, n_pruned = ld_prune(plink, merged_prefix, workdir)
    log(f"After LD prune: {n_pruned:,} variants")
    if n_pruned < 30000:
        log(f"WARNING: post-prune variant count {n_pruned:,} is low; "
            "PCA may be unstable.")

    # 7. PCA (reference-only on 1KG, then project the full merged set)
    log(banner(f"Step 7: PCA ({args.n_pcs} components, reference-only + projection)", "-"))
    pcs, eigenvec_path = run_pca(
        plink2, ldpruned_prefix, args.n_pcs, psam["IID"].tolist(), workdir)

    # 8. Label samples + Mahalanobis
    log(banner("Step 8: Label populations + compute Mahalanobis distance", "-"))
    labelled = label_samples(pcs, psam)
    n_hcp_in_pca = (labelled["source"] == "HCP").sum()
    n_1kg_in_pca = (labelled["source"] == "1KG").sum()
    log(f"PCA samples: HCP={n_hcp_in_pca}, 1KG={n_1kg_in_pca}")

    labelled, mu, ref_mask = mahalanobis_distances(
        labelled, ref_subpops, args.mahalanobis_pcs)
    log(f"EUR-ref samples (centroid + cov source): {ref_mask.sum()}")

    labelled, cutoff_info = apply_threshold(
        labelled, ref_mask, args.mahalanobis_pcs,
        args.cutoff_method, args.chi2_quantile, args.sd_cutoff)
    log(f"Cutoff: D={cutoff_info['D_thresh']:.4f}  D2={cutoff_info['D2_thresh']:.4f}")

    # 9. Outputs
    log(banner("Step 9: Write outputs", "-"))
    keep_path, fail_path = write_keep_fail(labelled, outdir)
    per_sample_path = write_per_sample(labelled, args.mahalanobis_pcs, args.n_pcs, outdir)
    log(f"  Keep list:    {keep_path}")
    log(f"  Fail list:    {fail_path}")
    log(f"  Per-sample:   {per_sample_path}")

    final_prefix = build_clean_triplet(
        plink, qc_prefix_path, keep_path, qcdir, args.qc_prefix)
    log(f"  Clean triplet: {final_prefix}.bed/bim/fam")

    # 10. Plots
    log(banner("Step 10: Plots", "-"))
    scatter_path = figures_dir / "B1b_PCA_scatter.png"
    dist_path = figures_dir / "B1b_mahalanobis_distribution.png"
    plot_pca_scatter(labelled, mu, ref_subpops, scatter_path)
    plot_distance_distribution(labelled, ref_mask, cutoff_info, dist_path)
    log(f"  Scatter:      {scatter_path}")
    log(f"  Distribution: {dist_path}")

    # 11. Report
    hcp = labelled[labelled["source"] == "HCP"]
    n_hcp = len(hcp)
    n_pass = int(hcp["pass"].sum())
    n_fail = n_hcp - n_pass
    nearest_all = nearest_subpop(labelled, args.mahalanobis_pcs)
    fail_idx = hcp.index[~hcp["pass"]]
    fail_by_subpop = nearest_all.loc[fail_idx].value_counts()

    # Sanity: EUR self-pass and FIN behaviour
    ref_pass_rate = labelled.loc[ref_mask, "pass"].mean()
    fin_mask = labelled["Population"] == "FIN"
    fin_pass_rate = (labelled.loc[fin_mask, "pass"].mean()
                     if fin_mask.any() else float("nan"))

    pc_cols_K = [f"PC{i}" for i in range(1, args.mahalanobis_pcs + 1)]
    centroid_str = ", ".join(f"PC{i+1}={mu[i]:.4f}" for i in range(len(mu)))

    sections = [
        banner("B1b: ANCESTRY PCA + MAHALANOBIS REPORT"),
        "",
        f"Date:                   {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Project:                {project}",
        f"B1 clean prefix:        {qc_prefix_path}",
        f"Reference dir:          {refdir}",
        f"Workdir:                {workdir}",
        "",
        "CLI ARGUMENTS:",
        *(f"  --{k.replace('_','-')}: {v}" for k, v in vars(args).items()),
        "",
        banner("INPUTS", "-"),
        f"  Initial HCP samples:  {initial_hcp_n}",
        f"  Initial HCP variants: {initial_hcp_snps}",
        f"  1000G psam samples:   {len(psam)}",
        "",
        banner("VARIANT FILTERING", "-"),
        f"  HCP chr:pos variants: {n_hcp:,}",
        f"  1KG chr:pos variants: {n_kg:,}",
        f"  Shared variants:      {n_shared:,}",
        f"  Merge first .missnp:  {merge_stats['first_missnp']:,}",
        f"  After flip .missnp:   {merge_stats['after_flip_missnp']:,}",
        f"  Excluded both sides:  {merge_stats['final_excluded']:,}",
        f"  Merged variants:      {n_merged_snps:,}",
        f"  Merged samples:       {n_merged_samples:,}",
        f"  After LD prune:       {n_pruned:,}",
        "",
        banner("PCA + MAHALANOBIS", "-"),
        f"  PCs computed:         {args.n_pcs}",
        f"  PCs used for D:       {args.mahalanobis_pcs}",
        f"  Reference subpops:    {','.join(ref_subpops)}",
        f"  Reference samples:    {int(ref_mask.sum())}",
        f"  Centroid (top {args.mahalanobis_pcs} PCs): {centroid_str}",
        f"  Cutoff method:        {cutoff_info['method']}",
        *(f"  {k}: {v}" for k, v in cutoff_info.items() if k != "method"),
        "",
        banner("HCP RESULTS", "-"),
        f"  HCP total:            {n_hcp}",
        f"  HCP pass:             {n_pass}",
        f"  HCP fail:             {n_fail}",
        "",
        "  HCP fail breakdown by nearest 1KG subpopulation:",
        *(f"    {pop:6s} {cnt}" for pop, cnt in fail_by_subpop.items()),
        "",
        banner("SANITY CHECKS", "-"),
        f"  EUR-reference self-pass rate: {ref_pass_rate*100:.2f}%   "
        f"(expected ~{args.chi2_quantile*100:.1f}% for chi2; "
        "deviations >1% suggest covariance estimate is off)",
        f"  FIN pass rate:                "
        f"{fin_pass_rate*100:.2f}% (excluded from reference; "
        "low rate confirms exclusion is biting)",
        "",
        banner("OUTPUTS", "-"),
        f"  Keep list:        {keep_path}",
        f"  Fail list:        {fail_path}",
        f"  Per-sample CSV:   {per_sample_path}",
        f"  Clean triplet:    {final_prefix}.bed/bim/fam",
        f"  PCA scatter:      {scatter_path}",
        f"  Distance dist:    {dist_path}",
        "",
        banner("END OF REPORT"),
    ]
    report_path = reports_dir / "B1b_ancestry_PCA_mahalanobis_report.txt"
    write_report(report_path, sections)
    log(f"Report: {report_path}")

    log("")
    log(banner("B1b complete"))
    log(f"  HCP pass: {n_pass} / {n_hcp}")
    log(f"  Clean triplet for B2: {final_prefix}.bed/bim/fam")


if __name__ == "__main__":
    main()
