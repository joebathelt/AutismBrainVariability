# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
ORIGINAL_BASE = Path('/Users/joebathelt/Documents/1_Projects/BrainCompensation')
DOCKER_BASE = Path('/Users/joebathelt/Documents/1_Projects/BrainCompensation_Docker')

ORIGINAL_PLINK = ORIGINAL_BASE / 'data/PLINK_anonymised'
DOCKER_PLINK = DOCKER_BASE / 'data/plink'

print("="*80)
print("SYSTEMATIC COMPARISON: ORIGINAL vs DOCKER PIPELINE")
print("="*80)

# ============================================================================
# Helper Functions
# ============================================================================
def compare_files(original_path, docker_path, id_cols, value_col, 
                  description, check_counts=True):
    """Compare two files and report correlation."""
    print(f"\n{description}")
    print("-" * 60)
    
    try:
        # Load files
        orig = pd.read_csv(original_path, sep='\s+', header=None if original_path.suffix == '.txt' else 'infer')
        docker = pd.read_csv(docker_path, sep='\s+', header=None if docker_path.suffix == '.txt' else 'infer')
        
        # Check row counts
        if check_counts:
            print(f"  Original rows: {len(orig)}")
            print(f"  Docker rows:   {len(docker)}")
            if len(orig) != len(docker):
                print(f"  ⚠️  WARNING: Different sample sizes!")
        
        # Merge and correlate
        if isinstance(id_cols, list):
            merged = pd.merge(orig, docker, on=id_cols, suffixes=('_orig', '_docker'))
            orig_col = f"{value_col}_orig"
            docker_col = f"{value_col}_docker"
        else:
            merged = pd.merge(orig, docker, left_on=id_cols, right_on=id_cols, 
                            suffixes=('_orig', '_docker'))
            orig_col = value_col if value_col in merged.columns else f"{value_col}_orig"
            docker_col = value_col if value_col in merged.columns else f"{value_col}_docker"
        
        # Ensure numeric
        merged[orig_col] = pd.to_numeric(merged[orig_col], errors='coerce')
        merged[docker_col] = pd.to_numeric(merged[docker_col], errors='coerce')
        
        # Remove NaN
        valid = merged[[orig_col, docker_col]].dropna()
        
        if len(valid) > 0:
            r, p = pearsonr(valid[orig_col], valid[docker_col])
            print(f"  Correlation: r = {r:.6f} (p = {p:.2e})")
            print(f"  Valid pairs: {len(valid)}")
            
            # Summary statistics
            print(f"  Original: mean={valid[orig_col].mean():.4f}, std={valid[orig_col].std():.4f}")
            print(f"  Docker:   mean={valid[docker_col].mean():.4f}, std={valid[docker_col].std():.4f}")
            
            # Interpretation
            if r > 0.99:
                print(f"  ✅ Excellent agreement (r > 0.99)")
            elif r > 0.95:
                print(f"  ✓  Good agreement (r > 0.95)")
            elif r > 0.85:
                print(f"  ⚠️  Moderate agreement (r > 0.85)")
            else:
                print(f"  ❌ Poor agreement (r < 0.85)")
            
            return r
        else:
            print(f"  ❌ No valid data pairs found!")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None

# ============================================================================
# STAGE 1: Input Data (Behavioral/Phenotypic)
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: INPUT DATA VERIFICATION")
print("="*80)

# Social cognition scores
compare_files(
    ORIGINAL_BASE / 'data/cfa_factor_scores_full_sample.csv',
    DOCKER_BASE / 'data/cfa_factor_scores_full_sample.csv',
    'Subject',
    'Social_Score',
    "Social Cognition Scores"
)

# ============================================================================
# STAGE 2: QC Pipeline Outputs (B1)
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: QC PIPELINE OUTPUTS (B1)")
print("="*80)

# Check if key QC files exist and compare sample sizes
qc_files = [
    ('Neuro_Chip_qc_nodup_sexfiltered.fam', 'Final QC sample'),
    ('Neuro_Chip_full_sample_pca.eigenvec', 'Full sample PCs'),
    ('Neuro_Chip_qc_final_unrelated.fam', 'Unrelated sample')
]

for filename, description in qc_files:
    print(f"\n{description} ({filename})")
    print("-" * 60)
    try:
        orig_count = len(pd.read_csv(ORIGINAL_PLINK / filename, sep='\s+', header=None))
        docker_count = len(pd.read_csv(DOCKER_PLINK / filename, sep='\s+', header=None))
        print(f"  Original: {orig_count} samples")
        print(f"  Docker:   {docker_count} samples")
        if orig_count == docker_count:
            print(f"  ✅ Sample counts match")
        else:
            print(f"  ❌ Sample counts differ by {abs(orig_count - docker_count)}")
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

# Compare PCs (first 5 components)
print(f"\nPrincipal Components (PC1-PC5)")
print("-" * 60)
try:
    orig_pcs = pd.read_csv(ORIGINAL_PLINK / 'Neuro_Chip_full_sample_pca.eigenvec', 
                           sep='\s+', header=None)
    docker_pcs = pd.read_csv(DOCKER_PLINK / 'Neuro_Chip_full_sample_pca.eigenvec', 
                             sep='\s+', header=None)
    
    # Merge on IID (column 1)
    merged_pcs = pd.merge(orig_pcs, docker_pcs, on=1, suffixes=('_orig', '_docker'))
    
    for pc in range(2, 7):  # PC1-PC5 are columns 2-6
        orig_col = f"{pc}_orig"
        docker_col = f"{pc}_docker"
        r, _ = pearsonr(merged_pcs[orig_col], merged_pcs[docker_col])
        print(f"  PC{pc-1}: r = {r:.6f}")
    
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# ============================================================================
# STAGE 3: PGS Calculation (PRSice Output - B1 Step 13)
# ============================================================================
print("\n" + "="*80)
print("STAGE 3: PRSICE PGS CALCULATION (B1 Step 13)")
print("="*80)

# Compare PRSice all_score file
print(f"\nPRSice PGS at p < 0.005")
print("-" * 60)
try:
    orig_prsice = pd.read_csv(ORIGINAL_PLINK / 'Neuro_Chip_unrelated_prsice.all_score', 
                              sep='\s+')
    docker_prsice = pd.read_csv(DOCKER_PLINK / 'Neuro_Chip_unrelated_prsice.all_score', 
                                sep='\s+')
    
    # Find common threshold column (e.g., Pt_0.00510005)
    threshold_cols = [col for col in orig_prsice.columns if col.startswith('Pt_')]
    if threshold_cols:
        test_col = threshold_cols[0]
        merged = pd.merge(orig_prsice[['FID', 'IID', test_col]], 
                         docker_prsice[['FID', 'IID', test_col]], 
                         on=['FID', 'IID'], 
                         suffixes=('_orig', '_docker'))
        r, p = pearsonr(merged[f'{test_col}_orig'], merged[f'{test_col}_docker'])
        print(f"  Threshold {test_col}")
        print(f"  Correlation: r = {r:.6f} (p = {p:.2e})")
        if r > 0.99:
            print(f"  ✅ Excellent agreement")
        else:
            print(f"  ⚠️  Correlation below 0.99")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# ============================================================================
# STAGE 4: Threshold Selection (B2 Output)
# ============================================================================
print("\n" + "="*80)
print("STAGE 4: THRESHOLD SELECTION (B2)")
print("="*80)

# Compare selected unrelated PGS scores
compare_files(
    ORIGINAL_PLINK / 'unrelated_prs_scores.txt',
    DOCKER_PLINK / 'unrelated_prs_scores.txt',
    [0, 1],  # FID, IID
    2,       # PGS value
    "Selected Threshold PGS (Unrelated Sample)",
    check_counts=True
)

# ============================================================================
# STAGE 5: BLUP Extension (B3 Outputs)
# ============================================================================
print("\n" + "="*80)
print("STAGE 5: BLUP EXTENSION (B3)")
print("="*80)

# Compare BLUP-extended PGS
print(f"\nBLUP-Extended PGS (full sample)")
print("-" * 60)
try:
    orig_blup = pd.read_csv(ORIGINAL_PLINK / 'full_prs_scores.snp.blp.profile', sep='\s+')
    docker_blup = pd.read_csv(DOCKER_PLINK / 'full_prs_scores.snp.blp.profile', sep='\s+')
    
    merged = pd.merge(orig_blup[['IID', 'SCORESUM']], 
                     docker_blup[['IID', 'SCORESUM']], 
                     on='IID', 
                     suffixes=('_orig', '_docker'))
    
    r, p = pearsonr(merged['SCORESUM_orig'], merged['SCORESUM_docker'])
    print(f"  Correlation: r = {r:.6f} (p = {p:.2e})")
    print(f"  Valid pairs: {len(merged)}")
    print(f"  Original: mean={merged['SCORESUM_orig'].mean():.4f}, std={merged['SCORESUM_orig'].std():.4f}")
    print(f"  Docker:   mean={merged['SCORESUM_docker'].mean():.4f}, std={merged['SCORESUM_docker'].std():.4f}")
    
    if r > 0.95:
        print(f"  ✅ Good BLUP agreement (r > 0.95)")
    elif r > 0.85:
        print(f"  ⚠️  Moderate BLUP agreement (0.85 < r < 0.95)")
    else:
        print(f"  ❌ Poor BLUP agreement (r < 0.85)")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# ============================================================================
# STAGE 6: Final Residualized PGS (B4 Output)
# ============================================================================
print("\n" + "="*80)
print("STAGE 6: FINAL RESIDUALIZED PGS (B4)")
print("="*80)

# Raw BLUP PGS
compare_files(
    ORIGINAL_BASE / 'data/prs_residuals.csv',
    DOCKER_BASE / 'data/prs_residuals.csv',
    'Subject',
    'blup_PRS',
    "Raw BLUP PGS (from prs_residuals.csv)"
)

# Residualized BLUP PGS
compare_files(
    ORIGINAL_BASE / 'data/prs_residuals.csv',
    DOCKER_BASE / 'data/prs_residuals.csv',
    'Subject',
    'blup_PRS_residuals',
    "Residualized BLUP PGS (PC/Age/Sex corrected)"
)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print("""
Key stages to check if correlations are low:

1. If STAGE 3 (PRSice) < 0.99:
   → QC pipeline (B1) produced different data
   → Check: sample filtering, SNP filtering, harmonization

2. If STAGE 4 (Threshold Selection) < 0.99:
   → Different threshold was selected OR
   → Different PC correction applied in B2

3. If STAGE 5 (BLUP) < 0.95:
   → BLUP procedure differs
   → Check: GRM calculation, GCTA inputs, genetic data differences

4. If STAGE 6 (Residuals) < 0.95:
   → Residualization differs
   → Check: which PCs/covariates used, sample composition

The correlation should improve at each stage, not get worse!
""")

print("="*80)
print("COMPARISON COMPLETE")
print("="*80)
# %%
