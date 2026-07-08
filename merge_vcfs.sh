#!/bin/bash
#merges all quality-filtered VCFs into a single multi-sample VCF.
#then converts to a feature matrix CSV using Python/cyvcf2.

# Outputs:
#   ~/tb_pipeline/merged.vcf.gz          multi-sample merged VCF
#   ~/tb_pipeline/variant_matrix.csv     raw feature matrix (samples x variants)
#   ~/tb_pipeline/variant_matrix_filtered.csv  after MAF frequency filtering

set -uo pipefail

BASE="$HOME/tb_pipeline"
INPUT_DIR="$BASE/vcf_filtered"
OUT_DIR="$BASE"
VCF_LIST="$BASE/vcf_list.txt"
MERGED_VCF="$BASE/merged.vcf.gz"
LOG_DIR="$BASE/logs"

# ── Activate environment ──────────────────────────────────────────────────────
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate tb_amr 2>/dev/null || true

if ! command -v bcftools &>/dev/null; then
    echo "ERROR: bcftools not found. Activate tb_amr first."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Step 1: Build sorted VCF file list ───────────────────────────────────────
echo "================================================================"
echo "Step 1: Building VCF file list"
echo "================================================================"

ls "$INPUT_DIR"/*.vcf.gz 2>/dev/null | sort > "$VCF_LIST"
total=$(wc -l < "$VCF_LIST")

if [ "$total" -eq 0 ]; then
    echo "ERROR: No .vcf.gz files found in $INPUT_DIR"
    exit 1
fi

echo "Found $total VCF files"
echo "VCF list saved to: $VCF_LIST"

# ── Step 2: Merge all VCFs ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Step 2: Merging $total VCFs with bcftools merge"
echo "================================================================"
echo "Started: $(date)"
echo "This may take 20-60 minutes for large datasets..."

if [ -f "$MERGED_VCF" ]; then
    echo "merged.vcf.gz already exists — skipping merge step."
    echo "Delete $MERGED_VCF to force re-merge."
else
    bcftools merge \
        --file-list "$VCF_LIST" \
        --missing-to-ref \
        --force-samples \
        -Oz \
        -o "$MERGED_VCF" \
        2>"$LOG_DIR/merge.log"

    if [ $? -ne 0 ] || [ ! -s "$MERGED_VCF" ]; then
        echo "ERROR: bcftools merge failed. Check $LOG_DIR/merge.log"
        exit 1
    fi

    bcftools index "$MERGED_VCF"
    echo "Merge complete: $(date)"
fi

# ── Step 3: Verify merged VCF ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Step 3: Verifying merged VCF"
echo "================================================================"

n_samples=$(bcftools query -l "$MERGED_VCF" | wc -l)
n_variants=$(bcftools view "$MERGED_VCF" 2>/dev/null | grep -vc "^#" || echo "?")

echo "Samples in merged VCF:  $n_samples"
echo "Variant positions:      $n_variants"
echo "File size:              $(du -sh "$MERGED_VCF" | cut -f1)"

if [ "$n_samples" -lt 10 ]; then
    echo "WARNING: Very few samples in merged VCF. Check that INPUT_DIR is correct."
fi

# ── Step 4: Convert to feature matrix CSV ────────────────────────────────────
echo ""
echo "================================================================"
echo "Step 4: Converting merged VCF to feature matrix CSV"
echo "================================================================"

# Install cyvcf2 if needed
python3 -c "import cyvcf2" 2>/dev/null || \
    mamba install -y -n tb_amr -c bioconda cyvcf2

python3 << 'PYEOF'
import os
import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "tb_pipeline")
MERGED_VCF = os.path.join(BASE, "merged.vcf.gz")
OUT_RAW = os.path.join(BASE, "variant_matrix.csv")
OUT_FILTERED = os.path.join(BASE, "variant_matrix_filtered.csv")

try:
    import cyvcf2
except ImportError:
    print("ERROR: cyvcf2 not installed. Run: mamba install -n tb_amr -c bioconda cyvcf2")
    exit(1)

print(f"Reading: {MERGED_VCF}")
vcf = cyvcf2.VCF(MERGED_VCF)
samples = vcf.samples
print(f"Samples: {len(samples)}")

variants = []
variant_ids = []
batch_size = 10000
batch_count = 0

for v in vcf:
    # Feature name: CHROM_POS_REF_ALT
    feat = f"{v.CHROM}_{v.POS}_{v.REF}_{v.ALT[0]}"

    # Extract genotype per sample: 1 if alt allele present, 0 if reference, NaN if missing
    gts = []
    for gt in v.genotypes:
        alleles = gt[:2]
        if -1 in alleles:
            gts.append(np.nan)
        elif 1 in alleles:
            gts.append(1)
        else:
            gts.append(0)

    variants.append(gts)
    variant_ids.append(feat)
    batch_count += 1

    if batch_count % batch_size == 0:
        print(f"  Processed {batch_count} variant positions...")

vcf.close()
print(f"Total variant positions: {len(variant_ids)}")

# Build feature matrix: rows = samples, columns = variant positions
print("Building feature matrix...")
matrix = np.array(variants, dtype=np.float32).T
df = pd.DataFrame(matrix, index=samples, columns=variant_ids)
df.index.name = "sample_id"

print(f"Raw feature matrix shape: {df.shape}")

# Save raw matrix
df.reset_index().to_csv(OUT_RAW, index=False)
print(f"Saved: variant_matrix.csv ({df.shape[0]} samples x {df.shape[1]} variants)")

# ── Frequency filtering ───────────────────────────────────────────────────────
print("\nApplying frequency filters...")
n_samples = df.shape[0]

# Count non-NaN alt allele presence per variant
alt_counts = df.sum(skipna=True)
nonmissing_counts = df.notna().sum()

# Remove variants present in fewer than 1% of samples
min_count = max(1, int(0.01 * n_samples))
# Remove variants present in more than 99% of samples
max_count = int(0.99 * n_samples)

keep_mask = (alt_counts >= min_count) & (alt_counts <= max_count)
df_filtered = df.loc[:, keep_mask]

print(f"Before frequency filter: {df.shape[1]} variants")
print(f"After MAF >=1%:          removed {(alt_counts < min_count).sum()} rare variants")
print(f"After MAF <=99%:         removed {(alt_counts > max_count).sum()} near-fixed variants")
print(f"After filtering:         {df_filtered.shape[1]} variants retained")

# Fill remaining NaN with 0 (treat missing as reference)
df_filtered = df_filtered.fillna(0).astype(int)

# Save filtered matrix
df_filtered.reset_index().to_csv(OUT_FILTERED, index=False)
print(f"Saved: variant_matrix_filtered.csv ({df_filtered.shape[0]} samples x {df_filtered.shape[1]} variants)")
print("\nFeature matrix construction complete.")
print("Next step: join with labels.csv to build final ML dataset.")
PYEOF

echo ""
echo "================================================================"
echo "ALL STEPS COMPLETE"
echo "================================================================"
echo "Outputs:"
echo "  $MERGED_VCF"
echo "  $BASE/variant_matrix.csv"
echo "  $BASE/variant_matrix_filtered.csv"
echo ""
echo "Next step:"
echo "  python3 ~/tb_pipeline/py_scripts/join_features_labels.py"
echo "================================================================"