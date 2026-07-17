#!/bin/bash
# merge_vcfs_fast.sh — Optimised VCF merger + feature matrix
# Speedup: bcftools pre-filter, chunked numpy

set -uo pipefail

BASE="$HOME/tb_pipeline"
INPUT_DIR="$BASE/vcf_filtered"
OUT_DIR="$BASE"
VCF_LIST="$BASE/vcf_list.txt"
MERGED_VCF="$BASE/merged.vcf.gz"
FILTERED_VCF="$BASE/merged_prefilt.vcf.gz"
LOG_DIR="$BASE/logs"
THREADS=8          # adjust to your CPU count
MIN_AF=0.01        # pre-filter: remove variants in <1% of samples
MAX_AF=0.99        # pre-filter: remove near-fixed variants

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate tb_amr 2>/dev/null || true

if ! command -v bcftools &>/dev/null; then
    echo "ERROR: bcftools not found."; exit 1
fi

mkdir -p "$LOG_DIR"

#making a VCF list 
echo "Step 1: VCF list"
ls "$INPUT_DIR"/*.vcf.gz 2>/dev/null | sort > "$VCF_LIST"
TOTAL=$(wc -l < "$VCF_LIST")
[ "$TOTAL" -eq 0 ] && { echo "ERROR: No VCFs in $INPUT_DIR"; exit 1; }
echo "Found $TOTAL VCF files"

#indexing any un-indexed VCFs in parallel
echo ""
echo "Step 2: Parallel indexing"
cat "$VCF_LIST" | xargs -P "$THREADS" -I {} sh -c '
    vcf="{}"
    if [ ! -f "${vcf}.csi" ]; then
        echo "  Indexing $(basename $vcf)"
        bcftools index --csi "$vcf"
    fi
'
echo "All VCFs indexed."

#merging with pre-filtering
echo ""
echo "Step 3: Merge + pre-filter ($(date))"

if [ ! -f "$FILTERED_VCF" ]; then
    bcftools merge \
        --file-list "$VCF_LIST" \
        --missing-to-ref \
        --force-samples \
        --output-type u \
        --threads "$THREADS" \
    | bcftools view \
        --min-af "${MIN_AF}:alt1" \
        --max-af "${MAX_AF}:alt1" \
        --output-type z \
        --output "$FILTERED_VCF" \
        --threads "$THREADS" \
        2>"$LOG_DIR/merge_prefilt.log"

    bcftools index --csi --threads "$THREADS" "$FILTERED_VCF"
else
    echo "  Pre-filtered VCF exists — skipping."
fi

N_SAMPLES=$(bcftools query -l "$FILTERED_VCF" | wc -l)
N_VARS=$(bcftools view -H "$FILTERED_VCF" | wc -l)
echo "  Samples: $N_SAMPLES  |  Variants after pre-filter: $N_VARS"
echo "  Merge+filter done: $(date)"

#dumping GT matrix via bcftools query
echo ""
echo "Step 4: Extract genotype table ($(date))"
GT_TSV="$BASE/gt_matrix.tsv.gz"

if [ ! -f "$GT_TSV" ]; then
    # Extract: CHROM_POS_REF_ALT then one GT column per sample
    (
        
        echo -n "variant_id"
        bcftools query -l "$FILTERED_VCF" | tr '\n' '\t' | sed 's/\t$/\n/'
        # Data lines
        bcftools query \
            --format '%CHROM\_%POS\_%REF\_%ALT[\t%GT]\n' \
            "$FILTERED_VCF"
    ) | bgzip -@ "$THREADS" > "$GT_TSV"
    echo "  GT table written: $GT_TSV"
else
    echo "  GT table exists — skipping."
fi

#fast chunked conversion to binary matrix
echo ""
echo "Step 5: Binary matrix conversion ($(date))"

python3 << 'PYEOF'
import os, gzip, time
import numpy as np
import pandas as pd

HOME   = os.path.expanduser("~")
BASE   = os.path.join(HOME, "tb_pipeline")
GT_TSV = os.path.join(BASE, "gt_matrix.tsv.gz")
OUT_NPZ  = os.path.join(BASE, "feature_matrix.npz")         # compressed numpy — fastest to save/load
OUT_CSV  = os.path.join(BASE, "variant_matrix_filtered.csv") # CSV for compatibility
OUT_META = os.path.join(BASE, "variant_metadata.csv")        # variant ID lookup

CHUNK_SIZE = 50_000   # variants per chunk — adjust down if RAM is tight

t0 = time.time()

print(f"Opening {GT_TSV} ...")
fh = gzip.open(GT_TSV, "rt")

#read header
header = fh.readline().rstrip("\n").split("\t")
variant_col = header[0]          # "variant_id"
sample_names = header[1:]
N = len(sample_names)
print(f"Samples: {N}")

def gt_to_bit(gt_str: str) -> np.uint8:
    """
    Fastest GT parser: no regex, no split overhead.
    Returns 1 if any ALT allele, 0 if all REF, 0 if missing.
    """
    if len(gt_str) == 1:
        return np.uint8(0) if gt_str in ("0", ".") else np.uint8(1)
    # diploid: check char 0 and char 2 (separator at index 1)
    a1 = gt_str[0]
    a2 = gt_str[2] if len(gt_str) > 2 else "0"
    return np.uint8(1 if (a1 not in ("0", ".") or a2 not in ("0", ".")) else 0)

all_chunks   = []
all_var_ids  = []
chunk_rows   = []
chunk_ids    = []
line_count   = 0

for line in fh:
    line = line.rstrip("\n")
    parts = line.split("\t")
    var_id = parts[0]
    gts    = parts[1:]

    #convert genotype strings to uint8 bits — pure numpy, no loop
    row = np.frombuffer(
        bytes([gt_to_bit(g) for g in gts]),
        dtype=np.uint8
    )
    chunk_rows.append(row)
    chunk_ids.append(var_id)
    line_count += 1

    if line_count % CHUNK_SIZE == 0:
        all_chunks.append(np.stack(chunk_rows, axis=0))   # shape (CHUNK_SIZE, N)
        all_var_ids.extend(chunk_ids)
        chunk_rows = []
        chunk_ids  = []
        elapsed = time.time() - t0
        print(f"  {line_count:,} variants processed — {elapsed:.0f}s elapsed")

#final partial chunk
if chunk_rows:
    all_chunks.append(np.stack(chunk_rows, axis=0))
    all_var_ids.extend(chunk_ids)

fh.close()

print(f"\nStacking {line_count:,} variants × {N} samples ...")
# Shape: (n_variants, n_samples)
matrix = np.concatenate(all_chunks, axis=0)   # uint8 — 8x smaller than float32

print(f"Matrix shape: {matrix.shape}")
print(f"Memory: {matrix.nbytes / 1e6:.1f} MB")

# Transpose → (n_samples, n_variants)
matrix_T = matrix.T    # shape: (N, n_variants)


#Save compressed numpy (instant load later for ML)
print(f"\nSaving NPZ (compressed)...")
np.savez_compressed(
    OUT_NPZ,
    matrix=matrix_T,
    samples=np.array(sample_names),
    variants=np.array(all_var_ids)
)
print(f"{OUT_NPZ}  ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")


#Save variant metadata
meta_parts = [v.rsplit("_", 2) for v in all_var_ids]
meta_df = pd.DataFrame(all_var_ids, columns=["variant_id"])
meta_df[["CHROM","POS","REF","ALT"]] = pd.DataFrame(
    [v.split("_", 3) for v in all_var_ids]
)
meta_df.to_csv(OUT_META, index=False)
print(f"{OUT_META}")


#Save CSV (chunked write — avoids OOM on large matrices)
print(f"\nWriting CSV in chunks...")
WRITE_CHUNK = 200   # samples per write chunk

with open(OUT_CSV, "w") as f:
    # Header
    f.write("sample_id," + ",".join(all_var_ids) + "\n")
    for start in range(0, N, WRITE_CHUNK):
        end = min(start + WRITE_CHUNK, N)
        rows = []
        for i in range(start, end):
            rows.append(sample_names[i] + "," + ",".join(map(str, matrix_T[i])))
        f.write("\n".join(rows) + "\n")
        if (start // WRITE_CHUNK) % 10 == 0:
            print(f"  Written {end}/{N} samples...")

print(f"{OUT_CSV}")

elapsed_total = time.time() - t0
print(f"\nDONE in {elapsed_total/60:.1f} minutes")
print(f"Matrix: {N} samples x {len(all_var_ids)} variants")
print(f"\nTo load in ML script:")
print(f"  data = np.load('{OUT_NPZ}')")
print(f"  X = data['matrix']        # shape (samples, variants)")
print(f"  samples = data['samples'] # sample IDs")
print(f"  variants = data['variants'] # variant IDs")
PYEOF

echo ""
echo "ALL DONE: $(date)"
echo "Outputs:"
echo "  $FILTERED_VCF                   — pre-filtered merged VCF"
echo "  $BASE/feature_matrix.npz        — compressed binary matrix (use for ML)"
echo "  $BASE/variant_matrix_filtered.csv — CSV version"
echo "  $BASE/variant_metadata.csv      — variant annotations"