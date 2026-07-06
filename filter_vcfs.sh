#!/bin/bash
# Filters all .vcf.gz files using QUAL>=20 and DP>=4 thresholds.
#
# Output:
#   ~/tb_pipeline/vcf_filtered/<ID>.vcf.gz
#   ~/tb_pipeline/vcf_filtered/<ID>.vcf.gz.csi

set -uo pipefail

BASE="$HOME/tb_pipeline"
INPUT_DIR="$BASE/vcf_output"
OUTPUT_DIR="$BASE/vcf_filtered"
LOG_FILE="$BASE/logs/filter_vcfs.log"
FAIL_LOG="$BASE/logs/filter_vcfs_failed.log"

#activating tb_amr env if not already active
if ! command -v bcftools &>/dev/null; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate tb_amr 2>/dev/null || true
fi

# Confirm bcftools is now available
if ! command -v bcftools &>/dev/null; then
    echo "ERROR: bcftools not found. Activate tb_amr first: mamba activate tb_amr"
    exit 1
fi

# --- Setup ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASE/logs"

# Clear old logs for this run
> "$LOG_FILE"
> "$FAIL_LOG"

#Count input files
input_files=("$INPUT_DIR"/*.vcf.gz)
total=${#input_files[@]}

if [ "$total" -eq 0 ] || [ ! -f "${input_files[0]}" ]; then
    echo "ERROR: No .vcf.gz files found in $INPUT_DIR"
    echo "Make sure you've run the consolidation step first."
    exit 1
fi
echo ""
echo "VCF Quality Filter"
echo "Input:    $INPUT_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Filter:   QUAL>=20 AND DP>=4 (exclude QUAL<20 OR DP<4)"
echo "Files:    $total"
echo "Started:  $(date)"

#estimating time from first 5 files
echo "Running quick speed test on 5 files to estimate total time..."
sample_files=("${input_files[@]:0:5}")
t_start=$(date +%s)

for vcf in "${sample_files[@]}"; do
    id=$(basename "$vcf" .vcf.gz)
    bcftools filter -e 'QUAL<20 || DP<4' -Oz -o "$OUTPUT_DIR/${id}.vcf.gz" "$vcf" 2>/dev/null
done

t_end=$(date +%s)
t_elapsed=$((t_end - t_start))

if [ "$t_elapsed" -gt 0 ]; then
    rate=$(echo "scale=2; $t_elapsed / 5" | bc)
    estimated=$(echo "scale=0; ($rate * $total) / 60" | bc)
    echo "Speed test: 5 files in ${t_elapsed}s (~${rate}s per file)"
    echo "Estimated total time: ~${estimated} minutes for $total files"
else
    echo "Speed test: 5 files in <1s — very fast"
    echo "Estimated total time: a few minutes at most"
fi
echo ""

#main filtration starts here
n_done=0
n_failed=0
n_skipped=0

for vcf in "${input_files[@]}"; do
    id=$(basename "$vcf" .vcf.gz)
    out="$OUTPUT_DIR/${id}.vcf.gz"

    #skips already filtered files
    if [ -f "$out" ] && [ -s "$out" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    #filtering 
    bcftools filter -e 'QUAL<20 || DP<4' -Oz -o "$out" "$vcf" 2>/dev/null

    if [ $? -ne 0 ] || [ ! -s "$out" ]; then
        echo "FAILED: $id" | tee -a "$FAIL_LOG"
        n_failed=$((n_failed + 1))
        rm -f "$out"
        continue
    fi

    #indexing the filtered vcf files
    bcftools index "$out" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "INDEX FAILED: $id" | tee -a "$FAIL_LOG"
        n_failed=$((n_failed + 1))
        rm -f "$out" "$out.csi"
        continue
    fi

    echo "$id" >> "$LOG_FILE"
    n_done=$((n_done + 1))

    #progress generation every 100 files filtered
    total_processed=$((n_done + n_skipped))
    if [ $((total_processed % 100)) -eq 0 ]; then
        echo "Progress: $total_processed / $total done (${n_failed} failed, ${n_skipped} skipped)"
    fi
done

echo ""
echo "Filtering complete: $(date)"
echo "Total input files:     $total"
echo "Newly filtered:        $n_done"
echo "Already existed (skipped): $n_skipped"
echo "Failed:                $n_failed"
echo "Output folder:         $OUTPUT_DIR"
echo ""

#counting and comparing variants for cross checking 
echo "Checking variant counts (sample of 3 files)..."
for vcf in "${input_files[@]:0:3}"; do
    id=$(basename "$vcf" .vcf.gz)
    raw_count=$(bcftools view "$vcf" 2>/dev/null | grep -vc "^#" || echo "?")
    filt_count=$(bcftools view "$OUTPUT_DIR/${id}.vcf.gz" 2>/dev/null | grep -vc "^#" || echo "?")
    echo "  $id: $raw_count raw variants → $filt_count after filtering"
done

echo ""
if [ "$n_failed" -gt 0 ]; then
    echo "Failed files logged to: $FAIL_LOG"
    echo "Re-run this script to retry failed files (they are auto-skipped if already done)"
fi