#!/bin/bash
#run_tbprofiler.sh
#runs TB-Profiler on all filtered VCFs to generate:
#  - drug resistance predictions which are labels for ml model
#  - lineage classifications
#  - compensatory mutation candidates
#automatically fixes chromosome naming in VCF headers (NC_000962.3 → Chromosome)
#uses --caller bcftools (matches how VCFs were generated)

set -uo pipefail

BASE="$HOME/tb_pipeline"
INPUT_DIR="$BASE/vcf_filtered"
OUT_DIR="$BASE/tbprofiler_results"
LOG_DIR="$BASE/logs"
SUCCESS_LOG="$LOG_DIR/tbprofiler_success.log"
FAIL_LOG="$LOG_DIR/tbprofiler_failed.log"

#activating the virtual environment
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate tb_amr 2>/dev/null || true

#checking if tb-profiler is installed
if ! command -v tb-profiler &>/dev/null; then
    echo "TB-Profiler not found. Installing into tb_amr..."
    mamba install -y -n tb_amr -c bioconda tb-profiler
fi

echo "TB-Profiler version: $(tb-profiler version 2>/dev/null || echo 'unknown')"

#fixing chromosome names in VCF headers (NC_000962.3 → Chromosome)
CHR_MAP_FILE="$BASE/reference/chr_map.txt"
mkdir -p "$(dirname "$CHR_MAP_FILE")"
echo -e "NC_000962.3\tChromosome" > "$CHR_MAP_FILE"

# Check if any VCF still has NC_000962.3 in its *data lines* (not header)
needs_fix=false
first_vcf="${INPUT_DIR}/$(ls -1 "$INPUT_DIR"/*.vcf.gz 2>/dev/null | head -1 | xargs basename 2>/dev/null)"
if [ -n "$first_vcf" ] && zgrep -q "^NC_000962.3" "$INPUT_DIR/$first_vcf" 2>/dev/null; then
    needs_fix=true
fi

if $needs_fix; then
    echo "Fixing chromosome names in all VCFs (NC_000962.3 → Chromosome)..."
    cd "$INPUT_DIR" || exit 1
    for vcf in *.vcf.gz; do
        echo "  $vcf"
        bcftools annotate --rename-chrs "$CHR_MAP_FILE" -Oz -o "${vcf}.tmp" "$vcf" && mv "${vcf}.tmp" "$vcf"
        bcftools index -f "$vcf" 2>/dev/null
    done
    echo "Chromosome fix complete."
else
    echo "Chromosome names already correct (Chromosome) - skipping rename."
fi


#file setup
mkdir -p "$OUT_DIR/results" "$LOG_DIR"
> "$SUCCESS_LOG"
> "$FAIL_LOG"

#input file count
input_files=("$INPUT_DIR"/*.vcf.gz)
total=${#input_files[@]}

if [ "$total" -eq 0 ] || [ ! -f "${input_files[0]}" ]; then
    echo "ERROR: No .vcf.gz files found in $INPUT_DIR"
    echo "Run filter_vcfs.sh first to generate filtered VCFs."
    exit 1
fi

echo ""
echo "TB-Profiler: Drug Resistance + Lineage + Compensatory Mutations"
echo ""
echo "Input:   $INPUT_DIR"
echo "Output:  $OUT_DIR"
echo "Samples: $total"
echo "Started: $(date)"
echo ""

#speed test on first 3 samples
echo "Running speed test on 3 samples to estimate total time..."
t_start=$(date +%s)
for vcf in "${input_files[@]:0:3}"; do
    id=$(basename "$vcf" .vcf.gz)
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller bcftools 2>/dev/null
done
t_end=$(date +%s)
t_elapsed=$((t_end - t_start))

if [ "$t_elapsed" -gt 0 ]; then
    rate=$(echo "scale=1; $t_elapsed / 3" | bc 2>/dev/null || echo "?")
    estimated=$(echo "scale=0; ($t_elapsed * $total) / (3 * 60)" | bc 2>/dev/null || echo "?")
    echo "Speed: ~${rate}s per sample"
    echo "Estimated total time: ~${estimated} minutes for $total samples"
else
    echo "Speed test: <1 second per sample — very fast"
fi
echo ""

#main loop
n_done=0
n_failed=0
n_skipped=0

for vcf in "${input_files[@]}"; do
    id=$(basename "$vcf" .vcf.gz)
    result_json="$OUT_DIR/results/${id}.results.json"

    #resume logic
    if [ -f "$result_json" ] && [ -s "$result_json" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    #running TB-Profiler
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller bcftools 2>>"$LOG_DIR/tbprofiler_run.log"

    if [ $? -eq 0 ] && [ -f "$result_json" ] && [ -s "$result_json" ]; then
        echo "$id" >> "$SUCCESS_LOG"
        n_done=$((n_done + 1))
    else
        echo "$id" >> "$FAIL_LOG"
        echo "FAILED: $id"
        n_failed=$((n_failed + 1))
    fi

    #progress report after every 50 samples
    if [ $((( n_done + n_skipped + n_failed ) % 50)) -eq 0 ]; then
        echo "Progress: $((n_done + n_skipped)) done / $total | Failed: $n_failed"
    fi
done

#collating all results into a summary table
echo ""
echo "Collating all results into summary table..."
tb-profiler collate --dir "$OUT_DIR" --prefix all_samples --outdir "$OUT_DIR" 2>/dev/null

echo ""
echo "TB-Profiler Complete: $(date)"
echo ""
echo "Total samples:          $total"
echo "Newly processed:        $n_done"
echo "Skipped (already done): $n_skipped"
echo "Failed:                 $n_failed"
echo ""

if [ -f "$OUT_DIR/all_samples.txt" ]; then
    echo "Summary table: $OUT_DIR/all_samples.txt"
    echo ""
    echo "Column headers:"
    head -1 "$OUT_DIR/all_samples.txt" | tr '\t' '\n' | nl
    echo ""
    echo "Total rows (samples) in summary:"
    tail -n +2 "$OUT_DIR/all_samples.txt" | wc -l
else
    echo "WARNING: Summary table not generated."
    echo "Try running manually: tb-profiler collate --dir $OUT_DIR --prefix all_samples --outdir $OUT_DIR"
fi

echo ""
if [ "$n_failed" -gt 0 ]; then
    echo "Failed samples logged to: $FAIL_LOG"
    echo "Re-run this script to retry — already-successful samples are skipped."
fi