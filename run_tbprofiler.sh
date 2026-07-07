#!/bin/bash
#run_tbprofiler.sh
#runs TB-Profiler on all filtered VCFs to generate:
#  - drug resistance predictions that will be the labels for the ML model
#  - lineage classifications
#  - compensatory mutation candidates

set -uo pipefail

BASE="$HOME/tb_pipeline"
INPUT_DIR="$BASE/vcf_filtered"
OUT_DIR="$BASE/tbprofiler_results"
LOG_DIR="$BASE/logs"
SUCCESS_LOG="$LOG_DIR/tbprofiler_success.log"
FAIL_LOG="$LOG_DIR/tbprofiler_failed.log"

#activating the vritual environment 
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate tb_amr 2>/dev/null || true

#check whether the tb-profiler is installed or not 
if ! command -v tb-profiler &>/dev/null; then
    echo "TB-Profiler not found. Installing into tb_amr..."
    mamba install -y -n tb_amr -c bioconda tb-profiler
fi

echo "TB-Profiler version: $(tb-profiler version 2>/dev/null || echo 'unknown')"

#file steup
mkdir -p "$OUT_DIR" "$LOG_DIR"
> "$SUCCESS_LOG"
> "$FAIL_LOG"

#input file count
input_files=("$INPUT_DIR"/*.vcf.gz)
total=${#input_files[@]}

if [ "$total" -eq 0 ] || [ ! -f "${input_files[0]}" ]; then
    echo "ERROR: No .vcf.gz files found in $INPUT_DIR"
    echo "Run filter_vcfs.sh first."
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
echo "Running speed test on 3 samples..."
t_start=$(date +%s)
for vcf in "${input_files[@]:0:3}"; do
    id=$(basename "$vcf" .vcf.gz)
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller freebayes 2>/dev/null
done
t_end=$(date +%s)
t_elapsed=$((t_end - t_start))
rate=$(echo "scale=1; $t_elapsed / 3" | bc 2>/dev/null || echo "?")
estimated=$(echo "scale=0; ($t_elapsed * $total) / (3 * 60)" | bc 2>/dev/null || echo "?")
echo "Speed: ~${rate}s per sample | Estimated total: ~${estimated} minutes"
echo ""

#main loop
n_done=0
n_failed=0
n_skipped=0

for vcf in "${input_files[@]}"; do
    id=$(basename "$vcf" .vcf.gz)
    result_json="$OUT_DIR/results/${id}.results.json"

    #resume logic
    if [ -f "$result_json" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    #running the tb-profiler
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller freebayes 2>>"$LOG_DIR/tbprofiler_run.log"

    if [ $? -eq 0 ] && [ -f "$result_json" ]; then
        echo "$id" >> "$SUCCESS_LOG"
        n_done=$((n_done + 1))
    else
        echo "$id" >> "$FAIL_LOG"
        echo "FAILED: $id"
        n_failed=$((n_failed + 1))
    fi

    #progress report after every 50 samples
    total_processed=$((n_done + n_skipped))
    if [ $((total_processed % 50)) -eq 0 ]; then
        echo "Progress: $total_processed / $total | Failed: $n_failed | Skipped: $n_skipped"
    fi
done

#collating all results into one summary table
echo ""
echo "Collating all results into summary table..."
tb-profiler collate --dir "$OUT_DIR" --prefix all_samples --outdir "$OUT_DIR" 2>/dev/null

echo ""
echo "TB-Profiler Complete: $(date)"
echo ""
echo "Total samples:     $total"
echo "Newly processed:   $n_done"
echo "Skipped (done):    $n_skipped"
echo "Failed:            $n_failed"
echo ""
echo "Results:"
echo "  Per-sample JSON:  $OUT_DIR/results/"
echo "  Summary table:    $OUT_DIR/all_samples.txt"
echo ""

if [ -f "$OUT_DIR/all_samples.txt" ]; then
    echo "Column headers in summary table:"
    head -1 "$OUT_DIR/all_samples.txt" | tr '\t' '\n' | nl
    echo ""
    echo "Total rows in summary:"
    wc -l < "$OUT_DIR/all_samples.txt"
fi

if [ "$n_failed" -gt 0 ]; then
    echo "Failed samples logged to: $FAIL_LOG"
    echo "Re-run this script to retry failed samples."
fi