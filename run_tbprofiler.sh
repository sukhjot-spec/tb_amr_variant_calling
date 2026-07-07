#!/bin/bash
#run_tbprofiler.sh
#runs TB-Profiler on all filtered VCFs to generate:
#  - drug resistance predictions that will be the labels for the ML model
#  - lineage classifications
#  - compensatory mutation candidates
#automatic chromosome name fix (NC_000962.3 vs Chromosome mismatch)

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

#fixing chromosome mismatch
#TB-Profiler's bundled reference uses "Chromosome" as the chromosome name.
#but the processed vcfs use "NC_000962.3" because they were aligned to the NCBI H37Rv reference.
#this fix renames TB-Profiler's internal reference files to match our VCFs.
#this is a one-time fix as subsequent runs detect it's already done and skip.

TBPROFILER_DB="$HOME/miniforge3/envs/tb_amr/share/tbprofiler/who_v2+"

if [ ! -d "$TBPROFILER_DB" ]; then
    #finding the database directory if not at default path
    TBPROFILER_DB=$(find "$HOME/miniforge3" -name "genome.fasta" -path "*/tbprofiler/*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -z "$TBPROFILER_DB" ]; then
        echo "ERROR: Could not find TB-Profiler database directory."
        echo "Looked in: $HOME/miniforge3/envs/tb_amr/share/tbprofiler/"
        exit 1
    fi
fi

echo "TB-Profiler database: $TBPROFILER_DB"

#check for whether the fix is already applied
CURRENT_CHROM=$(head -1 "$TBPROFILER_DB/genome.fasta" 2>/dev/null | tr -d '>')

if [ "$CURRENT_CHROM" = "Chromosome" ]; then
    echo ""
    echo "Applying chromosome name fix (Chromosome → NC_000962.3)..."

    # Fix each reference file that contains chromosome names
    sed -i 's/^>Chromosome/>NC_000962.3/' "$TBPROFILER_DB/genome.fasta"
    sed -i 's/^Chromosome/NC_000962.3/' "$TBPROFILER_DB/genome.gff"
    sed -i 's/^Chromosome/NC_000962.3/' "$TBPROFILER_DB/genes.bed"
    sed -i 's/^Chromosome/NC_000962.3/' "$TBPROFILER_DB/mask.bed"
    sed -i 's/^Chromosome/NC_000962.3/' "$TBPROFILER_DB/barcode.bed"

    #verifying whether the fix was applied correctly
    echo "Verifying fix..."
    all_ok=true
    for f in genome.fasta genome.gff genes.bed mask.bed barcode.bed; do
        chrom=$(grep -m1 "NC_000962.3\|Chromosome" "$TBPROFILER_DB/$f" 2>/dev/null | head -c 30)
        if echo "$chrom" | grep -q "Chromosome"; then
            echo "  WARNING: $f still contains 'Chromosome' — fix may be incomplete"
            all_ok=false
        else
            echo "  OK: $f → $chrom"
        fi
    done

    if [ "$all_ok" = false ]; then
        echo "ERROR: Chromosome fix incomplete. Check the files manually."
        exit 1
    fi
    echo "Chromosome fix applied successfully."

elif [ "$CURRENT_CHROM" = "NC_000962.3" ]; then
    echo "Chromosome name already fixed (NC_000962.3) — skipping fix step."
else
    echo "WARNING: Unexpected chromosome name in reference: '$CURRENT_CHROM'"
fi

echo ""

#file steup
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
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller freebayes 2>/dev/null
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

    #running the tb-profiler
    tb-profiler profile --vcf "$vcf" --prefix "$id" --dir "$OUT_DIR" --no_trim --caller freebayes 2>>"$LOG_DIR/tbprofiler_run.log"


    if [ $? -eq 0 ] && [ -f "$result_json" ] && [ -s "$result_json" ]; then
        echo "$id" >> "$SUCCESS_LOG"
        n_done=$((n_done + 1))
    else
        echo "$id" >> "$FAIL_LOG"
        echo "FAILED: $id"
        n_failed=$((n_failed + 1))
    fi

    #progress report after every 50 samples
    total_processed=$((n_done + n_skipped))
    if [ $((( n_done + n_skipped + n_failed ) % 50)) -eq 0 ]; then
        echo "Progress: $((n_done + n_skipped)) done / $total | Failed: $n_failed"
    fi
done

#collating all results into one summary table
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