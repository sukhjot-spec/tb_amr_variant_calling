#!/bin/bash
# Runs the full pipeline for one batch file and skips those samples that have already been ran 
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <batch_name>   e.g. $0 batch_000"
    exit 1
fi

BATCH_NAME="$1"
PARALLEL_JOBS=6 

ENV_NAME="tb_amr"
BASE="$HOME/tb_pipeline"
BATCH_FILE="$BASE/batches/$BATCH_NAME"
REF="$BASE/reference/H37Rv.fasta"
LOG_DIR="$BASE/logs"

#activate the tb_amr mamba environment so that the required tools are on the PATH
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "ERROR: could not activate mamba environment '$ENV_NAME'."
    echo "Make sure you've run install_tools.sh first, and that '$ENV_NAME' exists"
    exit 1
fi
echo "Activated environment: $ENV_NAME"

if [ ! -f "$BATCH_FILE" ]; then
    echo "ERROR: $BATCH_FILE not found."
    exit 1
fi

mkdir -p "$BASE/reference" "$BASE/vcf_output" "$LOG_DIR" "$BASE/scratch"


#downloading the reference genome, indexing it to use for alignment and variant calling
if [ ! -f "$REF" ]; then
    echo "Downloading H37Rv reference (NC_000962.3)..."
    curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000962.3&rettype=fasta&retmode=text" -o "$REF"
    if [ ! -s "$REF" ]; then
        echo "ERROR: reference download failed"
        exit 1
    fi
fi

if [ ! -f "$REF.bwt" ]; then
    echo "Building BWA index..."
    bwa index "$REF"
fi

if [ ! -f "$REF.fai" ]; then
    samtools faidx "$REF"
fi

#in case any run crashed: cleaning any leftover scratch dirs from a previous crashed run
find "$BASE/scratch" -mindepth 1 -maxdepth 1 -type d -mmin +120 -exec rm -rf {} \; 2>/dev/null || true

#load IDs
mapfile -t ids < "$BATCH_FILE"
ids=("${ids[@]}")
total=${#ids[@]}

already_done=0
for id in "${ids[@]}"; do
    [ -z "$id" ] && continue
    [ -f "$BASE/vcf_output/${id}.vcf.gz" ] && already_done=$((already_done+1))
done

echo "================================================================"
echo "Batch: $BATCH_NAME"
echo "Total IDs: $total"
echo "Already completed (will be skipped): $already_done"
echo "Remaining: $((total - already_done))"
echo "Running $PARALLEL_JOBS at a time..."
echo "================================================================"

chmod +x "$BASE/worker.sh" 2>/dev/null || true

cat "$BATCH_FILE" | parallel -j "$PARALLEL_JOBS" --joblog "$LOG_DIR/${BATCH_NAME}_joblog.tsv" "$BASE/worker.sh" {}

echo "================================================================"
echo "Batch $BATCH_NAME run finished."
echo "Run ./check_results.sh $BATCH_NAME to see what succeeded/failed."
echo "================================================================"
