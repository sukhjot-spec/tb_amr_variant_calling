#!/bin/bash
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <batch_name>   e.g. $0 batch_000"
    exit 1
fi

BATCH_NAME="$1"
BASE="$HOME/tb_pipeline"
BATCH_FILE="$BASE/batches/$BATCH_NAME"
SUCCESS_LOG="$BASE/logs/success.log"
FAIL_LOG="$BASE/logs/failed.log"

if [ ! -f "$BATCH_FILE" ]; then
    echo "ERROR: $BATCH_FILE not found."
    exit 1
fi

mapfile -t ids < "$BATCH_FILE"
total=${#ids[@]}

n_success=0
n_fail=0
n_pending=0

echo "================================================================"
echo "Results for $BATCH_NAME"
echo "================================================================"

declare -A fail_reasons

for id in "${ids[@]}"; do
    [ -z "$id" ] && continue
    if [ -f "$BASE/vcf_output/${id}.vcf.gz" ]; then
        n_success=$((n_success+1))
    elif [ -f "$FAIL_LOG" ] && grep -q "^${id}	" "$FAIL_LOG"; then
        n_fail=$((n_fail+1))
        reason=$(grep "^${id}	" "$FAIL_LOG" | tail -1 | cut -f2)
        fail_reasons["$id"]="$reason"
    else
        n_pending=$((n_pending+1))
    fi
done

echo "Total IDs in batch:  $total"
echo "Succeeded (VCF exists): $n_success"
echo "Failed:                 $n_fail"
echo "Not yet attempted:      $n_pending"
echo ""

if [ $n_fail -gt 0 ]; then
    echo "Failed IDs and reasons:"
    for id in "${!fail_reasons[@]}"; do
        echo "  $id: ${fail_reasons[$id]}"
    done
    echo ""
    echo "re-run the batch if any ID failed"
fi

if [ $n_pending -gt 0 ]; then
    echo "$n_pending samples haven't been attempted yet"
fi

total_vcfs=$(ls "$BASE/vcf_output"/*.vcf.gz 2>/dev/null | wc -l)
echo ""
echo "Total VCFs across ALL batches so far: $total_vcfs"
