#!/bin/bash
# Per-sample worker. Downloads one SRA accession, aligns to H37Rv, calls variants
#using pipe BWA directly into samtools sort (no SAM file will be written to disk)
#deleting .sra cache immediately after fasterq-dump
#deleting fastq immediately after alignment starts 
set -uo pipefail

ID="$1"
BASE="$HOME/tb_pipeline"
REF="$BASE/reference/H37Rv.fasta"

#ensuring the tb_amr env's tools are on PATH even when this script is launched as a subshell by GNU parallel
ENV_NAME="tb_amr"
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_NAME" 2>/dev/null || true

VCF_OUT_DIR="$BASE/vcf_output"
LOG_DIR="$BASE/logs"
SCRATCH="$BASE/scratch/$ID"

SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/failed.log"

FINAL_VCF="$VCF_OUT_DIR/${ID}.vcf.gz"
TMP_VCF="$VCF_OUT_DIR/.${ID}.vcf.gz.tmp"

#resume logic
if [ -f "$FINAL_VCF" ]; then
    echo "[$ID] SKIP - VCF already exists"
    exit 0
fi

#for cleaning up any stale scratch/tmp from a previous interrupted attempt on this ID
rm -rf "$SCRATCH"
rm -f "$TMP_VCF" "$TMP_VCF.csi"
mkdir -p "$SCRATCH"
cd "$SCRATCH" || exit 1

fail() {
    echo -e "${ID}\t$1" >> "$FAIL_LOG"
    echo "[$ID] FAILED - $1"
    cd "$BASE"
    rm -rf "$SCRATCH"
    rm -f "$TMP_VCF" "$TMP_VCF.csi"
    exit 1
}

#downloading the .SRA using prefetch and then converting that .SRA to fastq file using fasterq-dump
prefetch "$ID" -O "$SCRATCH" >/dev/null 2>"$SCRATCH/prefetch.err"
if [ $? -ne 0 ]; then
    fail "prefetch failed (bad accession or network issue)"
fi

fasterq-dump "$SCRATCH/$ID" -O "$SCRATCH" --split-3 -e 2 >/dev/null 2>"$SCRATCH/fasterq.err"
if [ $? -ne 0 ]; then
    fail "fasterq-dump failed"
fi

#deleting .sra cache immediately after conversion as it is no longer needed (saves storage space on my device)
rm -rf "$SCRATCH/$ID"
rm -rf "$HOME/ncbi/public/sra/${ID}.sra"* 2>/dev/null


#detecting layout from actual output files, whether it is paired or single 
R1="$SCRATCH/${ID}_1.fastq"
R2="$SCRATCH/${ID}_2.fastq"
SE="$SCRATCH/${ID}.fastq"

if [ -s "$R1" ] && [ -s "$R2" ]; then
    MODE="paired"
elif [ -s "$SE" ]; then
    MODE="single"
else
    fail "no usable fastq output produced"
fi

echo "[$ID] Mode detected: $MODE"

#Alignment: with BWA-MEM piped directly into samtools sort so that no SAM file is written to the drive and no secondary storage is consumed
# BWA writes to stdout -> samtools sort reads from stdin -> writes sorted BAM directly
BAM="$SCRATCH/${ID}.sorted.bam"

if [ "$MODE" = "paired" ]; then
    bwa mem -t 2 -R "@RG\tID:${ID}\tSM:${ID}" "$REF" "$R1" "$R2" 2>"$SCRATCH/bwa.err" \
        | samtools sort -@ 2 -o "$BAM" - 2>"$SCRATCH/sort.err"
else
    bwa mem -t 2 -R "@RG\tID:${ID}\tSM:${ID}" "$REF" "$SE" 2>"$SCRATCH/bwa.err" \
        | samtools sort -@ 2 -o "$BAM" - 2>"$SCRATCH/sort.err"
fi

if [ $? -ne 0 ] || [ ! -s "$BAM" ]; then
    fail "bwa mem / samtools sort failed"
fi

#deleting fastq files immediately after alignment
rm -f "$R1" "$R2" "$SE" 2>/dev/null

#indexing BAM file
samtools index "$BAM"

#variant calling
RAW_VCF="$SCRATCH/${ID}.raw.vcf.gz"
bcftools mpileup -f "$REF" "$BAM" 2>"$SCRATCH/mpileup.err" \
    | bcftools call -mv -Oz -o "$RAW_VCF" 2>"$SCRATCH/call.err"
if [ $? -ne 0 ] || [ ! -s "$RAW_VCF" ]; then
    fail "bcftools variant calling failed"
fi

bcftools index "$RAW_VCF"

#deleting BAM after variant calling
rm -f "$BAM" "$BAM.bai" 2>/dev/null

#atomic move into place
cp "$RAW_VCF" "$TMP_VCF"
cp "$RAW_VCF.csi" "$TMP_VCF.csi"
mv "$TMP_VCF" "$FINAL_VCF"
mv "$TMP_VCF.csi" "$FINAL_VCF.csi"

if [ -f "$FINAL_VCF" ]; then
    echo -e "${ID}\tOK" >> "$SUCCESS_LOG"
    echo "[$ID] SUCCESS - VCF saved"
else
    fail "final move failed"
fi

#deleting the scractch folder of the ID that has been processed into vcf
cd "$BASE"
rm -rf "$SCRATCH"

exit 0
