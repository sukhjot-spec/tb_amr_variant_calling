#!/bin/bash
# One-time setup script. for running inside WSL2 Ubuntu.
# Installs sra-tools, bwa, samtools, bcftools, parallel
set -euo pipefail

ENV_NAME="tb_amr"


echo "Installing pipeline tools into mamba env: $ENV_NAME"

#creating folder structure
mkdir -p ~/tb_pipeline/{batches,reference,vcf_output,logs,scratch}
echo "created ~/tb_pipeline folder structure."

#checking if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "ERROR: 'mamba' command not found."
    echo "check path"
    echo "or maybe run 'conda init bash'"
    exit 1
fi

#checking whether the tb_amr env exists 
if ! mamba env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "ERROR: mamba environment '$ENV_NAME' not found."
    echo "Available environments:"
    mamba env list
    exit 1
fi

echo "found environment '$ENV_NAME'. Installing tools into it..."
mamba install -y -n "$ENV_NAME" -c bioconda -c conda-forge sra-tools=3.1.1 bwa samtools bcftools parallel

echo ""
echo "verifying installation inside $ENV_NAME:"
mamba run -n "$ENV_NAME" which prefetch fasterq-dump bwa samtools bcftools parallel
mamba run -n "$ENV_NAME" fasterq-dump --version
mamba run -n "$ENV_NAME" bwa 2>&1 | head -3 || true
mamba run -n "$ENV_NAME" samtools --version | head -1
mamba run -n "$ENV_NAME" bcftools --version | head -1

echo ""
echo "setup complete"
