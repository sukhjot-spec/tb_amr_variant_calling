# M. tuberculosis WGS Variant Calling Pipeline
### Antimicrobial Resistance Prediction | African Clinical Isolates

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#3-dataset)
3. [Repository Structure](#4-repository-structure)
4. [Environment Setup](#5-environment-setup)
5. [Pipeline Architecture](#6-pipeline-architecture)
6. [Step-by-Step Pipeline Execution](#7-step-by-step-pipeline-execution)
7. [Tools and Software](#8-tools-and-software)
8. [Progress Status](#9-progress-status)
9. [Citation](#10-citation)

---

## 1. Project Overview

This repository contains the complete computational pipeline for whole-genome sequencing (WGS) based variant calling in *Mycobacterium tuberculosis* clinical isolates, developed as part of a research project

The pipeline processes raw sequencing data from NCBI's Sequence Read Archive (SRA), aligns reads to the standard H37Rv reference genome, and produces per-sample VCF (Variant Call Format) files. These variant files form the foundation for building a machine learning-ready feature matrix for antimicrobial resistance (AMR) prediction.

**Key characteristics:**
- Processes SRA accessions from African *M. tuberculosis* clinical isolates
- Fully resumable — safe to interrupt and restart at any point
- Disk-efficient design — peak scratch usage ~3–4 GB per sample
- Runs locally on WSL2 (Windows Subsystem for Linux 2) and in parallel on Google Colab

---

## 2. Dataset

### 2.1 Source

All sequencing data is sourced from NCBI's **Sequence Read Archive (SRA)**. The master metadata was obtained from NCBI SRA's public metadata, filtered specifically for African *M. tuberculosis* isolates with whole-genome sequencing (WGS) runs.

### 2.2 Metadata Filtering Process

The original NCBI SRA metadata for *M. tuberculosis* contained **19,000+ accession records**. The following filtering steps were applied to select a scientifically relevant subset:

1. **Organism filter** — Retained only *Mycobacterium tuberculosis* entries
2. **Library strategy filter** — Retained only `WGS` (Whole Genome Sequencing) runs
3. **Library source filter** — Retained only `GENOMIC` library sources
4. **Platform filter** — Retained only `ILLUMINA` platform runs (short-read, compatible with BWA-MEM alignment)
5. **Geographic filter** — Retained only runs with country/geographic origin metadata indicating African nations
6. **Run quality filter** — Removed runs with missing or ambiguous accession data

**Files in `data/`:**

| File | Description |
|---|---|
| `sra_master_metadata` | Full raw metadata downloaded from NCBI SRA for *M. tuberculosis* |
| `filtered_runs_with_country` | Metadata after filtering -includes country of origin, run accession, platform, layout, and library information |
| `african_accessions` | Final list of African accession IDs selected for processing |
| `SRR_Acc_List` | Complete SRA accession list used for batch file generation |
| `SRR_run` | Run-level accession records |

### 2.3 Dataset Summary

- **Total SRA records ran:** 3100
- **African isolates identified after filtering:** 1858
- **Accessions split into batches of 100** for pipeline processing
- **Accessions processed so far:** ~3,100 (across batch_000 to batch_030)
- **VCF files generated so far:** ~3,100 per-sample VCF files containing african and non african isolates
- **Reference genome:** H37Rv (*M. tuberculosis* H37Rv complete genome, NC_000962.3, 4,411,532 bp)

### 2.4 Geographic Coverage

Isolates span multiple African countries represented in the NCBI SRA, including countries across Sub-Saharan Africa, East Africa, West Africa, and South Africa. Exact country-level distribution is documented in `data/filtered_runs_with_country`.

---

## 3. Repository Structure

```
tb-amr-variant-calling/
│
├── batches/                        # Accession ID batch files (100 IDs each)
│   ├── batch_000                   # Plain text, one SRA/ERR accession per line
│   ├── batch_001
│   └── ...
│
├── colab_pipeline/
│   └──TB_variantCalling_pipeline.ipynb  # parallel pipeline running on google colab
│
├── data/                           # metadata and accession lists
│   ├── sra_master_metadata         # full raw NCBI SRA metadata
│   ├── filtered_runs_with_country  # filtered metadata with country information
│   ├── african_accessions          # final African accession ID list
│   ├── SRR_Acc_List                # SRR format accession list
│   └── SRR_run                     # run-level accession records
│
├── notebook/                       # analysis notebooks
│
├── py_scripts/                     # python utility scripts
│
├── install_tools.sh                # one-time environment setup script
├── run_batch.sh                    # main pipeline entry point
├── worker.sh                       # per-sample pipeline worker
├── check_results.sh                # progress monitoring script
├── filter_vcfs.sh                  # filters the vcfs based on QUAL and dp
└── README.md                      
```

**Not included in this repository (too large / available elsewhere):**
- `vcf_output/` — per-sample VCF files
- `reference/` — H37Rv reference genome (download from NCBI: NC_000962.3)
- `scratch/` — temporary pipeline working files (auto-cleaned after each sample)
- `logs/` — machine-specific run logs

---

## 4. Environment Setup

### 4.1 Requirements

- **Operating system:** Linux or WSL2 (Ubuntu 24, Windows 11)
- **Package manager:** Conda/Mamba (Miniforge recommended)
- **Conda environment:** `tb_amr`
- **Python:** 3.10

### 4.2 Installation

**Step 1 — Clone this repository:**
```bash
git clone https://github.com/sukhjot-spec/tb-amr-variant-calling.git
cd tb-amr-variant-calling
```

**Step 2 — Run the installation script:**
```bash
bash install_tools.sh
```

This script will:
- Verify the `tb_amr` conda environment exists
- Install all required bioinformatics tools into the `tb_amr` environment via Bioconda
- Create the required working directory structure under `~/tb_pipeline/`

Tools installed:
```
sra-tools==3.1.1    prefetch + fasterq-dump
bwa                 read alignment
samtools            BAM sorting and indexing
bcftools            variant calling and VCF manipulation
parallel            GNU Parallel for concurrent processing
```

**Step 3 — Download the H37Rv reference genome:**

The reference is downloaded automatically on the first pipeline run. To download manually:
```bash
mkdir -p ~/tb_pipeline/reference
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?\
db=nuccore&id=NC_000962.3&rettype=fasta&retmode=text" \
     -o ~/tb_pipeline/reference/H37Rv.fasta
```

**Step 4 — Place batch files:**
```bash
cp batches/batch_* ~/tb_pipeline/batches/
```

---

## 5. Pipeline Architecture

### 5.1 Overview

The pipeline follows a **reference-based variant calling** approach- the standard for *M. tuberculosis* WGS analysis. Each sample's reads are aligned to H37Rv and positions where the sample differs from reference are recorded as variants.

```
SRA Accession ID (e.g. SRR5181828)
        │
        ▼
  ┌───────────┐
  │  prefetch │   Downloads compressed .sra file from NCBI SRA
  └─────┬─────┘
        │ .sra file (2–4 GB)
        ▼
  ┌──────────────────┐
  │  fasterq-dump    │   Converts .sra to FASTQ, splits paired-end reads
  └────────┬─────────┘   .sra deleted immediately after this step
           │ _1.fastq + _2.fastq  (or .fastq for single-end)
           ▼
  ┌──────────────────────────────────────────────┐
  │  bwa mem  ──── pipe ────►  samtools sort     │
  │  (align to H37Rv)          (coordinate sort) │
  └──────────────────────┬───────────────────────┘
      FASTQs deleted ────┘        │ .sorted.bam
      immediately after           │ (SAM never written to disk)
      alignment starts            ▼
                          ┌──────────────────────────────────────────┐
                          │  bcftools mpileup ── pipe ──► bcftools call │
                          │  (per-position summary)   (variant calls) │
                          └──────────────────────────┬───────────────┘
                              BAM deleted ───────────┘  │ .vcf.gz + .csi
                              after calling             ▼
                                               vcf_output/<ID>.vcf.gz
```

### 5.2 Disk-Efficient Design

| Intermediate file | Initial approach | Present pipeline |
|---|---|---|
| SAM file | Written to disk (13–25 GB) | **Never written** — BWA piped directly to samtools |
| .sra cache | Kept until end | **Deleted immediately** after fasterq-dump |
| FASTQ files | Kept until end | **Deleted immediately** after alignment |
| BAM file | Kept until end | **Deleted immediately** after variant calling |
| **Peak scratch usage** | **~44 GB per sample** | **~3–4 GB per sample** |

### 5.3 Resumable Design

Before processing any sample, the pipeline checks whether a completed `.vcf.gz` already exists for that accession ID. If it does, the sample is skipped entirely — no network access, no computation. This means:

- Interruptions (power loss, laptop sleep, session timeout) never corrupt completed work
- Re-running `./run_batch.sh batch_000` automatically resumes from the last incomplete sample
- Completed samples are never reprocessed regardless of how many times the script is called

### 5.4 Parallelism

| Environment | Parallelism | Reasoning |
|---|---|---|
| Local WSL2 (i5-13450HX, 16 threads) | `-j 6` (6 concurrent samples) | 6 × 2 threads = 12 threads for BWA alignment, leaving 4 for OS/downloads |
| Google Colab (free tier, 2 vCPUs) | `-j 4` | Bottleneck is network bandwidth from NCBI, not CPU- confirmed by profiling |

Local and Colab runs process **different batch numbers simultaneously** — no overlap, no shared bottleneck, fully additive throughput.

---

## 6. Step-by-Step Pipeline Execution

### 6.1 Running a batch (local WSL2)

```bash
# Activate environment
mamba activate tb_amr

# Navigate to pipeline directory
cd ~/tb_pipeline

# Run a batch — processes 100 accession IDs, 6 at a time
./run_batch.sh batch_000

# Check results after the run
./check_results.sh batch_000
```

### 6.2 Monitoring progress

```bash
# Total VCFs generated
ls ~/tb_pipeline/vcf_output/*.vcf.gz | wc -l

# Successes and failures
cat ~/tb_pipeline/logs/success.log | wc -l
cat ~/tb_pipeline/logs/failed.log

# Currently running processes
ps aux | grep -E "prefetch|fasterq|bwa|samtools|bcftools" | grep -v grep

# Scratch disk usage
du -sh ~/tb_pipeline/scratch/
```

### 6.3 Resuming after interruption

```bash
# Re-run the same command — completed samples are automatically skipped
./run_batch.sh batch_000

# When batch_000 is complete, move to the next
./run_batch.sh batch_001
```

### 6.4 Verifying VCF output

```bash
# Inspect a completed VCF (header + first 19 variant rows)
zcat ~/tb_pipeline/vcf_output/SRR5181828.vcf.gz | grep -v "^##" | head -20

# Count total variants called in a sample
zcat ~/tb_pipeline/vcf_output/SRR5181828.vcf.gz | grep -vc "^#"

# Expected: 500–5,000 variants for a typical M. tb isolate vs H37Rv
```

---

## 7. Tools and Software

| Tool | Version | Role in pipeline |
|---|---|---|
| `prefetch` (SRA Toolkit) | 3.1.1 | Download compressed .sra files from NCBI SRA |
| `fasterq-dump` (SRA Toolkit) | 3.1.1 | Convert .sra to FASTQ format; split paired-end reads |
| `bwa mem` | 0.7.19 | Align short reads to H37Rv reference genome |
| `samtools sort` | 1.23.1 | Coordinate-sort aligned reads (required before variant calling) |
| `samtools index` | 1.23.1 | Build BAM index for random access |
| `bcftools mpileup` | 1.23.1 | Generate per-position read evidence summary |
| `bcftools call` | 1.23.1 | Statistical variant calling (multiallelic model) |
| `bcftools index` | 1.23.1 | Index VCF files for merging and random access |
| `GNU Parallel` | 20160622 | Run multiple samples concurrently |

**Reference genome:**
- H37Rv — *Mycobacterium tuberculosis* H37Rv complete genome
- NCBI accession: NC_000962.3
- Length: 4,411,532 bp, single chromosome

---

## 8. Progress Status

| Stage | Status | Notes |
|---|---|---|
| NCBI SRA metadata acquisition | ✅ Complete | 19,000+ records reviewed |
| African isolate filtering | ✅ Complete | ~2,773 accessions identified |
| Batch file generation | ✅ Complete | Split into batches of 100 |
| Reference genome download and indexing | ✅ Complete | H37Rv NC_000962.3 |
| Variant calling — local WSL2 pipeline | ✅ Complete | 15 batches were run successfuly |
| Variant calling — Google Colab pipeline | ✅ Complete | 16 batches were run successfuly |
| VCF quality filtering | ✅ Complete | filtered the vcfs based on QUAL and dp |
| VCF merging and feature matrix generation | 🔲 Pending | |
| ML model development | 🔲 Pending | |

---

## 9. Citation

This repository is part of an ongoing research project. Full citation details will be provided upon publication.

**If you use this pipeline, please cite the following tools:**

- **BWA:** Li, H. (2013). Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM. *arXiv:1303.3997*.
- **SAMtools / BCFtools:** Danecek, P. et al. (2021). Twelve years of SAMtools and BCFtools. *GigaScience*, 10(2), giab008.
- **GNU Parallel:** Tange, O. (2011). GNU Parallel — The Command-Line Power Tool. *;login: The USENIX Magazine*, 36(1), 42–47.
- **H37Rv reference genome:** Cole, S.T. et al. (1998). Deciphering the biology of *Mycobacterium tuberculosis* from the complete genome sequence. *Nature*, 393(6685), 537–544.

---