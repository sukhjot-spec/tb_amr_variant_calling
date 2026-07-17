# M. tuberculosis WGS Variant Calling Pipeline
### Comparative Genomic and Explainable Machine Learning Analysis of Compensatory Mutations Associated with Multidrug Resistance in African   Mycobacterium tuberculosis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Objectives](#2-research-objectives)
3. [Dataset](#3-dataset)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup](#5-environment-setup)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Step-by-Step Pipeline Execution](#7-step-by-step-pipeline-execution)
8. [Post-VCF Processing](#8-post-vcf-processing)
9. [Output Files and Their Role](#9-output-files-and-their-role)
10. [Tools and Software](#10-tools-and-software)
11. [Progress Status](#11-progress-status)
12. [Citation](#12-citation)

---

## 1. Project Overview

This repository contains the complete computational pipeline for whole-genome sequencing (WGS) based variant calling and antimicrobial resistance (AMR) analysis in *Mycobacterium tuberculosis* clinical isolates.

The pipeline processes raw sequencing data from NCBI's Sequence Read Archive (SRA), aligns reads to the H37Rv reference genome, calls variants, and produces a machine learning-ready feature matrix alongside drug resistance labels and candidate compensatory mutation annotations - enabling explainable ML-based AMR prediction focused on MDR-TB in African clinical isolates.

**Key characteristics:**
- Processes SRA accessions from confirmed African *M. tuberculosis* clinical isolates only
- Fully resumable - safe to interrupt and restart at any point without losing progress
- Disk-efficient design - peak scratch usage ~3–4 GB per sample (vs ~44 GB naive approach)
- Runs locally on WSL2 (Windows Subsystem for Linux 2) and in parallel on Google Colab
- Produces both genomic feature matrix and TB-Profiler resistance labels ready for ML

---

## 2. Research Objectives

**Objective 1** - Identify known antimicrobial resistance mutations and candidate compensatory mutations in multidrug-resistant African *M. tuberculosis* genomes using comparative genomic analysis.

**Objective 2** - Develop and evaluate an explainable machine learning model for prioritizing genomic features associated with multidrug resistance, including candidate compensatory mutations.

**Objective 3** - Characterize the evolutionary conservation, lineage distribution, and functional significance of candidate compensatory mutations using computational bioinformatics analyses.

---

## 3. Dataset

### 3.1 Source

All sequencing data is sourced from NCBI's **Sequence Read Archive (SRA)**. Master metadata was obtained from NCBI SRA's public database, filtered specifically for African *M. tuberculosis* isolates with Illumina WGS runs.

### 3.2 Metadata Filtering Process

The original NCBI SRA metadata for *M. tuberculosis* contained **19,000+ accession records**. The following sequential filters were applied:

1. **Organism filter** - Retained only *Mycobacterium tuberculosis* entries
2. **Library strategy filter** - Retained only `WGS` (Whole Genome Sequencing) runs
3. **Library source filter** - Retained only `GENOMIC` library sources
4. **Platform filter** - Retained only `ILLUMINA` platform runs (compatible with BWA-MEM)
5. **Geographic filter** - Retained only runs with African country of origin in metadata
6. **Run quality filter** - Removed runs with missing or ambiguous accession data

**Files in `data/`:**

| File | Description |
|---|---|
| `sra_master_metadata` | Full raw metadata from NCBI SRA for *M. tuberculosis* (19,000+ records) |
| `filtered_runs_with_country` | Metadata after filtering - country of origin, accession, platform, layout |
| `african_accessions` | Final list of 1,858 confirmed African accession IDs |
| `SRR_Acc_List` | Full SRA accession list used for batch file generation |
| `SRR_run` | Run-level accession records |

### 3.3 Dataset Summary

| Metric | Value |
|---|---|
| Total SRA records reviewed | 19,000+ |
| Accessions processed through variant calling | 3,100 (batch_000 – batch_030) |
| African isolates retained after geographic filtering | 1,858 |
| Geographic verification | 100% confirmed African origin via NCBI BioSample API |
| Reference genome | H37Rv (NC_000962.3), 4,411,532 bp, single chromosome |
| Quality filter applied | QUAL ≥ 20 AND DP ≥ 4 |
| Final variant features (after frequency filtering) | 94,583 |
| MDR samples (computed) | 871 / 1,857 (46.9%) |

### 3.4 Geographic Coverage

| Country | Isolates | Percentage |
|---|---|---|
| South Africa | 1,453 | 78.1% |
| Uganda (Kampala) | 241 | 12.9% |
| Ethiopia | 89 | 4.8% |
| Nigeria | 20 | 1.1% |
| Kenya | 19 | 1.0% |
| Other African | < 10 each | < 1% each |

Geographic origin was independently verified using `py_scripts/geo_check.py`, which queries NCBI BioSample API in batches of 500 for each sample.

### 3.5 TB-Profiler Resistance Profile

| Classification | Count | Percentage |
|---|---|---|
| Susceptible | 711 | 38.3% |
| Pre-XDR-TB | 497 | 26.7% |
| MDR-TB | 235 | 12.6% |
| XDR-TB | 160 | 8.6% |
| RR-TB | 115 | 6.2% |
| HR-TB | 85 | 4.6% |
| Other | 55 | 3.0% |

**Computed resistance flags:** MDR: 871 (46.9%) · Pre-XDR: 641 (34.5%) · XDR: 149 (8.0%)

**Lineage distribution:** Lineage 4: 68.1% · Lineage 2: 19.4% · Lineage 3: 6.3% · Lineage 1: 3.4%

---

## 4. Repository Structure

tb-amr-variant-calling/
│
├── batches/                         # Accession ID batch files (100 IDs each)
│   ├── batch_000                    # Plain text, one SRA/ERR accession per line
│   ├── batch_001
│   └── ...
│
├── colab_pipeline/
│   └── TB_variantCalling_pipeline.ipynb  # Parallel Colab pipeline notebook
│
├── data/                            # Metadata and accession lists
│   ├── sra_master_metadata          # Full raw NCBI SRA metadata
│   ├── filtered_runs_with_country   # Filtered metadata with country information
│   ├── african_accessions           # Final 1,858 African accession ID list
│   ├── SRR_Acc_List                 # SRR format accession list
│   └── SRR_run                      # Run-level accession records
│
├── py_scripts/                      # Python utility scripts
│   ├── geo_check.py                 # NCBI BioSample API geographic verification
│   ├── collate.py                   # Custom TB-Profiler JSON collation script
│   └── build_ml_dataset.py          # ML dataset construction from feature matrix + labels
|
├── ml_outputs/                      # All the output data files required for ML process (large files were not pushed to github)
|   ├── y_labels.csv
|   ├── variant_metadata_filtered.csv
|   └── ...
│
├── notebook/                        
│   ├── analysis.ipynb
|
├── tbprofiler_results/
|   ├── compensatory.csv
|   ├── dr_variants.csv
|   ├── labels.csv
|   ├── rpoB_nonRRDR.csv
|   └── summary_stats.txt
|
├── install_tools.sh                 # One-time environment setup
├── run_batch.sh                     # Main pipeline entry point
├── worker.sh                        # Per-sample variant calling worker
├── check_results.sh                 # Progress monitoring
├── filter_vcfs.sh                   # Quality-based VCF filtering (QUAL≥20, DP≥4)
├── run_tbprofiler.sh                # TB-Profiler batch runner with chromosome fix
├── merge_vcfs.sh                    # VCF merger + feature matrix construction
├── geo_check.py
├── .gitignore
└── README.md

**Not included in this repository (large data files):**
- `vcf_output/` - raw per-sample VCF files (~3,100 files)
- `vcf_filtered/` - quality-filtered VCF files (1,858 files)
- `tbprofiler_results/` - TB-Profiler JSON results and collated CSVs
- `reference/` - H37Rv reference genome (download from NCBI: NC_000962.3)
- `scratch/` - temporary pipeline working files (auto-cleaned after each sample)
- `logs/` - machine-specific run logs

---

## 5. Environment Setup

### 5.1 Requirements

- **Operating system:** Linux or WSL2 (Ubuntu 24, Windows 11)
- **Package manager:** Conda/Mamba (Miniforge recommended)
- **Conda environment:** `tb_amr`
- **Python:** 3.10+

### 5.2 Installation

**Step 1 - Clone this repository:**
```bash
git clone https://github.com/sukhjot-spec/tb-amr-variant-calling.git
cd tb_amr_variant_calling
```

**Step 2 - Run the installation script:**
```bash
bash install_tools.sh
```

This script verifies the `tb_amr` conda environment, installs all required bioinformatics tools via Bioconda, and creates the working directory structure under `~/tb_pipeline/`.

Tools installed:
sra-tools==3.1.1    prefetch + fasterq-dump
bwa                 read alignment
samtools            BAM sorting and indexing
bcftools            variant calling and VCF manipulation
parallel            GNU Parallel for concurrent processing
tb-profiler         drug resistance prediction (WHO v2+ database)

**Step 3 - Download the H37Rv reference genome:**
```bash
mkdir -p ~/tb_pipeline/reference
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?\
db=nuccore&id=NC_000962.3&rettype=fasta&retmode=text" \
     -o ~/tb_pipeline/reference/H37Rv.fasta
```

**Step 4 - Place batch files:**
```bash
cp batches/batch_* ~/tb_pipeline/batches/
```

---

## 6. Pipeline Architecture

### 6.1 Overview

The pipeline follows a **reference-based variant calling** approach - the standard for *M. tuberculosis* WGS analysis. Each sample's reads are aligned to H37Rv and positions where the sample differs from the reference are recorded as variants.

SRA Accession ID (e.g. SRR5181828)
│
▼
┌───────────┐
│  prefetch │   Downloads compressed .sra file from NCBI SRA
└─────┬─────┘
│ .sra file (2–4 GB) - deleted immediately after next step
▼
┌──────────────────┐
│  fasterq-dump    │   Converts .sra to FASTQ, splits paired-end reads
└────────┬─────────┘
│ _1.fastq + _2.fastq (or .fastq for single-end)
▼
┌──────────────────────────────────────────────┐
│  bwa mem  ──── pipe ────►  samtools sort     │
│  (align to H37Rv)          (coordinate sort) │
└──────────────────────┬───────────────────────┘
FASTQs deleted ────┘        │ .sorted.bam
immediately after           │ (SAM never written to disk)
alignment                   ▼
┌──────────────────────────────────────────────┐
│  bcftools mpileup ── pipe ──► bcftools call  │
│  (per-position read summary)  (variant calls)│
└──────────────────────────┬───────────────────┘
BAM deleted ───────────┘  │ .vcf.gz + .csi
after calling             ▼
vcf_output/<ID>.vcf.gz

### 6.2 Disk-Efficient Design

| Intermediate file | Naive approach | This pipeline |
|---|---|---|
| SAM file | Written to disk (13–25 GB) | **Never written** - BWA piped directly to samtools |
| .sra cache | Kept until end | **Deleted immediately** after fasterq-dump |
| FASTQ files | Kept until end | **Deleted immediately** after alignment |
| BAM file | Kept until end | **Deleted immediately** after variant calling |
| **Peak scratch per sample** | **~44 GB** | **~3–4 GB** |

### 6.3 Resumable Design

Before processing any sample, the pipeline checks whether a completed `.vcf.gz` already exists. If it does, the sample is skipped - no network access, no computation. Safe to interrupt and restart at any point.

### 6.4 Parallelism

| Environment | Jobs | Reasoning |
|---|---|---|
| Local WSL2 (i5-13450HX, 16 threads) | `-j 6` | 6 × 2 BWA threads = 12 threads used for alignment |
| Google Colab (free tier, 2 vCPUs) | `-j 4` | Bottleneck is NCBI download bandwidth, not CPU |

Local and Colab runs processed different batch numbers simultaneously - no overlap, fully additive throughput.

---

## 7. Step-by-Step Pipeline Execution

### 7.1 Running a batch (local WSL2)

```bash
mamba activate tb_amr
cd ~/tb_pipeline
./run_batch.sh batch_000
./check_results.sh batch_000
```

### 7.2 Monitoring progress

```bash
ls ~/tb_pipeline/vcf_output/*.vcf.gz | wc -l   # total VCFs completed
cat ~/tb_pipeline/logs/success.log | wc -l       # logged successes
cat ~/tb_pipeline/logs/failed.log                # failures with reasons
ps aux | grep -E "prefetch|fasterq|bwa|samtools|bcftools" | grep -v grep
du -sh ~/tb_pipeline/scratch/                    # current scratch usage
```

### 7.3 Quality filtering (QUAL≥20, DP≥4)

After variant calling, all VCFs are quality-filtered. Original VCFs are never modified - filtered copies go to `vcf_filtered/`:

```bash
mamba activate tb_amr
bash ~/tb_pipeline/filter_vcfs.sh
```

This removes variants where QUAL < 20 (less than 99% confidence) OR DP < 4 (fewer than 4 supporting reads). Typical effect: ~17% of raw variants removed per sample.

### 7.4 Geographic verification

After collecting 3,100 VCFs, African origin was verified and non-African isolates removed:

```bash
cd ~/tb_pipeline
python3 py_scripts/geo_check.py
# Queries NCBI BioSample API in batches of 500
# Output: data/filtered_runs_with_country, data/african_accessions
# Result: 1,858 of 3,100 VCFs confirmed African origin
```

---

## 8. Post-VCF Processing

### 8.1 Pre-processing - Chromosome Renaming

Before merging or running TB-Profiler, all 1,858 filtered VCFs were renamed from the NCBI chromosome name (`NC_000962.3`) to the TB-Profiler database chromosome name (`Chromosome`):

```bash
echo -e "NC_000962.3\tChromosome" > ~/tb_pipeline/reference/chr_map.txt
cd ~/tb_pipeline/vcf_filtered
for vcf in *.vcf.gz; do
    bcftools annotate --rename-chrs ~/tb_pipeline/reference/chr_map.txt \
        -Oz -o "${vcf}.tmp" "$vcf" && mv "${vcf}.tmp" "$vcf"
    bcftools index -f "$vcf"
done
```

### 8.2 Stream 1 - VCF Merging and Feature Matrix (merge_vcfs.sh)

Merges all 1,858 filtered VCFs into a single multi-sample variant matrix in five steps:

```bash
mamba activate tb_amr
bash ~/tb_pipeline/merge_vcfs.sh
```

**Step 1** - Build sorted VCF file list (`vcf_list.txt`)

**Step 2** - Parallel CSI index verification (8 threads via `xargs -P 8`)

**Step 3** - Merge + simultaneous frequency filtering (single-pass pipe):
```bash
bcftools merge --file-list vcf_list.txt --missing-to-ref --force-samples \
    --output-type u --threads 8 \
| bcftools view --min-af 0.01:alt1 --max-af 0.99:alt1 \
    --output-type z --output merged_prefilt.vcf.gz --threads 8
```
- Raw merged positions: 831,000+
- After 1%–99% allele frequency filter: **94,583 variants retained**
- `--missing-to-ref`: samples with no call at a position are assigned reference genotype (0/0) - produces a complete matrix with no missing values

**Step 4** - Extract genotype matrix via `bcftools query` → `gt_matrix_tsv.gz`
(format: `CHROM_POS_REF_ALT[\t%GT]` per variant, bgzip compressed)

**Step 5** - Python chunked binary conversion (50,000 variants/chunk):
- Genotype encoding: `0/0 → 0`, `1/1 or 0/1 → 1`, missing → 0
- Output: `feature_matrix.npz` (dense NumPy, shape 1858 × 94583) and `variant_matrix_filtered.csv`

### 8.3 Stream 2 - TB-Profiler Resistance Prediction (run_tbprofiler.sh)

Runs TB-Profiler on all 1,858 filtered VCFs to generate drug resistance labels:

```bash
mamba activate tb_amr
bash ~/tb_pipeline/run_tbprofiler.sh
```

- Tool: TB-Profiler with WHO v2+ resistance catalogue
- Mode: `--vcf` (uses pre-called VCF, does not re-call variants)
- Chromosome fix included automatically in the script
- Produces: one JSON result file per sample in `tbprofiler_results/results/`
- The built-in `tb-profiler collate` failed → custom `collate.py` used instead

### 8.4 Custom Collation (collate_tbprofiler.py)

Parses all 1,858 JSON files and produces five structured CSV files:

```bash
python3 ~/tb_pipeline/py_scripts/collate.py
```

Key design decisions:
- **rpoB RRDR separation**: codons 426–452 are primary resistance (→ `dr_variants.csv`); non-RRDR rpoB mutations in MDR samples are compensatory candidates (→ `rpoB_nonRRDR.csv`)
- **Compensatory gene validation**: only scientifically validated genes included (rpoA, rpoC, rpsA, rpsL, gyrB, tlyA, mmpL5, Rv0678) with mechanism, evidence, and `requires_MDR_context` flags
- **Excluded from compensatory set**: rplC, rplD, tlyA, fgd1, dprE1, pepQ (primary resistance genes, not fitness compensators)

### 8.5 ML Dataset Construction (build_ml_dataset.py)

Joins the feature matrix with labels and builds all ML-ready arrays:

```bash
python3 ~/tb_pipeline/py_scripts/build_ml_dataset.py
```

- Fixes malformed variant IDs (27,494 multi-allelic comma-containing IDs → first ALT kept)
- Inner join: 1,857 samples (SRR11922476 excluded - in labels but absent from feature matrix)
- Post-alignment MAF re-filter: removes zero-variance features after sample subsetting
- Builds Objective 1 enrichment tables for compensatory mutation analysis

---

## 9. Output Files and Their Role

### 9.1 From merge_vcfs.sh

| File | Dimensions | Description |
|---|---|---|
| `merged_prefilt_vcf.gz` | 1,858 × 94,583 | Frequency-filtered multi-sample merged VCF. Source of truth for all matrix files. |
| `gt_matrix_tsv.gz` | 94,583 rows × 1,859 cols | Raw genotype TSV from bcftools query. Preserves 0/0, 0/1, 1/1 notation before binary encoding. |
| `feature_matrix.npz` | (1858, 94583) uint8 | Compressed NumPy archive: `matrix`, `samples`, `variants` arrays. Fast loading for ML. |
| `variant_matrix_filtered.csv` | 1,858 × 94,584 | CSV feature matrix. sample_id + 94,583 binary variant columns (0=ref, 1=alt). |
| `variant_metadata.csv` | 94,583 rows × 5 cols | variant_id, CHROM, POS, REF, ALT. Required for annotating SHAP feature importances with gene names. |

### 9.2 From collate.py

| File | Records | Description |
|---|---|---|
| `labels.csv` | 1,858 × 49 cols | Per-sample: drtype, MDR/pre_XDR/XDR flags, has_rpoB_RRDR, main_lineage, per-drug R/S + binary encodings |
| `dr_variants.csv` | 14,734 records | All primary drug resistance variants. Columns: gene, change, drug, confidence, is_RRDR, sample_MDR |
| `compensatory.csv` | 24,718 records | Candidate compensatory variants with mechanism, drug_context, evidence, requires_MDR_context fields |
| `rpoB_nonRRDR.csv` | 1,335 records | rpoB non-RRDR variants in MDR+RRDR samples - intra-gene compensatory candidates |
| `other_variants.csv` | All remaining | Non-resistance, non-compensatory variants for exploratory analysis |
| `summary_stats.txt` | Text | Dataset-level statistics for Methods section reporting |

### 9.3 From build_ml_dataset.py (ml_outputs/)

| File | Dimensions | Used for |
|---|---|---|
| `X_array.npy` | (1857, 94583) uint8 | Primary feature matrix for all ML training |
| `y_mdr_array.npy` | (1857,) int8 | MDR binary target - 871 positive, 986 negative |
| `y_pre_xdr_array.npy` | (1857,) int8 | pre-XDR binary target - 641 positive |
| `y_xdr_array.npy` | (1857,) int8 | XDR binary target - 149 positive |
| `y_labels.csv` | 1,857 × 49 cols | Full aligned label file - all resistance and lineage columns |
| `sample_ids.txt` | 1,857 lines | Sample IDs in row order matching X_array |
| `feature_names_clean.txt` | 94,583 lines | Variant IDs in column order matching X_array - for SHAP annotation |
| `variant_metadata_filtered.csv` | 94,583 rows | Clean variant metadata aligned to final feature set |
| `excluded_samples.csv` | 1 row | SRR11922476 - in labels but missing from feature matrix |
| `obj1_mdr_vs_susceptible_comp.csv` | Per mutation | Enrichment table: MDR vs non-MDR frequencies for each compensatory mutation. Input for Fisher's exact test (Step 2) |
| `obj1_compensatory_in_MDR_samples.csv` | MDR records | Compensatory records in MDR context only |
| `obj1_compensatory_in_nonMDR_samples.csv` | non-MDR records | Compensatory records in non-MDR samples for comparison |
| `obj1_rpoB_nonRRDR_summary.csv` | Non-RRDR records | rpoB intra-gene compensatory candidates stratified by MDR status |

---

## 10. Tools and Software

| Tool | Version | Role |
|---|---|---|
| `prefetch` (SRA Toolkit) | 3.1.1 | Download .sra files from NCBI SRA |
| `fasterq-dump` (SRA Toolkit) | 3.1.1 | Convert .sra to FASTQ; split paired-end reads |
| `bwa mem` | 0.7.19 | Align reads to H37Rv reference genome |
| `samtools sort` | 1.23.1 | Coordinate-sort aligned reads |
| `samtools index` | 1.23.1 | BAM index for random access |
| `bcftools mpileup` | 1.23.1 | Per-position read evidence summary |
| `bcftools call` | 1.23.1 | Statistical variant calling (multiallelic model) |
| `bcftools merge` | 1.23.1 | Multi-sample VCF merging |
| `bcftools filter` | 1.23.1 | Quality-based variant filtering |
| `bcftools annotate` | 1.23.1 | Chromosome name renaming |
| `bcftools query` | 1.23.1 | Genotype matrix extraction |
| `bgzip` | htslib 1.23.1 | Genotype TSV compression |
| `GNU Parallel` | 20160622 | Concurrent multi-sample processing |
| `TB-Profiler` | Latest (WHO v2+) | Drug resistance prediction and lineage classification |
| `NumPy` | 1.26+ | Chunked binary matrix conversion and NPZ storage |
| `pandas` | 2.x+ | CSV handling in collation and ML dataset steps |

**Reference genome:** H37Rv - *Mycobacterium tuberculosis* H37Rv complete genome - NC_000962.3 - 4,411,532 bp

---

## 11. Progress Status

| Stage | Status | Details |
|---|---|---|
| NCBI SRA metadata acquisition | ✅ Complete | 19,000+ records reviewed |
| African isolate filtering | ✅ Complete | 1,858 accessions identified |
| Batch file generation | ✅ Complete | Split into batches of 100 |
| Reference genome download and indexing | ✅ Complete | H37Rv NC_000962.3 |
| Variant calling - local WSL2 pipeline | ✅ Complete | 15 batches processed |
| Variant calling - Google Colab pipeline | ✅ Complete | 16 batches processed |
| Geographic verification (geo_check.py) | ✅ Complete | 1,858 / 3,100 confirmed African - 100% verified |
| VCF quality filtering (QUAL≥20, DP≥4) | ✅ Complete | 1,858 African isolates filtered |
| Chromosome renaming (NC_000962.3 → Chromosome) | ✅ Complete | Applied to all 1,858 VCFs before merge and TB-Profiler |
| VCF merging and feature matrix (merge_vcfs.sh) | ✅ Complete | 94,583 variants · feature_matrix.npz · variant_matrix_filtered.csv |
| TB-Profiler resistance prediction (run_tbprofiler.sh) | ✅ Complete | 1,858 samples · WHO v2+ database · 0 failures |
| Custom collation (collate.py) | ✅ Complete | labels.csv · dr_variants.csv · compensatory.csv · rpoB_nonRRDR.csv |
| ML dataset construction (build_ml_dataset.py) | ✅ Complete | 1,857 samples × 94,583 features · MDR: 871 (46.9%) |
| Comparative genomic analysis - Objective 1 | 🔲 Pending | Fisher's exact test on compensatory mutation enrichment |
| ML model training - Objective 2 | 🔲 Pending | XGBoost + EBM (GlassBox) + SHAP |
| Lineage distribution analysis - Objective 3 | 🔲 Pending | Chi-squared lineage stratification |
| Functional annotation - Objective 3 | 🔲 Pending | UniProt / literature annotation |

---

## 12. Citation

This repository is part of an ongoing research project. Full citation details will be provided upon publication.

**Please cite the following tools if you use this pipeline:**

- **BWA:** Li, H. (2013). Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM. *arXiv:1303.3997*.
- **SAMtools / BCFtools:** Danecek, P. et al. (2021). Twelve years of SAMtools and BCFtools. *GigaScience*, 10(2), giab008.
- **GNU Parallel:** Tange, O. (2011). GNU Parallel - The Command-Line Power Tool. *;login: The USENIX Magazine*, 36(1), 42–47.
- **TB-Profiler:** Phelan, J. et al. (2019). Integrating informatics tools and portable sequencing technology for rapid detection of resistance to anti-tuberculous drugs. *Genome Medicine*, 11:41.
- **WHO Resistance Catalogue:** WHO (2023). Catalogue of mutations in *M. tuberculosis* complex and their association with drug resistance.
- **H37Rv reference genome:** Cole, S.T. et al. (1998). Deciphering the biology of *Mycobacterium tuberculosis* from the complete genome sequence. *Nature*, 393(6685), 537–544.
