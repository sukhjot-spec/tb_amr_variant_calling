# M. tuberculosis WGS Variant Calling & Compensatory Mutation Analysis Pipeline
### Comparative Genomic and Explainable Machine Learning Analysis of Compensatory Mutations Associated with Multidrug Resistance in African Mycobacterium tuberculosis

**Project status: COMPLETE.** All three research objectives, all eleven analytical steps, and a final publication-figures step have been executed, verified against real data at every stage, and (where problems were found along the way) fixed and re-verified. This README documents the full pipeline from raw sequencing reads through to the final, evidence-integrated candidate table and publication figures.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Objectives](#2-research-objectives)
3. [Dataset](#3-dataset)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup](#5-environment-setup)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Step-by-Step Pipeline Execution](#7-step-by-step-pipeline-execution)
8. [Post-VCF Processing (Steps 1-3, Objective 1)](#8-post-vcf-processing-steps-1-3-objective-1)
9. [Explainable Machine Learning (Steps 4-7, Objective 2)](#9-explainable-machine-learning-steps-4-7-objective-2)
10. [Evolutionary, Lineage, and Structural Analysis (Steps 8-10, Objective 3)](#10-evolutionary-lineage-and-structural-analysis-steps-8-10-objective-3)
11. [Final Integration and Publication Figures (Steps 11-12)](#11-final-integration-and-publication-figures-steps-11-12)
12. [Output Files and Their Role](#12-output-files-and-their-role)
13. [Tools and Software](#13-tools-and-software)
14. [Key Findings Summary](#14-key-findings-summary)
15. [Known Issues Found and Fixed During This Project](#15-known-issues-found-and-fixed-during-this-project)
16. [Progress Status](#16-progress-status)
17. [Citation](#17-citation)

---

## 1. Project Overview

This repository contains the complete computational pipeline for whole-genome sequencing (WGS) based variant calling, antimicrobial resistance (AMR) analysis, explainable machine learning, and multi-layer evolutionary/structural characterisation of candidate compensatory mutations in *Mycobacterium tuberculosis* clinical isolates.

The pipeline processes raw sequencing data from NCBI's Sequence Read Archive (SRA), aligns reads to the H37Rv reference genome, calls variants, identifies statistically significant compensatory-mutation candidates, trains and explains four machine learning models for MDR prediction, and characterises the surviving candidates' lineage distribution, evolutionary conservation, and protein-structural context - integrating every one of these evidence layers into a single, evidence-tiered master table.

**Key characteristics:**
- Processes SRA accessions from confirmed African *M. tuberculosis* clinical isolates only
- Fully resumable - safe to interrupt and restart at any point without losing progress
- Disk-efficient design - peak scratch usage ~3-4 GB per sample (vs ~44 GB naive approach)
- Runs locally on WSL2 (Windows Subsystem for Linux 2) and in parallel on Google Colab
- Combines statistical, machine-learning, phylogenetic, and structural-biology evidence for the same candidate set, integrated into one final table rather than left as separate, disconnected results
- Every fix and every reported number in this pipeline was verified directly against real data before being trusted, not assumed correct from code review alone (see Section 15)

---

## 2. Research Objectives

**Objective 1** - Identify known antimicrobial resistance mutations and candidate compensatory mutations in multidrug-resistant African *M. tuberculosis* genomes using comparative genomic analysis. *(Steps 1-3. Complete.)*

**Objective 2** - Develop and evaluate an explainable machine learning model for prioritising genomic features associated with multidrug resistance, including candidate compensatory mutations. *(Steps 4-7. Complete.)*

**Objective 3** - Characterise the evolutionary conservation, lineage distribution, and functional/structural significance of candidate compensatory mutations using computational bioinformatics analyses. *(Steps 8-10. Complete.)*

**Final integration** - Combine every evidence layer above into a single master table with an evidence tier per candidate, and produce a curated set of publication-ready figures. *(Steps 11-12. Complete.)*

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
| Accessions processed through variant calling | 3,100 (batch_000 - batch_030) |
| African isolates retained after geographic filtering | 1,858 |
| Geographic verification | 100% confirmed African origin via NCBI BioSample API |
| Reference genome | H37Rv (NC_000962.3), 4,411,532 bp, single chromosome |
| Quality filter applied | QUAL >= 20 AND DP >= 4 |
| Final variant features (after frequency filtering) | 94,583 |
| **Final ML-ready cohort** | **1,858 samples - the full cohort, zero exclusions** (see Section 15.1 for a row-shift bug found and fixed early in this project that had previously, incorrectly excluded one sample) |
| MDR samples (final, corrected) | 871 / 1,858 (46.9%) |
| Non-MDR samples (final, corrected) | 987 / 1,858 (53.1%) |

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

**Computed resistance flags:** MDR: 871 (46.9%) - Pre-XDR: 641 (34.5%) - XDR: 149 (8.0%)

**Lineage distribution:** Lineage 4: 68.1% - Lineage 2: 19.4% - Lineage 3: 6.3% - Lineage 1: 3.4% - remaining ~2.6% either "Unknown" or a compound multi-lineage call (see Section 15 for how this small residual group is handled in lineage-stratified analyses).

---

## 4. Repository Structure

```
tb-amr-variant-calling/
|
├── batches/                              # Accession ID batch files (100 IDs each)
│   ├── batch_000
│   └── ...
|
├── colab_pipeline/
│   └── TB_variantCalling_pipeline.ipynb  # Parallel Colab pipeline notebook
|
├── data/                                 # Metadata and accession lists
│   ├── sra_master_metadata
│   ├── filtered_runs_with_country
│   ├── african_accessions
│   ├── SRR_Acc_List
│   └── SRR_run
|
├── py_scripts/                           # Python scripts, one per pipeline step
│   ├── geo_check.py                      # Step: NCBI BioSample API geographic verification
│   ├── collate.py                        # Step: custom TB-Profiler JSON collation
│   ├── build_ml_dataset.py               # Step 1: ML dataset construction
│   ├── comparative_analysis.py           # Step 2: Fisher's exact test, Objective 1
│   ├── amr_summary.py                    # Step 3: clinical AMR catalogue, Objective 1
│   ├── codon_classify.py                 # Step 9 support: codon-level variant classification
│   ├── fitch.py                          # Step 9 support: root-fixed Fitch parsimony
│   ├── step9_1_build_tree.py             # Step 9: background phylogenetic tree construction
│   ├── step9_2_run_analysis.py           # Step 9: homoplasy + tree-corrected dN/dS
│   ├── step10_structural_context.py      # Step 10: protein structural/biochemical context
│   └── step11_final_integration.py       # Step 11: master evidence table + tiering
|
├── notebooks/
│   ├── train_models.ipynb                # Step 4: model training (LR, RF, XGBoost, EBM)
│   ├── shap_analysis.ipynb               # Step 5: SHAP explainability
│   ├── ebm_analysis.ipynb                # Step 6: EBM native explanation
│   ├── lineage_distribution_analysis.ipynb  # Step 8: lineage stratification
│   └── pub_figures.ipynb                 # Step 12: publication figure set
|
├── ml_outputs/                            # All step outputs (large files not pushed to GitHub)
│   ├── step1_dataset/                     # Step 1 outputs (X_array.npy, y_labels.csv, ...)
│   ├── step2_comparative/                 # Step 2 outputs
│   ├── step4_models/                      # Step 4 outputs (model pickles, comparison table)
│   ├── step5_shap/                        # Step 5 outputs
│   ├── step6_ebm/                         # Step 6 outputs
│   ├── step8_lineage/                     # Step 8 outputs
│   ├── step9_conservation/                # Step 9 outputs (tree files, homoplasy, dN/dS)
│   ├── step10_structural/                 # Step 10 outputs (structural context, cached PDB/CIF files)
│   ├── step11_integration/                # Step 11 outputs (master evidence table)
│   └── step12_publication_figures/        # Step 12 outputs (9 figure PNGs)
|
├── tbprofiler_results/
│   ├── compensatory.csv
│   ├── dr_variants.csv
│   ├── labels.csv
│   ├── rpoB_nonRRDR.csv
│   └── summary_stats.txt
|
|
├── install_tools.sh                       # One-time environment setup
├── run_batch.sh                           # Main pipeline entry point
├── worker.sh                              # Per-sample variant calling worker
├── check_results.sh                       # Progress monitoring
├── filter_vcfs.sh                         # Quality-based VCF filtering
├── run_tbprofiler.sh                      # TB-Profiler batch runner with chromosome fix
├── merge_vcfs.sh                          # VCF merger + feature matrix construction
├── .gitignore
└── README.md
```

**Not included in this repository (large data files):**
- `vcf_output/` - raw per-sample VCF files (~3,100 files)
- `vcf_filtered/` - quality-filtered VCF files (1,858 files)
- `reference/` - H37Rv reference genome (download from NCBI: NC_000962.3)
- `scratch/` - temporary pipeline working files (auto-cleaned after each sample)
- `logs/` - machine-specific run logs
- Model pickles (`*.pkl`) and cached protein structure files (`*.pdb`, `*.cif`) under `ml_outputs/`

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

This installs the core variant-calling toolchain via Bioconda and creates the working directory structure under `~/tb_pipeline/`.

**Step 3 - Install the analysis-stage packages** (needed for Steps 4 onward, not part of `install_tools.sh`):
```bash
mamba activate tb_amr
mamba install -c conda-forge -c bioconda "iqtree=2" biopython ete3 -y
pip install scikit-learn xgboost interpret shap openpyxl jupyter nbconvert --break-system-packages
```
See Section 13 for exactly which step needs which of these, and the version-pinning rationale for `iqtree`.

**Step 4 - Download the H37Rv reference genome:**
```bash
mkdir -p ~/tb_pipeline/reference
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?\
db=nuccore&id=NC_000962.3&rettype=fasta&retmode=text" \
     -o ~/tb_pipeline/reference/H37Rv.fasta
```

Also place the matching NCBI RefSeq GFF annotation (`GCF_000195955.2_ASM19595v2_genomic.gff`) in the same `reference/` directory - required by Step 9 and Step 10 for CDS coordinates.

**Step 5 - Place batch files:**
```bash
cp batches/batch_* ~/tb_pipeline/batches/
```

---

## 6. Pipeline Architecture

### 6.1 Overview

The pipeline follows a **reference-based variant calling** approach - the standard for *M. tuberculosis* WGS analysis. Each sample's reads are aligned to H37Rv and positions where the sample differs from the reference are recorded as variants.

```
SRA Accession ID (e.g. SRR5181828)
        |
        v
  prefetch          Downloads compressed .sra file from NCBI SRA
        |
        | .sra file (2-4 GB) - deleted immediately after next step
        v
  fasterq-dump      Converts .sra to FASTQ, splits paired-end reads
        |
        | _1.fastq + _2.fastq (or .fastq for single-end)
        v
  bwa mem  --pipe-->  samtools sort      (SAM never written to disk)
  (align to H37Rv)    (coordinate sort)
        |
        | FASTQs deleted immediately after alignment
        | .sorted.bam
        v
  bcftools mpileup --pipe--> bcftools call
  (per-position read summary)  (variant calls)
        |
        | BAM deleted after calling
        | .vcf.gz + .csi
        v
  vcf_output/<ID>.vcf.gz
```

### 6.2 Disk-Efficient Design

| Intermediate file | Naive approach | This pipeline |
|---|---|---|
| SAM file | Written to disk (13-25 GB) | **Never written** - BWA piped directly to samtools |
| .sra cache | Kept until end | **Deleted immediately** after fasterq-dump |
| FASTQ files | Kept until end | **Deleted immediately** after alignment |
| BAM file | Kept until end | **Deleted immediately** after variant calling |
| **Peak scratch per sample** | **~44 GB** | **~3-4 GB** |

### 6.3 Resumable Design

Before processing any sample, the pipeline checks whether a completed `.vcf.gz` already exists. If it does, the sample is skipped - no network access, no computation. Safe to interrupt and restart at any point. This same resumability principle carries into the analysis steps too - Step 9's tree-building, for example, is specifically designed to survive being interrupted partway through (see Section 10.2).

### 6.4 Parallelism

| Environment | Jobs | Reasoning |
|---|---|---|
| Local WSL2 (i5-13450HX, 16 threads) | `-j 6` | 6 x 2 BWA threads = 12 threads used for alignment |
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

### 7.3 Quality filtering (QUAL>=20, DP>=4)

```bash
mamba activate tb_amr
bash ~/tb_pipeline/filter_vcfs.sh
```

This removes variants where QUAL < 20 (less than 99% confidence) OR DP < 4 (fewer than 4 supporting reads). Typical effect: ~17% of raw variants removed per sample.

### 7.4 Geographic verification

```bash
cd ~/tb_pipeline
python3 py_scripts/geo_check.py
# Queries NCBI BioSample API in batches of 500
# Result: 1,858 of 3,100 VCFs confirmed African origin
```

---

## 8. Post-VCF Processing (Steps 1-3, Objective 1)

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

### 8.2 VCF Merging and Feature Matrix (merge_vcfs.sh)

```bash
mamba activate tb_amr
bash ~/tb_pipeline/merge_vcfs.sh
```

Merges all 1,858 filtered VCFs into a single multi-sample variant matrix. Merge + frequency filtering in one pass:
```bash
bcftools merge --file-list vcf_list.txt --missing-to-ref --force-samples \
    --output-type u --threads 8 \
| bcftools view --min-af 0.01:alt1 --max-af 0.99:alt1 \
    --output-type z --output merged_prefilt.vcf.gz --threads 8
```
- Raw merged positions: 831,000+
- After 1%-99% allele frequency filter: **94,583 variants retained**

Genotype matrix extraction (`bcftools query`) and chunked binary conversion (50,000 variants/chunk) follow, producing `feature_matrix.npz` (dense NumPy, 1858 x 94583).

### 8.3 TB-Profiler Resistance Prediction (run_tbprofiler.sh)

```bash
mamba activate tb_amr
bash ~/tb_pipeline/run_tbprofiler.sh
```
WHO v2+ resistance catalogue, `--vcf` mode (uses pre-called VCF). The built-in `tb-profiler collate` failed on this dataset, so a custom `collate.py` was written instead (see Section 8.4).

### 8.4 Custom Collation (collate.py)

```bash
python3 ~/tb_pipeline/py_scripts/collate.py
```

Key design decisions:
- **rpoB RRDR separation**: codons 426-452 are primary resistance (-> `dr_variants.csv`); non-RRDR rpoB mutations in MDR samples are compensatory candidates (-> `rpoB_nonRRDR.csv`)
- **Compensatory gene panel**: 12 scientifically validated genes (rpoA, rpoC, ahpC, kasA, ndh, gyrB, gid, mmpR5/Rv0678, mmpL5, embR, whiB7, eis), each with mechanism, evidence, and `requires_MDR_context` flags - see Section 15.2 for a real gene-panel consistency bug found and fixed between this script and later steps

### 8.5 ML Dataset Construction - Step 1 (build_ml_dataset.py)

```bash
python3 ~/tb_pipeline/py_scripts/build_ml_dataset.py
```

- Fixes malformed variant IDs (27,494 multi-allelic comma-containing IDs -> first ALT kept)
- Produces the final, correct **1,858-sample** cohort with **zero exclusions** (an earlier version of this join had a row-shift bug that silently excluded one sample, SRR11922476 - found and fixed; see Section 15.1)
- Post-alignment MAF re-filter: removes zero-variance features after sample subsetting

### 8.6 Comparative Genomic Analysis - Step 2 (comparative_analysis.py)

```bash
python3 ~/tb_pipeline/py_scripts/comparative_analysis.py
```

Genome-wide Fisher's exact test (MDR vs non-MDR) across all 94,583 variants, with Benjamini-Hochberg FDR correction. Produces the full statistical screen and the compensatory-panel-specific candidate list.

**Real, final results:** 92,968 of 94,583 variants significant overall; **1,209 significant compensatory-mutation candidates** (the population every later objective's analysis is built around).

### 8.7 Clinical AMR Catalogue - Step 3 (amr_summary.py)

```bash
python3 ~/tb_pipeline/py_scripts/amr_summary.py
```

Builds the clinical drug-resistance summary: 547 mutation-drug pairs, per-drug and per-gene breakdowns. Confirmed to have zero dependency on the compensatory gene-panel fix described in Section 15.2 (checked directly against its source code, then confirmed by an identical rerun).

---

## 9. Explainable Machine Learning (Steps 4-7, Objective 2)

### 9.1 Step 4 - Model Training (train_models.ipynb)

Trains four models via 5-fold stratified cross-validation, with chi-squared feature selection (top 5,000 of 94,583 features) performed *inside* each fold to avoid data leakage:

| Model | Cross-validated AUC (mean +/- SD) |
|---|---|
| XGBoost | **0.9937 +/- 0.0021** (best-performing) |
| Logistic Regression | 0.9892 +/- 0.0040 |
| EBM (Explainable Boosting Machine) | 0.9867 +/- 0.0049 |
| Random Forest | 0.9663 +/- 0.0080 |

A gene-panel naming bug and an EBM feature-naming bug were both found and fixed at this step (Section 15.2, 15.3), and a genuine cross-platform (WSL vs Windows) reproducibility issue affecting XGBoost and EBM specifically was investigated and resolved by designating canonical models (Section 15.4).

### 9.2 Step 5 - SHAP Explainability (shap_analysis.ipynb)

SHAP TreeExplainer applied to the canonical XGBoost model. Real result: only **2 of the SHAP top-100 most important variants are compensatory-panel genes** (both embR, ranks 70 and 73) - the model's predictive signal is overwhelmingly dominated by primary-resistance-gene variants once those are known.

### 9.3 Step 6 - EBM Native Explanation (ebm_analysis.ipynb)

EBM's own exact, native additive decomposition - main effects and interaction terms with no post-hoc approximation. Real result: the top interaction term is **PPE19 x aspB** (importance 0.0446); **zero** compensatory-panel genes appear in EBM's own top-100 main effects, the same pattern as Step 5's SHAP result via a completely different method.

### 9.4 Step 7 - Objective 2 Synthesis

Reconciles Steps 4-6's findings, formally closing Objective 2. Full write-up: `reports/Phase4C_Obj2_Report.docx`.

---

## 10. Evolutionary, Lineage, and Structural Analysis (Steps 8-10, Objective 3)

### 10.1 Step 8 - Lineage Distribution Analysis (lineage_distribution_analysis.ipynb)

Tests whether each significant compensatory candidate's carriage is concentrated in specific MTb lineages (chi-square test of independence) and whether its association with MDR status is directionally consistent across lineages (lineage-stratified Fisher's exact test) - guarding against MTb's clonal population structure confounding a pooled association result.

**Real results:** of the 1,209 significant candidates, **629 resolve to a single, unambiguous position in the ML matrix** (analysable) and 580 do not. Of the 629, **42 are significantly lineage-restricted**, and exactly **1** (embR p.Phe376Leu) shows a genuinely inconsistent MDR-association direction across lineages.

### 10.2 Step 9 - Evolutionary Conservation (step9_1_build_tree.py, step9_2_run_analysis.py, codon_classify.py, fitch.py)

Builds a maximum-parsimony phylogenetic tree from the 93,615 background (non-gene-set) sites specifically, to avoid circularity with the positions being tested, then runs root-fixed Fitch parsimony on every gene-set/rpoB variant to count independent origin events, plus gene-wide tree-corrected dN/dS.

```bash
python3 ~/tb_pipeline/py_scripts/step9_1_build_tree.py
python3 ~/tb_pipeline/py_scripts/step9_2_run_analysis.py
```

**Real results:** 968 gene-set/rpoB variants classified (678 synonymous, 190 nonsynonymous, 97 noncoding, 3 indel); **606 of 968 (62.6%) show real evolutionary convergence** (more than one independent origin). Gene-wide dN/dS finds **gid (0.722) and embR (0.710)** under markedly weaker purifying selection than the rest of the panel (0.055-0.116). embR p.Phe376Leu is the single most homoplasious candidate in the whole project (68 independent origins). Several real, consequential bugs were found and fixed during this step - see Section 15.5-15.7.

### 10.3 Step 10 - Functional/Structural Significance (step10_structural_context.py)

```bash
python3 ~/tb_pipeline/py_scripts/step10_structural_context.py
```

For every nonsynonymous candidate, resolves a real protein structure (experimental PDB preferred, AlphaFold DB fallback, both queried dynamically via UniProt/RCSB/AlphaFold APIs), then computes Grantham biochemical distance and relative solvent accessibility (Shrake-Rupley algorithm).

**Real results:** all 79 nonsynonymous candidates biochemically scored; **69 of 79 (87.3%) structurally resolved** to a real relative solvent accessibility value. 57.0% of substitutions are biochemically conservative, 32.9% moderately conservative. Two consequential bugs (resolution-based structure selection picking unusable fragments; chain identity assumed rather than verified in multi-subunit structures) were found and fixed - see Section 15.8-15.9.

---

## 11. Final Integration and Publication Figures (Steps 11-12)

### 11.1 Step 11 - Final Integration (step11_final_integration.py)

```bash
python3 ~/tb_pipeline/py_scripts/step11_final_integration.py --indir <dir with all Step 2/4/5/6/8/9/10 outputs> --outdir ml_outputs/step11_integration
```

Joins every evidence layer above onto Step 2's 1,209 significant candidates by variant_id, and assigns each candidate one of five evidence tiers.

**Real results:**

| Tier | Count | % of 1,209 |
|---|---|---|
| Flagged anomaly - requires caveat | 1 | 0.1% |
| Strong, convergent, lineage-consistent | 352 | 29.1% |
| Convergent but lineage-restricted | 33 | 2.7% |
| Single-origin (clonal inheritance only) | 243 | 20.1% |
| Not conservation-tested (unresolved position) | 580 | 48.0% |

**embR p.Phe376Leu** is the project's one flagged anomaly - independently caught by three unrelated methods (Step 8's lineage-inconsistency test, Step 9's homoplasy count, and Step 2's own significance screen). **mmpL5 p.Ile948Val** is the project's strongest positive example, with fully consistent evidence across every layer.

### 11.2 Step 12 - Publication Figures (pub_figures.ipynb)

Produces 9 curated, consistently-styled figures (5 main + 4 supplementary) built from the already-verified results above - a genome-wide volcano plot, ROC curves, a SHAP importance summary, the integrated multi-layer evidence figure (the one genuinely new visual this step produces), plus supplementary figures for lineage restriction, gene dN/dS, EBM interactions, and structural/biochemical distributions. Full details and what each figure tells you: `reports/Phase7_Step12_Report.docx`.

---

## 12. Output Files and Their Role

### 12.1 Step 1 (`ml_outputs/step1_dataset/`)

| File | Dimensions | Used for |
|---|---|---|
| `X_array.npy` | (1858, 94583) uint8 | Primary feature matrix for all ML training |
| `y_mdr_array.npy` | (1858,) int8 | MDR binary target - 871 positive, 987 negative |
| `y_labels.csv` | 1,858 x 49 cols | Full aligned label file - all resistance and lineage columns |
| `sample_ids.txt` | 1,858 lines | Sample IDs in row order matching X_array |
| `feature_names_clean.txt` | 94,583 lines | Variant IDs in column order matching X_array |
| `variant_metadata_with_genes.csv` | 94,583 rows | Gene/position metadata per variant, incl. compensatory/primary-DR flags |

### 12.2 Step 2 (`ml_outputs/step2_comparative/`)

| File | Description |
|---|---|
| `obj1_fisher_all_variants.csv` | Full genome-wide screen, all 94,583 variants |
| `obj1_compensatory_fisher.csv` | Compensatory-panel-specific candidate list with Fisher statistics |
| `obj1_fisher_significant.csv` | The 92,968 significant variants overall |

### 12.3 Step 4 (`ml_outputs/step4_models/`)

Model pickles (`lor_pipeline.pkl`, `rf_pipeline.pkl`, `xgb_pipeline.pkl`, `ebm_pipeline.pkl`), `obj2_model_comparison.csv` (the authoritative CV performance table), `obj2_rf_top30_features.csv`, `obj2_consensus_features.csv`.

### 12.4 Step 5 (`ml_outputs/step5_shap/`)

`obj2_shap_top100_annotated.csv`, `obj2_shap_consensus_features.csv`, PE_PGRS/PPE mapping-quality and paralogy check outputs.

### 12.5 Step 6 (`ml_outputs/step6_ebm/`)

`obj2_ebm_top100_annotated.csv`, `obj2_ebm_all_terms.csv`, `obj2_ebm_interactions_annotated.csv`.

### 12.6 Step 8 (`ml_outputs/step8_lineage/`)

| File | Rows | Description |
|---|---|---|
| `obj3_lineage_distribution.csv` | 629 | Analysable candidates, per-lineage carriage, restriction test result |
| `obj3_lineage_not_analyzable.csv` | 580 | The remaining significant candidates, with the reason they couldn't be resolved |
| `obj3_lineage_stratified_mdr_association.csv` | 2,516 | Per-candidate, per-lineage MDR association test (629 x 4) |
| `obj3_lineage_inconsistent_mdr_direction.csv` | 1 | The single flagged anomaly (embR p.Phe376Leu) |

### 12.7 Step 9 (`ml_outputs/step9_conservation/`)

| File | Description |
|---|---|
| `full_tree_complete.parstree` | The completed 1,858-tip background phylogenetic tree |
| `obj3_step9_geneset_homoplasy.csv` | All 968 gene-set/rpoB variants, homoplasy result per variant |
| `obj3_step9_gene_dNdS.csv` | Per-gene, tree-corrected dN/dS (12 compensatory-panel genes) |
| `obj3_step9_candidate_conservation.csv` | The 629 analysable candidates joined with their conservation result |

### 12.8 Step 10 (`ml_outputs/step10_structural/`)

`obj3_step10_structural_context.csv` (Grantham distance + RSA per nonsynonymous candidate), `obj3_step10_structure_provenance.csv` (which structure was used per gene, and why), plus a `structures/` subfolder caching the actual downloaded PDB/CIF files.

### 12.9 Step 11 (`ml_outputs/step11_integration/`)

| File | Description |
|---|---|
| `obj_final_master_evidence_table.csv` | **1,209 rows, 38 columns - every evidence layer, one row per candidate.** The single most complete artefact this project produces. |
| `obj_final_master_evidence_table.xlsx` | The same table, formatted for direct human review (colour-coded by evidence tier) |
| `obj_final_evidence_summary.csv` | The 5-row tier breakdown |

### 12.10 Step 12 (`ml_outputs/step12_publication_figures/`)

9 PNG figures at 300 DPI (see Section 11.2), plus `Figure_Captions.docx` for manuscript-ready caption text.

---

## 13. Tools and Software

### 13.1 Variant calling (Steps 1-3)

| Tool | Version | Role |
|---|---|---|
| `prefetch` / `fasterq-dump` (SRA Toolkit) | 3.1.1 | Download and convert SRA reads |
| `bwa mem` | 0.7.19 | Align reads to H37Rv |
| `samtools` | 1.23.1 | BAM sorting, indexing |
| `bcftools` | 1.23.1 | Variant calling, merging, filtering, annotation |
| `GNU Parallel` | 20160622 | Concurrent multi-sample processing |
| `TB-Profiler` | Latest (WHO v2+) | Drug resistance prediction and lineage classification |

### 13.2 Analysis stage (Steps 4-12)

| Tool | Role | Used in |
|---|---|---|
| `scikit-learn` | Logistic Regression, Random Forest, pipelines, chi2 feature selection | Steps 4, 5, 10, 12 |
| `xgboost` | XGBoost classifier | Steps 4, 5, 12 |
| `interpret` | Explainable Boosting Machine (EBM) | Steps 4, 6, 12 |
| `shap` | SHAP TreeExplainer | Step 5 |
| `iqtree2` (version 2.x specifically - see note below) | Maximum-parsimony phylogenetic tree construction | Step 9 |
| `ete3` | Tree manipulation (loading, tip reattachment) | Step 9 |
| `biopython` | Codon translation, protein structure parsing, solvent accessibility (Shrake-Rupley) | Steps 9, 10 |
| `openpyxl` | Formatted Excel output | Step 11 |
| `jupyter` / `nbconvert` | Notebook execution | Steps 4, 5, 6, 8, 12 |

**Note on `iqtree2` versioning:** pin to the 2.x line specifically (`mamba install -c bioconda "iqtree=2"`), not whatever bioconda currently defaults to (3.x at time of writing). This project's Step 9 log-parsing logic was found to break across even minor IQ-TREE version changes (2.0.7 vs 2.4.0 use different log formats for the same underlying event) before being redesigned to work independent of log format entirely - see Section 15.7.

**Reference genome:** H37Rv - *Mycobacterium tuberculosis* H37Rv complete genome - NC_000962.3 - 4,411,532 bp, plus the matching NCBI RefSeq GFF annotation for CDS coordinates (Steps 9, 10).

---

## 14. Key Findings Summary

This section pulls together the handful of results worth knowing before reading any individual step's own report in full.

1. **1,209 candidate compensatory mutations were identified as statistically significant** (Step 2), out of 94,583 genome-wide tested variants.

2. **XGBoost is this project's best-performing model** (cross-validated AUC 0.9937), and **compensatory-panel variants are consistently, independently found to add little predictive signal on top of primary resistance markers** - confirmed by two unrelated explainability methods (SHAP, Step 5: 2/100; EBM, Step 6: 0/100).

3. **Of the 629 candidates that could be evolutionarily and structurally characterised, 385 (61.2%) show real, independent evolutionary convergence with no lineage-consistency concerns** - a substantial, well-supported core of genuinely promising candidates.

4. **embR p.Phe376Leu is this project's single most important candidate to treat with caution, not without it.** Independently flagged as anomalous by three separate methods across three separate steps (Step 2's significance, Step 8's lineage-inconsistency test, Step 9's homoplasy count) - and Step 10's structural finding (buried, but only a conservative substitution) does not resolve the question either way.

5. **mmpL5 p.Ile948Val is this project's strongest positive example** - extremely significant, genuinely convergent (25 independent origins on 1,600 carriers), and a structurally/biochemically plausible substitution (exposed, conservative) - a coherent, mutually-reinforcing case across every evidence layer.

6. **gid and embR are under markedly weaker purifying selection than the rest of the compensatory panel** (gene-wide dN/dS 0.722 and 0.710 respectively, vs. 0.055-0.116 for the other seven genes with observable coding variation) - worth particular attention in any follow-up mechanistic work.

Full detail behind every one of these points is in the corresponding step's own report under `reports/`.

---

## 15. Known Issues Found and Fixed During This Project

This project's own working history surfaced a number of real, non-trivial bugs - documented here because several of them are the kind of issue that can silently recur in similar pipelines, and because this project's own numbers should be read alongside knowing what was already checked.

### 15.1 Row-shift bug in Step 1's sample join (fixed before this project's later stages)
An early version of `build_ml_dataset.py` silently excluded one sample (SRR11922476) via a row-misalignment in its join logic. Fixed; the current, correct pipeline output is the full 1,858-sample cohort with zero exclusions.

### 15.2 Gene-panel inconsistency between Step 2 and Step 5/9 (PRIMARY_DR_GENES / COMPENSATORY_GENES)
`comparative_analysis.py` and `shap_analysis.ipynb` used inconsistent gene lists for the same intended "authoritative" panel, additionally missing an alias (Rv0678 = mmpR5, the same physical gene under two names depending on which annotation source resolved a given variant - the reference GFF has no `gene=` tag at this locus). Fixed across every script that touches this gene panel, including `codon_classify.py` (Step 9) and `step10_structural_context.py`, each via the appropriate mechanism for that script's own design (a flat alias substitution in Steps 2/5; a `GENE_NAME_ALIASES` normalisation, not a duplicate panel entry, in Step 9, since duplicating the entry was tested and found to silently split this gene's statistics across two incomplete buckets).

### 15.3 EBM placeholder-name bug silently breaking Step 4's own consensus computation
`train_models.ipynb`'s EBM feature-importance cell used raw internal placeholder names (`feature_0001`, etc.) instead of resolved variant IDs, which cascaded into a separate cell's cross-model consensus computation never being able to match EBM's features against RF's and XGBoost's real variant IDs. Fixed with a one-line change reusing the already-correct name-resolution logic from an earlier cell.

### 15.4 Cross-platform (WSL vs Windows) reproducibility issue for XGBoost and EBM
Retraining on a different machine produced different XGBoost trees and different EBM chi2-selected features and interaction terms, traced to numpy/scipy build differences between platforms (not just version numbers - same versions, different compiled BLAS backends). Logistic Regression and Random Forest were confirmed fully reproducible across platforms; XGBoost and EBM were not. Resolved by designating canonical models explicitly (the original WSL-trained XGBoost; a Windows-trained EBM model, per this project's own decision) rather than treating either platform's output as silently authoritative.

### 15.5 mmCIF-only PDB structures, coverage-based vs resolution-based structure selection (Step 10)
Some real PDB entries (e.g. large cryo-EM structures) are only distributed in mmCIF format, not legacy PDB - `step10_structural_context.py` was fixed to try both formats. More significantly: selecting a gene's structure by best resolution alone picked a real but unusably small protein fragment for rpoC (covering 0 of its 20 real candidate residues) over a full-length structure ranked lower only because it had a worse nominal resolution. Fixed by selecting structures based on actual, verified coverage of the residues a given gene's real candidates need, not resolution ranking.

### 15.6 Chain identity assumed rather than verified in multi-subunit structures (Step 10)
An early version trusted a chain literally named "A" without checking it was the right protein - a real risk for multi-subunit structures (e.g. an RNA polymerase structure containing rpoA, rpoB, and rpoC as separate chains in one file). Caught by a visible inconsistency (coverage-counting reported more resolvable candidates than were actually computed) and fixed by verifying each candidate chain actually has the expected wild-type amino acid at the expected position before accepting it.

### 15.7 IQ-TREE silently drops genotype-identical taxa from its own tree output (Step 9)
IQ-TREE drops any sample genotypically identical to another (across the background sites used for tree-building) from the tree it constructs, intending to re-graft them once a full run completes - a step never reached on the single-core, interrupted-at-parsimony workflow this project uses. An initial fix parsed IQ-TREE's own log for the dropped-taxa mapping, but this was found to break across IQ-TREE versions (2.0.7 vs 2.4.0 use incompatible log formats for the same event). The final, version-independent fix computes identical-sequence groups directly from the alignment file this project's own code controls, and reattaches missing taxa by cross-referencing against the tree's actual tip set - unaffected by anything IQ-TREE's log does or doesn't say.

### 15.8 Grantham distance formula missing its scaling constant (Step 10)
An initial implementation of the Grantham biochemical-distance calculation omitted the published scaling factor (rho = 50.723), producing values roughly 50x too small and silently classifying every substitution as "conservative" regardless of true severity. Caught by the output looking suspiciously uniform, then verified against 9 known published reference values before being trusted.

### 15.9 Duplicate-row handling in downstream evidence-integration work
`df[df.duplicated(keep=False)]` followed by dropping every flagged row discards both the true duplicate copy *and* the first, genuinely unique occurrence of every duplicated row - a mistake that surfaced more than once across this project's later analysis scripts. The correct operation is `df.drop_duplicates()` (default `keep='first'`), which retains exactly one copy of each unique row. Where a lookup table's data applies identically regardless of which of several duplicate-position rows it is joined onto (Step 11), the lookup table itself is deduplicated by key before merging, rather than deduplicating the master table and losing a genuinely distinct record.

---

## 16. Progress Status

| Stage | Status | Details |
|---|---|---|
| NCBI SRA metadata acquisition | Complete | 19,000+ records reviewed |
| African isolate filtering | Complete | 1,858 accessions identified |
| Variant calling (WSL2 + Colab) | Complete | 3,100 accessions processed |
| Geographic verification | Complete | 1,858 / 3,100 confirmed African |
| VCF quality filtering, chromosome renaming | Complete | All 1,858 African isolates |
| VCF merging and feature matrix | Complete | 94,583 variants |
| TB-Profiler resistance prediction + collation | Complete | 1,858 samples, 0 failures |
| **Step 1** - ML dataset construction | Complete | 1,858 samples x 94,583 features |
| **Step 2** - Comparative genomic analysis (Objective 1) | Complete | 1,209 significant compensatory candidates |
| **Step 3** - Clinical AMR catalogue (Objective 1) | Complete | 547 mutation-drug pairs |
| **Step 4** - Model training (Objective 2) | Complete | 4 models, 5-fold CV |
| **Step 5** - SHAP explainability (Objective 2) | Complete | Top-100 importance ranking |
| **Step 6** - EBM native explanation (Objective 2) | Complete | Main effects + 7 interaction terms |
| **Step 7** - Objective 2 synthesis | Complete | Cross-model reconciliation |
| **Step 8** - Lineage distribution (Objective 3) | Complete | 629 analysable, 42 lineage-restricted |
| **Step 9** - Evolutionary conservation (Objective 3) | Complete | 968 variants, tree-corrected dN/dS |
| **Step 10** - Structural significance (Objective 3) | Complete | 69/79 nonsynonymous candidates resolved |
| **Step 11** - Final integration | Complete | 1,209-row evidence-tiered master table |
| **Step 12** - Publication figures | Complete | 9 figures, main + supplementary |

**All three objectives, all eleven analytical steps, and the publication-figures step are complete.**

---

## 17. Citation

This repository is part of an ongoing research project (manuscript preparation underway). Full citation details will be provided upon publication.

**Please cite the following tools if you use this pipeline:**

- **BWA:** Li, H. (2013). Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM. *arXiv:1303.3997*.
- **SAMtools / BCFtools:** Danecek, P. et al. (2021). Twelve years of SAMtools and BCFtools. *GigaScience*, 10(2), giab008.
- **GNU Parallel:** Tange, O. (2011). GNU Parallel - The Command-Line Power Tool. *;login: The USENIX Magazine*, 36(1), 42-47.
- **TB-Profiler:** Phelan, J. et al. (2019). Integrating informatics tools and portable sequencing technology for rapid detection of resistance to anti-tuberculous drugs. *Genome Medicine*, 11:41.
- **WHO Resistance Catalogue:** WHO (2023). Catalogue of mutations in *M. tuberculosis* complex and their association with drug resistance.
- **H37Rv reference genome:** Cole, S.T. et al. (1998). Deciphering the biology of *Mycobacterium tuberculosis* from the complete genome sequence. *Nature*, 393(6685), 537-544.
- **scikit-learn:** Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
- **XGBoost:** Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- **InterpretML (EBM):** Nori, H. et al. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv:1909.09223*.
- **SHAP:** Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- **IQ-TREE:** Minh, B.Q. et al. (2020). IQ-TREE 2: New Models and Efficient Methods for Phylogenetic Inference in the Genomic Era. *Molecular Biology and Evolution*, 37(5), 1530-1534.
- **ete3:** Huerta-Cepas, J. et al. (2016). ETE 3: Reconstruction, Analysis, and Visualization of Phylogenomic Data. *Molecular Biology and Evolution*, 33(6), 1635-1638.
- **Biopython:** Cock, P.J.A. et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*, 25(11), 1422-1423.
- **Grantham distance:** Grantham, R. (1974). Amino Acid Difference Formula to Help Explain Protein Evolution. *Science*, 185(4154), 862-864.
- **Fitch parsimony:** Fitch, W.M. (1971). Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology. *Systematic Zoology*, 20(4), 406-416.
