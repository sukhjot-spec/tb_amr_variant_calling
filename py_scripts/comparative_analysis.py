#!/usr/bin/env python3
"""
objective 1 — Comparative genomic analysis: MDR vs Susceptible

inputs (all from ml_outputs/):
  X_array.npy                        — 1857 x 94583 binary variant matrix
  y_mdr_array.npy                    — 1857 MDR labels
  y_labels.csv                       — full labels with lineage, drtype, per-drug
  feature_names_clean.txt            — 94583 variant IDs
  variant_metadata_filtered.csv      — CHROM/POS/REF/ALT per variant
  obj1_mdr_vs_susceptible_comp.csv   — pre-built compensatory enrichment table
  obj1_compensatory_in_MDR_samples.csv — compensatory records in MDR context
  obj1_rpoB_nonRRDR_summary.csv      — rpoB non-RRDR candidates

From tbprofiler_results/:
  dr_variants.csv                    — known primary DR mutations

Outputs (all to ml_outputs/):
  obj1_fisher_all_variants.csv       — all 94583 variants + Fisher p-values
  obj1_fisher_significant.csv        — FDR < 0.05 variants
  obj1_fisher_compensatory.csv       — FDR < 0.05 in compensatory genes
  obj1_known_amr_summary.csv         — known DR mutation frequency table
  obj1_gene_enrichment_summary.csv   — gene-level enrichment summary
  obj1_compensatory_fisher.csv       — Fisher test on pre-built comp table

(Fisher's test on 94,583 variants)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats import false_discovery_control
import warnings
warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/tb_pipeline")
ML_OUT   = os.path.join(BASE, "ml_outputs/obj1_outputs")
TB_OUT   = os.path.join(BASE, "tbprofiler_results")
REF_DIR  = os.path.join(BASE, "reference")
GFF_PATH = os.path.join(REF_DIR, "GCF_000195955.2_ASM19595v2_genomic.gff")

#gene set
COMPENSATORY_GENES = {
    "rpoA", "rpoC",           # rifampicin fitness compensation
    "ahpC", "kasA", "ndh",   # isoniazid pathway
    "gyrB",                   # fluoroquinolone pathway
    "gid",                    # streptomycin pathway
    "mmpR5", "mmpL5",        # bedaquiline/clofazimine efflux
    "embR",                   # ethambutol pathway
    "whiB7", "eis",          # broad fitness restorers
}

#tracking rpoB non-RRDR separately
PRIMARY_DR_GENES = {"rpoB", "katG", "inhA", "embB", "embA", "embC", "pncA", "gyrA", "rrs", "rpsL", "tlyA", "rplC", "fgd1", "dprE1", "pepQ", "Rv0678", "atpE"}

print("Comparative Genomic Analysis (MDR vs Susceptible)")

# 1. Load all Step 1 outputs
print("\n[1/8] Loading Step 1 outputs...")

X = np.load(os.path.join(ML_OUT, "X_array.npy"))
y = np.load(os.path.join(ML_OUT, "y_mdr_array.npy"))

with open(os.path.join(ML_OUT, "feature_names_clean.txt")) as f:
    variant_ids = [line.strip() for line in f if line.strip()]

meta   = pd.read_csv(os.path.join(ML_OUT, "variant_metadata_filtered.csv"))
labels = pd.read_csv(os.path.join(ML_OUT, "y_labels.csv"))

# Objective 1 pre-built files
comp_enrich = pd.read_csv(os.path.join(ML_OUT, "obj1_mdr_vs_susceptible_comp.csv"))
comp_mdr    = pd.read_csv(os.path.join(ML_OUT, "obj1_compensatory_in_MDR_samples.csv"))
rpob_nonRRDR = pd.read_csv(os.path.join(ML_OUT, "obj1_rpoB_nonRRDR_summary.csv"))

# Validation checks
assert X.shape[0] == y.shape[0],       "X/y row mismatch"
assert X.shape[1] == len(variant_ids), "X cols / variant_ids mismatch"
assert X.shape[0] == len(labels),      "X / labels row mismatch"

print(f"  X              : {X.shape}")
print(f"  y (MDR)        : {y.shape} | MDR={y.sum()} | non-MDR={(y==0).sum()}")
print(f"  variant_ids    : {len(variant_ids):,}")
print(f"  labels         : {labels.shape}")
print(f"  comp_enrich    : {comp_enrich.shape} ({comp_enrich['gene'].nunique()} genes)")
print(f"  comp_mdr       : {comp_mdr.shape}")
print(f"  rpoB_nonRRDR   : {rpob_nonRRDR.shape}")

X_mdr  = X[y == 1]
X_susc = X[y == 0]
n_mdr  = X_mdr.shape[0]
n_susc = X_susc.shape[0]

# 2 gene annotation from GFF
print("\n[2/8] Loading gene annotations...")

def parse_gff(gff_path: str) -> pd.DataFrame:
    """Parse H37Rv GFF3 -extract gene name and CDS coordinates."""
    rows = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] not in ("gene", "CDS"):
                continue
            start = int(parts[3])
            end   = int(parts[4])
            attrs = {kv.split("=")[0]: kv.split("=")[1]
                     for kv in parts[8].split(";")
                     if "=" in kv}
            gene = (attrs.get("gene") or
                    attrs.get("Name") or
                    attrs.get("locus_tag") or "")
            if gene:
                rows.append({"gene": gene, "start": start, "end": end})
    df = pd.DataFrame(rows).drop_duplicates()
    df = df.sort_values("start").reset_index(drop=True)
    return df

USE_GFF = False
gff_df  = pd.DataFrame(columns=["gene","start","end"])

if os.path.exists(GFF_PATH):
    gff_df  = parse_gff(GFF_PATH)
    USE_GFF = True
    print(f"  GFF loaded: {len(gff_df):,} gene/CDS records")
else:
    print(f"  GFF not found - gene names will be 'unannotated'")
    print(f"  Run this to download it:")
    print(f"    mkdir -p ~/tb_pipeline/reference")
    print(f"    wget -P ~/tb_pipeline/reference/ \\")
    print(f"    'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/"
          f"GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.gff.gz'")
    print(f"    gunzip ~/tb_pipeline/reference/"
          f"GCF_000195955.2_ASM19595v2_genomic.gff.gz")
    print(f"  Then rerun this script to get gene-level annotation.")

def pos_to_gene(pos: int, gff: pd.DataFrame) -> str:
    """Binary search - returns gene name or 'intergenic'."""
    hits = gff[(gff["start"] <= pos) & (gff["end"] >= pos)]
    return hits["gene"].iloc[0] if len(hits) > 0 else "intergenic"


# 3 fisher's exact test on all 94,583 variants
#    Contingency table for each variant
#    Null hypothesis: variant frequency is equal in MDR and non-MDR
print(f"\n[3/8] Fisher's exact test on {len(variant_ids):,} variants...")
print(f"  MDR n={n_mdr} | Non-MDR n={n_susc}")
print()

#vectorised count computation fast numpy
mdr_alt  = X_mdr.sum(axis=0).astype(np.int32)   # a
mdr_ref  = n_mdr  - mdr_alt                      # b
susc_alt = X_susc.sum(axis=0).astype(np.int32)   # c
susc_ref = n_susc - susc_alt                     # d

mdr_freq  = mdr_alt  / n_mdr
susc_freq = susc_alt / n_susc
freq_diff = mdr_freq - susc_freq   # positive = enriched in MDR

pvalues    = np.ones(len(variant_ids))
oddsratios = np.ones(len(variant_ids))

BATCH = 5000
for start in range(0, len(variant_ids), BATCH):
    end = min(start + BATCH, len(variant_ids))
    for i in range(start, end):
        a, b = int(mdr_alt[i]),  int(mdr_ref[i])
        c, d = int(susc_alt[i]), int(susc_ref[i])
        #skipping if variant absent in BOTH groups as it gives no information
        if a == 0 and c == 0:
            pvalues[i]    = 1.0
            oddsratios[i] = 1.0
        else:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            pvalues[i]    = p
            oddsratios[i] = odds if np.isfinite(odds) else (
                999.0 if a > 0 else 0.001
            )
    if start % 20000 == 0 or end == len(variant_ids):
        print(f"  {end:,} / {len(variant_ids):,} variants tested...")

print(f"  Fisher's tests complete ")

# 4 FDR correction - Benjamini-Hochberg
print("\n[4/8] Benjamini-Hochberg FDR correction...")
padj = false_discovery_control(pvalues, method="bh")

print(f"  FDR < 0.001 : {(padj < 0.001).sum():,} variants")
print(f"  FDR < 0.01  : {(padj < 0.01).sum():,} variants")
print(f"  FDR < 0.05  : {(padj < 0.05).sum():,} variants")
print(f"  FDR < 0.10  : {(padj < 0.10).sum():,} variants")

# 5 making annotated results dataframe
print("\n[5/8] Building annotated results table...")

#parsing variant IDs into components
parsed = []
for vid in variant_ids:
    parts = vid.split("_")
    parsed.append({
        "variant_id" : vid,
        "CHROM"      : parts[0],
        "POS"        : int(parts[1]),
        "REF"        : parts[2],
        "ALT"        : "_".join(parts[3:]),
    })

results_df = pd.DataFrame(parsed)
results_df["mdr_alt_count"]   = mdr_alt.astype(int)
results_df["susc_alt_count"]  = susc_alt.astype(int)
results_df["mdr_ref_count"]   = mdr_ref.astype(int)
results_df["susc_ref_count"]  = susc_ref.astype(int)
results_df["mdr_frequency"]   = mdr_freq.round(6)
results_df["susc_frequency"]  = susc_freq.round(6)
results_df["freq_diff"]       = freq_diff.round(6)
results_df["odds_ratio"]      = oddsratios.round(4)
results_df["pvalue"]          = pvalues
results_df["padj_BH"]         = padj
results_df["neg_log10_padj"]  = -np.log10(np.clip(padj, 1e-300, 1))
results_df["significant_005"] = (padj < 0.05).astype(int)
results_df["significant_001"] = (padj < 0.01).astype(int)
results_df["enriched_in"]     = np.where(freq_diff > 0, "MDR",np.where(freq_diff < 0, "Susceptible", "Neither"))

#gene annotation
if USE_GFF:
    print("  Annotating variants with gene names from GFF...")
    results_df["gene"] = results_df["POS"].apply(
        lambda p: pos_to_gene(p, gff_df)
    )
else:
    results_df["gene"] = "unannotated"

#flag compensatory and primary DR gene variants
results_df["is_compensatory_gene"] = (results_df["gene"].isin(COMPENSATORY_GENES)).astype(int)
results_df["is_primary_DR_gene"] = (results_df["gene"].isin(PRIMARY_DR_GENES)).astype(int)

#flag rpoB non-RRDR variants from the pre-built set
rpob_positions = set(rpob_nonRRDR["pos"].dropna().astype(int).tolist())
results_df["is_rpoB_nonRRDR"] = (
    (results_df["gene"] == "rpoB") &
    (results_df["POS"].isin(rpob_positions))
).astype(int)

results_df = results_df.sort_values("padj_BH").reset_index(drop=True)

print(f"  Annotated {len(results_df):,} variants")
sig_df = results_df[results_df["significant_005"] == 1]
print(f"  Significant (FDR<0.05): {len(sig_df):,}")
print(f"  Enriched in MDR       : {(sig_df['enriched_in']=='MDR').sum():,}")
print(f"  In compensatory genes : {sig_df['is_compensatory_gene'].sum():,}")
print(f"  In primary DR genes   : {sig_df['is_primary_DR_gene'].sum():,}")
print(f"  rpoB non-RRDR         : {sig_df['is_rpoB_nonRRDR'].sum():,}")

# 6 fisher's test on pre-built compensatory enrichment table
#    Uses the 2x2 counts already built in Step 1
#    Fixes the infinity enrichment_ratio problem with proper p-values
print("\n[6/8] Fisher's test on compensatory mutation table...")

comp_pvals = []
comp_odds  = []
for _, row in comp_enrich.iterrows():
    a = int(row["mdr_with_mutation"])
    b = int(row["mdr_without_mutation"])
    c = int(row["non_mdr_with_mutation"])
    d = int(row["non_mdr_without_mutation"])
    if a == 0 and c == 0:
        comp_pvals.append(1.0)
        comp_odds.append(1.0)
    else:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        comp_pvals.append(p)
        comp_odds.append(odds if np.isfinite(odds) else (999.0 if a > 0 else 0.001))

comp_enrich = comp_enrich.copy()
comp_enrich["pvalue"]       = comp_pvals
comp_enrich["padj_BH"]      = false_discovery_control(comp_pvals, method="bh")
comp_enrich["odds_ratio_fisher"] = comp_odds
comp_enrich["significant_005"]   = (comp_enrich["padj_BH"] < 0.05).astype(int)

# Replace inflated enrichment_ratio with log2 odds ratio
comp_enrich["log2_odds_ratio"] = np.log2(
    np.clip(comp_enrich["odds_ratio_fisher"], 1e-6, 1e6)
)

comp_enrich = comp_enrich.sort_values("padj_BH").reset_index(drop=True)

n_comp_sig = comp_enrich["significant_005"].sum()
print(f"  Significant comp mutations (FDR<0.05): {n_comp_sig:,} / {len(comp_enrich):,}")
print(f"  Top 10 by significance:")
top_cols = ["gene","change","drug_context","mdr_frequency",
            "non_mdr_frequency","odds_ratio_fisher","padj_BH"]
print(comp_enrich[comp_enrich["significant_005"]==1].head(10)[top_cols].to_string(index=False))

# 7. Known AMR mutation summary from dr_variants.csv
print("\n[7/8] Summarising known AMR mutations...")
dr = pd.read_csv(os.path.join(TB_OUT, "dr_variants.csv"))

# Restrict to aligned samples only
aligned_samples = set(labels["sample_id"])
dr = dr[dr["sample_id"].isin(aligned_samples)].copy()

mdr_set  = set(labels[labels["MDR"] == 1]["sample_id"])
susc_set = set(labels[labels["MDR"] == 0]["sample_id"])

amr_rows = []
for (gene, change, drug), grp in dr.groupby(["gene", "change", "drug"]):
    samps  = set(grp["sample_id"].unique())
    mdr_w  = len(samps & mdr_set)
    susc_w = len(samps & susc_set)

    #F: safe mode extraction handles NaN and empty mode results
    conf_series = grp["confidence"].dropna()
    if len(conf_series) > 0:
        mode_result = conf_series.mode()
        conf = mode_result.iloc[0] if len(mode_result) > 0 else ""
    else:
        conf = ""

    #F: safe is_RRDR check
    is_rrdr = int(grp["is_RRDR"].any()) if "is_RRDR" in grp.columns else 0

    amr_rows.append({
        "gene"           : gene,
        "change"         : change,
        "drug"           : drug,
        "who_confidence" : conf,
        "is_RRDR"        : is_rrdr,
        "total_samples"  : len(samps),
        "mdr_samples"    : mdr_w,
        "susc_samples"   : susc_w,
        "overall_freq"   : round(len(samps) / len(labels), 4),
        "mdr_freq"       : round(mdr_w / len(mdr_set), 4),
        "susc_freq"      : round(susc_w / len(susc_set), 4),
    })

amr_df = pd.DataFrame(amr_rows).sort_values(
    "total_samples", ascending=False
).reset_index(drop=True)

print(f"  Known AMR mutation-drug pairs: {len(amr_df):,}")
print(f"  Top 10 most frequent:")
print(amr_df.head(10)[["gene","change","drug",
    "total_samples","mdr_freq","susc_freq","who_confidence"]].to_string(index=False))

# Gene-level enrichment summary
if len(sig_df) > 0 and "gene" in sig_df.columns and sig_df["gene"].ne("unannotated").any():
    gene_summary = (sig_df.groupby("gene")
        .agg(
            n_significant_variants = ("variant_id",    "count"),
            median_odds_ratio      = ("odds_ratio",    "median"),
            median_mdr_freq        = ("mdr_frequency", "median"),
            median_susc_freq       = ("susc_frequency","median"),
            n_enriched_MDR         = ("enriched_in",
                                      lambda x: (x=="MDR").sum()),
        )
        .sort_values("n_significant_variants", ascending=False)
        .reset_index()
    )
    gene_summary["is_compensatory"] = gene_summary["gene"].isin(
        COMPENSATORY_GENES).astype(int)
    gene_summary["is_primary_DR"]   = gene_summary["gene"].isin(
        PRIMARY_DR_GENES).astype(int)
else:
    # GFF not available — build gene summary from compensatory results only
    gene_summary = comp_enrich[comp_enrich["significant_005"]==1].groupby("gene").agg(
        n_significant_mutations = ("change", "count"),
        median_mdr_freq         = ("mdr_frequency", "median"),
        median_non_mdr_freq     = ("non_mdr_frequency", "median"),
        median_odds_ratio       = ("odds_ratio_fisher", "median"),
    ).reset_index()
    gene_summary["is_compensatory"] = 1
    print(f"  Gene summary built from compensatory table (GFF not available)")



# 8. Save all outputs
print("\n[8/8] Saving outputs...")

results_df.to_csv(
    os.path.join(ML_OUT, "obj1_fisher_all_variants.csv"), index=False)
print(f"  obj1_fisher_all_variants.csv     ({len(results_df):,} variants)")

sig_df.to_csv(
    os.path.join(ML_OUT, "obj1_fisher_significant.csv"), index=False)
print(f"  obj1_fisher_significant.csv      ({len(sig_df):,} FDR<0.05)")

comp_sig = sig_df[sig_df["is_compensatory_gene"] == 1]
comp_sig.to_csv(
    os.path.join(ML_OUT, "obj1_fisher_compensatory.csv"), index=False)
print(f"  obj1_fisher_compensatory.csv     ({len(comp_sig):,} comp gene variants)")

rpob_sig = sig_df[sig_df["is_rpoB_nonRRDR"] == 1]
rpob_sig.to_csv(
    os.path.join(ML_OUT, "obj1_fisher_rpoB_nonRRDR.csv"), index=False)
print(f"  obj1_fisher_rpoB_nonRRDR.csv     ({len(rpob_sig):,} rpoB non-RRDR sig)")

comp_enrich.to_csv(
    os.path.join(ML_OUT, "obj1_compensatory_fisher.csv"), index=False)
print(f"  obj1_compensatory_fisher.csv     ({len(comp_enrich):,} mutations, "
      f"{n_comp_sig} significant)")

amr_df.to_csv(
    os.path.join(ML_OUT, "obj1_known_amr_summary.csv"), index=False)
print(f"  obj1_known_amr_summary.csv       ({len(amr_df):,} AMR records)")

if len(gene_summary) > 0:
    gene_summary.to_csv(
        os.path.join(ML_OUT, "obj1_gene_enrichment_summary.csv"), index=False)
    print(f"  obj1_gene_enrichment_summary.csv ({len(gene_summary):,} genes)")

print("\n" + "=" * 65)
print("STEP 2 COMPLETE")
print(f"  Variants tested            : {len(results_df):,}")
print(f"  Significant (FDR<0.05)     : {len(sig_df):,}")
print(f"  In compensatory genes      : {len(comp_sig):,}")
print(f"  rpoB non-RRDR significant  : {len(rpob_sig):,}")
print(f"  Comp mutations significant : {n_comp_sig:,} / {len(comp_enrich):,}")
print(f"  Known AMR pairs            : {len(amr_df):,}")

if not USE_GFF:
    print()
    print("  !!Gene annotation incomplete — download GFF and rerun")
    print("  See download command printed in step [2/8]")