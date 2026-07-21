"""
step3_amr_summary
Objective 1 - known AMR Mutation Summary Analysis

What this step does:
  1. Builds a complete catalogue of all 547 known AMR mutation-drug pairs
  2. Stratifies each mutation by DR type (MDR, Pre-XDR, XDR, Susceptible)
  3. Identifies the most clinically significant mutations by frequency
  4. Analyses rpoB RRDR vs non-RRDR mutation distribution
  5. Builds per-drug resistance profiles across the dataset
  6. Computes co-occurrence patterns of key resistance mutations
  7. Produces publication-ready summary tables for Objective 1

Inputs (from step1_dataset/ and step2_comparative/):
  dr_variants.csv                - 14,734 DR variant records from TB-Profiler
  obj1_known_amr_summary.csv     - 547 mutation-drug pairs from Step 2
  y_labels.csv                   - 1857 samples with MDR/XDR/lineage labels

Outputs (to step3_amr_summary/):
  obj1_amr_catalogue.csv         - full annotated catalogue, all 547 pairs
  obj1_amr_drtype_stratified.csv - mutation frequency by DR type (MDR/XDR/etc)
  obj1_amr_top_mutations.csv     - top mutations by frequency, confidence filtered
  obj1_rpoB_profile.csv          - complete rpoB mutation profile (RRDR + non-RRDR)
  obj1_drug_resistance_profile.csv - per-drug resistance summary across dataset
  obj1_mutation_cooccurrence.csv - co-occurrence of key mutation pairs
  obj1_amr_gene_summary.csv      - gene-level AMR summary table
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats import false_discovery_control
import warnings
warnings.filterwarnings('ignore')

HOME   = os.path.expanduser("~")
BASE   = os.path.join(HOME, "tb_pipeline")
TB_OUT = os.path.join(BASE, "tbprofiler_results")
ML_OUT = os.path.join(BASE, "ml_outputs")

STEP1  = os.path.join(ML_OUT, "step1_dataset")
STEP2  = os.path.join(ML_OUT, "step2_comparative")
STEP3  = os.path.join(ML_OUT, "step3_amr_summary")
os.makedirs(STEP3, exist_ok=True)

#WHO confidence level that counts as resistance
RESISTANT_CONF = {"Assoc w R", "Assoc w R - Interim"}

#DR type hierarchy for ordering
DRTYPE_ORDER = ["Susceptible", "HR-TB", "RR-TB", "MDR-TB", "Pre-XDR-TB", "XDR-TB", "Other"]

#drugs in clinical priority order
DRUG_ORDER = [
    "rifampicin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin", "levofloxacin", "bedaquiline", "linezolid", "delamanid", "pretomanid",
    "amikacin", "kanamycin", "capreomycin", "streptomycin", "ethionamide", "prothionamide", "clofazimine", "cycloserine", "para-aminosalicylic_acid", "rifapentine"
]

print("STEP 3: Known AMR Mutation Summary (Objective 1)")

#input loading 
print("\n[1/8] Loading input files...")

dr = pd.read_csv(os.path.join(TB_OUT, "dr_variants.csv"))
amr = pd.read_csv(os.path.join(STEP2, "obj1_known_amr_summary.csv"))
labels = pd.read_csv(os.path.join(STEP1, "y_labels.csv"))

#restricting dr_variants to the aligned 1857 samples only 
aligned_samples = set(labels["sample_id"])
dr = dr[dr["sample_id"].isin(aligned_samples)].copy()

#build sample -> drtype lookup
sample_drtype   = labels.set_index("sample_id")["drtype"].to_dict()
sample_mdr      = labels.set_index("sample_id")["MDR"].to_dict()
sample_pre_xdr  = labels.set_index("sample_id")["pre_XDR"].to_dict()
sample_xdr      = labels.set_index("sample_id")["XDR"].to_dict()
sample_lineage  = labels.set_index("sample_id")["main_lineage"].to_dict()

#group counts per DR type
drtype_counts = labels["drtype"].value_counts().to_dict()
mdr_n         = labels["MDR"].sum()
non_mdr_n     = (labels["MDR"] == 0).sum()
total_n       = len(labels)

print(f"  dr_variants.csv  : {len(dr):,} records | {dr['sample_id'].nunique()} samples")
print(f"  amr_summary.csv  : {len(amr):,} mutation-drug pairs")
print(f"  y_labels.csv     : {len(labels):,} samples")
print(f"  DR type breakdown:")
for dt in DRTYPE_ORDER:
    cnt = drtype_counts.get(dt, 0)
    if cnt > 0:
        print(f"    {dt:<15}: {cnt:>4} ({cnt/total_n*100:.1f}%)")


# STEP [2/8]: annotated AMR catalogue
#full catalogue of all 547 mutation-drug pairs with enrichment stats
print("\n[2/8] Building annotated AMR catalogue...")

catalogue_rows = []
mdr_set  = set(labels[labels["MDR"] == 1]["sample_id"])
susc_set = set(labels[labels["MDR"] == 0]["sample_id"])

for _, row in amr.iterrows():
    gene   = row["gene"]
    change = row["change"]
    drug   = row["drug"]

    #get all samples carrying this specific mutation
    mask = (
        (dr["gene"] == gene) &
        (dr["change"] == change) &
        (dr["drug"] == drug)
    )
    carriers = set(dr[mask]["sample_id"].unique())

    #2x2 for Fisher test
    a = len(carriers & mdr_set)
    b = mdr_n - a
    c = len(carriers & susc_set)
    d = non_mdr_n - c

    if a + c > 0:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        odds = min(odds, 999.0) if np.isfinite(odds) else 999.0
    else:
        odds, p = 1.0, 1.0

    #count carriers per drtype
    carrier_drtypes = {s: sample_drtype.get(s, "Unknown") for s in carriers}
    drtype_carrier_counts = {}
    for dt in DRTYPE_ORDER:
        drtype_carrier_counts[f"n_{dt.replace('-','_').replace(' ','_')}"] = \
            sum(1 for v in carrier_drtypes.values() if v == dt)

    #allele frequency stats from dr_variants
    grp = dr[mask]
    avg_freq  = grp["freq"].mean()  if len(grp) > 0 else None
    avg_depth = grp["depth"].mean() if len(grp) > 0 else None

    catalogue_rows.append({
        "gene"              : gene,
        "change"            : change,
        "drug"              : drug,
        "who_confidence"    : row["who_confidence"],
        "is_RRDR"           : row["is_RRDR"],
        "total_samples"     : len(carriers),
        "overall_freq"      : round(len(carriers) / total_n, 4),
        "mdr_samples"       : a,
        "susc_samples"      : c,
        "mdr_freq"          : round(a / mdr_n, 4),
        "susc_freq"         : round(c / non_mdr_n, 4),
        "odds_ratio"        : round(odds, 3),
        "pvalue"            : p,
        "avg_allele_freq"   : round(avg_freq, 4) if avg_freq else None,
        "avg_read_depth"    : round(avg_depth, 1) if avg_depth else None,
        **drtype_carrier_counts,
    })

catalogue_df = pd.DataFrame(catalogue_rows)
#add FDR correction
catalogue_df["padj_BH"] = false_discovery_control(
    catalogue_df["pvalue"].values, method="bh"
)
catalogue_df["significant_005"] = (catalogue_df["padj_BH"] < 0.05).astype(int)
catalogue_df = catalogue_df.sort_values("total_samples", ascending=False).reset_index(drop=True)

print(f"  Catalogue built: {len(catalogue_df):,} mutation-drug pairs")
print(f"  Significant (FDR<0.05): {catalogue_df['significant_005'].sum():,}")
print(f"  WHO Assoc w R: {(catalogue_df['who_confidence']=='Assoc w R').sum():,}")

#STEP [3/8]: DR TYPE STRATIFICATION
#how does each mutation distribute across DR types?
print("\n[3/8] DR type stratification of AMR mutations...")

drtype_rows = []
for _, row in amr.iterrows():
    gene   = row["gene"]
    change = row["change"]
    drug   = row["drug"]

    mask = (
        (dr["gene"] == gene) &
        (dr["change"] == change) &
        (dr["drug"] == drug)
    )
    grp = dr[mask]
    carriers = set(grp["sample_id"].unique())

    row_data = {
        "gene"           : gene,
        "change"         : change,
        "drug"           : drug,
        "who_confidence" : row["who_confidence"],
        "total_samples"  : len(carriers),
        "overall_freq"   : round(len(carriers) / total_n, 4),
    }

    # Frequency within each DR type
    for dt in DRTYPE_ORDER:
        dt_samples = set(labels[labels["drtype"] == dt]["sample_id"])
        dt_n       = len(dt_samples)
        dt_carriers = len(carriers & dt_samples)
        col_key = dt.replace("-","_").replace(" ","_")
        row_data[f"n_{col_key}"]    = dt_carriers
        row_data[f"freq_{col_key}"] = round(dt_carriers / dt_n, 4) if dt_n > 0 else 0.0

    drtype_rows.append(row_data)

drtype_df = pd.DataFrame(drtype_rows).sort_values(
    "total_samples", ascending=False
).reset_index(drop=True)

print(f"  DR type stratification built: {len(drtype_df):,} rows")

# STEP [4/8]: TOP MUTATIONS - Assoc w R only
#clinically most important mutations - confirmed resistance associations
print("\n[4/8] Building top confirmed AMR mutations table...")

top_df = catalogue_df[
    catalogue_df["who_confidence"].isin(RESISTANT_CONF)
].copy()
top_df = top_df.sort_values("total_samples", ascending=False).reset_index(drop=True)

print(f"  Confirmed resistance mutations (Assoc w R + Interim): {len(top_df):,}")
print(f"  Top 15 by sample count:")
print(top_df[["gene","change","drug","who_confidence","total_samples", "mdr_freq","susc_freq","odds_ratio"]].head(15).to_string(index=False))

# STEP [5/8]: rpoB PROFILE
#complete rpoB mutation analysis - RRDR vs non-RRDR
print("\n[5/8] Building rpoB mutation profile...")

rpob_dr    = dr[dr["gene"] == "rpoB"].copy()
rpob_amr   = amr[amr["gene"] == "rpoB"].copy()

rpob_rows = []
for _, row in rpob_amr.iterrows():
    change = row["change"]
    drug   = row["drug"]

    mask = (
        (dr["gene"] == "rpoB") &
        (dr["change"] == change) &
        (dr["drug"] == drug)
    )
    carriers = set(dr[mask]["sample_id"].unique())
    a = len(carriers & mdr_set)
    c = len(carriers & susc_set)

    rpob_rows.append({
        "change"         : change,
        "drug"           : drug,
        "who_confidence" : row["who_confidence"],
        "is_RRDR"        : row["is_RRDR"],
        "total_samples"  : len(carriers),
        "overall_freq"   : round(len(carriers) / total_n, 4),
        "mdr_freq"       : round(a / mdr_n, 4),
        "susc_freq"      : round(c / non_mdr_n, 4),
        "avg_allele_freq": round(dr[mask]["freq"].mean(), 4) if len(dr[mask]) > 0 else None,
    })

rpob_df = pd.DataFrame(rpob_rows).sort_values(
    ["is_RRDR","total_samples"], ascending=[False, False]
).reset_index(drop=True)

rrdr_muts    = rpob_df[rpob_df["is_RRDR"] == 1]
non_rrdr_mut = rpob_df[rpob_df["is_RRDR"] == 0]

print(f"  rpoB RRDR mutations      : {len(rrdr_muts):,} unique mutation-drug pairs")
print(f"  rpoB non-RRDR mutations  : {len(non_rrdr_mut):,} unique mutation-drug pairs")
print(f"  Top 5 RRDR mutations:")
print(rrdr_muts[["change","total_samples","mdr_freq","susc_freq","who_confidence"]].head(5).to_string(index=False))

# STEP [6/8]: PER-DRUG RESISTANCE PROFILE
print("\n[6/8] Building per-drug resistance profile...")

drug_rows = []
for drug in DRUG_ORDER:
    # All confirmed resistance mutations for this drug
    drug_mask = (
        (dr["drug"] == drug) &
        (dr["confidence"].isin(RESISTANT_CONF))
    )
    drug_carriers = set(dr[drug_mask]["sample_id"].unique())

    # From labels direct drug column
    drug_col = drug.replace("-","_").replace(" ","_")
    if drug_col in labels.columns:
        label_resistant = (labels[drug_col] == "R").sum()
        label_mdr_resistant = labels[
            (labels[drug_col] == "R") & (labels["MDR"] == 1)
        ].shape[0]
    else:
        label_resistant = len(drug_carriers)
        label_mdr_resistant = len(drug_carriers & mdr_set)

    # Count unique mutations
    drug_muts = dr[drug_mask].groupby(["gene","change"])["sample_id"].nunique()
    n_unique_muts = len(drug_muts)
    top_gene = dr[drug_mask]["gene"].value_counts().index[0] if len(dr[drug_mask]) > 0 else "N/A"

    drug_rows.append({
        "drug"                   : drug,
        "resistant_samples"      : label_resistant,
        "resistant_pct"          : round(label_resistant / total_n * 100, 1),
        "resistant_in_MDR"       : label_mdr_resistant,
        "resistant_pct_of_MDR"   : round(label_mdr_resistant / mdr_n * 100, 1),
        "n_unique_mutations"     : n_unique_muts,
        "top_gene"               : top_gene,
    })

drug_df = pd.DataFrame(drug_rows)
print(f"  Per-drug profile built: {len(drug_df)} drugs")
print(f"  Drug resistance rates:")
print(drug_df[["drug","resistant_samples","resistant_pct",
               "n_unique_mutations","top_gene"]].to_string(index=False))

# STEP [7/8]: MUTATION CO-OCCURRENCE
#which primary resistance mutations co-occur most frequently in MDR samples?
print("\n[7/8] Computing mutation co-occurrence...")

# Key mutations to check co-occurrence for
KEY_MUTS = {
    "katG_S315T" : (dr["gene"]=="katG") & (dr["change"].str.contains("Ser315|S315", na=False)),
    "rpoB_S450L" : (dr["gene"]=="rpoB") & (dr["change"].str.contains("Ser450|S450", na=False)),
    "embB_M306"  : (dr["gene"]=="embB") & (dr["change"].str.contains("Met306|M306", na=False)),
    "gyrA_A90V"  : (dr["gene"]=="gyrA") & (dr["change"].str.contains("Ala90|A90", na=False)),
    "rrs_1401"   : (dr["gene"]=="rrs")  & (dr["pos"]==1401),
    "inhA_prom"  : (dr["gene"]=="inhA") & (dr["change"].str.contains("c.-", na=False)),
}

# Sample sets for each key mutation
mut_samples = {}
for name, mask in KEY_MUTS.items():
    mut_samples[name] = set(dr[mask]["sample_id"].unique())
    print(f"  {name:<18}: {len(mut_samples[name]):>4} samples "
          f"({len(mut_samples[name])/total_n*100:.1f}% overall)")

# Build co-occurrence matrix
print()
co_rows = []
names = list(mut_samples.keys())
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if j <= i:
            continue
        overlap = mut_samples[n1] & mut_samples[n2]
        union   = mut_samples[n1] | mut_samples[n2]
        jaccard = len(overlap) / len(union) if union else 0
        co_rows.append({
            "mutation_1"       : n1,
            "mutation_2"       : n2,
            "samples_with_both": len(overlap),
            "samples_mut1"     : len(mut_samples[n1]),
            "samples_mut2"     : len(mut_samples[n2]),
            "pct_of_mut1"      : round(len(overlap) / len(mut_samples[n1]) * 100, 1) if mut_samples[n1] else 0,
            "pct_of_mut2"      : round(len(overlap) / len(mut_samples[n2]) * 100, 1) if mut_samples[n2] else 0,
            "jaccard_index"    : round(jaccard, 3),
        })

cooccur_df = pd.DataFrame(co_rows).sort_values(
    "samples_with_both", ascending=False
).reset_index(drop=True)

print("  Top 10 co-occurring mutation pairs:")
print(cooccur_df[["mutation_1","mutation_2","samples_with_both", "pct_of_mut1","jaccard_index"]].head(10).to_string(index=False))

# STEP [8/8]: GENE-LEVEL SUMMARY + SAVE ALL OUTPUTS
print("\n[8/8] Building gene summary and saving outputs...")

# Gene-level AMR summary
gene_summary = (catalogue_df.groupby("gene")
    .agg(
        n_mutation_drug_pairs  = ("change",          "count"),
        n_unique_mutations     = ("change",          "nunique"),
        n_drugs                = ("drug",            "nunique"),
        total_unique_samples   = ("total_samples",   "max"),
        max_mdr_freq           = ("mdr_freq",        "max"),
        max_susc_freq          = ("susc_freq",       "max"),
        n_assoc_w_R            = ("who_confidence",
                                  lambda x: x.isin(RESISTANT_CONF).sum()),
        n_uncertain            = ("who_confidence",
                                  lambda x: (x == "Uncertain significance").sum()),
        n_RRDR                 = ("is_RRDR",         "sum"),
        n_significant_FDR      = ("significant_005", "sum"),
    )
    .sort_values("total_unique_samples", ascending=False)
    .reset_index()
)

print(f"  Gene summary: {len(gene_summary)} genes")
print(f"  Top 10 genes by unique samples affected:")
print(gene_summary[["gene","n_unique_mutations","n_drugs", "max_mdr_freq","n_assoc_w_R"]].head(10).to_string(index=False))

catalogue_df.to_csv(
    os.path.join(STEP3, "obj1_amr_catalogue.csv"), index=False)
print(f"\n  obj1_amr_catalogue.csv           ({len(catalogue_df):,} rows)")

drtype_df.to_csv(
    os.path.join(STEP3, "obj1_amr_drtype_stratified.csv"), index=False)
print(f"  obj1_amr_drtype_stratified.csv    ({len(drtype_df):,} rows)")

top_df.to_csv(
    os.path.join(STEP3, "obj1_amr_top_mutations.csv"), index=False)
print(f"  obj1_amr_top_mutations.csv        ({len(top_df):,} rows)")

rpob_df.to_csv(
    os.path.join(STEP3, "obj1_rpoB_profile.csv"), index=False)
print(f"  obj1_rpoB_profile.csv             ({len(rpob_df):,} rows)")

drug_df.to_csv(
    os.path.join(STEP3, "obj1_drug_resistance_profile.csv"), index=False)
print(f"  obj1_drug_resistance_profile.csv  ({len(drug_df):,} rows)")

cooccur_df.to_csv(
    os.path.join(STEP3, "obj1_mutation_cooccurrence.csv"), index=False)
print(f"  obj1_mutation_cooccurrence.csv    ({len(cooccur_df):,} rows)")

gene_summary.to_csv(
    os.path.join(STEP3, "obj1_amr_gene_summary.csv"), index=False)
print(f"  obj1_amr_gene_summary.csv         ({len(gene_summary):,} rows)")

print("\n" + "=" * 65)
print("STEP 3 COMPLETE")
print(f"  AMR mutation-drug pairs     : {len(catalogue_df):,}")
print(f"  Confirmed (Assoc w R)       : {len(top_df):,}")
print(f"  Significant (FDR<0.05)      : {catalogue_df['significant_005'].sum():,}")
print(f"  rpoB RRDR mutations         : {len(rrdr_muts):,}")
print(f"  rpoB non-RRDR mutations     : {len(non_rrdr_mut):,}")
print(f"  Drugs profiled              : {len(drug_df)}")
print(f"  Genes with AMR mutations    : {len(gene_summary)}")
print(f"  Outputs in: {STEP3}/")