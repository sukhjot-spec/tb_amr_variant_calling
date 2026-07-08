#!/usr/bin/env python3
"""
collate_tbprofiler.py
Collates all TB-Profiler JSON result files into structured CSV files.

Outputs:
  labels.csv          - per-sample drug resistance labels + MDR flag + lineage
  dr_variants.csv     - all drug resistance variants across all samples
  other_variants.csv  - all other variants including compensatory candidates
  compensatory.csv    - candidate compensatory mutations (rpoA, rpoC, rpsA etc.)
  summary_stats.txt   - dataset-level statistics

mamba activate tb_amr
python3 ~/tb_pipeline/py_scripts/collate_tbprofiler.py
"""

import json
import os
import glob
import pandas as pd

#defining paths
HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "tb_pipeline")
RESULTS_DIR = os.path.join(BASE, "tbprofiler_results", "results")
OUT_DIR = os.path.join(BASE, "tbprofiler_results")
os.makedirs(OUT_DIR, exist_ok=True)

#known compensatory mutation genes for MDR-TB 
COMPENSATORY_GENES = {
    "rpoA", "rpoC",   # compensate rpoB fitness cost (rifampicin resistance)
    "rpsA", "rpsL",   # compensate pyrazinamide/streptomycin resistance
    "gyrB",           # compensates gyrA fluoroquinolone resistance
    "tlyA",           # compensates aminoglycoside resistance
    "atpE",           # compensates bedaquiline resistance
    "pepQ",           # compensates bedaquiline/clofazimine resistance
    "Rv0678", "mmpL5" # compensate bedaquiline/clofazimine resistance
}

#all drugs TB-Profiler reports
ALL_DRUGS = [
    "rifampicin", "rifapentine", "isoniazid", "ethambutol", "pyrazinamide",
    "moxifloxacin", "levofloxacin", "bedaquiline", "delamanid", "pretomanid",
    "linezolid", "streptomycin", "amikacin", "kanamycin", "capreomycin",
    "clofazimine", "ethionamide", "prothionamide", "para-aminosalicylic_acid",
    "cycloserine"
]

#loading all JSON files
json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.results.json")))
print(f"Found {len(json_files)} JSON result files in {RESULTS_DIR}")

if len(json_files) == 0:
    print(f"ERROR: No JSON files found at {RESULTS_DIR}/*.results.json")
    exit(1)

#parsing all JSONs
labels_rows = []
dr_variant_rows = []
other_variant_rows = []
compensatory_rows = []
failed_files = []

print("Parsing JSON files...")
for i, jf in enumerate(json_files):
    try:
        with open(jf) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not parse {jf}: {e}")
        failed_files.append(jf)
        continue

    sample_id = d.get("id", os.path.basename(jf).replace(".results.json", ""))
    drtype = d.get("drtype", "Unknown")
    main_lineage = d.get("main_lineage", "")
    sub_lineage = d.get("sub_lineage", "")
    total_variants = d.get("qc", {}).get("total_variants", None)

    #drug resistance labels
    drug_status = {drug: "S" for drug in ALL_DRUGS}

    for variant in d.get("dr_variants", []):
        for ann in variant.get("annotation", []):
            drug = ann.get("drug", "")
            conf = ann.get("confidence", "")
            if drug in drug_status and "Assoc w R" in conf:
                drug_status[drug] = "R"

    drug_binary = {f"{drug}_binary": (1 if drug_status[drug] == "R" else 0) for drug in ALL_DRUGS}

    # MDR = resistant to both isoniazid AND rifampicin
    mdr = int(drug_status.get("isoniazid") == "R" and drug_status.get("rifampicin") == "R")

    # Pre-XDR = MDR + resistant to any fluoroquinolone
    pre_xdr = int(mdr == 1 and
                  (drug_status.get("moxifloxacin") == "R" or
                   drug_status.get("levofloxacin") == "R"))

    # XDR = Pre-XDR + resistant to bedaquiline or linezolid
    xdr = int(pre_xdr == 1 and
              (drug_status.get("bedaquiline") == "R" or
               drug_status.get("linezolid") == "R"))

    label_row = {
        "sample_id": sample_id,
        "drtype": drtype,
        "main_lineage": main_lineage,
        "sub_lineage": sub_lineage,
        "MDR": mdr,
        "pre_XDR": pre_xdr,
        "XDR": xdr,
        "total_variants_qc": total_variants,
    }
    label_row.update(drug_status)
    label_row.update(drug_binary)
    labels_rows.append(label_row)

    #DR variants 
    for variant in d.get("dr_variants", []):
        for ann in variant.get("annotation", []):
            dr_variant_rows.append({
                "sample_id": sample_id,
                "gene": variant.get("gene_name", ""),
                "locus_tag": variant.get("locus_tag", ""),
                "change": variant.get("change", ""),
                "nucleotide_change": variant.get("nucleotide_change", ""),
                "type": variant.get("type", ""),
                "drug": ann.get("drug", ""),
                "confidence": ann.get("confidence", ""),
                "depth": variant.get("depth", None),
                "freq": variant.get("freq", None),
                "pos": variant.get("pos", None),
            })

    #Other variants (compensatory candidates) 
    for variant in d.get("other_variants", []):
        gene = variant.get("gene_name", "")
        is_compensatory = gene in COMPENSATORY_GENES
        drugs_annotated = []
        confidences = []
        for ann in variant.get("annotation", []):
            drugs_annotated.append(ann.get("drug", ""))
            confidences.append(ann.get("confidence", ""))

        other_row = {
            "sample_id": sample_id,
            "gene": gene,
            "locus_tag": variant.get("locus_tag", ""),
            "change": variant.get("change", ""),
            "nucleotide_change": variant.get("nucleotide_change", ""),
            "type": variant.get("type", ""),
            "depth": variant.get("depth", None),
            "freq": variant.get("freq", None),
            "pos": variant.get("pos", None),
            "is_compensatory_candidate": is_compensatory,
            "sample_MDR": mdr,
            "drugs_annotated": "|".join(drugs_annotated),
            "confidences": "|".join(confidences),
        }
        other_variant_rows.append(other_row)

        if is_compensatory:
            compensatory_rows.append(other_row.copy())

    if (i + 1) % 100 == 0:
        print(f"  Parsed {i+1} / {len(json_files)}")

print(f"Parsing complete. {len(failed_files)} files failed.")

#building dataframes and saving
df_labels = pd.DataFrame(labels_rows)
df_dr = pd.DataFrame(dr_variant_rows) if dr_variant_rows else pd.DataFrame()
df_other = pd.DataFrame(other_variant_rows) if other_variant_rows else pd.DataFrame()
df_comp = pd.DataFrame(compensatory_rows) if compensatory_rows else pd.DataFrame()

df_labels.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
print(f"Saved: labels.csv ({len(df_labels)} samples)")

if not df_dr.empty:
    df_dr.to_csv(os.path.join(OUT_DIR, "dr_variants.csv"), index=False)
    print(f"Saved: dr_variants.csv ({len(df_dr)} records)")

if not df_other.empty:
    df_other.to_csv(os.path.join(OUT_DIR, "other_variants.csv"), index=False)
    print(f"Saved: other_variants.csv ({len(df_other)} records)")

if not df_comp.empty:
    df_comp.to_csv(os.path.join(OUT_DIR, "compensatory.csv"), index=False)
    print(f"Saved: compensatory.csv ({len(df_comp)} records)")

#summary stats 
with open(os.path.join(OUT_DIR, "summary_stats.txt"), "w") as f:
    f.write("TB-Profiler Collation Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total samples: {len(df_labels)}\n")
    f.write(f"Failed files:  {len(failed_files)}\n\n")
    f.write("Drug Resistance Classification\n")
    for drtype, count in df_labels["drtype"].value_counts().items():
        f.write(f"  {drtype:<30}: {count:>5} ({count/len(df_labels)*100:.1f}%)\n")
    f.write(f"\nMDR/XDR\n")
    f.write(f"  MDR:     {df_labels['MDR'].sum():>5} ({df_labels['MDR'].mean()*100:.1f}%)\n")
    f.write(f"  Pre-XDR: {df_labels['pre_XDR'].sum():>5} ({df_labels['pre_XDR'].mean()*100:.1f}%)\n")
    f.write(f"  XDR:     {df_labels['XDR'].sum():>5} ({df_labels['XDR'].mean()*100:.1f}%)\n")
    f.write(f"\nLineage Distribution\n")
    for lin, count in df_labels["main_lineage"].value_counts().items():
        if lin:
            f.write(f"  {lin:<30}: {count:>5} ({count/len(df_labels)*100:.1f}%)\n")
    f.write(f"\nPer-Drug Resistance\n")
    for drug in ALL_DRUGS:
        if drug in df_labels.columns:
            r = (df_labels[drug] == "R").sum()
            f.write(f"  {drug:<35}: {r:>5} ({r/len(df_labels)*100:.1f}%)\n")

print("Saved: summary_stats.txt")
print("\n" + "=" * 50)
print(f"DONE — {len(df_labels)} samples collated")
print(f"MDR: {df_labels['MDR'].sum()} ({df_labels['MDR'].mean()*100:.1f}%)")
print(f"Files saved to: {OUT_DIR}")
print("=" * 50)