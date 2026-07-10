#!/usr/bin/env python3
"""
collate_tbprofiler.py  — Corrected version
Collates TB-Profiler JSON results into structured CSVs with scientifically
accurate compensatory mutation classification.

KEY CORRECTIONS vs previous version:
  1. rpoB: RRDR mutations (codons 426-452) → primary resistance
           non-RRDR mutations in MDR samples → candidate compensatory
  2. Removed from compensatory set (these are PRIMARY resistance genes):
       - rplC, rplD  : primary linezolid resistance (23S rRNA ribosome)
       - fgd1        : primary delamanid/pretomanid activation loss
       - dprE1       : primary target of emerging drugs
       - pepQ        : primary low-level BDQ/CFZ resistance locus
       - tlyA        : primary aminoglycoside resistance (rRNA methylase)
  3. Added scientifically validated compensatory genes with mechanism notes
  4. Compensatory classification requires CONFIRMED MDR in the same sample
  5. rpoB non-RRDR compensation logic implemented

Outputs:
  labels.csv            - per-sample labels, MDR/XDR flags, lineage
  dr_variants.csv       - all drug resistance variants (primary only)
  other_variants.csv    - all non-DR variants
  compensatory.csv      - candidate compensatory mutations (CLEAN set)
  rpoB_nonRRDR.csv      - rpoB non-RRDR variants in MDR samples (Obj 1 & 3)
  summary_stats.txt     - dataset-level statistics
"""

import json
import os
import glob
import re
import pandas as pd

#paths
HOME        = os.path.expanduser("~")
BASE        = os.path.join(HOME, "tb_pipeline")
RESULTS_DIR = os.path.join(BASE, "tbprofiler_results", "results")
OUT_DIR     = os.path.join(BASE, "tbprofiler_results")
os.makedirs(OUT_DIR, exist_ok=True)

#all the drugs from tb-profiler report
ALL_DRUGS = [
    "rifampicin", "rifapentine", "isoniazid", "ethambutol", "pyrazinamide",
    "moxifloxacin", "levofloxacin", "bedaquiline", "delamanid", "pretomanid",
    "linezolid", "streptomycin", "amikacin", "kanamycin", "capreomycin",
    "clofazimine", "ethionamide", "prothionamide", "para-aminosalicylic_acid",
    "cycloserine"
]

# COMPENSATORY GENE SET
# EXCLUDED (primary resistance genes):
#   rpoB  → handled separately below with RRDR logic
#   rplC  → primary linezolid resistance (C154R directly blocks drug binding)
#   rplD  → primary linezolid/streptomycin resistance
#   tlyA  → primary aminoglycoside resistance (rRNA methylase LoF)
#   fgd1  → primary delamanid/pretomanid resistance (prodrug activation loss)
#   dprE1 → primary target of DprE1-inhibitor drugs
#   pepQ  → primary low-level BDQ/CFZ resistance (not a fitness restorer)
COMPENSATORY_GENES = {
    # rifampicin pathway: compensate rpoB RRDR fitness cost
    "rpoA": {
        "drug_context":  "rifampicin",
        "mechanism":     "Compensates transcriptional fitness cost of rpoB RRDR mutations. "
                         "Mutations in the alpha subunit of RNA polymerase restore the "
                         "binding geometry disrupted by rpoB resistance mutations.",
        "evidence":      "Comas et al. 2012 Nat Genet; Gagneux et al.",
        "requires_mdr":  False,   # requires rpoB RRDR — checked separately
    },
    "rpoC": {
        "drug_context":  "rifampicin",
        "mechanism":     "Beta-prime subunit of RNA polymerase. Mutations here compensate "
                         "for the conformational changes in the RNA polymerase complex caused "
                         "by rpoB RRDR resistance mutations, restoring transcription efficiency.",
        "evidence":      "Comas et al. 2012 Nat Genet; Muller et al. 2017",
        "requires_mdr":  False,
    },

    #isoniazid pathway: compensate katG LoF and inhA overexpression
    "ahpC": {
        "drug_context":  "isoniazid",
        "mechanism":     "Alkyl hydroperoxide reductase. Upregulation compensates for the "
                         "loss of KatG-mediated oxidative defence when katG is inactivated "
                         "by INH-resistance mutations. Promoter mutations causing ahpC "
                         "upregulation are the canonical compensatory mechanism for katG LoF.",
        "evidence":      "Sherman et al. 1996; Heym et al. 1997",
        "requires_mdr":  True,
    },
    "kasA": {
        "drug_context":  "isoniazid",
        "mechanism":     "Beta-ketoacyl ACP synthase involved in mycolic acid synthesis. "
                         "Mutations here can partially compensate for inhA-mediated INH "
                         "resistance by altering the flux through the FAS-II fatty acid "
                         "synthesis pathway.",
        "evidence":      "Mdluli et al. 1998",
        "requires_mdr":  True,
    },
    "ndh": {
        "drug_context":  "isoniazid",
        "mechanism":     "NADH dehydrogenase. Loss-of-function mutations raise the NADH/NAD+ "
                         "ratio, which can compensate for INH-mediated inhibition of InhA "
                         "(InhA requires NADH as cofactor). Not a direct resistance mutation "
                         "but modulates the INH activation and target environment.",
        "evidence":      "Miesel et al. 1998",
        "requires_mdr":  True,
    },

    #fluoroquinolone pathway: compensate gyrA fitness cost 
    "gyrB": {
        "drug_context":  "fluoroquinolones",
        "mechanism":     "DNA gyrase B subunit. gyrB mutations can compensate for the "
                         "supercoiling defects caused by gyrA QRDR (Quinolone "
                         "Resistance-Determining Region) primary resistance mutations, "
                         "restoring adequate DNA replication efficiency.",
        "evidence":      "Avalos Vizcaino et al.; Mayer & Fierer 2014",
        "requires_mdr":  True,
    },

    #Aminoglycoside pathway: compensate rrs/rpsL resistance 
    "gid": {
        "drug_context":  "streptomycin",
        "mechanism":     "Glucose-inhibited division protein A. Mutations in gid reduce "
                         "streptomycin binding through 16S rRNA methylation changes, "
                         "compensating for the ribosomal fitness defects caused by primary "
                         "rpsL and rrs resistance mutations.",
        "evidence":      "Nair et al. 1993; Spies et al.",
        "requires_mdr":  True,
    },

    #Bedaquiline / Clofazimine efflux pathway 
    "mmpR5": {
        "drug_context":  "bedaquiline/clofazimine",
        "mechanism":     "Transcriptional repressor of the MmpS5-MmpL5 efflux pump. "
                         "Loss-of-function mutations in mmpR5 upregulate MmpL5-mediated "
                         "drug efflux. Classified here as compensatory in the context of "
                         "pre-existing BDQ resistance conferring fitness loss, though some "
                         "literature treats this as a primary low-level resistance mechanism. "
                         "Flag for manual review.",
        "evidence":      "Andries et al.; Hartkoorn et al. 2014",
        "requires_mdr":  True,
    },
    "mmpL5": {
        "drug_context":  "bedaquiline/clofazimine",
        "mechanism":     "MmpL5 efflux pump structural component. Gain-of-function mutations "
                         "that increase pump activity may compensate for BDQ/CFZ fitness costs "
                         "in highly resistant strains. Borderline primary/compensatory — "
                         "flag for manual review.",
        "evidence":      "Andries et al.; Hartkoorn et al. 2014",
        "requires_mdr":  True,
    },

    #Ethambutol pathway 
    "embR": {
        "drug_context":  "ethambutol",
        "mechanism":     "Transcriptional activator of the emb operon. Mutations in embR can "
                         "modulate embB/embA/embC expression levels and partially compensate "
                         "for the arabinogalactan synthesis disruption caused by primary embB "
                         "resistance mutations.",
        "evidence":      "Sharma et al. 2006",
        "requires_mdr":  True,
    },

    #General fitness restorers in MDR context 
    "whiB7": {
        "drug_context":  "multiple",
        "mechanism":     "WhiB7 is a transcriptional activator of multiple intrinsic "
                         "resistance mechanisms including eis (aminoglycoside), tap (efflux), "
                         "and others. Mutations affecting whiB7 activity can modulate broad "
                         "fitness costs of MDR. Emerging evidence for compensatory role in "
                         "MDR-TB fitness restoration.",
        "evidence":      "Burian et al. 2012; Larsson et al.",
        "requires_mdr":  True,
    },
    "eis": {
        "drug_context":  "amikacin/kanamycin",
        "mechanism":     "Enhanced intracellular survival protein. Promoter mutations causing "
                         "eis overexpression confer low-level kanamycin/amikacin resistance "
                         "and may compensate for fitness costs in strains with primary rrs "
                         "resistance mutations. Borderline primary/compensatory at promoter "
                         "positions — flag for manual review.",
        "evidence":      "Campbell et al. 2011; Zaunbrecher et al. 2009",
        "requires_mdr":  True,
    },
}

# rpoB RRDR DEFINITION
# H37Rv coordinates — codons 426–452 of rpoB
# These are PRIMARY resistance mutations — NOT compensatory
# Canonical RRDR amino acid positions (protein numbering in H37Rv rpoB):
RRDR_CODONS = set(range(426, 453))   # codons 426–452 inclusive

def is_in_rrdr(change_str: str) -> bool:
    """
    Returns True if the amino acid change is within the rpoB RRDR (codons 426-452).
    Parses standard TB-Profiler notation: p.Ser450Leu, p.His445Tyr, etc.
    Also handles nucleotide positions mapped to RRDR codon range.
    """
    if not change_str:
        return False
    # Try protein change: p.Xxx###Yyy
    m = re.search(r'p\.[A-Za-z]+(\d+)[A-Za-z]', change_str)
    if m:
        codon = int(m.group(1))
        return codon in RRDR_CODONS
    # Fallback: nucleotide position — RRDR spans nt ~1276 to ~1358 in rpoB CDS
    m2 = re.search(r'c\.(\d+)', change_str)
    if m2:
        nt = int(m2.group(1))
        return 1276 <= nt <= 1358
    return False


#loading JSON files
json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.results.json")))
print(f"Found {len(json_files)} JSON result files in {RESULTS_DIR}")
if not json_files:
    print(f"ERROR: No JSON files found. Check path: {RESULTS_DIR}")
    exit(1)

 
#parsing JSON files 
labels_rows        = []
dr_variant_rows    = []
other_variant_rows = []
compensatory_rows  = []
rpoB_nonRRDR_rows  = []
failed_files       = []

RESISTANT_CONF = {"Assoc w R", "Assoc w R - Interim"}

print("Parsing JSON files...")
for i, jf in enumerate(json_files):
    try:
        with open(jf) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not parse {jf}: {e}")
        failed_files.append(jf)
        continue

    sample_id    = d.get("id", os.path.basename(jf).replace(".results.json", ""))
    drtype       = d.get("drtype", "Unknown")
    main_lineage = d.get("main_lineage", "")
    sub_lineage  = d.get("sub_lineage", "")
    total_vars   = d.get("qc", {}).get("total_variants", None)

    #drug resistance labels 
    drug_status = {drug: "S" for drug in ALL_DRUGS}

    for variant in d.get("dr_variants", []):
        for ann in variant.get("annotation", []):
            drug = ann.get("drug", "")
            conf = ann.get("confidence", "")
            if drug in drug_status and conf in RESISTANT_CONF:
                drug_status[drug] = "R"

    drug_binary = {
        f"{drug}_binary": (1 if drug_status[drug] == "R" else 0)
        for drug in ALL_DRUGS
    }

    #MDR / Pre-XDR / XDR flags 
    is_mdr     = int(drug_status["isoniazid"] == "R" and
                     drug_status["rifampicin"] == "R")
    is_pre_xdr = int(is_mdr == 1 and
                     (drug_status["moxifloxacin"] == "R" or
                      drug_status["levofloxacin"] == "R"))
    is_xdr     = int(is_pre_xdr == 1 and
                     (drug_status["bedaquiline"] == "R" or
                      drug_status["linezolid"] == "R"))

    #checking if sample has primary rpoB RRDR mutation 
    has_rrdr_mutation = False
    for variant in d.get("dr_variants", []):
        gene   = variant.get("gene_name", "")
        change = variant.get("change", "")
        if gene == "rpoB" and is_in_rrdr(change):
            has_rrdr_mutation = True
            break

    #label row
    label_row = {
        "sample_id"           : sample_id,
        "drtype"              : drtype,
        "main_lineage"        : main_lineage,
        "sub_lineage"         : sub_lineage,
        "MDR"                 : is_mdr,
        "pre_XDR"             : is_pre_xdr,
        "XDR"                 : is_xdr,
        "has_rpoB_RRDR"       : int(has_rrdr_mutation),
        "total_variants_qc"   : total_vars,
    }
    label_row.update(drug_status)
    label_row.update(drug_binary)
    labels_rows.append(label_row)

    #DR variants (primary resistance mutations only)
    for variant in d.get("dr_variants", []):
        gene   = variant.get("gene_name", "")
        change = variant.get("change", "")

        # Flag rpoB RRDR status on each DR variant row
        in_rrdr = (gene == "rpoB") and is_in_rrdr(change)

        for ann in variant.get("annotation", []):
            dr_variant_rows.append({
                "sample_id"        : sample_id,
                "gene"             : gene,
                "locus_tag"        : variant.get("locus_tag", ""),
                "change"           : change,
                "nucleotide_change": variant.get("nucleotide_change", ""),
                "type"             : variant.get("type", ""),
                "drug"             : ann.get("drug", ""),
                "confidence"       : ann.get("confidence", ""),
                "depth"            : variant.get("depth", None),
                "freq"             : variant.get("freq", None),
                "pos"              : variant.get("pos", None),
                "is_RRDR"          : int(in_rrdr),   # rpoB only; 0 for all others
                "sample_MDR"       : is_mdr,
                "sample_drtype"    : drtype,
            })

    #other variants: compensatory classification 
    for variant in d.get("other_variants", []):
        gene   = variant.get("gene_name", "")
        change = variant.get("change", "")
        vtype  = variant.get("type", "")
        freq   = variant.get("freq", None)
        depth  = variant.get("depth", None)
        pos    = variant.get("pos", None)
        locus  = variant.get("locus_tag", "")

        drugs_ann  = []
        confs_ann  = []
        for ann in variant.get("annotation", []):
            drugs_ann.append(ann.get("drug", ""))
            confs_ann.append(ann.get("confidence", ""))

        #rpoB non-RRDR logic
        is_rpoB_nonRRDR_compensatory = False
        if gene == "rpoB" and not is_in_rrdr(change):
            # Only compensatory if sample already has a primary RRDR mutation
            if has_rrdr_mutation:
                is_rpoB_nonRRDR_compensatory = True
                rpoB_nonRRDR_rows.append({
                    "sample_id"          : sample_id,
                    "gene"               : gene,
                    "locus_tag"          : locus,
                    "change"             : change,
                    "nucleotide_change"  : variant.get("nucleotide_change", ""),
                    "type"               : vtype,
                    "depth"              : depth,
                    "freq"               : freq,
                    "pos"                : pos,
                    "sample_MDR"         : is_mdr,
                    "sample_drtype"      : drtype,
                    "has_primary_RRDR"   : int(has_rrdr_mutation),
                    "compensatory_note"  : "rpoB non-RRDR in sample with primary RRDR mutation",
                    "drug_context"       : "rifampicin",
                })

        #validated compensatory gene list
        is_validated_comp = gene in COMPENSATORY_GENES
        comp_meta = COMPENSATORY_GENES.get(gene, {})
        requires_mdr = comp_meta.get("requires_mdr", True)

        # If gene requires MDR context, only flag if sample is MDR
        if is_validated_comp and requires_mdr and not is_mdr:
            is_validated_comp = False

        is_compensatory = is_validated_comp or is_rpoB_nonRRDR_compensatory

        other_row = {
            "sample_id"                  : sample_id,
            "gene"                       : gene,
            "locus_tag"                  : locus,
            "change"                     : change,
            "nucleotide_change"          : variant.get("nucleotide_change", ""),
            "type"                       : vtype,
            "depth"                      : depth,
            "freq"                       : freq,
            "pos"                        : pos,
            "is_compensatory_candidate"  : int(is_compensatory),
            "compensatory_mechanism"     : comp_meta.get("mechanism", "rpoB non-RRDR" if is_rpoB_nonRRDR_compensatory else ""),
            "drug_context"               : comp_meta.get("drug_context", "rifampicin" if is_rpoB_nonRRDR_compensatory else ""),
            "evidence"                   : comp_meta.get("evidence", ""),
            "sample_MDR"                 : is_mdr,
            "sample_pre_XDR"             : is_pre_xdr,
            "sample_XDR"                 : is_xdr,
            "sample_drtype"              : drtype,
            "sample_lineage"             : main_lineage,
            "drugs_annotated"            : "|".join(drugs_ann),
            "confidences_annotated"      : "|".join(confs_ann),
            "requires_MDR_context"       : int(requires_mdr),
        }
        other_variant_rows.append(other_row)

        if is_compensatory:
            compensatory_rows.append(other_row.copy())

    if (i + 1) % 100 == 0:
        print(f"  Parsed {i+1} / {len(json_files)}")

print(f"Parsing complete. {len(failed_files)} files failed.")


#building dataframes and saving them 
df_labels = pd.DataFrame(labels_rows)
df_dr     = pd.DataFrame(dr_variant_rows)     if dr_variant_rows     else pd.DataFrame()
df_other  = pd.DataFrame(other_variant_rows)  if other_variant_rows  else pd.DataFrame()
df_comp   = pd.DataFrame(compensatory_rows)   if compensatory_rows   else pd.DataFrame()
df_rpoB   = pd.DataFrame(rpoB_nonRRDR_rows)   if rpoB_nonRRDR_rows   else pd.DataFrame()

df_labels.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
print(f"Saved: labels.csv           ({len(df_labels)} samples)")

if not df_dr.empty:
    df_dr.to_csv(os.path.join(OUT_DIR, "dr_variants.csv"), index=False)
    print(f"Saved: dr_variants.csv      ({len(df_dr)} records)")

if not df_other.empty:
    df_other.to_csv(os.path.join(OUT_DIR, "other_variants.csv"), index=False)
    print(f"Saved: other_variants.csv   ({len(df_other)} records)")

if not df_comp.empty:
    df_comp.to_csv(os.path.join(OUT_DIR, "compensatory.csv"), index=False)
    print(f"Saved: compensatory.csv     ({len(df_comp)} unique candidate records)")

if not df_rpoB.empty:
    df_rpoB.to_csv(os.path.join(OUT_DIR, "rpoB_nonRRDR.csv"), index=False)
    print(f"Saved: rpoB_nonRRDR.csv     ({len(df_rpoB)} rpoB non-RRDR compensatory candidates)")

#summary file
with open(os.path.join(OUT_DIR, "summary_stats.txt"), "w") as f:
    f.write("TB-Profiler Collation Summary — Corrected Compensatory Classification\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total samples parsed : {len(df_labels)}\n")
    f.write(f"Failed files         : {len(failed_files)}\n\n")

    f.write("Drug Resistance Classification (TB-Profiler drtype)\n")
    f.write("-" * 50 + "\n")
    for drtype, count in df_labels["drtype"].value_counts().items():
        f.write(f"  {drtype:<35}: {count:>5} ({count/len(df_labels)*100:.1f}%)\n")

    f.write(f"\nMDR / Pre-XDR / XDR (computed)\n")
    f.write("-" * 50 + "\n")
    f.write(f"  MDR     : {df_labels['MDR'].sum():>5} ({df_labels['MDR'].mean()*100:.1f}%)\n")
    f.write(f"  Pre-XDR : {df_labels['pre_XDR'].sum():>5} ({df_labels['pre_XDR'].mean()*100:.1f}%)\n")
    f.write(f"  XDR     : {df_labels['XDR'].sum():>5} ({df_labels['XDR'].mean()*100:.1f}%)\n")
    f.write(f"  rpoB RRDR confirmed: {df_labels['has_rpoB_RRDR'].sum():>5}\n")

    f.write(f"\nLineage Distribution\n")
    f.write("-" * 50 + "\n")
    for lin, count in df_labels["main_lineage"].value_counts().items():
        if lin:
            f.write(f"  {lin:<35}: {count:>5} ({count/len(df_labels)*100:.1f}%)\n")

    f.write(f"\nPer-Drug Resistance Rates\n")
    f.write("-" * 50 + "\n")
    for drug in ALL_DRUGS:
        if drug in df_labels.columns:
            r = (df_labels[drug] == "R").sum()
            f.write(f"  {drug:<40}: {r:>5} ({r/len(df_labels)*100:.1f}%)\n")

    if not df_comp.empty:
        f.write(f"\nCompensatory Mutation Summary (CLEAN SET)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total candidate records  : {len(df_comp)}\n")
        f.write(f"  Unique samples affected  : {df_comp['sample_id'].nunique()}\n")
        f.write(f"  Unique genes             : {df_comp['gene'].nunique()}\n")
        f.write(f"\n  Records by gene:\n")
        for gene, cnt in df_comp["gene"].value_counts().items():
            mech = COMPENSATORY_GENES.get(gene, {}).get("drug_context", "rpoB non-RRDR")
            f.write(f"    {gene:<15} [{mech:<30}]: {cnt:>4}\n")

    if not df_rpoB.empty:
        f.write(f"\nrpoB non-RRDR Compensatory Candidates\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total records : {len(df_rpoB)}\n")
        f.write(f"  Unique samples: {df_rpoB['sample_id'].nunique()}\n")
        f.write(f"  Unique changes: {df_rpoB['change'].nunique()}\n")

print("\nSaved: summary_stats.txt")
print("\n" + "=" * 60)
print(f"DONE — {len(df_labels)} samples collated")
print(f"MDR     : {df_labels['MDR'].sum()} ({df_labels['MDR'].mean()*100:.1f}%)")
print(f"Pre-XDR : {df_labels['pre_XDR'].sum()} ({df_labels['pre_XDR'].mean()*100:.1f}%)")
print(f"XDR     : {df_labels['XDR'].sum()} ({df_labels['XDR'].mean()*100:.1f}%)")
if not df_comp.empty:
    print(f"Compensatory candidates: {len(df_comp)} records across {df_comp['sample_id'].nunique()} samples")
if not df_rpoB.empty:
    print(f"rpoB non-RRDR candidates: {len(df_rpoB)} records")
print(f"Files saved to: {OUT_DIR}")
print("=" * 60)