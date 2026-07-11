
#build annotated ML-ready dataset
#outputs:
#   ml_outputs/X_features.csv          - variant feature matrix (samples x variants)
#   ml_outputs/y_labels.csv            - all labels (MDR, drtype, lineage, per-drug)
#   ml_outputs/ml_dataset.csv          - X + y joined
#   ml_outputs/X_array.npy             - numpy array for fast ML loading
#   ml_outputs/y_mdr_array.npy         - MDR binary target
#   ml_outputs/feature_names_clean.txt - cleaned feature names
#   ml_outputs/compensatory_mdr_summary.csv  - compensatory mutations in MDR context
#   ml_outputs/obj1_mdr_vs_susceptible_comp.csv - comp mutations stratified


# All issues from file inspection handled:
#   1. 27,494 multi-allelic variant IDs (comma in ALT) — first ALT kept, deduped
#   2. SRR11922476 in labels but missing from NPZ — inner join, 1857 samples used
#   3. 41 null main_lineage values — filled with 'Unknown'
#   4. compensatory.csv has 17,494 non-MDR records — stratified correctly
#   5. rpoB_nonRRDR has 187 non-MDR records — flagged, tracked separately
#   6. Matrix 97.4% sparse — confirmed correct, no fix needed

import pandas as pd
import numpy as np
import os
import re

BASE   = os.path.expanduser("~/tb_pipeline")
ML_OUT = os.path.join(BASE, "ml_outputs")
TB_OUT = os.path.join(BASE, "tbprofiler_results")
os.makedirs(ML_OUT, exist_ok=True)

print("STEP 1: Building ML-Ready Dataset")

#Loading NPZ file
print("\n[1/8] Loading feature matrix from NPZ...")
NPZ = os.path.join(BASE, "feature_matrix.npz")
assert os.path.exists(NPZ), f"ERROR: Not found: {NPZ}"

data        = np.load(NPZ, allow_pickle=True)
matrix      = data["matrix"]
sample_ids  = data["samples"].tolist()
variant_ids = data["variants"].tolist()

print(f"  Loaded  : {matrix.shape[0]} samples × {matrix.shape[1]} variants")
print(f"  RAM     : {matrix.nbytes / 1e6:.0f} MB")
print(f"  dtype   : {matrix.dtype}")
print(f"  Sparsity: {(matrix == 0).mean() * 100:.1f}% zeros (expected ~97% for WGS SNP data)")
print(f"  Sample IDs (first 3) : {sample_ids[:3]}")
print(f"  Variant IDs (first 3): {variant_ids[:3]}")


#fixing malformed variant IDs -> 27,494 multi-allelic IDs with comma in ALT
#    after renaming, deduplicating to keep only the first occurrence
print("\n[2/8] Fixing malformed variant IDs...")

def fix_variant_id(vid: str) -> str:
    """
    Fixes multi-allelic variant IDs by keeping only the first ALT allele.
    Chromosome_51_C_T,A  →  Chromosome_51_C_T
    Chromosome_255_T_G,C,A  →  Chromosome_255_T_G
    Also validates format — returns empty string if invalid.
    """
    vid = str(vid).strip()
    parts = vid.split("_")
    if len(parts) < 4:
        return ""
    chrom = parts[0]
    pos   = parts[1]
    ref   = parts[2]
    alt   = "_".join(parts[3:])  # rejoin in case of indels

    # Keep only first ALT if multi-allelic
    if "," in alt:
        alt = alt.split(",")[0].strip()

    # Validate
    if chrom not in ("Chromosome", "NC_000962.3"):
        return ""
    if not pos.isdigit():
        return ""
    if not re.match(r'^[ACTGNacgtn]+$', ref):
        return ""
    if not re.match(r'^[ACTGNacgtn]+$', alt):
        return ""

    return f"{chrom}_{pos}_{ref}_{alt}"

fixed_ids   = [fix_variant_id(v) for v in variant_ids]
invalid_mask = np.array([fid == "" for fid in fixed_ids])
print(f"  Multi-allelic (comma in ALT): {sum(',' in str(v) for v in variant_ids):,} → first ALT kept")
print(f"  Completely invalid (removed): {invalid_mask.sum():,}")

#removing fully invalid ones
valid_mask  = ~invalid_mask
matrix      = matrix[:, valid_mask]
fixed_ids   = [fid for fid, v in zip(fixed_ids, valid_mask) if v]

#deduplicating as after fixing multi-allelics some IDs may now collide
#keep first occurrence of each variant ID
seen        = {}
dedup_mask  = []
for i, fid in enumerate(fixed_ids):
    if fid not in seen:
        seen[fid] = i
        dedup_mask.append(True)
    else:
        dedup_mask.append(False)

dedup_mask  = np.array(dedup_mask)
n_dupes     = (~dedup_mask).sum()
matrix      = matrix[:, dedup_mask]
variant_ids = [fid for fid, k in zip(fixed_ids, dedup_mask) if k]

print(f"  Duplicates removed after fix : {n_dupes:,}")
print(f"  Final clean variant IDs      : {len(variant_ids):,}")
print(f"  Matrix after fix             : {matrix.shape}")
print(f"  Example fixed IDs            : {variant_ids[:5]}")


#loading labels
#issue: 41 null main_lineage
print("\n[3/8] Loading labels...")
labels = pd.read_csv(os.path.join(TB_OUT, "labels.csv"))

# Fix null lineages
null_lineage = labels["main_lineage"].isnull().sum()
labels["main_lineage"] = labels["main_lineage"].fillna("Unknown")
labels["sub_lineage"]  = labels["sub_lineage"].fillna("Unknown")
print(f"  Shape             : {labels.shape}")
print(f"  Null lineages fixed: {null_lineage} → filled with 'Unknown'")
print(f"  MDR               : {labels['MDR'].sum()} / {len(labels)} ({labels['MDR'].mean()*100:.1f}%)")
print(f"  Pre-XDR           : {labels['pre_XDR'].sum()} | XDR: {labels['XDR'].sum()}")
print(f"  has_rpoB_RRDR     : {labels['has_rpoB_RRDR'].sum()}")
print(f"  Lineage distribution:")
for lin, cnt in labels["main_lineage"].value_counts().head(8).items():
    print(f"    {lin}: {cnt} ({cnt/len(labels)*100:.1f}%)")

#aligning samples
#    issue: SRR11922476 in labels but NOT in NPZ (1858 labels, 1857 NPZ)
#    SRR11922476 excluded
print("\n[4/8] Aligning samples...")
labels_idx     = labels.set_index("sample_id")
feature_set    = set(sample_ids)
label_set      = set(labels_idx.index.tolist())
common_samples = sorted(feature_set & label_set)

only_in_npz    = feature_set - label_set
only_in_labels = label_set - feature_set

print(f"  Samples in NPZ       : {len(feature_set):,}")
print(f"  Samples in labels    : {len(label_set):,}")
print(f"  Common (inner join)  : {len(common_samples):,}")

if only_in_npz:
    print(f" In NPZ only (excluded): {len(only_in_npz)} — {list(only_in_npz)[:5]}")
if only_in_labels:
    print(f" In labels only (excluded — no VCF processed): "
          f"{len(only_in_labels)} — {list(only_in_labels)}")
    #saving excluded samples for record
    pd.DataFrame({"sample_id": list(only_in_labels),
                  "reason": "In labels but missing from feature matrix NPZ"}
                ).to_csv(os.path.join(ML_OUT, "excluded_samples.csv"), index=False)
    print(f"    → Saved to excluded_samples.csv")

#reordering matrix rows
sample_to_row  = {s: i for i, s in enumerate(sample_ids)}
row_order      = [sample_to_row[s] for s in common_samples]
matrix_aligned = matrix[row_order, :]
labels_aligned = labels_idx.loc[common_samples].reset_index()

print(f"  Aligned matrix : {matrix_aligned.shape}")
print(f"  Aligned labels : {labels_aligned.shape}")

#extracting y vectors
print("\n[5/8] Extracting target vectors...")
y_mdr     = labels_aligned["MDR"].values.astype(np.int8)
y_pre_xdr = labels_aligned["pre_XDR"].values.astype(np.int8)
y_xdr     = labels_aligned["XDR"].values.astype(np.int8)

print(f"  MDR     — 1:{y_mdr.sum():>5}  0:{(y_mdr==0).sum():>5}  ({y_mdr.mean()*100:.1f}% positive)")
print(f"  Pre-XDR — 1:{y_pre_xdr.sum():>5}  0:{(y_pre_xdr==0).sum():>5}  ({y_pre_xdr.mean()*100:.1f}% positive)")
print(f"  XDR     — 1:{y_xdr.sum():>5}  0:{(y_xdr==0).sum():>5}  ({y_xdr.mean()*100:.1f}% positive)")

#emoving zero-variance features post-alignment
print("\n[6/8] Removing zero-variance features post-alignment...")
col_sums  = matrix_aligned.sum(axis=0, dtype=np.int32)
n         = matrix_aligned.shape[0]
min_count = max(1, int(0.01 * n))   # present in ≥1% samples
max_count = int(0.99 * n)           # present in ≤99% samples

keep_mask      = (col_sums >= min_count) & (col_sums <= max_count)
removed        = (~keep_mask).sum()
matrix_final   = matrix_aligned[:, keep_mask]
variants_final = [variant_ids[i] for i, k in enumerate(keep_mask) if k]

print(f"  Min count threshold  : {min_count} samples (1%)")
print(f"  Max count threshold  : {max_count} samples (99%)")
print(f"  Removed (zero/fixed) : {removed:,}")
print(f"  Final matrix         : {matrix_final.shape[0]} samples × {matrix_final.shape[1]} variants")

#saving all outputs
print("\n[7/8] Saving outputs...")

# Core numpy arrays
np.save(os.path.join(ML_OUT, "X_array.npy"),
        matrix_final.astype(np.uint8))
np.save(os.path.join(ML_OUT, "y_mdr_array.npy"),     y_mdr)
np.save(os.path.join(ML_OUT, "y_pre_xdr_array.npy"), y_pre_xdr)
np.save(os.path.join(ML_OUT, "y_xdr_array.npy"),     y_xdr)
print(f" X_array.npy              {matrix_final.shape}")
print(f" y_mdr / y_pre_xdr / y_xdr arrays")

# Sample and variant ID lists
with open(os.path.join(ML_OUT, "sample_ids.txt"), "w") as f:
    f.write("\n".join(common_samples))
with open(os.path.join(ML_OUT, "feature_names_clean.txt"), "w") as f:
    f.write("\n".join(variants_final))
print(f" sample_ids.txt           ({len(common_samples)} samples)")
print(f" feature_names_clean.txt  ({len(variants_final)} variants)")

# Full aligned labels
labels_aligned.to_csv(os.path.join(ML_OUT, "y_labels.csv"), index=False)
print(f"  y_labels.csv             {labels_aligned.shape}")

# Variant metadata filtered to final clean variants
meta     = pd.read_csv(os.path.join(BASE, "variant_metadata.csv"))
# Fix multi-allelic in metadata ALT, keeping first ALT only
meta["ALT"] = meta["ALT"].str.split(",").str[0]
meta["variant_id"] = (meta["CHROM"] + "_" + meta["POS"].astype(str) +
                      "_" + meta["REF"] + "_" + meta["ALT"])
meta_filt = meta[meta["variant_id"].isin(set(variants_final))].drop_duplicates("variant_id")
meta_filt.to_csv(os.path.join(ML_OUT, "variant_metadata_filtered.csv"), index=False)
print(f"  variant_metadata_filtered.csv ({len(meta_filt):,} variants)")

#Objective 1: Compensatory mutation summaries 
print("\n  Building Objective 1 compensatory summaries...")
comp = pd.read_csv(os.path.join(TB_OUT, "compensatory.csv"))
rpob = pd.read_csv(os.path.join(TB_OUT, "rpoB_nonRRDR.csv"))

#restricting to samples in the aligned dataset only
comp = comp[comp["sample_id"].isin(set(common_samples))].copy()
rpob = rpob[rpob["sample_id"].isin(set(common_samples))].copy()

mdr_set       = set(labels_aligned[labels_aligned["MDR"] == 1]["sample_id"])
non_mdr_set   = set(labels_aligned[labels_aligned["MDR"] == 0]["sample_id"])
total_mdr     = len(mdr_set)
total_non_mdr = len(non_mdr_set)

#stratifying compensatory records
comp_in_mdr  = comp[comp["sample_MDR"] == 1]
comp_not_mdr = comp[comp["sample_MDR"] == 0]

print(f"  Compensatory records in MDR samples     : {len(comp_in_mdr):,}")
print(f"  Compensatory records in non-MDR samples : {len(comp_not_mdr):,}")
print(f"  rpoB non-RRDR in MDR samples            : {(rpob['sample_MDR']==1).sum():,}")
print(f"  rpoB non-RRDR in non-MDR samples        : {(rpob['sample_MDR']==0).sum():,}")

#per-mutation MDR vs non-MDR enrichment table
rows = []
for (gene, change), grp in comp.groupby(["gene", "change"]):
    samps     = set(grp["sample_id"].unique())
    mdr_w     = len(samps & mdr_set)
    non_mdr_w = len(samps & non_mdr_set)
    rows.append({
        "gene"                     : gene,
        "change"                   : change,
        "drug_context"             : grp["drug_context"].iloc[0],
        "compensatory_mechanism"   : grp["compensatory_mechanism"].iloc[0],
        "evidence"                 : grp["evidence"].iloc[0],
        "mdr_with_mutation"        : mdr_w,
        "non_mdr_with_mutation"    : non_mdr_w,
        "mdr_without_mutation"     : total_mdr - mdr_w,
        "non_mdr_without_mutation" : total_non_mdr - non_mdr_w,
        "mdr_frequency"            : round(mdr_w / total_mdr, 6) if total_mdr else 0,
        "non_mdr_frequency"        : round(non_mdr_w / total_non_mdr, 6) if total_non_mdr else 0,
        "enrichment_ratio"         : round(
            (mdr_w / total_mdr) / (non_mdr_w / total_non_mdr + 1e-6), 4
        ),
    })

df_enrich = pd.DataFrame(rows).sort_values("enrichment_ratio", ascending=False)

#saving
df_enrich.to_csv(os.path.join(ML_OUT, "obj1_mdr_vs_susceptible_comp.csv"),    index=False)
comp_in_mdr.to_csv(os.path.join(ML_OUT, "obj1_compensatory_in_MDR_samples.csv"), index=False)
comp_not_mdr.to_csv(os.path.join(ML_OUT, "obj1_compensatory_in_nonMDR_samples.csv"), index=False)
rpob.to_csv(os.path.join(ML_OUT, "obj1_rpoB_nonRRDR_summary.csv"),            index=False)

print(f" obj1_mdr_vs_susceptible_comp.csv        ({len(df_enrich):,} unique mutations)")
print(f" obj1_compensatory_in_MDR_samples.csv    ({len(comp_in_mdr):,} records)")
print(f" obj1_compensatory_in_nonMDR_samples.csv ({len(comp_not_mdr):,} records)")
print(f" obj1_rpoB_nonRRDR_summary.csv           ({len(rpob):,} records)")

#CSV fallback
print("\n  Writing X_features.csv (chunked)...")
x_csv = os.path.join(ML_OUT, "X_features.csv")
CHUNK = 100
with open(x_csv, "w") as f:
    f.write("sample_id," + ",".join(variants_final) + "\n")
    for start in range(0, len(common_samples), CHUNK):
        end = min(start + CHUNK, len(common_samples))
        for i in range(start, end):
            f.write(common_samples[i] + "," +
                    ",".join(map(str, matrix_final[i])) + "\n")
        if start % 500 == 0:
            print(f"    {end}/{len(common_samples)} samples written...")
print(f" X_features.csv  {matrix_final.shape}")

#check
print("\n[8/8] Sanity check...")
X_ = np.load(os.path.join(ML_OUT, "X_array.npy"))
y_ = np.load(os.path.join(ML_OUT, "y_mdr_array.npy"))

assert X_.shape[0] == y_.shape[0],   f"MISMATCH: X rows {X_.shape[0]} != y {y_.shape[0]}"
assert X_.shape[1] == len(variants_final), "MISMATCH: X cols != variant list"
assert set(np.unique(X_)) <= {0, 1},  f"UNEXPECTED values in X: {np.unique(X_)}"
assert set(np.unique(y_)) <= {0, 1},  f"UNEXPECTED values in y: {np.unique(y_)}"

print(f"  X shape    : {X_.shape}  ")
print(f"  y shape    : {y_.shape}  ")
print(f"  X dtype    : {X_.dtype}")
print(f"  MDR rate   : {y_.mean()*100:.1f}%")
print(f"  Sparsity   : {(X_==0).mean()*100:.1f}% zeros")
print(f"  All assertions passed ")

mdr_m  = X_[y_ == 1].mean(axis=0)
susc_m = X_[y_ == 0].mean(axis=0)
diff   = np.abs(mdr_m - susc_m)
top5   = diff.argsort()[-5:][::-1]

print(f"\n  Top 5 most differentially present variants (MDR vs Susceptible):")
print(f"  {'Variant':<45} {'MDR_freq':>9} {'SUSC_freq':>9} {'|Diff|':>7}")
for idx in top5:
    print(f"  {variants_final[idx]:<45} {mdr_m[idx]:>9.3f} "
          f"{susc_m[idx]:>9.3f} {diff[idx]:>7.3f}")

print("Complete")
print(f"  Samples  : {X_.shape[0]} (1 excluded — SRR11922476, no VCF in NPZ)")
print(f"  Features : {X_.shape[1]} variant positions")
print(f"  MDR      : {y_.sum()} positive / {(y_==0).sum()} negative")
print(f"  Outputs  : {ML_OUT}/")
