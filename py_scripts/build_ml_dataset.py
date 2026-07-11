
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


import pandas as pd
import numpy as np
import os

BASE   = os.path.expanduser("~/tb_pipeline")
ML_OUT = os.path.join(BASE, "ml_outputs")
TB_OUT = os.path.join(BASE, "tbprofiler_results")
os.makedirs(ML_OUT, exist_ok=True)

print("STEP 1: Building ML-Ready Dataset (using the NPZ file)")

#loading from NPZ
print("\n[1/8] Loading feature matrix from NPZ...")
NPZ = os.path.join(BASE, "feature_matrix.npz")
assert os.path.exists(NPZ), f"Not found: {NPZ}"

data        = np.load(NPZ, allow_pickle=True)
matrix      = data["matrix"]            # (n_samples, n_variants)
sample_ids  = data["samples"].tolist()  # list of sample ID strings
variant_ids = data["variants"].tolist() # list of variant ID strings

print(f"  Loaded: {matrix.shape[0]} samples × {matrix.shape[1]} variants")
print(f"  RAM used: {matrix.nbytes / 1e6:.0f} MB")
print(f"  Sample IDs (first 3): {sample_ids[:3]}")
print(f"  Variant IDs (first 3): {variant_ids[:3]}")

#fixing malformed variant IDs 
print("\n[2/8] Fixing malformed variant IDs...")
import re

def is_valid_variant_id(vid: str) -> bool:
    parts = str(vid).split("_")
    if len(parts) < 4:
        return False
    chrom, pos = parts[0], parts[1]
    ref = parts[2]
    alt = "_".join(parts[3:])
    if chrom not in ("Chromosome", "NC_000962.3"):
        return False
    if not pos.isdigit():
        return False
    if not re.match(r'^[ACTGNacgtn]+$', ref):
        return False
    if not re.match(r'^[ACTGNacgtn]+$', alt):
        return False
    return True

good_mask   = np.array([is_valid_variant_id(v) for v in variant_ids])
bad_count   = (~good_mask).sum()
bad_examples = [v for v, g in zip(variant_ids, good_mask) if not g][:5]

print(f"  Valid variants  : {good_mask.sum():,}")
print(f"  Malformed removed: {bad_count:,} — examples: {bad_examples}")

matrix      = matrix[:, good_mask]
variant_ids = [v for v, g in zip(variant_ids, good_mask) if g]
print(f"  Matrix after fix: {matrix.shape}")

#loading labels 
print("\n[3/8] Loading labels...")
labels = pd.read_csv(os.path.join(TB_OUT, "labels.csv"))
print(f"  Shape: {labels.shape}")
print(f"  MDR: {labels['MDR'].sum()} / {len(labels)} ({labels['MDR'].mean()*100:.1f}%)")
print(f"  Pre-XDR: {labels['pre_XDR'].sum()} | XDR: {labels['XDR'].sum()}")
print(f"  has_rpoB_RRDR: {labels['has_rpoB_RRDR'].sum()}")
print(f"  Lineage distribution:")
for lin, cnt in labels["main_lineage"].value_counts().head(6).items():
    print(f"    {str(lin) if lin else 'Unknown'}: {cnt} ({cnt/len(labels)*100:.1f}%)")

#Aligning samples between matrix and labels 
print("\n[4/8] Aligning samples...")
labels_idx     = labels.set_index("sample_id")
feature_set    = set(sample_ids)
label_set      = set(labels_idx.index.tolist())
common_samples = sorted(feature_set & label_set)

print(f"  Samples in NPZ      : {len(feature_set):,}")
print(f"  Samples in labels   : {len(label_set):,}")
print(f"  Common (intersection): {len(common_samples):,}")

only_feat = feature_set - label_set
only_lab  = label_set - feature_set
if only_feat:
    print(f"  In NPZ but not labels : {len(only_feat)} — {list(only_feat)[:3]}")
if only_lab:
    print(f"  In labels but not NPZ : {len(only_lab)} — {list(only_lab)[:3]}")

#reordering matrix rows to match common_samples
sample_to_row = {s: i for i, s in enumerate(sample_ids)}
row_order     = [sample_to_row[s] for s in common_samples]
matrix_aligned = matrix[row_order, :]

#aligning labels
labels_aligned = labels_idx.loc[common_samples].reset_index()
print(f"  Aligned matrix : {matrix_aligned.shape}")
print(f"  Aligned labels : {labels_aligned.shape}")

#extracting y vectors
print("\n[5/8] Extracting target vectors...")
y_mdr     = labels_aligned["MDR"].values.astype(np.int8)
y_pre_xdr = labels_aligned["pre_XDR"].values.astype(np.int8)
y_xdr     = labels_aligned["XDR"].values.astype(np.int8)

print(f"  MDR     — 1: {y_mdr.sum():,}  0: {(y_mdr==0).sum():,}  ({y_mdr.mean()*100:.1f}% positive)")
print(f"  Pre-XDR — 1: {y_pre_xdr.sum():,}  0: {(y_pre_xdr==0).sum():,}  ({y_pre_xdr.mean()*100:.1f}% positive)")
print(f"  XDR     — 1: {y_xdr.sum():,}  0: {(y_xdr==0).sum():,}  ({y_xdr.mean()*100:.1f}% positive)")

#removing zero-variance features post-alignment 
print("\n[6/8] Removing zero-variance features post-alignment...")
col_sums  = matrix_aligned.sum(axis=0, dtype=np.int32)
n         = matrix_aligned.shape[0]
min_count = max(1, int(0.01 * n))
max_count = int(0.99 * n)

keep_mask     = (col_sums >= min_count) & (col_sums <= max_count)
removed       = (~keep_mask).sum()
matrix_final  = matrix_aligned[:, keep_mask]
variants_final = [variant_ids[i] for i, k in enumerate(keep_mask) if k]

print(f"  Removed {removed:,} zero/near-fixed features")
print(f"  Final matrix: {matrix_final.shape[0]} samples × {matrix_final.shape[1]} variants")

#saving all outputs
print("\n[7/8] Saving outputs...")

# Fast numpy saves
np.save(os.path.join(ML_OUT, "X_array.npy"), matrix_final.astype(np.uint8))
print(f"  ✅ X_array.npy          {matrix_final.shape}")

np.save(os.path.join(ML_OUT, "y_mdr_array.npy"),     y_mdr)
np.save(os.path.join(ML_OUT, "y_pre_xdr_array.npy"), y_pre_xdr)
np.save(os.path.join(ML_OUT, "y_xdr_array.npy"),     y_xdr)
print(f"  ✅ y_mdr / y_pre_xdr / y_xdr arrays saved")

# Sample and variant ID lists
with open(os.path.join(ML_OUT, "sample_ids.txt"), "w") as f:
    f.write("\n".join(common_samples))
print(f"  ✅ sample_ids.txt       ({len(common_samples)} samples)")

with open(os.path.join(ML_OUT, "feature_names_clean.txt"), "w") as f:
    f.write("\n".join(variants_final))
print(f"  ✅ feature_names_clean.txt ({len(variants_final)} variants)")

# Full labels aligned
labels_aligned.to_csv(os.path.join(ML_OUT, "y_labels.csv"), index=False)
print(f"  ✅ y_labels.csv         {labels_aligned.shape}")

# Variant metadata filtered to final variants
meta = pd.read_csv(os.path.join(BASE, "variant_metadata.csv"))
meta_filt = meta[meta["variant_id"].isin(set(variants_final))]
meta_filt.to_csv(os.path.join(ML_OUT, "variant_metadata_filtered.csv"), index=False)
print(f"  ✅ variant_metadata_filtered.csv ({len(meta_filt):,} variants)")

# Objective 1 compensatory files
print("\n  Building Objective 1 compensatory summaries...")
comp      = pd.read_csv(os.path.join(TB_OUT, "compensatory.csv"))
rpob      = pd.read_csv(os.path.join(TB_OUT, "rpoB_nonRRDR.csv"))

comp_in_mdr  = comp[comp["sample_MDR"] == 1]
comp_not_mdr = comp[comp["sample_MDR"] == 0]

mdr_set     = set(labels_aligned[labels_aligned["MDR"] == 1]["sample_id"])
non_mdr_set = set(labels_aligned[labels_aligned["MDR"] == 0]["sample_id"])
total_mdr     = len(mdr_set)
total_non_mdr = len(non_mdr_set)

rows = []
for (gene, change), grp in comp.groupby(["gene", "change"]):
    samps = set(grp["sample_id"].unique())
    mdr_w     = len(samps & mdr_set)
    non_mdr_w = len(samps & non_mdr_set)
    rows.append({
        "gene"                    : gene,
        "change"                  : change,
        "drug_context"            : grp["drug_context"].iloc[0],
        "compensatory_mechanism"  : grp["compensatory_mechanism"].iloc[0],
        "evidence"                : grp["evidence"].iloc[0],
        "mdr_with_mutation"       : mdr_w,
        "non_mdr_with_mutation"   : non_mdr_w,
        "mdr_without_mutation"    : total_mdr - mdr_w,
        "non_mdr_without_mutation": total_non_mdr - non_mdr_w,
        "mdr_frequency"           : mdr_w / total_mdr if total_mdr else 0,
        "non_mdr_frequency"       : non_mdr_w / total_non_mdr if total_non_mdr else 0,
        "enrichment_ratio"        : (mdr_w / total_mdr) / (non_mdr_w / total_non_mdr + 1e-6),
    })

df_enrich = pd.DataFrame(rows).sort_values("enrichment_ratio", ascending=False)
df_enrich.to_csv(os.path.join(ML_OUT, "obj1_mdr_vs_susceptible_comp.csv"), index=False)
comp_in_mdr.to_csv(os.path.join(ML_OUT, "obj1_compensatory_in_MDR_samples.csv"), index=False)
rpob.to_csv(os.path.join(ML_OUT, "obj1_rpoB_nonRRDR_summary.csv"), index=False)
print(f"  ✅ obj1_mdr_vs_susceptible_comp.csv  ({len(df_enrich)} mutations)")
print(f"  ✅ obj1_compensatory_in_MDR_samples.csv ({len(comp_in_mdr)} records)")
print(f"  ✅ obj1_rpoB_nonRRDR_summary.csv     ({len(rpob)} records)")

# CSV fallback — write chunked to avoid OOM
print("\n  Writing X_features.csv (chunked — may take 2-3 min)...")
x_csv = os.path.join(ML_OUT, "X_features.csv")
with open(x_csv, "w") as f:
    f.write("sample_id," + ",".join(variants_final) + "\n")
    for start in range(0, len(common_samples), 100):
        end = min(start + 100, len(common_samples))
        for i in range(start, end):
            f.write(common_samples[i] + "," +
                    ",".join(map(str, matrix_final[i])) + "\n")
print(f"  ✅ X_features.csv       {matrix_final.shape}")

#check
print("\n[8/8] Sanity check...")
X_  = np.load(os.path.join(ML_OUT, "X_array.npy"))
y_  = np.load(os.path.join(ML_OUT, "y_mdr_array.npy"))
assert X_.shape[0] == y_.shape[0], "MISMATCH: X rows != y length"
print(f"  X shape    : {X_.shape}")
print(f"  y shape    : {y_.shape}")
print(f"  Match      : ✅")
print(f"  MDR rate   : {y_.mean()*100:.1f}%")
print(f"  X dtype    : {X_.dtype}")
print(f"  Sparsity   : {(X_==0).mean()*100:.1f}% zeros")

mdr_m  = X_[y_==1].mean(axis=0)
susc_m = X_[y_==0].mean(axis=0)
diff   = np.abs(mdr_m - susc_m)
top5   = diff.argsort()[-5:][::-1]
print(f"\n  Top 5 differentially present variants (MDR vs Susceptible):")
print(f"  {'Variant':<45} {'MDR_freq':>9} {'SUSC_freq':>9} {'|Diff|':>7}")
for idx in top5:
    print(f"  {variants_final[idx]:<45} {mdr_m[idx]:>9.3f} {susc_m[idx]:>9.3f} {diff[idx]:>7.3f}")

print("\n")
print("STEP 1 COMPLETE")
print(f"  {X_.shape[0]} samples × {X_.shape[1]} variants")
print(f"  MDR: {y_.sum()} positive / {(y_==0).sum()} negative")
print(f"  All outputs in: {ML_OUT}/")
