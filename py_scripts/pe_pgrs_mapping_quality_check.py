#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


BASE       = os.path.expanduser("~/tb_pipeline")
VCF_DIR    = os.path.join(BASE, "vcf_filtered")
STEP5      = os.path.join(BASE, "ml_outputs", "step5_shap")
CHROM_NAME = "Chromosome"
OUT_DIR    = STEP5
TOP100_CSV = os.path.join(STEP5, "obj2_shap_top100_annotated.csv")
os.makedirs(OUT_DIR, exist_ok=True)


# BUG FIX (see conversation record): this used to be two hardcoded dicts,
# hand-transcribed from obj2_shap_top100_annotated.csv at some earlier
# point. That snapshot went stale the moment Step 5 was rerun on the
# corrected data -- every position/rank pairing in the old dicts matched
# the OLD pre-fix top100 file and silently mismatched the current one
# (e.g. position 840217 is genuinely rank 8 in the current top100, one of
# the highest-priority features to check, but the stale dict labeled it
# "rank 56", and 24 of the 33 real current PE_PGRS/PPE positions weren't
# in the dict at all). Reading the CSV directly at runtime instead of
# hardcoding a snapshot means this can't go stale again the next time
# Step 5 reruns.
def load_target_positions():
    """Build {pos: label} dicts for PE_PGRS/PPE and primary-DR positions
    directly from the current top100 file -- no hardcoded snapshot."""
    if not os.path.exists(TOP100_CSV):
        sys.exit(f"obj2_shap_top100_annotated.csv not found at {TOP100_CSV}\n"
                  f"Run Step 5 (SHAP analysis) first, or edit TOP100_CSV above "
                  f"if it lives elsewhere.")
    df = pd.read_csv(TOP100_CSV)

    def make_label(row):
        return f"{row.variant_id} ({row.gene}, rank {row.rank})"

    is_pe_ppe = df["gene"].str.contains("PE_PGRS|PPE", na=False)
    pe_pgrs_df = df[is_pe_ppe]
    primary_dr_df = df[df["is_primary_DR"] == 1]

    pos_from_vid = lambda v: int(v.split("_")[1])
    pe_pgrs_positions = {
        pos_from_vid(r.variant_id): make_label(r) for r in pe_pgrs_df.itertuples()
    }
    primary_dr_positions = {
        pos_from_vid(r.variant_id): make_label(r) for r in primary_dr_df.itertuples()
    }
    return pe_pgrs_positions, primary_dr_positions


def build_regions_file(path: str, pe_pgrs_positions: dict, primary_dr_positions: dict):
    """One line per target position, 1-based, matching bcftools -R format."""
    all_pos = sorted(set(pe_pgrs_positions) | set(primary_dr_positions))
    with open(path, "w") as f:
        for pos in all_pos:
            f.write(f"{CHROM_NAME}\t{pos}\t{pos}\n")
    return len(all_pos)


def ensure_indexed(vcf_path: str):
    """filter_vcfs.sh indexes every file it produces, but this is a cheap,
    defensive check in case any file was copied/regenerated without its
    .csi (bcftools query -R requires an index)."""
    csi = vcf_path + ".csi"
    tbi = vcf_path + ".tbi"
    if not (os.path.exists(csi) or os.path.exists(tbi)):
        subprocess.run(["bcftools", "index", vcf_path], capture_output=True)


def query_one_vcf(vcf_path: str, regions_path: str) -> list:
    """Returns list of (pos, qual, dp) tuples for target positions found in this VCF."""
    ensure_indexed(vcf_path)
    # %INFO/DP prints '.' if absent; [%DP] prints the sample's own FORMAT/DP.
    fmt = r"%POS\t%QUAL\t%INFO/DP\t[%DP]\n"
    cmd = ["bcftools", "query", "-R", regions_path, "-f", fmt, vcf_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():  # splitlines(), NOT .strip().splitlines() -
        line = line.rstrip("\n")          # a whole-blob .strip() eats the trailing tab
        if not line:                      # off only the LAST line's empty FORMAT/DP
            continue                      # field, silently dropping that one row.
        parts = line.split("\t")
        if len(parts) < 3:                # POS, QUAL, INFO/DP are required; the
            continue                      # trailing [%DP] field is optional/may be empty
        pos_s, qual_s, info_dp_s = parts[0], parts[1], parts[2]
        fmt_dp_s = parts[3] if len(parts) > 3 else ""
        try:
            pos = int(pos_s)
        except ValueError:
            continue
        qual = None if qual_s in (".", "") else _to_float(qual_s)
        dp = _to_float(info_dp_s) if info_dp_s not in (".", "") else _to_float(fmt_dp_s)
        rows.append((pos, qual, dp))
    return rows


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-max-samples", type=int, default=None,
                     help="Limit to N sample VCFs for a quick test run.")
    ap.add_argument("-vcf-dir", default=VCF_DIR)
    args = ap.parse_args()

    if not os.path.isdir(args.vcf_dir):
        sys.exit(f"VCF directory not found: {args.vcf_dir}\n"
                  f"Edit VCF_DIR / pass -vcf-dir if your filtered VCFs live elsewhere.")

    vcf_files = sorted(glob.glob(os.path.join(args.vcf_dir, "*.vcf.gz")))
    if not vcf_files:
        sys.exit(f"No *.vcf.gz files found in {args.vcf_dir}")
    if args.max_samples:
        vcf_files = vcf_files[: args.max_samples]
    print(f"Found {len(vcf_files)} filtered VCF(s) to scan.")

    pe_pgrs_positions, primary_dr_positions = load_target_positions()
    print(f"Loaded target positions from {TOP100_CSV}: "
          f"{len(pe_pgrs_positions)} PE_PGRS/PPE, {len(primary_dr_positions)} primary-DR "
          f"(current top-100 ranks, not a hardcoded snapshot)")

    regions_path = os.path.join(OUT_DIR, "_pe_pgrs_check_regions.tsv")
    n_positions = build_regions_file(regions_path, pe_pgrs_positions, primary_dr_positions)
    print(f"Querying {n_positions} target positions "
          f"({len(pe_pgrs_positions)} PE_PGRS/PPE, {len(primary_dr_positions)} primary-DR)...")

    records = []
    for i, vcf in enumerate(vcf_files, 1):
        sample_id = os.path.basename(vcf).replace(".filtered.vcf.gz", "").replace(".vcf.gz", "")
        for pos, qual, dp in query_one_vcf(vcf, regions_path):
            group = ("PE_PGRS_PPE" if pos in pe_pgrs_positions
                     else "Primary_DR" if pos in primary_dr_positions else None)
            if group is None:
                continue
            records.append({
                "sample_id": sample_id, "pos": pos, "group": group,
                "qual": qual, "dp": dp,
            })
        if i % 200 == 0 or i == len(vcf_files):
            print(f"  {i}/{len(vcf_files)} VCFs scanned, {len(records):,} calls collected so far...")

    if not records:
        sys.exit("No matching calls found at any target position across the scanned VCFs. "
                  "Check CHROM_NAME and VCF_DIR assumptions at the top of this script.")

    df = pd.DataFrame(records)
    raw_path = os.path.join(OUT_DIR, "pe_pgrs_call_confidence_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw per-call QUAL/DP: {raw_path} ({len(df):,} rows)")

    # - per-position summary -
    label_map = {**pe_pgrs_positions, **primary_dr_positions}
    per_pos = df.groupby(["pos", "group"]).agg(
        n_calls=("qual", "size"),
        mean_qual=("qual", "mean"),
        median_qual=("qual", "median"),
        mean_dp=("dp", "mean"),
        median_dp=("dp", "median"),
    ).reset_index()
    per_pos["label"] = per_pos["pos"].map(label_map)
    per_pos_path = os.path.join(OUT_DIR, "pe_pgrs_call_confidence_per_position.csv")
    per_pos.to_csv(per_pos_path, index=False)
    print(f"Saved per-position summary: {per_pos_path}")

    # - group-level comparison -
    pe = df[df["group"] == "PE_PGRS_PPE"]
    dr = df[df["group"] == "Primary_DR"]

    print("\n" + "=" * 70)
    print("GROUP-LEVEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20}{'PE_PGRS/PPE (n=' + str(len(pe)) + ')':<28}{'Primary-DR (n=' + str(len(dr)) + ')'}")
    print(f"{'Mean QUAL':<20}{pe['qual'].mean():<28.2f}{dr['qual'].mean():.2f}")
    print(f"{'Median QUAL':<20}{pe['qual'].median():<28.2f}{dr['qual'].median():.2f}")
    print(f"{'Mean DP':<20}{pe['dp'].mean():<28.2f}{dr['dp'].mean():.2f}")
    print(f"{'Median DP':<20}{pe['dp'].median():<28.2f}{dr['dp'].median():.2f}")

    qual_u, qual_p = mannwhitneyu(pe["qual"].dropna(), dr["qual"].dropna(), alternative="two-sided")
    dp_u, dp_p = mannwhitneyu(pe["dp"].dropna(), dr["dp"].dropna(), alternative="two-sided")
    print(f"\nMann-Whitney U test (PE_PGRS/PPE vs Primary-DR):")
    print(f"  QUAL: U={qual_u:.1f}, p={qual_p:.2e}"
          f"  {'*** significant difference' if qual_p < 0.05 else '(not significant)'}")
    print(f"  DP  : U={dp_u:.1f}, p={dp_p:.2e}"
          f"  {'*** significant difference' if dp_p < 0.05 else '(not significant)'}")

    
    if qual_p < 0.05 and pe["qual"].median() < dr["qual"].median():
        print("INTERPRETATION: PE_PGRS/PPE calls show significantly LOWER QUAL than "
              "primary-DR calls - consistent with (though not proof of) a mapping-"
              "quality concern in these repetitive regions.")
    elif qual_p < 0.05:
        print("INTERPRETATION: A significant QUAL difference was found, but PE_PGRS/PPE "
              "calls are NOT the weaker group - re-examine before drawing conclusions.")
    else:
        print("INTERPRETATION: No significant QUAL difference between groups at this "
              "sample size - does not, on its own, support the mapping-artefact "
              "concern, though it does not rule out multi-mapping ambiguity that QUAL/DP "
              "alone cannot detect (see the framing note at the top of this script).")
    

    os.remove(regions_path)


if __name__ == "__main__":
    main()