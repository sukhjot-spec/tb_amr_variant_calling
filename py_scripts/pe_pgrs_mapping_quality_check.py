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
os.makedirs(OUT_DIR, exist_ok=True)


# these are the target positions pulled directly from the real Step 5 output
#    (obj2_shap_top100_annotated.csv): every PE_PGRS/PPE-family variant
#    in the top 100 (42 positions) vs every primary-DR-gene variant in the top 100 (9 positions), used here as the confidence baseline.
PE_PGRS_POSITIONS = {
    3935771: "Chromosome_3935771_C_G (PE_PGRS54, rank 5)",
    1636143: "Chromosome_1636143_G_T (PE_PGRS28, rank 11)",
    839794:  "Chromosome_839794_C_T (PE_PGRS10, rank 12)",
    3948347: "Chromosome_3948347_T_G (PE_PGRS57, rank 13)",
    3947125: "Chromosome_3947125_C_G (PE_PGRS57, rank 14)",
    3948362: "Chromosome_3948362_T_G (PE_PGRS57, rank 15)",
    338648:  "Chromosome_338648_T_G (PE_PGRS4, rank 16)",
    3738516: "Chromosome_3738516_C_CTGCCGCCGCTGCCGCCGT (PE_PGRS50, rank 18)",
    3941723: "Chromosome_3941723_ACC_AC (PE_PGRS55, rank 24)",
    3843678: "Chromosome_3843678_CACACCATGGTGA_C (PPE58, rank 27)",
    1189773: "Chromosome_1189773_T_A (PE_PGRS19, rank 29)",
    3940802: "Chromosome_3940802_A_G (PE_PGRS55, rank 30)",
    839790:  "Chromosome_839790_C_A (PE_PGRS10, rank 35)",
    839821:  "Chromosome_839821_G_C (PE_PGRS10, rank 42)",
    1637228: "Chromosome_1637228_A_G (PE_PGRS28, rank 44)",
    839271:  "Chromosome_839271_C_A (PE_PGRS10, rank 45)",
    3732194: "Chromosome_3732194_A_G (PPE54, rank 50)",
    3948417: "Chromosome_3948417_T_G (PE_PGRS57, rank 54)",
    338669:  "Chromosome_338669_T_C (PE_PGRS4, rank 55)",
    840217:  "Chromosome_840217_C_G (PE_PGRS10, rank 56)",
    3934699: "Chromosome_3934699_G_A (PE_PGRS54, rank 61)",
    3947128: "Chromosome_3947128_C_G (PE_PGRS57, rank 63)",
    338618:  "Chromosome_338618_C_G (PE_PGRS4, rank 65)",
    839295:  "Chromosome_839295_T_C (PE_PGRS10, rank 66)",
    362466:  "Chromosome_362466_A_C (PE_PGRS5, rank 68)",
    3738512: "Chromosome_3738512_A_G (PE_PGRS50, rank 69)",
    338777:  "Chromosome_338777_G_C (PE_PGRS4, rank 72)",
    3941516: "Chromosome_3941516_G_A (PE_PGRS55, rank 73)",
    338719:  "Chromosome_338719_T_C (PE_PGRS4, rank 75)",
    1864424: "Chromosome_1864424_G_C (PE_PGRS30, rank 78)",
    840358:  "Chromosome_840358_C_G (PE_PGRS10, rank 79)",
    839471:  "Chromosome_839471_T_C (PE_PGRS10, rank 80)",
    1488434: "Chromosome_1488434_T_G (PE_PGRS24, rank 81)",
    839291:  "Chromosome_839291_T_C (PE_PGRS10, rank 87)",
    1637018: "Chromosome_1637018_G_C (PE_PGRS28, rank 88)",
    3842452: "Chromosome_3842452_C_A (PPE57, rank 89)",
    840430:  "Chromosome_840430_A_C (PE_PGRS10, rank 90)",
    1488435: "Chromosome_1488435_C_A (PE_PGRS24, rank 95)",
    337175:  "Chromosome_337175_G_A (PE_PGRS4, rank 96)",
    338690:  "Chromosome_338690_A_G (PE_PGRS4, rank 98)",
    424320:  "Chromosome_424320_T_TC (PPE7, rank 99)",
    1339741: "Chromosome_1339741_C_G (PPE18, rank 100)",
}

PRIMARY_DR_POSITIONS = {
    2155168: "Chromosome_2155168_C_G (katG, rank 1)",
    761155:  "Chromosome_761155_C_T (rpoB, rank 3)",
    7570:    "Chromosome_7570_C_T (gyrA, rank 6)",
    7581:    "Chromosome_7581_G_C (gyrA, rank 22)",
    761110:  "Chromosome_761110_A_T (rpoB, rank 25)",
    1473177: "Chromosome_1473177_G_A (rrs, rank 51)",
    1472895: "Chromosome_1472895_C_T (rrs, rank 58)",
    7582:    "Chromosome_7582_A_G (gyrA, rank 59)",
    7362:    "Chromosome_7362_G_C (gyrA, rank 67)",
}


def build_regions_file(path: str):
    """One line per target position, 1-based, matching bcftools -R format."""
    all_pos = sorted(set(PE_PGRS_POSITIONS) | set(PRIMARY_DR_POSITIONS))
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

    regions_path = os.path.join(OUT_DIR, "_pe_pgrs_check_regions.tsv")
    n_positions = build_regions_file(regions_path)
    print(f"Querying {n_positions} target positions "
          f"({len(PE_PGRS_POSITIONS)} PE_PGRS/PPE, {len(PRIMARY_DR_POSITIONS)} primary-DR)...")

    records = []
    for i, vcf in enumerate(vcf_files, 1):
        sample_id = os.path.basename(vcf).replace(".filtered.vcf.gz", "").replace(".vcf.gz", "")
        for pos, qual, dp in query_one_vcf(vcf, regions_path):
            group = ("PE_PGRS_PPE" if pos in PE_PGRS_POSITIONS
                     else "Primary_DR" if pos in PRIMARY_DR_POSITIONS else None)
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
    label_map = {**PE_PGRS_POSITIONS, **PRIMARY_DR_POSITIONS}
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