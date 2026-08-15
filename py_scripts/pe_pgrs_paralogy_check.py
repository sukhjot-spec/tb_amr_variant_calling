#!/usr/bin/env python3
"""
pe_pgrs_paralogy_check.py

Follow-up to pe_pgrs_mapping_quality_check.py. That script measures WHETHER
a position has low call confidence (QUAL/DP); this one tests WHY, for
whichever positions came back low. No BAM/alignment access needed - this
works entirely from the reference FASTA and the confidence-check's own
output CSV.

The concrete, checkable hypothesis: a position has low call confidence
because its surrounding sequence has a near-identical match elsewhere in
the genome (or repeated within itself), which causes short reads from
there to multi-map ambiguously. That's a real, testable mechanism, not
just "PE_PGRS genes are generally repetitive" - this finds the specific
paralogous region for each flagged position, or confirms none exists.

Which positions get checked is NOT hardcoded - it's read fresh from
pe_pgrs_call_confidence_per_position.csv every run, with the flagging
threshold derived from that same file's own Primary-DR group (whatever
the current worst confidently-called primary-DR position's QUAL is, any
PE_PGRS/PPE position below that gets checked). Same lesson as the
mapping_quality_check fix earlier in this project: no hardcoded snapshot
that can go stale the next time Step 5 reruns and the top-100 list changes.

Usage:
    python3 pe_pgrs_paralogy_check.py
    python3 pe_pgrs_paralogy_check.py --qual-threshold 100   # override
    python3 pe_pgrs_paralogy_check.py --window 200 --kmer 30
"""
import argparse
import os

import pandas as pd

BASE = os.path.expanduser("~/tb_pipeline")
FASTA_PATH = os.path.join(BASE, "reference", "H37Rv.fasta")
STEP5 = os.path.join(BASE, "ml_outputs", "step5_shap")
CONFIDENCE_CSV = os.path.join(STEP5, "pe_pgrs_call_confidence_per_position.csv")
OUT_CSV = os.path.join(STEP5, "pe_pgrs_paralogy_check.csv")


def load_genome(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()


def gc_content(s):
    if not s:
        return 0.0
    return 100 * (s.count("G") + s.count("C")) / len(s)


def find_all(genome, kmer):
    """All start positions (0-based) of kmer in genome, including overlaps."""
    hits = []
    start = 0
    while True:
        idx = genome.find(kmer, start)
        if idx == -1:
            break
        hits.append(idx)
        start = idx + 1
    return hits


def get_flagged_positions(confidence_csv, qual_threshold=None):
    """Read the confidence-check output and select PE_PGRS/PPE positions
    below the flagging threshold. If qual_threshold is None, derive it
    from the current Primary-DR group's own worst (minimum) mean_qual --
    i.e. "any PE_PGRS/PPE position less confidently called than the
    worst-called primary-DR-gene position in this same run"."""
    if not os.path.exists(confidence_csv):
        raise SystemExit(
            f"Confidence file not found: {confidence_csv}\n"
            f"Run pe_pgrs_mapping_quality_check.py first."
        )
    df = pd.read_csv(confidence_csv)

    if qual_threshold is None:
        dr = df[df["group"] == "Primary_DR"]
        if len(dr) == 0:
            raise SystemExit("No Primary_DR rows in confidence file -- "
                              "can't derive a threshold, pass --qual-threshold explicitly.")
        qual_threshold = dr["mean_qual"].min()
        print(f"Derived flagging threshold from current data: "
              f"{qual_threshold:.1f} (worst Primary-DR mean_qual)")
    else:
        print(f"Using explicit flagging threshold: {qual_threshold:.1f}")

    pe = df[df["group"] == "PE_PGRS_PPE"].copy()
    flagged = pe[pe["mean_qual"] < qual_threshold].sort_values("mean_qual")
    print(f"Flagged {len(flagged)}/{len(pe)} PE_PGRS/PPE positions "
          f"(mean_qual < {qual_threshold:.1f})\n")
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qual-threshold", type=float, default=None,
                     help="Flag PE_PGRS/PPE positions below this mean_qual. "
                          "Default: derived from the worst Primary-DR position "
                          "in the current confidence file.")
    ap.add_argument("--window", type=int, default=150,
                     help="bp each side of the position to search (default 150, "
                          "roughly a short-read insert size)")
    ap.add_argument("--kmer", type=int, default=25,
                     help="k-mer size for the paralogy search (default 25 -- "
                          "specific enough that any hit in a 4.4 Mb genome is "
                          "meaningful, sensitive enough to catch divergent paralogs)")
    ap.add_argument("--fasta", default=FASTA_PATH)
    ap.add_argument("--confidence-csv", default=CONFIDENCE_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if not os.path.exists(args.fasta):
        raise SystemExit(f"Reference FASTA not found: {args.fasta}")

    flagged = get_flagged_positions(args.confidence_csv, args.qual_threshold)
    if len(flagged) == 0:
        print("Nothing flagged -- no PE_PGRS/PPE positions fall below the "
              "threshold in the current confidence file. Nothing to check.")
        return

    genome = load_genome(args.fasta)
    print(f"Genome length: {len(genome):,} bp\n")

    print(f"{'pos':>9} {'gene':<12} {'mean_qual':>10} {'GC%':>6} "
          f"{'paralog_hits':>13} {'nearest_paralog':>20}")
    print("-" * 85)

    results = []
    for _, row in flagged.iterrows():
        pos = int(row["pos"])
        gene_label = row["label"] if "label" in row and pd.notna(row["label"]) else str(pos)
        # label looks like "Chromosome_840217_C_G (PE_PGRS10, rank 8)" -- pull the gene name out
        gene = gene_label.split("(")[-1].split(",")[0].strip() if "(" in gene_label else "?"

        win_start = max(0, pos - args.window - 1)  # pos is 1-based
        win_end = min(len(genome), pos + args.window)
        window_seq = genome[win_start:win_end]
        gc = gc_content(window_seq)

        paralog_positions = set()
        for i in range(0, len(window_seq) - args.kmer, 10):  # step 10bp for speed
            kmer = window_seq[i:i + args.kmer]
            if "N" in kmer:
                continue
            hits = find_all(genome, kmer)
            true_hits = [h for h in hits if not (win_start - args.kmer <= h <= win_end)]
            paralog_positions.update(true_hits)

        n_hits = len(paralog_positions)
        if paralog_positions:
            nearest = min(paralog_positions, key=lambda h: abs(h - pos)) + 1
            dist_kb = abs(nearest - pos) / 1000
            nearest_str = f"{nearest:,} ({dist_kb:.0f}kb away)"
        else:
            nearest_str = "none found"

        results.append({
            "pos": pos, "gene": gene, "mean_qual": row["mean_qual"],
            "gc_pct": round(gc, 1), "paralog_hits": n_hits,
            "nearest_paralog_pos": nearest if paralog_positions else None,
            "nearest_paralog_label": nearest_str,
        })
        print(f"{pos:>9} {gene:<12} {row['mean_qual']:>10.1f} {gc:>5.1f}% "
              f"{n_hits:>13} {nearest_str:>20}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

    print()
    print("Interpretation:")
    print("  paralog_hits > 0  -> concrete evidence of near-identical sequence")
    print("                       elsewhere in the genome (or repeated within")
    print("                       the region itself, if nearest is ~0kb away);")
    print("                       multi-mapping is a real, specific risk here,")
    print("                       not just a general 'repetitive family' guess.")
    print("  paralog_hits == 0 -> no k-mer match found elsewhere. Doesn't rule")
    print("                       out lower-identity paralogy below this k-mer's")
    print("                       sensitivity, but does rule out an exact/near-")
    print("                       exact duplicate as the explanation.")
    print("  Nearest ~0kb away -> internal tandem repeat within the gene itself")
    print("                       (the classic PE_PGRS repeat-domain signature),")
    print("                       not a separate paralogous gene elsewhere.")
    print(f"  GC% context: H37Rv genome average is ~65.6%; markedly higher GC")
    print(f"  is a secondary, independent signal of low-complexity sequence.")
    print()
    
if __name__ == "__main__":
    main()
