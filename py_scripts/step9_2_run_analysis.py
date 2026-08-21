"""
Step 9, script 2 of 2 (formerly script 3 of 3 - renumbered now that
step9_1_build_genotype_matrix.py has been dropped; see step9_1_build_tree.py's
load_step1_data() for why).

Takes the tree (script 1) and Step 1's own genotype matrix (loaded
directly, not rebuilt - see below), and:

  Part A - Homoplasy / convergence counting
    For every variant in the compensatory gene panel + rpoB (the full
    set that any of Step 8's 629 significant, lineage-restricted
    candidates could belong to), runs Fitch parsimony (fitch.py) against
    the tree to get the minimum number of independent origin events.
    steps==1 means the variant's presence pattern is fully explained by
    one ancestral acquisition, inherited clonally (weak evidence of
    selection, however many samples carry it). steps>1 means the raw
    carriage count overstates independent evolutionary support - either
    true convergent (independent) origins, or a gain plus reversion(s).

  Part B - Gene-wide, tree-corrected dN/dS
    Classifies every gene-set SNP as synonymous/nonsynonymous by codon
    context (codon_classify.py), then computes dN/dS per gene using
    independent-origin counts (not raw sample counts) as the numerator
    - this is what corrects for MTb's clonal population structure,
    since a variant inherited by 500 samples from one ancestor is one
    event, not 500. Normalized by Nei-Gojobori possible N/S sites per
    gene. Scoped to the 12-gene compensatory panel only (not rpoB,
    which is the primary rifampicin-resistance gene, not part of this
    panel).

GENOTYPE SOURCE: this script reads Step 1's X_array.npy directly
(via load_step1_data(), duplicated from step9_1_build_tree.py rather
than factored into a shared module, since this project's other steps
are single self-contained scripts) instead of a separately-rebuilt
matrix. X_array.npy is (1858 samples, 94583 sites) - sample-major,
the opposite orientation of what an earlier version of this script
used, so site lookups below are X[:, pos_index[pos]] (a column) rather
than a row.

Inputs expected:
    ~/tb_pipeline/ml_outputs/step9_conservation/full_tree_complete.parstree
        from step9_1_build_tree.py's complete_tree() - NOT
        full_tree.parstree directly; see that function's docstring for
        why (IQ-TREE drops genotypically-identical taxa from its own
        direct output, and complete_tree() reattaches them)
    ~/tb_pipeline/ml_outputs/step1_dataset/X_array.npy,
        feature_names_clean.txt, sample_ids.txt
        Step 1's own outputs, loaded directly
    ~/tb_pipeline/ml_outputs/step1_dataset/variant_metadata_with_genes.csv
        has REF/ALT/gene per variant
    ~/tb_pipeline/ml_outputs/step8_lineage/obj3_lineage_distribution.csv
        the 629 significant, lineage-restricted named candidates

Outputs (written to OUTDIR):
    obj3_step9_geneset_homoplasy.csv     every gene-set variant, homoplasy + syn/nonsyn
    obj3_step9_gene_dNdS.csv             per-gene event-based dN/dS (compensatory panel only)
    obj3_step9_candidate_conservation.csv  the 629 named candidates, joined with the above
    obj3_step9_gene_dNdS_plot.png        bar chart of gene-wide dN/dS

CAVEATS surfaced by an earlier run of this analysis, worth a manual look
on your real re-run rather than trusted as still-current (numbers below
predate three separate fixes since - the row-shift/MDR-filtering bugs,
the mmpR5/Rv0678 gene-alias fix, and the IQ-TREE dropped-identical-taxa
fix - so treat these as "worth specifically re-checking", not as
current findings):
  - embR p.Phe376Leu: 176 independent origins on only 519 carriers, a
    large outlier relative to everything else in the table (next
    highest is 81). Independently worth noting: this is the same
    variant Step 8 flagged as the one candidate with inconsistent
    MDR-association direction across lineages
    (obj3_lineage_inconsistent_mdr_direction.csv) - two independent
    analyses flagging the same variant as anomalous is a real signal
    worth extra scrutiny, not a coincidence to wave away.
  - mmpR5 p.Asp47Glu: 60 steps on 66 carriers, and it's the dataset's
    one indel in a homopolymer-adjacent context (TC->TCC). That
    combination (indel + homopolymer) is a classic sequencing/calling
    artifact hotspot.
"""
import os
import csv
import time
from collections import Counter

import numpy as np
from ete3 import Tree

from fitch import fitch_rooted_batch
from codon_classify import (
    build_gene_contexts, possible_sites, COMPENSATORY_GENES, EXTRA_GENES,
    ALL_GENES, GENE_NAME_ALIASES,
)

STEP1 = os.path.expanduser("~/tb_pipeline/ml_outputs/step1_dataset")
INDIR = os.path.expanduser("~/tb_pipeline/ml_outputs/step9_conservation")
OUTDIR = os.path.expanduser("~/tb_pipeline/ml_outputs/step9_conservation")
os.makedirs(OUTDIR, exist_ok=True)
VARIANT_METADATA_PATH = os.path.join(STEP1, "variant_metadata_with_genes.csv")
CANDIDATES_PATH = os.path.expanduser(
    "~/tb_pipeline/ml_outputs/step8_lineage/obj3_lineage_distribution.csv")


def load_step1_data():
    """Read Step 1's own matrix + ID files directly, rather than
    rebuilding anything from the raw VCF. Duplicated verbatim from
    step9_1_build_tree.py - see that file's module docstring for why
    this replaced the old step9_1_build_genotype_matrix.py entirely.
    Returns X (samples x sites, shape (1858, 94583)), positions (POS per
    column, same order as X's columns), sample_ids (per row, same order
    as X's rows)."""
    X = np.load(os.path.join(STEP1, "X_array.npy"))  # (1858, 94583)
    with open(os.path.join(STEP1, "feature_names_clean.txt")) as f:
        feature_names = [l.strip() for l in f if l.strip()]
    positions = [int(v.split("_")[1]) for v in feature_names]
    with open(os.path.join(STEP1, "sample_ids.txt")) as f:
        sample_ids = [l.strip() for l in f if l.strip()]
    assert X.shape == (len(sample_ids), len(positions)), (
        f"X_array.npy shape {X.shape} doesn't match "
        f"({len(sample_ids)} sample_ids, {len(positions)} feature_names) - "
        f"check these three files are all from the same Step 1 run."
    )
    return X, positions, sample_ids


def load_tree_and_matrix():
    tree = Tree(f"{INDIR}/full_tree_complete.parstree")
    X, positions, sample_ids = load_step1_data()
    pos_index = {p: i for i, p in enumerate(positions)}
    sample_index = {s: i for i, s in enumerate(sample_ids)}
    return tree, X, pos_index, sample_index


def load_and_classify_geneset_variants(contexts):
    """Load every variant in ALL_GENES (compensatory panel + rpoB) from
    variant_metadata_with_genes.csv and classify each by codon context.

    Gene names are normalized via GENE_NAME_ALIASES (Rv0678 -> mmpR5,
    same physical gene under two names depending on which annotation
    source resolved a given variant - see codon_classify.py) before
    BOTH the ALL_GENES membership check and the GeneContext lookup, and
    the normalized name is written back into the row itself (not just
    used transiently), so every downstream consumer - the per-gene
    grouping in compute_gene_dNdS(), the output CSV's own "gene" column
    - sees one consistent gene name rather than a mix of the two.
    Without this, contexts["Rv0678"] would raise a KeyError outright
    (only "mmpR5" is ever built as a GeneContext), and even a defensive
    fix that avoided the crash would still leave that variant's data
    invisible to compute_gene_dNdS()'s `v["gene"] == "mmpR5"` grouping."""
    rows = []
    with open(VARIANT_METADATA_PATH) as f:
        for row in csv.DictReader(f):
            gene = GENE_NAME_ALIASES.get(row["gene"], row["gene"])
            if gene in ALL_GENES:
                row = dict(row)
                row["gene"] = gene
                rows.append(row)

    results = []
    n_base_mismatch = 0
    for row in rows:
        gene = row["gene"]
        pos = int(row["POS"])
        ref, alt = row["REF"], row["ALT"]
        ctx = contexts[gene]
        cls = ctx.classify_variant_row(pos, ref, alt)
        if cls.get("base_check_ok") is False:
            n_base_mismatch += 1
        results.append({**row, **cls})

    print(f"Classified {len(results)} gene-set variants "
          f"({len(COMPENSATORY_GENES)}-gene compensatory panel + rpoB)")
    print("  classification breakdown:",
          dict(Counter(r["classification"] for r in results)))
    if n_base_mismatch:
        print(f"  WARNING: {n_base_mismatch} REF-base mismatches vs genome "
              f"(should be 0 - investigate before trusting results)")
    return results


def run_homoplasy(tree, X, pos_index, sample_index, variants, root_state=0):
    """Root-fixed Fitch parsimony (root_state=0, i.e. reference/absent).

    Fixing the root matters concretely for this dataset, not just in
    theory: on an earlier run, a small number of named candidates had an
    unconstrained-Fitch root set that was NOT purely {0} (one, mmpL5
    p.Ile948Val at 86% carriage, had root set {1} exclusively) - meaning
    unconstrained parsimony's minimum-cost solution for those candidates
    either requires or permits "the mutation was ancestral and
    independently lost," which is biologically backwards given H37Rv
    (this project's reference throughout) is drug-susceptible. That
    specific count was measured against the old, pre-fix 391-candidate
    set and against a tree missing 31 samples IQ-TREE had dropped (see
    step9_1_build_tree.py's complete_tree()) - worth re-measuring on a
    real re-run rather than assumed unchanged, though the underlying
    rationale for forcing the root is unaffected by either fix. See
    fitch.fitch_rooted_gains_losses for the implementation and full
    rationale.
    """
    tree_tips = set(tree.get_leaf_names())
    sites_tip_states = {}
    for v in variants:
        pos = int(v["POS"])
        if pos not in pos_index:
            continue
        col = X[:, pos_index[pos]]  # X is samples x sites - column, not row
        tip_states = {s: int(col[sample_index[s]]) for s in tree_tips if s in sample_index}
        sites_tip_states[v["variant_id"]] = tip_states

    print(f"Running root-fixed Fitch parsimony on {len(sites_tip_states)} sites "
          f"({len(tree_tips)} tree tips, root fixed to {root_state})...")
    t0 = time.time()
    homoplasy = fitch_rooted_batch(tree, sites_tip_states, root_state=root_state)
    print(f"  done in {time.time() - t0:.1f}s")

    for v in variants:
        vid = v["variant_id"]
        if vid in homoplasy:
            gains, losses = homoplasy[vid]
            v["n_independent_origins"] = gains
            v["n_losses"] = losses
            v["fitch_steps"] = gains + losses  # kept for continuity with earlier output
            v["n_carriers_in_tree"] = sum(sites_tip_states[vid].values())
            v["convergence_pattern"] = ("convergent" if gains >= 2
                                         else "single_origin_clonal" if gains == 1
                                         else "absent_in_tree")
    return variants


def write_geneset_homoplasy_csv(variants):
    rows = []
    for v in variants:
        if "fitch_steps" not in v:
            continue
        rows.append({
            "variant_id": v["variant_id"],
            "gene": v["gene"],
            "POS": v["POS"],
            "REF": v["REF"],
            "ALT": v["ALT"],
            "is_compensatory_candidate": v["is_compensatory"],
            "classification": v["classification"],
            "n_carriers_in_tree": v["n_carriers_in_tree"],
            "n_independent_origins": v["n_independent_origins"],
            "n_losses": v["n_losses"],
            "convergence_pattern": v["convergence_pattern"],
        })
    path = f"{OUTDIR}/obj3_step9_geneset_homoplasy.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    return rows


def compute_gene_dNdS(variants, contexts):
    """Event-based dN/dS, scoped to the compensatory panel only (not rpoB -
    see module docstring)."""
    gene_stats = []
    for gene in COMPENSATORY_GENES:
        if gene not in contexts:
            continue
        ps = possible_sites(contexts[gene])
        gv = [v for v in variants if v["gene"] == gene and "fitch_steps" in v]
        # Event count = independent ORIGINS only (gains), not gains+losses -
        # dN/dS measures the rate at which nonsyn/syn changes arise, and with
        # the root now correctly fixed to 0, "gains" is the direct count of
        # origin events for each category.
        N_events = sum(v["n_independent_origins"] for v in gv if v["classification"] == "nonsynonymous")
        S_events = sum(v["n_independent_origins"] for v in gv if v["classification"] == "synonymous")
        n_nonsyn_obs = sum(1 for v in gv if v["classification"] == "nonsynonymous")
        n_syn_obs = sum(1 for v in gv if v["classification"] == "synonymous")
        dN = N_events / ps["N_sites"] if ps["N_sites"] > 0 else None
        dS = S_events / ps["S_sites"] if ps["S_sites"] > 0 else None
        dNdS = (dN / dS) if (dN is not None and dS not in (None, 0)) else None
        gene_stats.append({
            "gene": gene,
            "cds_length_bp": contexts[gene].length,
            "possible_N_sites": round(ps["N_sites"], 1),
            "possible_S_sites": round(ps["S_sites"], 1),
            "n_nonsyn_variants_observed": n_nonsyn_obs,
            "n_syn_variants_observed": n_syn_obs,
            "N_events_fitch_sum": N_events,
            "S_events_fitch_sum": S_events,
            "dN": round(dN, 5) if dN is not None else None,
            "dS": round(dS, 5) if dS is not None else None,
            "dN_dS_event_based": round(dNdS, 3) if dNdS is not None else "undefined_dS_zero",
        })

    path = f"{OUTDIR}/obj3_step9_gene_dNdS.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gene_stats[0].keys()))
        w.writeheader()
        w.writerows(gene_stats)
    print(f"Wrote {path} ({len(gene_stats)} genes)")
    for g in gene_stats:
        print(" ", g)
    return gene_stats


def write_candidate_conservation_csv(homoplasy_rows, gene_stats):
    homoplasy_by_vid = {r["variant_id"]: r for r in homoplasy_rows}
    gene_dnds = {r["gene"]: r for r in gene_stats}

    rows = []
    with open(CANDIDATES_PATH) as f:
        for cand in csv.DictReader(f):
            vid = cand["variant_id"]
            # Defensive normalization, same as load_and_classify_geneset_variants():
            # not currently triggered by the real candidate list (no candidate is
            # gene=="Rv0678" as of this writing), but cand["gene"] comes from a
            # different file (Step 8's output) than the alias fix was applied to,
            # so nothing guarantees that stays true on a future rerun.
            gene = GENE_NAME_ALIASES.get(cand["gene"], cand["gene"])
            h = homoplasy_by_vid.get(vid)
            gd = gene_dnds.get(gene, {})
            rows.append({
                "gene": gene,
                "change": cand["change"],
                "variant_id": vid,
                "drug_context": cand["drug_context"],
                "overall_carriage_pct": cand["overall_carriage_pct"],
                "lineage_restricted": cand["lineage_restricted"],
                "coding_classification": h["classification"] if h else "not_found",
                "n_independent_origins": h["n_independent_origins"] if h else None,
                "n_losses": h["n_losses"] if h else None,
                "n_carriers_in_tree": h["n_carriers_in_tree"] if h else None,
                "convergence_pattern": h["convergence_pattern"] if h else None,
                "gene_dN_dS_event_based": gd.get(
                    "dN_dS_event_based",
                    "n/a (rpoB not in compensatory panel)" if gene == "rpoB" else "n/a",
                ),
            })

    n_missing = sum(1 for r in rows if r["coding_classification"] == "not_found")
    if n_missing:
        print(f"WARNING: {n_missing} candidates have no homoplasy result - "
              f"check whether their gene is covered by ALL_GENES in codon_classify.py")

    path = f"{OUTDIR}/obj3_step9_candidate_conservation.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows, {n_missing} unresolved)")
    print("  convergence pattern breakdown:",
          dict(Counter(r["convergence_pattern"] for r in rows)))
    return rows


def make_dNdS_plot(gene_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genes, dnds = [], []
    for g in gene_stats:
        v = g["dN_dS_event_based"]
        if v == "undefined_dS_zero":
            continue
        genes.append(g["gene"])
        dnds.append(float(v))

    order = sorted(range(len(genes)), key=lambda i: dnds[i])
    genes = [genes[i] for i in order]
    dnds = [dnds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1F4E5F" if d < 0.3 else ("#c0392b" if d > 0.6 else "#5b8a99") for d in dnds]
    ax.barh(genes, dnds, color=colors)
    ax.axvline(1.0, color="gray", linestyle="-", linewidth=1, label="dN/dS = 1 (neutral)")
    ax.set_xlabel("Event-based dN/dS (Fitch-parsimony substitution events / possible sites)")
    ax.set_title("Step 9: Gene-wide dN/dS, tree-corrected for MTb clonality\n"
                  "(compensatory gene panel; genes with no coding variants observed omitted)")
    ax.legend()
    plt.tight_layout()
    path = f"{OUTDIR}/obj3_step9_gene_dNdS_plot.png"
    plt.savefig(path, dpi=150)
    print(f"Wrote {path}")


def main():
    tree, X, pos_index, sample_index = load_tree_and_matrix()
    print(f"Tree: {len(tree.get_leaf_names())} tips")

    contexts = build_gene_contexts()

    variants = load_and_classify_geneset_variants(contexts)
    variants = run_homoplasy(tree, X, pos_index, sample_index, variants)

    homoplasy_rows = write_geneset_homoplasy_csv(variants)
    gene_stats = compute_gene_dNdS(variants, contexts)
    write_candidate_conservation_csv(homoplasy_rows, gene_stats)
    make_dNdS_plot(gene_stats)


if __name__ == "__main__":
    main()
