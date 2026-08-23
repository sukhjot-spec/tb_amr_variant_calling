#!/usr/bin/env python3
"""
Step 11 - Final integration. Closes the project by joining every evidence
layer this project has produced, for every one of Step 2's 1,209 significant
compensatory-mutation candidates, into a single master table, and assigns
each candidate an overall evidence tier based on how many independent lines
of evidence support it (or flag it as anomalous).

This does not recompute anything Steps 2-10 already established - it reads
their own saved outputs directly and joins them by variant_id (falling back
to gene+change where a file only has that), exactly as every other step in
this project reads from a previous step's saved files rather than
duplicating logic.

THE FOUR EVIDENCE LAYERS (see the accompanying Phase 5B report for the full
rationale behind each):
  1. Statistical significance   (Step 2  - obj1_compensatory_fisher.csv)
  2. ML importance              (Steps 4-6 - SHAP/EBM/RF top rankings,
                                  the 5-feature strict triple-consensus set)
  3. Lineage context             (Step 8  - obj3_lineage_distribution.csv)
  4. Evolutionary conservation   (Step 9  - obj3_step9_candidate_conservation.csv)
  5. Structural/functional context (Step 10 - obj3_step10_structural_context.csv)

NOTE ON THE ML-CONSENSUS DEFINITION USED HERE: this project's saved
obj2_consensus_features.csv (11 rows, ranked by mean rank across RF/XGBoost/
EBM) uses a different, more permissive definition than the "5-feature"
strict triple-intersection (present in SHAP top-100 AND RF top-30 AND EBM
top-30 simultaneously) already used in this project's finalised Phase 4B/4C
reports. This script recomputes the STRICT definition directly from Step
4-6's own real top-ranking files, for consistency with those already-
delivered reports - confirmed to independently reproduce the same 5
features (katG, rpoB, embB x2, inhA) on this project's real, final data
before being trusted here.

Inputs expected (all in the current working directory, or pass -indir):
    obj1_compensatory_fisher.csv        Step 2
    obj2_shap_top100_annotated.csv      Step 5
    obj2_rf_top30_features.csv          Step 4
    obj2_ebm_all_terms.csv              Step 6
    obj3_lineage_distribution.csv,
    obj3_lineage_not_analyzable.csv     Step 8
    obj3_step9_candidate_conservation.csv  Step 9
    obj3_step10_structural_context.csv  Step 10

Output:
    obj_final_master_evidence_table.csv   1,209 rows, one per Step 2
                                           significant candidate, every layer
                                           joined, with an evidence_tier column
    obj_final_evidence_summary.csv        tier counts and a few headline stats
"""
import argparse
import os

import pandas as pd

pd.set_option("display.width", 160)


def load_ml_consensus(indir):
    """Recompute the strict SHAP-top100 ∩ RF-top30 ∩ EBM-top30(main effects)
    consensus set directly from Step 4-6's own real ranking files - see
    module docstring for why this, not the saved 11-row mean-rank file, is
    the definition used here."""
    shap100 = pd.read_csv(os.path.join(indir, "obj2_shap_top100_annotated.csv"))
    rf30 = pd.read_csv(os.path.join(indir, "obj2_rf_top30_features.csv"))
    ebm_all = pd.read_csv(os.path.join(indir, "obj2_ebm_all_terms.csv"))
    ebm_main30 = ebm_all[ebm_all["term_type"] == "main_effect"].sort_values("rank").head(30)

    shap_rank = dict(zip(shap100["variant_id"], shap100["rank"]))
    rf_rank = dict(zip(rf30["variant_id"], rf30["rank"]))
    ebm_rank = dict(zip(ebm_main30["feature_a"], ebm_main30["rank"]))

    consensus_set = set(shap_rank) & set(rf_rank) & set(ebm_rank)
    print(f"ML consensus (SHAP top100 ∩ RF top30 ∩ EBM top30 main effects): "
          f"{len(consensus_set)} features")
    return shap_rank, rf_rank, ebm_rank, consensus_set


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-indir", default=".", help="directory containing all Step 2-10 output files")
    ap.add_argument("-outdir", default=".", help="directory to write the master table to")
    args = ap.parse_args()

    # Layer 1: Step 2 significance + Step 8's own position resolution
    # Base population (gene+change+variant_id+resolution flag) comes from Step 8's
    # own two output files, which already solved the gene+change -> variant_id
    # resolution problem (Phase 5A Step 8 report, Section 3.2) - reused here
    # rather than re-deriving it. Step 2's own statistics are pulled separately,
    # directly from obj1_compensatory_fisher.csv (gene+change keyed, no variant_id
    # column at all), since Step 8's two files use inconsistent column names for
    # these stats between themselves (obj3_lineage_distribution.csv renames Step 2's
    # own padj_BH to padj_BH_step2; obj3_lineage_not_analyzable.csv keeps it as
    # padj_BH) - getting them from one single, consistent source avoids that
    # mismatch rather than trying to reconcile two different naming conventions.
    lineage = pd.read_csv(os.path.join(args.indir, "obj3_lineage_distribution.csv"))
    not_an = pd.read_csv(os.path.join(args.indir, "obj3_lineage_not_analyzable.csv"))

    analysable = lineage[["gene", "change", "drug_context", "variant_id"]].copy()
    analysable["step8_analysable"] = True
    not_analysable = not_an[["gene", "change", "drug_context", "variant_id"]].copy()
    not_analysable["step8_analysable"] = False

    master = pd.concat([analysable, not_analysable], ignore_index=True)
    print(f"Step 2 + Step 8 population: {len(master)} significant candidates "
          f"({master['step8_analysable'].sum()} analysable / "
          f"{(~master['step8_analysable']).sum()} not analysable)")

    fisher_sig = pd.read_csv(os.path.join(args.indir, "obj1_compensatory_fisher.csv"))
    fisher_sig = fisher_sig[fisher_sig["padj_BH"] < 0.05]
    step2_stat_cols = ["gene", "change", "compensatory_mechanism", "evidence",
                       "mdr_frequency", "non_mdr_frequency", "enrichment_ratio",
                       "pvalue", "padj_BH", "odds_ratio_fisher", "log2_odds_ratio"]
    step2_stat_cols = [c for c in step2_stat_cols if c in fisher_sig.columns]
    master = master.merge(
        fisher_sig[step2_stat_cols].rename(
            columns={c: f"step2_{c}" for c in step2_stat_cols if c not in ("gene", "change")}),
        on=["gene", "change"], how="left")

    # - Layer 2: ML importance (Steps 4-6) -
    shap_rank, rf_rank, ebm_rank, consensus_set = load_ml_consensus(args.indir)
    master["step56_shap_rank"] = master["variant_id"].map(shap_rank)
    master["step4_rf_rank"] = master["variant_id"].map(rf_rank)
    master["step6_ebm_rank"] = master["variant_id"].map(ebm_rank)
    master["step456_in_ml_consensus"] = master["variant_id"].isin(consensus_set)

    # Layer 3: Step 8 lineage-restriction detail (analysable candidates only)
    # A handful of genomic positions (8 in this project's real data) carry two
    # distinct TB-Profiler (gene, change) labels resolving to the same matrix
    # column - both are kept as separate rows above (they are independently
    # significant Step 2 candidates), but the position-level data being merged
    # in from here on (lineage detail, conservation, structure) is a property
    # of the POSITION, identical regardless of which label is attached, so the
    # lookup table itself is deduplicated by variant_id first - otherwise a
    # shared position's data would be joined in twice for one label and zero
    # times for the other, or compound across successive merges.
    lin_extra_cols = ["variant_id", "overall_carriage_pct", "lineage_restricted", "padj_lineage_BH"]
    lin_extra_cols = [c for c in lin_extra_cols if c in lineage.columns]
    lineage_dedup = lineage[lin_extra_cols].drop_duplicates(subset="variant_id")
    master = master.merge(
        lineage_dedup.rename(
            columns={c: f"step8_{c}" for c in lin_extra_cols if c != "variant_id"}),
        on="variant_id", how="left")

    inconsistent_path = os.path.join(args.indir, "obj3_lineage_inconsistent_mdr_direction.csv")
    if os.path.exists(inconsistent_path):
        inconsistent = pd.read_csv(inconsistent_path)
        inconsistent_ids = set(inconsistent["variant_id"]) if "variant_id" in inconsistent.columns else set()
    else:
        inconsistent_ids = set()
    master["step8_mdr_direction_inconsistent"] = master["variant_id"].isin(inconsistent_ids)

    n_step8 = master["step8_analysable"].sum()
    print(f"Step 8: {n_step8} of {len(master)} candidates have lineage context "
          f"({len(master) - n_step8} not analysable in the ML matrix - see Step 8's own report)")

    # Layer 4: Step 9 evolutionary conservation
    cons9 = pd.read_csv(os.path.join(args.indir, "obj3_step9_candidate_conservation.csv"))
    c9_cols = ["variant_id", "coding_classification", "n_independent_origins", "n_losses",
               "n_carriers_in_tree", "convergence_pattern", "gene_dN_dS_event_based"]
    c9_cols = [c for c in c9_cols if c in cons9.columns]
    master = master.merge(
        cons9[c9_cols].drop_duplicates(subset="variant_id").rename(
            columns={c: f"step9_{c}" for c in c9_cols if c != "variant_id"}),
        on="variant_id", how="left")
    n_step9 = master["step9_convergence_pattern"].notna().sum()
    print(f"Step 9: {n_step9} of {len(master)} candidates have conservation data")

    #Layer 5: Step 10 structural/functional context
    struct10 = pd.read_csv(os.path.join(args.indir, "obj3_step10_structural_context.csv"))
    s10_cols = ["variant_id", "protein_residue_number", "wt_aa", "mut_aa",
                "grantham_distance", "grantham_tier", "structure_source",
                "structure_id", "rsa", "burial_tier"]
    s10_cols = [c for c in s10_cols if c in struct10.columns]
    master = master.merge(
        struct10[s10_cols].drop_duplicates(subset="variant_id").rename(
            columns={c: f"step10_{c}" for c in s10_cols if c != "variant_id"}),
        on="variant_id", how="left")
    n_step10_nonsyn = master["step10_grantham_distance"].notna().sum()
    n_step10_rsa = master["step10_rsa"].notna().sum()
    print(f"Step 10: {n_step10_nonsyn} nonsynonymous candidates biochemically scored, "
          f"{n_step10_rsa} with a verified structural RSA value")

    # Evidence tier assignment
    # NOTE: the strict ML-consensus set (step456_in_ml_consensus) is entirely
    # primary-DR-gene variants (katG, rpoB, embB, inhA) - confirmed directly:
    # zero of this project's real, final significant compensatory candidates
    # overlap with the SHAP/EBM/RF top rankings at all, in either direction.
    # This is not a gap in this script - it is the same finding already
    # established in the Phase 4B/4C reports (a compensatory mutation
    # correlated with, but co-occurring alongside, a primary-DR marker
    # contributes little additional information to a model that already
    # knows the primary marker is present). Gating the top evidence tier on
    # ML-consensus membership would therefore leave it permanently empty for
    # this population by construction, not because of anything about any
    # individual candidate - so ML ranking is reported as an informational
    # column only, and tiering instead uses the three signals that CAN
    # meaningfully differentiate compensatory candidates from each other:
    # lineage consistency, evolutionary convergence, and (where available)
    # structural context.
    def assign_tier(row):
        anomaly = bool(row.get("step8_mdr_direction_inconsistent", False))
        analysable = bool(row.get("step8_analysable", False))
        convergent = row.get("step9_convergence_pattern") == "convergent"
        lineage_restricted = bool(row.get("step8_lineage_restricted", False))

        if anomaly:
            return "Flagged anomaly - requires caveat"
        if not analysable:
            return "Not conservation-tested (unresolved position)"
        if convergent and not lineage_restricted:
            return "Strong, convergent, lineage-consistent"
        if convergent and lineage_restricted:
            return "Convergent but lineage-restricted"
        return "Single-origin (clonal inheritance only)"

    master["evidence_tier"] = master.apply(assign_tier, axis=1)

    tier_order = ["Flagged anomaly - requires caveat", "Strong, convergent, lineage-consistent",
                  "Convergent but lineage-restricted", "Single-origin (clonal inheritance only)",
                  "Not conservation-tested (unresolved position)"]
    master["evidence_tier"] = pd.Categorical(master["evidence_tier"], categories=tier_order, ordered=True)
    master = master.sort_values(["evidence_tier", "step2_padj_BH"]).reset_index(drop=True)

    out_path = os.path.join(args.outdir, "obj_final_master_evidence_table.csv")
    master.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(master)} rows, {len(master.columns)} columns)")

    #Summary
    tier_counts = master["evidence_tier"].value_counts().reindex(tier_order)
    print("\nEvidence tier breakdown:")
    print(tier_counts.to_string())

    summary_rows = [{"tier": t, "count": int(c), "pct_of_1209": round(100 * c / len(master), 1)}
                     for t, c in tier_counts.items()]
    summary_path = os.path.join(args.outdir, "obj_final_evidence_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
