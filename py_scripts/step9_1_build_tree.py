"""
Step 9, script 1 of 2 (formerly script 2 of 3 - step9_1_build_genotype_matrix.py
has been dropped entirely; see load_step1_data() below for why).

genotype source -> phylogenetic tree.

Builds a binary-character FASTA alignment (one sequence per sample, one
character per site, '0'/'1') from Step 1's own X_array.npy, then runs
IQ-TREE2 to get a tree topology covering the full cohort.

WHY THIS NO LONGER REBUILDS A GENOTYPE MATRIX FROM THE RAW VCF: the
previous step9_1_build_genotype_matrix.py re-derived a binarized matrix
from merged_prefilt.vcf.gz using the identical 0/0->0, anything-else->1
rule Step 1's build_ml_dataset.py already applied and saved to
X_array.npy - duplicated work for zero benefit. An earlier version of
this docstring additionally claimed this rebuild "silently reintroduced
SRR11922476, the one sample Step 1 excluded" and created a cohort-size
mismatch against the rest of the project. That claim was checked directly
against the current pipeline and does not hold: Step 1's row-shift bug
(the one that caused SRR11922476 to appear excluded) was fixed earlier
in this project, and X_array.npy, y_labels.csv, obj3_lineage_distribution.csv,
and every other file from Step 1 onward now consistently reflect the
full, correct 1,858-sample cohort with zero exclusions - there is no
cohort-size inconsistency to avoid. The actual reason to use X_array.npy
directly, which still fully holds, is simpler: it is Step 1's own
authoritative output, already correctly built, and there is nothing to
gain from re-deriving the identical matrix a second time from the raw
VCF. X_array.npy has everything this script and script 2 need for
presence/absence lookups.

CIRCULARITY FIX - why the gene-set/rpoB positions are excluded from
tree-building: script 2 tests exactly those positions for homoplasy via
Fitch parsimony ON this tree. If the tree itself is built by minimizing
character changes across an alignment that INCLUDES those same positions,
the topology gets partly shaped to explain them economically - which
then makes those specific positions look artificially more clonal (fewer
independent origins) when re-tested against a tree that was already
bent toward accommodating their pattern. This is standard practice in
comparative-methods phylogenetics (never test a character for convergence
against a tree built partly from that same character) and was flagged as
a real problem here, not a theoretical one, since MDR status is itself
phylogenetically structured and these are exactly the positions most
likely to have influenced the topology if left in.
Fix: build the tree from the background sites only (94,583 minus the
gene-set/rpoB positions, identified via variant_metadata_with_genes.csv's
own gene column - the same source of truth script 2 uses, including the
mmpR5/Rv0678 alias normalization - see codon_classify.py's
GENE_NAME_ALIASES), then test the held-out positions against the
resulting tree in script 2. This fully breaks the circularity regardless
of which tree-building tool is used. The exact count of excluded
positions is computed fresh and printed at runtime by
get_excluded_positions() below, rather than hardcoded here, since it can
shift slightly if the upstream gene-annotation pipeline changes - as it
already has once in this project's history.

DESIGN NOTE - why a maximum-parsimony tree, not maximum-likelihood:
IQ-TREE's default "-fast" workflow builds a quick starting parsimony
tree, then refines it under an ML substitution model. On a single CPU
core, the ML refinement's model-parameter-estimation loop did not
converge in any reasonable number of resumed runs (each resume redid
the same iterations, since IQ-TREE's checkpoint only saves at phase
boundaries, not per-iteration - so resuming was not making progress).

The parsimony starting tree, however, completed in ~90 seconds and is
methodologically well-matched to this project's purpose: Step 9's
downstream homoplasy analysis (fitch.py) is itself a maximum-parsimony
method. There is no benefit to an ML-refined tree for a parsimony-based
ancestral reconstruction, so this script stops at the parsimony tree
rather than waiting on ML convergence.

If more compute is available (multi-core machine), you can let IQ-TREE
run to full ML convergence instead by removing "-fast" or increasing
"-nt" to use more threads - complete_tree() (below) already prefers
full_tree.treefile over full_tree.parstree automatically if it finds
one, so no manual file substitution is needed either way.

Requires: iqtree2 (tested with 2.0.7), ete3 (for complete_tree()'s
identical-taxa reattachment - see that function's docstring).

Output:
    binary_alignment.fasta   one sequence per sample, '0'/'1' per site,
                             EXCLUDING the gene-set/rpoB positions (count
                             printed at runtime, see get_excluded_positions())
    excluded_positions.txt   the POS values held out of tree-building
                             (for the next script / for anyone auditing this)
    full_tree.parstree       the maximum-parsimony tree (Newick), IQ-TREE's
                             raw output - NOTE: missing any taxa IQ-TREE
                             found genotypically identical to another taxon
                             (confirmed on this project's real data: 31 of
                             1,858 samples). Left untouched for auditing;
                             do not read this file directly in script 2.
    full_tree.log            IQ-TREE's run log (also the source
                             complete_tree() parses to find dropped taxa)
    full_tree_complete.parstree  the tree to actually use downstream -
                             full_tree.parstree/.treefile with any dropped-
                             identical taxa reattached (zero branch length,
                             as a sibling of whichever taxon they matched).
                             See complete_tree() and
                             reattach_identical_sequences() below for why
                             this step exists and is not optional.
"""
import os
import subprocess
import csv
from collections import defaultdict
import numpy as np

from codon_classify import ALL_GENES, GENE_NAME_ALIASES

STEP1 = os.path.expanduser("~/tb_pipeline/ml_outputs/step1_dataset")
OUTDIR = os.path.expanduser("~/tb_pipeline/ml_outputs/step9_conservation")
os.makedirs(OUTDIR, exist_ok=True)
VARIANT_METADATA_PATH = os.path.join(STEP1, "variant_metadata_with_genes.csv")


def load_step1_data():
    """Read Step 1's own matrix + ID files directly, rather than
    rebuilding anything from the raw VCF (see module docstring).
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


def get_excluded_positions():
    """POS values belonging to the 12-gene compensatory panel + rpoB,
    per variant_metadata_with_genes.csv's own gene column - the same
    source of truth script 2 uses to select which variants to classify
    and homoplasy-test. This is deliberately NOT limited to CDS
    coordinate ranges: it also excludes promoter/upstream positions
    already annotated to these genes (e.g. "ahpC c.-88G>A"), since those
    are still part of the character set being tested downstream and
    must be held out of tree-building for the same reason.

    Gene names are normalized via GENE_NAME_ALIASES (currently just
    Rv0678 -> mmpR5, same physical gene under two names depending on
    which annotation source resolved a given variant) before the
    membership check, so a GFF-fallback-resolved Rv0678 variant is
    correctly excluded exactly like a TB-Profiler-resolved mmpR5 one -
    without this, that variant would silently stay in the "background"
    alignment, reintroducing a small amount of the exact circularity
    this whole exclusion mechanism exists to prevent."""
    excluded = set()
    with open(VARIANT_METADATA_PATH) as f:
        for row in csv.DictReader(f):
            gene = GENE_NAME_ALIASES.get(row["gene"], row["gene"])
            if gene in ALL_GENES:
                excluded.add(int(row["POS"]))
    return excluded


def build_alignment():
    X, positions, sample_ids = load_step1_data()

    excluded = get_excluded_positions()
    keep_mask = np.array([p not in excluded for p in positions])
    n_excluded_found = len(positions) - keep_mask.sum()
    print(f"Excluding {n_excluded_found} of {len(excluded)} gene-set/rpoB positions "
          f"found in the matrix from tree-building "
          f"({keep_mask.sum()} background sites remain)")
    if n_excluded_found != len(excluded):
        print(f"  NOTE: {len(excluded)} positions are flagged for exclusion but only "
              f"{n_excluded_found} were found in this matrix - expected if some "
              f"gene-set variants didn't survive the 1-99% AF filter (Step 2's "
              f"significant-candidate count and the raw gene-set-variant count "
              f"in this matrix are expected to differ for this same reason).")

    with open(f"{OUTDIR}/excluded_positions.txt", "w") as f:
        f.write("\n".join(str(p) for p, keep in zip(positions, keep_mask) if not keep))

    X_bg = X[:, keep_mask]  # samples x background_sites - no transpose needed,
                            # X_array.npy is already samples x sites
    char_mat = (X_bg + ord("0")).astype(np.uint8)

    with open(f"{OUTDIR}/binary_alignment.fasta", "wb") as f:
        for i, sid in enumerate(sample_ids):
            f.write(b">" + sid.encode() + b"\n")
            f.write(char_mat[i].tobytes())
            f.write(b"\n")

    print(f"Wrote binary_alignment.fasta: {len(sample_ids)} sequences x {X_bg.shape[1]} "
          f"background sites (gene-set/rpoB positions excluded)")


def run_iqtree(poll_interval=5, grace_period=15):
    """Runs IQ-TREE2 and returns once the parsimony tree is ready, without
    requiring you to manually watch the log and interrupt it.

    Why this isn't just a blocking subprocess.run() call: on a single-core
    machine (this project's documented, anticipated setup), IQ-TREE's ML
    refinement phase after the parsimony tree completes does not converge
    in any practical amount of time (see module docstring's DESIGN NOTE).
    A plain blocking call would hang here indefinitely, and manually
    Ctrl+C-ing it kills this whole script - including complete_tree(),
    which would then never run. Confirmed directly: this is exactly what
    happens if you call this the naive way.

    Instead: launch IQ-TREE as a background process, poll for
    full_tree.parstree to appear (written once the parsimony construction
    step finishes, well before the ML phase that never converges), wait
    a short grace period for IQ-TREE to finish flushing its log, then
    terminate the still-running ML-refinement phase ourselves. On a
    multi-core machine where you've removed "-fast" for full ML
    convergence, this will instead simply wait for the process to exit
    normally once full_tree.treefile appears - either way you end up
    back in this function once there is something for complete_tree()
    to read.
    """
    import time as _time

    cmd = (
        f"iqtree2 -s {OUTDIR}/binary_alignment.fasta -st BIN -m GTR2 -fast "
        f"-nt 1 -pre {OUTDIR}/full_tree -redo"
    )
    print(f"$ {cmd}")
    parstree = f"{OUTDIR}/full_tree.parstree"
    treefile = f"{OUTDIR}/full_tree.treefile"
    if os.path.exists(parstree):
        os.remove(parstree)

    proc = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Started IQ-TREE (pid {proc.pid}), polling every {poll_interval}s "
          f"for full_tree.parstree...")
    t0 = _time.time()
    while proc.poll() is None:
        if os.path.exists(treefile):
            print(f"  full_tree.treefile appeared after {_time.time()-t0:.0f}s "
                  f"(full ML run) - waiting for the process to exit normally.")
            proc.wait()
            break
        if os.path.exists(parstree):
            print(f"  full_tree.parstree appeared after {_time.time()-t0:.0f}s - "
                  f"waiting {grace_period}s for the log to finish flushing, then "
                  f"stopping the (unneeded) ML refinement phase.")
            _time.sleep(grace_period)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        _time.sleep(poll_interval)
    else:
        print(f"  IQ-TREE process exited on its own after {_time.time()-t0:.0f}s "
              f"(return code {proc.returncode}).")

    if not os.path.exists(parstree) and not os.path.exists(treefile):
        raise SystemExit(
            "Neither full_tree.parstree nor full_tree.treefile was produced - "
            "check full_tree.log for an IQ-TREE error (e.g. a malformed "
            "alignment) rather than assuming this is just slow."
        )


def reattach_identical_sequences(tree, alignment_path, sample_ids):
    """IQ-TREE drops any taxon whose character pattern is identical to
    another taxon's from the tree it actually constructs, planning to
    graft each one back on once the full run completes. On the
    single-core, interrupted-at-the-parsimony-stage workflow this
    project actually uses (see run_iqtree()'s NOTE), that final assembly
    step is never reached, so the tree is silently missing these taxa.

    This matters concretely, not just for tip-count completeness: a
    taxon dropped here is only guaranteed identical to its match across
    the BACKGROUND sites used for tree-building - it can still differ
    at any of the held-out gene-set/rpoB positions (excluded from
    tree-building specifically, see the module docstring's CIRCULARITY
    FIX section, and therefore never part of the comparison that made
    IQ-TREE call it "identical" in the first place). Silently dropping
    it would silently lose real carriage information for the Fitch
    homoplasy counts script 2 computes on exactly those positions.

    HOW THIS WORKS, AND WHY NOT BY PARSING full_tree.log: an earlier
    version of this function parsed IQ-TREE's log for lines naming which
    taxon is identical to which. That broke across IQ-TREE versions -
    confirmed directly: IQ-TREE 2.0.7 prints one "NOTE: X (identical to
    Y) is ignored but added at the end" line per dropped taxon; IQ-TREE
    2.4.0 instead prints a single summary count ("NOTE: 31 identical
    sequences (see below) will be ignored...") with no per-pair mapping
    in the log text at all. Rather than chase IQ-TREE's log format
    across versions, this instead computes identical-sequence groups
    directly from binary_alignment.fasta (which this project's own code
    controls, so its format is stable) and cross-references against
    which taxa the tree is actually missing (computed from the tree
    itself and sample_ids, not from any count IQ-TREE reports) -
    fully independent of IQ-TREE's internal grouping algorithm or log
    wording. Confirmed on real data that this matters, not just as a
    theoretical robustness improvement: on one real run, IQ-TREE's own
    reported count (31 dropped) did not match the number of samples
    found in multi-member identical-sequence groups by direct
    computation (38) - IQ-TREE evidently keeps more than one
    representative for some duplicate clusters. This function sidesteps
    needing to understand why: it only needs to know, for each taxon
    the tree is actually missing, whether ANY other member of its
    identical-sequence group is present in the tree to attach it next
    to - and since every present member of a group is, by construction,
    identical to every other member across the tree-building sites, it
    does not matter which present member gets used as the attachment
    point.

    Idempotent: a taxon already present in the tree (e.g. because you
    let IQ-TREE run to full ML convergence and it completed its own
    re-grafting step already) is left alone.
    """
    tree_tips = set(tree.get_leaf_names())
    missing = set(sample_ids) - tree_tips
    if not missing:
        print("Tree already covers the full cohort - nothing to reattach.")
        return tree

    seq_to_samples = defaultdict(list)
    with open(alignment_path) as f:
        sample = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                sample = line[1:]
            else:
                seq_to_samples[line].append(sample)
    sample_to_group = {}
    for members in seq_to_samples.values():
        if len(members) > 1:
            for m in members:
                sample_to_group[m] = members

    n_attached, n_unresolvable = 0, 0
    for m in sorted(missing):
        group = sample_to_group.get(m)
        anchor = next((x for x in group if x in tree_tips), None) if group else None
        if anchor is None:
            n_unresolvable += 1
            print(f"  WARNING: {m} is missing from the tree but has no identical-"
                  f"sequence match present in the tree to attach it next to - "
                  f"this is a real gap, not just a dropped duplicate. Investigate.")
            continue
        node = tree.search_nodes(name=anchor)[0]
        node.up.add_child(name=m, dist=0.0)
        tree_tips.add(m)
        n_attached += 1

    print(f"Reattached {n_attached}/{len(missing)} taxa missing from the tree "
          f"via identical-sequence matching ({n_unresolvable} unresolvable)")
    return tree


def complete_tree():
    """Load whichever tree run_iqtree() produced (prefers the fully-converged
    full_tree.treefile if present, falls back to the parsimony-only
    full_tree.parstree - see run_iqtree()'s NOTE), reattach any taxa
    missing from it (see reattach_identical_sequences()), and write the
    result to full_tree_complete.parstree. This is the file script 2
    should read, NOT full_tree.parstree/full_tree.treefile directly -
    those are left untouched as IQ-TREE's raw output, for anyone
    auditing this step."""
    from ete3 import Tree as ETree
    treefile = f"{OUTDIR}/full_tree.treefile"
    parstree = f"{OUTDIR}/full_tree.parstree"
    if os.path.exists(treefile):
        print(f"Using {treefile} (full ML-converged run found)")
        tree = ETree(treefile)
    elif os.path.exists(parstree):
        print(f"Using {parstree} (parsimony-only - no full_tree.treefile found, "
              f"consistent with an interrupted single-core run)")
        tree = ETree(parstree)
    else:
        raise SystemExit(
            f"Neither {treefile} nor {parstree} found - did run_iqtree() run "
            f"(or get interrupted before writing anything at all)?")

    with open(f"{STEP1}/sample_ids.txt") as f:
        sample_ids = [l.strip() for l in f if l.strip()]

    tree = reattach_identical_sequences(
        tree, f"{OUTDIR}/binary_alignment.fasta", sample_ids)

    expected = set(sample_ids)
    actual = set(tree.get_leaf_names())
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        print(f"  WARNING: tree does not cover the full cohort after reattachment - "
              f"{len(missing)} sample(s) missing, {len(extra)} unexpected tip(s). "
              f"Investigate before trusting script 2's output.")
        if missing:
            print(f"    missing: {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
    else:
        print(f"  Verified: all {len(expected)} samples are present as tree tips.")

    out_path = f"{OUTDIR}/full_tree_complete.parstree"
    tree.write(outfile=out_path, format=1)
    print(f"Wrote {out_path}")
    return tree


if __name__ == "__main__":
    build_alignment()
    run_iqtree()
    complete_tree()
