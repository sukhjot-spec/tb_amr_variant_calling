"""
Step 9 module: load H37Rv reference, extract CDS sequences, and classify
SNPs as synonymous/nonsynonymous by codon context. Uses NCBI translation
table 11 (bacterial), matching the GFF's stated transl_table=11.

Covers two gene sets:
  COMPENSATORY_GENES  - the project's canonical 12-gene compensatory
                          panel. Gene-wide dN/dS (Step 9's Part B) is
                          scoped to exactly this panel. mmpR5/Rv0678 are
                          the same physical gene under two names (see
                          GENE_NAME_ALIASES below) and count as one entry.
  EXTRA_GENES         - genes outside the compensatory panel that still
                          need codon classification because individual
                          candidate positions from Step 2's Fisher test
                          fall inside them. Currently just rpoB: a small
                          number of the 629 named candidates (Step 8's
                          lineage-restricted, significant compensatory set)
                          are intragenic rpoB compensatory mutations
                          (compensating for rpoB's own RRDR resistance
                          mutations), which is a real, separate biological
                          category from the cross-gene compensatory panel.
                          These are classified for completeness (no
                          candidate left without a classification) but are
                          NOT included in gene-wide dN/dS, since that's
                          specifically scoped to the compensatory panel.

Validation performed on this classification (see conversation record):
against TB-Profiler's own p.-notation amino acid change annotations for
55 checked positions (48 compensatory-panel + 7 rpoB), 54/55 matched
exactly. The one mismatch (Chromosome_764206_T_C, rpoC) is a case where
this module's codon-level computation (GAT->GAC, both Asp, synonymous)
is independently verifiable against the standard genetic code and
disagrees with TB-Profiler's stated "p.Asp279Glu" label - treated here
as a TB-Profiler-side labeling anomaly, not a bug in this module, since
GAT/GAC are unambiguously synonymous under any standard codon table.
This specific 55-position spot-check was run before the mmpR5/Rv0678
alias fix and before Step 2's candidate count was corrected from 391 to
629 (see GENE_NAME_ALIASES and the step9 scripts' own module docstrings);
it is unaffected by either - none of the 55 checked positions were the
one Rv0678-labeled variant - but is noted here since it predates both
fixes and a fresh spot-check against the current 629-candidate set would
be a reasonable thing to do before treating Step 9's output as final.

SECOND CONFIRMED INSTANCE of the same anomaly class, found during Step 10
development while cross-checking this module's residue/amino-acid output
against TB-Profiler's change labels for all 79 nonsynonymous candidates
in the real, corrected 629-candidate set: Chromosome_3877553_C_T (rpoA)
computes as ref_codon=GAG, alt_codon=AAG, i.e. Glu319Lys (E319K) -
base_check_ok=True, confirming GAG genuinely is the real H37Rv sequence
at this position, and GAG/AAG are unambiguous Glu/Lys under the standard
genetic code - yet TB-Profiler's own label for this variant is
"p.Glu319Gln" (E319Q), which would require the codon's first base to
change G->C, not the G->A actually present in the VCF. 78 of the 79
nonsynonymous candidates matched TB-Profiler's label exactly; this is
the one exception, and like the rpoC case above, the discrepancy is
independently resolvable via the genetic code itself rather than being
ambiguous - this module's E319K is correct.
"""
import re
from Bio.Seq import Seq
from Bio.Data import CodonTable

import os

FASTA_PATH = os.path.expanduser("~/tb_pipeline/reference/H37Rv.fasta")
GFF_PATH = os.path.expanduser("~/tb_pipeline/reference/GCF_000195955.2_ASM19595v2_genomic.gff")

COMPENSATORY_GENES = ["rpoA", "rpoC", "ahpC", "kasA", "ndh", "gyrB", "gid",
                       "mmpR5", "mmpL5", "embR", "whiB7", "eis"]
# NOTE on mmpR5/Rv0678: the GFF has no gene= tag at this locus (only
# Name=Rv0678), so variant_metadata_with_genes.csv's GFF-fallback path
# labels variants here "Rv0678" while TB-Profiler-covered ones resolve to
# "mmpR5" - same physical gene, two names. This module deliberately keeps
# COMPENSATORY_GENES as a single canonical name (mmpR5) rather than listing
# both, because both step9 scripts build one GeneContext / one dN-dS row
# PER NAME - listing "Rv0678" separately here would silently split this
# gene's variants and codon-possible-sites calculation across two duplicate
# buckets instead of merging them (confirmed directly: doing so doubles
# this gene's entry in build_gene_contexts() with identical coordinates).
# Instead, both step9 scripts normalize "Rv0678" -> "mmpR5" via
# GENE_NAME_ALIASES (below) at the point they read
# variant_metadata_with_genes.csv, before anything ever checks membership
# in ALL_GENES or looks up a GeneContext by name. This mirrors the same
# underlying fix already applied to comparative_analysis.py and
# shap_analysis.ipynb (see those files for the fuller rationale), adapted
# here to fit this module's per-gene-context design instead of duplicating
# a flat set entry.

GENE_NAME_ALIASES = {"Rv0678": "mmpR5"}

EXTRA_GENES = ["rpoB"]

ALL_GENES = COMPENSATORY_GENES + EXTRA_GENES

# Genes with no gene= synonym tag in this GFF release, keyed by locus_tag
# instead (mmpR5 = Rv0678). Coordinates taken directly from the GFF's CDS
# record for that locus_tag.
MANUAL_CDS = {
    "mmpR5": (778990, 779487, "+"),
}

CODON_TABLE = CodonTable.unambiguous_dna_by_id[11]


def load_genome():
    seq = []
    with open(FASTA_PATH) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()


def load_cds_coords(genes=ALL_GENES):
    """Return {gene: (start_1based, end_1based, strand)}"""
    coords = {g: v for g, v in MANUAL_CDS.items() if g in genes}
    with open(GFF_PATH) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            attrs = parts[8]
            for g in genes:
                if g in coords:
                    continue
                if re.search(rf"gene={re.escape(g)}(;|$)", attrs):
                    coords[g] = (int(parts[3]), int(parts[4]), parts[6])
    return coords


def revcomp(s):
    return str(Seq(s).reverse_complement())


class GeneContext:
    """Holds CDS sequence (5'->3', coding strand) and coordinate mapping
    for one gene, to support codon lookup by genomic position."""

    def __init__(self, gene, start, end, strand, genome):
        self.gene = gene
        self.start = start  # 1-based, genomic, always start<end regardless of strand
        self.end = end
        self.strand = strand
        raw = genome[start - 1:end]  # genomic-orientation slice
        self.cds_seq = raw if strand == "+" else revcomp(raw)
        self.length = len(self.cds_seq)

    def genomic_pos_to_cds_index(self, pos):
        """0-based index into self.cds_seq (5'->3' coding orientation)."""
        if self.strand == "+":
            return pos - self.start
        else:
            return self.end - pos

    def in_cds(self, pos):
        return self.start <= pos <= self.end

    def classify_snp(self, pos, ref, alt):
        """Return dict with codon context and syn/nonsyn classification.
        ref/alt are as given on the genomic (+) strand."""
        if not self.in_cds(pos):
            return {"in_cds": False}

        idx = self.genomic_pos_to_cds_index(pos)
        codon_start = (idx // 3) * 3
        codon_pos_in_codon = idx % 3  # 0,1,2
        ref_codon = self.cds_seq[codon_start:codon_start + 3]
        if len(ref_codon) != 3:
            return {"in_cds": True, "error": "incomplete_codon"}

        # ref/alt given on + strand; convert to coding-strand base if gene is on -
        if self.strand == "+":
            coding_ref, coding_alt = ref, alt
        else:
            coding_ref, coding_alt = revcomp(ref), revcomp(alt)

        # sanity check: does coding_ref match the reference genome at this codon position?
        expected_base = ref_codon[codon_pos_in_codon]
        base_ok = (expected_base == coding_ref)

        alt_codon = (ref_codon[:codon_pos_in_codon] + coding_alt +
                     ref_codon[codon_pos_in_codon + 1:])

        try:
            ref_aa = str(Seq(ref_codon).translate(table=11))
            alt_aa = str(Seq(alt_codon).translate(table=11))
        except Exception:
            return {"in_cds": True, "error": "translation_failed"}

        is_syn = (ref_aa == alt_aa)
        return {
            "in_cds": True,
            "base_check_ok": base_ok,
            "codon_index": codon_start // 3,
            "pos_in_codon": codon_pos_in_codon,
            "ref_codon": ref_codon,
            "alt_codon": alt_codon,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "synonymous": is_syn,
        }

    def classify_variant_row(self, pos, ref, alt):
        """Convenience wrapper: full classification label + details for a
        variant, handling non-SNP (indel) inputs gracefully."""
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            return {"classification": "indel_not_classified", "in_cds": None}
        cls = self.classify_snp(pos, ref, alt)
        if not cls["in_cds"]:
            return {"classification": "noncoding_upstream", **cls}
        label = "synonymous" if cls["synonymous"] else "nonsynonymous"
        return {"classification": label, **cls}


def build_gene_contexts(genes=ALL_GENES):
    genome = load_genome()
    coords = load_cds_coords(genes)
    contexts = {}
    for g in genes:
        if g not in coords:
            print(f"WARNING: no CDS coords found for {g}")
            continue
        start, end, strand = coords[g]
        contexts[g] = GeneContext(g, start, end, strand, genome)
    return contexts


def possible_sites(gene_ctx):
    """Nei-Gojobori-style possible N/S site counts for a CDS.
    For each codon position, for each of the 3 possible single-nt
    substitutions, determine the fraction that are synonymous vs
    nonsynonymous; sum (fraction synonymous) as S-sites and
    (fraction nonsynonymous) as N-sites, per codon position, over the CDS."""
    seq = gene_ctx.cds_seq
    n_codons = len(seq) // 3
    bases = "ACGT"
    S_sites = 0.0
    N_sites = 0.0
    for c in range(n_codons):
        codon = seq[c * 3:c * 3 + 3]
        if len(codon) != 3 or any(b not in bases for b in codon):
            continue
        try:
            ref_aa = str(Seq(codon).translate(table=11))
        except Exception:
            continue
        if ref_aa == "*":
            continue  # skip stop codon (shouldn't be internal, but be safe)
        for pos in range(3):
            syn_count = 0
            total = 0
            for b in bases:
                if b == codon[pos]:
                    continue
                alt_codon = codon[:pos] + b + codon[pos + 1:]
                try:
                    alt_aa = str(Seq(alt_codon).translate(table=11))
                except Exception:
                    continue
                total += 1
                if alt_aa == ref_aa:
                    syn_count += 1
            if total > 0:
                S_sites += syn_count / total
                N_sites += (total - syn_count) / total
    return {"S_sites": S_sites, "N_sites": N_sites, "n_codons": n_codons}


if __name__ == "__main__":
    contexts = build_gene_contexts()
    print(f"Built gene contexts for {len(contexts)}/{len(ALL_GENES)} genes "
          f"({len(COMPENSATORY_GENES)} compensatory panel + {len(EXTRA_GENES)} extra)")
    for g, ctx in contexts.items():
        tag = "compensatory panel" if g in COMPENSATORY_GENES else "extra (rpoB intragenic)"
        print(f"  {g}: {ctx.length} bp CDS, strand {ctx.strand}  [{tag}]")
