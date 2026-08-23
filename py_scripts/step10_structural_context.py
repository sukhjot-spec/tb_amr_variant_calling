#!/usr/bin/env python3
"""
step10_structural_context.py

Step 10 - Functional/structural significance of candidate compensatory
mutations. Takes Step 9's candidate_conservation table (629 candidates,
each already carrying a statistical-significance verdict from Step 2 and
an evolutionary-conservation verdict from Step 9) and adds a third,
independent layer: where does this mutation sit in the actual folded
protein, and does that location suggest it could plausibly matter
functionally?

For each NONSYNONYMOUS candidate (the only ones with a direct protein-
structural interpretation - synonymous and noncoding candidates pass
through with structural columns left blank, not dropped, since Step 11
still needs the full 629-row table):
  1. Resolve the gene's UniProt accession (dynamic lookup, not hardcoded
     - see GENE_LOCUS_TAGS and resolve_uniprot_accession() below for why)
  2. Acquire a structure: prefer an experimental PDB structure if one
     covers this protein, otherwise fall back to the AlphaFold DB
     predicted model (both are queried at runtime; nothing here assumes
     in advance which genes have which)
  3. Map the genomic position to a protein residue number, reusing
     codon_classify.py's GeneContext (the same gene-coordinate machinery
     Step 9 already validated) rather than re-deriving it
  4. Compute two independent, complementary severity signals:
       a. Structural: relative solvent accessibility (RSA) at that
          residue via the Shrake-Rupley algorithm (pure Python, part of
          biopython - no external DSSP binary required, unlike
          secondary-structure assignment, which does need one and is
          deliberately NOT attempted here for that reason)
       b. Biochemical: a Grantham-distance-style physicochemical
          difference between the wild-type and mutant amino acid
          (hydrophobicity, volume, polarity) - computable for every
          nonsynonymous candidate regardless of whether a structure was
          found at all, so structure-acquisition failures don't leave a
          candidate with zero information

IMPORTANT - WHAT WAS AND WASN'T TESTED DIRECTLY BEFORE THIS WAS HANDED
TO YOU: the genomic-to-residue mapping (step 3 above) and the
biochemical severity metric (step 4b) need no network access and were
both tested directly against real data (this project's actual candidate
list and actual GFF/genome files) before delivery - see the
accompanying test log. The UniProt/PDB/AlphaFold lookups (steps 1-2) and
the SASA computation (step 4a) need outbound internet access this
sandbox does not have, so those paths are implemented defensively (clear
errors on failure, nothing silently guessed) but could only be verified
against real external services in your own environment, not mine. Watch
the first real run's console output closely, and see VERIFICATION.md
(if provided alongside this script) for exactly what to check.

Inputs expected:
    ~/tb_pipeline/ml_outputs/step9_conservation/obj3_step9_candidate_conservation.csv
        Step 9's output - 629 candidates, each with a coding_classification
        column this script uses to select which rows get structural analysis
    ~/tb_pipeline/reference/H37Rv.fasta, GCF_000195955.2_ASM19595v2_genomic.gff
        via codon_classify.py, same as Steps 2/5/9
    codon_classify.py must be importable (same directory or on PYTHONPATH)

Outputs (written to OUTDIR):
    obj3_step10_structural_context.csv
        One row per candidate (all 629, same row count as the Step 9
        input - nothing is dropped). Nonsynonymous rows carry the full
        structural annotation; synonymous/noncoding/indel rows carry
        the identifying columns with structural fields left blank, so
        Step 11 can left-join this onto the Step 9 table with no row-
        count surprises.
    obj3_step10_structure_provenance.csv
        One row per gene actually analysed: which UniProt accession was
        used, whether the structure came from PDB or AlphaFold, the
        specific structure ID, and (for PDB) resolution/method - so any
        structural claim in Step 11 or a report can be traced back to a
        specific, named structure rather than an opaque "AlphaFold said so".

Requires: biopython (already installed for Step 9), requests.
"""
import os
import sys
import csv
import json
import time
import urllib.request
import urllib.error
import urllib.parse

from codon_classify import build_gene_contexts, COMPENSATORY_GENES, EXTRA_GENES, ALL_GENES

STEP9 = os.path.expanduser("~/tb_pipeline/ml_outputs/step9_conservation")
OUTDIR = os.path.expanduser("~/tb_pipeline/ml_outputs/step10_structural")
os.makedirs(OUTDIR, exist_ok=True)
CANDIDATES_PATH = os.path.join(STEP9, "obj3_step9_candidate_conservation.csv")
STRUCTURE_CACHE_DIR = os.path.join(OUTDIR, "structures")
os.makedirs(STRUCTURE_CACHE_DIR, exist_ok=True)

# Rv locus tags: extracted directly from this project's own reference GFF
# (GCF_000195955.2_ASM19595v2_genomic.gff), the same file codon_classify.py
# already trusts as its single source of truth - NOT typed from memory.
# mmpR5 has no gene= tag in this GFF release (only Name=Rv0678, see
# codon_classify.py's GENE_NAME_ALIASES); its locus tag is taken from the
# GFF's Name= attribute at that CDS instead.
GENE_LOCUS_TAGS = {
    "rpoA": "Rv3457c", "rpoB": "Rv0667", "rpoC": "Rv0668",
    "ahpC": "Rv2428", "kasA": "Rv2245", "ndh": "Rv1854c",
    "gyrB": "Rv0005", "gid": "Rv3919c", "mmpR5": "Rv0678",
    "mmpL5": "Rv0676c", "embR": "Rv1267c", "whiB7": "Rv3197A",
    "eis": "Rv2416c",
}

# UniProt accessions confirmed directly against RCSB PDB structure records
# during this script's development (real cryo-EM RNAP structures naming
# these three chains explicitly) - used as a cache to skip a redundant
# network round-trip for these three specifically; every other gene is
# resolved fresh via resolve_uniprot_accession() below, and these three
# would resolve to the same values that way too if this cache were removed.
KNOWN_UNIPROT_ACCESSIONS = {
    "rpoA": "P9WGZ1",
    "rpoB": "P9WGY9",
    "rpoC": "P9WGY7",
}


def resolve_uniprot_accession(gene, timeout=15):
    """Query UniProt's REST API for this gene's H37Rv accession, keyed by
    the GFF-verified Rv locus tag (organism_id 83332 = M. tuberculosis
    H37Rv specifically, not just any Mycobacterium tuberculosis strain).
    Prefers a reviewed (Swiss-Prot) entry if one exists.

    Returns the accession string, or None with a printed warning if the
    lookup fails for any reason (no network, no match, API change) -
    never guesses or falls back to a hardcoded value silently."""
    if gene in KNOWN_UNIPROT_ACCESSIONS:
        return KNOWN_UNIPROT_ACCESSIONS[gene]

    locus = GENE_LOCUS_TAGS.get(gene)
    if locus is None:
        print(f"  WARNING: no known locus tag for {gene}, cannot resolve UniProt accession")
        return None

    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene:{locus}+AND+organism_id:83332"
        "&format=json&fields=accession,reviewed"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  WARNING: UniProt lookup failed for {gene} ({locus}): {e}")
        return None

    results = data.get("results", [])
    if not results:
        print(f"  WARNING: UniProt returned no entries for {gene} ({locus})")
        return None

    reviewed = [r for r in results if r.get("entryType", "").startswith("UniProtKB reviewed")]
    chosen = reviewed[0] if reviewed else results[0]
    accession = chosen["primaryAccession"]
    if not reviewed:
        print(f"  NOTE: {gene} ({locus}) has no reviewed Swiss-Prot entry; "
              f"using unreviewed accession {accession}")
    return accession


def find_pdb_structure(uniprot_accession, timeout=15, max_candidates=5):
    """Search RCSB PDB for experimental structures whose UniProt cross-
    reference matches this accession. Returns a LIST of up to
    max_candidates PDB IDs, best-resolution first, or an empty list if
    nothing is found - callers should fall back to AlphaFold DB in that
    case, not treat this as an error.

    Returns a list, not a single "best" ID, because "best resolution"
    and "most useful for this gene's candidates" are not the same thing
    - confirmed as a real, not hypothetical, problem: for rpoC (1,316
    residues), the best-resolution hit (5UH7) turned out to be a small
    24-154 fragment that covers NONE of this project's real 20 rpoC
    candidates (162-1252), while a full-length structure exists (8EOT,
    solved by cryo-EM, therefore lower nominal resolution, therefore
    ranked below the fragment by resolution alone). See
    select_best_pdb_structure() below, which downloads each candidate in
    turn and picks by ACTUAL coverage of the residues this gene's
    candidates actually need, not resolution."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers"
                             ".reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_accession,
            },
        },
        "return_type": "entry",
        "request_options": {
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "results_content_type": ["experimental"],
            "return_all_hits": False,
            "paginate": {"start": 0, "rows": max_candidates},
        },
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 204:
            return []  # RCSB's documented "no matches" response
        print(f"    PDB search failed for {uniprot_accession}: HTTP {e.code}")
        return []
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    PDB search failed for {uniprot_accession}: {e}")
        return []

    if not body.strip():
        # RCSB returns an empty 200 body (not a JSON object) when there are
        # zero matches - this is the normal "nothing found" case, not an
        # error, and callers should fall back to AlphaFold DB silently.
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        print(f"    PDB search returned unparseable data for {uniprot_accession}: {e}")
        return []

    hits = data.get("result_set", [])
    return [h["identifier"] for h in hits[:max_candidates]]


def structure_residue_coverage(structure_path, file_format, residue_aa_pairs):
    """How many of residue_aa_pairs (list of (residue_number, expected_aa)
    tuples) are actually modeled in this structure WITH THE CORRECT AMINO
    ACID IDENTITY - not just "some residue exists at that number in some
    chain". Used to pick between multiple candidate structures for the
    same protein by real, verified coverage, rather than by resolution
    or chain count alone.

    Checking identity, not just residue-number presence, is not optional
    caution here - it was proven necessary on this project's own real
    data. An earlier version only checked whether a residue existed at
    that number in ANY chain, which silently overcounts for multi-
    subunit structures: e.g. 7KIN (used for both rpoA and rpoC, an RNA
    polymerase complex with both proteins as separate chains in one
    file) can have residue number N present in the rpoA chain and
    residue number N present, coincidentally, in the rpoC chain too -
    two entirely different amino acids that happen to share a position
    number. A real run's own numbers exposed this directly: per-gene
    coverage summed to 79/79 "covered" while only 58 residues actually
    got a computed RSA value in compute_residue_rsa() (which DOES check
    identity) - a 21-residue gap traceable to exactly this.

    Returns (n_covered, n_total)."""
    from Bio.PDB import PDBParser, MMCIFParser

    parser = MMCIFParser(QUIET=True) if file_format == "cif" else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", structure_path)
    except Exception:
        return 0, len(residue_aa_pairs)

    modeled = {}  # resseq -> set of amino acids found there, across all chains
    for chain in structure[0]:
        for residue in chain:
            het, resseq, icode = residue.id
            aa = AA3TO1.get(residue.resname)
            if het == " " and aa:
                modeled.setdefault(resseq, set()).add(aa)

    covered = sum(1 for r, aa in residue_aa_pairs if aa and aa in modeled.get(r, set()))
    return covered, len(residue_aa_pairs)


def select_best_pdb_structure(candidate_pdb_ids, residue_aa_pairs, timeout=60):
    """Download each candidate PDB structure in turn (best-resolution
    first) and keep whichever actually covers the most of the residues
    this gene's real candidates need - see structure_residue_coverage()
    for why coverage means identity-verified coverage, not just residue-
    number presence. Stops early if a candidate covers every needed
    residue - no reason to keep downloading once nothing more can be
    gained. Returns (path, format, pdb_id, coverage_str), or
    (None, None, None, None) if every candidate failed to download."""
    best = None  # (n_covered, path, fmt, pdb_id)
    for pdb_id in candidate_pdb_ids:
        path, fmt = _fetch_pdb_entry(pdb_id, timeout)
        if path is None:
            continue
        n_covered, n_total = structure_residue_coverage(path, fmt, residue_aa_pairs)
        print(f"    {pdb_id}: covers {n_covered}/{n_total} of this gene's candidate residues")
        if best is None or n_covered > best[0]:
            best = (n_covered, path, fmt, pdb_id)
        if n_covered == n_total:
            break  # full coverage - no candidate can do better than this
    if best is None:
        return None, None, None, None
    n_covered, path, fmt, pdb_id = best
    return path, fmt, pdb_id, f"{n_covered}/{len(residue_aa_pairs)}"


def fetch_structure_file(pdb_id=None, uniprot_accession=None, timeout=60):
    """Download either a PDB structure or, if no pdb_id is given, the
    AlphaFold DB predicted model for uniprot_accession. Caches to
    STRUCTURE_CACHE_DIR so a rerun doesn't re-download.

    Returns (local_path, file_format) where file_format is "pdb" or
    "cif", or (None, None) on failure - callers must pass file_format
    through to compute_residue_rsa(), which needs to know which parser
    to use.
    """
    if pdb_id:
        return _fetch_pdb_entry(pdb_id, timeout)
    else:
        return _fetch_alphafold_entry(uniprot_accession, timeout)


def _fetch_pdb_entry(pdb_id, timeout):
    """Try legacy PDB format first, fall back to mmCIF. A meaningful
    fraction of real PDB entries - anything with >62 chains, >99,999
    atoms, multi-character chain IDs, or sufficiently complex topology -
    are ONLY distributed in mmCIF, never legacy PDB (this is a real,
    common, well-documented RCSB limitation, not an edge case: confirmed
    directly against this project's real run, where 2 of 7 experimental
    structures found - 6X6I for eis, 6Y2I for kasA - 404'd on .pdb but
    are almost certainly available as .cif)."""
    for ext, fmt in [("pdb", "pdb"), ("cif", "cif")]:
        local_path = os.path.join(STRUCTURE_CACHE_DIR, f"{pdb_id}.{ext}")
        if os.path.exists(local_path):
            return local_path, fmt
        url = f"https://files.rcsb.org/download/{pdb_id}.{ext}"
        try:
            urllib.request.urlretrieve(url, local_path)
            if fmt == "cif":
                print(f"    (legacy .pdb format unavailable for {pdb_id} - "
                      f"used .cif instead, a known limitation for large/complex structures)")
            return local_path, fmt
        except (urllib.error.URLError, TimeoutError) as e:
            if os.path.exists(local_path):
                os.remove(local_path)  # don't cache a partial/failed download
            if ext == "pdb":
                continue  # try .cif next
            print(f"    Structure download failed for PDB:{pdb_id} (tried both "
                  f".pdb and .cif): {e}")
            return None, None
    return None, None


def _fetch_alphafold_entry(uniprot_accession, timeout):
    """Query AlphaFold DB's own prediction API for the authoritative,
    current download URL rather than constructing one from a guessed
    naming convention - confirmed necessary, not just extra caution:
    a direct guess (https://alphafold.ebi.ac.uk/files/AF-{accession}-
    F1-model_v4.pdb) 404'd on this project's real run for a real M.
    tuberculosis protein (ndh, P95160) despite AlphaFold DB's full-
    proteome coverage of M. tuberculosis, meaning the naming convention
    alone isn't reliable enough to hardcode. The prediction API returns
    the real, current URL for whichever accession it has, and clearly
    fails (empty result) for one it doesn't - a meaningful distinction
    a raw file-URL guess can't make (a 404 there is ambiguous between
    "no model exists" and "URL guessed wrong").

    NOTE: AlphaFold DB's prediction API fields were mid-transition as of
    this script's writing (old fields' documented sunset date, 25 June
    2026, has already passed) - this checks both old (pdbUrl) and new
    field names defensively rather than assuming either is current."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}"
    try:
        with urllib.request.urlopen(api_url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"    AlphaFold DB has no entry for {uniprot_accession}")
        else:
            print(f"    AlphaFold API request failed for {uniprot_accession}: HTTP {e.code}")
        return None, None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    AlphaFold API request failed for {uniprot_accession}: {e}")
        return None, None

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print(f"    AlphaFold API returned unparseable data for {uniprot_accession}")
        return None, None

    entry = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
    if entry is None:
        print(f"    AlphaFold API returned no prediction for {uniprot_accession}")
        return None, None

    # Try every field name this API has used, old and new, in preference order.
    pdb_url = (entry.get("pdbUrl") or entry.get("pdb_url")
               or (entry.get("files", {}) or {}).get("pdb"))
    cif_url = (entry.get("cifUrl") or entry.get("cif_url")
               or (entry.get("files", {}) or {}).get("cif"))

    for url, fmt in [(pdb_url, "pdb"), (cif_url, "cif")]:
        if not url:
            continue
        local_path = os.path.join(STRUCTURE_CACHE_DIR, f"AF-{uniprot_accession}.{fmt}")
        if os.path.exists(local_path):
            return local_path, fmt
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path, fmt
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    AlphaFold file download failed for {uniprot_accession} ({fmt}): {e}")
            continue

    print(f"    AlphaFold API response for {uniprot_accession} had no usable pdbUrl/cifUrl field "
          f"(API schema may have changed further - inspect the raw response if this recurs)")
    return None, None


# ----------------------
# Biochemical severity: Grantham-distance-style physicochemical difference
# between wild-type and mutant residue. Needs no structure at all, so every
# nonsynonymous candidate gets this even if structure acquisition fails.
# Values from Grantham (1974) Science 185:862-864, the standard reference
# scale for this purpose.
# ----------------------
GRANTHAM_COMPOSITION = {  # atomic composition c
    "A": 0.00, "R": 0.65, "N": 1.33, "D": 1.38, "C": 2.75, "Q": 0.89, "E": 0.92,
    "G": 0.74, "H": 0.58, "I": 0.00, "L": 0.00, "K": 0.33, "M": 0.00, "F": 0.00,
    "P": 0.39, "S": 1.42, "T": 0.71, "W": 0.13, "Y": 0.20, "V": 0.00,
}
GRANTHAM_POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3,
    "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2,
    "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
GRANTHAM_VOLUME = {
    "A": 31.0, "R": 124.0, "N": 56.0, "D": 54.0, "C": 55.0, "Q": 85.0, "E": 83.0,
    "G": 3.0, "H": 96.0, "I": 111.0, "L": 111.0, "K": 119.0, "M": 105.0, "F": 132.0,
    "P": 32.5, "S": 32.0, "T": 61.0, "W": 170.0, "Y": 136.0, "V": 84.0,
}


def grantham_distance(aa1, aa2):
    """Grantham's own weighting, INCLUDING the rho=50.723 scaling factor
    from his original paper (Grantham 1974, Science 185:862-864, page 863
    and the Table 2 caption) - without this final multiplication the
    formula only produces the correct RELATIVE ordering between amino
    acid pairs, not the actual published distance values (e.g. Gly-Ser
    comes out as ~1.09 without rho, vs the real, published value of 56).
    Verified directly against 9 known published values from Grantham's
    own Table 2 (Ser-Arg=110, Ser-Leu=145, Ser-Pro=74, Ser-Thr=58,
    Ser-Ala=99, Ser-Val=124, Ser-Gly=56, Ser-Ile=142, Ser-Phe=155) before
    this script was finalized - see the accompanying test log.
    Returns None for stop codons or non-standard symbols this table doesn't cover."""
    if aa1 not in GRANTHAM_VOLUME or aa2 not in GRANTHAM_VOLUME or aa1 == aa2:
        return None
    dc = GRANTHAM_COMPOSITION[aa1] - GRANTHAM_COMPOSITION[aa2]
    dp = GRANTHAM_POLARITY[aa1] - GRANTHAM_POLARITY[aa2]
    dv = GRANTHAM_VOLUME[aa1] - GRANTHAM_VOLUME[aa2]
    RHO = 50.723
    return round(RHO * (1.833 * dc * dc + 0.1018 * dp * dp + 0.000399 * dv * dv) ** 0.5, 1)


def grantham_severity_tier(distance):
    """Grantham's own published tiers."""
    if distance is None:
        return None
    if distance < 51:
        return "conservative"
    if distance < 101:
        return "moderately_conservative"
    if distance < 151:
        return "moderately_radical"
    return "radical"



# Structural severity: relative solvent accessibility via Shrake-Rupley.
# Pure Python (part of biopython's Bio.PDB.SASA), no external DSSP binary.
MAX_ASA = {  # Tien et al. 2013 theoretical max ASA (A^2), the standard
    # normalization reference for RSA
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 225, "E": 223,
    "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
    "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def compute_residue_rsa(structure_path, file_format, chain_id, residue_number, expected_wt_aa=None):
    """Relative solvent accessibility (0=fully buried, 1=fully exposed) for
    one residue in a structure file (PDB or mmCIF - see file_format).
    Returns None (with a printed reason) if the file can't be parsed, no
    chain can be confidently identified as the right one, or the residue
    isn't a standard amino acid - never a fabricated placeholder value.

    CHAIN SELECTION IS ALWAYS IDENTITY-VERIFIED WHEN expected_wt_aa IS
    GIVEN - chain_id is only ever used as a fallback when it isn't.
    An earlier version of this function trusted chain_id directly
    whenever a chain by that name existed in the structure, only
    falling back to identity-checking other chains when chain_id was
    ABSENT. That is unsafe and was proven wrong on this project's own
    real data: multi-subunit structures (e.g. 7KIN, used for both rpoA
    and rpoC - an RNA polymerase complex containing both proteins as
    separate chains in one file) commonly have a chain literally named
    "A", but there is no guarantee "A" is the chain for whatever gene
    you're actually asking about. Confirmed directly with a synthetic
    test: requesting a residue with an expected amino acid that exists
    in NEITHER chain still returned a value (silently pulled from
    chain A regardless), and a real run's coverage-count-vs-actual-
    success numbers (79 counted-as-covered vs 58 actually computed)
    pointed at exactly this kind of chain misidentification.

    Fix: when expected_wt_aa is given, every candidate chain in the
    structure is checked for the (residue_number, expected_wt_aa) pair,
    and only a chain that actually has it is used - chain_id is
    consulted only to break ties if more than one chain matches (e.g.
    homodimer copies, where any match is an equally valid answer).
    chain_id is used directly, unverified, ONLY if expected_wt_aa is
    None - a defensive fallback for callers that don't have it
    available, not the normal path.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = MMCIFParser(QUIET=True) if file_format == "cif" else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", structure_path)
    except Exception as e:
        print(f"      structure parse failed ({file_format}): {e}")
        return None

    model = structure[0]
    available = list(model.child_dict.keys())

    resolved_chain_id = None
    if expected_wt_aa is not None:
        matches = []
        found_but_wrong = {}
        for candidate in available:
            for residue in model[candidate]:
                het, resseq, icode = residue.id
                if het == " " and resseq == residue_number:
                    actual_aa = AA3TO1.get(residue.resname)
                    if actual_aa == expected_wt_aa:
                        matches.append(candidate)
                    else:
                        found_but_wrong[candidate] = actual_aa or residue.resname
                    break
        if matches:
            # Prefer chain_id itself if it's among the verified matches
            # (keeps behavior stable/predictable for single-chain and
            # AlphaFold cases); otherwise any verified match is equally
            # valid (homodimer copies of the correct protein).
            resolved_chain_id = chain_id if chain_id in matches else matches[0]
        elif found_but_wrong:
            print(f"      residue {residue_number} exists but with the wrong identity "
                  f"in every chain of this structure (expected {expected_wt_aa}, found "
                  f"{found_but_wrong}) - possible numbering mismatch between this "
                  f"structure and this project's residue count; not treated as a match")
            return None
        else:
            print(f"      residue {residue_number} isn't modeled in any chain "
                  f"{available} of this structure at all (likely unresolved density "
                  f"- common near chain termini, not necessarily a problem)")
            return None
    elif chain_id in model:
        resolved_chain_id = chain_id
    elif len(available) == 1:
        resolved_chain_id = available[0]
    else:
        print(f"      chain {chain_id} not found (available: {available}, "
              f"no expected_wt_aa given to disambiguate)")
        return None

    chain = model[resolved_chain_id]
    sr = ShrakeRupley()
    sr.compute(structure[0], level="R")

    for residue in chain:
        het, resseq, icode = residue.id
        if het != " " or resseq != residue_number:
            continue
        resname = AA3TO1.get(residue.resname)
        if resname is None:
            return None
        sasa = residue.sasa
        max_asa = MAX_ASA.get(resname)
        if max_asa is None:
            return None
        return round(min(sasa / max_asa, 1.0), 3)

    return None  # shouldn't happen given the identity check above, but stay defensive


# Main pipeline

def load_candidates():
    with open(CANDIDATES_PATH) as f:
        return list(csv.DictReader(f))


def compute_candidate_residue_info(cand, ctx):
    """Genomic position -> protein residue number + wt/mut amino acid +
    Grantham severity for one nonsynonymous candidate, reusing
    codon_classify.py's already-validated GeneContext. Returns a dict
    with keys residue_number/wt_aa/mut_aa/grantham_distance/grantham_tier
    (values "" where not computable), or None if this candidate can't be
    placed in ctx's CDS at all (wrong gene context, non-coding position,
    unparseable variant_id, etc.) - extracted as its own function so it
    can be computed ONCE per candidate and reused both for structure
    selection (which needs every gene's full residue list up front) and
    for building the final output row, rather than recomputed twice."""
    try:
        pos = int(cand["variant_id"].split("_")[1])
    except (IndexError, ValueError, KeyError):
        return None
    if ctx is None or not ctx.in_cds(pos):
        return None

    cds_index = ctx.genomic_pos_to_cds_index(pos)
    residue_number = cds_index // 3 + 1  # 1-based
    info = {"residue_number": residue_number, "wt_aa": "", "mut_aa": "",
            "grantham_distance": "", "grantham_tier": ""}

    parts = cand["variant_id"].split("_")
    if len(parts) >= 4:
        ref, alt = parts[-2], parts[-1]
        cls = ctx.classify_snp(pos, ref, alt)
        if cls.get("in_cds") and "ref_aa" in cls:
            info["wt_aa"], info["mut_aa"] = cls["ref_aa"], cls["alt_aa"]
            gd = grantham_distance(cls["ref_aa"], cls["alt_aa"])
            info["grantham_distance"] = gd if gd is not None else ""
            info["grantham_tier"] = grantham_severity_tier(gd) or ""
    return info


def main():
    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates from Step 9")

    nonsyn = [c for c in candidates if c.get("coding_classification") == "nonsynonymous"]
    print(f"{len(nonsyn)} are nonsynonymous - these get full structural analysis. "
          f"The remaining {len(candidates) - len(nonsyn)} pass through with "
          f"structural columns blank (not dropped).")

    genes_needed = sorted(set(c["gene"] for c in nonsyn))
    print(f"Genes represented among nonsynonymous candidates: {genes_needed}")

    contexts = build_gene_contexts(genes=genes_needed)

    # Compute every candidate's residue/amino-acid info up front (once),
    # both to drive coverage-based structure selection below and to reuse
    # when building the final output rows - see
    # compute_candidate_residue_info()'s docstring for why this isn't done
    # inline twice.
    residue_info = {}  # variant_id -> info dict
    # (residue_number, wt_aa) PAIRS, not bare residue numbers - coverage
    # checking needs the expected amino acid to verify identity, not just
    # residue-number presence (see structure_residue_coverage()'s docstring).
    genes_residue_aa_pairs = {g: [] for g in genes_needed}
    for cand in nonsyn:
        info = compute_candidate_residue_info(cand, contexts.get(cand["gene"]))
        if info is None:
            continue
        residue_info[cand["variant_id"]] = info
        if info["wt_aa"]:
            genes_residue_aa_pairs[cand["gene"]].append((info["residue_number"], info["wt_aa"]))

    provenance_rows = []
    structure_for_gene = {}
    for gene in genes_needed:
        print(f"\n- {gene} -")
        accession = resolve_uniprot_accession(gene)
        if accession is None:
            provenance_rows.append({"gene": gene, "uniprot_accession": None,
                                     "structure_source": "unresolved", "structure_id": None,
                                     "residue_coverage": None})
            continue
        print(f"  UniProt: {accession}")

        needed_residues = genes_residue_aa_pairs.get(gene, [])
        pdb_candidates = find_pdb_structure(accession)
        path = fmt = pdb_id = coverage = None
        if pdb_candidates:
            print(f"  {len(pdb_candidates)} experimental structure candidate(s) found: "
                  f"{pdb_candidates} - selecting by actual coverage of this gene's "
                  f"{len(needed_residues)} candidate residue(s), not resolution alone")
            path, fmt, pdb_id, coverage = select_best_pdb_structure(pdb_candidates, needed_residues)
            if path and coverage is not None and coverage.startswith("0/") and needed_residues:
                # Every PDB candidate covered NONE of this gene's actual
                # candidate residues - confirmed a real scenario, not
                # hypothetical (rpoC's best-resolution hit, 5UH7, is a
                # 24-154 fragment covering 0/20 real candidates, all of
                # which sit outside that range). A structure that can't
                # answer a single question this run needs is no better
                # than no structure at all; AlphaFold's model covers the
                # full sequence by construction, so it's a strictly
                # better fallback than keeping a 0%-coverage PDB hit.
                print(f"  Best PDB candidate ({pdb_id}) covers 0 of {len(needed_residues)} "
                      f"needed residues - treating as unusable, falling back to AlphaFold DB")
                path = fmt = pdb_id = coverage = None

        if path:
            print(f"  Using PDB {pdb_id} (covers {coverage} of this gene's candidate residues)")
            source, sid = "PDB", pdb_id
        else:
            print(f"  No usable experimental structure found; falling back to AlphaFold DB")
            path, fmt = fetch_structure_file(uniprot_accession=accession)
            source, sid = "AlphaFold", accession
            if path:
                n_cov, n_tot = structure_residue_coverage(path, fmt, needed_residues)
                coverage = f"{n_cov}/{n_tot}"
                print(f"  AlphaFold model covers {coverage} of this gene's candidate residues")

        if path is None:
            print(f"  WARNING: could not obtain any structure for {gene}")
            provenance_rows.append({"gene": gene, "uniprot_accession": accession,
                                     "structure_source": "download_failed", "structure_id": sid,
                                     "residue_coverage": None})
            continue

        structure_for_gene[gene] = {"path": path, "format": fmt, "source": source, "chain": "A"}
        provenance_rows.append({"gene": gene, "uniprot_accession": accession,
                                 "structure_source": source, "structure_id": sid,
                                 "residue_coverage": coverage})
        time.sleep(0.5)  # be polite to the free APIs between genes

    with open(os.path.join(OUTDIR, "obj3_step10_structure_provenance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "uniprot_accession", "structure_source",
                                           "structure_id", "residue_coverage"])
        w.writeheader()
        w.writerows(provenance_rows)
    print(f"\nWrote obj3_step10_structure_provenance.csv ({len(provenance_rows)} genes)")

    out_fields = [
        "variant_id", "gene", "change", "coding_classification",
        "protein_residue_number", "wt_aa", "mut_aa",
        "grantham_distance", "grantham_tier",
        "structure_source", "structure_id", "rsa", "burial_tier",
    ]
    out_rows = []
    n_structural_success = 0
    for cand in candidates:
        row = {k: cand.get(k, "") for k in ["variant_id", "gene", "change", "coding_classification"]}
        for k in out_fields[4:]:
            row[k] = ""

        info = residue_info.get(cand.get("variant_id"))
        if cand.get("coding_classification") != "nonsynonymous" or info is None:
            out_rows.append(row)
            continue

        gene = cand["gene"]
        residue_number = info["residue_number"]
        row["protein_residue_number"] = residue_number
        row["wt_aa"], row["mut_aa"] = info["wt_aa"], info["mut_aa"]
        row["grantham_distance"] = info["grantham_distance"]
        row["grantham_tier"] = info["grantham_tier"]

        struct = structure_for_gene.get(gene)
        if struct and row["wt_aa"]:
            row["structure_source"] = struct["source"]
            row["structure_id"] = next(
                (p["structure_id"] for p in provenance_rows if p["gene"] == gene), "")
            rsa = compute_residue_rsa(struct["path"], struct["format"], struct["chain"],
                                       residue_number, expected_wt_aa=row["wt_aa"])
            if rsa is not None:
                row["rsa"] = rsa
                row["burial_tier"] = ("buried" if rsa < 0.10 else
                                       "partially_buried" if rsa < 0.25 else "exposed")
                n_structural_success += 1

        out_rows.append(row)

    with open(os.path.join(OUTDIR, "obj3_step10_structural_context.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote obj3_step10_structural_context.csv ({len(out_rows)} rows, "
          f"matching Step 9's {len(candidates)} candidates exactly)")
    print(f"  {sum(1 for r in out_rows if r['grantham_distance'] != '')} rows have a "
          f"Grantham biochemical-severity score")
    print(f"  {n_structural_success} rows have a real structural RSA/burial value "
          f"(the rest either had no structure available, or the residue wasn't "
          f"resolved/modeled at that position in the structure that was found)")


if __name__ == "__main__":
    main()
