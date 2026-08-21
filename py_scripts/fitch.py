"""
Standard Fitch (1971) parsimony for a binary character on a tree.
Returns the minimum number of state-change events required to explain
the observed tip states given the tree topology (the "homoplasy count"
for that site). A count of 1 = perfectly consistent with a single
origin (clonal spread down one clade). A count > 1 means the raw
presence pattern cannot be explained by one inherited event - either
true convergent (independent) origins, or a gain plus reversion(s).
"""
from ete3 import Tree


def fitch_steps(tree, tip_states):
    """
    tree: ete3 Tree (already loaded, with tip names matching tip_states keys)
    tip_states: dict {tip_name: 0 or 1}
    Returns: (n_steps, root_state_set)
    Tips not present in tip_states are treated as missing and pruned
    from consideration for this site (their subtree is ignored if all
    descendants are missing).
    """
    steps = 0

    # Post-order: process leaves first, then internal nodes bottom-up
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if node.name in tip_states:
                node.add_feature("fset", {tip_states[node.name]})
            else:
                node.add_feature("fset", None)  # missing
        else:
            child_sets = [c.fset for c in node.children if c.fset is not None]
            if not child_sets:
                node.add_feature("fset", None)
                continue
            inter = set.intersection(*child_sets) if len(child_sets) > 1 else set(child_sets[0])
            if inter:
                node.add_feature("fset", inter)
            else:
                node.add_feature("fset", set.union(*child_sets))
                steps += 1

    root_set = tree.fset
    return steps, root_set


def fitch_steps_batch(tree, sites_tip_states):
    """
    Run fitch_steps for many sites efficiently by doing a single
    postorder traversal per site (ete3 doesn't support numpy vectorization
    natively, so this loops per site - fine for hundreds of sites).
    sites_tip_states: dict {site_id: {tip_name: 0/1}}
    Returns: dict {site_id: n_steps}
    """
    results = {}
    for site_id, tip_states in sites_tip_states.items():
        n_steps, _ = fitch_steps(tree, tip_states)
        results[site_id] = n_steps
    return results


def fitch_rooted_gains_losses(tree, tip_states, root_state=0):
    """
    Fitch parsimony with the root FORCED to root_state, rather than left to
    the down-pass's own (possibly ambiguous, possibly root=1) resolution.

    Why this matters here, concretely (not just in theory): for a small but
    real subset of this project's own candidates, the down-pass root Fitch
    set comes back as {1} or {0,1} - meaning unconstrained parsimony's
    minimum-cost solution either requires or permits "the mutation was
    ancestral, and independently LOST in the isolates that lack it," which
    is biologically backwards for a derived resistance/compensatory allele
    relative to a drug-susceptible reference genome (H37Rv). Forcing the
    root to 0 (reference/absent) may cost one or more extra parsimony steps
    versus the unconstrained minimum, but encodes the actual biological
    prior correctly. On an earlier run of this project's real candidates,
    a small number (including one high-carriage headline candidate, mmpL5
    p.Ile948Val, 86% carriage) had a non-{0} unconstrained root set. That
    specific count was measured against the old, pre-fix 391-candidate
    set (Step 2's candidate count was later corrected to 629) and against
    a tree still missing 31 IQ-TREE-dropped samples (see
    step9_1_build_tree.py's complete_tree()) - re-verify on a real
    re-run rather than assume unchanged; the underlying rationale for
    forcing the root is unaffected by either fix and does not depend on
    the exact count.

    Returns: (n_gains, n_losses)
        n_gains  = number of 0->1 transitions on the tree (independent
                   origin events - this is the homoplasy/convergence count)
        n_losses = number of 1->0 transitions (reversions)
    """
    # Down-pass: compute each node's Fitch state set, same as fitch_steps.
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("fset", {tip_states[node.name]} if node.name in tip_states else None)
        else:
            child_sets = [c.fset for c in node.children if c.fset is not None]
            if not child_sets:
                node.add_feature("fset", None)
                continue
            inter = set.intersection(*child_sets) if len(child_sets) > 1 else set(child_sets[0])
            node.add_feature("fset", inter if inter else set.union(*child_sets))

    # Up-pass: assign states top-down, root fixed, each other node preferring
    # to match its parent's assigned state when its own Fitch set allows it
    # (the standard MPR tie-breaking rule - avoids introducing unnecessary
    # extra changes for ambiguous internal nodes).
    tree.add_feature("assigned", root_state)
    gains, losses = 0, 0
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        parent_state = node.up.assigned
        if node.fset is None:
            node.add_feature("assigned", parent_state)  # missing tip: inherit, no event
            continue
        node.add_feature("assigned", parent_state if parent_state in node.fset else next(iter(node.fset)))
        if parent_state == 0 and node.assigned == 1:
            gains += 1
        elif parent_state == 1 and node.assigned == 0:
            losses += 1
    return gains, losses


def fitch_rooted_batch(tree, sites_tip_states, root_state=0):
    """Batch version of fitch_rooted_gains_losses.
    Returns: dict {site_id: (n_gains, n_losses)}"""
    results = {}
    for site_id, tip_states in sites_tip_states.items():
        results[site_id] = fitch_rooted_gains_losses(tree, tip_states, root_state)
    return results
