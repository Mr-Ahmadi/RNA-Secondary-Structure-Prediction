"""Estimate the evolutionary model and grammar from Rfam seed families.

For each family we need an alignment, its consensus structure and a tree.
Every family gets the same total weight (1000 split over its sequences), so
large families do not dominate.

Frequencies are counted over all sequences. Substitution rates follow Knudsen
& Hein (1999): for ordered pairs of sequences with at least 85% identity, count
the substitutions ``X -> Y`` in unpaired columns (and ``XY -> X'Y'`` in paired
columns) and divide by the expected exposure,

    R_XY = c_XY / (P_s * P_X * sum_p t_p N_p),     R_XX = -sum_{Y != X} R_XY

where ``t_p`` is the tree distance of pair ``p``, ``N_p`` its number of
comparable columns and ``P_s`` the fraction of single (unpaired) positions.
Counts from a sequence are averaged over its similar partners. Paired
counts are symmetrised (``XY -> X'Y'`` also counts ``YX -> Y'X'``).
Ambiguous IUPAC symbols are spread uniformly over compatible nucleotides and
gaps are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import read_phylip, tip_likelihoods
from .evolution import EvolutionModel
from .grammar import Grammar
from .structure import nested_part, pairs_from_dotbracket
from .tree import Node, parse_newick

IDENTITY_THRESHOLD = 0.85
FAMILY_WEIGHT = 1000.0
_SWAP = np.array([4 * b + a for a in range(4) for b in range(4)])  # XY -> YX


@dataclass
class Family:
    name: str
    alignment: dict[str, str]
    structure: str
    tree: Node

    @classmethod
    def load(cls, root: str | Path, name: str) -> "Family":
        root = Path(root)
        return cls(
            name,
            read_phylip(root / "alignments" / f"{name}.phylip"),
            (root / "structures" / f"{name}.structure").read_text().strip(),
            parse_newick((root / "trees" / f"{name}.nwk").read_text()),
        )


def _base_weights(sequences: list[str]) -> np.ndarray:
    """(n_seqs, n_cols, 4): IUPAC symbols spread uniformly, gaps all zero."""
    tips = tip_likelihoods(sequences)
    gap = np.array([[c == "-" for c in s] for s in sequences])
    tips[gap] = 0.0
    return tips / np.maximum(tips.sum(-1, keepdims=True), 1)


def _leaf_distances(tree: Node, names: list[str]) -> np.ndarray:
    """Patristic distance matrix in the order of ``names``."""
    index = {name: k for k, name in enumerate(names)}
    depth, leaves_below = {id(tree): 0.0}, {}
    for node in reversed(list(tree.postorder())):  # pre-order: parents first
        for child in node.children:
            depth[id(child)] = depth[id(node)] + child.length
    lca_depth = np.zeros((len(names), len(names)))
    for node in tree.postorder():
        if node.is_leaf():
            leaves_below[id(node)] = [index[node.name]]
            continue
        groups = [leaves_below.pop(id(c)) for c in node.children]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                lca_depth[np.ix_(groups[a], groups[b])] = depth[id(node)]
                lca_depth[np.ix_(groups[b], groups[a])] = depth[id(node)]
        leaves_below[id(node)] = [k for group in groups for k in group]
    leaf_depth = np.zeros(len(names))
    for node in tree.leaves():
        leaf_depth[index[node.name]] = depth[id(node)]
    return leaf_depth[:, None] + leaf_depth[None, :] - 2 * lca_depth


def estimate_evolution(families: list[Family]) -> EvolutionModel:
    single_count, pair_count = np.zeros(4), np.zeros(16)
    single_subst, pair_subst = np.zeros((4, 4)), np.zeros((16, 16))
    exposure = 0.0

    for family in families:
        names = list(family.alignment)
        sequences = [family.alignment[n] for n in names]
        weight = FAMILY_WEIGHT / len(sequences)
        pairs = nested_part(pairs_from_dotbracket(family.structure))
        left, right = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        unpaired = np.setdiff1d(np.arange(len(family.structure)), np.concatenate([left, right]))

        base = _base_weights(sequences)
        single = base[:, unpaired]  # (seqs, u, 4)
        dinuc = (base[:, left, :, None] * base[:, right, None, :]).reshape(len(sequences), len(pairs), 16)
        dinuc_sym = dinuc + dinuc[:, :, _SWAP]

        single_count += weight * single.sum((0, 1))
        pair_count += weight * dinuc_sym.sum((0, 1))

        codes = np.array([np.frombuffer(s.encode(), dtype=np.uint8) for s in sequences])
        identity = sum(((codes == c) * 1.0) @ (codes == c).T for c in np.unique(codes))
        similar = identity / codes.shape[1] >= IDENTITY_THRESHOLD
        np.fill_diagonal(similar, False)
        distance = _leaf_distances(family.tree, names)

        single_mass = single.sum(-1)  # (seqs, u): 1 for a base, 0 for a gap
        dinuc_mass = dinuc.sum(-1)
        for s in np.flatnonzero(similar.any(1)):
            partners = np.flatnonzero(similar[s])
            joint_single = np.einsum("ux,puy->xy", single[s], single[partners])
            joint_pair = np.einsum("ux,puy->xy", dinuc[s], dinuc[partners])
            joint_pair = joint_pair + joint_pair[np.ix_(_SWAP, _SWAP)]
            columns = weight * (single_mass[partners] @ single_mass[s] + 2 * dinuc_mass[partners] @ dinuc_mass[s])
            single_subst += weight * joint_single / len(partners)
            pair_subst += weight * joint_pair / len(partners)
            exposure += float(distance[s, partners] @ columns) / len(partners)

    total_single, total_pair = single_count.sum(), pair_count.sum()
    p_single = total_single / (total_single + total_pair)
    single_freq, pair_freq = single_count / total_single, pair_count / total_pair

    def rate_matrix(subst, freq, scale):
        rates = subst.copy()
        np.fill_diagonal(rates, 0.0)
        rates = scale * rates / (freq[:, None] * exposure)
        np.fill_diagonal(rates, -rates.sum(1))
        return rates

    return EvolutionModel(
        single_freq,
        pair_freq,
        rate_matrix(single_subst, single_freq, 1.0 / p_single),
        # a paired column holds two positions, hence the factor 2 / P_pair
        rate_matrix(pair_subst, pair_freq, 2.0 / (1.0 - p_single)),
    )


def estimate_grammar(families: list[Family]) -> Grammar:
    structures = [nested_part(pairs_from_dotbracket(f.structure)) for f in families]
    return Grammar.estimate(structures, [len(f.structure) for f in families])
