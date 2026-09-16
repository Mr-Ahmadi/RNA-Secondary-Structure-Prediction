"""End-to-end consensus structure prediction (two-pass Gap-Bracket CYK)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .evolution import EvolutionModel, optimise_branch_lengths
from .grammar import Grammar
from .parser import PassParams, crossing_counts, cyk
from .structure import dotbracket
from .tree import Node, estimate_tree, midpoint_root, neighbour_joining, p_distance_matrix

DEFAULT_PARAMS = Path(__file__).parent / "data" / "parameters.json"


@dataclass
class Prediction:
    structure: str
    nested: list[tuple[int, int]]
    crossing: list[tuple[int, int]]
    tree: Node = field(repr=False)


@dataclass
class ColumnEvidence:
    """Tree and column log-likelihoods: everything the parser needs.

    Computing the evidence is the expensive step, so tuning caches it and
    re-parses with different ratios.
    """

    tree: Node
    single: np.ndarray
    pair: np.ndarray


class EKH:
    def __init__(self, evolution: EvolutionModel, grammar: Grammar,
                 first: PassParams = PassParams(), second: PassParams | None = PassParams()):
        self.evolution = evolution
        self.grammar = grammar
        self.first = first
        self.second = second  # None disables the pseudoknot pass

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS) -> "EKH":
        data = json.loads(Path(path).read_text())
        passes = data.get("passes", {})
        second = passes.get("second", {})
        return cls(
            EvolutionModel.from_json(data["evolution"]),
            Grammar.from_json(data["grammar"]),
            PassParams(**passes.get("first", {})),
            None if second is None else PassParams(**second),
        )

    def evidence(self, alignment: dict[str, str], tree: Node | None = None,
                 phyml: str | None = None) -> ColumnEvidence:
        if tree is None:
            tree = estimate_tree(alignment, phyml) if phyml else self.estimate_tree(alignment)
        single, pair = self.evolution.column_log_likelihoods(alignment, tree)
        return ColumnEvidence(tree, single, pair)

    def estimate_tree(self, alignment: dict[str, str], sweeps: int = 2) -> Node:
        """Neighbour-joining topology with maximum-likelihood branch lengths
        under the model's own single-base substitution process."""
        names = list(alignment)
        if len(names) < 3:
            return estimate_tree(alignment)
        tree = neighbour_joining(names, p_distance_matrix([alignment[n] for n in names]))
        optimise_branch_lengths(tree, alignment, self.evolution.single_rates,
                                self.evolution.single_freq, sweeps)
        return midpoint_root(tree)

    def parse(self, evidence: ColumnEvidence) -> Prediction:
        n = len(evidence.single)
        _, nested = cyk(evidence.single, evidence.pair, self.grammar, self.first)
        crossing: list[tuple[int, int]] = []
        if self.second is not None:
            paired = {k for p in nested for k in p}
            free = np.array([k for k in range(n) if k not in paired], dtype=np.int64)
            if len(free):
                _, found = cyk(
                    evidence.single[free],
                    evidence.pair[np.ix_(free, free)],
                    self.grammar,
                    self.second,
                    crossing_counts(free, nested),
                )
                crossing = [(int(free[a]), int(free[b])) for a, b in found]
        return Prediction(dotbracket(n, nested, crossing), nested, crossing, evidence.tree)

    def predict(self, alignment: dict[str, str], tree: Node | None = None,
                phyml: str | None = None) -> Prediction:
        return self.parse(self.evidence(alignment, tree, phyml))
