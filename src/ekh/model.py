"""End-to-end consensus structure prediction (two-pass Gap-Bracket CYK)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .evolution import EvolutionModel, optimise_branch_lengths
from .grammar import Grammar
from .parser import PassSettings, crossing_counts, cyk, structure_score
from .structure import dotbracket, split_layers
from .tree import Node, midpoint_root, neighbour_joining, nj_tree, p_distance_matrix

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


@dataclass(frozen=True)
class Decoding:
    """Settings of the two parsing passes.

    ``evidence_weight`` (beta) multiplies every column log-likelihood before
    parsing. With beta < 1 the grammar counts for more against the alignment,
    which makes up for columns and sequences not being as independent as the
    phylogenetic model assumes. ``min_hairpin`` is the fewest columns a pair
    must enclose (3 is the physical minimum; the bare KH-99 grammar allows 2).
    ``pseudoknot_ratio`` multiplies the probability of every second-layer pair,
    so that a pseudoknot must be better supported than a nested helix.
    """

    first: PassSettings = PassSettings()
    second: PassSettings | None = PassSettings()  # None disables the pseudoknot pass
    evidence_weight: float = 1.0
    min_hairpin: int = 3
    pseudoknot_ratio: float = 1.0  # weight of every second-layer (pseudoknot) pair
    refinement_rounds: int = 0  # joint re-parsing rounds of the two layers

    @classmethod
    def kh99(cls) -> "Decoding":
        """Plain KH-99 CYK: one pass, all ratios 1, no evidence weight, hairpins of 2."""
        return cls(PassSettings(), None, 1.0, 2, 1.0, 0)

    def to_json(self) -> dict:
        return {"first": asdict(self.first), "second": asdict(self.second) if self.second else None,
                "evidence_weight": self.evidence_weight, "min_hairpin": self.min_hairpin,
                "pseudoknot_ratio": self.pseudoknot_ratio, "refinement_rounds": self.refinement_rounds}

    @classmethod
    def from_json(cls, data: dict) -> "Decoding":
        second = data.get("second", {})
        return cls(PassSettings(**data.get("first", {})), None if second is None else PassSettings(**second),
                   float(data.get("evidence_weight", 1.0)), int(data.get("min_hairpin", 3)),
                   float(data.get("pseudoknot_ratio", 1.0)), int(data.get("refinement_rounds", 0)))


class EKH:
    def __init__(self, evolution: EvolutionModel, grammar: Grammar, decoding: Decoding = Decoding()):
        self.evolution = evolution
        self.grammar = grammar
        self.decoding = decoding

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS) -> "EKH":
        data = json.loads(Path(path).read_text())
        return cls(EvolutionModel.from_json(data["evolution"]), Grammar.from_json(data["grammar"]),
                   Decoding.from_json(data.get("decoding", {})))

    def with_decoding(self, decoding: Decoding) -> "EKH":
        return EKH(self.evolution, self.grammar, decoding)

    def evidence(self, alignment: dict[str, str], tree: Node | None = None) -> ColumnEvidence:
        tree = tree or self.estimate_tree(alignment)
        single, pair = self.evolution.column_log_likelihoods(alignment, tree)
        return ColumnEvidence(tree, single, pair)

    def estimate_tree(self, alignment: dict[str, str], sweeps: int = 2) -> Node:
        """Neighbour-joining topology with maximum-likelihood branch lengths
        under the model's own single-base substitution process."""
        names = list(alignment)
        if len(names) < 2:
            return nj_tree(alignment)
        tree = neighbour_joining(names, p_distance_matrix([alignment[n] for n in names]))
        optimise_branch_lengths(tree, alignment, self.evolution.single_rates,
                                self.evolution.single_freq, sweeps)
        return midpoint_root(tree)

    def first_pass(self, evidence: ColumnEvidence) -> list[tuple[int, int]]:
        d = self.decoding
        return self._parse_layer(d.evidence_weight * evidence.single, d.evidence_weight * evidence.pair,
                                 d.first, excluded=[], crossed=[])

    def pairs(self, evidence: ColumnEvidence, nested: list[tuple[int, int]] | None = None):
        """Both layers; returns ``(first-layer pairs, second-layer pairs)``.

        The first pass parses all columns; the second parses the columns the
        first leaves unpaired. With ``refinement_rounds > 0`` each layer is
        then re-parsed on the columns the other leaves free, and a round is
        kept only if the joint score of the two layers increases.
        ``nested`` can pass in a cached first-pass result.
        """
        d = self.decoding
        if d.second is None:
            return (self.first_pass(evidence) if nested is None else nested), []
        single, pair, pair_second = self._weighted(evidence)
        first = self.first_pass(evidence) if nested is None else nested
        second = self._parse_layer(single, pair_second, d.second, excluded=first, crossed=first)
        if d.refinement_rounds:
            score = self.joint_score(evidence, first, second)
            for _ in range(d.refinement_rounds):
                new_first = self._parse_layer(single, pair, d.first, excluded=second, crossed=[])
                new_second = self._parse_layer(single, pair_second, d.second, excluded=new_first, crossed=new_first)
                new_score = self.joint_score(evidence, new_first, new_second)
                if new_score <= score + 1e-9:
                    break
                first, second, score = new_first, new_second, new_score
        return first, second

    def joint_score(self, evidence: ColumnEvidence, first: list[tuple[int, int]],
                    second: list[tuple[int, int]]) -> float:
        """Score of the first layer on the columns the second leaves free, plus the
        score of the second layer (with its pair and crossing ratios) on the columns the first leaves free."""
        d = self.decoding
        single, pair, pair_second = self._weighted(evidence)
        return (self._layer_score(single, pair, d.first, first, excluded=second, crossed=[])
                + self._layer_score(single, pair_second, d.second, second, excluded=first, crossed=first))

    def _weighted(self, evidence: ColumnEvidence):
        """Evidence-weighted column scores, and pair scores for the second layer."""
        d = self.decoding
        pair = d.evidence_weight * evidence.pair
        return d.evidence_weight * evidence.single, pair, pair + np.log(d.pseudoknot_ratio)

    def _free(self, n: int, excluded: list[tuple[int, int]]) -> np.ndarray:
        used = {k for p in excluded for k in p}
        return np.array([k for k in range(n) if k not in used], dtype=np.int64)

    def _parse_layer(self, single, pair, settings: PassSettings, excluded, crossed) -> list[tuple[int, int]]:
        free = self._free(len(single), excluded)
        if not len(free):
            return []
        crossings = crossing_counts(free, crossed) if crossed else None
        _, found = cyk(single[free], pair[np.ix_(free, free)], self.grammar, settings, crossings, self._allowed(free))
        return [(int(free[a]), int(free[b])) for a, b in found]

    def _layer_score(self, single, pair, settings: PassSettings, layer, excluded, crossed) -> float:
        free = self._free(len(single), excluded)
        local = {c: k for k, c in enumerate(free)}
        crossings = crossing_counts(free, crossed) if crossed else None
        return structure_score(single[free], pair[np.ix_(free, free)], self.grammar, settings,
                               [(local[i], local[j]) for i, j in layer], crossings)

    def _allowed(self, positions: np.ndarray) -> np.ndarray:
        """Pairs of ``positions`` that enclose at least ``min_hairpin`` alignment columns."""
        return positions[None, :] - positions[:, None] > self.decoding.min_hairpin

    def parse(self, evidence: ColumnEvidence) -> Prediction:
        first, second = self.pairs(evidence)
        layers = split_layers(first + second)
        structure = dotbracket(len(evidence.single), *layers)
        return Prediction(structure, layers[0] if layers else [], sorted(sum(layers[1:], [])), evidence.tree)

    def predict(self, alignment: dict[str, str], tree: Node | None = None) -> Prediction:
        return self.parse(self.evidence(alignment, tree))
