"""Continuous-time substitution models and column likelihoods on a tree.

Single columns evolve under a 4-state rate matrix, paired columns under a
16-state rate matrix over dinucleotides (Knudsen & Hein 1999). Likelihoods are
computed with Felsenstein's pruning algorithm, vectorised over all columns
(single) or all column pairs (paired), with per-node rescaling so that large
alignments do not underflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import NUCLEOTIDES, tip_likelihoods
from .tree import Node

DINUCLEOTIDES = [a + b for a in NUCLEOTIDES for b in NUCLEOTIDES]


def expm(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series."""
    norm = np.abs(matrix).sum(1).max()
    squarings = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    a = matrix / 2**squarings
    result = term = np.eye(len(matrix))
    for k in range(1, 20):
        term = term @ a / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


@dataclass
class EvolutionModel:
    single_freq: np.ndarray  # (4,)
    pair_freq: np.ndarray  # (16,)
    single_rates: np.ndarray  # (4, 4)
    pair_rates: np.ndarray  # (16, 16)

    def to_json(self) -> dict:
        return {
            "single_frequencies": dict(zip(NUCLEOTIDES, self.single_freq.tolist())),
            "pair_frequencies": dict(zip(DINUCLEOTIDES, self.pair_freq.tolist())),
            "single_rates": {"order": list(NUCLEOTIDES), "matrix": self.single_rates.tolist()},
            "pair_rates": {"order": DINUCLEOTIDES, "matrix": self.pair_rates.tolist()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "EvolutionModel":
        assert data["single_rates"]["order"] == list(NUCLEOTIDES)
        assert data["pair_rates"]["order"] == DINUCLEOTIDES
        return cls(
            np.array([data["single_frequencies"][x] for x in NUCLEOTIDES]),
            np.array([data["pair_frequencies"][x] for x in DINUCLEOTIDES]),
            np.array(data["single_rates"]["matrix"]),
            np.array(data["pair_rates"]["matrix"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvolutionModel":
        return cls.from_json(json.loads(Path(path).read_text())["evolution"])

    def column_log_likelihoods(self, alignment: dict[str, str], tree: Node):
        """Return ``(single, pair)``: ``log P(column i)`` of shape ``(n,)`` and
        ``log P(columns i, j evolve as a base pair)`` of shape ``(n, n)``.

        Identical columns have identical likelihoods, so both are computed
        once per distinct column pattern and then expanded.
        """
        names = [leaf.name for leaf in tree.leaves()]
        if sorted(names) != sorted(alignment):
            raise ValueError("tree leaves and alignment names differ")
        tips = tip_likelihoods([alignment[name] for name in names])  # (leaves, n, 4)
        _, first, column_pattern = np.unique(tips.transpose(1, 0, 2).reshape(tips.shape[1], -1), axis=0,
                                             return_index=True, return_inverse=True)
        patterns = tips[:, first]  # (leaves, u, 4)
        tip_of = dict(zip(names, patterns))
        u = patterns.shape[1]

        single = _prune(tree, lambda leaf: tip_of[leaf.name], self.single_rates, self.single_freq)
        pair = _prune(
            tree,
            lambda leaf: (tip_of[leaf.name][:, None, :, None] * tip_of[leaf.name][None, :, None, :]).reshape(u, u, 16),
            self.pair_rates,
            self.pair_freq,
        )
        column_pattern = column_pattern.ravel()
        return single[column_pattern], pair[np.ix_(column_pattern, column_pattern)]


class Transition:
    """``P(t) = exp(R t)`` for many ``t`` from one eigendecomposition of ``R``.

    Falls back to :func:`expm` when ``R`` is not (numerically) diagonalisable
    with real eigenvalues.
    """

    def __init__(self, rates: np.ndarray):
        self.rates = rates
        w, v = np.linalg.eig(rates)
        self.exact = np.allclose(w.imag, 0) and np.linalg.cond(v) < 1e8
        if self.exact:
            self.w, self.v, self.v_inv = w.real, v.real, np.linalg.inv(v.real)

    def __call__(self, t: float) -> np.ndarray:
        if not self.exact:
            return expm(self.rates * t)
        return (self.v * np.exp(self.w * t)) @ self.v_inv


def _prune(tree: Node, tip, rates: np.ndarray, freq: np.ndarray) -> np.ndarray:
    transition = Transition(rates)
    partial: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for node in tree.postorder():
        if node.is_leaf():
            like = tip(node).astype(float)
            log_scale = np.zeros(like.shape[:-1])
        else:
            like, log_scale = 1.0, 0.0
            for child in node.children:
                child_like, child_scale = partial.pop(id(child))
                like = like * (child_like @ transition(child.length).T)
                log_scale = log_scale + child_scale
            peak = like.max(-1)
            peak = np.where(peak > 0, peak, 1.0)
            like = like / peak[..., None]
            log_scale = log_scale + np.log(peak)
        partial[id(node)] = (like, log_scale)
    like, log_scale = partial[id(tree)]
    with np.errstate(divide="ignore"):
        return np.log(like @ freq) + log_scale


def optimise_branch_lengths(tree: Node, alignment: dict[str, str], rates: np.ndarray,
                            freq: np.ndarray, sweeps: int = 2, iterations: int = 20) -> Node:
    """Maximum-likelihood branch lengths for a fixed topology (in place).

    Each sweep computes inside (pruning) and outside partial likelihoods and
    then maximises every branch by golden-section search on ``log t`` in
    ``[1e-6, 3]`` with the rest of the tree held fixed. Identical columns are
    collapsed into one weighted site pattern first.
    """
    leaves = tree.leaves()
    tips = tip_likelihoods([alignment[leaf.name] for leaf in leaves])
    patterns, counts = np.unique(tips.transpose(1, 0, 2), axis=0, return_counts=True)
    tip_of = {leaf.name: patterns[:, k] for k, leaf in enumerate(leaves)}
    transition = Transition(rates)
    order = list(tree.postorder())
    ratio = (np.sqrt(5) - 1) / 2

    for _ in range(sweeps):
        inside: dict[int, np.ndarray] = {}
        for node in order:
            if node.is_leaf():
                inside[id(node)] = tip_of[node.name]
            else:
                like = 1.0
                for child in node.children:
                    like = like * (inside[id(child)] @ transition(child.length).T)
                inside[id(node)] = like / like.max(-1, keepdims=True)
        outside = {id(tree): np.broadcast_to(freq, inside[id(tree)].shape)}
        for node in reversed(order):  # parents before children
            if node.is_leaf():
                continue
            messages = [inside[id(c)] @ transition(c.length).T for c in node.children]
            for k, child in enumerate(node.children):
                above = outside[id(node)]
                for m, message in enumerate(messages):
                    if m != k:
                        above = above * message
                above = above / above.max(-1, keepdims=True)
                below = inside[id(child)]

                def log_like(log_t):
                    site = ((above @ transition(np.exp(log_t))) * below).sum(-1)
                    return counts @ np.log(site.clip(1e-300))

                a, b = np.log(1e-6), np.log(3.0)
                x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
                f1, f2 = log_like(x1), log_like(x2)
                for _ in range(iterations):
                    if f1 < f2:
                        a, x1, f1 = x1, x2, f2
                        x2 = a + ratio * (b - a)
                        f2 = log_like(x2)
                    else:
                        b, x2, f2 = x2, x1, f1
                        x1 = b - ratio * (b - a)
                        f1 = log_like(x1)
                child.length = float(np.exp((a + b) / 2))
                messages[k] = inside[id(child)] @ transition(child.length).T
                outside[id(child)] = above @ transition(child.length)
    return tree
