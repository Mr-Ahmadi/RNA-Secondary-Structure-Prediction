"""The KH-99 structure grammar in Chomsky normal form.

    S  -> L S | $M E | s        (a loop region: one or more elements)
    L  -> $M E | s              (one element: an unpaired base or a helix)
    F  -> $M E | L S            (inside a pair: a stacked pair or a loop)
    $M -> B F                   (left base of a pair and its interior)
    B  -> d,  E -> d            (paired bases)

``$M E`` generates a base pair ``d F d``. The grammar is unambiguous: every
nested structure has exactly one derivation, so the maximum-likelihood rule
probabilities are relative frequencies of the rules used by the training
structures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

RULES = {
    "S": ["L S", "$M E", "s"],
    "L": ["$M E", "s"],
    "F": ["$M E", "L S"],
    "$M": ["B F"],
    "B": ["d"],
    "E": ["d"],
}


@dataclass
class Grammar:
    prob: dict[tuple[str, str], float]

    def log(self, lhs: str, rhs: str) -> float:
        p = self.prob[lhs, rhs]
        return float(np.log(p)) if p > 0 else -np.inf

    def to_json(self) -> dict:
        return {f"{lhs} -> {rhs}": self.prob[lhs, rhs] for lhs, rhss in RULES.items() for rhs in rhss}

    @classmethod
    def from_json(cls, data: dict) -> "Grammar":
        prob = {}
        for rule, p in data.items():
            lhs, rhs = rule.split(" -> ")
            prob[lhs, rhs] = float(p)
        missing = {(l, r) for l, rs in RULES.items() for r in rs} - set(prob)
        if missing:
            raise ValueError(f"grammar is missing rules {sorted(missing)}")
        return cls(prob)

    @classmethod
    def estimate(cls, structures: list[list[tuple[int, int]]], lengths: list[int]) -> "Grammar":
        """Relative-frequency estimate from nested structures (pair lists)."""
        counts: Counter = Counter()
        for pairs, length in zip(structures, lengths):
            _count_rules(pairs, length, counts)
        prob = {}
        for lhs, rhss in RULES.items():
            total = sum(counts[lhs, rhs] for rhs in rhss)
            for rhs in rhss:
                prob[lhs, rhs] = counts[lhs, rhs] / total if total else 1.0 / len(rhss)
        return cls(prob)


def _count_rules(pairs: list[tuple[int, int]], length: int, counts: Counter) -> None:
    partner = {i: j for i, j in pairs}
    if len(partner) != len(pairs) or any(j in partner for _, j in pairs):
        raise ValueError("a base is used by more than one pair")

    def elements(start: int, end: int) -> list[tuple[int, int]]:
        """Top-level elements of [start, end): unpaired bases and helices."""
        out, i = [], start
        while i < end:
            j = partner.get(i, i)
            if j >= end:
                raise ValueError(f"pair {(i, j)} is not nested")
            out.append((i, j))
            i = j + 1
        return out

    # Explicit stack instead of recursion: long molecules nest deeply.
    stack = [("S", 0, length)]
    while stack:
        symbol, start, end = stack.pop()
        if symbol == "pair":  # (start, end) is the base pair itself
            counts["$M", "B F"] += 1
            counts["B", "d"] += 1
            counts["E", "d"] += 1
            stack.append(("F", start + 1, end))
            continue
        items = elements(start, end)
        if symbol == "F":
            if len(items) == 1 and items[0][0] != items[0][1]:
                counts["F", "$M E"] += 1
                stack.append(("pair", *items[0]))
            elif len(items) >= 2:
                counts["F", "L S"] += 1
                stack.append(("L", items[0][0], items[0][1] + 1))
                stack.append(("S", items[1][0], end))
            else:
                raise ValueError(f"hairpin closed at {start - 1} encloses fewer than 2 bases")
        elif symbol == "S" and len(items) > 1:
            counts["S", "L S"] += 1
            stack.append(("L", items[0][0], items[0][1] + 1))
            stack.append(("S", items[1][0], end))
        else:  # S or L with a single element
            i, j = items[0]
            if i == j:
                counts[symbol, "s"] += 1
            else:
                counts[symbol, "$M E"] += 1
                stack.append(("pair", i, j))
