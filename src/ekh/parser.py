"""Gap-Bracket CYK: Viterbi parsing of an alignment with the KH-99 grammar.

The KH-99 grammar is *extended* with column likelihoods: an unpaired column
``i`` is emitted with ``P_single(i)`` and a pair of columns ``(i, j)`` with
``P_pair(i, j)``. Instead of writing one grammar rule per column pattern and
running a generic CYK, the extended grammar is specialised into the recursions
below (all quantities are log-probabilities, ``q`` are rule probabilities):

    Pair(i, j)  = M(i, j-1) + log q(E -> d) + P_pair(i, j) + chi(i, j) * log crossing
    M(i, j)     = log q($M -> B F) + log q(B -> d)
                  + max(F_stack(i+1, j) + log extension, F_loop(i+1, j) + log initiation)
    F_stack     = log q(F -> $M E) + Pair
    F_loop(i,j) = log q(F -> L S) + max_k L(i, k) + S(k+1, j)
    L(i, j)     = log q(L -> s) + P_single(i)    if i == j,  else log q(L -> $M E) + Pair(i, j)
    S(i, j)     = log q(S -> s) + P_single(i)    if i == j,
                  else max(log q(S -> $M E) + Pair(i, j), log q(S -> L S) + max_k L(i, k) + S(k+1, j))

The *helix initiation ratio* weights a pair that closes a loop (the innermost
pair of a helix) and the *helix extension ratio* a pair stacked on another
pair, so ``extension > initiation`` favours long helices. ``F`` is split into
its two derivations so that the ratios are applied exactly. The *crossing
ratio* is applied once for every first-pass pair that a second-pass pair
crosses (``chi`` counts them).

Each span length is filled in one vectorised step, so a pass costs O(n^3)
array operations but only O(n) Python iterations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .grammar import Grammar, _count_rules

NEG = -np.inf


@dataclass(frozen=True)
class PassSettings:
    initiation: float = 1.0  # helix initiation ratio
    extension: float = 1.0  # helix extension ratio
    crossing: float = 1.0  # crossing ratio (second pass only)


def cyk(
    single: np.ndarray,
    pair: np.ndarray,
    grammar: Grammar,
    settings: PassSettings = PassSettings(),
    crossings: np.ndarray | None = None,
    allowed: np.ndarray | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Most probable nested structure; returns ``(log probability, pairs)``.

    ``crossings[i, j]`` counts the pairs of an earlier pass that ``(i, j)``
    crosses. ``allowed`` (boolean ``n x n``) restricts which columns may pair,
    e.g. to enforce a minimum hairpin loop in alignment coordinates.
    """
    n = len(single)
    if n == 0:
        return 0.0, []
    g = grammar.log
    pair_score = pair + g("E", "d")
    if crossings is not None and settings.crossing != 1.0:
        pair_score = pair_score + crossings * np.log(settings.crossing)
    if allowed is not None:
        pair_score = np.where(allowed, pair_score, NEG)
    m_const = g("$M", "B F") + g("B", "d")
    log_init, log_ext = np.log(settings.initiation), np.log(settings.extension)
    s_single, s_pair, s_split = g("S", "s"), g("S", "$M E"), g("S", "L S")
    l_pair, f_pair, f_split = g("L", "$M E"), g("F", "$M E"), g("F", "L S")

    S = np.full((n, n), NEG)
    L = np.full((n, n), NEG)
    P = np.full((n, n), NEG)  # Pair(i, j) without the rule of the parent
    F_stack = np.full((n, n), NEG)
    F_loop = np.full((n, n), NEG)
    M = np.full((n, n), NEG)
    split = np.zeros((n, n), dtype=np.int64)  # best k of L(i,k) S(k+1,j)
    # The same L and S cells indexed by (start, span) and (end, span), so the
    # bifurcation max_k L(i,k) + S(k+1,j) reads contiguous slices, not gathers.
    L_by_start = np.full((n, n), NEG)
    S_by_end = np.full((n, n), NEG)
    idx = np.arange(n)
    with np.errstate(invalid="ignore"):
        S[idx, idx] = S_by_end[:, 0] = s_single + single
        L[idx, idx] = L_by_start[:, 0] = g("L", "s") + single
        for d in range(1, n):
            i = idx[: n - d]
            j = i + d
            if d >= 2:  # $M spans B plus an interior F of at least two columns
                M[i, j] = m_const + np.maximum(F_stack[i + 1, j] + log_ext, F_loop[i + 1, j] + log_init)
            if d >= 3:
                P[i, j] = M[i, j - 1] + pair_score[i, j]
                L[i, j] = L_by_start[i, d] = l_pair + P[i, j]
                F_stack[i, j] = f_pair + P[i, j]
            # k = i + t for t = 0..d-1: L(i, i+t) is L_by_start[i, t], S(i+t+1, j) is S_by_end[j, d-1-t]
            bif = L_by_start[: n - d, :d] + S_by_end[d:, d - 1::-1]
            best = np.argmax(bif, axis=1)
            split[i, j] = i + best
            bif_best = bif[i, best]
            F_loop[i, j] = f_split + bif_best
            S[i, j] = S_by_end[j, d] = np.maximum(s_pair + P[i, j], s_split + bif_best)

    score = float(S[0, n - 1])
    if not np.isfinite(score):
        return score, []
    return score, _traceback(S, P, F_stack, F_loop, split, s_pair, log_init, log_ext)


def _traceback(S, P, F_stack, F_loop, split, s_pair, log_init, log_ext):
    pairs = []
    stack = [("S", 0, len(S) - 1)]
    while stack:
        symbol, i, j = stack.pop()
        if symbol == "pair":
            pairs.append((i, j))
            # M(i, j-1) -> B F(i+1, j-1): follow the F derivation M chose
            if F_stack[i + 1, j - 1] + log_ext >= F_loop[i + 1, j - 1] + log_init:
                stack.append(("pair", i + 1, j - 1))
            else:
                stack.append(("loop", i + 1, j - 1))
        elif symbol == "loop":
            k = split[i, j]
            stack.append(("L", i, k))
            stack.append(("S", k + 1, j))
        elif i == j:
            continue  # S -> s or L -> s
        elif symbol == "L" or s_pair + P[i, j] >= S[i, j]:
            stack.append(("pair", i, j))
        else:
            stack.append(("loop", i, j))
    return sorted(pairs)


def crossing_counts(positions: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """``out[a, b]``: how many of ``pairs`` cross a pair of ``positions[a], positions[b]``."""
    out = np.zeros((len(positions), len(positions)))
    a, b = positions[:, None], positions[None, :]
    for x, y in pairs:
        out += ((a < x) & (x < b) & (b < y)) | ((x < a) & (a < y) & (y < b))
    return out


def structure_score(
    single: np.ndarray,
    pair: np.ndarray,
    grammar: Grammar,
    settings: PassSettings,
    pairs: list[tuple[int, int]],
    crossings: np.ndarray | None = None,
) -> float:
    """Log score of one fixed nested structure under the same model as :func:`cyk`
    (``-inf`` if the grammar cannot derive it, e.g. a hairpin loop of one column)."""
    n = len(single)
    rules: Counter = Counter()
    try:
        _count_rules(pairs, n, rules)
    except ValueError:
        return NEG
    score = sum(count * grammar.log(*rule) for rule, count in rules.items())
    paired = {k for p in pairs for k in p}
    score += float(sum(single[k] for k in range(n) if k not in paired))
    partner = dict(pairs)
    for i, j in pairs:
        score += pair[i, j] + np.log(settings.extension if partner.get(i + 1) == j - 1 else settings.initiation)
        if crossings is not None:
            score += crossings[i, j] * np.log(settings.crossing)
    return float(score)

