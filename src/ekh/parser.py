"""Gap-Bracket CYK: Viterbi parsing of an alignment with the KH-99 grammar.

The KH-99 grammar is *extended* with column likelihoods: an unpaired column
``i`` is emitted with ``P_single(i)`` and a pair of columns ``(i, j)`` with
``P_pair(i, j)``. Instead of writing one grammar rule per column pattern and
running a generic CYK, the extended grammar is specialised into the recursions
below (all quantities are log-probabilities, ``q`` are rule probabilities):

    Pair(i, j)  = M(i, j-1) + log q(E -> d) + P_pair(i, j) + crossings(i, j) * log flag
    M(i, j)     = log q($M -> B F) + log q(B -> d)
                  + max(F_stack(i+1, j) + log accelerate, F_loop(i+1, j) + log start)
    F_stack     = log q(F -> $M E) + Pair
    F_loop(i,j) = log q(F -> L S) + max_k L(i, k) + S(k+1, j)
    L(i, j)     = log q(L -> s) + P_single(i)    if i == j,  else log q(L -> $M E) + Pair(i, j)
    S(i, j)     = log q(S -> s) + P_single(i)    if i == j,
                  else max(log q(S -> $M E) + Pair(i, j), log q(S -> L S) + max_k L(i, k) + S(k+1, j))

``start`` weights a pair that closes a loop (the first pair of a helix, seen
from the inside) and ``accelerate`` weights a pair stacked on another pair, so
``accelerate > start`` favours long helices. ``F`` is split into its two
derivations so the ratio is applied exactly (a single ``F`` cell would keep
only the best derivation and lose the other one). ``flag`` penalises (``< 1``)
or rewards (``> 1``) each first-pass pair that a second-pass pair crosses.

Each span length is filled in one vectorised step, so a pass costs O(n^3)
array operations but only O(n) Python iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grammar import Grammar

NEG = -np.inf


@dataclass(frozen=True)
class PassParams:
    start: float = 1.0
    accelerate: float = 1.0
    flag: float = 1.0


def cyk(
    single: np.ndarray,
    pair: np.ndarray,
    grammar: Grammar,
    params: PassParams = PassParams(),
    crossings: np.ndarray | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Most probable nested structure; returns ``(log probability, pairs)``."""
    n = len(single)
    if n == 0:
        return 0.0, []
    g = grammar.log
    pair_score = pair + g("E", "d")
    if crossings is not None and params.flag != 1.0:
        pair_score = pair_score + crossings * np.log(params.flag)
    m_const = g("$M", "B F") + g("B", "d")
    log_start, log_acc = np.log(params.start), np.log(params.accelerate)

    S = np.full((n, n), NEG)
    L = np.full((n, n), NEG)
    P = np.full((n, n), NEG)  # Pair(i, j) without the rule of the parent
    F_stack = np.full((n, n), NEG)
    F_loop = np.full((n, n), NEG)
    M = np.full((n, n), NEG)
    split = np.zeros((n, n), dtype=np.int64)  # best k of L(i,k) S(k+1,j)
    idx = np.arange(n)
    with np.errstate(invalid="ignore"):
        S[idx, idx] = g("S", "s") + single
        L[idx, idx] = g("L", "s") + single
        for d in range(1, n):
            i = idx[: n - d]
            j = i + d
            if d >= 2:  # $M spans B plus an interior F of at least two columns
                M[i, j] = m_const + np.maximum(F_stack[i + 1, j] + log_acc, F_loop[i + 1, j] + log_start)
            if d >= 3:
                P[i, j] = M[i, j - 1] + pair_score[i, j]
                L[i, j] = g("L", "$M E") + P[i, j]
                F_stack[i, j] = g("F", "$M E") + P[i, j]
            ks = i[:, None] + np.arange(d)[None, :]
            bif = L[i[:, None], ks] + S[ks + 1, j[:, None]]
            best = np.argmax(bif, axis=1)
            split[i, j] = i + best
            bif_best = bif[np.arange(len(i)), best]
            F_loop[i, j] = g("F", "L S") + bif_best
            S[i, j] = np.maximum(g("S", "$M E") + P[i, j], g("S", "L S") + bif_best)

    score = float(S[0, n - 1])
    if not np.isfinite(score):
        return score, []
    return score, _traceback(S, P, F_stack, F_loop, split, grammar, log_start, log_acc)


def _traceback(S, P, F_stack, F_loop, split, grammar, log_start, log_acc):
    g = grammar.log
    pairs = []
    stack = [("S", 0, len(S) - 1)]
    while stack:
        symbol, i, j = stack.pop()
        if symbol == "pair":
            pairs.append((i, j))
            # M(i, j-1) -> B F(i+1, j-1): follow the F derivation M chose
            if F_stack[i + 1, j - 1] + log_acc >= F_loop[i + 1, j - 1] + log_start:
                stack.append(("pair", i + 1, j - 1))
            else:
                stack.append(("loop", i + 1, j - 1))
        elif symbol == "loop":
            k = split[i, j]
            stack.append(("L", i, k))
            stack.append(("S", k + 1, j))
        elif i == j:
            continue  # S -> s or L -> s
        elif symbol == "L" or g("S", "$M E") + P[i, j] >= S[i, j]:
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
