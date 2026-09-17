"""Base-pair sets: dot-bracket parsing and rendering, nesting and pseudoknots."""

from __future__ import annotations

import numpy as np

OPENERS = "([{<"
CLOSERS = ")]}>"
_MATCH = dict(zip(CLOSERS, OPENERS))

Pairs = list[tuple[int, int]]


def pairs_from_dotbracket(structure: str) -> Pairs:
    """Return sorted base pairs ``(i, j)``, ``i < j``, of a dot-bracket string.

    Every bracket type has its own stack, so pseudoknots written with a second
    bracket type are supported, as are Rfam WUSS pseudoknot letters (``A`` pairs
    with ``a``). Any other character is unpaired (this covers ``SS_cons``
    symbols such as ``,:_-~``).
    """
    stacks: dict[str, list[int]] = {}
    pairs = []
    for j, char in enumerate(structure):
        if char in OPENERS or char.isupper():
            stacks.setdefault(char, []).append(j)
        elif char in _MATCH or char.islower():
            stack = stacks.get(_MATCH.get(char, char.upper()))
            if not stack:
                raise ValueError(f"unbalanced {char!r} at position {j}")
            pairs.append((stack.pop(), j))
    for opener, stack in stacks.items():
        if stack:
            raise ValueError(f"unbalanced {opener!r} at position {stack[-1]}")
    return sorted(pairs)


def crosses(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return p[0] < q[0] < p[1] < q[1] or q[0] < p[0] < q[1] < p[1]


def is_nested(pairs: Pairs) -> bool:
    ordered = sorted(pairs)
    return all(not crosses(p, q) for k, p in enumerate(ordered) for q in ordered[k + 1:] if q[0] < p[1])


def pseudoknotted_pairs(pairs: Pairs) -> Pairs:
    """Pairs that cross at least one other pair of the same structure.

    This does not depend on how a structure is split into bracket layers: both
    helices of a pseudoknot count.
    """
    ordered = sorted(pairs)
    out = set()
    for k, p in enumerate(ordered):
        for q in ordered[k + 1:]:
            if q[0] > p[1]:
                break
            if crosses(p, q):
                out.update((p, q))
    return sorted(out)


def largest_nested(pairs: Pairs) -> Pairs:
    """Largest nested subset of a set of base pairs (exact interval DP).

    Runs over the paired columns only: ``best[a, b]`` is the size of the
    largest nested subset of the pairs within paired columns ``a..b-1``.
    """
    columns = sorted(k for p in pairs for k in p)
    index = {c: a for a, c in enumerate(columns)}
    partner = {index[i]: index[j] for i, j in pairs}
    m = len(columns)
    best = np.zeros((m + 1, m + 1), dtype=np.int64)

    def take(a: int, b: int) -> int:
        k = partner.get(a)
        return 1 + best[a + 1, k] + best[k + 1, b] if k is not None and k < b else -1

    for a in range(m - 1, -1, -1):
        for b in range(a + 1, m + 1):
            best[a, b] = max(best[a + 1, b], take(a, b))
    nested, stack = [], [(0, m)]
    while stack:
        a, b = stack.pop()
        if a >= b:
            continue
        if take(a, b) >= best[a + 1, b]:
            k = partner[a]
            nested.append((columns[a], columns[k]))
            stack += [(a + 1, k), (k + 1, b)]
        else:
            stack.append((a + 1, b))
    return sorted(nested)


def split_layers(pairs: Pairs) -> list[Pairs]:
    """Layers for ``()``, ``[]``, ``{}``, ``<>``: each is the largest nested subset of the pairs left."""
    layers, rest = [], sorted(pairs)
    while rest:
        layers.append(largest_nested(rest))
        rest = sorted(set(rest) - set(layers[-1]))
    if len(layers) > len(OPENERS):
        raise ValueError(f"pairs need more than {len(OPENERS)} bracket types")
    return layers


def dotbracket(length: int, *layers: Pairs) -> str:
    """Render pair layers with ``()``, ``[]``, ``{}``, ``<>`` in that order."""
    chars = ["."] * length
    for (opener, closer), layer in zip(zip(OPENERS, CLOSERS), layers):
        for i, j in layer:
            if chars[i] != "." or chars[j] != ".":
                raise ValueError(f"position reused by pair {(i, j)}")
            chars[i], chars[j] = opener, closer
    return "".join(chars)


def to_dotbracket(length: int, pairs: Pairs) -> str:
    """Canonical dot-bracket string of a pair set (largest nested layer first)."""
    return dotbracket(length, *split_layers(pairs))
