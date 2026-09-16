"""Dot-bracket structures: parsing, rendering and conversion to pair sets."""

from __future__ import annotations

OPENERS = "([{<"
CLOSERS = ")]}>"
_MATCH = dict(zip(CLOSERS, OPENERS))


def pairs_from_dotbracket(structure: str) -> list[tuple[int, int]]:
    """Return sorted base pairs ``(i, j)``, ``i < j``, of a dot-bracket string.

    Every bracket type has its own stack, so pseudoknots written with a second
    bracket type are supported. Any non-bracket character is unpaired (this
    covers Rfam ``SS_cons`` symbols such as ``,:_-~``).
    """
    stacks: dict[str, list[int]] = {c: [] for c in OPENERS}
    pairs = []
    for j, char in enumerate(structure):
        if char in stacks:
            stacks[char].append(j)
        elif char in _MATCH:
            stack = stacks[_MATCH[char]]
            if not stack:
                raise ValueError(f"unbalanced {char!r} at position {j}")
            pairs.append((stack.pop(), j))
    for opener, stack in stacks.items():
        if stack:
            raise ValueError(f"unbalanced {opener!r} at position {stack[-1]}")
    return sorted(pairs)


def nested_part(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy maximal nested subset: drop pairs that cross an earlier-kept pair.

    Pairs are visited from the outermost/longest inward, so helices that span
    most of the molecule are kept and short crossing helices are dropped.
    """
    kept: list[tuple[int, int]] = []
    for i, j in sorted(pairs, key=lambda p: (p[0] - p[1], p[0])):
        if all(not (a < i < b < j or i < a < j < b) for a, b in kept):
            kept.append((i, j))
    return sorted(kept)


def dotbracket(length: int, *layers: list[tuple[int, int]]) -> str:
    """Render pair layers with ``()``, ``[]``, ``{}``, ``<>`` in that order."""
    chars = ["."] * length
    for (opener, closer), layer in zip(zip(OPENERS, CLOSERS), layers):
        for i, j in layer:
            if chars[i] != "." or chars[j] != ".":
                raise ValueError(f"position reused by pair {(i, j)}")
            chars[i], chars[j] = opener, closer
    return "".join(chars)
