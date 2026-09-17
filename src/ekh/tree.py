"""Phylogenetic trees: Newick parsing, neighbour joining and midpoint rooting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    name: str | None = None
    length: float = 0.0  # branch length to the parent
    children: list["Node"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children

    def postorder(self):
        stack, out = [self], []
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children)
        return reversed(out)

    def leaves(self) -> list["Node"]:
        return [n for n in self.postorder() if n.is_leaf()]


def parse_newick(text: str) -> Node:
    """Parse Newick; internal-node labels (e.g. bootstrap values) are ignored.

    The parser keeps an explicit stack, so deep (caterpillar) trees of large
    seed alignments do not hit Python's recursion limit.
    """
    text = text.strip().rstrip(";")
    root = current = Node()  # "(" makes the current node internal
    parents: list[Node] = []
    pos = 0

    def label() -> str:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] not in ",():":
            pos += 1
        return text[start:pos].strip()

    while pos < len(text):
        char = text[pos]
        if char == "(":
            child = Node()
            current.children.append(child)
            parents.append(current)
            current = child
            pos += 1
        elif char == ",":
            child = Node()
            parents[-1].children.append(child)
            current = child
            pos += 1
        elif char == ")":
            current = parents.pop()
            pos += 1
            label()  # internal label, ignored
        elif char == ":":
            pos += 1
            current.length = float(label())
        else:
            current.name = label()
    if parents:
        raise ValueError("unbalanced parentheses in Newick tree")
    return root


def p_distance_matrix(sequences: list[str]) -> np.ndarray:
    """Jukes-Cantor corrected distances over columns where both have a base."""
    codes = np.array([np.frombuffer(s.encode(), dtype=np.uint8) for s in sequences])
    is_base = np.isin(codes, np.frombuffer(b"ACGU", dtype=np.uint8))
    n = len(sequences)
    dist = np.zeros((n, n))
    for a in range(n):
        both = is_base[a] & is_base[a + 1:]
        diff = (codes[a] != codes[a + 1:]) & both
        p = diff.sum(1) / np.maximum(both.sum(1), 1)
        p = np.minimum(p, 0.74)
        dist[a, a + 1:] = dist[a + 1:, a] = -0.75 * np.log(1 - 4 * p / 3)
    return dist


def neighbour_joining(names: list[str], dist: np.ndarray) -> Node:
    """Saitou & Nei neighbour joining; returns an unrooted (trifurcating) tree."""
    nodes = [Node(name=n) for n in names]
    d = dist.astype(float).copy()
    while len(nodes) > 3:
        m = len(nodes)
        r = d.sum(1)
        q = (m - 2) * d - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        a, b = np.unravel_index(np.argmin(q), q.shape)
        la = 0.5 * d[a, b] + (r[a] - r[b]) / (2 * (m - 2))
        nodes[a].length, nodes[b].length = max(la, 0.0), max(d[a, b] - la, 0.0)
        joined = Node(children=[nodes[a], nodes[b]])
        new_row = 0.5 * (d[a] + d[b] - d[a, b])
        keep = [k for k in range(m) if k not in (a, b)]
        d = np.vstack([np.append(d[np.ix_(keep, keep)], new_row[keep][:, None], 1),
                       np.append(new_row[keep], 0.0)])
        nodes = [nodes[k] for k in keep] + [joined]
    root = Node(children=nodes)
    if len(nodes) == 3:
        d01, d02, d12 = d[0, 1], d[0, 2], d[1, 2]
        lengths = [(d01 + d02 - d12) / 2, (d01 + d12 - d02) / 2, (d02 + d12 - d01) / 2]
        for node, length in zip(nodes, lengths):
            node.length = max(length, 0.0)
    elif len(nodes) == 2:
        nodes[0].length = nodes[1].length = d[0, 1] / 2
    return root


def midpoint_root(tree: Node) -> Node:
    """Re-root on the midpoint of the longest leaf-to-leaf path."""
    adjacency: dict[int, list[tuple[Node, float]]] = {}
    for node in tree.postorder():
        for child in node.children:
            adjacency.setdefault(id(node), []).append((child, child.length))
            adjacency.setdefault(id(child), []).append((node, child.length))

    def farthest(start: Node):
        best, parent, stack = (0.0, start), {id(start): None}, [(start, 0.0)]
        while stack:
            node, depth = stack.pop()
            if depth > best[0]:
                best = (depth, node)
            for nxt, length in adjacency.get(id(node), []):
                if id(nxt) not in parent:
                    parent[id(nxt)] = (node, length)
                    stack.append((nxt, depth + length))
        return best, parent

    (_, a), _ = farthest(tree.leaves()[0])
    (total, b), parent = farthest(a)
    if total == 0.0:
        return tree
    # Walk from b back towards a until the midpoint edge is found.
    remaining, node = total / 2, b
    while True:
        up, length = parent[id(node)]
        if length >= remaining or parent[id(up)] is None:  # guard against rounding
            remaining = min(remaining, length)
            break
        remaining -= length
        node = up

    new_root = Node(children=[Node(length=remaining), Node(length=length - remaining)])
    stack = [(node, up, new_root.children[0]), (up, node, new_root.children[1])]
    while stack:  # copy the tree, re-oriented away from the new root
        current, came_from, copy = stack.pop()
        copy.name = current.name if current.is_leaf() else None
        for nxt, edge in adjacency.get(id(current), []):
            if nxt is not came_from:
                child = Node(length=edge)
                copy.children.append(child)
                stack.append((nxt, current, child))
    return new_root


def nj_tree(alignment: dict[str, str]) -> Node:
    """Midpoint-rooted neighbour-joining tree from Jukes-Cantor distances."""
    names = list(alignment)
    if len(names) == 1:
        return Node(children=[Node(name=names[0])])
    return midpoint_root(neighbour_joining(names, p_distance_matrix(list(alignment.values()))))
