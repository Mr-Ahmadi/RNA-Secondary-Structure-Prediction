"""Phylogenetic trees: Newick I/O, neighbour joining and midpoint rooting."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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

    def newick(self) -> str:
        def fmt(node: Node) -> str:
            label = node.name or ""
            if node.children:
                label = "(" + ",".join(fmt(c) for c in node.children) + ")" + label
            return f"{label}:{node.length:.8f}"

        return fmt(self).rsplit(":", 1)[0] + ";"


def parse_newick(text: str) -> Node:
    """Parse Newick; internal-node labels (e.g. bootstrap values) are ignored."""
    text = text.strip().rstrip(";")
    pos = 0

    def label() -> str:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] not in ",():;":
            pos += 1
        return text[start:pos].strip()

    def subtree() -> Node:
        nonlocal pos
        node = Node()
        if text[pos] == "(":
            pos += 1
            node.children.append(subtree())
            while text[pos] == ",":
                pos += 1
                node.children.append(subtree())
            pos += 1  # ')'
            label()
        else:
            node.name = label()
        if pos < len(text) and text[pos] == ":":
            pos += 1
            node.length = float(label())
        return node

    return subtree()


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
    while parent[id(node)] is not None:
        up, length = parent[id(node)]
        if length >= remaining:
            break
        remaining -= length
        node = up
    up, length = parent[id(node)]

    def build(current: Node, came_from: Node | None, length_to_parent: float) -> Node:
        new = Node(name=current.name if current.is_leaf() else None, length=length_to_parent)
        for nxt, edge in adjacency.get(id(current), []):
            if nxt is not came_from:
                new.children.append(build(nxt, current, edge))
        return new

    return Node(children=[build(node, up, remaining), build(up, node, length - remaining)])


def estimate_tree(alignment: dict[str, str], phyml: str | None = None) -> Node:
    """Midpoint-rooted tree: PhyML (GTR) when a binary is given, otherwise NJ."""
    names = list(alignment)
    if len(names) < 3 or phyml is None:
        if len(names) == 1:
            return Node(children=[Node(name=names[0])])
        return midpoint_root(neighbour_joining(names, p_distance_matrix(list(alignment.values()))))
    return midpoint_root(_run_phyml(alignment, phyml))


def _run_phyml(alignment: dict[str, str], phyml: str) -> Node:
    binary = shutil.which(phyml) or phyml
    ids = {f"s{k}": name for k, name in enumerate(alignment)}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aln.phy"
        rows = [f"{sid}  {alignment[name]}" for sid, name in ids.items()]
        path.write_text(f"{len(ids)} {len(next(iter(alignment.values())))}\n" + "\n".join(rows) + "\n")
        subprocess.run([binary, "-i", str(path), "-m", "GTR", "-b", "0"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tree = parse_newick((Path(tmp) / "aln.phy_phyml_tree.txt").read_text())
    for leaf in tree.leaves():
        leaf.name = ids[leaf.name]
    return tree


def pairwise_leaf_distances(tree: Node) -> dict[tuple[str, str], float]:
    """Patristic distances between all pairs of leaves."""
    depth: dict[int, float] = {id(tree): 0.0}
    ancestors: dict[int, list[Node]] = {id(tree): [tree]}
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in node.children:
            depth[id(child)] = depth[id(node)] + child.length
            ancestors[id(child)] = ancestors[id(node)] + [child]
            stack.append(child)
    leaves = tree.leaves()
    out = {}
    for a in leaves:
        anc_a = {id(x) for x in ancestors[id(a)]}
        for b in leaves:
            lca = next(x for x in reversed(ancestors[id(b)]) if id(x) in anc_a)
            out[a.name, b.name] = depth[id(a)] + depth[id(b)] - 2 * depth[id(lca)]
    return out
