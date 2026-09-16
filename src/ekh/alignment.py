"""Alignment I/O and the per-sequence nucleotide likelihood encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUCLEOTIDES = "ACGU"

# IUPAC codes -> compatible nucleotides. Gaps and unknown symbols are treated
# as "any nucleotide" (all ones), following Knudsen & Hein (2003).
IUPAC = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "R": "AG", "Y": "CU", "S": "CG", "W": "AU", "K": "GU", "M": "AC",
    "B": "CGU", "D": "AGU", "H": "ACU", "V": "ACG", "N": "ACGU",
}


def normalise(sequence: str) -> str:
    return sequence.upper().replace("T", "U").replace(".", "-")


def tip_likelihoods(sequences: list[str]) -> np.ndarray:
    """Encode an alignment as ``(n_seqs, n_cols, 4)`` indicator vectors.

    Entry ``[s, i, x]`` is 1 when nucleotide ``x`` is compatible with the symbol
    of sequence ``s`` at column ``i``; gaps are all ones (missing data).
    """
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"aligned sequences must have equal length, got {sorted(lengths)}")
    table = np.ones((256, 4))
    for code, compatible in IUPAC.items():
        table[ord(code)] = [x in compatible for x in NUCLEOTIDES]
    codes = np.frombuffer("".join(normalise(s) for s in sequences).encode(), dtype=np.uint8)
    return table[codes].reshape(len(sequences), lengths.pop(), 4)


def read_phylip(path: str | Path) -> dict[str, str]:
    """Read a relaxed, interleaved or sequential PHYLIP alignment."""
    lines = [line.rstrip("\n") for line in Path(path).read_text().splitlines()]
    n_seqs, n_cols = map(int, lines[0].split()[:2])
    body = [line for line in lines[1:] if line.strip()]
    names, chunks = [], []
    for line in body[:n_seqs]:
        name, _, rest = line.strip().partition(" ")
        names.append(name)
        chunks.append([rest.replace(" ", "")])
    for k, line in enumerate(body[n_seqs:]):
        chunks[k % n_seqs].append(line.replace(" ", ""))
    alignment = {name: normalise("".join(parts)) for name, parts in zip(names, chunks)}
    bad = [name for name, seq in alignment.items() if len(seq) != n_cols]
    if bad:
        raise ValueError(f"{path}: {len(bad)} sequences do not have {n_cols} columns")
    return alignment


def read_fasta(path: str | Path) -> dict[str, str]:
    alignment: dict[str, str] = {}
    name = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            alignment[name] = ""
        elif line and name is not None:
            alignment[name] += line
    return {k: normalise(v) for k, v in alignment.items()}


def read_stockholm(path: str | Path) -> tuple[dict[str, str], str | None]:
    """Return the alignment and the ``#=GC SS_cons`` line (if present)."""
    alignment: dict[str, str] = {}
    ss_cons = ""
    for line in Path(path).read_text().splitlines():
        if line.startswith("#=GC SS_cons"):
            ss_cons += line.split()[-1]
        elif line and not line.startswith(("#", "//")):
            name, seq = line.split()[:2]
            alignment[name] = alignment.get(name, "") + seq
    return {k: normalise(v) for k, v in alignment.items()}, ss_cons or None


def read_alignment(path: str | Path) -> dict[str, str]:
    path = Path(path)
    head = path.read_text()[:200].lstrip()
    if head.startswith("# STOCKHOLM"):
        return read_stockholm(path)[0]
    if head.startswith(">"):
        return read_fasta(path)
    return read_phylip(path)
