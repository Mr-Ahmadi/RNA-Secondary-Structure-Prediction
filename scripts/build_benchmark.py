"""Build the Rfam benchmark: clan-disjoint training, tuning and test sets.

Source: Rfam 15.1 seed alignments (``data/rfam/Rfam.seed.gz``) and clan
membership (``data/rfam/Rfam.clanin``); ``scripts/fetch_rfam.sh`` downloads them.

Selection
    Families whose consensus structure comes from the literature (``#=GF SS``
    starting with ``Published`` or ``Pseudobase``); structures annotated as
    ``Predicted`` (e.g. by RNAalifold or CMfinder) are excluded, since they
    would reward agreement with those tools. At least four seed sequences, and
    a structure as long as the alignment.

Split
    Families of one Rfam clan stay together; a family without a clan is its own
    group. Groups are shuffled with a fixed seed within two strata (with or
    without a pseudoknotted family) and dealt 50 / 15 / 35 % to training,
    tuning and test.

Alignments
    Training families keep up to 100 random seed sequences, used to estimate
    the substitution model and the grammar. Each tuning and test family
    contributes one alignment of 4-12 random seed sequences, as a user would
    supply. In every alignment, columns where at most half of the chosen
    sequences have a base are dropped, together with the reference pairs that
    use them. Alignments with fewer than 3 reference pairs are skipped, and so
    are tuning and test alignments with more than 400 columns.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ekh.alignment import normalise  # noqa: E402
from ekh.structure import pairs_from_dotbracket, pseudoknotted_pairs, to_dotbracket  # noqa: E402

SEED = 2026
FRACTIONS = {"training": 0.50, "tuning": 0.15, "test": 0.35}
MAX_TRAINING_SEQS, MIN_SEQS, MAX_SEQS, MAX_COLUMNS, MIN_PAIRS = 100, 4, 12, 400, 3


def read_seed(path: Path):
    family = None
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("# STOCKHOLM"):
                family = {"gf": defaultdict(str), "seqs": {}, "ss": ""}
            elif line.startswith("#=GF"):
                family["gf"][line[5:7]] += line[10:].strip() + " "
            elif line.startswith("#=GC SS_cons"):
                family["ss"] += line.split()[-1]
            elif line.startswith("//"):
                yield family
            elif line.strip() and not line.startswith("#"):
                name, seq = line.split()[:2]
                family["seqs"][name] = family["seqs"].get(name, "") + seq


def project(seqs: list[str], pairs, min_fraction: float = 0.5):
    """Keep columns where more than ``min_fraction`` of the sequences have a base."""
    n = len(seqs)
    keep = [c for c in range(len(seqs[0])) if sum(s[c] not in "-." for s in seqs) > min_fraction * n]
    column = {c: k for k, c in enumerate(keep)}
    kept_pairs = [(column[i], column[j]) for i, j in pairs if i in column and j in column]
    return ["".join(s[c] for c in keep) for s in seqs], kept_pairs


def main() -> None:
    clan_of = {}
    for line in (ROOT / "data/rfam/Rfam.clanin").read_text().splitlines():
        clan, *members = line.split()
        clan_of.update({member: clan for member in members})

    families = []
    for fam in read_seed(ROOT / "data/rfam/Rfam.seed.gz"):
        source = fam["gf"]["SS"].strip()
        if not source.startswith(("Published", "Pseudobase")) or len(fam["seqs"]) < MIN_SEQS:
            continue
        if {len(s) for s in fam["seqs"].values()} != {len(fam["ss"])}:
            continue  # malformed entry (structure and alignment lengths differ)
        try:
            pairs = pairs_from_dotbracket(fam["ss"])
        except ValueError:
            continue
        accession, name = fam["gf"]["AC"].strip(), fam["gf"]["ID"].strip()
        families.append({
            "accession": accession, "name": name, "clan": clan_of.get(name, accession),
            "source": source.split(";")[0], "type": fam["gf"]["TP"].strip(),
            "seqs": {k: normalise(v) for k, v in fam["seqs"].items()}, "pairs": pairs,
            "pseudoknot": bool(pseudoknotted_pairs(pairs)),
        })

    groups = defaultdict(list)
    for fam in families:
        groups[fam["clan"]].append(fam)
    rng = np.random.default_rng(SEED)
    split_of = {}
    for stratum in (True, False):
        clans = sorted(c for c, fams in groups.items() if any(f["pseudoknot"] for f in fams) == stratum)
        clans = [clans[k] for k in rng.permutation(len(clans))]
        bounds = np.cumsum([round(FRACTIONS[s] * len(clans)) for s in ("training", "tuning")])
        for k, clan in enumerate(clans):
            split_of[clan] = "training" if k < bounds[0] else "tuning" if k < bounds[1] else "test"

    out = {split: {} for split in FRACTIONS}
    for fam in sorted(families, key=lambda f: f["accession"]):
        split = split_of[fam["clan"]]
        names = sorted(fam["seqs"])
        size = min(len(names), MAX_TRAINING_SEQS) if split == "training" else \
            min(len(names), int(rng.integers(MIN_SEQS, MAX_SEQS + 1)))
        chosen = sorted(rng.choice(len(names), size, replace=False))
        seqs = [fam["seqs"][names[k]] for k in chosen]
        aligned, pairs = project(seqs, fam["pairs"])
        if len(pairs) < MIN_PAIRS or (split != "training" and len(aligned[0]) > MAX_COLUMNS):
            continue
        out[split][fam["accession"]] = {
            "name": fam["name"], "clan": fam["clan"], "source": fam["source"], "type": fam["type"],
            "pseudoknot": bool(pseudoknotted_pairs(pairs)),
            "alignment": {names[k]: s for k, s in zip(chosen, aligned)},
            "structure": to_dotbracket(len(aligned[0]), pairs),
        }

    for split, data in out.items():
        path = ROOT / "data/benchmark" / f"{split}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        lengths = [len(next(iter(f["alignment"].values()))) for f in data.values()]
        print(f"{split:<9} {len(data):>4} alignments, {sum(f['pseudoknot'] for f in data.values()):>3} pseudoknotted, "
              f"{len({f['clan'] for f in data.values()}):>4} clan groups, columns {min(lengths)}-{max(lengths)}")


if __name__ == "__main__":
    main()
