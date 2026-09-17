"""Tuning of the decoding settings, with the tuning method chosen by cross-validation.

A decoding has seven tunable settings: the helix initiation and extension
ratios of each pass, the crossing ratio of the second pass, the evidence
weight and the pseudoknot pair ratio. Each takes values on a fixed logarithmic grid, so a candidate is a
vector of grid indices. The minimum hairpin loop is fixed (see ``Decoding``).

Objective
    Pooled base-pair F1 (the mean over tuning sets if there are several).
    Ties go to the candidate closest to neutral (all 1).

Tuning methods
    ``shared-exhaustive``
        Both passes share one initiation and one extension ratio, and the
        crossing ratio is 1. The remaining four settings (initiation,
        extension, evidence weight, pseudoknot pair ratio) are scored on their
        whole grid (17 x 17 x 9 x 9 candidates), so the global optimum is found.
    ``separate-coordinate``
        All seven settings, by coordinate ascent from neutral: one setting at a
        time, until a full sweep changes nothing.
    ``separate-genetic``
        All seven settings, by a seeded genetic algorithm (tournament selection,
        uniform crossover, 1-2 step mutations, elitism), polished by
        coordinate ascent. This is the kind of search the 2025 notebooks used.

Choosing the method
    :func:`cross_validate` splits the tuning alignments into folds by group
    (Rfam clan), tunes on all folds but one and scores the fold left out.
    Pooling these held-out counts estimates how each method does on unseen
    clans. ``ekh tune`` keeps the method with the best estimate (ties: fewer
    settings) and runs it on all tuning data.

Base-pair counts are cached per (candidate, alignment), so folds and methods
share work, and uncached alignments are parsed in parallel worker processes.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .model import EKH, ColumnEvidence, Decoding
from .parser import PassSettings

RATIO_GRID = tuple(float(x) for x in np.round(2.0 ** (np.arange(-8, 9) / 3), 3))  # 0.157 .. 6.35
WEIGHT_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25)
PSEUDOKNOT_GRID = tuple(float(x) for x in np.round(10.0 ** (-np.arange(9) / 4), 3))  # 1 .. 0.01
SETTINGS = ("first.initiation", "first.extension", "second.initiation", "second.extension", "second.crossing",
            "evidence_weight", "pseudoknot_ratio")
GRIDS = (RATIO_GRID,) * 5 + (WEIGHT_GRID, PSEUDOKNOT_GRID)
NEUTRAL = tuple(grid.index(1.0) for grid in GRIDS)
WEIGHT = SETTINGS.index("evidence_weight")
PSEUDOKNOT = SETTINGS.index("pseudoknot_ratio")
METHODS = ("shared-exhaustive", "separate-coordinate", "separate-genetic")  # fewest settings first

Point = tuple[int, ...]


def to_decoding(point: Point, base: Decoding = Decoding()) -> Decoding:
    """Decoding with the grid values of ``point``; other fields come from ``base``."""
    v = [grid[k] for grid, k in zip(GRIDS, point)]
    return replace(base, first=PassSettings(v[0], v[1]), second=PassSettings(v[2], v[3], v[4]), evidence_weight=v[5],
                   pseudoknot_ratio=v[6])


def to_point(decoding: Decoding) -> Point:
    """Nearest grid point of a decoding."""
    second = decoding.second or PassSettings()
    values = (decoding.first.initiation, decoding.first.extension, second.initiation, second.extension,
              second.crossing, decoding.evidence_weight, decoding.pseudoknot_ratio)
    return tuple(int(np.argmin([abs(np.log(g / v)) for g in grid])) for grid, v in zip(GRIDS, values))


def distance_from_neutral(point: Point) -> float:
    return float(sum(abs(np.log(grid[k])) for grid, k in zip(GRIDS, point)))


@dataclass
class Item:
    set_name: str
    group: str  # alignments of one group (Rfam clan) are held out together
    name: str
    evidence: ColumnEvidence
    reference: frozenset[tuple[int, int]]


# ---------------------------------------------------------------- evaluation

_WORKER: dict = {}


def _init_worker(model: EKH, items: list[Item], base: Decoding) -> None:
    _WORKER.update(model=model, items=items, base=base, first={})


def _count(item_index: int, points: list[Point]) -> list[tuple[int, int, int]]:
    """(tp, fp, fn) base-pair counts of one alignment for many candidates."""
    model, item, first = _WORKER["model"], _WORKER["items"][item_index], _WORKER["first"]
    out = []
    for point in points:
        m = model.with_decoding(to_decoding(point, _WORKER["base"]))
        key = (item_index, point[0], point[1], point[WEIGHT])  # the first pass ignores second-pass settings
        if key not in first:
            first[key] = m.first_pass(item.evidence)
        predicted = set(sum(m.pairs(item.evidence, nested=first[key]), []))
        tp = len(predicted & item.reference)
        out.append((tp, len(predicted) - tp, len(item.reference) - tp))
    return out


class Evaluator:
    """Cached, parallel base-pair counts of candidates on the tuning alignments."""

    def __init__(self, model: EKH, items: list[Item], jobs: int | None = None, base: Decoding = Decoding()):
        self.items, self.base = items, base
        self.cache: dict[Point, np.ndarray] = {}  # point -> (n_items, 3)
        jobs = jobs or min(8, os.cpu_count() or 1)
        _init_worker(model, items, base)
        self.pool = ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(model, items, base)) if jobs > 1 else None
        self.set_names = sorted({it.set_name for it in items})
        self.set_masks = np.array([[it.set_name == s for it in items] for s in self.set_names])
        self.everything = np.ones(len(items), dtype=bool)

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc) -> None:
        if self.pool:
            self.pool.shutdown()

    def counts(self, points: list[Point]) -> np.ndarray:
        """(n_points, n_items, 3) array of (tp, fp, fn)."""
        missing = list(dict.fromkeys(p for p in points if p not in self.cache))
        if missing:
            chunks = [missing[k:k + 64] for k in range(0, len(missing), 64)]
            tasks = [(i, chunk) for chunk in chunks for i in range(len(self.items))]
            if self.pool:
                results = self.pool.map(_count, *zip(*tasks), chunksize=max(1, len(tasks) // 64))
            else:
                results = (_count(i, chunk) for i, chunk in tasks)
            for (i, chunk), result in zip(tasks, results):
                for point, row in zip(chunk, result):
                    self.cache.setdefault(point, np.zeros((len(self.items), 3), dtype=np.int64))[i] = row
        return np.array([self.cache[p] for p in points])

    def set_f1(self, counts: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Pooled F1 per set of ``counts`` (n_items, 3) over the items in ``mask``."""
        f1 = self._set_f1(counts[None], mask)[0]
        return {name: float(f1[k]) for k, name in enumerate(self.set_names) if (self.set_masks[k] & mask).any()}

    def _set_f1(self, counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """(n_points, n_sets) pooled F1 for (n_points, n_items, 3) counts."""
        out = []
        for set_mask in self.set_masks:
            tp, fp, fn = np.moveaxis(counts[:, set_mask & mask].sum(1), -1, 0)
            out.append(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
        return np.array(out).T

    def objective(self, points: list[Point], mask: np.ndarray) -> np.ndarray:
        used = [(s & mask).any() for s in self.set_masks]
        return self._set_f1(self.counts(points), mask)[:, used].mean(1)

    def best(self, points: list[Point], mask: np.ndarray) -> Point:
        scores = np.round(self.objective(points, mask), 10)
        return max(zip(scores, points), key=lambda sp: (sp[0], -distance_from_neutral(sp[1])))[1]


# ---------------------------------------------------------------- tuning methods


def shared_grid(fix_weight: bool = False) -> list[Point]:
    """All candidates with shared ratios in both passes and crossing ratio 1."""
    one = NEUTRAL[4]
    weights = [NEUTRAL[WEIGHT]] if fix_weight else range(len(WEIGHT_GRID))
    return [(s, e, s, e, one, w, k) for s in range(len(RATIO_GRID)) for e in range(len(RATIO_GRID))
            for w in weights for k in range(len(PSEUDOKNOT_GRID))]


def coordinate_ascent(evaluator: Evaluator, mask: np.ndarray, start: Point = NEUTRAL,
                      frozen: tuple[int, ...] = (), max_sweeps: int = 20) -> Point:
    current = start
    for _ in range(max_sweeps):
        previous = current
        for k, grid in enumerate(GRIDS):
            if k not in frozen:
                current = evaluator.best([current[:k] + (v,) + current[k + 1:] for v in range(len(grid))], mask)
        if current == previous:
            break
    return current


def genetic(evaluator: Evaluator, mask: np.ndarray, frozen: tuple[int, ...] = (), population: int = 32,
            generations: int = 30, elite: int = 4, seed: int = 0) -> Point:
    rng = np.random.default_rng(seed)
    free = [k for k in range(len(GRIDS)) if k not in frozen]

    def clip(genes: list[int]) -> Point:
        for k in frozen:
            genes[k] = NEUTRAL[k]
        return tuple(int(np.clip(v, 0, len(GRIDS[k]) - 1)) for k, v in enumerate(genes))

    def ranked(points: list[Point]) -> list[Point]:
        scores = np.round(evaluator.objective(points, mask), 10)
        order = sorted(range(len(points)), key=lambda i: (scores[i], -distance_from_neutral(points[i])), reverse=True)
        return [points[i] for i in order]

    pop = ranked([NEUTRAL] + [clip([int(rng.integers(len(g))) for g in GRIDS]) for _ in range(population - 1)])
    stale = 0
    for _ in range(generations):
        children = pop[:elite]
        while len(children) < population:
            a, b = (min(rng.choice(population, 3, replace=False)) for _ in range(2))  # tournaments
            take_a = rng.random(len(GRIDS)) < 0.5
            genes = [pop[a][k] if take_a[k] else pop[b][k] for k in range(len(GRIDS))]
            for k in free:
                if rng.random() < 1.5 / len(free):
                    genes[k] += int(rng.choice([-2, -1, 1, 2]))
            children.append(clip(genes))
        leader = pop[0]
        pop = ranked(children)
        stale = stale + 1 if pop[0] == leader else 0
        if stale >= 8:
            break
    return coordinate_ascent(evaluator, mask, pop[0], frozen)


def fit(method: str, evaluator: Evaluator, mask: np.ndarray, fix_weight: bool = False) -> Point:
    """Tune on the items in ``mask``; ``fix_weight`` keeps the evidence weight at 1."""
    frozen = (WEIGHT,) if fix_weight else ()
    if method == "shared-exhaustive":
        return evaluator.best(shared_grid(fix_weight), mask)
    if method == "separate-coordinate":
        return coordinate_ascent(evaluator, mask, NEUTRAL, frozen)
    if method == "separate-genetic":
        return genetic(evaluator, mask, frozen)
    raise ValueError(f"unknown tuning method {method!r}; choose from {METHODS}")


def group_folds(items: list[Item], n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Held-out masks: groups are shuffled with a fixed seed and dealt round-robin."""
    groups = sorted({it.group for it in items})
    order = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {groups[g]: k % n_folds for k, g in enumerate(order)}
    labels = np.array([fold_of[it.group] for it in items])
    return [labels == k for k in range(min(n_folds, len(groups)))]


def cross_validate(evaluator: Evaluator, method: str, fix_weight: bool = False, n_folds: int = 10,
                   log: Callable[[str], None] = print) -> dict:
    """Grouped K-fold estimate of a tuning method on unseen groups (Rfam clans)."""
    held_out = np.zeros((len(evaluator.items), 3), dtype=np.int64)
    chosen = []
    for test in group_folds(evaluator.items, n_folds):
        point = fit(method, evaluator, ~test, fix_weight)
        held_out[test] = evaluator.counts([point])[0][test]
        chosen.append(to_decoding(point, evaluator.base).to_json())
    f1 = evaluator.set_f1(held_out, evaluator.everything)
    distinct = len({json.dumps(c, sort_keys=True) for c in chosen})
    log(f"{method}{' (evidence weight 1)' if fix_weight else ''}: held-out F1 "
        + ", ".join(f"{k} {v:.3f}" for k, v in f1.items()) + f"; {distinct} distinct settings over {len(chosen)} folds")
    return {"f1": f1, "objective": float(np.mean(list(f1.values()))), "distinct_settings": distinct,
            "fold_decodings": chosen}


def neutral_score(evaluator: Evaluator) -> dict:
    """Score of the untuned two-pass parser (all settings 1); a floor for any tuning method."""
    f1 = evaluator.set_f1(evaluator.counts([NEUTRAL])[0], evaluator.everything)
    return {"f1": f1, "objective": float(np.mean(list(f1.values())))}
