"""Accuracy of predicted structures against reference structures."""

from __future__ import annotations

from dataclasses import dataclass

from .structure import pairs_from_dotbracket


@dataclass
class Counts:
    tp: int = 0  # correctly predicted base pairs
    fp: int = 0
    fn: int = 0
    unpaired_tp: int = 0  # columns correctly predicted unpaired
    unpaired_fp: int = 0
    unpaired_fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(*(a + b for a, b in zip(vars(self).values(), vars(other).values())))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """Base-pair F1, the standard secondary-structure accuracy measure."""
        return _f1(self.tp, self.fp, self.fn)

    @property
    def weighted_f1(self) -> float:
        """Legacy score of the original notebooks: pair F1 and unpaired-column F1
        averaged with weights 2 x (reference pairs) and (reference unpaired)."""
        pair_weight = 2 * (self.tp + self.fn)
        unpaired_weight = self.unpaired_tp + self.unpaired_fn
        total = pair_weight + unpaired_weight
        unpaired_f1 = _f1(self.unpaired_tp, self.unpaired_fp, self.unpaired_fn)
        return (pair_weight * self.f1 + unpaired_weight * unpaired_f1) / total if total else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def compare(predicted: str, reference: str) -> Counts:
    if len(predicted) != len(reference):
        raise ValueError("structures have different lengths")
    pred, ref = set(pairs_from_dotbracket(predicted)), set(pairs_from_dotbracket(reference))
    pred_free = set(range(len(predicted))) - {k for p in pred for k in p}
    ref_free = set(range(len(reference))) - {k for p in ref for k in p}
    return Counts(
        len(pred & ref), len(pred - ref), len(ref - pred),
        len(pred_free & ref_free), len(pred_free - ref_free), len(ref_free - pred_free),
    )
