"""Accuracy of predicted structures against reference structures.

All base pairs
    PPV (precision), sensitivity (recall) and F1 over base pairs, pooled
    over alignments.
Pseudoknotted pairs
    The same measures restricted to pairs that cross at least one other pair
    of their own structure (:func:`ekh.structure.pseudoknotted_pairs`), i.e.
    both helices of every pseudoknot, independent of bracket layers. A
    predicted pseudoknotted pair is correct if the reference contains it as a
    pseudoknotted pair.
Pseudoknot detection
    Per alignment: does the prediction contain a pseudoknot, given that the
    reference does (sensitivity) or not (specificity)?
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .structure import pairs_from_dotbracket, pseudoknotted_pairs


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pk_tp: int = 0
    pk_fp: int = 0
    pk_fn: int = 0
    alignments: int = 0
    reference_pk: int = 0  # alignments whose reference has a pseudoknot
    detected_pk: int = 0  # ... and whose prediction has one
    false_pk: int = 0  # alignments without a reference pseudoknot but with a predicted one

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    @property
    def ppv(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def sensitivity(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _f1(self.tp, self.fp, self.fn)

    @property
    def pk_ppv(self) -> float:
        return _ratio(self.pk_tp, self.pk_tp + self.pk_fp)

    @property
    def pk_sensitivity(self) -> float:
        return _ratio(self.pk_tp, self.pk_tp + self.pk_fn)

    @property
    def pk_f1(self) -> float:
        return _f1(self.pk_tp, self.pk_fp, self.pk_fn)

    @property
    def detection_sensitivity(self) -> float:
        return _ratio(self.detected_pk, self.reference_pk)

    @property
    def detection_specificity(self) -> float:
        return 1.0 - _ratio(self.false_pk, self.alignments - self.reference_pk)

    def summary(self) -> dict[str, float]:
        return {"ppv": self.ppv, "sensitivity": self.sensitivity, "f1": self.f1,
                "pk_ppv": self.pk_ppv, "pk_sensitivity": self.pk_sensitivity, "pk_f1": self.pk_f1,
                "pk_detection_sensitivity": self.detection_sensitivity,
                "pk_detection_specificity": self.detection_specificity}


def _ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def compare_pairs(predicted: list[tuple[int, int]], reference: list[tuple[int, int]]) -> Counts:
    pred, ref = set(predicted), set(reference)
    pred_pk, ref_pk = set(pseudoknotted_pairs(predicted)), set(pseudoknotted_pairs(reference))
    return Counts(
        len(pred & ref), len(pred - ref), len(ref - pred),
        len(pred_pk & ref_pk), len(pred_pk - ref_pk), len(ref_pk - pred_pk),
        1, int(bool(ref_pk)), int(bool(ref_pk) and bool(pred_pk)), int(not ref_pk and bool(pred_pk)),
    )


def compare(predicted: str, reference: str) -> Counts:
    if len(predicted) != len(reference):
        raise ValueError("structures have different lengths")
    return compare_pairs(pairs_from_dotbracket(predicted), pairs_from_dotbracket(reference))
