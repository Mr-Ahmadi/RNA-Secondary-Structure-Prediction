"""Command line interface: ``ekh predict | estimate | tune | evaluate``."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .alignment import read_alignment
from .estimate import Family, estimate_evolution, estimate_grammar
from .metrics import Counts, compare
from .model import DEFAULT_PARAMS, EKH
from .parser import PassParams
from .tree import parse_newick

TRAINING_FAMILIES = ["RF00001", "RF00005", "RF03000"]
GRID = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
FLAG_GRID = [0.25, 0.5, 0.7, 1.0, 1.4, 2.0]
# Ratios found by the genetic algorithm of the original (2025) notebooks.
ORIGINAL_GA_RATIOS = (PassParams(0.19, 1.48), PassParams(0.88, 1.78, 0.98))


def load_benchmark(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())


def cmd_predict(args) -> None:
    model = EKH.load(args.params)
    alignment = read_alignment(args.alignment)
    tree = parse_newick(Path(args.tree).read_text()) if args.tree else None
    prediction = model.predict(alignment, tree=tree, phyml=args.phyml)
    width = max(map(len, alignment))
    for name, seq in alignment.items():
        print(f"{name:<{width}}  {seq}")
    print(f"{'structure':<{width}}  {prediction.structure}")


def cmd_estimate(args) -> None:
    families = [Family.load(args.data, name) for name in args.families]
    evolution, grammar = estimate_evolution(families), estimate_grammar(families)
    out = Path(args.output)
    previous = json.loads(out.read_text()) if out.exists() else {}
    previous.update({"training_families": args.families,
                     "evolution": evolution.to_json(), "grammar": grammar.to_json()})
    out.write_text(json.dumps(previous, indent=2) + "\n")
    print(f"wrote {out}")


def _score(model: EKH, evidence: dict, data: dict) -> tuple[Counts, dict[str, str]]:
    total, structures = Counts(), {}
    for name, ev in evidence.items():
        structures[name] = model.parse(ev).structure
        total = total + compare(structures[name], data[name]["structure"])
    return total, structures


def cmd_tune(args) -> None:
    """Grid search of the pass ratios on the validation set (pooled pair F1).

    The first pass is tuned with the second pass disabled; the second pass is
    then tuned on top of the chosen first pass. Ties go to the setting closest
    to the neutral value 1.
    """
    model = EKH.load(args.params)
    data = load_benchmark(args.validation)
    evidence = {name: model.evidence(fam["alignment"], phyml=args.phyml) for name, fam in data.items()}
    distance = lambda values: sum(abs(np.log(v)) for v in values)  # noqa: E731

    model.second = None
    first = max(
        (PassParams(s, a) for s, a in itertools.product(GRID, GRID)),
        key=lambda p: (round(_score(_with(model, first=p), evidence, data)[0].f1, 10), -distance([p.start, p.accelerate])),
    )
    model.first = first
    second = max(
        (PassParams(s, a, f) for s, a, f in itertools.product(GRID, GRID, FLAG_GRID)),
        key=lambda p: (round(_score(_with(model, second=p), evidence, data)[0].f1, 10),
                       -distance([p.start, p.accelerate, p.flag])),
    )
    one_pass = _score(model, evidence, data)[0]
    model.second = second
    two_pass = _score(model, evidence, data)[0]
    use_second = two_pass.f1 > one_pass.f1
    params_path = Path(args.params)
    params = json.loads(params_path.read_text())
    params["passes"] = {"first": asdict(first), "second": asdict(second) if use_second else None}
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps({"first": asdict(first), "second": asdict(second),
                      "validation_f1_one_pass": one_pass.f1, "validation_f1_two_pass": two_pass.f1,
                      "second_pass_enabled": use_second}, indent=2))


def _with(model: EKH, **passes) -> EKH:
    return EKH(model.evolution, model.grammar, passes.get("first", model.first),
               passes.get("second", model.second))


def cmd_evaluate(args) -> None:
    base = EKH.load(args.params)
    variants = {
        "KH-99 CYK (1 pass, no ratios)": EKH(base.evolution, base.grammar, PassParams(), None),
        "Gap-Bracket, 1st pass only": EKH(base.evolution, base.grammar, base.first, None),
        "Gap-Bracket, 2 passes": EKH(base.evolution, base.grammar, base.first,
                                      base.second or PassParams()),
        "Gap-Bracket, 2 passes, 2025 GA ratios": EKH(base.evolution, base.grammar, *ORIGINAL_GA_RATIOS),
    }
    report = {"params": str(args.params), "sets": {}}
    for set_path in args.sets:
        data = {k: v for k, v in load_benchmark(set_path).items() if k not in args.exclude}
        started = time.perf_counter()
        evidence = {name: base.evidence(fam["alignment"], phyml=args.phyml) for name, fam in data.items()}
        evidence_seconds = time.perf_counter() - started
        result = {"evidence_seconds": evidence_seconds, "variants": {}}
        for label, model in variants.items():
            started = time.perf_counter()
            families = {}
            total = Counts()
            for name, ev in evidence.items():
                prediction = model.parse(ev)
                counts = compare(prediction.structure, data[name]["structure"])
                total = total + counts
                families[name] = {
                    "reference": data[name]["structure"], "predicted": prediction.structure,
                    "n_seqs": len(data[name]["alignment"]), "length": len(prediction.structure),
                    "precision": counts.precision, "recall": counts.recall,
                    "f1": counts.f1, "weighted_f1": counts.weighted_f1,
                }
            result["variants"][label] = {
                "parse_seconds": time.perf_counter() - started,
                "pooled": {"precision": total.precision, "recall": total.recall,
                           "f1": total.f1, "weighted_f1": total.weighted_f1},
                "mean_family_f1": float(np.mean([f["f1"] for f in families.values()])),
                "families": families,
            }
        report["sets"][Path(set_path).stem] = result
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    _print_summary(report)


def _print_summary(report: dict) -> None:
    for set_name, result in report["sets"].items():
        print(f"\n== {set_name} (evidence {result['evidence_seconds']:.2f}s)")
        for label, v in result["variants"].items():
            p = v["pooled"]
            print(f"{label:<38} P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f} "
                  f"meanF1={v['mean_family_f1']:.3f} weightedF1={p['weighted_f1']:.3f} "
                  f"({v['parse_seconds']:.2f}s)")


def main(argv: list[str] | None = None) -> None:
    sys.setrecursionlimit(100_000)  # Newick parsing of large seed trees
    parser = argparse.ArgumentParser(prog="ekh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    phyml_help = "PhyML binary for ML trees (default: neighbour joining + ML branch lengths)"

    p = sub.add_parser("predict", help="predict the consensus structure of an alignment")
    p.add_argument("alignment", help="FASTA, Stockholm or PHYLIP alignment")
    p.add_argument("--tree", help="Newick tree for the alignment")
    p.add_argument("--phyml", help=phyml_help)
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("estimate", help="estimate evolution and grammar parameters")
    p.add_argument("--data", default="data/training")
    p.add_argument("--families", nargs="+", default=TRAINING_FAMILIES)
    p.add_argument("--output", default=DEFAULT_PARAMS)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("tune", help="tune pass ratios on the validation set")
    p.add_argument("--validation", default="data/benchmark/validation.json")
    p.add_argument("--phyml", help=phyml_help)
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("evaluate", help="score the model on benchmark sets")
    p.add_argument("sets", nargs="+")
    p.add_argument("--phyml", help=phyml_help)
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.add_argument("--exclude", nargs="*", default=[], help="families to leave out")
    p.add_argument("--output")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
