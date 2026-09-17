"""Command line interface: ``ekh predict | estimate | tune | evaluate``."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .alignment import read_alignment
from .estimate import estimate
from .metrics import Counts, compare
from .model import DEFAULT_PARAMS, EKH, Decoding
from .structure import pairs_from_dotbracket, pseudoknotted_pairs
from . import tune
from .tree import parse_newick

TRAINING_SET = "data/benchmark/training.json"
TUNING_SETS = ["data/benchmark/tuning.json"]

MODEL = "Gap-Bracket"
KH99 = "KH-99 CYK"
FIRST_LAYER = "Gap-Bracket, nested layer only"
NO_REFINEMENT = "Gap-Bracket, no refinement"
HAIRPIN_2 = "Gap-Bracket, minimum hairpin 2"
NO_WEIGHT = "Gap-Bracket, evidence weight 1"
METHOD_NAMES = {"shared-exhaustive": "shared ratios, exhaustive search",
                "separate-coordinate": "separate ratios, coordinate ascent",
                "separate-genetic": "separate ratios, genetic algorithm"}


def load_benchmark(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())


def cmd_predict(args) -> None:
    model = EKH.load(args.params)
    alignment = read_alignment(args.alignment)
    tree = parse_newick(Path(args.tree).read_text()) if args.tree else None
    prediction = model.predict(alignment, tree=tree)
    width = max(map(len, alignment))
    for name, seq in alignment.items():
        print(f"{name:<{width}}  {seq}")
    print(f"{'structure':<{width}}  {prediction.structure}")


def cmd_estimate(args) -> None:
    evolution, grammar, families = estimate(args.training, args.jobs)
    out = Path(args.output)
    previous = json.loads(out.read_text()) if out.exists() else {}
    previous.update({"training": {"set": str(args.training), "families": len(families)},
                     "evolution": evolution.to_json(), "grammar": grammar.to_json()})
    previous.pop("training_families", None)
    out.write_text(json.dumps(previous, indent=2) + "\n")
    print(f"estimated from {len(families)} training families; wrote {out}")


def _tuning_items(model: EKH, paths: list[str]) -> list[tune.Item]:
    items = []
    for path in paths:
        for name, fam in load_benchmark(path).items():
            items.append(tune.Item(Path(path).stem, fam.get("clan", name), name, model.evidence(fam["alignment"]),
                                   frozenset(pairs_from_dotbracket(fam["structure"]))))
    return items


def cmd_tune(args) -> None:
    """Choose the tuning method by leave-one-family-out CV, then tune on all sets.

    Writes the chosen fit as the decoding. As baselines it stores the fits of
    the other methods, the chosen method with the evidence weight fixed at 1
    (ablation) and the historical settings.
    """
    started = time.perf_counter()
    model = EKH.load(args.params)
    methods = tune.METHODS if args.method == "auto" else (args.method,)
    base = Decoding(refinement_rounds=args.refinement_rounds)
    with tune.Evaluator(model, _tuning_items(model, args.sets), args.jobs, base) as evaluator:
        print(f"{len(evaluator.items)} tuning alignments; 10-fold cross-validation grouped by clan")
        cv = {"neutral (no tuning)": tune.neutral_score(evaluator)}
        for method in methods:
            cv[method] = tune.cross_validate(evaluator, method)
        chosen = max(methods, key=lambda m: (round(cv[m]["objective"], 10), -methods.index(m)))
        cv[f"{chosen}, evidence weight 1"] = tune.cross_validate(evaluator, chosen, fix_weight=True)

        fits = {MODEL: (chosen, False), NO_WEIGHT: (chosen, True)}
        fits.update({f"Gap-Bracket, {METHOD_NAMES[m]}": (m, False) for m in methods if m != chosen})
        final = {}
        for label, (method, fix_weight) in fits.items():
            point = tune.fit(method, evaluator, evaluator.everything, fix_weight)
            final[label] = {"method": method, "decoding": tune.to_decoding(point, evaluator.base).to_json(),
                            "tuning_f1": evaluator.set_f1(evaluator.counts([point])[0], evaluator.everything)}
        n_candidates = len(evaluator.cache)

    summary = {"tuning_sets": args.sets, "chosen_method": chosen, "cross_validation": cv, "fits": final,
               "candidates_scored": n_candidates, "seconds": round(time.perf_counter() - started, 1)}
    if args.cv_output:
        Path(args.cv_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"chosen: {chosen}")
    print(json.dumps({label: fit["decoding"] for label, fit in final.items()}, indent=2))
    print(f"{n_candidates} candidates scored in {summary['seconds']} s")
    if args.dry_run:
        return
    params_path = Path(args.params)
    params = json.loads(params_path.read_text())
    params["decoding"] = final[MODEL]["decoding"]
    params["tuning"] = {"sets": args.sets, "method": chosen, "cross_validated_objective": cv[chosen]["objective"]}
    params["baselines"] = {label: fit["decoding"] for label, fit in final.items() if label != MODEL}
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {params_path}")


def evaluation_variants(path: str | Path) -> dict[str, EKH]:
    """The model, its ablations, and the decodings stored as baselines by ``ekh tune``."""
    base = EKH.load(path)
    d = base.decoding
    variants = {KH99: Decoding.kh99(), MODEL: d, FIRST_LAYER: replace(d, second=None, refinement_rounds=0)}
    if d.refinement_rounds:
        variants[NO_REFINEMENT] = replace(d, refinement_rounds=0)
    variants[HAIRPIN_2] = replace(d, min_hairpin=2)
    baselines = json.loads(Path(path).read_text()).get("baselines", {})
    variants.update({label: Decoding.from_json(v) for label, v in baselines.items()})
    return {label: base.with_decoding(decoding) for label, decoding in variants.items()}


def _predict_all(job: tuple) -> dict:
    """Worker: predictions of every variant for one alignment."""
    params, alignment = job
    variants = evaluation_variants(params)
    started = time.perf_counter()
    evidence = variants[MODEL].evidence(alignment)
    evidence_seconds = time.perf_counter() - started
    out = {}
    for label, model in variants.items():
        started = time.perf_counter()
        structure = model.parse(evidence).structure
        out[label] = {"structure": structure, "seconds": evidence_seconds + time.perf_counter() - started}
    return out


def cmd_evaluate(args) -> None:
    report = {"params": Path(args.params).name, "sets": {}}
    for set_path in args.sets:
        data = load_benchmark(set_path)
        names = sorted(data)
        with ProcessPoolExecutor(args.jobs or min(8, os.cpu_count() or 1)) as pool:
            rows = list(pool.map(_predict_all, [(args.params, data[n]["alignment"]) for n in names]))
        predictions: dict[str, dict] = {label: {} for label in rows[0]}
        for name, row in zip(names, rows):
            for label, prediction in row.items():
                predictions[label][name] = prediction
        external = Path(args.baselines) / f"{Path(set_path).stem}.json" if args.baselines else None
        if external and external.exists():
            predictions.update(json.loads(external.read_text()))

        result = {}
        for label, by_name in predictions.items():
            total, alignments = Counts(), {}
            for name in names:
                counts = compare(by_name[name]["structure"], data[name]["structure"])
                total = total + counts
                alignments[name] = {"predicted": by_name[name]["structure"], "seconds": by_name[name]["seconds"],
                                    "counts": asdict(counts)}
            result[label] = {"pooled": total.summary(),
                             "mean_f1": float(np.mean([compare(by_name[n]["structure"], data[n]["structure"]).f1
                                                       for n in names])),
                             "seconds": float(sum(a["seconds"] for a in alignments.values())),
                             "alignments": alignments}
        report["sets"][Path(set_path).stem] = {
            "references": {n: {"structure": data[n]["structure"], "pseudoknot": bool(pseudoknotted_pairs(pairs_from_dotbracket(data[n]["structure"]))),
                               "sequences": len(data[n]["alignment"]), "columns": len(data[n]["structure"])}
                           for n in names},
            "methods": result}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    _print_summary(report)


def _print_summary(report: dict) -> None:
    for set_name, result in report["sets"].items():
        refs = result["references"]
        print(f"\n== {set_name}: {len(refs)} alignments, {sum(r['pseudoknot'] for r in refs.values())} with pseudoknots")
        print(f"{'method':<44} {'PPV':>6} {'Sens':>6} {'F1':>6} | {'PK PPV':>6} {'PK Sen':>6} {'PK F1':>6} "
              f"| {'det.Se':>6} {'det.Sp':>6} | {'time':>7}")
        for label, v in result["methods"].items():
            p = v["pooled"]
            print(f"{label:<44} {p['ppv']:6.3f} {p['sensitivity']:6.3f} {p['f1']:6.3f} | {p['pk_ppv']:6.3f} "
                  f"{p['pk_sensitivity']:6.3f} {p['pk_f1']:6.3f} | {p['pk_detection_sensitivity']:6.3f} "
                  f"{p['pk_detection_specificity']:6.3f} | {v['seconds']:6.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ekh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict the consensus structure of an alignment")
    p.add_argument("alignment", help="FASTA, Stockholm or PHYLIP alignment")
    p.add_argument("--tree", help="Newick tree (e.g. from PhyML or IQ-TREE); default: built-in ML tree")
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("estimate", help="estimate the substitution model and grammar")
    p.add_argument("--training", default=TRAINING_SET, help="benchmark JSON file of training families")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--output", default=DEFAULT_PARAMS)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("tune", help="tune the decoding settings on held-out families")
    p.add_argument("--sets", nargs="+", default=TUNING_SETS, help="benchmark JSON files to tune on")
    p.add_argument("--method", default="auto", choices=("auto",) + tune.METHODS)
    p.add_argument("--refinement-rounds", type=int, default=0, help="joint layer refinement rounds (fixed)")
    p.add_argument("--jobs", type=int, default=None, help="worker processes (default: up to 8)")
    p.add_argument("--cv-output", default="results/tuning_cv.json")
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.add_argument("--dry-run", action="store_true", help="do not write --params")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("evaluate", help="score the model, its ablations and baselines on benchmark sets")
    p.add_argument("sets", nargs="+")
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.add_argument("--baselines", default="results/baselines", help="directory of external predictions")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--output")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
