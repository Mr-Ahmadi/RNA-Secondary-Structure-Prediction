"""Confidence intervals, subset scores, running times and the paper figures.

Reads ``results/evaluation.json`` (written by ``ekh evaluate``). Alignments are
resampled with replacement (10,000 bootstrap samples) and pooled measures are
recomputed, giving 95% intervals and paired intervals for the difference to
Gap-Bracket. Writes ``results/summary.json`` and, with matplotlib installed,
``report/images/accuracy.png`` and ``report/images/runtime.png``.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ekh.metrics import Counts  # noqa: E402

MODEL = "Gap-Bracket"
FIELDS = list(Counts.__dataclass_fields__)
MEASURES = {"f1": ("tp", "fp", "fn"), "pk_f1": ("pk_tp", "pk_fp", "pk_fn")}
FIGURE_METHODS = {MODEL: "#2a78d6", "RNAalifold": "#eb6834", "IPknot": "#1baf7a", "KH-99 CYK": "#8c8b86"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
NESTED_ONLY = {"RNAalifold", "KH-99 CYK"}  # cannot predict pseudoknots


def count_matrix(method: dict, names: list[str]) -> np.ndarray:
    return np.array([[method["alignments"][n]["counts"][f] for f in FIELDS] for n in names], dtype=float)


def f1_of(totals: np.ndarray, measure: str) -> np.ndarray:
    tp, fp, fn = (totals[..., FIELDS.index(f)] for f in MEASURES[measure])
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def summarise_set(result: dict, rng: np.random.Generator) -> dict:
    names = sorted(result["references"])
    pk = np.array([result["references"][n]["pseudoknot"] for n in names])
    samples = rng.integers(0, len(names), (10_000, len(names)))
    counts = {label: count_matrix(m, names) for label, m in result["methods"].items()}
    model_boot = {k: f1_of(counts[MODEL][samples].sum(1), k) for k in MEASURES}
    out = {}
    for label, c in counts.items():
        boot = {k: f1_of(c[samples].sum(1), k) for k in MEASURES}
        entry = {"pooled": Counts(*c.sum(0).astype(int)).summary()}
        for k in MEASURES:
            entry[f"{k}_ci95"] = np.percentile(boot[k], [2.5, 97.5]).round(3).tolist()
            if label != MODEL:
                entry[f"{k}_difference_to_model"] = {
                    "mean": round(float(f1_of(counts[MODEL].sum(0), k) - f1_of(c.sum(0), k)), 4),
                    "ci95": np.percentile(model_boot[k] - boot[k], [2.5, 97.5]).round(3).tolist()}
        entry["f1_pseudoknotted_alignments"] = float(f1_of(c[pk].sum(0), "f1"))
        entry["f1_nested_alignments"] = float(f1_of(c[~pk].sum(0), "f1"))
        seconds = np.array([result["methods"][label]["alignments"][n]["seconds"] for n in names])
        entry["seconds"] = {"total": round(float(seconds.sum()), 2), "median_ms": round(1000 * float(np.median(seconds)), 1),
                            "max_ms": round(1000 * float(seconds.max()), 1)}
        out[label] = entry
    return out


def plot(report: dict, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    test = summary["test"]
    methods = [m for m in FIGURE_METHODS if m in test]

    # accuracy: pair F1 and pseudoknot-pair F1 with 95% bootstrap intervals
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.9), sharey=True)
    for ax, (measure, title) in zip(axes, [("f1", "All base pairs: F1"), ("pk_f1", "Pseudoknotted pairs: F1")]):
        y = np.arange(len(methods))[::-1]
        for yi, m in zip(y, methods):
            value = test[m]["pooled"][measure]
            lo, hi = test[m][f"{measure}_ci95"]
            if measure == "pk_f1" and m in NESTED_ONLY:
                ax.text(0.015, yi, "nested structures only", va="center", color=MUTED, style="italic")
                continue
            ax.barh(yi, value, height=0.62, color=FIGURE_METHODS[m], edgecolor="white", linewidth=2)
            ax.plot([lo, hi], [yi, yi], color=INK, linewidth=1)
            ax.text(max(hi, value) + 0.015, yi, f"{value:.3f}", va="center", color=INK)
        ax.set_yticks(y, methods)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, loc="left", color=INK, fontsize=8.5)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "report/images/accuracy.png", dpi=300)

    # running time against alignment length
    refs = report["sets"]["test"]["references"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in [x for x in methods if x != "KH-99 CYK"]:
        alignments = report["sets"]["test"]["methods"][m]["alignments"]
        cols = [refs[n]["columns"] for n in alignments]
        secs = [1000 * a["seconds"] for a in alignments.values()]
        ax.scatter(cols, secs, s=9, color=FIGURE_METHODS[m], edgecolors="white", linewidths=0.4, label=m)
    ax.set_yscale("log")
    ax.set_xlabel("alignment columns")
    ax.set_ylabel("time per alignment (ms)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "report/images/runtime.png", dpi=300)


def main() -> None:
    report = json.loads((ROOT / "results/evaluation.json").read_text())
    rng = np.random.default_rng(0)
    summary = {name: summarise_set(result, rng) for name, result in report["sets"].items()}
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, methods in summary.items():
        print(f"== {name}")
        for label, e in methods.items():
            p = e["pooled"]
            diff = e.get("f1_difference_to_model", {})
            print(f"{label:<48} F1 {p['f1']:.3f} {e['f1_ci95']}  PK F1 {p['pk_f1']:.3f} {e['pk_f1_ci95']}  "
                  f"model-minus {diff.get('mean', '')} {diff.get('ci95', '')}  median {e['seconds']['median_ms']} ms")
    try:
        plot(report, summary)
    except ImportError:
        print("matplotlib not installed: figures skipped")


if __name__ == "__main__":
    main()
