"""Bootstrap confidence intervals and figure for results/evaluation.json.

Families are resampled with replacement and pooled base-pair F1 is recomputed,
so the intervals reflect how few families the benchmark contains.
Requires matplotlib for the figure.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ekh.metrics import compare  # noqa: E402

BASELINE = "KH-99 CYK (1 pass, no ratios)"
MODEL = "Gap-Bracket, 2 passes"


def counts(variant):
    names = sorted(variant["families"])
    rows = [compare(variant["families"][n]["predicted"], variant["families"][n]["reference"]) for n in names]
    return names, np.array([[c.tp, c.fp, c.fn] for c in rows], dtype=float)


def pooled_f1(tpfpfn):
    tp, fp, fn = tpfpfn.sum(-2).T if tpfpfn.ndim == 3 else tpfpfn.sum(0)
    return 2 * tp / (2 * tp + fp + fn)


def main():
    report = json.loads((ROOT / "results" / "evaluation.json").read_text())
    rng = np.random.default_rng(0)
    summary = {}
    for set_name, result in report["sets"].items():
        variants = result["variants"]
        _, model = counts(variants[MODEL])
        idx = rng.integers(0, len(model), (10_000, len(model)))
        summary[set_name] = {}
        for label, variant in variants.items():
            _, c = counts(variant)
            f1 = pooled_f1(c[idx])
            entry = {"f1": float(pooled_f1(c)), "ci95": np.percentile(f1, [2.5, 97.5]).round(4).tolist()}
            if label != MODEL:
                diff = pooled_f1(model[idx]) - f1
                entry[f"delta_vs_{MODEL}"] = {
                    "mean": float(pooled_f1(model) - pooled_f1(c)),
                    "ci95": np.percentile(diff, [2.5, 97.5]).round(4).tolist(),
                }
            summary[set_name][label] = entry
    out = ROOT / "results" / "bootstrap.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    plot(report)


def plot(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [BASELINE, "Gap-Bracket, 1st pass only", MODEL]
    colors = ["#9aa5b1", "#5b8def", "#1f3a93"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [8, 8]}, sharey=True)
    for ax, (set_name, result) in zip(axes, report["sets"].items()):
        families = list(result["variants"][MODEL]["families"])
        x = np.arange(len(families))
        for k, label in enumerate(labels):
            f1 = [result["variants"][label]["families"][f]["f1"] for f in families]
            ax.bar(x + (k - 1) * 0.27, f1, 0.27, label=label.replace(" (1 pass, no ratios)", ""), color=colors[k])
        ax.set_xticks(x, families, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{set_name} set", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("base-pair F1")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "report" / "images" / "family_f1.png", dpi=250)


if __name__ == "__main__":
    main()
