"""Run the external baselines on benchmark sets and store their predictions.

    RNAalifold  ViennaRNA 2.7.2, consensus minimum free energy (nested only)
    IPknot      1.1.0, aligned mode: averaged McCaskill + RNAalifold pair
                probabilities (``-e McCaskill -e Alifold``), pseudoknots allowed

The tools come from the isolated environment ``.bench-env`` (see README).
Output: ``results/baselines/<set>.json`` with, per tool, the predicted
dot-bracket string and wall-clock seconds for every alignment.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".bench-env"
IPKNOT = ENV / "src/ipknot/build/ipknot"
RNAALIFOLD = ENV / "bin/RNAalifold"


def clustal(alignment: dict[str, str]) -> str:
    rows = [f"seq{k}  {seq}" for k, seq in enumerate(alignment.values())]
    return "CLUSTAL W\n\n" + "\n".join(rows) + "\n"


def run(command: list[str], text: str) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".aln", delete=False) as handle:
        handle.write(text)
    started = time.perf_counter()
    try:
        out = subprocess.run(command + [handle.name], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(handle.name)
    return out, time.perf_counter() - started


def rnaalifold(alignment: dict[str, str]) -> tuple[str, float]:
    out, seconds = run([str(RNAALIFOLD), "--noPS"], clustal(alignment))
    structure = out.splitlines()[1].split()[0]
    return structure, seconds


def ipknot(alignment: dict[str, str]) -> tuple[str, float]:
    out, seconds = run([str(IPKNOT), "-e", "McCaskill", "-e", "Alifold"], clustal(alignment))
    lines = [line for line in out.splitlines() if line and not line.startswith(">")]
    structure = lines[-1]
    return structure, seconds


TOOLS = {"RNAalifold": rnaalifold, "IPknot": ipknot}


def main(sets: list[str]) -> None:
    out_dir = ROOT / "results/baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_path in sets:
        data = json.loads(Path(set_path).read_text())
        result = {}
        for tool, predict in TOOLS.items():
            with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
                predictions = dict(zip(data, pool.map(lambda fam: predict(fam["alignment"]), data.values())))
            for name, (structure, _) in predictions.items():
                length = len(data[name]["structure"])
                if len(structure) != length or not re.fullmatch(r"[.()\[\]{}<>]+", structure):
                    raise ValueError(f"{tool} returned an invalid structure for {name}: {structure!r}")
            result[tool] = {name: {"structure": s, "seconds": t} for name, (s, t) in predictions.items()}
            print(f"{Path(set_path).stem}: {tool} done ({sum(t for _, t in predictions.values()):.1f} s total)")
        (out_dir / f"{Path(set_path).stem}.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/benchmark/tuning.json", "data/benchmark/test.json"])
