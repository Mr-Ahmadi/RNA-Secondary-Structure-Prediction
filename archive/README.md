# Archive: original 2025 notebooks

These are the notebooks the report was first written from, kept unchanged for
reference. They are superseded by the `ekh` package in `src/`.

* `KH-99/` — the baseline KH-99 model on RF03000.
* `EKH-25/` — the Gap-Bracket model: `1. parameters` (frequencies, rates, grammar),
  `2. hyperparams` (genetic algorithm), `3. test`, `4. full_model`.

The notebooks `os.chdir` into an absolute path and call bundled PhyML binaries
(`EKH-25/phyml` for macOS x86-64, `KH-99/phyml.exe` for Windows). They need
`biopython`, `scipy`, `networkx` and `matplotlib`. The training data they use
is also in `data/training/`, and their validation and test sets are in
`data/benchmark/`.
