# Gap-Bracket: two-layer phylogenetic SCFG parsing of RNA consensus structures with pseudoknots

**Author:** Ali Ahmadi Esfidi

`ekh` implements **Gap-Bracket**, which predicts the consensus secondary structure of an
RNA alignment, pseudoknots included. It combines the Knudsen–Hein (KH-99) stochastic
context-free grammar with a phylogenetic model of alignment columns, and parses the
alignment in two layers: the first CYK pass predicts the nested structure, the second
predicts pseudoknotted pairs among the remaining columns.

On a clan-disjoint test set of **246 Rfam alignments** (47 pseudoknotted, literature-derived
structures):

| Method | Pair F1 | Pseudoknotted-pair F1 | PK detection (sens. / spec.) | Median time |
| --- | ---: | ---: | ---: | ---: |
| **Gap-Bracket** | 0.759 | **0.541** | **0.60 / 0.94** | 15 ms |
| RNAalifold 2.7.2 | **0.767** | nested only | – | 11 ms |
| IPknot 1.1.0 (aligned) | 0.757 | 0.113 | 0.32 / 0.81 | 116 ms |
| KH-99 CYK | 0.746 | nested only | – | 13 ms |

Overall pair F1 is statistically indistinguishable from RNAalifold and IPknot; on
pseudoknotted pairs Gap-Bracket leads IPknot by 0.429 (95% CI 0.297 to 0.541).

The [paper](report/main.pdf) describes the model, benchmark, tuning protocol and ablations.
The only runtime dependency is NumPy.

## Install and use

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test]'

.venv/bin/ekh predict alignment.fasta                  # FASTA, Stockholm or PHYLIP
.venv/bin/ekh predict alignment.sto --tree tree.nwk    # optional external tree
```

```python
from ekh import EKH

model = EKH.load()
prediction = model.predict({"seq1": "GGGGAAAACCCC", "seq2": "GCGGAAAACCGC", "seq3": "GGCGAAAACGCC"})
print(prediction.structure)   # ((((....))))
```

## Reproduce

```bash
scripts/fetch_rfam.sh                          # Rfam 15.1 seed alignments and clans (6 MB)
python scripts/build_benchmark.py              # data/benchmark/{training,tuning,test}.json
.venv/bin/ekh estimate                         # substitution model + grammar   (~5 s)
.venv/bin/ekh tune                             # clan-grouped CV tuning          (~17 min)

scripts/setup_baselines.sh                     # RNAalifold + IPknot via conda   (~3 min)
python scripts/run_baselines.py                # baseline predictions            (~2 min)

.venv/bin/ekh evaluate data/benchmark/tuning.json data/benchmark/test.json --output results/evaluation.json
python scripts/summarise.py                    # CIs and figures (pip install -e '.[figures]')
.venv/bin/pytest
```

## Repository layout

```
src/ekh/       the package; src/ekh/data/parameters.json holds the fitted model
data/          benchmark sets, downloaded Rfam files, legacy 2025 data
results/       evaluations, baseline predictions, tuning cross-validation
report/        paper (LaTeX)          presentation/  slides
scripts/       data, benchmark, baselines, summary
tests/         unit tests, including exhaustive-search checks
archive/       original 2025 notebooks
```

See [CHANGELOG.md](CHANGELOG.md) for what changed from the original notebooks.

## References

* Knudsen & Hein (1999, 2003). *RNA secondary structure prediction using stochastic context-free grammars and evolutionary history*; *Pfold*.
* Felsenstein (1981). *Evolutionary trees from DNA sequences: a maximum likelihood approach*.
* Kato et al. (2006). *Stochastic multiple context-free grammar for RNA pseudoknot modeling*.

## Contact

Ali Ahmadi Esfidi · [aliahmadiesfidi@outlook.com](mailto:aliahmadiesfidi@outlook.com) · [GitHub](https://github.com/Mr-Ahmadi) · [LinkedIn](https://linkedin.com/in/ali-ahmadi-esfidi-6a0848375)
