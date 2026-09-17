# Gap-Bracket: two-layer phylogenetic SCFG parsing of RNA consensus structures with pseudoknots

**Author:** Ali Ahmadi Esfidi

`ekh` implements **Gap-Bracket**, which predicts the consensus secondary structure of an
RNA alignment, pseudoknots included. It combines the Knudsen–Hein (KH-99) stochastic
context-free grammar with a phylogenetic model of alignment columns, and parses the
alignment into two nested layers: the first CYK pass predicts the nested structure, the
second predicts pseudoknotted pairs among the remaining columns at a tuned cost per pair.

On a clan-disjoint test set of **246 Rfam alignments** (47 pseudoknotted, literature-derived
structures):

| Method | Pair F1 | Pseudoknotted-pair F1 | PK detection (sens. / spec.) | Median time |
| --- | ---: | ---: | ---: | ---: |
| **Gap-Bracket** | 0.759 | **0.541** | **0.60 / 0.94** | 15 ms |
| RNAalifold 2.7.2 | **0.767** | nested only | – | 11 ms |
| IPknot 1.1.0 (aligned) | 0.757 | 0.113 | 0.32 / 0.81 | 116 ms |
| KH-99 CYK | 0.746 | nested only | – | 13 ms |

* **Overall pair F1:** differences to RNAalifold (−0.008, 95% CI −0.034 to +0.015) and
  IPknot (+0.002, CI −0.019 to +0.022) are not significant.
* **Pseudoknotted pairs:** Gap-Bracket is 0.429 ahead of IPknot (CI 0.297 to 0.541).

The full write-up is the [paper](report/main.pdf). The only runtime dependency is NumPy.

## Model

| Symbol | Setting | Value |
| --- | --- | ---: |
| ρ_init | helix initiation ratio (both layers) | 0.16 |
| ρ_ext | helix extension ratio (both layers) | 1.59 |
| ρ_cross | crossing ratio | 1 |
| ρ_pk | pseudoknot pair ratio | 0.56 |
| β | evidence weight | 0.8 |
| h_min | minimum hairpin loop | 3 |

1. **Column evidence.** Felsenstein pruning on a neighbour-joining tree with
   maximum-likelihood branch lengths gives `log P(column i)` (4-state model) and
   `log P(columns i, j pair)` (16-state model), once per distinct column pattern.
   Gaps are unknown nucleotides. `--tree` accepts an external Newick tree.
2. **Grammar.** KH-99 in Chomsky normal form, with rule probabilities from rule counts in
   the training structures:

   ```
   S -> L S | $M E | s      L -> $M E | s      F -> $M E | L S
   $M -> B F                B -> d             E -> d
   ```

3. **Two-layer CYK.**
   * A pair closing a loop is weighted by ρ_init, and a pair stacked on another pair by ρ_ext.
   * **Layer 1** parses all columns.
   * **Layer 2** parses the columns layer 1 leaves free. It weights each pair by ρ_pk, and by
     ρ_cross once for each layer-1 pair it crosses.
   * All column log-likelihoods are multiplied by β, and pairs must enclose at least
     h_min = 3 columns.
4. **Output.** Pairs are drawn with the largest nested subset as `()`, then `[]`, `{}`
   (exact interval DP). This affects the drawing only.

An optional joint refinement (`refinement_rounds`) re-parses each layer given the other and
keeps a round only if the joint score rises. Cross-validation showed no gain, so it is off.
All dynamic programs are tested against exhaustive enumeration.

## Benchmark

`scripts/build_benchmark.py` builds `data/benchmark/{training,tuning,test}.json` from
Rfam 15.1 seed alignments:

* **Families:** only the 769 whose structure is *published* or from *PseudoBase*. Predicted
  structures are excluded, because they would favour the tool that made them.
* **Split:** by **clan**, stratified by pseudoknot presence, 50 / 15 / 35 % with a fixed seed.
* **Alignments:** training families keep ≤ 100 sequences. Tuning and test families give one
  alignment of 4–12 random sequences. Columns where at most half of the sequences have a
  base are removed.

| Set | Alignments | Pseudoknotted | Clans | Sequences | Columns |
| --- | ---: | ---: | ---: | ---: | ---: |
| training | 337 | 66 | 316 | 4–100 | 23–1208 |
| tuning | 100 | 22 | 94 | 4–12 | 19–307 |
| test | 246 | 47 | 216 | 4–12 | 20–383 |

**Metrics** (`ekh.metrics`):
* **All base pairs:** PPV, sensitivity and F1.
* **Pseudoknotted pairs:** the same three, on pairs that cross another pair of their own structure.
* **Pseudoknot detection:** per-alignment sensitivity and specificity.

`scripts/summarise.py` adds bootstrap CIs over alignments.

## Tuning

`ekh tune` compares three tuning methods by **10-fold cross-validation grouped by clan** on the
tuning set, then fits the best one (ties go to fewer settings) on the whole set.

| Tuning method | Held-out F1 | + joint refinement | Distinct picks over folds |
| --- | ---: | ---: | ---: |
| none (all settings 1) | 0.728 | 0.731 | – |
| separate ratios, coordinate ascent | 0.745 | 0.759 | 6 |
| separate ratios, genetic algorithm | 0.755 | 0.754 | 9 |
| **shared ratios, exhaustive search** | **0.755** | 0.755 | **5** |
| shared ratios, β fixed at 1 | 0.753 | – | 4 |

For reference, RNAalifold scores 0.758 and IPknot 0.757 on the same alignments.
*Shared ratios* means both layers share ρ_init and ρ_ext and ρ_cross = 1. The full
17 × 17 × 9 × 9 grid is scored, so the optimum is global. The protocol scores 31,830
candidates in about 17 minutes, with cached counts and parallel workers.

## Ablations (test set)

| Variant | Pair F1 | Δ | PK-pair F1 | Δ |
| --- | ---: | ---: | ---: | ---: |
| **Gap-Bracket** | 0.759 | | 0.541 | |
| first layer only | 0.747 | −0.012* | – | |
| h_min = 2 | 0.756 | −0.003* | 0.528 | −0.013 |
| β = 1 (retuned) | 0.764 | +0.005 | 0.506 | −0.036 |
| separate ratios, coordinate ascent | 0.758 | −0.001 | 0.538 | −0.003 |
| separate ratios, genetic algorithm | 0.760 | +0.001 | 0.549 | +0.008 |

\* 95% paired bootstrap interval excludes zero.

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

## Reproduce everything

```bash
scripts/fetch_rfam.sh                          # Rfam 15.1 seed alignments and clans (6 MB)
python scripts/build_benchmark.py              # data/benchmark/{training,tuning,test}.json
.venv/bin/ekh estimate                         # substitution model + grammar from training   (~5 s)
.venv/bin/ekh tune                             # CV-selected tuning; results/tuning_cv.json  (~17 min)
.venv/bin/ekh tune --refinement-rounds 3 --dry-run --cv-output results/tuning_cv_refinement.json

scripts/setup_baselines.sh                     # RNAalifold 2.7.2 + IPknot 1.1.0 in .bench-env (~3 min; conda)
python scripts/run_baselines.py                # results/baselines/{tuning,test}.json (~2 min)

.venv/bin/ekh evaluate data/benchmark/tuning.json data/benchmark/test.json --output results/evaluation.json
python scripts/summarise.py                    # CIs, subsets, timings, paper figures (pip install -e '.[figures]')
.venv/bin/pytest
```

## Repository layout

```
src/ekh/            package: alignment, tree, evolution, grammar, parser, model, tune, metrics, estimate, structure, cli
src/ekh/data/       parameters.json (substitution model, grammar, decoding, tuned baselines)
data/benchmark/     training, tuning and test sets (Rfam 15.1, clan-disjoint)
data/rfam/          downloaded Rfam files (scripts/fetch_rfam.sh)
data/legacy/        the 2025 training families and 8-family validation/test sets
results/            evaluations, baseline predictions, tuning cross-validation, summary
report/             paper (LaTeX)
tests/              unit tests, including exhaustive-search checks
scripts/            data download, benchmark builder, baseline setup and runner, result summary
archive/            original 2025 notebooks (KH-99 and EKH-25), kept for reference
literature/         papers the method builds on
presentation/       slides
```

## What changed in v2.0

* **Benchmark.** A clan-disjoint Rfam 15.1 benchmark of 683 alignments with literature-derived
  structures replaces the 8-family validation/test sets. Pseudoknot-specific metrics are added,
  and RNAalifold and IPknot serve as external baselines.
* **Training.** The substitution model and grammar are estimated from 337 families instead of 3,
  with two-stage tree estimation and identity computed over non-gap columns.
* **Model.**
  * A pseudoknot pair ratio ρ_pk for the second layer.
  * An optional joint refinement (evaluated, not adopted).
  * Exact largest-nested-subset handling throughout.
* **Tuning.** 10-fold cross-validation grouped by clan, with the pseudoknot pair ratio in the grid.
* **Removed.** The legacy weighted score and the old 3-family estimation path. The 2025 data now
  lives in `data/legacy/`.

## What changed from the original notebooks (v1.0)

* The notebooks are now a tested package. Nothing hard-codes absolute paths,
  and no grammar files are written to disk for each prediction.
* **Speed.** The generic dictionary-based CYK over a grammar with one rule per
  column pattern became vectorised recursions over column indices. Column
  likelihoods are array operations instead of a symbolic enumeration of column
  patterns. Result: 50× to over 10,000× faster; the largest alignments gain the most.
* **Correctness fixes:**
  * The stacking ratio used to depend on which derivation of `F` happened to
    score best. Pass 1 and pass 2 now split `F` into its stacked and loop
    derivations, so the ratio is applied exactly. The parser is checked
    against exhaustive enumeration.
  * The crossing penalty used to count only up to the column before the
    closing base. It now counts over the whole span of the pair.
  * Grammar probabilities are exact rule frequencies from the training
    structures. Before, they came from inside–outside on `d/s` strings, which
    loses which bases pair with which.
  * Ambiguous IUPAC symbols are now handled the same way in frequencies and
    rates.
  * The evaluation used to skip unmatched brackets silently. It now rejects
    them. One validation reference (RF01722) had an unmatched `(`, which is now
    written as `.`, matching how it was scored before.
* **Reproduction check.** The re-estimated substitution rates match the
  original pickled parameters to 1e-4 relative. With PhyML trees and the
  original ratios, the new code comes close to the 2025 report (weighted 0.817 vs
  0.824; four of the eight families agree within 0.1 points; see
  [`results/evaluation_phyml_trees.json`](results/evaluation_phyml_trees.json)).

## References

* Knudsen & Hein (1999, 2003). *RNA secondary structure prediction using stochastic context-free grammars and evolutionary history*; *Pfold*.
* Sükösd et al. (2012). *PPfold 3.0*.
* Felsenstein (1981). *Evolutionary trees from DNA sequences: a maximum likelihood approach*.
* Kato et al. (2006). *Stochastic multiple context-free grammar for RNA pseudoknot modeling*.
* Andrikos et al. (2023). *Knotify*.
* Yi et al. (2011). *Efficient parallel CKY parsing on GPUs*.

## Contact

Ali Ahmadi Esfidi · [aliahmadiesfidi@outlook.com](mailto:aliahmadiesfidi@outlook.com) · [GitHub](https://github.com/Mr-Ahmadi) · [LinkedIn](https://linkedin.com/in/ali-ahmadi-esfidi-6a0848375)
