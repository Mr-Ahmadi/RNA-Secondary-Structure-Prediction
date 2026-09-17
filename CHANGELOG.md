# Changelog

## v2.0

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

## v1.0: from the original notebooks

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
