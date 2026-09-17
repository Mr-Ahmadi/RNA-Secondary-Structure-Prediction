import json

import numpy as np
import pytest

from ekh.alignment import tip_likelihoods
from ekh.cli import main
from ekh.evolution import Transition, _prune, expm, optimise_branch_lengths
from ekh.model import EKH, Decoding
from ekh.parser import PassSettings, cyk
from ekh.structure import pairs_from_dotbracket
from ekh.tree import midpoint_root, neighbour_joining, parse_newick, p_distance_matrix

HAIRPINS = {
    "a": "GGGGAAAACCCCUU", "b": "GCGGAAAACCGCUA", "c": "GGCGAAAACGCCAU", "d": "CGGGAAAACCCGUU",
}


def test_wuss_pseudoknot_letters():
    assert pairs_from_dotbracket("<A.>a") == [(0, 3), (1, 4)]
    with pytest.raises(ValueError):
        pairs_from_dotbracket("A..")


def test_deep_newick_tree_needs_no_recursion():
    depth = 5000
    text = "(" * depth + "x0:1" + "".join(f",x{k}:1):1" for k in range(1, depth + 1)) + ";"
    tree = parse_newick(text)
    assert len(tree.leaves()) == depth + 1
    rooted = midpoint_root(tree)
    assert len(rooted.leaves()) == depth + 1


def test_newick_lengths_and_single_leaf():
    tree = parse_newick("((a:1,b:2)90:0.5,c:3);")
    assert {l.name: l.length for l in tree.leaves()} == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert parse_newick("(x:0.0);").children[0].name == "x"


def test_transition_matches_expm():
    rates = EKH.load().evolution.pair_rates
    transition = Transition(rates)
    for t in (1e-6, 0.1, 2.5):
        assert np.allclose(transition(t), expm(rates * t), atol=1e-10)


def test_branch_optimisation_increases_likelihood():
    model = EKH.load().evolution
    seqs = ["GGGGAAAACCCCUUAG", "GCGGAAUACCGCUAAG", "GGCGAAAACGCCAUGG", "CGGGAGAACCCGUUAA", "CGGGAGAACCCGUUCA"]
    alignment = dict(zip("abcde", seqs))
    tree = neighbour_joining(list(alignment), p_distance_matrix(seqs))
    for node in tree.postorder():
        node.length = 0.5
    tips = tip_likelihoods(seqs)
    tip_of = dict(zip("abcde", tips))

    def log_like():
        return _prune(tree, lambda leaf: tip_of[leaf.name], model.single_rates, model.single_freq).sum()

    before = log_like()
    optimise_branch_lengths(tree, alignment, model.single_rates, model.single_freq)
    assert log_like() > before + 1.0


def test_evidence_weight_and_min_hairpin_reach_the_parser():
    base = EKH.load()
    evidence = base.evidence(HAIRPINS)
    decoding = Decoding(PassSettings(0.5, 2.0), None, 0.6, 3)
    positions = np.arange(len(evidence.single))
    expected = cyk(0.6 * evidence.single, 0.6 * evidence.pair, base.grammar, decoding.first,
                   allowed=positions[None, :] - positions[:, None] > 3)[1]
    assert base.with_decoding(decoding).parse(evidence).nested == expected
    assert Decoding.from_json(decoding.to_json()) == decoding


def test_pattern_compression_matches_per_column_likelihoods():
    model = EKH.load()
    alignment = {"a": "GGGAAACCCGG", "b": "GGGAAACCCGG", "c": "GCGAUACGCGG"}  # repeated columns
    tree = model.estimate_tree(alignment)
    single, pair = model.evolution.column_log_likelihoods(alignment, tree)
    for k in range(len(single)):
        column = {name: seq[k] for name, seq in alignment.items()}
        s, _ = model.evolution.column_log_likelihoods(column, tree)
        assert single[k] == pytest.approx(s[0])
    i, j = 0, 8
    two = {name: seq[i] + seq[j] for name, seq in alignment.items()}
    assert pair[i, j] == pytest.approx(model.evolution.column_log_likelihoods(two, tree)[1][0, 1])


def _toy_evaluator(model):
    from ekh.tune import Evaluator, Item

    evidence = model.evidence(HAIRPINS)
    references = ["((((....))))..", "(((......)))..", ".............."]
    items = [Item(s, f"fam{k}", f"fam{k}", evidence, frozenset(pairs_from_dotbracket(ref)))
             for s in ("a", "b") for k, ref in enumerate(references)]
    return Evaluator(model, items, jobs=1)


def test_tuning_methods_never_lose_to_neutral():
    from ekh.tune import METHODS, NEUTRAL, fit, shared_grid, to_decoding, to_point

    with _toy_evaluator(EKH.load()) as evaluator:
        everything = evaluator.everything
        neutral = evaluator.objective([NEUTRAL], everything)[0]
        for method in METHODS:
            point = fit(method, evaluator, everything, fix_weight=True)
            assert evaluator.objective([point], everything)[0] >= neutral
            assert to_decoding(point).evidence_weight == 1.0
            assert to_point(to_decoding(point)) == point
        best = fit("shared-exhaustive", evaluator, everything)
        grid = shared_grid()
        assert evaluator.objective([best], everything)[0] == evaluator.objective(grid, everything).max()


def test_cross_validation_leaves_out_whole_families():
    from ekh.tune import cross_validate

    with _toy_evaluator(EKH.load()) as evaluator:
        result = cross_validate(evaluator, "shared-exhaustive", log=lambda *_: None)
    assert len(result["fold_decodings"]) == 3  # one fold per group (fam0, fam1, fam2)
    assert 0.0 <= result["objective"] <= 1.0


def test_cli_predict_and_evaluate(tmp_path, capsys):
    fasta = tmp_path / "aln.fasta"
    fasta.write_text("".join(f">{k}\n{v}\n" for k, v in HAIRPINS.items()))
    main(["predict", str(fasta)])
    assert capsys.readouterr().out.strip().splitlines()[-1].split()[-1] == "((((....)))).."
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"toy": {"alignment": HAIRPINS, "structure": "((((....)))).."}}))
    out = tmp_path / "eval.json"
    main(["evaluate", str(bench), "--output", str(out), "--jobs", "1", "--baselines", str(tmp_path / "no-baselines")])
    report = json.loads(out.read_text())
    assert report["sets"]["bench"]["methods"]["Gap-Bracket"]["pooled"]["f1"] == 1.0




def test_split_layers_keeps_largest_nested_layer():
    from itertools import combinations

    from ekh.structure import is_nested as _crossing_free, largest_nested as _largest_nested, split_layers
    from ekh.structure import dotbracket

    # a helix continued by the second pass, and two crossing helices drawn the wrong way round
    first = [(2, 20), (3, 19), (28, 38), (29, 37)]
    second = [(0, 22), (1, 21), (24, 34), (25, 33), (26, 32)]
    assert split_layers(first + second) == [
        [(0, 22), (1, 21), (2, 20), (3, 19), (24, 34), (25, 33), (26, 32)], [(28, 38), (29, 37)]]

    rng = np.random.default_rng(0)
    for _ in range(40):
        cols = rng.permutation(16)[:12]
        ps = [tuple(sorted(map(int, cols[k:k + 2]))) for k in range(0, 12, 2)]
        largest = _largest_nested(ps)
        best = max(len(s) for r in range(len(ps) + 1) for s in combinations(ps, r) if _crossing_free(list(s)))
        assert _crossing_free(largest) and len(largest) == best
        # the union of two nested passes always renders to a string that parses back to the same pairs
        other = _largest_nested(sorted(set(ps) - set(largest)))
        layers = split_layers(largest + other)
        assert all(_crossing_free(layer) for layer in layers)
        assert sorted(pairs_from_dotbracket(dotbracket(16, *layers))) == sorted(largest + other)


def test_pseudoknot_metrics():
    from ekh.metrics import compare
    from ekh.structure import pseudoknotted_pairs

    assert pseudoknotted_pairs(pairs_from_dotbracket("((..[[..))..]]")) == [(0, 9), (1, 8), (4, 13), (5, 12)]
    assert pseudoknotted_pairs(pairs_from_dotbracket("((..((..))..))")) == []
    counts = compare("((..[[..))..]]", "((......))....")
    assert (counts.tp, counts.fp, counts.fn) == (2, 2, 0)
    assert (counts.pk_tp, counts.pk_fp, counts.pk_fn) == (0, 4, 0)  # reference has no pseudoknot
    assert (counts.reference_pk, counts.detected_pk, counts.false_pk) == (0, 0, 1)
    assert compare("((..[[..))..]]", "((..[[..))..]]").pk_f1 == 1.0


def test_structure_score_matches_cyk_optimum():
    from ekh.grammar import RULES, Grammar
    from ekh.parser import structure_score

    rng = np.random.default_rng(3)
    for _ in range(5):
        probs = {}
        for lhs, rhss in RULES.items():
            probs.update({(lhs, r): v for r, v in zip(rhss, rng.dirichlet(np.ones(len(rhss))))})
        grammar = Grammar(probs)
        n = 30
        single, pair = rng.normal(-1.5, 1, n), rng.normal(-2, 2.5, (n, n))
        settings = PassSettings(float(rng.uniform(0.2, 3)), float(rng.uniform(0.2, 3)), float(rng.uniform(0.3, 2)))
        crossings = rng.integers(0, 3, (n, n)).astype(float)
        score, pairs = cyk(single, pair, grammar, settings, crossings)
        assert structure_score(single, pair, grammar, settings, pairs, crossings) == pytest.approx(score)


def test_refinement_never_lowers_joint_score():
    from dataclasses import replace

    from ekh.model import ColumnEvidence

    base = EKH.load()
    rng = np.random.default_rng(5)
    for _ in range(4):
        n = 40
        pair = rng.normal(-6, 3, (n, n))
        evidence = ColumnEvidence(None, rng.normal(-1.2, 0.3, n), (pair + pair.T) / 2)
        two_pass = base.with_decoding(replace(base.decoding, refinement_rounds=0))
        refined = base.with_decoding(replace(base.decoding, refinement_rounds=5))
        assert refined.joint_score(evidence, *refined.pairs(evidence)) >= \
            two_pass.joint_score(evidence, *two_pass.pairs(evidence)) - 1e-9
