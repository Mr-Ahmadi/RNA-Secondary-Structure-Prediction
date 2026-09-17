import numpy as np
import pytest

from ekh.alignment import tip_likelihoods
from ekh.evolution import expm
from ekh.grammar import Grammar
from ekh.metrics import compare
from ekh.model import EKH
from ekh.structure import dotbracket, largest_nested, pairs_from_dotbracket
from ekh.tree import midpoint_root, neighbour_joining, parse_newick, p_distance_matrix


def test_dotbracket_roundtrip_with_pseudoknot():
    s = "((..[[..))..]]"
    pairs = pairs_from_dotbracket(s)
    nested = largest_nested(pairs)
    assert nested == [(0, 9), (1, 8)]
    assert dotbracket(len(s), nested, sorted(set(pairs) - set(nested))) == s
    with pytest.raises(ValueError):
        pairs_from_dotbracket("(()")


def test_expm_against_eigendecomposition():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1, (16, 16))
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(1))
    w, v = np.linalg.eig(q * 3.0)
    assert np.allclose(expm(q * 3.0), (v * np.exp(w)) @ np.linalg.inv(v), atol=1e-9)


def test_grammar_counts_single_hairpin():
    g = Grammar.estimate([[(0, 5), (1, 4)]], [7])
    # ((..)). : top S -> L S, then S -> s at the end and inside the hairpin
    assert g.prob["F", "$M E"] == pytest.approx(0.5)
    assert g.prob["S", "L S"] == pytest.approx(1 / 3)
    assert g.prob["S", "s"] == pytest.approx(2 / 3)


def test_tip_likelihoods_gap_and_iupac():
    tips = tip_likelihoods(["AR-", "UTN"])
    assert tips[0, 0].tolist() == [1, 0, 0, 0]
    assert tips[0, 1].tolist() == [1, 0, 1, 0]
    assert tips[0, 2].tolist() == [1, 1, 1, 1]
    assert tips[1, 1].tolist() == [0, 0, 0, 1]


def test_tree_tools():
    tree = parse_newick("((a:1,b:2):0.5,c:3);")
    assert sorted(l.name for l in tree.leaves()) == ["a", "b", "c"]
    seqs = ["ACGUACGUAC", "ACGUACGUAA", "ACGAACGUAA", "UCGAACCUAA"]
    nj = midpoint_root(neighbour_joining(list("abcd"), p_distance_matrix(seqs)))
    assert len(nj.children) == 2 and sorted(l.name for l in nj.leaves()) == list("abcd")


def test_pair_likelihood_matches_single_leaf_frequency():
    model = EKH.load().evolution
    tree = parse_newick("(x:0.0);")
    single, pair = model.column_log_likelihoods({"x": "GC"}, tree)
    assert np.exp(single[0]) == pytest.approx(model.single_freq[2])
    assert np.exp(pair[0, 1]) == pytest.approx(model.pair_freq[4 * 2 + 1])


def test_predicts_obvious_hairpin():
    seqs = {
        "a": "GGGGAAAACCCC", "b": "GCGGAAAACCGC", "c": "GGCGAAAACGCC", "d": "CGGGAAAACCCG",
    }
    prediction = EKH.load().predict(seqs)
    assert compare(prediction.structure, "((((....))))").f1 == 1.0
