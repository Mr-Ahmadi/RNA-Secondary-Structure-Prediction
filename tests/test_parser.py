import itertools
from collections import Counter

import numpy as np
import pytest

from ekh.grammar import Grammar, RULES, _count_rules
from ekh.parser import PassParams, crossing_counts, cyk


def nested_structures(i, j):
    """All KH-99-derivable nested pair sets on columns i..j (hairpins >= 2)."""
    if i > j:
        yield []
        return
    yield from ([] + rest for rest in nested_structures(i + 1, j))  # i unpaired
    for k in range(i + 3, j + 1):  # i pairs with k
        for inner in nested_structures(i + 1, k - 1):
            for rest in nested_structures(k + 1, j):
                yield [(i, k)] + inner + rest


def brute_force_score(pairs, n, single, pair, grammar, params, crossings):
    counts = Counter()
    _count_rules(pairs, n, counts)
    score = sum(c * grammar.log(*rule) for rule, c in counts.items())
    paired = {k for p in pairs for k in p}
    score += sum(single[k] for k in range(n) if k not in paired)
    partner = dict(pairs)
    for i, j in pairs:
        score += pair[i, j] + crossings[i, j] * np.log(params.flag)
        stacked = partner.get(i + 1) == j - 1
        score += np.log(params.accelerate if stacked else params.start)
    return score


@pytest.mark.parametrize("seed", range(6))
def test_cyk_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    n = 9
    probs = {}
    for lhs, rhss in RULES.items():
        p = rng.dirichlet(np.ones(len(rhss)))
        probs.update({(lhs, r): v for r, v in zip(rhss, p)})
    grammar = Grammar(probs)
    single = rng.normal(-1.5, 1.0, n)
    pair = rng.normal(-2.0, 2.0, (n, n))
    crossings = rng.integers(0, 3, (n, n)).astype(float)
    params = PassParams(start=float(rng.uniform(0.2, 3)), accelerate=float(rng.uniform(0.2, 3)),
                        flag=float(rng.uniform(0.3, 2)))

    best = max(
        nested_structures(0, n - 1),
        key=lambda s: brute_force_score(s, n, single, pair, grammar, params, crossings),
    )
    expected = brute_force_score(best, n, single, pair, grammar, params, crossings)
    score, pairs = cyk(single, pair, grammar, params, crossings)
    assert score == pytest.approx(expected)
    assert brute_force_score(pairs, n, single, pair, grammar, params, crossings) == pytest.approx(expected)


def test_crossing_counts():
    positions = np.array([0, 2, 5, 7])
    out = crossing_counts(positions, [(1, 4), (3, 6)])
    assert out[0, 1] == 1  # (0,2) crosses (1,4)
    assert out[1, 2] == 2  # (2,5) crosses (1,4) and (3,6)
    assert out[0, 3] == 0  # (0,7) encloses both
