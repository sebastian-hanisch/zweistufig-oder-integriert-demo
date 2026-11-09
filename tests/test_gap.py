"""Lücke Δ (zoi_gap): Handbeispiele, Randfälle, Eigenschaften. Die Aufzählung gegen den Löser steht im Orakel-Test."""

import numpy as np
import pytest

import zoi_gap as G
from zoi_grid import PathSet, two_paths


def test_two_paths_example_by_hand():
    """Wahre Kosten c = (3, 5): bester Wert 3, zweitbester 5, Lücke 2, relative Lücke 2/3."""
    W_all = two_paths().W_all
    gap, best = G.gap_to_second_best(W_all, np.array([[3.0, 5.0]]))
    assert gap.tolist() == [2.0] and best.tolist() == [3.0]
    assert G.relative_gap(W_all, np.array([[3.0, 5.0]])).tolist() == [pytest.approx(2.0 / 3.0)]


def test_tie_gives_gap_zero_and_is_not_counted_as_small_by_the_theory_condition():
    W_all = two_paths().W_all
    rel = G.relative_gap(W_all, np.array([[4.0, 4.0], [4.0, 4.1]]))
    assert rel[0] == 0.0 and rel[1] == pytest.approx(0.1 / 4.0)
    assert G.share_leq(rel, 0.05) == 1.0                      # Gleichstand zählt als klein
    assert G.share_leq(rel, 0.05, strict_positive=True) == 0.5   # 0 < Lücke <= 0,05: nur der zweite Fall hat Lücke 0,025
    assert G.share_leq(rel, 0.0) == 0.5 and G.share_leq(rel, 0.0, strict_positive=True) == 0.0


def test_three_paths_by_hand_second_smallest_not_second_in_list():
    W_all = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    gap, best = G.gap_to_second_best(W_all, np.array([[7.0, 2.0, 5.0], [1.0, 9.0, 1.5]]))
    assert gap.tolist() == [3.0, 0.5] and best.tolist() == [2.0, 1.0]


def test_gap_needs_at_least_two_paths_and_all_zero_costs_have_zero_relative_gap():
    with pytest.raises(ValueError):
        G.gap_to_second_best(np.array([[1.0, 0.0]]), np.array([[1.0, 2.0]]))
    assert G.relative_gap(two_paths().W_all, np.zeros((1, 2))).tolist() == [0.0]


def test_gap_is_nonnegative_and_scale_invariant_in_the_relative_version(grid5):
    rng = np.random.default_rng(0)
    M = rng.uniform(1, 4, size=(50, grid5.d))
    W_all = grid5.paths()
    gap, best = G.gap_to_second_best(W_all, M)
    assert np.all(gap >= 0) and np.all(best > 0)
    assert np.allclose(G.relative_gap(W_all, 3.0 * M), G.relative_gap(W_all, M))        # relative Lücke ist skaleninvariant
    assert np.allclose(G.gap_to_second_best(W_all, 3.0 * M)[0], 3.0 * gap)              # die absolute wächst mit dem Faktor


def test_mean_regret_per_case_is_zero_for_the_oracle_forecast_and_positive_for_a_wrong_choice(grid5):
    rng = np.random.default_rng(1)
    M = rng.uniform(1, 4, size=(30, grid5.d))
    W_all = grid5.paths()
    assert np.allclose(G.mean_regret_per_case(W_all, M, M, grid5), 0.0, atol=1e-12)
    wrong = G.mean_regret_per_case(W_all, M, -M, grid5)                # die Vorhersage -M wählt den teuersten Weg
    assert np.all(wrong >= -1e-12) and wrong.min() >= 0 and wrong.max() > 0


def test_mean_regret_with_a_path_set_solver_by_hand():
    """Wege (1, 0) und (0, 1), mittlere Kosten (3, 5), Vorhersage (4, 3) wählt Weg 2: Regret 5 - 3 = 2."""
    solver = two_paths()
    r = G.mean_regret_per_case(solver.W_all, np.array([[3.0, 5.0]]), np.array([[4.0, 3.0]]), solver)
    assert r.tolist() == [2.0]
    assert isinstance(solver, PathSet)
