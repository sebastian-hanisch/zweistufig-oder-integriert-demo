"""Newsvendor (zoi_news): Kosten von Hand, Quantile, Gewichte, Welten, Reproduzierbarkeit. Unabhängige Rechenwege (Integration, Aufzählung) stehen im Orakel-Test."""

import numpy as np
import pytest

import zoi_news as N


def test_critical_costs_by_hand():
    assert N.critical_costs(0.5) == (1.0, 1.0)
    b, h = N.critical_costs(0.8)
    assert b == pytest.approx(4.0) and h == 1.0
    assert N.critical_costs(0.95)[0] == pytest.approx(19.0)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            N.critical_costs(bad)


def test_empirical_quantile_uses_the_rank_ceil_tau_n():
    v = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    assert N.empirical_quantile(v, 0.5) == 3.0       # Rang ceil(2,5) = 3
    assert N.empirical_quantile(v, 0.8) == 4.0       # Rang 4 (0,8 * 5 = 4 ist ganzzahlig: Rang 4)
    assert N.empirical_quantile(v, 0.95) == 5.0
    assert N.empirical_quantile(v, 0.01) == 1.0
    assert N.empirical_quantile(np.array([7.0]), 0.3) == 7.0
    with pytest.raises(ValueError):
        N.empirical_quantile(np.array([]), 0.5)


def test_weighted_quantile_by_hand():
    """Nachfragen (10, 30, 20) mit Gewichten (0,2, 0,3, 0,5): sortiert 10 (0,2), 20 (0,5), 30 (0,3), kumuliert 0,2 / 0,7 / 1,0."""
    d = np.array([10.0, 30.0, 20.0])
    w = np.array([[0.2, 0.3, 0.5]])
    assert N.weighted_quantile(w, d, 0.1).tolist() == [10.0]
    assert N.weighted_quantile(w, d, 0.2).tolist() == [10.0]      # kumuliert 0,2 erreicht tau = 0,2 (Toleranz)
    assert N.weighted_quantile(w, d, 0.5).tolist() == [20.0]
    assert N.weighted_quantile(w, d, 0.7).tolist() == [20.0]
    assert N.weighted_quantile(w, d, 0.71).tolist() == [30.0]
    assert N.weighted_quantile(w, d, 1.0).tolist() == [30.0]


def test_equal_weights_give_the_empirical_quantile():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 10, size=37)
    w = np.full((1, 37), 1 / 37)
    for tau in (0.5, 0.8, 0.95):
        assert N.weighted_quantile(w, d, tau)[0] == N.empirical_quantile(d, tau)


def test_knn_weights_and_neighbour_order_by_hand():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    Q = np.array([[0.9], [2.9]])
    order = N.neighbour_order(N.sqdist(Q, X))
    assert order.tolist() == [[1, 0, 2, 3], [2, 1, 0, 3]]
    w = N.knn_weights(order, 2)
    assert w.tolist() == [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]
    assert N.knn_weights(order, 99).tolist() == [[0.25] * 4, [0.25] * 4]     # k größer als n: alle gleich
    with pytest.raises(ValueError):
        N.knn_weights(order, 0)


def test_neighbour_order_breaks_distance_ties_by_the_smaller_index():
    X = np.array([[1.0], [-1.0], [1.0]])
    order = N.neighbour_order(N.sqdist(np.array([[0.0]]), X))
    assert order.tolist() == [[0, 1, 2]]


def test_kernel_weights_sum_to_one_and_the_limits_are_equal_weights_and_nearest_neighbour():
    X = np.array([[0.0], [1.0], [3.0]])
    d2 = N.sqdist(np.array([[0.2]]), X)
    w = N.kernel_weights(d2, 1.0)
    assert w.sum() == pytest.approx(1.0) and w[0, 0] > w[0, 1] > w[0, 2]
    assert np.allclose(N.kernel_weights(d2, 1e6), 1 / 3, atol=1e-6)
    assert N.kernel_weights(d2, 1e-3).tolist() == [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        N.kernel_weights(d2, 0.0)
    # Weight by hand: exp(-d²/2) mit d² = (0,04, 0,64, 7,84) bei h = 1, normiert
    raw = np.exp(-0.5 * np.array([0.04, 0.64, 7.84]))
    assert np.allclose(w[0], raw / raw.sum())


def test_pinball_by_hand():
    """tau = 0,8: Nachfrage 10, Menge 6: Unterdeckung 4, Verlust 0,8 * 4 = 3,2; Menge 12: Überdeckung 2, Verlust 0,2 * 2 = 0,4."""
    assert N.pinball(np.array([6.0]), np.array([10.0]), 0.8) == pytest.approx(3.2)
    assert N.pinball(np.array([12.0]), np.array([10.0]), 0.8) == pytest.approx(0.4)


def test_k_and_h_grids():
    assert N.k_grid(100) == [5, 10, 20, 40, 80]            # nur k <= n - 1
    assert N.k_grid(1000) == [5, 10, 20, 40, 80, 160]
    assert N.k_grid(3) == [5]                              # mindestens der kleinste Wert
    assert N.h_grid(3) == list(N.HGRID)
    assert N.h_grid(12) == pytest.approx([h * 2.0 for h in N.HGRID])


def test_loo_losses_have_one_value_per_grid_point_and_handle_n_equals_one():
    data = N.make_news_data("quadratisch", 3, 60, seed=1)
    assert len(N.loo_losses_knn(data.X, data.y, 0.8, [5, 10, 20])) == 3
    assert len(N.loo_losses_kernel(data.X, data.y, 0.8, N.h_grid(3))) == len(N.HGRID)
    assert N.loo_losses_knn(data.X[:1], data.y[:1], 0.8, [5]) == [0.0]


def test_worlds_are_reproducible_and_independent_of_n():
    a, b = N.make_news_data("quadratisch", 3, 100, seed=7), N.make_news_data("quadratisch", 3, 100, seed=7)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.Xt, b.Xt) and a.world == b.world
    c = N.make_news_data("quadratisch", 3, 500, seed=7)
    assert c.world == a.world and np.array_equal(c.Xt, a.Xt)               # Welt und Testsatz hängen nicht von n ab
    assert not np.array_equal(N.make_news_data("quadratisch", 3, 100, seed=8).y, a.y)
    assert np.isclose(np.linalg.norm(a.world.beta), 1.0)
    assert a.y.min() >= 0 and a.X.shape == (100, 3) and a.Xt.shape == (N.N_TEST, 3)
    with pytest.raises(ValueError):
        N.make_news_data("betrag", 3, 10, 0)
    with pytest.raises(ValueError):
        N.make_news_data("linear", 3, 0, 0)


def test_quadratic_world_differs_from_the_linear_one_only_by_the_x1_term():
    lin, quad = N.make_news_data("linear", 3, 10, 2), N.make_news_data("quadratisch", 3, 10, 2)
    diff = quad.world.mu(lin.Xt) - lin.world.mu(lin.Xt)
    assert np.allclose(diff, N.MU_QUAD * (lin.Xt[:, 0] ** 2 - 1.0))


def test_oracle_quantile_is_the_tau_quantile_of_the_truncated_demand():
    w = N.make_world("linear", 3, 0)
    X = np.zeros((1, 3))                                     # mu = 60, sigma = 8
    assert w.oracle_quantile(X, 0.5)[0] == pytest.approx(60.0)
    assert w.oracle_quantile(X, 0.95)[0] == pytest.approx(60.0 + 8.0 * 1.6448536269514722)


def test_oracle_has_zero_extra_cost_and_every_other_rule_nonnegative_extra_cost():
    data = N.make_news_data("quadratisch", 3, 100, seed=3)
    for tau in N.TAUS:
        q0 = data.world.oracle_quantile(data.Xt, tau)
        assert N.extra_cost_percent(data, q0, tau) == pytest.approx(0.0, abs=1e-9)
        assert N.extra_cost_percent(data, q0 + 3.0, tau) > 0 and N.extra_cost_percent(data, q0 - 3.0, tau) > 0


def test_quantreg_input_checks_and_exact_fit():
    with pytest.raises(ValueError):
        N.quantreg_fit(np.zeros((3, 1)), np.zeros(3), 1.0)
    X = np.linspace(-1, 1, 12)[:, None]
    y = 5.0 + 2.0 * X[:, 0]                                 # exakt linear: jede Quantilregression findet sie
    coef = N.quantreg_fit(X, y, 0.8)
    assert coef == pytest.approx([5.0, 2.0], abs=1e-7)


def test_kq_emp_on_an_exactly_linear_world_without_spread_is_the_line_plus_the_residual_quantile():
    X = np.linspace(-1, 1, 10)[:, None]
    y = 3.0 + 4.0 * X[:, 0]
    data = N.NewsData(N.NewsWorld("linear", (1.0, 0.0, 0.0)), X, y, np.array([[0.0], [0.5]]))
    assert np.allclose(N.rule_kq_emp(data, 0.8), [3.0, 5.0])        # Residuen alle 0


def test_rules_on_one_instance_follow_the_class_logic_as_bands():
    """Welt quadratisch, n = 1000, tau = 0,95, ein Seed: die Gewichte liegen weit unter den linearen Regeln (in der Messung dieser Demo im Median 4- bis 5-mal darunter); Welt linear: die lineare Regel liegt unter den Gewichten."""
    q = N.run_news("quadratisch", 3, 1000, 0.95, 0)
    assert q["knn"] < 0.5 * min(q["kq_emp"], q["direkt"]) and q["kern"] < 0.5 * min(q["kq_emp"], q["direkt"])
    li = N.run_news("linear", 3, 1000, 0.95, 0)
    assert li["kq_emp"] < li["knn"] and li["kq_emp"] < li["kern"] and li["kq_emp"] < 2.0
    assert set(q) == {"seed", "k", "h", "kq_emp", "direkt", "knn", "kern"} and q["k"] in N.KGRID and q["h"] in N.h_grid(3)
