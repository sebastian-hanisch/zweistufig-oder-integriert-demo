"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, n kleiner als die Parameterzahl, Gleichstände, e = 0, Nullen, kleinstes Gitter, Grenzen der Regler."""

import numpy as np
import pytest

import zoi_evaluation as ev
import zoi_gap as G
import zoi_kernel as K
import zoi_news as N
import zoi_runs as R
from zoi_grid import Grid
from zoi_world import WorldSpec, make_dataset


def test_world_run_on_the_smallest_grid_does_not_break():
    run = R.run_world(R.RunSpec(h=2, deg=2, noise=0.0, n=20, seed=1, steps=10, n_test=30))
    assert run.grid.d == 4 and run.data.C_test.shape == (30, 4)
    assert run.regret["orakel"] == pytest.approx(0.0, abs=1e-12)
    assert all(np.isfinite(v) for v in run.regret.values())


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6, 7])
def test_runs_with_fewer_cases_than_parameters_do_not_break(n):
    """Das lineare Modell hat 6 x 40 = 240 Parameter: für n < 240 ist KQ unterbestimmt (kleinste Norm), SPO+ startet dort; Kern-ETO braucht ab n = 2 eine Validierung."""
    run = R.run_world(R.RunSpec(deg=4, noise=0.5, n=n, seed=3, steps=5, n_test=40))
    assert all(np.isfinite(v) and v >= -1e-12 for v in run.regret.values()), (n, run.regret)
    assert np.all(np.isfinite(run.chat["kern"])) and run.chat["kern"].shape == (40, 40)


def test_kernel_eto_with_a_single_case_predicts_that_case_everywhere():
    data = make_dataset(WorldSpec(deg=4), 40, 1, 20, seed=1)
    model = K.fit_kernel_eto(data.X_train, data.C_train, seed=1)
    assert np.allclose(model.predict(data.X_test), np.tile(data.C_train[0], (20, 1)))


def test_kernel_eto_without_features_is_the_constant_model():
    """Welt ohne Merkmale (alle Zeilen von X gleich): jeder Fall sieht gleich aus, die Vorhersage ist das Mittel der Kosten (alle Verfahren in dieser Klasse gleich)."""
    X = np.zeros((30, 5))
    C = np.random.default_rng(0).uniform(1, 3, size=(30, 4))
    model = K.fit_kernel_ridge(X, C, gamma=0.1, ridge=0.5)
    assert np.allclose(model.predict(np.zeros((3, 5))), np.tile(C.mean(axis=0), (3, 1)), atol=1e-9)   # K = 11^T, Residuen ohne Mittel: Summe 0


def test_exactly_linear_noiseless_world_every_method_has_tiny_regret(grid5):
    """Plan: in einer exakt linearen, rauschfreien Welt (Grad 1, Rauschen 0) haben KQ und SPO+ Regret 0 (KQ exakt bis auf Rundung); Kern-ETO bleibt knapp darüber (Strafe und endliche Bandbreite), als Band."""
    run = R.run_world(R.RunSpec(deg=1, noise=0.0, n=200, seed=0, n_test=300))
    assert run.regret["orakel"] == pytest.approx(0.0, abs=1e-12)
    assert run.regret["kq"] < 1e-6
    assert run.regret["spo"] < 1e-3
    assert run.regret["kern"] < 0.005                      # unter 0,5 % (Band; gemessen in den Bändern dieser Demo: im Median 0,00 %)


def test_ties_in_the_best_value_give_a_zero_gap_and_a_valid_solver_path(grid5):
    M = np.ones((2, grid5.d))
    gap, best = G.gap_to_second_best(grid5.paths(), M)
    assert gap.tolist() == [0.0, 0.0] and best.tolist() == [8.0, 8.0]
    W, z = grid5.solve(M)
    assert z.tolist() == [8.0, 8.0] and np.all(W.sum(axis=1) == 8)


def test_zero_noise_world_has_an_oracle_with_regret_zero_and_noise_world_a_positive_floor():
    r0 = R.run_world(R.RunSpec(deg=6, noise=0.0, n=100, seed=4, steps=3, with_kernel=False))
    r5 = R.run_world(R.RunSpec(deg=6, noise=0.5, n=100, seed=4, steps=3, with_kernel=False))
    assert abs(r0.regret["orakel"]) < 1e-12 and r5.regret["orakel"] > 0.05


def test_more_noise_raises_the_oracle_floor():
    """Der Rauschboden (Regret des Orakels) wächst mit der Rauschstufe: gleiche Welt, drei Stufen (Median über vier Seeds, Wachstum in allen drei Schritten)."""
    medians = []
    for noise in (0.0, 0.25, 0.5):
        vals = [R.run_world(R.RunSpec(deg=4, noise=noise, n=10, seed=s, steps=1, n_test=300, with_kernel=False)).regret["orakel"] for s in range(4)]
        medians.append(float(np.median(vals)))
    assert medians[0] < medians[1] < medians[2]


def test_newsvendor_with_the_smallest_samples_runs():
    for kind in N.KINDS:
        for n in (1, 2, 5):
            data = N.make_news_data(kind, 3, n, seed=2)
            for rule in (N.rule_kq_emp,):
                assert np.all(np.isfinite(rule(data, 0.8)))
            assert np.all(np.isfinite(N.rule_knn(data, 0.8)[0])) and np.all(np.isfinite(N.rule_kern(data, 0.8)[0]))
    one = N.make_news_data("linear", 3, 1, seed=2)
    assert np.allclose(N.rule_knn(one, 0.95)[0], one.y[0])      # eine Beobachtung: ihre Nachfrage wird bestellt


def test_quantile_regression_with_fewer_cases_than_parameters_still_returns_a_solution():
    data = N.make_news_data("linear", 10, 6, seed=3)
    coef = N.quantreg_fit(data.X, data.y, 0.8)
    assert coef.shape == (11,) and np.all(np.isfinite(coef))


def test_grid_class_rejects_a_wrong_cost_width():
    with pytest.raises(ValueError):
        Grid(5).solve(np.ones((1, 41)))


def test_settings_at_the_slider_corners_give_complete_tables():
    from zoi_constants import DEG_CHOICES, N_BAND, NOISE_CHOICES
    for deg in DEG_CHOICES:
        rows = ev.grid_rows(deg)
        assert [(r["n"], r["noise"]) for r in rows] == [(n, e) for e in NOISE_CHOICES for n in N_BAND]
    for n in (100, 1000):
        assert [r["key"] for r in ev.class_rows(6, n, 0.5)] == ["orakel", "kq", "spo", "kern", "wald"]
    assert [r["key"] for r in ev.class_rows(6, 300, 0.5)] == ["orakel", "kq", "spo", "kern"]      # bei n = 300 gibt es keinen Zufallswald der Vormessung
    assert [r["key"] for r in ev.class_rows(6, 5000, 0.5)] == ["orakel", "kq", "spo", "wald"]    # bei n = 5000 keinen Kern-ETO
