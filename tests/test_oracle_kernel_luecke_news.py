"""Orakel-Tests für Kern-ETO, Lücke Δ, Parameterwahl und Newsvendor (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5).

Jeder Test rechnet auf einem anderen Weg als der Prüfling:

1. Kern-Ridge gegen die Eigenzerlegung (`scipy.linalg.eigh`, nicht `numpy.linalg.solve`) mit einem Gauß-Kern aus Doppelschleifen, und gegen scikit-learn `KernelRidge` (falls installiert).
2. Parameterwahl (Bandbreite, Strafe) gegen die Aufzählung aller 16 Paare mit `scipy.linalg.solve(assume_a="pos")`.
3. Lücke Δ und Regret in Erwartung gegen die Aufzählung der rekursiv erzeugten 70 Wege (`conftest.independent_paths`).
4. Normierter Regret eines Laufs gegen den Aufzählungs-Löser.
5. Newsvendor: erwartete Kosten gegen numerische Integration (`scipy.integrate.quad`), Orakelmenge gegen Rastersuche, Quantilregression gegen die Aufzählung der Ecken (p = 1) und scikit-learn
   `QuantileRegressor` (falls installiert), Leave-one-out gegen eine Schleife.

Im Scratch-Lauf (nicht im Repo) mit mehr Instanzen wiederholt (Kern-Ridge 200, Lücke 200, Newsvendor-Kosten 100), jeweils ohne Abweichung über der Toleranz."""

import itertools

import numpy as np
import pytest
import scipy.linalg
from conftest import independent_paths, independent_solve
from scipy import integrate

import zoi_gap as G
import zoi_kernel as K
import zoi_news as N
import zoi_runs as R


def _gauss(a, b, gamma):
    """Gauß-Kern per Doppelschleife (kein Matrixprodukt, anderer Rechenweg als `rbf_kernel`)."""
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i, j] = np.exp(-gamma * sum((u - v) ** 2 for u, v in zip(x, y)))
    return out


def test_kernel_ridge_equals_the_eigendecomposition_solution():
    """alpha = (K + aI)^-1 r über K = Q diag(l) Q^T: alpha = Q diag(1 / (l + a)) Q^T r."""
    rng = np.random.default_rng(0)
    for n, p, d in ((12, 2, 3), (30, 5, 7)):
        X, C = rng.normal(size=(n, p)), rng.uniform(1, 4, size=(n, d))
        Xq = rng.normal(size=(9, p))
        gamma, ridge = 0.2, 0.05
        model = K.fit_kernel_ridge(X, C, gamma, ridge)
        lam, Q = scipy.linalg.eigh(_gauss(X, X, gamma))
        mu = C.mean(axis=0)
        alpha = Q @ ((Q.T @ (C - mu)) / (lam + ridge)[:, None])
        expected = mu + _gauss(Xq, X, gamma) @ alpha
        assert np.abs(model.predict(Xq) - expected).max() < 1e-9


def test_kernel_ridge_equals_scikit_learn():
    sk = pytest.importorskip("sklearn.kernel_ridge")
    rng = np.random.default_rng(1)
    X, C, Xq = rng.normal(size=(40, 5)), rng.uniform(1, 4, size=(40, 6)), rng.normal(size=(10, 5))
    for gamma, ridge in ((0.05, 0.01), (0.3, 1.0)):
        mu = C.mean(axis=0)
        ref = sk.KernelRidge(kernel="rbf", gamma=gamma, alpha=ridge).fit(X, C - mu).predict(Xq) + mu
        assert np.abs(K.fit_kernel_ridge(X, C, gamma, ridge).predict(Xq) - ref).max() < 1e-8


def test_parameter_choice_equals_enumeration_of_the_grid():
    """Das Paar mit dem kleinsten Validierungsfehler, selbst ausgezählt: Schnitt aus `validation_split`, Anpassung mit `scipy.linalg.solve`, erster Wert bei Gleichstand."""
    rng = np.random.default_rng(2)
    for seed in (0, 1, 2):
        X = rng.normal(size=(50, 3))
        C = X[:, :1] ** 2 + 0.3 * rng.normal(size=(50, 2))
        fit, val = K.validation_split(len(X), seed)
        best, best_err = None, np.inf
        for g in K.GAMMAS:
            for a in K.ALPHAS:
                mu = C[fit].mean(axis=0)
                alpha = scipy.linalg.solve(_gauss(X[fit], X[fit], g) + a * np.eye(len(fit)), C[fit] - mu, assume_a="pos")
                err = float(np.mean((mu + _gauss(X[val], X[fit], g) @ alpha - C[val]) ** 2))
                if err < best_err - 1e-15:
                    best, best_err = (g, a), err
        g, a, err_matrix = K.choose_parameters(X, C, seed)
        assert (g, a) == best and err_matrix.min() == pytest.approx(best_err, rel=1e-9)


def test_gap_equals_the_enumeration_of_the_recursive_paths(grid5):
    """Δ über die rekursiv erzeugten Wege gegen `gap_to_second_best` (Wegmatrix `Grid.paths`); der beste Wert gegen den Löser."""
    rng = np.random.default_rng(3)
    M = rng.uniform(0.5, 5.0, size=(40, 40))
    paths = independent_paths(grid5)
    gap, best = G.gap_to_second_best(grid5.paths(), M)
    for r, m in enumerate(M):
        costs = sorted(sum(m[k] for k in arcs) for _, arcs in paths)
        assert gap[r] == pytest.approx(costs[1] - costs[0], abs=1e-12) and best[r] == pytest.approx(costs[0], abs=1e-12)
    assert np.allclose(best, grid5.solve(M)[1])


def test_gap_with_integer_costs_counts_all_ties(grid5):
    rng = np.random.default_rng(4)
    M = rng.integers(1, 3, size=(40, 40)).astype(float)
    paths = independent_paths(grid5)
    gap, _ = G.gap_to_second_best(grid5.paths(), M)
    for r, m in enumerate(M):
        costs = sorted(sum(m[k] for k in arcs) for _, arcs in paths)
        assert gap[r] == costs[1] - costs[0]


def test_mean_regret_per_case_equals_the_enumeration(grid5):
    rng = np.random.default_rng(5)
    M = rng.uniform(1, 4, size=(25, 40))
    Chat = M + rng.normal(0, 0.6, size=M.shape)
    paths = independent_paths(grid5)
    got = G.mean_regret_per_case(grid5.paths(), M, Chat, grid5)
    for r in range(len(M)):
        chosen = min(paths, key=lambda pa: sum(Chat[r, k] for k in pa[1]))            # erster kleinster Weg unter Chat
        expected = sum(M[r, k] for k in chosen[1]) - min(sum(M[r, k] for k in pa[1]) for pa in paths)
        assert got[r] == pytest.approx(expected, abs=1e-12)


def test_normalized_regret_of_a_run_equals_the_enumeration_on_a_small_world():
    run = R.run_world(R.RunSpec(deg=4, noise=0.5, n=60, seed=2, steps=5, n_test=30))
    _, zstar = independent_solve(run.grid, run.data.C_test)
    for m in ("kq", "spo", "kern", "orakel"):
        W, _ = independent_solve(run.grid, run.chat[m])
        assert run.regret[m] == pytest.approx((np.sum(run.data.C_test * W) - np.sum(zstar)) / np.sum(zstar), abs=1e-12)


def test_expected_newsvendor_cost_equals_numerical_integration():
    world = N.make_world("quadratisch", 3, 1)
    rng = np.random.default_rng(6)
    X = rng.normal(size=(4, 3))
    for tau in (0.5, 0.8, 0.95):
        b, h = N.critical_costs(tau)
        q = np.maximum(0.0, world.oracle_quantile(X, tau) + rng.normal(0, 5, size=4))
        got = N.expected_cost_at(world, X, q, b, h)
        for i in range(4):
            m = float(world.mu(X)[i])

            def density(y, m=m):
                return np.exp(-0.5 * ((y - m) / N.SIGMA) ** 2) / (N.SIGMA * np.sqrt(2 * np.pi))

            def cost(y, i=i):
                demand = max(y, 0.0)                                             # D = max(0, Y)
                return b * max(demand - q[i], 0.0) + h * max(q[i] - demand, 0.0)

            ref = integrate.quad(lambda y: cost(y) * density(y), m - 12 * N.SIGMA, m + 12 * N.SIGMA, points=[0.0, float(q[i])], limit=400)[0]
            assert got[i] == pytest.approx(ref, rel=1e-7, abs=1e-8)


def test_oracle_quantile_minimizes_the_expected_cost_on_a_grid():
    world = N.make_world("linear", 3, 2)
    X = np.array([[0.3, -1.0, 0.5], [2.0, 1.0, -1.0]])
    for tau in (0.5, 0.8, 0.95):
        b, h = N.critical_costs(tau)
        q0 = world.oracle_quantile(X, tau)
        c0 = N.expected_cost_at(world, X, q0, b, h)
        for delta in (-2.0, -0.5, -0.05, 0.05, 0.5, 2.0):
            assert np.all(N.expected_cost_at(world, X, q0 + delta, b, h) > c0)


def test_quantile_regression_equals_the_enumeration_of_the_vertices():
    """p = 1: eine Optimallösung der Quantilregression geht durch zwei Datenpunkte (Ecken des Programms); der kleinste Pinball-Verlust über alle Paare ist der Wert des LP."""
    rng = np.random.default_rng(7)
    for tau in (0.5, 0.8, 0.95):
        X = rng.normal(size=(9, 1))
        y = 2 + 1.5 * X[:, 0] + rng.standard_t(3, size=9)
        best = np.inf
        for i, j in itertools.combinations(range(9), 2):
            slope = (y[j] - y[i]) / (X[j, 0] - X[i, 0])
            icpt = y[i] - slope * X[i, 0]
            r = y - icpt - slope * X[:, 0]
            best = min(best, float(np.sum(np.maximum(tau * r, (tau - 1) * r))))
        coef = N.quantreg_fit(X, y, tau)
        r = y - coef[0] - coef[1] * X[:, 0]
        assert float(np.sum(np.maximum(tau * r, (tau - 1) * r))) == pytest.approx(best, abs=1e-7)


def test_quantile_regression_equals_scikit_learn_if_available():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    y = 10 + X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=80) * (1 + 0.5 * np.abs(X[:, 0]))
    for tau in (0.5, 0.8):
        ref = sk.QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X, y)
        coef = N.quantreg_fit(X, y, tau)

        def pinball_sum(c0, c):
            r = y - c0 - X @ c
            return float(np.sum(np.maximum(tau * r, (tau - 1) * r)))

        assert pinball_sum(coef[0], coef[1:]) == pytest.approx(pinball_sum(ref.intercept_, ref.coef_), abs=1e-6)


def test_leave_one_out_losses_equal_a_naive_loop():
    data = N.make_news_data("quadratisch", 3, 40, seed=3)
    tau, ks = 0.8, [3, 5, 10]
    got = N.loo_losses_knn(data.X, data.y, tau, ks)
    for k, g in zip(ks, got):
        total = 0.0
        for i in range(40):
            idx = [j for j in range(40) if j != i]
            near = sorted(idx, key=lambda j: (float(np.sum((data.X[i] - data.X[j]) ** 2)), j))[:k]
            vals = np.sort(data.y[near])                                           # gleiche Gewichte 1/k: das ceil(tau k)-kleinste
            q = vals[int(np.ceil(tau * k - 1e-12)) - 1]
            r = data.y[i] - q
            total += max(tau * r, (tau - 1) * r)
        assert g == pytest.approx(total / 40, abs=1e-12)
