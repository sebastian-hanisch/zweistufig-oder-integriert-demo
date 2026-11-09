"""Kern-ETO (zoi_kernel): Handbeispiel, Grenzfälle, Auswahl per Validierung. Der Vergleich mit unabhängigen Rechenwegen steht in `test_oracle_zweistufig_oder_integriert.py`."""

import numpy as np
import pytest

import zoi_kernel as K


def test_rbf_kernel_by_hand():
    """Abstand 1 bei gamma = ln 2 gibt exp(-ln 2) = 1/2; Abstand 0 gibt 1; Abstand 2 (quadratisch 4) gibt 2^-4 = 1/16."""
    g = np.log(2.0)
    A = np.array([[0.0], [1.0], [2.0]])
    Kmat = K.rbf_kernel(A, A, g)
    assert np.allclose(np.diag(Kmat), 1.0)
    assert Kmat[0, 1] == pytest.approx(0.5) and Kmat[1, 2] == pytest.approx(0.5) and Kmat[0, 2] == pytest.approx(1 / 16)
    assert np.allclose(Kmat, Kmat.T)


def test_two_point_example_by_hand():
    """Zwei Fälle x = 0 und x = 1, ein Bogen, Kosten 2 und 6. Mittel mu = 4, zentriert r = (-2, +2).
    gamma = ln 2 (k = 1/2), Strafe a = 1/2: K + aI hat auf dem Vektor (1, -1) den Eigenwert 1 - k + a = 1, also alpha = r / 1 = (-2, 2).
    Vorhersage bei x = 0: 4 + (1 * (-2) + k * 2) = 4 - 2 + 1 = 3; bei x = 1: 4 + (k * (-2) + 1 * 2) = 4 + 1 = 5."""
    X = np.array([[0.0], [1.0]])
    C = np.array([[2.0], [6.0]])
    model = K.fit_kernel_ridge(X, C, gamma=np.log(2.0), ridge=0.5)
    assert model.mu.tolist() == [4.0]
    assert np.allclose(model.alpha_coef, [[-2.0], [2.0]])
    assert np.allclose(model.predict(X), [[3.0], [5.0]])
    # bei x = 0,5 (Abstand 1/2 zu beiden Fällen, k = 2^-1/4): die Summe der Gewichte ist 0, die Vorhersage das Mittel
    assert model.predict(np.array([[0.5]]))[0, 0] == pytest.approx(4.0)


def test_zero_gamma_means_infinite_bandwidth_and_gives_the_constant_mean_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    C = rng.uniform(1, 4, size=(30, 6))
    model = K.fit_kernel_ridge(X, C, gamma=0.0, ridge=0.1)
    Xq = rng.normal(size=(7, 5))
    assert np.allclose(model.predict(Xq), np.tile(C.mean(axis=0), (7, 1)), atol=1e-10)


def test_huge_ridge_gives_the_mean_and_tiny_ridge_nearly_interpolates():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    C = rng.uniform(1, 4, size=(20, 4))
    big = K.fit_kernel_ridge(X, C, gamma=0.3, ridge=1e9)
    assert np.allclose(big.predict(X), np.tile(C.mean(axis=0), (20, 1)), atol=1e-6)
    small = K.fit_kernel_ridge(X, C, gamma=0.3, ridge=1e-10)
    assert np.abs(small.predict(X) - C).max() < 1e-4        # Interpolation (der Kern ist gut konditioniert genug bei 20 Punkten in 3 Dimensionen)


def test_input_checks():
    X, C = np.zeros((3, 2)), np.ones((3, 2))
    with pytest.raises(ValueError):
        K.fit_kernel_ridge(X, C, gamma=0.1, ridge=0.0)
    with pytest.raises(ValueError):
        K.fit_kernel_ridge(X, C, gamma=-1.0, ridge=1.0)


def test_validation_split_is_a_partition_deterministic_and_has_both_parts():
    for n in (2, 3, 5, 10, 100, 1000):
        fit, val = K.validation_split(n, seed=4)
        assert len(fit) >= 1 and len(val) >= 1 and len(fit) + len(val) == n
        assert sorted(np.concatenate([fit, val]).tolist()) == list(range(n))
        assert np.array_equal(fit, K.validation_split(n, seed=4)[0])
    fit, val = K.validation_split(100, 4)
    assert len(fit) == 80 and len(val) == 20
    assert not np.array_equal(K.validation_split(100, 4)[0], K.validation_split(100, 5)[0])


def test_selection_picks_the_smallest_validation_error_and_the_first_on_ties():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    C = (X[:, :1] ** 2) + 0.01 * rng.normal(size=(60, 1))
    g, a, err = K.choose_parameters(X, C, seed=1)
    assert err.shape == (len(K.GAMMAS), len(K.ALPHAS))
    i, j = K.GAMMAS.index(g), K.ALPHAS.index(a)
    assert err[i, j] == err.min()
    # Gleichstand: zwei gleiche Werte in der Strafliste geben dieselbe Spalte, gewählt wird die erste
    g2, a2, _ = K.choose_parameters(X, C, seed=1, gammas=(0.1, 0.1), ridges=(0.5, 0.5))
    assert (g2, a2) == (0.1, 0.5)


def test_single_case_gives_the_constant_model_and_two_cases_run():
    one = K.fit_kernel_eto(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]]), seed=0)
    assert np.allclose(one.predict(np.array([[9.0, -9.0], [0.0, 0.0]])), [[3.0, 4.0, 5.0]] * 2)
    two = K.fit_kernel_eto(np.array([[0.0], [1.0]]), np.array([[1.0, 2.0], [3.0, 4.0]]), seed=0)
    assert np.all(np.isfinite(two.predict(np.array([[0.5]])))) and two.predict(np.array([[0.5]])).shape == (1, 2)


def test_the_chosen_model_is_refit_on_all_training_cases():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    C = rng.uniform(1, 3, size=(40, 3))
    model = K.fit_kernel_eto(X, C, seed=5)
    g, a, _ = K.choose_parameters(X, C, seed=5)
    assert (model.gamma, model.ridge) == (g, a)
    assert model.X.shape == (40, 2) and model.alpha_coef.shape == (40, 3)


def test_a_smooth_truth_is_learned_much_better_than_by_a_linear_fit():
    """Wahrheit c = x^2 (eine Dimension): der Kern-Ridge trifft sie, eine Gerade nicht (Band, keine feste Zahl)."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-2, 2, size=(200, 1))
    C = X ** 2
    Xt = np.linspace(-2, 2, 101)[:, None]
    kern = K.fit_kernel_eto(X, C, seed=0).predict(Xt)
    slope = np.polyfit(X[:, 0], C[:, 0], 1)
    lin = np.polyval(slope, Xt[:, 0])[:, None]
    assert np.mean((kern - Xt ** 2) ** 2) < 0.1 * np.mean((lin - Xt ** 2) ** 2)
