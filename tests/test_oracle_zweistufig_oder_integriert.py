"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Löser, SPO+ und Teilgradient gegen unabhängige Rechenwege.

1. Der Löser (Rückwärtsrechnung) gegen die Aufzählung der 70 Wege. Die Wege werden in `conftest.independent_paths` rekursiv erzeugt (andere Konstruktion als
   `Grid.paths`), die Kosten je Weg als Summe der Bogenkosten gerechnet; Gleichstände (ganzzahlige Kosten) folgen derselben Regel (erster kleinster in der Reihenfolge R vor D).
2. SPO+ gegen das Brute-Force-Maximum über alle 70 Wege.
3. Teilgradient gegen den Differenzenquotienten in Richtungen ohne Knick (SPO+ ist stückweise linear in ch: der zentrale Differenzenquotient ist dort exakt).
4. Gesamt-Teilgradient des Trainings (Kettenregel über das lineare Modell) gegen den Differenzenquotienten in den Parametern.
5. KQ gegen die Normalengleichungen, SplitMix64 steht in test_rng.py (Python-Ganzzahlreferenz).

Im Scratch-Lauf (nicht im Repo) mit 300 Instanzen für den Löser, 2000 für SPO+ und Teilgradient wiederholt, jeweils ohne Abweichung."""

import numpy as np
import pytest
from conftest import independent_paths, independent_solve

import zoi_losses as L
from zoi_training import add_intercept, fit_least_squares, subgradient_of_mean_loss


def test_solver_equals_enumeration_on_random_instances(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 5.0, size=(60, 40))
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.abs(z - zo).max() < 1e-12
    assert np.array_equal(W, Wo)                      # stetige Kosten: der beste Weg ist eindeutig


def test_solver_equals_enumeration_with_ties_on_integer_costs(grid5):
    rng = np.random.default_rng(1)
    C = rng.integers(0, 3, size=(60, 40)).astype(float)
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.array_equal(z, zo) and np.array_equal(W, Wo)   # auch die gewählte Wahl bei Gleichstand (R vor D) stimmt überein


def test_solver_equals_enumeration_on_smaller_grids(grid3, grid2):
    rng = np.random.default_rng(2)
    for grid in (grid3, grid2):
        C = rng.uniform(0, 5, size=(40, grid.d))
        W, z = grid.solve(C)
        Wo, zo = independent_solve(grid, C)
        assert np.abs(z - zo).max() < 1e-12 and np.array_equal(W, Wo)


def test_independent_enumeration_has_70_paths_and_matches_the_hand_count(grid5):
    paths = independent_paths(grid5)
    assert len(paths) == 70 and all(len(arcs) == 8 for _, arcs in paths)
    keys = ["".join(m).replace("R", "0").replace("D", "1") for m, _ in paths]
    assert keys == sorted(keys) and len(set(keys)) == 70      # lexikographisch mit R vor D, alle verschieden


def test_spo_plus_equals_brute_force_maximum_over_all_paths(grid5):
    rng = np.random.default_rng(3)
    P = grid5.paths()
    C = rng.uniform(0.5, 4, size=(80, 40))
    Chat = C + rng.normal(0, 1.2, size=C.shape)
    _, zstar = independent_solve(grid5, C)
    Wstar, _ = independent_solve(grid5, C)
    brute = np.max((C - 2 * Chat) @ P.T, axis=1) + 2 * np.sum(Chat * Wstar, axis=1) - zstar
    assert np.abs(L.spo_plus_loss(grid5, C, Chat) - brute).max() < 1e-10


def test_spo_loss_equals_brute_force_regret(grid5):
    rng = np.random.default_rng(4)
    P = grid5.paths()
    C = rng.uniform(0.5, 4, size=(80, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    chosen = np.argmin(Chat @ P.T, axis=1)
    brute = np.sum(C * P[chosen], axis=1) - (C @ P.T).min(axis=1)
    assert np.abs(L.spo_loss(grid5, C, Chat) - brute).max() < 1e-12


def test_subgradient_equals_central_difference_where_there_is_no_kink(grid5):
    rng = np.random.default_rng(5)
    n, eps = 400, 1e-6
    C = rng.uniform(0.5, 4, size=(n, 40))
    Chat = rng.normal(2, 1.2, size=(n, 40))
    V = rng.normal(size=(n, 40))                      # eine Richtung je Fall
    plus = L.spo_plus_loss(grid5, C, Chat + eps * V)
    minus = L.spo_plus_loss(grid5, C, Chat - eps * V)
    numeric = (plus - minus) / (2 * eps)
    analytic = np.sum(L.spo_plus_subgradient(grid5, C, Chat) * V, axis=1)
    # dieselbe Wegwahl vor und nach dem Schritt: kein Knick im Intervall
    same = np.all(grid5.solve(2 * (Chat + eps * V) - C)[0] == grid5.solve(2 * (Chat - eps * V) - C)[0], axis=1)
    assert same.sum() >= 0.95 * n
    assert np.abs(numeric[same] - analytic[same]).max() < 1e-6


def test_training_subgradient_equals_central_difference_in_the_parameters(grid5):
    rng = np.random.default_rng(6)
    n = 120
    X = rng.normal(size=(n, 5))
    C = rng.uniform(0.5, 4, size=(n, 40))
    coef = rng.normal(0.3, 0.2, size=(6, 40))
    V = rng.normal(size=coef.shape)
    Xa = add_intercept(X)
    eps = 1e-7

    def mean_loss(c):
        return float(np.mean(L.spo_plus_loss(grid5, C, Xa @ c)))

    numeric = (mean_loss(coef + eps * V) - mean_loss(coef - eps * V)) / (2 * eps)
    analytic = float(np.sum(subgradient_of_mean_loss(grid5, X, C, coef) * V))
    assert numeric == pytest.approx(analytic, abs=1e-5)


def test_least_squares_equals_the_normal_equations():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 5))
    C = rng.normal(size=(60, 7))
    Xa = add_intercept(X)
    expected = np.linalg.solve(Xa.T @ Xa, Xa.T @ C)
    assert np.abs(fit_least_squares(X, C) - expected).max() < 1e-10
