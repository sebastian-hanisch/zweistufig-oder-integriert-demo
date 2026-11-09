"""Kern-ETO: Gauß-Kern-Ridge als flexible Modellklasse für die zweistufige Regel (keine Streamlit-Abhängigkeit, nur numpy).

Zweistufig heißt hier: erst die Bogenkosten vorhersagen (Kleinste Quadrate auf dem Kostenfehler, hier mit einem Gauß-Kern), dann den Löser auf der Vorhersage laufen lassen.
Das Modell ist je Bogen derselbe Kern-Ridge-Schätzer, alle 40 Bögen werden zugleich gerechnet (eine Matrix als rechte Seite):

    k(x, x') = exp(-gamma |x - x'|^2),   K = k(X, X),   mu = Mittel der Trainingskosten (je Bogen),
    alpha = (K + a I)^(-1) (C - mu)       (ein lineares System, `numpy.linalg.solve`),
    c_hat(x) = mu + k(x, X) alpha.

Parameter (gamma = 1 / (2 h²) mit der Bandbreite h, a = Strafe) werden per Validierung gewählt: ein fester Zufallsschnitt der Trainingsfälle (80 % Anpassung,
20 % Validierung), der Fehler ist der mittlere quadratische Fehler der vorhergesagten Kosten auf den Validierungsfällen, das Raster steht in `GAMMAS` und `ALPHAS`
(16 Paare). Gewählt wird das Paar mit dem kleinsten Validierungsfehler, bei Gleichstand das erste in der Reihenfolge gamma aufsteigend, dann a aufsteigend. Danach wird
mit dem gewählten Paar auf allen Trainingsfällen neu angepasst.

Herkunft: neu geschrieben nach der Beschreibung der Vormessung V1d (`entscheidungsorientiert-planung/vormessung/eo_v1d_kernel.py`, dort mit scikit-learn `KernelRidge`);
dasselbe Raster und derselbe Aufbau (zentrierte Kosten, Validierungsschnitt 80/20), aber eine andere Zufallsquelle für den Schnitt. Die Zahlen sind deshalb
vergleichbar, nicht gleich.

Grenzfälle: gamma -> 0 (Bandbreite unendlich) macht alle Kernwerte 1, die Vorhersage ist dann das konstante Mittel mu (siehe `tests/test_kernel.py`); n = 1 liefert das
konstante Modell, n = 2 und 3 haben mindestens einen Validierungsfall."""

from dataclasses import dataclass

import numpy as np

from zoi_rng import SplitMix64, derive_seed

GAMMAS = (0.02, 0.05, 0.1, 0.3)
ALPHAS = (0.001, 0.01, 0.1, 1.0)
VALID_SHARE = 0.8     # Anteil der Trainingsfälle, auf dem je Paar angepasst wird
TAG_SPLIT = 4         # Etikett des Teilstroms für den Validierungsschnitt (1 bis 3 sind Gewichte, Training, Test)


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """Gauß-Kern exp(-gamma |a - b|^2) zwischen allen Zeilen von A und B (Binomialformel, auf 0 begrenzt gegen Rundungsreste)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    d2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.exp(-gamma * np.maximum(d2, 0.0))


@dataclass(frozen=True)
class KernelModel:
    """Angepasster Kern-Ridge: Trainingsmerkmale X, Gewichte alpha (n x d), Mittel mu (d,), Kernparameter."""

    X: np.ndarray
    alpha_coef: np.ndarray
    mu: np.ndarray
    gamma: float
    ridge: float

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        return self.mu + rbf_kernel(Xq, self.X, self.gamma) @ self.alpha_coef


def fit_kernel_ridge(X: np.ndarray, C: np.ndarray, gamma: float, ridge: float) -> KernelModel:
    """Kern-Ridge mit festen Parametern: alpha = (K + ridge I)^(-1) (C - mu), gelöst mit `numpy.linalg.solve`."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if ridge <= 0:
        raise ValueError("die Strafe muss positiv sein")
    if gamma < 0:
        raise ValueError("gamma darf nicht negativ sein")
    mu = C.mean(axis=0)
    K = rbf_kernel(X, X, gamma)
    coef = np.linalg.solve(K + ridge * np.eye(len(X)), C - mu)
    return KernelModel(X, coef, mu, float(gamma), float(ridge))


def validation_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fester Zufallsschnitt der Zeilen 0 bis n - 1 in Anpassung und Validierung (mindestens je eine Zeile, wenn n >= 2)."""
    order = np.argsort(SplitMix64(derive_seed(seed, TAG_SPLIT)).uniform(n), kind="stable")
    n_fit = max(1, min(n - 1, int(VALID_SHARE * n)))
    return order[:n_fit], order[n_fit:]


def validation_errors(X: np.ndarray, C: np.ndarray, seed: int, gammas=GAMMAS, ridges=ALPHAS) -> np.ndarray:
    """Validierungsfehler (mittlerer quadratischer Fehler der Kosten) je Paar: Matrix (len(gammas), len(ridges))."""
    fit, val = validation_split(len(X), seed)
    out = np.zeros((len(gammas), len(ridges)))
    for i, g in enumerate(gammas):
        for j, a in enumerate(ridges):
            m = fit_kernel_ridge(X[fit], C[fit], g, a)
            out[i, j] = float(np.mean((m.predict(X[val]) - C[val]) ** 2))
    return out


def choose_parameters(X: np.ndarray, C: np.ndarray, seed: int, gammas=GAMMAS, ridges=ALPHAS) -> tuple[float, float, np.ndarray]:
    """Das Paar (gamma, ridge) mit dem kleinsten Validierungsfehler (bei Gleichstand das erste im Raster) und die Fehlermatrix."""
    err = validation_errors(X, C, seed, gammas, ridges)
    i, j = np.unravel_index(int(np.argmin(err)), err.shape)   # argmin nimmt bei Gleichstand das erste (zeilenweise)
    return float(gammas[i]), float(ridges[j]), err


def fit_kernel_eto(X: np.ndarray, C: np.ndarray, seed: int, gammas=GAMMAS, ridges=ALPHAS) -> KernelModel:
    """Die zweistufige Regel mit flexibler Klasse: Parameter per Validierung wählen, dann auf allen Trainingsfällen anpassen.

    Bei n = 1 gibt es nichts zu validieren: das Modell ist die Konstante (der einzige Fall) und die Vorhersage immer dieselbe."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if len(X) < 2:
        return KernelModel(X, np.zeros((len(X), C.shape[1])), C.mean(axis=0), float(gammas[0]), float(ridges[-1]))
    g, a, _ = choose_parameters(X, C, seed, gammas, ridges)
    return fit_kernel_ridge(X, C, g, a)
