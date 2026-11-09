"""Die Lücke Δ zwischen bestem und zweitbestem Weg (Rauschbedingung der Theorie) im Gitter (keine Streamlit-Abhängigkeit, nur numpy).

Hu, Kallus und Mao (*Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022, arXiv 2011.03030, Nummern nach arXiv v3, Volltext gelesen) bezeichnen mit
Δ(x) die Lücke zwischen dem besten und dem zweitbesten Extrempunkt unter den **wahren Mittelkosten** f*(x) und verlangen in ihrer Rauschbedingung (Annahme 2), dass
Δ(X) selten klein ist: P(0 < Δ(X) <= δ) <= (γ δ / B)^α. Im Gitter ist Δ messbar, denn die 70 Wege sind die Extrempunkte (Wegmatrix aus `zoi_grid.Grid.paths`).

Hier: Δ(x) = (zweitkleinster) - (kleinster) Wert von m(x)^T w über die 70 Wege w, absolut und relativ zum besten Wert z*(m(x)). Gleichstände im besten Wert liefern Δ = 0
(mehrere beste Wege); die Theorie schließt Δ = 0 aus ihrer Bedingung aus ("0 < Δ"), die Zahl der Fälle mit Δ = 0 wird deshalb getrennt gezählt (`share_leq`
mit `strict_positive`).

Die Rechnung ist die Aufzählung aller Wege (nicht der Löser): sie ist zugleich der zweite, unabhängige Rechenweg zum Löser (`tests/test_gap.py`)."""

import numpy as np


def path_values(W_all: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Werte m^T w jedes Weges (Wege x Bögen) für jede Zeile von M: Matrix (Fälle x Wege)."""
    return np.atleast_2d(np.asarray(M, dtype=float)) @ np.asarray(W_all, dtype=float).T


def gap_to_second_best(W_all: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rückgabe (Δ, z*): Lücke zwischen bestem und zweitbestem Weg und der beste Wert je Zeile von M (Aufzählung über W_all)."""
    values = path_values(W_all, M)
    if values.shape[1] < 2:
        raise ValueError("die Lücke braucht mindestens zwei Wege")
    two = np.partition(values, 1, axis=1)[:, :2]   # die zwei kleinsten Werte je Zeile (nicht sortiert)
    best = np.minimum(two[:, 0], two[:, 1])
    second = np.maximum(two[:, 0], two[:, 1])
    return second - best, best


def relative_gap(W_all: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Δ geteilt durch den besten Wert z*(m) (dimensionslos). Bei z* = 0 (alle Kosten 0) ist Δ = 0 und die relative Lücke 0."""
    gap, best = gap_to_second_best(W_all, M)
    return np.where(best > 0, gap / np.where(best > 0, best, 1.0), 0.0)


def share_leq(rel_gap: np.ndarray, delta: float, strict_positive: bool = False) -> float:
    """Anteil der Fälle mit relativer Lücke <= delta; mit `strict_positive` nur die Fälle mit 0 < Lücke <= delta (so steht die Bedingung der Theorie)."""
    g = np.asarray(rel_gap, dtype=float)
    hit = (g <= delta) & (g > 0) if strict_positive else (g <= delta)
    return float(np.mean(hit))


def mean_regret_per_case(W_all: np.ndarray, M: np.ndarray, Chat: np.ndarray, solver) -> np.ndarray:
    """Regret in Erwartung je Fall: m^T w(c_hat) - z*(m), absolut (>= 0). `solver.solve(Chat)` wählt die Wege; bewertet wird mit den mittleren Kosten m."""
    W, _ = solver.solve(Chat)
    _, best = gap_to_second_best(W_all, M)
    return np.sum(np.atleast_2d(M) * W, axis=1) - best
