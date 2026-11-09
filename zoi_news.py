"""Experiment 4: der Newsvendor mit Merkmalen in der falschen Klasse (keine Streamlit-Abhängigkeit, numpy und scipy).

Dieselbe Frage wie im Gitter, in der kleinsten Welt: Welche Bestellregel gewinnt, wenn die lineare Modellklasse die Wahrheit trifft, und welche, wenn sie sie nicht trifft?

Welten (Nachfrage D = max(0, Y), Y ~ Normal(mu(x), 8²), Merkmale x ~ Normal(0, I_p), beta ein zufälliger Einheitsvektor):

* `linear`       mu(x) = 60 + 10 * (beta . x): die lineare Klasse ist richtig,
* `quadratisch`  mu(x) = 60 + 10 * (beta . x) + 8 * (x1² - 1): die lineare Klasse ist falsch.

Regeln (jede liefert eine Bestellmenge je Testfall; kritisches Verhältnis tau = b / (b + h), h = 1, b = tau / (1 - tau)):

* `kq_emp`  Kleinste Quadrate für den Mittelwert plus empirisches tau-Quantil der Residuen: zweistufig in der linearen Klasse,
* `direkt`  lineare Quantilregression zum Niveau tau (Pinball-Verlust als lineares Programm, `scipy.optimize.linprog`, HiGHS): auf den Kosten trainiert, dieselbe Klasse,
* `knn`     gewichtete Stichprobe mit den k nächsten Nachbarn, k per Leave-one-out,
* `kern`    gewichtete Stichprobe mit Gauß-Kerngewichten, Bandbreite per Leave-one-out.

Zufallswald-Gewichte werden hier NICHT gerechnet (kein scikit-learn zur Laufzeit); ihre Zahlen stehen nur als eingefrorene Vormessung in `zoi_vormessung.json`.

Die Kosten einer Regel werden exakt als Erwartung über die Nachfrage gerechnet (geschlossene Formel der gestutzten Normalverteilung) und über einen festen Testsatz von Merkmalen
gemittelt; es gibt kein Rauschen der Nachfrage im Test. Mehrkosten = (Kosten der Regel - Kosten des Orakels) / Kosten des Orakels in Prozent.

Herkunft: Welt, Kostenformel und LP-Quantilregression aus `newsvendor-direkt-demo` (`nvd_scenario.py`, `nvd_algorithm.py`), Gewichte, gewichtetes Quantil und
Leave-one-out aus `gewichtete-stichprobe-demo` (`gws_weights.py`, `gws_algorithm.py`), jeweils kopiert, gekürzt (nur die zwei Welten, ohne L1-Strafe) und mit dem Präfix zoi
geführt; der Zufall kommt aus `zoi_rng`. Die Einheiten sind in diesem Repo erneut getestet (`tests/test_news.py`, `tests/test_oracle_zweistufig_oder_integriert.py`)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtr, ndtri

from zoi_rng import SplitMix64, derive_seed

KINDS = ("linear", "quadratisch")
TAUS = (0.5, 0.8, 0.95)
RULES = ("kq_emp", "direkt", "knn", "kern")
MU0, MU_SLOPE, MU_QUAD, SIGMA = 60.0, 10.0, 8.0, 8.0
N_TEST = 2000
KGRID = (5, 10, 20, 40, 80, 160)          # Nachbarzahlen, Wahl per Leave-one-out (nur k <= n - 1)
HGRID = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)    # Bandbreiten für p = 3; gerechnet wird h * sqrt(p / 3)
LOO_TOL = 1e-12
TAG_BETA, TAG_X, TAG_Z, TAG_XT = 11, 12, 13, 14


@dataclass(frozen=True)
class NewsWorld:
    kind: str
    beta: tuple

    def mu(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        m = MU0 + MU_SLOPE * (X @ np.asarray(self.beta))
        if self.kind == "quadratisch":
            m = m + MU_QUAD * (X[:, 0] ** 2 - 1.0)
        return m

    def oracle_quantile(self, X: np.ndarray, tau: float) -> np.ndarray:
        """Optimale Bestellmenge: das tau-Quantil von D = max(0, Y), also max(0, mu + sigma * z_tau)."""
        return np.maximum(0.0, self.mu(X) + SIGMA * float(ndtri(tau)))


@dataclass(frozen=True)
class NewsData:
    world: NewsWorld
    X: np.ndarray     # Trainingsmerkmale (n, p)
    y: np.ndarray     # beobachtete Nachfragen (n,)
    Xt: np.ndarray    # feste Testmerkmale (N_TEST, p)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def p(self) -> int:
        return self.X.shape[1]


def make_world(kind: str, p: int, seed: int) -> NewsWorld:
    if kind not in KINDS:
        raise ValueError(f"unbekannte Welt: {kind}")
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    beta = SplitMix64(derive_seed(seed, TAG_BETA)).uniform(p) * 2.0 - 1.0
    norm = float(np.linalg.norm(beta))
    beta = beta / norm if norm > 0 else np.eye(p)[0]
    return NewsWorld(kind, tuple(float(b) for b in beta))


def make_news_data(kind: str, p: int, n: int, seed: int, n_test: int = N_TEST) -> NewsData:
    """Deterministisch aus (Welt, p, n, Seed); die Welt und der Testsatz hängen nicht von n ab."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    world = make_world(kind, p, seed)
    X = SplitMix64(derive_seed(seed, TAG_X)).normal(n * p).reshape(n, p)
    z = SplitMix64(derive_seed(seed, TAG_Z)).normal(n)
    y = np.maximum(0.0, world.mu(X) + SIGMA * z)
    Xt = SplitMix64(derive_seed(seed, TAG_XT)).normal(n_test * p).reshape(n_test, p)
    return NewsData(world, X, y, Xt)


def critical_costs(tau: float) -> tuple[float, float]:
    """Kosten (b, h) je fehlende und je überschüssige Einheit zum kritischen Verhältnis tau = b / (b + h) mit h = 1."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau muss zwischen 0 und 1 liegen (ohne die Grenzen)")
    return tau / (1.0 - tau), 1.0


def expected_cost_at(world: NewsWorld, X: np.ndarray, q: np.ndarray, b: float, h: float) -> np.ndarray:
    """Erwartete Kosten je Merkmalspunkt: b * E[(D - q)+] + h * E[(q - D)+] für D = max(0, Y), exakt.

    Für q >= 0 gilt (D - q)+ = (Y - q)+ und (q - D)+ = (q - Y)+ - (-Y)+; die Erwartungen der Normalverteilung sind
    E[(Y - q)+] = s phi(z) - (q - m)(1 - Phi(z)) und E[(q - Y)+] = s phi(z) + (q - m) Phi(z) mit z = (q - m) / s. Bestellmengen unter 0 gelten als 0."""
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    m = world.mu(X)
    s = SIGMA

    def phi(z):
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    z = (q - m) / s
    under = s * phi(z) - (q - m) * (1.0 - ndtr(z))
    over_y = s * phi(z) + (q - m) * ndtr(z)
    z0 = -m / s
    neg = s * phi(z0) - m * ndtr(z0)
    return b * under + h * (over_y - neg)


def expected_cost(world: NewsWorld, X: np.ndarray, q: np.ndarray, tau: float) -> float:
    b, h = critical_costs(tau)
    return float(np.mean(expected_cost_at(world, X, q, b, h)))


def extra_cost_percent(data: NewsData, q: np.ndarray, tau: float) -> float:
    """Mehrkosten der Bestellmengen q (auf den Testmerkmalen) gegen das Orakel in Prozent der Orakelkosten."""
    oracle = expected_cost(data.world, data.Xt, data.world.oracle_quantile(data.Xt, tau), tau)
    return 100.0 * (expected_cost(data.world, data.Xt, q, tau) - oracle) / oracle


# --- Lineare Regeln --------------------------------------------------------------------------------------------------------------------------

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt: Koeffizienten (c0, c1, ..., cp)."""
    return np.linalg.lstsq(add_intercept(X), y, rcond=None)[0]


def predict_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + np.atleast_2d(X) @ coef[1:]


def empirical_quantile(values: np.ndarray, tau: float) -> float:
    """Das ceil(tau * n)-kleinste Element (Rangregel)."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        raise ValueError("leere Stichprobe")
    rank = min(max(int(np.ceil(tau * len(v) - 1e-12)), 1), len(v))
    return float(v[rank - 1])


def quantreg_fit(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Lineare Quantilregression: min tau Σu + (1 - tau) Σv mit y - c0 - X c = u - v, u, v >= 0 (LP, HiGHS). Koeffizienten (c0, ..., cp).

    Bei Mehrdeutigkeit (etwa n <= p) liefert HiGHS eine von mehreren gleich guten Lösungen."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau muss zwischen 0 und 1 liegen (ohne die Grenzen)")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    cost = np.concatenate([np.zeros(p + 1), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    A = sparse.hstack([sparse.csr_matrix(add_intercept(X)), sparse.identity(n), -sparse.identity(n)], format="csr")
    bounds = [(None, None)] * (p + 1) + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Quantilregression nicht gelöst: {res.message}")
    return res.x[: p + 1]


def rule_kq_emp(data: NewsData, tau: float) -> np.ndarray:
    coef = ols_fit(data.X, data.y)
    shift = empirical_quantile(data.y - predict_linear(coef, data.X), tau)
    return predict_linear(coef, data.Xt) + shift


def rule_direkt(data: NewsData, tau: float) -> np.ndarray:
    return predict_linear(quantreg_fit(data.X, data.y, tau), data.Xt)


# --- Gewichtete Stichprobe -------------------------------------------------------------------------------------------------------------------

def sqdist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Quadratische euklidische Abstände (len(a), len(b)), elementweise Differenzen (gleiche Punkte haben exakt gleichen Abstand)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    out = np.empty((len(a), len(b)))
    for s in range(0, len(a), chunk):
        diff = a[s:s + chunk, None, :] - b[None, :, :]
        out[s:s + chunk] = (diff * diff).sum(axis=2)
    return out


def neighbour_order(d2: np.ndarray) -> np.ndarray:
    """Spaltenindizes je Zeile aufsteigend nach Abstand; bei Gleichstand die kleinere Nummer zuerst (stabile Sortierung)."""
    return np.argsort(d2, axis=1, kind="stable")


def knn_weights(order: np.ndarray, k: int, n_cols: int | None = None) -> np.ndarray:
    """Gewicht 1/k für die k nächsten Beobachtungen (k wird auf die verfügbaren Spalten begrenzt)."""
    m, avail = order.shape
    n_cols = avail if n_cols is None else n_cols
    if k < 1:
        raise ValueError("k muss mindestens 1 sein")
    k = min(k, avail)
    w = np.zeros((m, n_cols))
    w[np.arange(m)[:, None], order[:, :k]] = 1.0 / k
    return w


def kernel_weights(d2: np.ndarray, h: float) -> np.ndarray:
    """Gauß-Kerngewichte exp(-d²/(2h²)), je Zeile auf Summe 1 normiert; der kleinste Abstand wird vor dem Exponieren abgezogen (kein 0 / 0 bei kleiner Bandbreite)."""
    if h <= 0:
        raise ValueError("Bandbreite muss positiv sein")
    d2 = np.asarray(d2, dtype=float)
    ref = d2.min(axis=1, keepdims=True)
    w = np.exp(-0.5 * (d2 - ref) / (h * h))
    return w / w.sum(axis=1, keepdims=True)


def weighted_quantile(w: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    """Kleinstes D_(j), dessen kumuliertes Gewicht (nach Nachfrage sortiert, Gleichstände nach Nummer) tau erreicht; je Zeile von `w`."""
    w = np.atleast_2d(w)
    d = np.asarray(d, dtype=float)
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[:, order], axis=1)
    pos = (cum >= tau * cum[:, -1:] - LOO_TOL).argmax(axis=1)
    return d[order][pos]


def pinball(q: np.ndarray, d: np.ndarray, tau: float) -> float:
    """Mittlerer Pinball-Verlust max(tau (D - q), (tau - 1)(D - q)) = Newsvendor-Kosten bis auf den Faktor b + h."""
    r = np.asarray(d, dtype=float) - q
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def k_grid(n: int) -> list[int]:
    ks = [k for k in KGRID if k <= n - 1]
    return ks or [min(KGRID)]


def h_grid(p: int) -> list[float]:
    return [h * float(np.sqrt(p / 3.0)) for h in HGRID]


def _loo_matrix(X: np.ndarray) -> np.ndarray:
    d2 = sqdist(X, X)
    np.fill_diagonal(d2, np.inf)
    return d2


def loo_losses_knn(X: np.ndarray, y: np.ndarray, tau: float, ks: list[int]) -> list[float]:
    """Leave-one-out-Pinball-Verlust je k: jede Trainingsbeobachtung wird aus ihren Nachbarn ohne sich selbst vorhergesagt."""
    n = len(y)
    if n < 2:
        return [0.0 for _ in ks]
    order = neighbour_order(_loo_matrix(X))[:, : n - 1]
    return [pinball(weighted_quantile(knn_weights(order, k, n), y, tau), y, tau) for k in ks]


def loo_losses_kernel(X: np.ndarray, y: np.ndarray, tau: float, hs: list[float]) -> list[float]:
    n = len(y)
    if n < 2:
        return [0.0 for _ in hs]
    dd = _loo_matrix(X)
    return [pinball(weighted_quantile(kernel_weights(dd, h), y, tau), y, tau) for h in hs]


def rule_knn(data: NewsData, tau: float) -> tuple[np.ndarray, int]:
    """kNN-Gewichte, k aus KGRID per Leave-one-out; Rückgabe (Bestellmengen der Testfälle, gewähltes k)."""
    ks = k_grid(data.n)
    losses = loo_losses_knn(data.X, data.y, tau, ks)
    kb = ks[int(np.argmin(losses))]
    q = weighted_quantile(knn_weights(neighbour_order(sqdist(data.Xt, data.X)), kb, data.n), data.y, tau)
    return q, kb


def rule_kern(data: NewsData, tau: float) -> tuple[np.ndarray, float]:
    """Gauß-Kerngewichte, Bandbreite aus HGRID (für p umgerechnet) per Leave-one-out; Rückgabe (Bestellmengen, gewählte Bandbreite)."""
    hs = h_grid(data.p)
    losses = loo_losses_kernel(data.X, data.y, tau, hs)
    hb = hs[int(np.argmin(losses))]
    q = weighted_quantile(kernel_weights(sqdist(data.Xt, data.X), hb), data.y, tau)
    return q, hb


def run_news(kind: str, p: int, n: int, tau: float, seed: int) -> dict:
    """Eine Instanz: Mehrkosten in Prozent je Regel, dazu k und Bandbreite."""
    data = make_news_data(kind, p, n, seed)
    q_knn, k = rule_knn(data, tau)
    q_kern, h = rule_kern(data, tau)
    qs = {"kq_emp": rule_kq_emp(data, tau), "direkt": rule_direkt(data, tau), "knn": q_knn, "kern": q_kern}
    out = {"seed": seed, "k": k, "h": h}
    for name, q in qs.items():
        out[name] = extra_cost_percent(data, q, tau)
    return out
