"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Lauf einer Welt, vorgerechnete Bänder, Lücke Δ, Newsvendor.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`). Verglichen wird immer der Wert, wie er mit den angezeigten Stellen erscheint (`_shown`)."""

from dataclasses import dataclass

import numpy as np

import zoi_bands as B
import zoi_constants as C
from zoi_gap import mean_regret_per_case, relative_gap, share_leq
from zoi_runs import RunSpec


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


def _digits(value: float) -> int:
    """Stellen für Regret- und Mehrkostenwerte: zwei Stellen unter 1 (0,05 statt 0,1), sonst eine."""
    return 2 if abs(float(value)) < 1 else 1


def de_pct(value: float) -> str:
    return de(value, _digits(value))


def _shown_pct(value: float) -> float:
    """Der Wert so, wie `de_pct` ihn anzeigt."""
    return _shown(value, _digits(value))


@dataclass(frozen=True)
class Settings:
    deg: int = C.DEFAULT_DEG
    noise: float = C.DEFAULT_NOISE
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    news_kind: str = C.DEFAULT_NEWS_KIND
    news_tau: float = C.DEFAULT_NEWS_TAU
    news_n: int = C.DEFAULT_NEWS_N

    def run_spec(self) -> RunSpec:
        return RunSpec(deg=self.deg, noise=self.noise, n=self.n, seed=self.seed, p=C.N_FEATURES, h=C.GRID_H, steps=C.STEPS, lr=C.LR, n_test=C.N_TEST)


METHOD_NAMES = {
    "orakel": "Orakel (kennt die mittleren Kosten)",
    "kq": "KQ (linear, zweistufig)",
    "spo": "SPO+ (linear, Löser im Training)",
    "kern": "Kern-ETO (flexibel, zweistufig)",
    "wald": "Zufallswald (nicht live)",
}


# --- Experiment 1: drei Klassen, ein Gitter ------------------------------------------------------------------------------------------------

def class_rows(deg: int, n: int, noise: float) -> list[dict]:
    """Zeilen für eine Zelle: Orakel, KQ, SPO+, Kern-ETO (eigene Messung, Kern-ETO nur bis n = 1000) und Zufallswald (Vormessung, nur n = 100 / 1000 / 5000)."""
    c = B.cell(deg, n, noise)
    s = c["summary"]
    rows = [
        {"key": "orakel", "Verfahren": METHOD_NAMES["orakel"], "quelle": "eigene Messung", "regret": s["regret_orakel"], "mse": None},
        {"key": "kq", "Verfahren": METHOD_NAMES["kq"], "quelle": "eigene Messung", "regret": s["regret_kq"], "mse": s["mse_kq"]["median"]},
        {"key": "spo", "Verfahren": METHOD_NAMES["spo"], "quelle": "eigene Messung", "regret": s["regret_spo"], "mse": s["mse_spo"]["median"]},
    ]
    if "regret_kern" in s:
        rows.append({"key": "kern", "Verfahren": METHOD_NAMES["kern"], "quelle": "eigene Messung", "regret": s["regret_kern"], "mse": s["mse_kern"]["median"]})
    w = B.wald_cell(deg, n, noise)
    if w is not None:
        rows.append({"key": "wald", "Verfahren": METHOD_NAMES["wald"], "quelle": "Vormessung mit scikit-learn", "regret": w["regret_wald"], "mse": w["mse_wald"]})
    return rows


CLEAR_SEEDS_SHARE = 0.9   # „klar“: Kern-ETO muss in mindestens 90 % der Seeds unter KQ liegen, sonst heißt das Verdikt „kein klarer Gewinner“


def verdict_classes(r_kq: float, r_spo: float, r_kern: float | None, r_orakel: float, seeds: int,
                    kern_better_spo: int | None = None, kern_better_kq: int | None = None) -> tuple[str, str]:
    """Verdikt über die drei Sorten in einer Zelle (Median-Regret in Prozent über `seeds` feste Zufalls-Seeds; `r_kern` None heißt: Kern-ETO dort nicht gerechnet).

    Rangfolge nach den angezeigten Werten (zwei Stellen unter 1 %, sonst eine). Fälle ohne Kern-ETO: SPO+ unter KQ oder nicht. Mit Kern-ETO: alle drei gleich; die flexible Klasse liegt im Median unter beiden und in mindestens 90 % der Seeds unter KQ (Erfolg) beziehungsweise in weniger Seeds (kein klarer Gewinner); keine Alternative unter KQ; sonst liegt SPO+ unter KQ und nicht über Kern-ETO (Kern-ETO unter KQ und nicht unter SPO+ kann nicht vorkommen: dann läge SPO+ über KQ)."""
    scope = f"Median über {seeds} Zufalls-Seeds"
    kq, sp = _shown_pct(r_kq), _shown_pct(r_spo)
    a, b, o = de_pct(r_kq), de_pct(r_spo), de_pct(r_orakel)
    if r_kern is None:
        if sp < kq:
            return "info", f"SPO+ ({b} %) liegt unter KQ ({a} %), Orakel {o} % ({scope}); Kern-ETO ist bei dieser Datenmenge nicht gerechnet."
        return "info", f"SPO+ ({b} %) liegt nicht unter KQ ({a} %), Orakel {o} % ({scope}); Kern-ETO ist bei dieser Datenmenge nicht gerechnet."
    ke = _shown_pct(r_kern)
    c = de_pct(r_kern)
    extra = ""
    if kern_better_spo is not None and kern_better_kq is not None:
        extra = f" Kern-ETO war in {kern_better_spo} von {seeds} Seeds besser als SPO+ und in {kern_better_kq} von {seeds} besser als KQ."
    if kq == sp == ke:
        return "info", f"Alle drei Sorten liegen bei {a} % (Orakel {o} %; {scope}): hier gibt es nichts zu gewinnen."
    if ke < sp and ke < kq and kern_better_kq is not None and kern_better_kq < CLEAR_SEEDS_SHARE * seeds:
        return "info", (
            f"Kern-ETO ({c} %) liegt im Median unter SPO+ ({b} %) und unter KQ ({a} %), aber nicht in den meisten Seeds: kein klarer Gewinner; Orakel {o} % ({scope}).{extra}"
        )
    if ke < sp and ke < kq:
        return "success", (
            f"Die flexible Klasse (Kern-ETO, {c} %) liegt unter SPO+ ({b} %) und unter KQ ({a} %), obwohl sie den Löser im Training nicht benutzt; Orakel {o} % ({scope}).{extra}"
        )
    if min(sp, ke) >= kq:
        return "warning", f"Keine der beiden Alternativen liegt unter KQ ({a} %): SPO+ {b} %, Kern-ETO {c} %; Orakel {o} % ({scope}).{extra}"
    return "info", f"SPO+ ({b} %) liegt unter KQ ({a} %) und nicht über Kern-ETO ({c} %); Orakel {o} % ({scope}).{extra}"


def ranking_row(rows: list[dict]) -> str:
    """Name (Kurzform) des Verfahrens mit dem kleinsten Median-Regret unter KQ, SPO+, Kern-ETO (angezeigte Stelle; bei Gleichstand das erste in dieser Reihenfolge)."""
    cand = [r for r in rows if r["key"] in ("kq", "spo", "kern")]
    best = min(cand, key=lambda r: _shown_pct(r["regret"]["median"]))
    return best["Verfahren"].split(" (")[0]


# --- Experiment 2: Datenmenge ------------------------------------------------------------------------------------------------------------------

def grid_rows(deg: int) -> list[dict]:
    """Die acht Zellen (n = 100 / 300 / 1000 / 5000 mal Rauschen) zu einem Grad aus der eigenen Messung (Kern-ETO bis n = 1000)."""
    rows = []
    for noise in C.NOISE_CHOICES:
        for n in C.N_BAND:
            c = B.cell(deg, n, noise)
            s = c["summary"]
            rows.append({
                "n": n, "noise": noise, "orakel": s["regret_orakel"]["median"], "kq": s["regret_kq"], "spo": s["regret_spo"],
                "kern": s.get("regret_kern"), "besser": c["spo_besser_als_kq"], "seeds": len(c["regret_kq"]),
                "kern_besser_spo": c.get("kern_besser_als_spo"),
            })
    return rows


def spo_below_kq(row: dict) -> bool:
    """Liegt in der Zelle der Median-Regret von SPO+ unter dem von KQ (angezeigte Stelle)?"""
    return _shown_pct(row["spo"]["median"]) < _shown_pct(row["kq"]["median"])


def verdict_grid(deg: int, rows: list[dict]) -> tuple[str, str]:
    """Verdikt über die Zellen: in wie vielen liegt SPO+ unter KQ, in wie vielen Kern-ETO unter SPO+ (nur Zellen mit Kern-ETO), und die Ausnahmen von SPO+ unter KQ."""
    def lab(r):
        return f"n = {r['n']} {'mit Rauschen' if r['noise'] > 0 else 'ohne Rauschen'}"

    lower = [r for r in rows if spo_below_kq(r)]
    rest = [r for r in rows if not spo_below_kq(r)]
    with_kern = [r for r in rows if r["kern"] is not None]
    kern_lower = [r for r in with_kern if _shown_pct(r["kern"]["median"]) < _shown_pct(r["spo"]["median"])]
    head = (
        f"Bei Grad {deg} liegt der Median-Regret von SPO+ in {len(lower)} von {len(rows)} Zellen unter dem von KQ; "
        f"Kern-ETO liegt in {len(kern_lower)} von {len(with_kern)} Zellen (bis n = 1000) unter SPO+"
    )
    if not rest:
        return "info", head + "."
    details = "; ".join(f"{lab(r)}: SPO+ {de_pct(r['spo']['median'])} % gegen KQ {de_pct(r['kq']['median'])} % ({r['besser']} von {r['seeds']} Seeds mit SPO+ besser)" for r in rest)
    return "info", f"{head}. SPO+ nicht unter KQ in: {details}."


# --- Experiment 3: die Lücke Δ -----------------------------------------------------------------------------------------------------------------

def gap_table(run) -> dict:
    """Lücke Δ der Testfälle einer Welt und Fehlgriffe der Verfahren nach Dritteln der Lücke.

    Δ(x) = zweitbester minus bester Wegwert unter den mittleren Kosten m(x), relativ zum besten Wert. Die Testfälle werden nach Δ geordnet und in drei gleich große Drittel geteilt
    (unteres = kleinste Lücken). Je Verfahren: Anteil der Fälle, in denen der gewählte Weg nicht der beste Weg unter m ist (Fehlgriff), und der Regret in Erwartung
    (mittlere Kosten, `mean_regret_per_case`) je Fall in Prozent des besten Wertes. Rückgabe: Dictionary mit Δ je Fall, Anteilen und den Dritteln."""
    grid = run.grid
    W_all = grid.paths()
    M = run.data.M_test
    rel = relative_gap(W_all, M)
    W_best, z_best = grid.solve(M)
    order = np.argsort(rel, kind="stable")
    thirds = np.array_split(order, 3)
    out = {
        "rel": rel, "n": len(rel),
        "anteil_leq": {d: share_leq(rel, d) for d in C.GAP_DELTAS},
        "median_rel": float(np.median(rel)),
        "third_median_rel": [float(np.median(rel[t])) for t in thirds],
        "methods": {},
    }
    for m in ("kq", "spo", "kern"):
        W, _ = grid.solve(run.chat[m])
        wrong = np.any(W != W_best, axis=1)
        reg = mean_regret_per_case(W_all, M, run.chat[m], grid) / np.maximum(z_best, 1e-12)
        out["methods"][m] = {
            "fehlgriff": [float(np.mean(wrong[t])) for t in thirds],
            "fehlgriff_gesamt": float(np.mean(wrong)),
            "regret_je_fall": [100.0 * float(np.mean(reg[t])) for t in thirds],
        }
    return out


def verdict_gap(table: dict) -> tuple[str, str]:
    """Verdikt der Lücke in dieser Welt: Anteil kleiner Lücken und ob der Fehlgriff mit der Lücke abnimmt (unteres gegen oberes Drittel) für alle drei Verfahren."""
    share = table["anteil_leq"][0.01]
    meth = table["methods"]
    ratios = {m: (v["fehlgriff"][0], v["fehlgriff"][2]) for m, v in meth.items()}
    falls = all(_shown(100 * lo, 0) > _shown(100 * hi, 0) for lo, hi in ratios.values())   # verglichen wird, was angezeigt wird (ganze Prozent)
    names = {"kq": "KQ", "spo": "SPO+", "kern": "Kern-ETO"}
    detail = "; ".join(f"{names[m]} {de(100 * lo, 0)} % gegen {de(100 * hi, 0)} %" for m, (lo, hi) in ratios.items())
    head = f"In {de(100 * share, 0)} % der Testfälle liegt die Lücke zwischen bestem und zweitbestem Weg unter 1 % des besten Wertes (Median {de(100 * table['median_rel'], 2)} %). "
    if falls:
        return "info", head + f"Der Anteil gewählter Nicht-Bestwege ist im unteren Drittel der Lücke größer als im oberen: {detail}."
    return "warning", head + f"Der Anteil gewählter Nicht-Bestwege nimmt hier nicht für alle Verfahren mit der Lücke ab (unteres gegen oberes Drittel: {detail})."


# --- Experiment 4: Newsvendor in der falschen Klasse ---------------------------------------------------------------------------------------

NEWS_NAMES = {
    "kq_emp": "KQ + empirisches Quantil (linear, zweistufig)",
    "direkt": "direkt: lineare Quantilregression (linear, auf den Kosten)",
    "knn": "kNN-Gewichte (ohne Modell)",
    "kern": "Kerngewichte (ohne Modell)",
    "wald": "Wald-Gewichte (nicht live)",
}


def news_rows(kind: str, n: int, tau: float) -> list[dict]:
    """Zeilen für Experiment 4: Mehrkosten gegen das Orakel in Prozent (Median, Minimum, Maximum über 20 Seeds, eigene Messung) und Wald-Gewichte aus der Vormessung (30 Seeds, nicht live)."""
    c = B.news_cell(kind, n, tau)
    rows = [{"key": k, "Verfahren": NEWS_NAMES[k], "quelle": "eigene Messung", "extra": c["summary"][k]} for k in ("kq_emp", "direkt", "knn", "kern")]
    v = B.news_vormessung(kind, C.NEWS_P, n, tau)
    if v is not None:
        rows.append({"key": "wald", "Verfahren": NEWS_NAMES["wald"], "quelle": "Vormessung mit scikit-learn", "extra": v["wald"]})
    return rows


def verdict_news(kind: str, n: int, tau: float, rows: list[dict]) -> tuple[str, str]:
    """Verdikt: welche Regel hat in dieser Zelle die kleinsten Median-Mehrkosten; ob die lineare Klasse (KQ + empirisch, direkt) oder die Gewichte vorn liegen."""
    by = {r["key"]: _shown_pct(r["extra"]["median"]) for r in rows}
    weights = {k: v for k, v in by.items() if k in ("knn", "kern", "wald")}
    linear = {k: v for k, v in by.items() if k in ("kq_emp", "direkt")}
    best_w = min(weights, key=weights.get)
    best_l = min(linear, key=linear.get)
    short = {"kq_emp": "KQ + empirisch", "direkt": "direkt", "knn": "kNN-Gewichte", "kern": "Kerngewichte", "wald": "Wald-Gewichte"}
    scope = f"Welt {kind}, n = {n}, τ = {de(tau, 2)}, Median über 20 Seeds (Wald: 30 Seeds der Vormessung)"
    d, k = by["direkt"], by["kq_emp"]
    if d < k:
        within = f"Direktes Lernen in der linearen Klasse senkt die Mehrkosten gegen KQ + empirisch von {de_pct(k)} % auf {de_pct(d)} %."
    elif d == k:
        within = f"Direktes Lernen in der linearen Klasse ändert die Mehrkosten gegen KQ + empirisch nicht ({de_pct(d)} %)."
    else:
        within = f"Direktes Lernen in der linearen Klasse senkt die Mehrkosten gegen KQ + empirisch nicht ({de_pct(d)} % gegen {de_pct(k)} %)."
    w_txt = f"{short[best_w]} {de_pct(weights[best_w])} %"
    l_txt = f"{short[best_l]} {de_pct(linear[best_l])} %"
    if weights[best_w] < linear[best_l]:
        return "success", f"Hier liegen die Gewichte vorn: {w_txt} gegen {l_txt} Mehrkosten. {within} ({scope})."
    if weights[best_w] == linear[best_l]:
        return "info", f"Gewichte und lineare Klasse liegen gleichauf: {w_txt} gegen {l_txt} Mehrkosten. {within} ({scope})."
    return "warning", f"Hier liegt die lineare Klasse vorn: {l_txt} gegen {w_txt} Mehrkosten der Gewichte (sie zahlen den Preis ohne Modellannahme). {within} ({scope})."
