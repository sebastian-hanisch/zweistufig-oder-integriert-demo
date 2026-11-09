"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `zoi_bands.json` (eigene Messung, `tools/measure_bands.py`) und `zoi_vormessung.json`
(Vormessung mit scikit-learn, in dieser Demo NICHT live gerechnet; `tools/freeze_vormessung.py`)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "zoi_bands.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def vormessung() -> dict:
    return json.loads((HERE / "zoi_vormessung.json").read_text(encoding="utf-8"))


def cell(deg: int, n: int, noise: float) -> dict:
    """Eigene Messung einer Zelle (Grad, n, Rauschen): Listen je Seed, Zusammenfassungen und Zählungen. Kern-ETO (`regret_kern`) gibt es nur bis n = 1000."""
    for c in own()["cells"]:
        if (c["deg"], c["n"], c["noise"]) == (deg, n, noise):
            return c
    raise KeyError((deg, n, noise))


def wald_cell(deg: int, n: int, noise: float) -> dict | None:
    """Zelle der Vormessung (Zufallswald, KQ, SPO+ mit 600 Schritten); None, wo die Vormessung nicht gerechnet hat (n = 300)."""
    for c in vormessung()["gitter_wald"]:
        if (c["deg"], c["n"], c["noise"]) == (deg, n, noise):
            return c
    return None


def kern_sklearn_cell(n: int, noise: float) -> dict | None:
    """Zelle der Vormessung V1d (Grad 6, Kern-Ridge mit scikit-learn); None außerhalb von n = 100 / 300 / 1000."""
    for c in vormessung()["gitter_kern"]:
        if (c["n"], c["noise"]) == (n, noise):
            return c
    return None


def stabil() -> dict:
    return own()["stabil"]


def luecke_teil_a(deg: int) -> list[dict]:
    return [r for r in own()["luecke"]["teil_a"] if r["deg"] == deg]


def luecke_teil_b(deg: int, noise: float) -> list[dict]:
    return [r for r in own()["luecke"]["teil_b"] if (r["deg"], r["noise"]) == (deg, noise)]


def news_cell(kind: str, n: int, tau: float) -> dict:
    """Eigene Messung des Newsvendors (p = 3) für eine Welt, eine Datenmenge und ein kritisches Verhältnis."""
    for c in own()["news"]:
        if (c["welt"], c["n"], c["tau"]) == (kind, n, tau):
            return c
    raise KeyError((kind, n, tau))


def news_vormessung(kind: str, p: int, n: int, tau: float) -> dict | None:
    """Zelle der Vormessung V3 (Wald-Gewichte mit scikit-learn); None, wo nicht gerechnet."""
    for c in vormessung()["news"]:
        if (c["welt"], c["p"], c["n"], c["tau"]) == (kind, p, n, tau):
            return c
    return None


def median_of(rows: list[dict], key) -> float:
    return float(np.median([key(r) for r in rows]))
