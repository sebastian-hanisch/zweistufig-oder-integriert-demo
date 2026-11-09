"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `zoi_bands.json` (feste Zufalls-Seeds 0 bis 19).

    python tools/measure_bands.py [--jobs 2] [--out zoi_bands.json]

Inhalt:
(1) Raster Grad (1, 2, 4, 6) x Datenmenge (100, 300, 1000, 5000) x Rauschen (0, 0,5), 20 Seeds: normierter Regret in Prozent für Orakel, KQ, SPO+ (Adam, 300 Schritte, Lernrate 0,05,
    Start bei KQ) und Kern-ETO (Gauß-Kern-Ridge, Parameter per Validierung) sowie der mittlere Quadratfehler je Bogen. Kern-ETO nur bis n = 1000 (bei n = 5000 ist die Kernmatrix zu groß
    für eine Demo; die Spalte fehlt dort und die App sagt es).
(2) Stabilitätsprobe: Grad 6, n = 100, Rauschen 0,5, 100 weitere Seeds (20 bis 119), alle vier Verfahren.
(3) Lücke Δ (Experiment 3): Teil A und B der Vormessung V4 (`tools/vormessung_v4.py`), 20 Seeds.
(4) Newsvendor (Experiment 4): Welten linear und quadratisch, p = 3, n = 100 und 1000, tau = 0,5 / 0,8 / 0,95, 20 Seeds: Mehrkosten gegen das Orakel in Prozent für KQ + empirisch,
    direkte Quantilregression, kNN- und Kerngewichte (Zufallswald nur als eingefrorene Vormessung in `zoi_vormessung.json`).

Zufall nur über den ganzzahligen SplitMix64 (`zoi_rng`). Die Zahlen sind trainiert beziehungsweise validiert und plattformabhängig; Tests prüfen sie nur als Bänder und Ungleichungen.
Nie mehr als zwei Prozesse (`--jobs`)."""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # ein BLAS-Thread je Prozess: bei mehreren gleichzeitigen Prozessen sonst Überbuchung der Kerne

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import vormessung_v4 as V4  # noqa: E402

from zoi_news import run_news  # noqa: E402
from zoi_runs import RunSpec, cell_result, run_world  # noqa: E402

SEEDS = tuple(range(20))
STAB_SEEDS = tuple(range(20, 120))
DEGS, NS, NOISES = (1, 2, 4, 6), (100, 300, 1000, 5000), (0.0, 0.5)
NEWS_KINDS, NEWS_NS, NEWS_TAUS, NEWS_P = ("linear", "quadratisch"), (100, 1000), (0.5, 0.8, 0.95), 3
KERNEL_MAX_N = 1000


def _grid_cell(args):
    deg, n, noise, seed = args
    run = run_world(RunSpec(deg=deg, noise=noise, n=n, seed=seed, with_kernel=n <= KERNEL_MAX_N))
    return (deg, n, noise, seed, cell_result(run))


def _stab_cell(seed):
    run = run_world(RunSpec(deg=6, noise=0.5, n=100, seed=seed))
    return (seed, cell_result(run))


def _news_cell(args):
    kind, n, tau, seed = args
    return (kind, n, tau, seed, run_news(kind, NEWS_P, n, tau, seed))


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def _r(values, digits=4):
    return [round(float(v), digits) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "zoi_bands.json"))
    args = ap.parse_args()
    t0 = time.time()
    with Pool(max(1, min(args.jobs, 2))) as pool:
        cells = sorted([(deg, n, e, s) for deg in DEGS for n in NS for e in NOISES for s in SEEDS], key=lambda c: -c[1])   # teuerste zuerst
        grid_res = pool.map(_grid_cell, cells, chunksize=4)
        stab_res = pool.map(_stab_cell, STAB_SEEDS, chunksize=5)
        luecke_a = pool.map(V4._part_a, [(d, s) for d in V4.DEGS for s in SEEDS], chunksize=4)
        luecke_b = pool.map(V4._part_b, [(d, e, s) for d in (4, 6) for e in (0.0, 0.5) for s in SEEDS], chunksize=1)
        news_res = pool.map(_news_cell, [(k, n, t, s) for k in NEWS_KINDS for n in NEWS_NS for t in NEWS_TAUS for s in SEEDS], chunksize=2)

    out = {
        "meta": {
            "beschreibung": "Eigene Messung dieser Demo (tools/measure_bands.py): 5 x 5 Gitter, 5 Merkmale, 1000 Testfälle, Zufalls-Seeds 0 bis 19; KQ und SPO+ (Adam, Lernrate 0,05, 300 Schritte, Start bei KQ) "
                            "in der linearen Klasse, Kern-ETO = Gauß-Kern-Ridge mit Validierung (nur bis n = 1000).",
            "seeds": list(SEEDS), "steps": 300, "lr": 0.05, "n_test": 1000, "kernel_max_n": KERNEL_MAX_N,
            "einheiten": "Regret in Prozent der besten Wegkosten (Summe über die Testfälle); mse = mittlerer quadratischer Fehler je Bogen; Newsvendor: Mehrkosten gegen das Orakel in Prozent",
            "laufzeit_s": round(time.time() - t0),
        },
        "cells": [],
        "stabil": {},
        "luecke": {"teil_a": luecke_a, "teil_b": luecke_b, "seeds": list(SEEDS)},
        "news": [],
    }
    for deg in DEGS:
        for n in NS:
            for e in NOISES:
                rows = sorted([r for (d, nn, ee, s, r) in grid_res if (d, nn, ee) == (deg, n, e)], key=lambda r: r["seed"])
                assert len(rows) == len(SEEDS)
                keys = ["regret_orakel", "regret_kq", "regret_spo", "mse_kq", "mse_spo"] + (["regret_kern", "mse_kern"] if n <= KERNEL_MAX_N else [])
                cell = {"deg": deg, "n": n, "noise": e, **{k: _r([r[k] for r in rows]) for k in keys}}
                cell["summary"] = {k: _summary(cell[k]) for k in keys}
                cell["spo_besser_als_kq"] = int(sum(r["regret_spo"] < r["regret_kq"] for r in rows))
                if n <= KERNEL_MAX_N:
                    cell["kern_besser_als_spo"] = int(sum(r["regret_kern"] < r["regret_spo"] for r in rows))
                    cell["kern_besser_als_kq"] = int(sum(r["regret_kern"] < r["regret_kq"] for r in rows))
                out["cells"].append(cell)
    rows = sorted(stab_res)
    out["stabil"] = {"deg": 6, "n": 100, "noise": 0.5, "seeds": [s for s, _ in rows],
                     **{k: _r([r[k] for _, r in rows]) for k in ("regret_kq", "regret_spo", "regret_kern", "regret_orakel")}}
    for kind in NEWS_KINDS:
        for n in NEWS_NS:
            for tau in NEWS_TAUS:
                rows = sorted([r for (k, nn, t, s, r) in news_res if (k, nn, t) == (kind, n, tau)], key=lambda r: r["seed"])
                cell = {"welt": kind, "p": NEWS_P, "n": n, "tau": tau, **{k: _r([r[k] for r in rows], 3) for k in ("kq_emp", "direkt", "knn", "kern")}}
                cell["summary"] = {k: _summary(cell[k]) for k in ("kq_emp", "direkt", "knn", "kern")}
                out["news"].append(cell)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
