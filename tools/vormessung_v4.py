"""Vormessung V4: Verteilung der Lücke Δ zwischen bestem und zweitbestem Weg im Gitter und ihr Zusammenhang mit der Regret-Lücke.

    python tools/vormessung_v4.py [--jobs 2] [--seeds 20] [--out tools/out_vormessung_v4.json]

Hintergrund: Die Rauschbedingung der Theorie (Hu, Kallus, Mao, Annahme 2) verlangt, dass die Lücke Δ(x) zwischen bestem und zweitbestem Weg unter den wahren Mittelkosten selten klein
ist. Diese Messung beantwortet vor dem Bau von Experiment 3 zwei Fragen an die Welt dieser Demo (5 x 5 Gitter, 70 Wege, p = 5 Merkmale, Kostenformel mit Grad deg):

  (A) Wie ist die relative Lücke Δ(x) / z*(m(x)) verteilt? Gerechnet wird sie durch Aufzählung der 70 Wege (Teil A braucht kein Training; Grad 1, 2, 4, 6; 20 Welten, je 2000 Testpunkte).
  (B) Hängt der Fehlgriff der Verfahren an der Lücke? Gerechnet wird für die Zellen Grad 4 und 6, Rauschstufe 0 und 0,5, n = 1000: je Testfall, ob das Verfahren einen anderen
      als den besten Weg wählt (der beste Weg unter den mittleren Kosten; bei Gleichstand im besten Wert zählt der Löser-Weg der mittleren Kosten) und sein Regret in Erwartung
      (mittlere Kosten, `zoi_gap.mean_regret_per_case`), getrennt nach Dritteln der Testfälle, geordnet nach relativer Lücke.

VORAB FESTGELEGTES KRITERIUM (vor dem ersten Lauf in diesem Docstring festgehalten): Experiment 3 trägt, wenn beides gilt:
  (1) Die Verteilung ist nicht entartet und hängt vom Grad ab: bei Grad 6 liegt der Anteil der Fälle mit relativer Lücke <= 5 % zwischen 5 % und 95 %, und er ist bei Grad 6 mindestens
      2 Prozentpunkte vom Anteil bei Grad 1 entfernt;
  (2) Der Fehlgriff nimmt mit der Lücke ab: in jeder der vier Zellen (Grad 4 und 6, Rauschen 0 und 0,5) und für jedes der drei Verfahren (KQ, SPO+, Kern-ETO) ist der Anteil gewählter
      Nicht-Bestwege im untersten Drittel der Lücke in mindestens 90 % der Seeds größer als im obersten Drittel.
Der Regret je Fall nach Dritteln wird berichtet, ist aber kein Kriterium (ein Fehlgriff bei kleiner Lücke kostet höchstens etwa die Lücke).
Trägt eines nicht, entfällt Experiment 3, und die Rauschbedingung bleibt ein Text mit der gemessenen Verteilung als Einordnung.

Zufall nur über SplitMix64 (`zoi_rng`); trainierte Zahlen sind plattformabhängig und werden nur als Bänder gelesen."""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # gleichzeitig laufende Prozesse: ein BLAS-Thread je Prozess (sonst Überbuchung der Kerne)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zoi_gap import (  # noqa: E402
    gap_to_second_best,
    mean_regret_per_case,
    relative_gap,
    share_leq,
)
from zoi_grid import Grid  # noqa: E402
from zoi_runs import METHODS, RunSpec, run_world  # noqa: E402
from zoi_world import TAG_TEST, WorldSpec, draw_data, draw_weights  # noqa: E402

DEGS = (1, 2, 4, 6)
DELTAS = (0.01, 0.02, 0.05, 0.10, 0.20)
N_TEST_A = 2000


def _part_a(args):
    deg, seed = args
    grid = Grid(5)
    W_all = grid.paths()
    spec = WorldSpec(p=5, deg=deg, noise=0.0)
    B = draw_weights(grid.d, spec, seed)
    _, _, M = draw_data(B, spec, N_TEST_A, seed, TAG_TEST)
    rel = relative_gap(W_all, M)
    return {
        "deg": deg, "seed": seed,
        "quantile": {str(q): float(np.quantile(rel, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "anteil_leq": {str(d): share_leq(rel, d) for d in DELTAS},
        "anteil_null": float(np.mean(rel == 0)),
    }


def _part_b(args):
    deg, noise, seed = args
    run = run_world(RunSpec(deg=deg, noise=noise, n=1000, seed=seed))
    grid = run.grid
    W_all = grid.paths()
    M = run.data.M_test
    rel = relative_gap(W_all, M)
    gap, best = gap_to_second_best(W_all, M)
    W_best, _ = grid.solve(M)
    order = np.argsort(rel, kind="stable")
    thirds = np.array_split(order, 3)           # unteres, mittleres, oberes Drittel der Lücke
    out = {"deg": deg, "noise": noise, "seed": seed, "rel_gap_third_median": [float(np.median(rel[t])) for t in thirds], "methods": {}}
    for m in METHODS:
        W, _ = grid.solve(run.chat[m])
        wrong = np.any(W != W_best, axis=1)
        reg = mean_regret_per_case(W_all, M, run.chat[m], grid) / np.maximum(best, 1e-12)
        out["methods"][m] = {
            "fehlgriff": [float(np.mean(wrong[t])) for t in thirds],
            "regret_je_fall": [float(100 * np.mean(reg[t])) for t in thirds],
            "regret_anteil": [float(np.sum(reg[t]) / max(np.sum(reg), 1e-12)) for t in thirds],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "tools" / "out_vormessung_v4.json"))
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t0 = time.time()
    with Pool(max(1, min(args.jobs, 2))) as pool:
        a = pool.map(_part_a, [(d, s) for d in DEGS for s in seeds], chunksize=4)
        b = pool.map(_part_b, [(d, e, s) for d in (4, 6) for e in (0.0, 0.5) for s in seeds], chunksize=1)
    Path(args.out).write_text(json.dumps({"teil_a": a, "teil_b": b, "seeds": seeds, "laufzeit_s": round(time.time() - t0)}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"Teil A: relative Lücke Δ / z* (Median über {len(seeds)} Welten; je Welt {N_TEST_A} Testpunkte)")
    print(f"{'Grad':>5} | " + " | ".join(f"Anteil <= {int(100 * d)} %" for d in DELTAS) + " | Anteil = 0 | Quantile 10/50/90 %")
    for d in DEGS:
        rows = [r for r in a if r["deg"] == d]
        sh = [np.median([r["anteil_leq"][str(x)] for r in rows]) for x in DELTAS]
        q = [np.median([r["quantile"][k] for r in rows]) for k in ("0.1", "0.5", "0.9")]
        print(f"{d:>5} | " + " | ".join(f"{100 * s:>11.1f} %" for s in sh) + f" | {100 * np.median([r['anteil_null'] for r in rows]):>8.1f} % | " + " / ".join(f"{100 * x:.2f}" for x in q))
    print("Teil B: je Drittel der Testfälle (nach relativer Lücke: unteres, mittleres, oberes), n = 1000")
    for d in (4, 6):
        for e in (0.0, 0.5):
            rows = [r for r in b if r["deg"] == d and r["noise"] == e]
            print(f"--- Grad {d}, Rauschen {e}: Median der Lücke je Drittel " + " / ".join(f"{100 * np.median([r['rel_gap_third_median'][i] for r in rows]):.2f} %" for i in range(3)))
            for m in METHODS:
                fg = np.array([r["methods"][m]["fehlgriff"] for r in rows])
                rg = np.array([r["methods"][m]["regret_je_fall"] for r in rows])
                an = np.array([r["methods"][m]["regret_anteil"] for r in rows])
                mono = int(np.sum(fg[:, 0] > fg[:, 2]))
                print(f"    {m:>5}: Fehlgriff je Drittel " + " / ".join(f"{100 * np.median(fg[:, i]):5.1f} %" for i in range(3))
                      + " | Regret je Fall " + " / ".join(f"{np.median(rg[:, i]):5.2f} %" for i in range(3))
                      + " | Anteil am Regret " + " / ".join(f"{100 * np.median(an[:, i]):4.0f} %" for i in range(3))
                      + f" | Fehlgriff unten > oben in {mono} von {len(rows)} Seeds")
    print(f"Laufzeit {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
