"""Friert die Zahlen der Vormessungen in `zoi_vormessung.json` ein (Vormessung mit scikit-learn, in der App nicht live gerechnet).

    python tools/freeze_vormessung.py <Ordner mit den Rohausgaben> [--out zoi_vormessung.json]

Der Ordner ist `entscheidungsorientiert-planung/vormessung/`. Gelesen werden drei Rohausgaben:

* `out_eo_v1_sp.txt` (Gitter, Grad 1 bis 6, n = 100 / 1000 / 5000, Rauschen 0 und 0,5): Regret von KQ, SPO+ (dort 600 Adam-Schritte) und dem **Zufallswald** (100 Bäume, mindestens
  5 Fälle je Blatt, nicht abgestimmt, scikit-learn), 20 Seeds, 1000 Testfälle; Median, Minimum, Maximum in Prozent.
* `out_eo_v1d_kernel.txt` (Gitter, Grad 6, n = 100 / 300 / 1000): Regret von KQ, SPO+ (300 Schritte) und **Kern-ETO mit scikit-learn `KernelRidge`**, 20 Seeds. Dient als Gegenprobe
  zur eigenen Kern-ETO-Rechnung dieser Demo (`zoi_kernel.py`, numpy).
* `out_eo_v3_gewichtet.txt` (Newsvendor, Welten linear und quadratisch („nichtlinear“ in der Rohausgabe), p = 3 und 10, n = 100 und 1000, tau = 0,5 / 0,8 / 0,95): Mehrkosten von
  KQ + empirisch, direkt, kNN-Gewichten und **Wald-Gewichten** (50 Bäume, scikit-learn), 30 Seeds, 5000 Testfälle (Monte Carlo); Median, Minimum, Maximum in Prozent.

Zufallsquellen, Seeds und Welten der Vormessungen sind NICHT dieselben wie in dieser Demo: die Zahlen sind vergleichbar, nicht gleich."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRIPLE = r"\s*(-?\d+\.\d+)\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"


def _triple(g, i):
    return {"median": float(g[3 * i]), "min": float(g[3 * i + 1]), "max": float(g[3 * i + 2])}


def parse_v1(text: str) -> list[dict]:
    line = re.compile(r"^\s*(\d+)\s+(\d+)\s*\|" + r"\s*\|".join([TRIPLE] * 4) + r"\s*\|\s*(\d+)/(\d+)\s*\|\s*([\d.]+) / ([\d.]+) / ([\d.]+)\s*$")
    noise, cells = None, []
    for ln in text.splitlines():
        m = re.match(r"--- Rauschen e = ([\d.]+)", ln)
        if m:
            noise = float(m.group(1))
            continue
        m = line.match(ln)
        if not m or noise is None:
            continue
        g = m.groups()[2:14]
        cells.append({
            "deg": int(m.group(1)), "n": int(m.group(2)), "noise": noise,
            "regret_orakel": _triple(g, 0), "regret_kq": _triple(g, 1), "regret_spo600": _triple(g, 2), "regret_wald": _triple(g, 3),
            "spo600_besser_als_kq": int(m.group(15)), "seeds": int(m.group(16)),
            "mse_kq": float(m.group(17)), "mse_spo600": float(m.group(18)), "mse_wald": float(m.group(19)),
        })
    return cells


def parse_v1d(text: str) -> list[dict]:
    line = re.compile(r"^\s*(\d+)\s*\|" + r"\s*\|".join([TRIPLE] * 4) + r"\s*$")
    noise, cells = None, []
    for ln in text.splitlines():
        m = re.match(r"--- Welt deg 6, Rauschen e = ([\d.]+);", ln)
        if m:
            noise = float(m.group(1))
            continue
        m = line.match(ln)
        if not m or noise is None:
            continue
        g = m.groups()[1:13]
        cells.append({"deg": 6, "n": int(m.group(1)), "noise": noise, "seeds": 20,
                      "regret_orakel": _triple(g, 0), "regret_kq": _triple(g, 1), "regret_spo300": _triple(g, 2), "regret_kern_sklearn": _triple(g, 3)})
    return cells


def parse_v3(text: str) -> list[dict]:
    line = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s*\|" + r"\s*\|".join([TRIPLE] * 4) + r"\s*\|\s*(\d+)\s*$")
    kind, cells = None, []
    names = {"linear": "linear", "nichtlinear": "quadratisch"}
    for ln in text.splitlines():
        m = re.match(r"--- Welt (\S+)", ln)
        if m:
            kind = names.get(m.group(1))   # nur linear und nichtlinear (= quadratisch); die anderen Welten gehören zu Stück 5
            continue
        m = line.match(ln)
        if not m or kind is None:
            continue
        g = m.groups()[3:15]
        cells.append({"welt": kind, "p": int(m.group(1)), "n": int(m.group(2)), "tau": float(m.group(3)), "seeds": 30,
                      "kq_emp": _triple(g, 0), "direkt": _triple(g, 1), "knn": _triple(g, 2), "wald": _triple(g, 3), "k_median": int(m.group(16))})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(ROOT / "zoi_vormessung.json"))
    args = ap.parse_args()
    d = Path(args.folder)
    v1 = parse_v1((d / "out_eo_v1_sp.txt").read_text(encoding="utf-8"))
    v1d = parse_v1d((d / "out_eo_v1d_kernel.txt").read_text(encoding="utf-8"))
    v3 = parse_v3((d / "out_eo_v3_gewichtet.txt").read_text(encoding="utf-8"))
    if (len(v1), len(v1d), len(v3)) != (24, 6, 24):
        sys.exit(f"erwartet 24 / 6 / 24 Zellen, gefunden {len(v1)} / {len(v1d)} / {len(v3)}")
    out = {
        "quelle": "Vormessungen in entscheidungsorientiert-planung/vormessung/ (2026-11-09): out_eo_v1_sp.txt, out_eo_v1d_kernel.txt, out_eo_v3_gewichtet.txt",
        "hinweis": "Mit scikit-learn gerechnet (Zufallswald, KernelRidge), NICHT live in dieser Demo; Zufallsquellen und Seeds anders als in der Demo. Zahlen in Prozent (Median, Minimum, Maximum über Seeds).",
        "gitter_wald": v1, "gitter_kern": v1d, "news": v3,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(v1)} + {len(v1d)} + {len(v3)} Zellen nach {args.out}")


if __name__ == "__main__":
    main()
