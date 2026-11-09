"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`zoi_bands.json`, `zoi_vormessung.json`).

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zoi_bands as B  # noqa: E402
import zoi_constants as C  # noqa: E402
from zoi_evaluation import de, de_pct  # noqa: E402


def _band(s: dict) -> str:
    return f"{de_pct(s['median'])} [{de_pct(s['min'])}; {de_pct(s['max'])}]"


def main_table(noise: float) -> list[str]:
    """Eigene Messung: je Grad und Datenmenge Median [Minimum; Maximum] des Regrets in Prozent für Orakel, KQ, SPO+, Kern-ETO und die Zählungen über 20 Seeds."""
    lines = ["| Grad | n | Orakel | KQ | SPO+ | Kern-ETO | SPO+ besser als KQ | Kern-ETO besser als KQ | Kern-ETO besser als SPO+ |", "|---|---|---|---|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for n in C.N_BAND:
            c = B.cell(deg, n, noise)
            s = c["summary"]
            k = "nicht gerechnet" if "regret_kern" not in s else _band(s["regret_kern"])
            kk = "–" if "regret_kern" not in s else f"{c['kern_besser_als_kq']} von {len(c['regret_kq'])}"
            ks = "–" if "regret_kern" not in s else f"{c['kern_besser_als_spo']} von {len(c['regret_kq'])}"
            lines.append(f"| {deg} | {n} | {de_pct(s['regret_orakel']['median'])} | {_band(s['regret_kq'])} | {_band(s['regret_spo'])} | {k} | {c['spo_besser_als_kq']} von {len(c['regret_kq'])} | {kk} | {ks} |")
    return lines


def stability_table() -> list[str]:
    sb = B.stabil()
    kq, sp, ke, orc = (np.array(sb[k]) for k in ("regret_kq", "regret_spo", "regret_kern", "regret_orakel"))
    lines = ["| Verfahren | Regret (Median über 100 Seeds) | besser als KQ | Median der Differenz zu KQ |", "|---|---|---|---|",
             f"| Orakel | {de_pct(np.median(orc))} % | – | – |",
             f"| KQ | {de_pct(np.median(kq))} % | – | – |",
             f"| SPO+ | {de_pct(np.median(sp))} % | {int((sp < kq).sum())} von {len(kq)} | {de(np.median(sp - kq), 2)} Punkte |",
             f"| Kern-ETO | {de_pct(np.median(ke))} % | {int((ke < kq).sum())} von {len(kq)} | {de(np.median(ke - kq), 2)} Punkte |"]
    return lines


def forest_table() -> list[str]:
    """Vormessung (scikit-learn, nicht live): Zufallswald, KQ und SPO+ (600 Schritte) zu Grad 6; n = 100 / 1000 / 5000."""
    lines = ["| n | Rauschstufe | Zufallswald | KQ | SPO+ (600 Schritte) |", "|---|---|---|---|---|"]
    for noise in C.NOISE_CHOICES:
        for n in (100, 1000, 5000):
            w = B.wald_cell(6, n, noise)
            lines.append(f"| {n} | {de(noise)} | {de(w['regret_wald']['median'])} % | {de(w['regret_kq']['median'])} % | {de(w['regret_spo600']['median'])} % |")
    return lines


def kernel_sklearn_table() -> list[str]:
    """Vormessung V1d (scikit-learn `KernelRidge`) neben dem Kern-ETO dieser Demo (numpy), Grad 6, n = 100 / 300 / 1000."""
    lines = ["| Rauschstufe | n | Kern-ETO Vormessung (scikit-learn) | Kern-ETO dieser Demo (numpy) | SPO+ Vormessung (300 Schritte) | SPO+ dieser Demo |", "|---|---|---|---|---|---|"]
    for noise in C.NOISE_CHOICES:
        for n in (100, 300, 1000):
            v = B.kern_sklearn_cell(n, noise)
            s = B.cell(6, n, noise)["summary"]
            lines.append(f"| {de(noise)} | {n} | {de_pct(v['regret_kern_sklearn']['median'])} % | {de_pct(s['regret_kern']['median'])} % | {de_pct(v['regret_spo300']['median'])} % | {de_pct(s['regret_spo']['median'])} % |")
    return lines


def gap_share_table() -> list[str]:
    lines = ["| Grad | Lücke ≤ 1 % | Lücke ≤ 2 % | Lücke ≤ 5 % | Lücke ≤ 10 % | Median der Lücke |", "|---|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        ra = B.luecke_teil_a(deg)
        sh = [float(np.median([r["anteil_leq"][k] for r in ra])) for k in ("0.01", "0.02", "0.05", "0.1")]
        med = float(np.median([r["quantile"]["0.5"] for r in ra]))
        lines.append(f"| {deg} | " + " | ".join(f"{de(100 * v, 0)} %" for v in sh) + f" | {de(100 * med, 2)} % |")
    return lines


def gap_thirds_table() -> list[str]:
    """Teil B der Vormessung V4: Fehlgriff, Regret je Fall und Anteil am Regret je Drittel der Lücke, Median über 20 Seeds (n = 1000)."""
    lines = ["| Grad | Rauschstufe | Verfahren | Fehlgriff (Lücke klein / mittel / groß) | Regret je Fall | Anteil am gesamten Regret |", "|---|---|---|---|---|---|"]
    names = {"kq": "KQ", "spo": "SPO+", "kern": "Kern-ETO"}
    for deg in (4, 6):
        for noise in (0.0, 0.5):
            rows = B.luecke_teil_b(deg, noise)
            for m in ("kq", "spo", "kern"):
                fg = [float(np.median([r["methods"][m]["fehlgriff"][i] for r in rows])) for i in range(3)]
                rg = [float(np.median([r["methods"][m]["regret_je_fall"][i] for r in rows])) for i in range(3)]
                an = [float(np.median([r["methods"][m]["regret_anteil"][i] for r in rows])) for i in range(3)]
                lines.append(f"| {deg} | {de(noise)} | {names[m]} | " + " / ".join(f"{de(100 * v, 0)} %" for v in fg) + " | " + " / ".join(f"{de_pct(v)} %" for v in rg)
                             + " | " + " / ".join(f"{de(100 * v, 0)} %" for v in an) + " |")
    return lines


def news_table(kind: str) -> list[str]:
    lines = ["| n | τ | KQ + empirisch | direkt (Quantilregression) | kNN-Gewichte | Kerngewichte | Wald-Gewichte (Vormessung) |", "|---|---|---|---|---|---|---|"]
    for n in C.NEWS_N_CHOICES:
        for tau in C.NEWS_TAUS:
            s = B.news_cell(kind, n, tau)["summary"]
            w = B.news_vormessung(kind, C.NEWS_P, n, tau)["wald"]
            lines.append(f"| {n} | {de(tau, 2)} | {_band(s['kq_emp'])} | {_band(s['direkt'])} | {_band(s['knn'])} | {_band(s['kern'])} | {_band(w)} |")
    return lines


def news_vormessung_table() -> list[str]:
    """Die Zellen der Vormessung V3, die der Plan nennt (quadratisch und linear, p = 3, n = 1000, τ = 0,95): KQ + empirisch, direkt, kNN, Wald."""
    lines = ["| Welt | KQ + empirisch | direkt | kNN-Gewichte | Wald-Gewichte |", "|---|---|---|---|---|"]
    for kind in C.NEWS_KINDS:
        v = B.news_vormessung(kind, C.NEWS_P, 1000, 0.95)
        lines.append(f"| {kind} | {de(v['kq_emp']['median'])} % | {de(v['direkt']['median'])} % | {de(v['knn']['median'])} % | {de(v['wald']['median'])} % |")
    return lines


if __name__ == "__main__":
    blocks = (("Raster ohne Rauschen", main_table(0.0)), ("Raster mit Rauschen", main_table(0.5)), ("Stabilität", stability_table()), ("Wald", forest_table()),
              ("Kern sklearn", kernel_sklearn_table()), ("Lücke Anteile", gap_share_table()), ("Lücke Drittel", gap_thirds_table()),
              ("Newsvendor quadratisch", news_table("quadratisch")), ("Newsvendor linear", news_table("linear")), ("Newsvendor Vormessung", news_vormessung_table()))
    for title, lines in blocks:
        print(f"### {title}")
        print("\n".join(lines))
        print()
