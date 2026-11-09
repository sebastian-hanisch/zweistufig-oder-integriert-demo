"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test:

| README | Test |
|---|---|
| alle erzeugten Tabellen (Raster ohne und mit Rauschen, Stabilitätsprobe, Zufallswald, Kern-Ridge der Vormessung, Lücke, Newsvendor) | `test_readme_contains_every_generated_table_line` (wörtlich aus den JSON-Dateien erzeugt) |
| Messung dieser Demo (Seeds, Schritte, Lernrate, Zellen) | `test_band_file_describes_the_documented_setup` |
| Klasse passt (Grad 1) | `test_class_fits_the_linear_truth` |
| Klasse passt nicht, ohne Rauschen (Grad 4 und 6) | `test_wrong_class_without_noise` |
| Klasse passt nicht, mit Rauschen (Grad 6), Lücke zum Orakel 31 % / 51 % | `test_wrong_class_with_noise_and_the_gap_to_the_oracle` |
| Datenmenge (7 von 8, 6 von 6) | `test_data_amount_at_degree_six` |
| Stabilitätsprobe (n = 100, Rauschen 0,5, 100 Seeds) | `test_stability_probe_small_data_with_noise` |
| Verteilung der Lücke (Anteile, Kriterium V4) | `test_gap_distribution_and_the_v4_criterion` |
| Fehlgriff nach Dritteln der Lücke (12 Kombinationen), Kern-ETO 73 % / KQ 24 % / 36 % | `test_mismatch_falls_with_the_gap_in_every_combination` |
| Newsvendor (falsche und richtige Klasse) | `test_newsvendor_wrong_class`, `test_newsvendor_right_class` |
| Zahlen der Vormessungen (V1 Wald, V1d Kern-Ridge, V3 Newsvendor), Abweichung der Kern-Ridge-Spalten | `test_frozen_pre_measurements_hold_the_numbers_of_the_raw_outputs` |
| Korrekturen gegenüber dem Plan | `test_plan_numbers_versus_this_measurement` |
| exakt lineare, rauschfreie Welt | `test_exactly_linear_noiseless_world` |
| Orakel-Vergleiche der Scratch-Läufe (300 / 200 / 100 Instanzen) | nicht im Repo: Messung dieser README; im Repo die kleinen Fassungen in den Orakel-Tests |
| Rohausgabe V4 (`tools/out_vormessung_v4.txt`) | `test_v4_raw_output_agrees_with_the_bands_in_structure` (Struktur und Eckwerte) |

Trainierte und validierte Zahlen hängen von der Plattform ab: gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur Rechnungen ohne Training (`test_kernel.py`, `test_gap.py`, `test_news.py`).
"""

from pathlib import Path

import numpy as np
import pytest
import readme_tables as T

import zoi_bands as B
import zoi_constants as C
import zoi_evaluation as ev
from zoi_evaluation import _shown_pct, de, de_pct
from zoi_runs import RunSpec, run_world

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _s(deg, n, noise, key):
    return B.cell(deg, n, noise)["summary"][key]["median"]


def _shown_less(a, b):
    return _shown_pct(a) < _shown_pct(b)


def test_readme_contains_every_generated_table_line():
    blocks = [T.main_table(0.0), T.main_table(0.5), T.stability_table(), T.forest_table(), T.kernel_sklearn_table(), T.gap_share_table(), T.gap_thirds_table(),
              T.news_table("quadratisch"), T.news_table("linear"), T.news_vormessung_table()]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_band_file_describes_the_documented_setup():
    meta = B.own()["meta"]
    assert meta["seeds"] == list(C.BAND_SEEDS) and meta["steps"] == C.STEPS and meta["lr"] == C.LR and meta["n_test"] == C.N_TEST and meta["kernel_max_n"] == C.KERNEL_MAX_N
    assert len(B.own()["cells"]) == len(C.DEG_CHOICES) * len(C.N_BAND) * len(C.NOISE_CHOICES) == 32
    assert all(len(c["regret_kq"]) == 20 for c in B.own()["cells"])
    assert len(B.own()["news"]) == 12 and all(len(c["knn"]) == 20 for c in B.own()["news"])
    assert len(B.stabil()["seeds"]) == 100 and B.stabil()["seeds"][0] == 20 and B.stabil()["seeds"][-1] == 119
    for text in ("Zufalls-Seeds 0 bis 19", "300 Schritte", "Lernrate 0,05", "1000 Testfälle", "2000 feste Testmerkmale"):
        assert text in README, text


def test_class_fits_the_linear_truth():
    for n in (100, 300, 1000):
        c = B.cell(1, n, 0.0)
        assert [de_pct(c["summary"][k]["median"]) for k in ("regret_kq", "regret_spo", "regret_kern")] == ["0,00"] * 3
        assert c["spo_besser_als_kq"] == 0 and c["kern_besser_als_kq"] == 0
    c5000 = B.cell(1, 5000, 0.0)
    assert de_pct(c5000["summary"]["regret_kq"]["median"]) == "0,00" and de_pct(c5000["summary"]["regret_spo"]["median"]) == "0,00" and c5000["spo_besser_als_kq"] == 0
    shown = {}
    for n in (100, 300, 1000):
        kq, sp, ke = (_s(1, n, 0.5, k) for k in ("regret_kq", "regret_spo", "regret_kern"))
        assert _shown_pct(kq) <= _shown_pct(ke) and _shown_pct(kq) < _shown_pct(sp)                  # KQ nicht über Kern-ETO, unter SPO+ (angezeigte Stelle)
        shown[n] = (de_pct(kq), de_pct(ke), de_pct(sp))
    assert shown == {100: ("19,6", "19,6", "20,5"), 300: ("18,3", "18,5", "18,9"), 1000: ("18,1", "18,2", "18,3")}
    assert de_pct(_s(1, 5000, 0.5, "regret_kq")) == de_pct(_s(1, 5000, 0.5, "regret_spo")) == "17,8"
    for text in ("19,6 gegen 19,6 und 20,5", "18,3 gegen 18,5 und 18,9", "18,1 gegen 18,2 und 18,3", "bei n = 5000 sind KQ und SPO+ gleich (17,8 %)"):
        assert text in README, text


def test_wrong_class_without_noise():
    for deg in (4, 6):
        for n in (100, 300, 1000):
            kq, sp, ke = (_s(deg, n, 0.0, k) for k in ("regret_kq", "regret_spo", "regret_kern"))
            assert ke < sp < kq, (deg, n)
            assert _shown_less(ke, sp) and _shown_less(sp, kq)
    c = B.cell(6, 1000, 0.0)
    assert (de_pct(c["summary"]["regret_kern"]["median"]), de_pct(c["summary"]["regret_spo"]["median"]), de_pct(c["summary"]["regret_kq"]["median"]), de(c["summary"]["regret_kq"]["median"], 2)) == ("0,05", "0,51", "2,0", "1,96")
    assert c["kern_besser_als_spo"] == 20 and c["kern_besser_als_kq"] == 20
    assert c["summary"]["regret_orakel"]["median"] == pytest.approx(0.0, abs=1e-9)
    for text in ("Median-Regret von Kern-ETO bei 0,05 %, von SPO+ bei 0,51 %, von KQ bei 1,96 %", "Kern-ETO 0,05 %, SPO+ 0,51 %, KQ 2,0 %", "in 20 von 20 Seeds besser als SPO+"):
        assert text in README, text


def test_wrong_class_with_noise_and_the_gap_to_the_oracle():
    c = B.cell(6, 1000, 0.5)
    kq, sp, ke, orc = (c["summary"][k]["median"] for k in ("regret_kq", "regret_spo", "regret_kern", "regret_orakel"))
    assert (de_pct(ke), de_pct(sp), de_pct(kq), de_pct(orc)) == ("11,8", "12,3", "13,1", "10,5")
    assert round(100 * (kq - sp) / (kq - orc)) == 31 and round(100 * (kq - ke) / (kq - orc)) == 51
    assert c["spo_besser_als_kq"] == 20 and c["kern_besser_als_kq"] == 20
    assert "SPO+ schließt 31 % der Lücke zwischen KQ und Orakel, Kern-ETO 51 %" in README


def test_data_amount_at_degree_six():
    cells = [B.cell(6, n, e) for e in C.NOISE_CHOICES for n in C.N_BAND]
    below = [c for c in cells if _shown_less(c["summary"]["regret_spo"]["median"], c["summary"]["regret_kq"]["median"])]
    assert len(below) == 7 and [c for c in cells if c not in below] == [B.cell(6, 100, 0.5)]
    assert (de_pct(_s(6, 100, 0.5, "regret_spo")), de_pct(_s(6, 100, 0.5, "regret_kq"))) == ("14,4", "14,0")
    with_kern = [c for c in cells if "regret_kern" in c["summary"]]
    assert len(with_kern) == 6 and all(c["summary"]["regret_kern"]["median"] < c["summary"]["regret_spo"]["median"] for c in with_kern)
    for text in ("SPO+ liegt in 7 von 8 Zellen", "(14,4 % gegen 14,0 %)", "Kern-ETO liegt in allen 6 Zellen mit Kern-ETO (n ≤ 1000) unter SPO+"):
        assert text in README, text


def test_stability_probe_small_data_with_noise():
    sb = B.stabil()
    kq, sp, ke = (np.array(sb[k]) for k in ("regret_kq", "regret_spo", "regret_kern"))
    assert (int((sp < kq).sum()), int((ke < kq).sum())) == (29, 64)
    assert (de(np.median(sp - kq), 2), de(np.median(ke - kq), 2)) == ("0,53", "-0,45")
    assert float(np.median(ke - kq)) > -0.5 and int((ke >= kq).sum()) == 36
    c = B.cell(6, 100, 0.5)
    assert (c["spo_besser_als_kq"], c["kern_besser_als_kq"]) == (10, 16)
    assert (de_pct(c["summary"]["regret_kern"]["median"]), de_pct(c["summary"]["regret_kq"]["median"])) == ("13,8", "14,0")
    for text in ("in 10 und Kern-ETO in 16 von 20 Seeds besser als KQ", "SPO+ in 29 und Kern-ETO in 64 von 100 Seeds unter KQ", "0,53 Punkte darüber beziehungsweise 0,45 Punkte darunter",
                 "in 36 von 100 Seeds kein Vorsprung", "(29 von 100 für SPO+)"):
        assert text in README, text


def test_gap_distribution_and_the_v4_criterion():
    share = {deg: {k: float(np.median([r["anteil_leq"][k] for r in B.luecke_teil_a(deg)])) for k in ("0.01", "0.05")} for deg in C.DEG_CHOICES}
    assert 0.05 <= share[6]["0.05"] <= 0.95 and share[1]["0.05"] - share[6]["0.05"] >= 0.02           # Kriterium (1)
    assert (round(100 * share[6]["0.01"]), round(100 * share[1]["0.01"])) == (51, 85)
    assert all(len(B.luecke_teil_a(deg)) == 20 for deg in C.DEG_CHOICES)
    med = {deg: float(np.median([r["quantile"]["0.5"] for r in B.luecke_teil_a(deg)])) for deg in C.DEG_CHOICES}
    assert med[1] < med[2] < med[4] < med[6]                                                          # die Lücke wächst mit dem Grad
    ties = [r["anteil_null"] for r in B.luecke_teil_a(1) if r["anteil_null"] > 0]
    assert len(ties) == 5 and (de(100 * min(ties), 2), de(100 * max(ties), 1)) == ("0,45", "40,3")          # Gleichstand im besten Wert: nur bei Grad 1
    assert all(r["anteil_null"] == 0.0 for dg in (2, 4, 6) for r in B.luecke_teil_a(dg))
    for text in ("in 51 % (Grad 6) bis 85 % (Grad 1) der Fälle kleiner als 1 % des besten Wertes", "zwischen 5 % und 95 % liegt", "in 5 von 20 Welten bei 0,45 % bis 40,3 % der Testpunkte"):
        assert text in README, text


def test_mismatch_falls_with_the_gap_in_every_combination():
    combos = 0
    for deg in (4, 6):
        for noise in (0.0, 0.5):
            rows = B.luecke_teil_b(deg, noise)
            assert len(rows) == 20
            for m in ("kq", "spo", "kern"):
                wins = sum(r["methods"][m]["fehlgriff"][0] > r["methods"][m]["fehlgriff"][2] for r in rows)
                assert wins >= 18, (deg, noise, m)                                                    # Kriterium (2): mindestens 90 % der Seeds
                assert wins == 20, (deg, noise, m)                                                    # gemessen: alle 20
                combos += 1
    assert combos == 12
    rows = B.luecke_teil_b(6, 0.0)
    med = lambda m, key, i: float(np.median([r["methods"][m][key][i] for r in rows]))  # noqa: E731
    assert round(100 * med("kern", "regret_anteil", 0)) == 73 and round(100 * med("kq", "regret_anteil", 0)) == 24
    assert round(100 * med("kq", "fehlgriff", 2)) == 36 and round(100 * med("kern", "fehlgriff", 2)) == 0
    for text in ("In allen 12 Kombinationen", "in 20 von 20 Seeds größer als im obersten", "fallen 73 % des Regrets von Kern-ETO ins unterste Drittel der Lücke, bei KQ 24 %",
                 "wählt KQ in 36 % der Fälle einen anderen als den besten Weg, Kern-ETO in 0 %"):
        assert text in README, text


def test_newsvendor_wrong_class():
    quad = {(n, t): B.news_cell("quadratisch", n, t)["summary"] for n in C.NEWS_N_CHOICES for t in C.NEWS_TAUS}
    for key, s in quad.items():
        assert max(s["knn"]["median"], s["kern"]["median"]) < min(s["kq_emp"]["median"], s["direkt"]["median"]), key
    s = quad[(1000, 0.95)]
    assert [de(s[k]["median"]) for k in ("knn", "kern", "kq_emp", "direkt")] == ["30,1", "28,7", "140,7", "141,1"]
    diffs = {key: s["direkt"]["median"] - s["kq_emp"]["median"] for key, s in quad.items()}
    assert max(abs(v) for v in diffs.values()) == pytest.approx(7.1, abs=0.05) and max(diffs, key=lambda k: abs(diffs[k])) == (100, 0.95) and diffs[(100, 0.95)] > 0
    below = {k: v for k, v in diffs.items() if v < 0}
    assert len(below) == 3 and max(abs(v) for v in below.values()) == pytest.approx(1.1, abs=0.05)         # (100; 0,5), (1000; 0,5), (1000; 0,8)
    for text in ("kNN- und Kerngewichte in allen 6 Zellen", "30,1 % und 28,7 % gegen 140,7 % (KQ + empirisch) und 141,1 % (direkt)", "um mehr als 7,1 Punkte", "in 3 von 6 Zellen im Median knapp darunter (um höchstens 1,1 Punkte)"):
        assert text in README, text


def test_newsvendor_right_class():
    lin = {(n, t): B.news_cell("linear", n, t)["summary"] for n in C.NEWS_N_CHOICES for t in C.NEWS_TAUS}
    for key, s in lin.items():
        assert s["kq_emp"]["median"] < min(s["knn"]["median"], s["kern"]["median"]), key
        assert s["direkt"]["median"] > s["kq_emp"]["median"], key
    k1000 = [lin[(1000, t)]["kq_emp"]["median"] for t in C.NEWS_TAUS]
    w1000 = [lin[(1000, t)][k]["median"] for t in C.NEWS_TAUS for k in ("knn", "kern")]
    assert (de_pct(min(k1000)), de_pct(max(k1000))) == ("0,10", "0,30") and (de(min(w1000)), de(max(w1000))) == ("5,0", "9,3")
    assert [de_pct(lin[(1000, t)]["direkt"]["median"]) for t in C.NEWS_TAUS] == ["0,13", "0,26", "0,72"]
    for text in ("KQ + empirisch in allen 6 Zellen im Median unter kNN- und Kerngewichten", "zwischen 0,10 und 0,30 % gegen 5,0 bis 9,3 %", "0,13 / 0,26 / 0,72 % gegen 0,10 / 0,10 / 0,30 %"):
        assert text in README, text


def test_frozen_pre_measurements_hold_the_numbers_of_the_raw_outputs():
    v = B.vormessung()
    assert "out_eo_v1_sp.txt" in v["quelle"] and "out_eo_v3_gewichtet.txt" in v["quelle"] and "scikit-learn" in v["hinweis"]
    assert (len(v["gitter_wald"]), len(v["gitter_kern"]), len(v["news"])) == (24, 6, 24)
    # V1: Zufallswald, Zellen Grad 6 (Spot-Check gegen die Rohausgabe)
    cells = {(c["deg"], c["n"], c["noise"]): c for c in v["gitter_wald"]}
    c = cells[(6, 5000, 0.5)]
    assert (c["regret_kq"]["median"], c["regret_spo600"]["median"], c["regret_wald"]["median"]) == (12.2, 11.6, 11.5)
    assert "Wald (11,5 %) und SPO+ der Vormessung (11,6 %) gleichauf, KQ liegt bei 12,2 %" in README
    # V1d: Kern-Ridge mit scikit-learn
    k = {(c["n"], c["noise"]): c for c in v["gitter_kern"]}
    assert [k[(n, 0.0)]["regret_kern_sklearn"]["median"] for n in (100, 300, 1000)] == [0.49, 0.12, 0.04]
    assert [k[(n, 0.5)]["regret_kern_sklearn"]["median"] for n in (100, 300, 1000)] == [13.54, 12.50, 11.33]
    assert [k[(n, 0.5)]["regret_spo300"]["median"] for n in (100, 300, 1000)] == [14.35, 12.44, 11.84]
    diffs = [abs(k[(n, e)]["regret_kern_sklearn"]["median"] - _s(6, n, e, "regret_kern")) for n in (100, 300, 1000) for e in (0.0, 0.5)]
    assert max(diffs) == pytest.approx(0.45, abs=0.01)
    assert "bis zu 0,45 Punkte (n = 1000, Rauschstufe 0,5)" in README
    # V3: Newsvendor, die Zellen des Plans
    nv = {kind: B.news_vormessung(kind, 3, 1000, 0.95) for kind in C.NEWS_KINDS}
    assert [nv["quadratisch"][key]["median"] for key in ("kq_emp", "direkt", "knn", "wald")] == [134.9, 136.5, 30.4, 29.7]
    assert [nv["linear"][key]["median"] for key in ("kq_emp", "direkt", "knn", "wald")] == [0.3, 0.6, 9.3, 12.4]
    assert nv["quadratisch"]["seeds"] == 30


def test_plan_numbers_versus_this_measurement():
    """Korrekturen gegenüber dem Plan: Messabweichung aus Stück 3, Kern-ETO gegen SPO+ in der Zelle n = 300 mit Rauschen, exakt lineare Welt."""
    w = B.wald_cell(6, 100, 0.5)
    assert (w["spo600_besser_als_kq"], w["regret_spo600"]["median"], w["regret_kq"]["median"]) == (2, 14.5, 13.4)         # Vormessung (Plan: „2 von 20“)
    assert B.cell(6, 100, 0.5)["spo_besser_als_kq"] == 10                                                                  # eigene Messung
    assert (int((np.array(B.stabil()["regret_spo"]) < np.array(B.stabil()["regret_kq"])).sum())) == 29
    k = B.kern_sklearn_cell(300, 0.5)
    assert k["regret_kern_sklearn"]["median"] > k["regret_spo300"]["median"]                                               # V1d: 12,50 gegen 12,44
    assert B.cell(6, 300, 0.5)["summary"]["regret_kern"]["median"] < B.cell(6, 300, 0.5)["summary"]["regret_spo"]["median"] and B.cell(6, 300, 0.5)["kern_besser_als_spo"] == 13
    assert ev.CLEAR_SEEDS_SHARE == 0.9 and B.cell(6, 100, 0.5)["kern_besser_als_kq"] < ev.CLEAR_SEEDS_SHARE * 20
    assert "„kein klarer Gewinner“, wenn Kern-ETO in weniger als 90 % der Seeds unter KQ liegt (bei n = 100 mit Rauschen: 16 von 20)" in README
    for text in ("2 von 20 Seeds besser", "in 10 von 20 Seeds besser als KQ (Median 14,4 gegen 14,0 %)", "29 von 100 und im Median 0,53 Punkte zu Lasten von SPO+",
                 "(12,50 gegen 12,44 %); hier 12,3 gegen 12,6 % bei 13 von 20 Seeds"):
        assert text in README, text


def test_exactly_linear_noiseless_world():
    worst = {"kq": 0.0, "spo": 0.0, "kern": 0.0}
    for n in (100, 200, 1000):
        for seed in (0, 1, 2):
            run = run_world(RunSpec(deg=1, noise=0.0, n=n, seed=seed, n_test=500))
            for m in worst:
                worst[m] = max(worst[m], run.regret[m])
    assert worst["kq"] < 1e-12 and worst["kern"] < 1e-4 and worst["spo"] < 1e-3, worst
    assert "In 9 Läufen (n = 100 / 200 / 1000, Zufalls-Seeds 0 bis 2)" in README


def test_live_run_reproduces_the_bands():
    """Zwei Seeds der Zelle Grad 6, ohne Rauschen, n = 1000 neu gerechnet: jeder Wert liegt (mit Toleranz für die Plattform) beim Band der Messung."""
    cell = B.cell(6, 1000, 0.0)
    for seed in (0, 1):
        run = run_world(RunSpec(deg=6, noise=0.0, n=1000, seed=seed))
        for m in ("kq", "spo", "kern"):
            assert 100 * run.regret[m] == pytest.approx(cell[f"regret_{m}"][seed], abs=0.25)
        assert run.regret["kern"] < run.regret["spo"] < run.regret["kq"]
        assert abs(run.regret["orakel"]) < 1e-12


def test_v4_raw_output_agrees_with_the_bands_in_structure():
    text = (ROOT / "tools" / "out_vormessung_v4.txt").read_text(encoding="utf-8")
    assert "Teil A" in text and "Teil B" in text and text.count("Fehlgriff unten > oben in 20 von 20 Seeds") == 12
    assert "VORAB FESTGELEGTES KRITERIUM" in (ROOT / "tools" / "vormessung_v4.py").read_text(encoding="utf-8")
