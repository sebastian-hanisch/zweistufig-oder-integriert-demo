"""Verdikte als reine Funktionen: Schwellen und jede Zahl im Text an handgesetzten Werten (DEMO-PLAYBOOK §12); zusätzlich jedes Preset zeigt das erwartete Verdikt."""

import pytest

import zoi_bands as B
import zoi_constants as C
import zoi_evaluation as ev


def test_de_and_de_pct_formatting():
    assert ev.de(1.25) == "1,2" and ev.de(-0.04) == "0,0" and ev.de(3.14159, 2) == "3,14"
    assert ev.de_pct(0.049) == "0,05" and ev.de_pct(0.5) == "0,50" and ev.de_pct(1.0) == "1,0" and ev.de_pct(13.04) == "13,0" and ev.de_pct(0.0) == "0,00"
    assert ev._shown_pct(0.996) == 1.0 and ev._shown_pct(0.994) == 0.99 and ev._shown_pct(2.04) == 2.0


# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

def test_verdict_classes_flexible_class_wins():
    kind, text = ev.verdict_classes(r_kq=1.96, r_spo=0.51, r_kern=0.05, r_orakel=0.0, seeds=20, kern_better_spo=19, kern_better_kq=20)
    assert kind == "success"
    assert "Kern-ETO, 0,05 %" in text and "SPO+ (0,51 %)" in text and "KQ (2,0 %)" in text and "Orakel 0,00 %" in text and "20 Zufalls-Seeds" in text
    assert "in 19 von 20 Seeds besser als SPO+ und in 20 von 20 besser als KQ" in text


def test_verdict_classes_flexible_class_ahead_in_the_median_but_not_clear():
    kind, text = ev.verdict_classes(r_kq=14.03, r_spo=14.39, r_kern=13.80, r_orakel=10.54, seeds=20, kern_better_spo=16, kern_better_kq=16)
    assert kind == "info" and "Kern-ETO (13,8 %) liegt im Median unter SPO+ (14,4 %) und unter KQ (14,0 %), aber nicht in den meisten Seeds: kein klarer Gewinner" in text
    assert "in 16 von 20 Seeds besser als SPO+ und in 16 von 20 besser als KQ" in text
    kind, _ = ev.verdict_classes(14.03, 14.39, 13.80, 10.54, 20, 16, 18)          # 18 von 20 = 90 %: klar
    assert kind == "success"
    kind, _ = ev.verdict_classes(14.03, 14.39, 13.80, 10.54, 20, 16, 17)          # 17 von 20 = 85 %: nicht klar
    assert kind == "info"


def test_verdict_classes_all_equal():
    kind, text = ev.verdict_classes(0.0, 0.0, 0.0, 0.0, 20)
    assert kind == "info" and "Alle drei Sorten liegen bei 0,00 %" in text and "nichts zu gewinnen" in text
    kind, text = ev.verdict_classes(0.001, 0.003, 0.004, 0.0, 20)            # gleich nach der Anzeige (0,00 %)
    assert kind == "info" and text.startswith("Alle drei Sorten")


def test_verdict_classes_no_alternative_below_kq():
    kind, text = ev.verdict_classes(r_kq=14.03, r_spo=14.39, r_kern=14.50, r_orakel=10.54, seeds=20, kern_better_spo=3, kern_better_kq=6)
    assert kind == "warning" and "Keine der beiden Alternativen liegt unter KQ (14,0 %)" in text and "SPO+ 14,4 %" in text and "Kern-ETO 14,5 %" in text
    assert "in 3 von 20 Seeds besser als SPO+ und in 6 von 20 besser als KQ" in text
    kind, text = ev.verdict_classes(14.03, 14.03, 14.03, 10.5, 20)             # Gleichstand mit KQ ist nicht darunter, aber alle gleich -> "alle drei"
    assert kind == "info" and text.startswith("Alle drei")
    kind, _ = ev.verdict_classes(14.03, 14.03, 14.5, 10.5, 20)                  # SPO+ gleich KQ, Kern darüber: keine Alternative unter KQ
    assert kind == "warning"


def test_verdict_classes_spo_below_kq_but_not_below_kern():
    kind, text = ev.verdict_classes(2.0, 0.5, 0.5, 0.0, 20)                    # Gleichstand SPO+ und Kern nach der Anzeige: SPO+ liegt nicht über Kern
    assert kind == "info" and "SPO+ (0,50 %) liegt unter KQ (2,0 %) und nicht über Kern-ETO (0,50 %)" in text
    kind, text = ev.verdict_classes(2.0, 0.5, 0.4, 0.0, 20)
    assert kind == "success" and "Kern-ETO, 0,40 %" in text
    kind, text = ev.verdict_classes(13.0, 12.0, 12.5, 10.0, 20)                  # Kern-ETO unter KQ, aber über SPO+: SPO+ liegt unter KQ und nicht über Kern-ETO
    assert kind == "info" and "SPO+ (12,0 %) liegt unter KQ (13,0 %) und nicht über Kern-ETO (12,5 %)" in text


def test_verdict_classes_without_kernel_eto():
    kind, text = ev.verdict_classes(1.97, 0.5, None, 0.0, 20)
    assert kind == "info" and "SPO+ (0,50 %) liegt unter KQ (2,0 %)" in text and "Kern-ETO ist bei dieser Datenmenge nicht gerechnet" in text
    kind, text = ev.verdict_classes(1.0, 1.2, None, 0.0, 20)
    assert "liegt nicht unter KQ" in text


def test_ranking_row_picks_the_smallest_shown_median_and_the_first_on_ties():
    def row(key, name, v):
        return {"key": key, "Verfahren": name, "regret": {"median": v, "min": v, "max": v}, "mse": 1.0}

    rows = [row("orakel", "Orakel (x)", 0.0), row("kq", "KQ (linear, zweistufig)", 2.0), row("spo", "SPO+ (linear)", 0.5), row("kern", "Kern-ETO (flexibel)", 0.04), row("wald", "Zufallswald (x)", 0.0)]
    assert ev.ranking_row(rows) == "Kern-ETO"
    rows[3] = row("kern", "Kern-ETO (flexibel)", 0.5)
    assert ev.ranking_row(rows) == "SPO+"          # Gleichstand SPO+ und Kern nach der Anzeige: SPO+ steht in der Reihenfolge vorn
    assert ev.ranking_row(rows[:3]) == "SPO+"      # ohne Kern-ETO


# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

def _grid_row(n, noise, kq, spo, kern, better, kern_better=None):
    return {"n": n, "noise": noise, "orakel": 0.0, "kq": {"median": kq, "min": kq, "max": kq}, "spo": {"median": spo, "min": spo, "max": spo},
            "kern": None if kern is None else {"median": kern, "min": kern, "max": kern}, "besser": better, "seeds": 20, "kern_besser_spo": kern_better}


def test_verdict_grid_counts_and_lists_the_exceptions():
    rows = [_grid_row(100, 0.0, 2.0, 1.0, 0.5, 20, 18), _grid_row(1000, 0.0, 2.0, 0.5, 0.05, 20, 20), _grid_row(5000, 0.0, 2.0, 0.5, None, 20),
            _grid_row(100, 0.5, 14.0, 14.4, 13.8, 10, 16)]
    kind, text = ev.verdict_grid(6, rows)
    assert kind == "info"
    assert text.startswith("Bei Grad 6 liegt der Median-Regret von SPO+ in 3 von 4 Zellen unter dem von KQ; Kern-ETO liegt in 3 von 3 Zellen (bis n = 1000) unter SPO+")
    assert "n = 100 mit Rauschen: SPO+ 14,4 % gegen KQ 14,0 % (10 von 20 Seeds mit SPO+ besser)" in text


def test_verdict_grid_without_exceptions_has_no_exception_list():
    rows = [_grid_row(100, 0.0, 2.0, 1.0, 0.5, 20), _grid_row(1000, 0.0, 2.0, 0.5, 1.5, 20)]
    kind, text = ev.verdict_grid(4, rows)
    assert text.endswith("unter SPO+.") and "nicht unter KQ" not in text and "1 von 2 Zellen (bis n = 1000)" in text


def test_spo_below_kq_compares_the_shown_values():
    assert ev.spo_below_kq(_grid_row(100, 0.0, 14.04, 13.96, None, 0)) is False       # beide zeigen 14,0
    assert ev.spo_below_kq(_grid_row(100, 0.0, 14.1, 13.9, None, 0)) is True


# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

def _gap_table(kq, spo, kern, share=0.52, median=0.0093):
    return {"anteil_leq": {0.01: share}, "median_rel": median, "methods": {"kq": {"fehlgriff": kq}, "spo": {"fehlgriff": spo}, "kern": {"fehlgriff": kern}}}


def test_verdict_gap_decreasing_mismatch_in_every_method():
    kind, text = ev.verdict_gap(_gap_table([0.55, 0.40, 0.28], [0.45, 0.20, 0.08], [0.32, 0.04, 0.0]))
    assert kind == "info"
    assert "In 52 % der Testfälle" in text and "(Median 0,93 %)" in text and "KQ 55 % gegen 28 %" in text and "SPO+ 45 % gegen 8 %" in text and "Kern-ETO 32 % gegen 0 %" in text
    assert "im unteren Drittel der Lücke größer als im oberen" in text


def test_verdict_gap_warns_if_one_method_does_not_decrease():
    kind, text = ev.verdict_gap(_gap_table([0.55, 0.40, 0.28], [0.45, 0.20, 0.45], [0.32, 0.04, 0.0]))
    assert kind == "warning" and "nicht für alle Verfahren" in text
    kind, _ = ev.verdict_gap(_gap_table([0.554, 0.4, 0.5], [0.45, 0.2, 0.08], [0.32, 0.04, 0.0]))      # 55 % gegen 50 %: nach der Anzeige größer
    assert kind == "info"
    kind, _ = ev.verdict_gap(_gap_table([0.55, 0.4, 0.549], [0.45, 0.2, 0.08], [0.32, 0.04, 0.0]))      # beide zeigen 55 %: nicht größer
    assert kind == "warning"


# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

def _news_row(key, v):
    return {"key": key, "Verfahren": ev.NEWS_NAMES[key], "quelle": "x", "extra": {"median": v, "min": v, "max": v}}


def test_verdict_news_weights_win_in_the_wrong_class():
    rows = [_news_row("kq_emp", 140.7), _news_row("direkt", 141.1), _news_row("knn", 30.1), _news_row("kern", 28.7), _news_row("wald", 29.7)]
    kind, text = ev.verdict_news("quadratisch", 1000, 0.95, rows)
    assert kind == "success"
    assert "Kerngewichte 28,7 % gegen KQ + empirisch 140,7 % Mehrkosten" in text and "nicht (141,1 % gegen 140,7 %)" in text
    assert "Welt quadratisch, n = 1000, τ = 0,95" in text


def test_verdict_news_linear_class_wins_in_the_right_class():
    rows = [_news_row("kq_emp", 0.3), _news_row("direkt", 0.7), _news_row("knn", 9.3), _news_row("kern", 8.8), _news_row("wald", 12.4)]
    kind, text = ev.verdict_news("linear", 1000, 0.95, rows)
    assert kind == "warning" and "Hier liegt die lineare Klasse vorn: KQ + empirisch 0,30 %" in text and "Kerngewichte 8,8 %" in text and "nicht (0,70 % gegen 0,30 %)" in text


def test_verdict_news_direct_learning_helps_and_ties():
    rows = [_news_row("kq_emp", 40.0), _news_row("direkt", 10.0), _news_row("knn", 30.0), _news_row("kern", 31.0)]
    kind, text = ev.verdict_news("linear", 100, 0.5, rows)
    assert kind == "warning" and "direkt 10,0 %" in text and "von 40,0 % auf 10,0 %" in text
    rows = [_news_row("kq_emp", 5.0), _news_row("direkt", 5.0), _news_row("knn", 5.0), _news_row("kern", 6.0)]
    kind, text = ev.verdict_news("linear", 100, 0.5, rows)
    assert kind == "info" and "gleichauf" in text and "ändert die Mehrkosten gegen KQ + empirisch nicht (5,0 %)" in text


# --- Presets durchlaufen ----------------------------------------------------------------------------------------------------------------------

EXPECTED = {
    # Preset -> erwartete Art des Verdikts der Kernfrage (aus den Bändern dieser Demo)
    "Falsche lineare Klasse, ohne Rauschen (Voreinstellung)": "success",
    "Lineare Wahrheit: zweistufig passt": "info",
    "Mit Rauschen, viele Daten": "success",
    "Mit Rauschen, wenig Daten": "info",           # Median-Regret Kern-ETO 13,80 % unter KQ 14,03 % und SPO+ 14,39 %, aber nur in 16 von 20 Seeds besser als KQ: kein klarer Gewinner
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_the_expected_verdict(name):
    p = C.PRESETS[name]
    rows = ev.class_rows(p["deg"], p["n"], p["noise"])
    by = {r["key"]: r["regret"]["median"] for r in rows}
    band = B.cell(p["deg"], p["n"], p["noise"])
    kind, text = ev.verdict_classes(by["kq"], by["spo"], by.get("kern"), by["orakel"], len(band["regret_kq"]), band.get("kern_besser_als_spo"), band.get("kern_besser_als_kq"))
    assert kind == EXPECTED[name], (name, text)
