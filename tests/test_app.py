"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen und prüft,
dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig
identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen.

Diese Demo hat keinen `st.slider`, sondern `st.select_slider`, ein Zahlenfeld und einen Radioknopf; die Vorlage-Prüfung der `st.slider` ist deshalb durch die Prüfung aller Regler aus
`SETTING_SPECS` ersetzt (`test_every_widget_of_the_settings_at_min_and_max_does_not_raise`)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import zoi_constants as C
import zoi_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, state_key):
    for art in ("select_slider", "number_input", "radio", "slider"):
        try:
            return getattr(at, art)(key=state_key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {state_key} nicht gefunden")


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_widget_of_the_settings_at_min_and_max_does_not_raise():
    for state_key, spec in P.SETTING_SPECS.items():
        values = spec.choices if spec.choices else (spec.lo, spec.hi)
        for value in (values[0], values[-1]):
            at = _fresh_app()
            _widget(at, state_key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (state_key, value, [str(e) for e in at.exception])


def test_worst_case_at_the_largest_live_world_finishes_without_error():
    """Größte Live-Welt (n = 1000, Grad 6, Rauschstufe 0,5) mit dem Newsvendor bei n = 1000: Timeout statt Zeit-Assertion."""
    at = _fresh_app()
    at.select_slider(key="noise_slider").set_value(0.5)
    at.select_slider(key="n_slider").set_value(max(C.N_CHOICES))
    at.select_slider(key="news_n").set_value(max(C.NEWS_N_CHOICES))
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_snaps_values_to_the_allowed_choices():
    at = AppTest.from_file(APP_PATH)
    at.query_params["deg"] = "3"
    at.query_params["noise"] = "0.3"
    at.query_params["n"] = "250"
    at.query_params["seed"] = "-4"
    at.query_params["nk"] = "betrag"
    at.query_params["ntau"] = "0.9"
    at.query_params["nn"] = "abc"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert (s["deg_slider"], s["noise_slider"], s["n_slider"], s["seed_input"], s["news_kind"], s["news_tau"], s["news_n"]) == (2, 0.5, 300, 0, C.DEFAULT_NEWS_KIND, 0.95, C.DEFAULT_NEWS_N)


def test_permalink_roundtrip_writes_the_settings_to_the_address():
    at = _fresh_app()
    at.select_slider(key="deg_slider").set_value(4).run(timeout=TIMEOUT)
    assert dict(at.query_params) == {"deg": "4", "noise": "0.0", "n": "1000", "seed": "0", "nk": "quadratisch", "ntau": "0.95", "nn": "1000"}


@pytest.mark.parametrize("name,expected_kind,fragment", [
    ("Falsche lineare Klasse, ohne Rauschen (Voreinstellung)", "success", "Die flexible Klasse (Kern-ETO, 0,05 %)"),
    ("Lineare Wahrheit: zweistufig passt", "info", "Alle drei Sorten liegen bei 0,00 %"),
    ("Mit Rauschen, viele Daten", "success", "Die flexible Klasse (Kern-ETO,"),
    ("Mit Rauschen, wenig Daten", "info", "kein klarer Gewinner"),
])
def test_every_preset_shows_the_expected_core_question_verdict(name, expected_kind, fragment):
    at = _fresh_app()
    at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    shown = {"success": at.success, "info": at.info, "warning": at.warning}[expected_kind]
    assert any(fragment in e.value for e in shown), (name, [e.value for e in list(at.success) + list(at.info) + list(at.warning)])


def test_all_four_experiments_the_theory_the_boundaries_and_the_formulas_are_there():
    at = _fresh_app()
    headings = [s.value for s in at.subheader]
    for i in range(1, 5):
        assert any(h.startswith(f"🔬 Experiment {i}:") for h in headings), i
    assert not any(h.startswith("🔬 Experiment 5") for h in headings)
    assert "📚 Was die Theorie sagt, und was hier gemessen wird" in headings
    assert "🚧 Wo die Annahmen enden" in headings
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So ist der Vergleich aufgebaut") for e in at.expander)


def test_experiment_four_controls_change_the_table_and_the_verdict():
    at = _fresh_app()
    quad = [e.value for e in at.success] + [e.value for e in at.warning]
    at.radio(key="news_kind").set_value("linear").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    lin = [e.value for e in at.success] + [e.value for e in at.warning]
    assert any("Welt quadratisch" in t for t in quad) and any("Welt linear" in t for t in lin)
    assert any("Hier liegt die lineare Klasse vorn" in t for t in lin)


def test_noise_free_linear_world_with_all_regrets_zero_runs_through():
    """Grad 1 ohne Rauschen: alle Regrets 0 (Nenner der Verhältnisse 0 in keinem Text); die App läuft durch und zeigt die Lücke trotzdem."""
    at = _fresh_app()
    at.button(key="preset_Lineare Wahrheit: zweistufig passt").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Lücke" in e.value for e in list(at.info) + list(at.warning))
