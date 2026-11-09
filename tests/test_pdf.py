"""PDF-Export: erzeugt ein gültiges PDF für jedes Preset (Umlaute und Sonderzeichen brechen die Kernschrift nicht)."""

import pytest

import zoi_constants as C
import zoi_evaluation as ev
from zoi_pdf_export import _s, build_pdf
from zoi_runs import run_world


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_pdf_is_built_for_every_preset(name):
    p = C.PRESETS[name]
    s = ev.Settings(deg=p["deg"], noise=p["noise"], n=p["n"], seed=p["seed"])
    pdf = build_pdf(s, run_world(s.run_spec()))
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_special_characters_are_replaced_by_latin1_safe_text():
    assert _s("ĉ → 3 – Δ τ λ") == "c^ -> 3 - Delta tau ?"
    assert _s("Löser für Gitter ß") == "Löser für Gitter ß"
