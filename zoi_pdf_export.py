"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt.

Herkunft: nach `skw_pdf_export.py` (`spo-kuerzester-weg-demo`) kopiert und für diese Demo angepasst."""

from __future__ import annotations

from fpdf import FPDF

from zoi_evaluation import Settings, class_rows, de, verdict_classes

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "−": "-", "≥": ">=", "≤": "<=", "·": "*", "×": "x", "σ": "sigma", "τ": "tau", "Δ": "Delta", "€": "EUR", "²": "^2", "ˆ": "^"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run) -> bytes:
    """`run`: das `WorldRun` zu den Einstellungen (Live-Lauf einer Instanz)."""
    rows = class_rows(settings.deg, settings.n, settings.noise)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Zweistufig oder integriert: auf die Modellklasse kommt es an"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {settings.deg}, Rauschstufe {de(settings.noise)}, {settings.n} Trainingsfälle, Zufalls-Seed {settings.seed}, "
        f"{run.spec.n_test} Testfälle. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Diese Instanz (ein Zufalls-Seed)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    names = {"orakel": "Orakel", "kq": "KQ (linear)", "spo": "SPO+ (linear)", "kern": "Kern-ETO (flexibel)"}
    pdf.multi_cell(0, 5, _s("Normierter Regret: " + ", ".join(f"{names[k]} {de(100 * run.regret[k])} %" for k in ("orakel", "kq", "spo", "kern") if k in run.regret) + "."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Median über 20 Zufalls-Seeds (vorgerechnet)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        pdf.multi_cell(0, 5, _s(f"{r['Verfahren']}: Regret {de(r['regret']['median'])} % (Spanne {de(r['regret']['min'])} bis {de(r['regret']['max'])} %)  [{r['quelle']}]"),
                       new_x="LMARGIN", new_y="NEXT")
    by = {r["key"]: r["regret"]["median"] for r in rows}
    _, text = verdict_classes(by["kq"], by["spo"], by.get("kern"), by["orakel"], 20)
    pdf.ln(1)
    pdf.multi_cell(0, 5, _s(text.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Zweistufig oder integriert: KQ, SPO+ und Kern-ETO im Gitter. Zufallswald: Vormessung mit scikit-learn, nicht live. Quelle: zweistufig-oder-integriert-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
