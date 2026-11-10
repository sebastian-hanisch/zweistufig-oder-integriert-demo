"""Zweistufig oder integriert: auf die Modellklasse kommt es an - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Im 5 x 5 Gitter (kürzester Weg) und im Newsvendor wird gemessen, wann die zweistufige Regel (erst Kosten schätzen, dann optimieren) und wann das Training auf der Entscheidung
(SPO+) oder die gewichtete Stichprobe gewinnt. Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: zoi_rng, zoi_grid, zoi_world, zoi_losses, zoi_training, zoi_kernel, zoi_gap,
zoi_runs, zoi_news.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import zoi_bands as B
import zoi_constants as C
from zoi_evaluation import (
    Settings,
    class_rows,
    de,
    de_pct,
    gap_table,
    grid_rows,
    news_rows,
    ranking_row,
    verdict_classes,
    verdict_gap,
    verdict_grid,
    verdict_news,
)
from zoi_layout import dunkles_design, kopf_css
from zoi_news import run_news
from zoi_pdf_export import build_pdf
from zoi_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from zoi_runs import run_world
from zoi_visualization import (
    build_class_figure,
    build_gap_histogram,
    build_gap_thirds_figure,
    build_news_figure,
    build_scan_figure,
)

st.set_page_config(page_title="Zweistufig oder integriert – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Training läuft: KQ, SPO+ mit 300 Adam-Schritten und Kern-ETO mit Validierung (bei 1000 Trainingsfällen dauert der Kern-Ridge spürbar) …")
def _world_run(deg: int, noise: float, n: int, seed: int):
    return run_world(Settings(deg=deg, noise=noise, n=n, seed=seed).run_spec())


@st.cache_data(show_spinner="Newsvendor: lineares Programm, Leave-one-out für kNN- und Kerngewichte …")
def _news_run(kind: str, n: int, tau: float, seed: int) -> dict:
    return run_news(kind, C.NEWS_P, n, tau, seed)


def _pct(x: float) -> str:
    return f"{de_pct(100 * x)} %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zoi_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Zweistufig oder integriert: auf die Modellklasse kommt es an")
st.markdown(
    """
Wer Kosten vorhersagt und dann optimiert, kann **zweistufig** vorgehen (erst schätzen, dann den Löser fragen, **ETO**), auf der **Entscheidung** trainieren (**SPO+**, der Löser läuft im Training) oder die Stichprobe
**gewichten** (ohne Modell). Welcher Weg gewinnt, entscheidet in den hier gemessenen Welten vor allem die **Modellklasse**, bei kleinen Datenmengen und Rauschen aber auch die Datenmenge: Passt sie zur Wahrheit, gewinnt in dieser Welt keine Alternative etwas gegenüber der zweistufigen Regel; passt sie nicht, hilft ein anderer Verlust
in derselben Klasse weniger als eine andere, flexiblere Klasse (im Gitter ohne Rauschen und bei genug Daten). Gemessen wird im 5 × 5 Gitter (40 Bögen, 70 Wege) und im Newsvendor mit Merkmalen; Theorien dazu werden nur eingeordnet, nicht nachgemessen.

Weiter unten: **🔬 vier Experimente**, **📚 Was die Theorie sagt**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So ist der Vergleich aufgebaut", expanded=True):
    st.markdown(
        """
1. **Dieselbe Welt, dieselben Daten.** Aus fünf Merkmalen sind die Kosten der 40 Bögen vorherzusagen; die Wahrheit ist ein Polynom vom Grad 1 (linear) bis 6, dazu optional multiplikatives Rauschen. Ein exakter Löser (Rückwärtsrechnung, das Gitter ist azyklisch) wählt aus den vorhergesagten Kosten den Weg; bewertet wird er mit den wahren Kosten (**Regret**).
2. **Drei Sorten, zwei Klassen.** **KQ** und **SPO+** benutzen dasselbe *lineare* Modell mit 240 Parametern (KQ minimiert den Fehler der Kosten, SPO+ den Regret mit dem Löser in jeder Trainingsrunde). **Kern-ETO** ist zweistufig mit einer *flexiblen* Klasse (Gauß-Kern-Ridge, Bandbreite und Strafe per Validierung); der Löser kommt im Training nicht vor.
3. **Das Orakel** kennt die mittleren Kosten und zeigt, wie viel Regret unvermeidbar ist (Rauschen).
4. **Median über 20 feste Zufalls-Seeds**, nie ein einzelner Seed als Befund. Der **Zufallswald** ist eine eingefrorene Vormessung mit scikit-learn (nicht live).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Welt -----------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt im Gitter**")
    deg = st.select_slider("Grad der Wahrheit", options=list(C.DEG_CHOICES), key="deg_slider",
                           help="1 = die Kosten hängen linear von den Merkmalen ab (die lineare Klasse enthält die Wahrheit); je größer, desto stärker weicht die Wahrheit von jedem linearen Modell ab.")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de(v),
                             help="Jeder Bogen kostet das Mittel mal einen Faktor aus [1 − e, 1 + e]. Bei 0 sind die Kosten ohne Rauschen, bei 0,5 liegt auch das Orakel beim Regret deutlich über null.")
    n = st.select_slider("Trainingsfälle n", options=list(C.N_CHOICES), key="n_slider",
                         help="Zahl der Fälle für die Live-Rechnung; das lineare Modell hat 6 × 40 = 240 Parameter. Die vorgerechneten Bänder gehen zusätzlich bis n = 5000 (ohne Kern-ETO).")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption("Newsvendor-Regler (Experiment 4) stehen beim Experiment.")

run = _world_run(int(deg), float(noise), int(n), int(seed))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(f"Eine Instanz (Zufalls-Seed {int(seed)}): {int(n)} Trainingsfälle, {C.N_TEST} Testfälle, normierter Regret = Summe der SPO-Verluste geteilt durch die Summe der besten Wegkosten.")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret Orakel", _pct(run.regret["orakel"]), help="Das Orakel kennt die mittleren Kosten der Testfälle; sein Regret ist der unvermeidbare Teil (Rauschen).")
m2.metric("Regret KQ (linear)", _pct(run.regret["kq"]), delta=f"{de_pct(100 * (run.regret['kq'] - run.regret['orakel']))} Punkte über dem Orakel", delta_color="off")
m3.metric("Regret SPO+ (linear)", _pct(run.regret["spo"]), delta=f"{de_pct(100 * (run.regret['spo'] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse")
m4.metric("Regret Kern-ETO (flexibel)", _pct(run.regret["kern"]), delta=f"{de_pct(100 * (run.regret['kern'] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse",
          help=f"Gauß-Kern-Ridge, zweistufig; gewählte Parameter dieser Instanz: γ = {de(run.kernel.gamma, 2)}, Strafe {de(run.kernel.ridge, 3)}.")

settings = Settings(int(deg), float(noise), int(n), int(seed), str(st.session_state.get("news_kind", C.DEFAULT_NEWS_KIND)),
                    float(st.session_state.get("news_tau", C.DEFAULT_NEWS_TAU)), int(st.session_state.get("news_n", C.DEFAULT_NEWS_N)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="zweistufig-oder-integriert.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann gewinnt die zweistufige Regel, wann die Entscheidung?")
rows1 = class_rows(int(deg), int(n), float(noise))
by1 = {r["key"]: r["regret"]["median"] for r in rows1}
band = B.cell(int(deg), int(n), float(noise))
seeds1 = len(band["regret_kq"])
kind, text = verdict_classes(by1["kq"], by1["spo"], by1.get("kern"), by1["orakel"], seeds1, band.get("kern_besser_als_spo"), band.get("kern_besser_als_kq"))
getattr(st, kind)(text)
st.caption(
    f"Das Verdikt gilt für diese Zelle (Grad {int(deg)}, Rauschstufe {de(noise)}, n = {int(n)}) und diese Welt: Median über {seeds1} Zufalls-Seeds, vorgerechnet mit `tools/measure_bands.py`; "
    f"SPO+ war in {band['spo_besser_als_kq']} von {seeds1} Seeds besser als KQ und Kern-ETO in {band['kern_besser_als_kq']} von {seeds1} Seeds besser als KQ."
)

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: drei Sorten, ein Gitter")
st.markdown(
    f"KQ (linear, zweistufig), SPO+ (linear, auf der Entscheidung) und Kern-ETO (flexibel, zweistufig) auf derselben Welt: Grad {int(deg)}, Rauschstufe {de(noise)}, n = {int(n)}. "
    f"Balken: **Median über {seeds1} Zufalls-Seeds**, Linie: Spanne von Minimum bis Maximum. Das **Orakel** zeigt den unvermeidbaren Teil. "
    "Der **Zufallswald** ist eine eingefrorene Vormessung mit scikit-learn und steht nur dort, wo sie gerechnet wurde (n = 100, 1000, 5000)."
)
st.plotly_chart(build_class_figure(rows1), width="stretch", key="class_chart")
table1 = pd.DataFrame([
    {"Verfahren": r["Verfahren"], "Regret (Median)": f"{de_pct(r['regret']['median'])} %", "Spanne über Seeds": f"{de_pct(r['regret']['min'])} bis {de_pct(r['regret']['max'])} %",
     "Diese Instanz": f"{de_pct(100 * run.regret[r['key']])} %" if r["key"] in run.regret else "–",
     "Quadratfehler je Bogen (Median)": "–" if r["mse"] is None else de(r["mse"], 2), "Quelle": r["quelle"]} for r in rows1
])
st.dataframe(table1, hide_index=True, width="stretch")
st.caption(f"Das kleinste Median-Regret unter den drei Sorten hat in dieser Zelle: **{ranking_row(rows1)}**.")

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Datenmenge und Rauschen")
st.markdown(
    f"Alle acht Zellen aus Datenmenge (n = 100 / 300 / 1000 / 5000) und Rauschen (0 / 0,5) zum gewählten **Grad {int(deg)}**, Median über 20 Zufalls-Seeds (vorgerechnet); Linien-Enden: Minimum und Maximum. "
    "**Kern-ETO gibt es nur bis n = 1000** (die Kernmatrix bei n = 5000 ist für eine Demo zu groß). Wo liegt SPO+ unter KQ, wo nicht?"
)
rows2 = grid_rows(int(deg))
st.plotly_chart(build_scan_figure(rows2), width="stretch", key="scan_chart")
table2 = pd.DataFrame([
    {"n": r["n"], "Rauschstufe": de(r["noise"]), "Orakel": f"{de_pct(r['orakel'])} %", "KQ": f"{de_pct(r['kq']['median'])} %", "SPO+": f"{de_pct(r['spo']['median'])} %",
     "Kern-ETO": "nicht gerechnet" if r["kern"] is None else f"{de_pct(r['kern']['median'])} %",
     "SPO+ besser als KQ": f"{r['besser']} von {r['seeds']} Seeds",
     "Kern-ETO besser als SPO+": "–" if r["kern_besser_spo"] is None else f"{r['kern_besser_spo']} von {r['seeds']} Seeds"} for r in rows2
])
st.dataframe(table2, hide_index=True, width="stretch")
kind, text = verdict_grid(int(deg), rows2)
getattr(st, kind)(text)
with st.expander("Stabilitätsprobe: Grad 6, n = 100, Rauschstufe 0,5 mit 100 weiteren Zufalls-Seeds"):
    sb = B.stabil()
    kq_, sp_, ke_ = (np.array(sb[k]) for k in ("regret_kq", "regret_spo", "regret_kern"))
    st.markdown(
        f"Die 20 Zufalls-Seeds der Bänder sind bei kleinem n knapp; deshalb dieselbe Zelle mit den Zufalls-Seeds {sb['seeds'][0]} bis {sb['seeds'][-1]} ({len(sb['seeds'])} Seeds): "
        f"Median-Regret KQ {de(np.median(kq_))} %, SPO+ {de(np.median(sp_))} %, Kern-ETO {de(np.median(ke_))} %. "
        f"SPO+ ist in {int(np.sum(sp_ < kq_))} von {len(kq_)} Seeds besser als KQ (im Median der Paare {de(np.median(sp_ - kq_))} Punkte), Kern-ETO in {int(np.sum(ke_ < kq_))} von {len(kq_)} "
        f"(im Median der Paare {de(np.median(ke_ - kq_))} Punkte)."
    )

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: die Lücke zwischen bestem und zweitbestem Weg")
st.markdown(
    "Die Theorie (Hu, Kallus, Mao) verlangt für schnelle Raten der zweistufigen Regel eine **Rauschbedingung**: die Lücke Δ(x) zwischen bestem und zweitbestem Weg unter den wahren Mittelkosten soll selten klein sein. "
    "Im Gitter ist Δ messbar (Aufzählung der 70 Wege). Hier die Lücke der **1000 Testfälle dieser Welt**, relativ zum besten Wert (Δ / z*), und der Anteil der Fälle, in denen ein Verfahren einen anderen als den besten Weg wählt, "
    "getrennt nach Dritteln der Lücke. Ob die Raten gelten, wird nicht gemessen."
)
gt = gap_table(run)
st.plotly_chart(build_gap_histogram(gt["rel"]), width="stretch", key="gap_hist")
g_cols = st.columns(len(C.GAP_DELTAS))
for col, d_ in zip(g_cols, C.GAP_DELTAS):
    col.metric(f"Lücke ≤ {de(100 * d_, 0)} % des besten Wertes", f"{de(100 * gt['anteil_leq'][d_], 0)} %", help="Anteil der Testfälle, in denen die relative Lücke höchstens so groß ist (Gleichstand im besten Wert, Lücke 0, eingeschlossen).")
st.plotly_chart(build_gap_thirds_figure(gt), width="stretch", key="gap_thirds")
table3 = pd.DataFrame([
    {"Verfahren": name, "Fehlgriff klein / mittel / groß": " / ".join(f"{de(100 * v, 0)} %" for v in gt["methods"][k]["fehlgriff"]),
     "Regret je Fall klein / mittel / groß": " / ".join(f"{de(v, 2)} %" for v in gt["methods"][k]["regret_je_fall"])}
    for k, name in (("kq", "KQ (linear)"), ("spo", "SPO+ (linear)"), ("kern", "Kern-ETO (flexibel)"))
])
st.dataframe(table3, hide_index=True, width="stretch")
kind, text = verdict_gap(gt)
getattr(st, kind)(text)
st.caption(
    "Fehlgriff = der gewählte Weg ist nicht der beste Weg unter den mittleren Kosten. Regret je Fall = mittlere Kosten des gewählten Weges minus bester Wert, in Prozent des besten Wertes (in Erwartung, ohne das Rauschen der Beobachtung). "
    "Drittel nach der Lücke geordnet: klein = die kleinsten Lücken dieser Welt."
)
with st.expander("Verteilung der Lücke je Grad (20 Welten, vorgerechnet)"):
    rows_a = []
    for dg in C.DEG_CHOICES:
        ra = B.luecke_teil_a(dg)
        med = lambda key, ra=ra: float(np.median([r["anteil_leq"][key] for r in ra]))  # noqa: E731
        rows_a.append({"Grad": dg, **{f"Lücke ≤ {de(100 * d_, 0)} %": f"{de(100 * med(str(d_)), 0)} %" for d_ in (0.01, 0.02, 0.05, 0.10)},
                       "Median der Lücke": f"{de(100 * float(np.median([r['quantile']['0.5'] for r in ra])), 2)} %"})
    st.markdown("Anteil der Testpunkte mit relativer Lücke höchstens so groß, Median über 20 Welten (je 2000 Testpunkte, rauschfreie Mittelkosten; die Lücke hängt nicht vom Rauschen ab):")
    st.dataframe(pd.DataFrame(rows_a), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Newsvendor in der falschen Klasse")
st.markdown(
    "Dieselbe Frage in der kleinsten Welt: Nachfrage $D = \\max(0, \\mu(x) + 8z)$ mit drei Merkmalen, bestellt wird das $\\tau$-Quantil. In der **linearen** Welt trifft die lineare Klasse die Wahrheit, in der **quadratischen** "
    "($\\mu$ enthält $8\\,(x_1^2 - 1)$) nicht. Verglichen werden die zweistufige Regel in der linearen Klasse (KQ + empirisches Quantil), das direkte Lernen auf den Kosten in **derselben** Klasse (Quantilregression) "
    "und die gewichtete Stichprobe (kNN-, Kern- und Zufallswald-Gewichte, ohne Modell). Mehr dazu in der Demo „Gewichtete Stichprobe“ derselben Reihe."
)
e1, e2, e3 = st.columns(3)
with e1:
    nkind = st.radio("Welt", list(C.NEWS_KINDS), key="news_kind", format_func=lambda k: C.NEWS_KIND_NAMES[k], help="Lineare Welt: die lineare Klasse ist richtig. Quadratische Welt: sie ist falsch.")
with e2:
    ntau = st.select_slider("Kritisches Verhältnis τ", options=list(C.NEWS_TAUS), key="news_tau", format_func=lambda v: de(v, 2),
                            help="τ = b / (b + h): Kosten b je fehlende, h je überschüssige Einheit (h = 1). Bei τ = 0,95 ist eine fehlende Einheit 19-mal so teuer wie eine überschüssige.")
with e3:
    nn_ = st.select_slider("Trainingsfälle n (Newsvendor)", options=list(C.NEWS_N_CHOICES), key="news_n", help="Zahl der beobachteten Nachfragen; drei Merkmale.")
rows4 = news_rows(str(nkind), int(nn_), float(ntau))
st.plotly_chart(build_news_figure(rows4), width="stretch", key="news_chart")
live4 = _news_run(str(nkind), int(nn_), float(ntau), int(seed))
table4 = pd.DataFrame([
    {"Regel": r["Verfahren"], "Mehrkosten (Median)": f"{de(r['extra']['median'])} %", "Spanne über Seeds": f"{de(r['extra']['min'])} bis {de(r['extra']['max'])} %",
     "Diese Instanz": f"{de(live4[r['key']])} %" if r["key"] in live4 else "–", "Quelle": r["quelle"]} for r in rows4
])
st.dataframe(table4, hide_index=True, width="stretch")
kind, text = verdict_news(str(nkind), int(nn_), float(ntau), rows4)
getattr(st, kind)(text)
st.caption(
    f"Mehrkosten = Kosten der Regel minus Kosten des Orakels (wahres Quantil), in Prozent der Orakelkosten, exakt über die Nachfrage gerechnet. Eigene Messung: 20 Zufalls-Seeds (0 bis 19); die Spalte „Diese Instanz“ gehört zum Zufalls-Seed {int(seed)} der Seitenleiste "
    f"(gewähltes k = {live4['k']}, Bandbreite {de(live4['h'], 2)}). Der Zufallswald ist eine Vormessung mit scikit-learn (30 Seeds, 5000 Testfälle, 50 Bäume) und nicht live."
)

st.markdown("---")

# --- Theorie ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("📚 Was die Theorie sagt, und was hier gemessen wird")
st.markdown(
    """
Die beiden Arbeiten, auf die sich die Frage stützt, sind **asymptotische** Aussagen für große Datenmengen (Volltext gelesen, Nummern nach der genannten Fassung; Beweise und Anhänge nicht gelesen):

- **Elmachtoub, Lam, Zhang, Zhao** (arXiv 2304.06833, Fassung v4): Das Modell ist eine parametrische *Verteilungsfamilie*; ETO schätzt sie per Maximum Likelihood und optimiert dann, IEO wählt den Parameter über den empirischen Entscheidungsverlust, SAA benutzt kein Modell. **Satz 2:** bei Wohlspezifikation und den Annahmen 1 bis 3 (Konsistenz, Regularität, Glattheit) konvergiert $n \\cdot$ Regret in Verteilung, und die Grenzverteilungen sind für ETO stochastisch kleiner als für IEO und für IEO kleiner als für SAA. **Satz 3:** bei Fehlspezifikation kehrt sich die Ordnung um. Die Autoren nennen die Lücke zwischen Asymptotik und endlichen, leicht falsch spezifizierten Fällen selbst. IEO dort ist nicht dasselbe wie das hier trainierte SPO+.
- **Hu, Kallus, Mao** (arXiv 2011.03030, Fassung v3): Für ein lineares Ziel über einem Polytop mit wahrer Bedingungserwartung in der Modellklasse erreicht die auf die Entscheidung trainierte Regel ohne weitere Annahme $n^{-1/2}$; unter einer **Rauschbedingung** an die Lücke Δ ($P(0 < \\Delta(X) \\le \\delta) \\le (\\gamma\\delta/B)^{\\alpha}$) erreicht sie (mit weiteren technischen Annahmen) $n^{-(1+\\alpha)/(2+\\alpha)}$, die zweistufige Plug-in-Regel (mit weiteren Annahmen an den Schätzer) $n^{-(1+\\alpha)/2}$ (bei $\\alpha = 1$: $n^{-1}$ gegen $n^{-2/3}$). Ihr Gitterversuch (5 × 5, Polynom, drei Klassen) zeigt qualitativ dieselbe Anordnung wie Experiment 1; die Theorie gilt laut den Autoren nicht direkt für SPO+.

**Was diese Demo daraus macht:** Sie misst endliche Stichproben in *einer* Familie von Welten und **prüft die Raten nicht**. Experiment 1 und 2 zeigen, in welchen Zellen welche Sorte vorn liegt; Experiment 3 zeigt, wie die Lücke Δ im Gitter verteilt ist
(relativ zum besten Wert, die Theorie rechnet mit einer anderen Skala) und dass der Fehlgriff mit ihr abnimmt; Experiment 4 zeigt dasselbe Muster in einer zweiten, kleineren Welt.
"""
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
_cell_lin = B.cell(1, 1000, 0.0)
_cell_small = B.cell(6, 100, 0.5)
_stab = B.stabil()
_stab_n = len(_stab["seeds"])
_stab_kq, _stab_sp, _stab_ke = (np.array(_stab[k]) for k in ("regret_kq", "regret_spo", "regret_kern"))
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Eine Familie von Welten** | Nur dieses 5 × 5 Gitter mit dieser Kostenformel (Grad 1, 2, 4, 6), Rauschstufe 0 und 0,5, fünf Merkmale; im Newsvendor nur zwei Welten, drei Merkmale, drei Werte von τ. Aussagen über andere Gitter, Wahrheiten, Rauschen oder Löser sind nicht gemessen. |
| **Die Klasse „flexibel“ ist ein Gauß-Kern-Ridge** | Kern-ETO ist ein bestimmter flexibler Lerner mit einem festen Raster für Bandbreite und Strafe, kein Maßstab für „flexible Klassen“. Andere Lerner (Wälder, Netze) sind nur als Vormessung des Zufallswaldes ohne Abstimmung dabei. |
| **Genug Daten für die Klasse** | Grad 6, Rauschstufe 0,5, n = 100: Median-Regret KQ {de(_cell_small['summary']['regret_kq']['median'])} %, SPO+ {de(_cell_small['summary']['regret_spo']['median'])} %, Kern-ETO {de(_cell_small['summary']['regret_kern']['median'])} % in den 20 Seeds; in den {_stab_n} weiteren Seeds liegt SPO+ in {int(np.sum(_stab_sp < _stab_kq))} und Kern-ETO in {int(np.sum(_stab_ke < _stab_kq))} Seeds unter KQ. |
| **Wohlspezifikation lässt sich nicht beweisen** | Bei Grad 1 ohne Rauschen und n = 1000 liegt KQ bei {de(_cell_lin['summary']['regret_kq']['median'])} %; in echten Daten kennt man die Form nicht. Die Demo kann nur zeigen, was passiert, wenn man sie kennt. |
| **Raten** | Die schnellen Raten der Theorie gelten asymptotisch und unter Zusatzbedingungen; hier wird weder $\\alpha$ geschätzt noch die Konvergenzgeschwindigkeit gemessen. |
| **Rechenzeit** | Gemessen sind Regret und Mehrkosten, keine Sekunden; die App nennt keine feste Zeit. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind Fälle $(x_i, c_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und beobachteten Bogenkosten $c_i \in \mathbb{R}^d$ und ein Löser, der zu Kosten $c$ einen kürzesten Weg $w(c)$ liefert. Gesucht ist ein Modell $\hat c(x)$, dessen gewählter Weg $w(\hat c(x))$ unter den wahren Kosten möglichst gut ist.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $c$, $\hat c$, $m(x)$ | beobachtete Kosten, vorhergesagte Kosten, mittlere Kosten $E[c \mid x]$ |
| $w(c)$, $z^*(c)$ | der vom Löser gewählte Weg (0/1-Vektor über die Bögen) und sein Wert $c^\top w(c)$ |
| $W$ | die Menge der 70 Wege |
| $\gamma$, $a$ | Kernparameter ($\gamma = 1/(2h^2)$ mit Bandbreite $h$) und Strafe des Kern-Ridge |
| $\Delta(x)$ | Lücke zwischen zweitbestem und bestem Wegwert unter $m(x)$ |

**Normierter Regret.** Der SPO-Verlust (Regret) eines Falles ist $\ell(\hat c, c) = c^\top w(\hat c) - z^*(c) \ge 0$, der normierte Regret einer Testmenge ist $\sum_i \ell(\hat c_i, c_i) / \sum_i z^*(c_i)$. Das Orakel setzt $\hat c = m(x)$; bei Rauschen bleibt sein Regret positiv, weil $z^*$ auf den *beobachteten* Kosten bestimmt wird.

**Die drei Sorten.**

- *KQ:* $\hat c(x) = \theta^\top (1, x)$ mit $\theta = \arg\min \sum_i \lVert c_i - \theta^\top(1, x_i)\rVert^2$ (`numpy.linalg.lstsq`), dann $w(\hat c)$.
- *SPO+:* dasselbe lineare Modell, trainiert auf
$$\ell_{\mathrm{SPO+}}(\hat c, c) = \max_{w \in W}\,(c - 2\hat c)^\top w + 2\,\hat c^\top w^*(c) - z^*(c),$$
einer konvexen oberen Schranke des SPO-Verlustes mit Teilgradient $2\,(w^*(c) - w(2\hat c - c))$ (zwei Löseraufrufe je Fall); Adam, Lernrate 0,05, 300 Schritte, Start bei der KQ-Lösung.
- *Kern-ETO:* mit $K_{ij} = \exp(-\gamma \lVert x_i - x_j\rVert^2)$, $\mu = \tfrac1n \sum_i c_i$ und $\alpha = (K + a I)^{-1}(C - \mu)$ ist
$$\hat c(x) = \mu + k(x, X)\,\alpha, \qquad k(x, X)_j = \exp(-\gamma \lVert x - x_j\rVert^2),$$
ein lineares System (`numpy.linalg.solve`); $(\gamma, a)$ aus 4 × 4 Werten per Validierung (80 % Anpassung, 20 % Validierung, kleinster mittlerer quadratischer Fehler der Kosten), danach Neuanpassung auf allen Fällen. Für $\gamma \to 0$ wird $K = \mathbf 1 \mathbf 1^\top$ und die Vorhersage das konstante Mittel $\mu$.

**Die Lücke.** Sind $v_1 \le v_2 \le \dots \le v_{70}$ die sortierten Werte $m(x)^\top w$ über die 70 Wege, ist $\Delta(x) = v_2 - v_1$ und die relative Lücke $\Delta(x)/v_1$. Gleichstand im besten Wert gibt $\Delta = 0$.

**Newsvendor.** Nachfrage $D = \max(0, Y)$, $Y \sim \mathcal N(\mu(x), 8^2)$; bei Kosten $b$ je fehlende und $h$ je überschüssige Einheit ist die optimale Menge das $\tau$-Quantil von $D$ mit $\tau = b/(b+h)$. Die Kosten $b\,(D - q)^+ + h\,(q - D)^+$ sind bis auf den Faktor $b + h$ der Pinball-Verlust $\rho_\tau(D - q) = \max(\tau (D - q), (\tau - 1)(D - q))$. *KQ + empirisch:* Kleinste Quadrate, dazu das $\lceil \tau n \rceil$-kleinste Residuum. *Direkt:* lineare Quantilregression, ein lineares Programm (`scipy.optimize.linprog`, HiGHS). *Gewichte:* zu einem Fall $x$ bekommt jede Beobachtung ein Gewicht $w_i(x) \ge 0$, $\sum_i w_i(x) = 1$ (kNN: $1/k$ für die $k$ nächsten; Kern: proportional zu $\exp(-\lVert x - x_i\rVert^2 / 2h^2)$), bestellt wird das gewichtete $\tau$-Quantil der Nachfragen; $k$ beziehungsweise $h$ per Leave-one-out auf dem Pinball-Verlust.

**Welt.** $x \sim \mathcal N(0, I_p)$, $B \in \{0,1\}^{d \times p}$ mit unabhängigen fairen Münzwürfen, mittlere Kosten $m_j(x) = \big((Bx)_j/\sqrt{p} + 3\big)^{\deg} / 3{,}5^{\deg} + 1$, beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$.

**Literatur.** Elmachtoub, Lam, Zhang, Zhao (arXiv 2304.06833, Fassung v4 vom 22.05.2025; Volltext gelesen, Satznummern nach dieser Fassung). Hu, Kallus, Mao, *Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022 (arXiv 2011.03030, Volltext gelesen, Nummern nach arXiv v3). Elmachtoub und Grigas, *Smart „Predict, then Optimize“*, Management Science 68(1), 2022: SPO+ (nur Abstract gelesen). Bertsimas und Kallus, *From Predictive to Prescriptive Analytics*, Management Science 2020: gewichtete Stichprobe (nur Abstract gelesen). Das Gitter und die Kostenformel sind aus dem Gedächtnis nach dem Schema dieser Literatur nachgebaut, nicht gegen sie geprüft.
        """
    )

sync_query_params({
    "deg_slider": int(deg), "noise_slider": float(noise), "n_slider": int(n), "seed_input": int(seed),
    "news_kind": str(nkind), "news_tau": float(ntau), "news_n": int(nn_),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
