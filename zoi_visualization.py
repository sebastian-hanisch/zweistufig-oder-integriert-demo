"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import zoi_constants as C
from zoi_evaluation import de, de_pct

_LOCK = dict(fixedrange=True)
_COLORS = {"orakel": C.COLOR_ORACLE, "kq": C.COLOR_KQ, "spo": C.COLOR_SPO, "kern": C.COLOR_KERN, "wald": C.COLOR_WALD,
           "kq_emp": C.COLOR_KQEMP, "direkt": C.COLOR_DIRECT, "knn": C.COLOR_KNN}
_SHORT = {"orakel": "Orakel", "kq": "KQ", "spo": "SPO+", "kern": "Kern-ETO", "wald": "Zufallswald (Vormessung)"}


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


def _bar_figure(labels, keys, medians, mins, maxs, x_title, sources, unit="%") -> go.Figure:
    """Balken je Verfahren (Median) mit Spanne von Minimum bis Maximum über die Seeds; `sources` steht im Hovertext."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=medians, orientation="h", marker_color=[_COLORS[k] for k in keys],
        error_x=dict(type="data", symmetric=False, array=[mx - md for md, mx in zip(medians, maxs)], arrayminus=[md - mn for md, mn in zip(medians, mins)], color="rgba(60,60,60,0.8)", thickness=1.5, width=5),
        customdata=[[de_pct(mn), de_pct(mx), s] for mn, mx, s in zip(mins, maxs, sources)],
        hovertemplate="%{y}<br>Median %{x:.2f} " + unit + " (Spanne %{customdata[0]} bis %{customdata[1]} " + unit + ")<br>%{customdata[2]}<extra></extra>",
    ))
    top = max(maxs) if maxs else 1.0
    for label, md, mx in zip(labels, medians, maxs):   # Zahl rechts vom Ende der Spanne (Anmerkung statt Balkentext, damit die Linie sie nicht überdeckt)
        fig.add_annotation(x=max(mx, md), y=label, text=f"{de_pct(md)} {unit}", showarrow=False, xanchor="left", xshift=6, font=dict(size=12))
    fig.update_xaxes(title_text=x_title, range=[min(0.0, min(mins) * 1.1), max(top * 1.3, 0.5)])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 52 * len(labels), margin=dict(l=10, r=20, t=10, b=45), showlegend=False)
    return _lock(fig)


# --- Experiment 1: drei Klassen -------------------------------------------------------------------------------------------------------------

def build_class_figure(rows: list[dict]) -> go.Figure:
    """Median-Regret je Verfahren in einer Zelle, Balkenende = Maximum und Minimum über die Zufalls-Seeds."""
    return _bar_figure([r["Verfahren"] for r in rows], [r["key"] for r in rows], [r["regret"]["median"] for r in rows],
                       [r["regret"]["min"] for r in rows], [r["regret"]["max"] for r in rows], "Normierter Regret in % (Median, Spanne über die Zufalls-Seeds)", [r["quelle"] for r in rows])


# --- Experiment 2: Datenmenge ---------------------------------------------------------------------------------------------------------------

def build_scan_figure(rows: list[dict]) -> go.Figure:
    """Median-Regret von KQ, SPO+ und Kern-ETO über der Datenmenge, je ein Feld je Rauschstufe; Kern-ETO nur bis n = 1000; das Orakel gestrichelt."""
    noises = sorted({r["noise"] for r in rows})
    fig = make_subplots(rows=1, cols=len(noises), subplot_titles=[f"Rauschstufe {de(e)}" for e in noises], horizontal_spacing=0.1)
    for col, e in enumerate(noises, start=1):
        sub = [r for r in rows if r["noise"] == e]
        xs = [str(r["n"]) for r in sub]
        series = [("KQ", "kq", [r["kq"] for r in sub]), ("SPO+", "spo", [r["spo"] for r in sub])]
        kern = [(str(r["n"]), r["kern"]) for r in sub if r["kern"] is not None]
        for name, key, vals in series:
            fig.add_trace(go.Scatter(x=xs, y=[v["median"] for v in vals], mode="lines+markers", name=name, legendgroup=key, showlegend=col == 1,
                                     line=dict(color=_COLORS[key], width=2.5), marker=dict(size=8),
                                     error_y=dict(type="data", symmetric=False, array=[v["max"] - v["median"] for v in vals], arrayminus=[v["median"] - v["min"] for v in vals], thickness=1, width=3),
                                     hovertemplate="n = %{x}: %{y:.2f} %<extra>" + name + "</extra>"), row=1, col=col)
        if kern:
            fig.add_trace(go.Scatter(x=[x for x, _ in kern], y=[v["median"] for _, v in kern], mode="lines+markers", name="Kern-ETO", legendgroup="kern", showlegend=col == 1,
                                     line=dict(color=_COLORS["kern"], width=2.5), marker=dict(size=8, symbol="diamond"),
                                     error_y=dict(type="data", symmetric=False, array=[v["max"] - v["median"] for _, v in kern], arrayminus=[v["median"] - v["min"] for _, v in kern], thickness=1, width=3),
                                     hovertemplate="n = %{x}: %{y:.2f} %<extra>Kern-ETO</extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=xs, y=[r["orakel"] for r in sub], mode="lines", name="Orakel", legendgroup="orakel", showlegend=col == 1,
                                 line=dict(color=_COLORS["orakel"], width=2, dash="dash"), hovertemplate="n = %{x}: %{y:.2f} %<extra>Orakel</extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Trainingsfälle n", type="category", row=1, col=col)
    fig.update_yaxes(title_text="Regret in % (Median)", row=1, col=1)
    fig.update_layout(height=360, margin=dict(l=50, r=20, t=40, b=50), legend=dict(orientation="h", y=-0.28))
    return _lock(fig)


# --- Experiment 3: die Lücke Δ ---------------------------------------------------------------------------------------------------------------

def build_gap_histogram(rel_gap: np.ndarray) -> go.Figure:
    """Verteilung der relativen Lücke Δ / z* (Prozent) über die Testfälle; Werte über der Achse (GAP_HIST_MAX) sammelt der letzte Balken."""
    edges = np.linspace(0.0, 100 * C.GAP_HIST_MAX, 21)
    clipped = np.minimum(100 * np.asarray(rel_gap), edges[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    over = int(np.sum(100 * np.asarray(rel_gap) >= edges[-1]))
    fig = go.Figure(go.Bar(x=centers, y=100.0 * counts / max(len(rel_gap), 1), width=np.diff(edges) * 0.92, marker_color=C.COLOR_KQ,
                           hovertemplate="Lücke %{x:.2f} %: %{y:.1f} % der Fälle<extra></extra>"))
    fig.update_xaxes(title_text=f"Relative Lücke in % des besten Wertes (letzter Balken: ab {de(100 * C.GAP_HIST_MAX, 0)} %, {over} Fälle)", range=[0, edges[-1]])
    fig.update_yaxes(title_text="Anteil der Testfälle in %")
    fig.update_layout(height=300, margin=dict(l=50, r=20, t=10, b=60), showlegend=False)
    return _lock(fig)


def build_gap_thirds_figure(table: dict) -> go.Figure:
    """Anteil gewählter Nicht-Bestwege je Verfahren in den drei Dritteln der Lücke (unteres Drittel = kleinste Lücken)."""
    names = {"kq": "KQ", "spo": "SPO+", "kern": "Kern-ETO"}
    labels = [f"Lücke klein (Median {de(100 * table['third_median_rel'][0], 2)} %)", f"mittel ({de(100 * table['third_median_rel'][1], 2)} %)", f"groß ({de(100 * table['third_median_rel'][2], 2)} %)"]
    fig = go.Figure()
    for m in ("kq", "spo", "kern"):
        fig.add_trace(go.Bar(name=names[m], x=labels, y=[100 * v for v in table["methods"][m]["fehlgriff"]], marker_color=_COLORS[m],
                             text=[f"{de(100 * v, 0)} %" for v in table["methods"][m]["fehlgriff"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>%{y:.1f} % der Fälle mit anderem als dem besten Weg<extra>" + names[m] + "</extra>"))
    fig.update_yaxes(title_text="Fälle mit Nicht-Bestweg in %", range=[0, 110])
    fig.update_layout(barmode="group", height=340, margin=dict(l=50, r=20, t=10, b=50), legend=dict(orientation="h", y=-0.2))
    return _lock(fig)


# --- Experiment 4: Newsvendor ------------------------------------------------------------------------------------------------------------------

def build_news_figure(rows: list[dict]) -> go.Figure:
    """Mehrkosten gegen das Orakel je Regel (Median, Spanne über die Seeds)."""
    return _bar_figure([r["Verfahren"] for r in rows], [r["key"] for r in rows], [r["extra"]["median"] for r in rows],
                       [r["extra"]["min"] for r in rows], [r["extra"]["max"] for r in rows], "Mehrkosten gegen das Orakel in % (Median, Spanne über die Seeds)", [r["quelle"] for r in rows])
