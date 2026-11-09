"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt (aus `spo-kuerzester-weg-demo`) ----------------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
N_TEST = 1000                    # Testfälle je Welt
STEPS, LR = 300, 0.05            # Adam für SPO+: Schritte und Lernrate
BAND_SEEDS = tuple(range(20))    # Zufalls-Seeds der vorgerechneten Bänder (zoi_bands.json)

DEG_CHOICES = (1, 2, 4, 6)       # Grad der Wahrheit: 1 = linear
NOISE_CHOICES = (0.0, 0.5)       # Rauschstufe e
N_CHOICES = (100, 300, 1000)     # Trainingsfälle der Live-Rechnung (die Kernmatrix bei n = 5000 ist für eine Demo zu groß)
N_BAND = (100, 300, 1000, 5000)  # Trainingsfälle der vorgerechneten Bänder (Kern-ETO nur bis KERNEL_MAX_N)
KERNEL_MAX_N = 1000
DEFAULT_DEG, DEFAULT_NOISE, DEFAULT_N, DEFAULT_SEED = 6, 0.0, 1000, 0
SEED_MAX = 999999

# --- Newsvendor (Experiment 4) -------------------------------------------------------------------------------------------------------------
NEWS_KINDS = ("linear", "quadratisch")
NEWS_KIND_NAMES = {"linear": "linear (die lineare Klasse trifft die Wahrheit)", "quadratisch": "quadratisch (die lineare Klasse trifft sie nicht)"}
NEWS_TAUS = (0.5, 0.8, 0.95)     # kritisches Verhältnis b / (b + h)
NEWS_N_CHOICES = (100, 1000)
NEWS_P = 3                       # Zahl der Merkmale
DEFAULT_NEWS_KIND, DEFAULT_NEWS_TAU, DEFAULT_NEWS_N = "quadratisch", 0.95, 1000

# --- Lücke Δ (Experiment 3) -----------------------------------------------------------------------------------------------------------------
GAP_DELTAS = (0.01, 0.02, 0.05, 0.10)   # Schwellen der relativen Lücke (Bruchteil des besten Wertes)
GAP_HIST_MAX = 0.10                      # Achse des Histogramms: relative Lücke bis 10 %; größere Werte sammelt der letzte Balken

# --- Anzeige ------------------------------------------------------------------------------------------------------------------------------
COLOR_ORACLE = "#7f7f7f"
COLOR_KQ = "#4c78a8"
COLOR_SPO = "#2F6B65"
COLOR_KERN = "#b8860b"
COLOR_WALD = "#8a6bb0"
COLOR_DIRECT = "#d62728"
COLOR_KNN = "#2F6B65"
COLOR_KQEMP = "#4c78a8"
COLOR_THIRDS = ("#c6dbef", "#6baed6", "#2171b5")

# Die vier Welten der Voreinstellungen; jedes Preset legt alle Regler fest (die Regler von Experiment 4 stehen auf ihren Standardwerten).
PRESETS = {
    "Falsche lineare Klasse, ohne Rauschen (Voreinstellung)": {"deg": 6, "noise": 0.0, "n": 1000, "seed": 0},
    "Lineare Wahrheit: zweistufig passt": {"deg": 1, "noise": 0.0, "n": 1000, "seed": 0},
    "Mit Rauschen, viele Daten": {"deg": 6, "noise": 0.5, "n": 1000, "seed": 0},
    "Mit Rauschen, wenig Daten": {"deg": 6, "noise": 0.5, "n": 100, "seed": 0},
}
PRESET_HELP = {
    "Falsche lineare Klasse, ohne Rauschen (Voreinstellung)": "Grad 6, Rauschstufe 0, 1000 Trainingsfälle: ein lineares Modell passt nicht zur Wahrheit, die Kosten sind exakt bekannt.",
    "Lineare Wahrheit: zweistufig passt": "Grad 1, Rauschstufe 0, 1000 Trainingsfälle: die lineare Klasse enthält die Wahrheit, Kleinste Quadrate trifft jede Entscheidung.",
    "Mit Rauschen, viele Daten": "Grad 6, Rauschstufe 0,5, 1000 Trainingsfälle: ein Teil des Regrets ist unvermeidbar, auch das Orakel hat Regret.",
    "Mit Rauschen, wenig Daten": "Grad 6, Rauschstufe 0,5, nur 100 Trainingsfälle bei 240 Parametern des linearen Modells.",
}
