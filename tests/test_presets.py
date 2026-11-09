"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellung, Einrasten der Adresszeile, Bänder zu jedem Preset vorhanden."""

import pytest

import zoi_bands as B
import zoi_constants as C
import zoi_evaluation as ev
import zoi_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_choices_and_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert p["deg"] in C.DEG_CHOICES and p["noise"] in C.NOISE_CHOICES and p["n"] in C.N_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    d = ev.Settings()
    assert C.PRESETS[first] == {"deg": d.deg, "noise": d.noise, "n": d.n, "seed": d.seed}
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == {
        "deg_slider": d.deg, "noise_slider": d.noise, "n_slider": d.n, "seed_input": d.seed,
        "news_kind": d.news_kind, "news_tau": d.news_tau, "news_n": d.news_n,
    }


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_preset_names_describe_their_values():
    """Preset-Namen gegen die tatsächlichen Werte (DEMO-PLAYBOOK §11 Punkt 5)."""
    p = C.PRESETS
    assert p["Lineare Wahrheit: zweistufig passt"]["deg"] == 1 and p["Lineare Wahrheit: zweistufig passt"]["noise"] == 0.0
    assert p["Mit Rauschen, viele Daten"]["noise"] == 0.5 and p["Mit Rauschen, viele Daten"]["n"] == max(C.N_CHOICES)
    assert p["Mit Rauschen, wenig Daten"]["noise"] == 0.5 and p["Mit Rauschen, wenig Daten"]["n"] == min(C.N_CHOICES)
    assert p["Falsche lineare Klasse, ohne Rauschen (Voreinstellung)"]["deg"] == 6 and p["Falsche lineare Klasse, ohne Rauschen (Voreinstellung)"]["noise"] == 0.0


def test_every_cell_of_the_choice_grid_has_a_band_with_kernel_eto():
    for deg in C.DEG_CHOICES:
        for n in C.N_CHOICES:
            for noise in C.NOISE_CHOICES:
                c = B.cell(deg, n, noise)
                assert len(c["regret_kq"]) == len(C.BAND_SEEDS) and len(c["regret_kern"]) == len(C.BAND_SEEDS)


def test_every_cell_of_the_band_grid_exists_and_kernel_eto_stops_at_the_limit():
    for deg in C.DEG_CHOICES:
        for n in C.N_BAND:
            for noise in C.NOISE_CHOICES:
                c = B.cell(deg, n, noise)
                assert ("regret_kern" in c) == (n <= C.KERNEL_MAX_N)


def test_every_newsvendor_cell_of_the_choice_grid_has_a_band_and_a_forest_cell():
    for kind in C.NEWS_KINDS:
        for n in C.NEWS_N_CHOICES:
            for tau in C.NEWS_TAUS:
                assert len(B.news_cell(kind, n, tau)["knn"]) == 20
                assert B.news_vormessung(kind, C.NEWS_P, n, tau)["seeds"] == 30


def test_snap_rounds_to_the_nearest_allowed_value_and_limits_the_rest():
    spec = P.SETTING_SPECS
    assert P.snap(spec["deg_slider"], 3) == 2          # genau in der Mitte von 2 und 4: der kleinere
    assert P.snap(spec["deg_slider"], 5) == 4 and P.snap(spec["deg_slider"], 99) == 6 and P.snap(spec["deg_slider"], -3) == 1
    assert P.snap(spec["noise_slider"], 0.3) == 0.5 and P.snap(spec["noise_slider"], 0.2) == 0.0 and P.snap(spec["noise_slider"], 7.0) == 0.5
    assert P.snap(spec["n_slider"], 10) == 100 and P.snap(spec["n_slider"], 2000) == 1000 and P.snap(spec["n_slider"], 250) == 300
    assert P.snap(spec["seed_input"], -5) == 0 and P.snap(spec["seed_input"], 10**9) == C.SEED_MAX
    assert P.snap(spec["news_tau"], 0.9) == 0.95 and P.snap(spec["news_tau"], 0.1) == 0.5 and P.snap(spec["news_n"], 500) == 100     # näher an 100 als an 1000
    assert P.snap(spec["news_kind"], "linear") == "linear" and P.snap(spec["news_kind"], "betrag") == C.DEFAULT_NEWS_KIND


@pytest.mark.parametrize("deg,noise,n", [(d, e, n) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES for n in C.N_CHOICES])
def test_settings_run_spec_matches_the_constants(deg, noise, n):
    spec = ev.Settings(deg=deg, noise=noise, n=n).run_spec()
    assert (spec.p, spec.h, spec.steps, spec.lr, spec.n_test) == (C.N_FEATURES, C.GRID_H, C.STEPS, C.LR, C.N_TEST)
    assert (spec.deg, spec.noise, spec.n) == (deg, noise, n)
