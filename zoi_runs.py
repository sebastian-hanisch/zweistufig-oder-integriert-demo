"""Läufe für die Klassenfrage: ein Lauf je Welt mit vier Verfahren, Zusammenfassung über Zufalls-Seeds (keine Streamlit-Abhängigkeit, nur numpy).

Ein Lauf (`run_world`) zieht Trainings- und Testdaten zu einem Zufalls-Seed und trainiert drei Verfahren auf denselben Fällen:

* `kq`    Kleinste Quadrate, lineare Klasse, zweistufig (erst Kosten schätzen, dann den Löser fragen),
* `spo`   SPO+ (Löser in der Schleife), dieselbe lineare Klasse, Adam, Start bei KQ,
* `kern`  Kern-ETO: zweistufig mit flexibler Klasse (Gauß-Kern-Ridge, Parameter per Validierung), der Löser kommt im Training nicht vor,
* `orakel` kennt die mittleren Kosten der Testfälle.

Gemessen wird der normierte Regret auf den Testfällen (Summe der SPO-Verluste geteilt durch die Summe der besten Wegkosten, in Prozent) und der Regret in Erwartung
(mit den mittleren statt der beobachteten Kosten, `zoi_gap.mean_regret_per_case`). Herkunft des Gerüsts: `skw_runs.py` aus `spo-kuerzester-weg-demo`, erweitert um Kern-ETO."""

from dataclasses import dataclass, replace

import numpy as np

from zoi_grid import Grid
from zoi_kernel import KernelModel, fit_kernel_eto
from zoi_losses import mse_per_arc, normalized_regret, spo_loss
from zoi_training import fit_least_squares, predict, train_spo_plus
from zoi_world import Dataset, WorldSpec, make_dataset

METHODS = ("kq", "spo", "kern")


@dataclass(frozen=True)
class RunSpec:
    """Alles, was einen Lauf festlegt."""

    deg: int = 6
    noise: float = 0.0
    n: int = 1000
    seed: int = 0
    p: int = 5
    h: int = 5
    steps: int = 300
    lr: float = 0.05
    n_test: int = 1000
    with_kernel: bool = True   # Kern-ETO mitrechnen (bei n = 5000 zu teuer für die Bänder)

    @property
    def world(self) -> WorldSpec:
        return WorldSpec(p=self.p, deg=self.deg, noise=self.noise)


@dataclass
class WorldRun:
    spec: RunSpec
    grid: Grid
    data: Dataset
    kernel: KernelModel | None
    chat: dict          # Vorhersagen der Testkosten je Verfahren ("kq", "spo", "kern", "orakel"): Matrix (n_test x d)
    regret: dict        # normierter Regret auf den Testfällen (Bruchteil, nicht Prozent)
    mse: dict           # mittlerer quadratischer Fehler je Bogen auf den Testkosten
    best_step: int      # Schritt, nach dem der gewählte SPO+-Parameter erreicht war


def run_world(spec: RunSpec) -> WorldRun:
    grid = Grid(spec.h)
    data = make_dataset(spec.world, grid.d, spec.n, spec.n_test, spec.seed)
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    res = train_spo_plus(grid, data.X_train, data.C_train, spec.steps, spec.lr, 0.0, init=coef_kq)
    chat = {"kq": predict(coef_kq, data.X_test), "spo": predict(res.coef, data.X_test), "orakel": data.M_test}
    kernel = None
    if spec.with_kernel:
        kernel = fit_kernel_eto(data.X_train, data.C_train, spec.seed)
        chat["kern"] = kernel.predict(data.X_test)
    regret = {k: normalized_regret(grid, data.C_test, v) for k, v in chat.items()}
    mse = {k: mse_per_arc(data.C_test, v) for k, v in chat.items() if k != "orakel"}
    return WorldRun(spec, grid, data, kernel, chat, regret, mse, res.best_step)


def cell_result(run: WorldRun) -> dict:
    """Die Zahlen eines Laufs: normierter Regret in Prozent und mittlerer Quadratfehler je Bogen (ohne die großen Felder)."""
    out = {"seed": run.spec.seed}
    for k, v in run.regret.items():
        out[f"regret_{k}"] = 100.0 * v
    for k, v in run.mse.items():
        out[f"mse_{k}"] = v
    return out


def run_seeds(spec: RunSpec, seeds) -> list[dict]:
    """Läuft über feste Zufalls-Seeds; je Seed ein Wörterbuch aus `cell_result`."""
    return [cell_result(run_world(replace(spec, seed=int(s)))) for s in seeds]


def summarize(values) -> dict:
    """Median, Minimum, Maximum einer Liste."""
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def count_better(rows: list[dict], better: str, than: str) -> int:
    """In wie vielen Seeds ist `better` kleiner als `than` (streng)? Beispiel: `count_better(rows, "regret_spo", "regret_kq")`."""
    return int(sum(r[better] < r[than] for r in rows))


def case_regret_relative(run: WorldRun, method: str) -> np.ndarray:
    """SPO-Verlust je Testfall in Bruchteilen der besten Kosten dieses Falles (für die Einzelfallansicht)."""
    _, zstar = run.grid.solve(run.data.C_test)
    return spo_loss(run.grid, run.data.C_test, run.chat[method]) / zstar
