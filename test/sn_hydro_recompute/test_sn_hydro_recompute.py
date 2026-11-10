"""sn_hydro_recompute (narrow, with a measurement): GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK on top of the default deposit after the
closing half-kick (gpu_kokkos_unified af507c293): a step in which mechanical feedback reached any cell repeats the hydro pass, as
stella_dev's FB_G17_BOUNDARY_INJECTION. Adiabatic gas (the one test exception to the production model), one 10 Msun star dying 200 yr
in, 20^3 cells of 1 Msun or of 0.1 Msun (same density), 2 ranks, 5 kyr.

Bars: built without the option no repeat; with it at least one repeat. The gas energy excess over start + 1e51 erg at full
synchronisation is printed for the four runs (the single-SN ladder at 1 and 0.1 Msun), not a bar: stella_dev's study gave +2.0 %
(default) and +1.2 % (both ingredients) at 1 Msun with its old deposit point."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

NAME = "sn_hydro_recompute"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L1, N_SIDE, M, Z, T_DEATH = 28.6, 20, 10.0, 0.0142, 200.0
FLAG = "GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK"


def run(tag, exe, m_cell, tables):
    d = HERE / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    L = L1 * m_cell ** (1 / 3)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M), np.log10(Z)
    age = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - T_DEATH
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), [(M, age, np.full(3, 0.5 * L + 0.17 * m_cell ** (1 / 3)))],
                            gas_mass=m_cell, hsml=2.5 * m_cell ** (1 / 3))
    p = (HERE / f"{NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / "run.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2, 1) + [str(HERE / exe), "run.params", "2", "0"], cwd=d, stdout=o, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    a = np.loadtxt(d / "output" / "energy_ledger.txt", ndmin=2)
    E = a[:, 2] + a[:, 3]
    after = E > E[0] + 0.5
    ex = E[after] - E[0] - 1.0
    return {"repeats": (d / "run.out").read_text().count("HYDRO_RECOMPUTE_AFTER_FEEDBACK: repeating"),
            "excess_max": float(ex.max()) if after.any() else np.nan, "excess_end": float(ex[-1]) if after.any() else np.nan, "rows": int(after.sum())}


@pytest.fixture(scope="module")
def ladder():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(NAME)
    os.replace(HERE / "GIZMO", HERE / "GIZMO_default")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG,))
    out = {}
    for m in (1.0, 0.1):
        out[("default", m)] = run(f"default_{m}", "GIZMO_default", m, t)
        out[("recompute", m)] = run(f"recompute_{m}", "GIZMO", m, t)
    for k, v in out.items():
        print(f"{k[0]:9s} {k[1]:4.1f} Msun: repeats {v['repeats']}, energy excess max {v['excess_max']:+.5f}, end {v['excess_end']:+.5f} of 1e51 erg ({v['rows']} rows)")
    return out


def test_no_repeat_without_the_option(ladder):
    assert ladder[("default", 1.0)]["repeats"] == 0 and ladder[("default", 0.1)]["repeats"] == 0


def test_repeat_with_the_option(ladder):
    assert ladder[("recompute", 1.0)]["repeats"] >= 1 and ladder[("recompute", 0.1)]["repeats"] >= 1
    assert all(v["rows"] > 3 for v in ladder.values()), "the supernova went off in every run"
