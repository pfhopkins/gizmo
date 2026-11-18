"""sn_hydro_recompute: feedback after the closing kick with the hydro pass redone (stella_dev RESOLVED_ISM_FB_AFTER_CLOSING_KICK, c68d0ced8,
and RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB, 667879eb8). The port deposits mechanical feedback after the closing half-kick by default; with
RESOLVED_ISM_FB_AFTER_CLOSING_KICK a step whose SN or Ia deposit reached a cell repeats the hydro pass, and with
RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB wind and AGB deposits do too (radiation pressure: test/fb_after_closing_kick_radp).
Adiabatic gas (the one test exception to the production model), one 10 Msun star blowing its wind and dying 200 yr in, 20^3 cells
of 1 Msun or of 0.1 Msun (same density), 2 ranks, 5 kyr. Turbulent metal diffusion across a metallicity step (3 x solar for x < L/4)
far from the star.

Bars (set before the runs):
- built without the flags: no repeat;
- RESOLVED_ISM_FB_AFTER_CLOSING_KICK: at least one repeat (the SN), fewer than with RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB added
  (its wind steps repeat too);
- the repeated pass applies no second metal exchange: cells at least 0.2 L from the star end with the metallicity of the run
  without the flag, max |Z_flag - Z_default| <= 1e-6 x max |Z_default - Z_start| over those cells (a second exchange on a repeat
  step adds a whole step's exchange); the exchange is non-zero there;
- the old names (FB_G17_BOUNDARY_INJECTION, GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK), RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB alone
  and the flag with MFV stop the build with their message.
The gas energy excess over start + 1e51 erg at full synchronisation is printed for the runs, not a bar."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

NAME = "sn_hydro_recompute"
HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L1, N_SIDE, M, Z, T_DEATH = 28.6, 20, 10.0, 0.0142, 200.0
FLAG = "RESOLVED_ISM_FB_AFTER_CLOSING_KICK"
FLAG_ALL = "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB"
REPEAT = "repeating the hydro pass"
RUN_TIMEOUT = 1800  # s


def star_pos(L, m_cell):
    return np.full(3, 0.5 * L + 0.17 * m_cell ** (1 / 3))


def run(tag, exe, m_cell, tables):
    d = HERE / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    L = L1 * m_cell ** (1 / 3)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M), np.log10(Z)
    age = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - T_DEATH
    x = lattice(N_SIDE, L)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, x, [(M, age, star_pos(L, m_cell))],
                            gas_mass=m_cell, hsml=2.5 * m_cell ** (1 / 3), z_gas=np.where(x[:, 0] < 0.25 * L, 3 * Z, Z))
    p = (HERE / f"{NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / "run.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2, 1) + [str(HERE / exe), "run.params", "2", "0"], cwd=d, stdout=o, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=RUN_TIMEOUT)
    a = np.loadtxt(d / "output" / "energy_ledger.txt", ndmin=2)
    E = a[:, 2] + a[:, 3]
    after = E > E[0] + 0.5
    ex = E[after] - E[0] - 1.0
    return {"repeats": (d / "run.out").read_text().count(REPEAT), "dir": d, "L": L, "m": m_cell,
            "excess_max": float(ex.max()) if after.any() else np.nan, "excess_end": float(ex[-1]) if after.any() else np.nan, "rows": int(after.sum())}


def metals(r, snap):
    with h5py.File(r["dir"] / "output" / snap, "r") as f:
        g = f["PartType0"]; o = np.argsort(g["ParticleIDs"][()])
        return g["ParticleIDs"][()][o], g["Coordinates"][()][o], g["Metallicity"][()][o][:, 0]


@pytest.fixture(scope="module")
def ladder():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(NAME)
    os.replace(HERE / "GIZMO", HERE / "GIZMO_default")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG, FLAG_ALL))
    os.replace(HERE / "GIZMO", HERE / "GIZMO_all")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG,))
    out = {}
    for m in (1.0, 0.1):
        out[("default", m)] = run(f"default_{m}", "GIZMO_default", m, t)
        out[("closing", m)] = run(f"closing_{m}", "GIZMO", m, t)
    out[("all", 1.0)] = run("all_1.0", "GIZMO_all", 1.0, t)
    for k, v in out.items():
        print(f"{k[0]:8s} {k[1]:4.1f} Msun: repeats {v['repeats']}, energy excess max {v['excess_max']:+.5f}, end {v['excess_end']:+.5f} of 1e51 erg ({v['rows']} rows)")
    return out


def test_no_repeat_without_the_flags(ladder):
    assert ladder[("default", 1.0)]["repeats"] == 0 and ladder[("default", 0.1)]["repeats"] == 0


def test_sn_repeats_winds_only_with_the_second_flag(ladder):
    assert all(v["rows"] > 3 for v in ladder.values()), "the supernova went off in every run"
    assert ladder[("closing", 1.0)]["repeats"] >= 1 and ladder[("closing", 0.1)]["repeats"] >= 1
    assert ladder[("all", 1.0)]["repeats"] > ladder[("closing", 1.0)]["repeats"]


def test_metal_exchange_once_per_step(ladder):
    for m in (1.0, 0.1):
        a, b = ladder[("default", m)], ladder[("closing", m)]
        ida, x, z0 = metals(a, "snapshot_000.hdf5")
        ida1, _, za = metals(a, "snapshot_001.hdf5")
        idb1, _, zb = metals(b, "snapshot_001.hdf5")
        assert np.array_equal(ida, ida1) and np.array_equal(ida, idb1)
        d = x - star_pos(a["L"], m); d -= a["L"] * np.round(d / a["L"])
        far = np.linalg.norm(d, axis=1) > 0.2 * a["L"]
        exch, diff = np.max(np.abs(za - z0)[far]), np.max(np.abs(zb - za)[far])
        print(f"{m} Msun: {far.sum()} far cells; metal exchange max |dZ| {exch:.3e}; flag vs default max |dZ| {diff:.3e} ({diff / exch:.2e})")
        assert exch > 0 and diff <= 1e-6 * exch


BUILD_ERRORS = [
    (("FB_G17_BOUNDARY_INJECTION",), "FB_G17_BOUNDARY_INJECTION was renamed RESOLVED_ISM_FB_AFTER_CLOSING_KICK"),
    (("GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK",), "GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK was replaced by RESOLVED_ISM_FB_AFTER_CLOSING_KICK"),
    ((FLAG_ALL,), "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB extends the hydro recompute of RESOLVED_ISM_FB_AFTER_CLOSING_KICK: set both"),
    ((FLAG, "HYDRO_MESHLESS_FINITE_VOLUME"), "RESOLVED_ISM_FB_AFTER_CLOSING_KICK repeats the hydro pass on feedback steps: MFV"),
]


@pytest.mark.parametrize("flags,message", BUILD_ERRORS, ids=["old_name", "old_port_name", "second_alone", "mfv"])
def test_build_stops(flags, message):
    cfg = [l for l in (HERE / "Config.sh").read_text().splitlines() if not (flags[-1] == "HYDRO_MESHLESS_FINITE_VOLUME" and l.strip() == "HYDRO_MESHLESS_FINITE_MASS")]
    (REPO / "src" / "Config.sh").write_text("\n".join(cfg + list(flags)) + "\n")
    r = subprocess.run("make clean > /dev/null && make -j8", shell=True, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode != 0 and message in r.stdout + r.stderr
