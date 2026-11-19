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
- the repeated pass applies no second metal exchange: every repeated pass of the flag runs (TEST_PRINT_METAL_CHANGE_OF_REPEATED_HYDRO_PASS,
  largest change of any gas cell's metallicity in that pass) changes no metallicity by more than 1e-6 x max |Z_default - Z_start|
  over the cells beyond the remnant (at least 0.4 L from the star; the Sedov radius after 5 kyr is about 0.2 L), which must be
  non-zero; one such line per repeat. Cells are matched by ID; IDs that a split repeats are left out. The far cells' flag vs
  default difference is printed, not a bar: feedback after the closing kick changes the flow the far shear layer sees, so the two
  runs differ there by about 1e-2 of the exchange with or without a second exchange (job 231836 at 2f8805d23: 1.02e-2 with the
  guard, 3.20e-2 with the guard removed, b1; one repeat in 33 steps).
  A shear layer (v_y = 2 km/s exp(-(d/0.1 L)^2), d the distance to the x = 0 interface) drives the exchange there.
  Record: the first version (job 231729) selected cells beyond 0.2 L, inside the remnant, and wrote 11 metal slots into this
  one-slot build's ICs, which the reader took as a flattened array; the second (job 231819) had static gas at the interface, where
  the exchange was 5e-13 (round-off of the shear-driven coefficient) and the flag and default runs differed by 7e-14 with or without
  the guard; shear layer added after that run (job 231836: the flag-vs-default comparison then failed by the flow difference
  above); the comparison moved to the repeated pass itself (2026-11-19). The 1e-6 bar is unchanged throughout;
- the old names (FB_G17_BOUNDARY_INJECTION, GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK), RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB alone,
  the flag with MFV and the flag with DM_HEATING stop the build with their message.
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
HOOK = "TEST_PRINT_METAL_CHANGE_OF_REPEATED_HYDRO_PASS"
V_SHEAR = 2.0  # km/s, shear layer at the x = 0 interface
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
    with h5py.File(d / "output" / "snapshot_000.hdf5", "r+") as f:  # this build carries total Z only: one metal slot
        for t in ("PartType0", "PartType4"):
            z = f[f"{t}/Metallicity"][()][:, 0]
            del f[f"{t}/Metallicity"]
            f[f"{t}/Metallicity"] = z
        f["Header"].attrs["Flag_Metals"] = 1
        xg = f["PartType0/Coordinates"][()][:, 0]; v = f["PartType0/Velocities"][()]
        dx = np.minimum(xg, L - xg)  # distance to the x = 0 metallicity interface (periodic)
        v[:, 1] = V_SHEAR * np.exp(-(dx / (0.1 * L)) ** 2)  # a shear layer there drives the turbulent metal exchange; nil at the star
        f["PartType0/Velocities"][...] = v
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
        g = f["PartType0"]
        z = g["Metallicity"][()]
        return g["ParticleIDs"][()], g["Coordinates"][()], (z if z.ndim == 1 else z[:, 0])


def by_unique_id(runs):
    """the (ID, x, Z) of every run on the IDs that occur once in each run, sorted by ID"""
    keep = None
    for ids, _, _ in runs:
        u, c = np.unique(ids, return_counts=True)
        keep = set(u[c == 1]) if keep is None else keep & set(u[c == 1])
    keep = np.array(sorted(keep))
    out = []
    for ids, x, z in runs:
        sel = np.isin(ids, keep); o = np.argsort(ids[sel])
        out.append((ids[sel][o], x[sel][o], z[sel][o]))
    return out


@pytest.fixture(scope="module")
def ladder():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(NAME)
    os.replace(HERE / "GIZMO", HERE / "GIZMO_default")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG, FLAG_ALL, HOOK))
    os.replace(HERE / "GIZMO", HERE / "GIZMO_all")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG, HOOK))
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
    for arm, m in (("closing", 1.0), ("closing", 0.1), ("all", 1.0)):  # "all": the wind steps repeat the pass too
        a, b = ladder[("default", m)], ladder[(arm, m)]
        (_, x, z0), (_, _, za), (_, _, zb) = by_unique_id([metals(a, "snapshot_000.hdf5"), metals(a, "snapshot_001.hdf5"), metals(b, "snapshot_001.hdf5")])
        d = x - star_pos(a["L"], m); d -= a["L"] * np.round(d / a["L"])
        far = np.linalg.norm(d, axis=1) > 0.4 * a["L"]
        exch, diff = np.max(np.abs(za - z0)[far]), np.max(np.abs(zb - za)[far])
        rep = [float(l.split()[-1]) for l in (b["dir"] / "run.out").read_text().splitlines() if l.startswith(HOOK + ": max |dZ|")]
        print(f"{arm} {m} Msun: {far.sum()} cells beyond 0.4 L; metal exchange max |dZ| {exch:.3e}; flag vs default max |dZ| {diff:.3e} "
              f"({diff / exch:.2e}, not a bar); {len(rep)} repeated passes, their largest metallicity change {max(rep, default=np.nan):.3e}")
        assert exch > 0 and len(rep) == b["repeats"] >= 1 and max(rep) <= 1e-6 * exch


BUILD_ERRORS = [
    (("FB_G17_BOUNDARY_INJECTION",), "FB_G17_BOUNDARY_INJECTION was renamed RESOLVED_ISM_FB_AFTER_CLOSING_KICK"),
    (("GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK",), "GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK was replaced by RESOLVED_ISM_FB_AFTER_CLOSING_KICK"),
    ((FLAG_ALL,), "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB extends the hydro recompute of RESOLVED_ISM_FB_AFTER_CLOSING_KICK: set both"),
    ((FLAG, "HYDRO_MESHLESS_FINITE_VOLUME"), "RESOLVED_ISM_FB_AFTER_CLOSING_KICK repeats the hydro pass on feedback steps: MFV"),
    ((FLAG, "DM_HEATING"), "RESOLVED_ISM_FB_AFTER_CLOSING_KICK repeats the hydro pass, which resets DtInternalEnergy: DM_HEATING's addition"),
]


@pytest.mark.parametrize("flags,message", BUILD_ERRORS, ids=["old_name", "old_port_name", "second_alone", "mfv", "dm_heating"])
def test_build_stops(flags, message):
    cfg = [l for l in (HERE / "Config.sh").read_text().splitlines() if not (flags[-1] == "HYDRO_MESHLESS_FINITE_VOLUME" and l.strip() == "HYDRO_MESHLESS_FINITE_MASS")]
    (REPO / "src" / "Config.sh").write_text("\n".join(cfg + list(flags)) + "\n")
    r = subprocess.run("make clean > /dev/null && make -j8", shell=True, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode != 0 and message in r.stdout + r.stderr
