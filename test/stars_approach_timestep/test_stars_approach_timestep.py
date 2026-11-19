"""stars_approach_timestep: a star's step resolves its approach to the nearest star (STARS_APPROACH_TIMESTEP, stella_dev
64dd07bf7 + a694e2761: STARFORGE's two-body step dt = sqrt(2 eta) S / (1/t_approach + 1/t_freefall) to the nearest star from the
gravity walk, on by default for Stella single stars).

Problem: Stella cloud production flags; two 2 Msun main-sequence stars 1 pc apart approaching at 50 km/s with impact parameter
0.05 pc (outside the softened kernel, 2.8 x 0.01 pc, so the pair potential is Newtonian) in thin gas (n_H = 1e-3, 0.035 Msun in
the 10 pc box), 40 kyr; MaxStellarTimestep 1e4 yr, MaxSizeTimestep 5 kyr (code units pc, km/s: 1 = 0.978 Myr).
Derivation: without the approach step the stars' steps follow the acceleration criterion sqrt(2 eta eps / a), about 7 kyr at
closest approach (a = G 2 Msun / (0.05 pc)^2), against an encounter time b / v = 1 kyr.
Record: these bars replace a first version (final pair energy within 1e-3 of E_kin) that passed with and without the step. They
were set after seeing that version's outputs, and were checked against those outputs before the rerun. Physics behind them: the
encounter's only observable effect on the pair is the perpendicular kick, which the exact two-body orbit gives; a leapfrog step
that skips the closest approach samples the force at one phase, so its kick error is of order unity, while a resolved encounter
(about 25 steps per b / v) is second-order accurate, (dt / t_enc)^2 ~ 2e-3, below 1 %. The pair energy at any snapshot measures
the same error on the scale of the energy exchanged, G m1 m2 / b; 5 % allows the resolved error with margin and is far below a
skipped approach.
test_encounter_kick: the change of the pair's relative velocity
  perpendicular to the approach over the run equals the exact two-body orbit's (scipy, rtol 1e-12, from the first snapshot)
  to 1 %; the largest change of the pair energy m v^2 / 2 - G m1 m2 / r over the snapshots stays below 5 % of G m1 m2 / b
  (the potential at closest approach; the kinetic energy is 3600 x larger, so the final pair energy is not a check:
  the leapfrog returns it whatever the step); the closest sampled separation is below 0.1 pc.
test_mass: gas plus star mass constant to 1e-12 (snapshots in double precision).
test_control_without_the_step_fails_the_bar: negative control, built with STARS_APPROACH_TIMESTEP_OFF the kick error exceeds 1 %.
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

NAME = "stars_approach_timestep"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

G = 4.300917e-3  # pc (km/s)^2 / Msun
L, NG, M_STAR, V, B = 10.0, 8, 2.0, 25.0, 0.05


def _run(tag, flags=()):
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(NAME, extra_config_flags=flags)
    d = HERE / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    m_gas = 1e-3 * 1.6726e-24 / 0.7065 * (3.085678e18 ** 3) / 1.989e33 * L ** 3 / NG ** 3
    c = L / 2
    stars = [(M_STAR, 5.0e7, (c - 0.5, c - B / 2, c)), (M_STAR, 5.0e7, (c + 0.5, c + B / 2, c))]
    make_treecol_pixels_ics(str(ic), L, lattice(NG, L), stars, gas_mass=m_gas, T=100.0, hsml=2 * L / NG,
                            tracabund=np.tile([0.0, 1.0e-4, 0.0], (NG ** 3, 1)))
    with h5py.File(ic, "r+") as f:
        f["PartType4/Velocities"][...] = np.array([[V, 0, 0], [-V, 0, 0]])
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text().replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    (d / f"{NAME}.params").write_text(p)
    with open(d / f"test_{NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=1500)
    assert "Final time=" in (d / f"test_{NAME}.out").read_text(), "run did not reach TimeMax"
    return sorted(glob.glob(str(d / "output" / "snapshot_*.hdf5")))


def pair(s):
    with h5py.File(s, "r") as f:
        st = f["PartType4"]; o = np.argsort(st["ParticleIDs"][()])
        x, v, m = st["Coordinates"][()][o], st["Velocities"][()][o], st["Masses"][()][o]
        mt = m.sum() + f["PartType0/Masses"][()].sum()
    r = np.linalg.norm(x[1] - x[0])
    return 0.5 * np.sum(m * np.sum(v ** 2, axis=1)) - G * m[0] * m[1] / r, r, 0.5 * np.sum(m * np.sum(v ** 2, axis=1)), mt


@pytest.fixture(scope="module")
def run_off():
    return _run("run_off", ("STARS_APPROACH_TIMESTEP_OFF",))


@pytest.fixture(scope="module")
def run(run_off):  # the control first: both builds share src/ and the test directory's GIZMO
    return _run("run")


def kick_error(snaps):
    from scipy.integrate import solve_ivp
    (x0, v0, m, t0), (x1, v1, _, t1) = stars(snaps[0]), stars(snaps[-1])
    gm = G * m.sum()
    orbit = solve_ivp(lambda t, y: np.r_[y[3:], -gm * y[:3] / np.linalg.norm(y[:3]) ** 3], (t0, t1),
                      np.r_[x0[1] - x0[0], v0[1] - v0[0]], rtol=1e-12, atol=1e-14).y[:, -1]
    return ((v1[1, 1] - v1[0, 1]) - (v0[1, 1] - v0[0, 1])) / (orbit[4] - (v0[1, 1] - v0[0, 1])) - 1


def test_control_without_the_step_fails_the_bar(run_off):
    """negative control: built with STARS_APPROACH_TIMESTEP_OFF the encounter is under-resolved and the kick bar fails"""
    err = kick_error(run_off)
    print(f"step off: perpendicular kick rel. error {err:.3e}")
    assert abs(err) > 1e-2


def stars(s):
    with h5py.File(s, "r") as f:
        st = f["PartType4"]; o = np.argsort(st["ParticleIDs"][()])
        return st["Coordinates"][()][o], st["Velocities"][()][o], st["Masses"][()][o], f["Header"].attrs["Time"]


def test_encounter_kick(run):
    from scipy.integrate import solve_ivp
    p = [pair(s) for s in run]
    e, r = np.array([q[0] for q in p]), np.array([q[1] for q in p])
    (x0, v0, m, t0), (x1, v1, _, t1) = stars(run[0]), stars(run[-1])
    gm = G * m.sum()
    orbit = solve_ivp(lambda t, y: np.r_[y[3:], -gm * y[:3] / np.linalg.norm(y[:3]) ** 3], (t0, t1),
                      np.r_[x0[1] - x0[0], v0[1] - v0[0]], rtol=1e-12, atol=1e-14).y[:, -1]
    kick_ref = orbit[4] - (v0[1, 1] - v0[0, 1])
    kick = (v1[1, 1] - v1[0, 1]) - (v0[1, 1] - v0[0, 1])
    phi_b = G * m[0] * m[1] / B
    print(f"{len(run)} snapshots; closest sampled separation {r.min():.4f} pc; perpendicular kick code {kick:.6e}, two-body {kick_ref:.6e}, "
          f"rel. error {kick / kick_ref - 1:.3e}; largest pair-energy change {np.max(np.abs(e - e[0])) / phi_b:.3e} of G m1 m2 / b")
    assert r.min() < 0.1 and r[-1] > 0.5
    assert abs(kick / kick_ref - 1) < 1e-2
    assert np.max(np.abs(e - e[0])) < 0.05 * phi_b


def test_mass(run):
    m0, m1 = pair(run[0])[3], pair(run[-1])[3]
    assert abs(m1 / m0 - 1) < 1e-12
