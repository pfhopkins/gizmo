"""energy_step_trace_sn: the energy step trace (OUTPUT_ENERGY_STEP_TRACE, stella_dev bc6a2572f..baedabbee with the fix/trace-imf-2026-11-17
changes) and the mass step trace (OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS) close in a production run with winds and a supernova.

Problem: Stella cloud production flags (SGChem net 5, MHD, photoionisation, radiation pressure, turbulent metal diffusion); one 10 Msun
star dying 1 kyr in and one 60 Msun star of 0.1 Myr blowing its wind every step, gas of n_H = 10 at 100 K, 8^3 cells of 0.68 Msun in a 10 pc periodic box, 10 kyr, 2 ranks.

Bars (the checker's, fixed in gizmo.conservation_trace; not set per test):
- mass and metal ledger: |delta| / value <= 1e-12 at every checkpoint;
- exact energy stages (feedback deposit, cooling): no STAGE line, i.e. measured change minus bookings within 1e-12 of E_abs + |booked|;
- time integration: |int_res| <= 1e-2 of the first sync's |E| + |E_pot|, or of the SN energy booked so far once the SN went off.
Also: the trace has SYNC lines, and the SN booked its 1e51 erg (b_fb_sn within 1e-6 of 1e51 erg, as the tables give for 10 Msun),
and the winds booked energy after the first sync (the trace's baseline; bookings before it are part of E_0).
Record: the first version (job 231729) had only the 10 Msun star, whose last kyr owes no wind after the first step; the 60 Msun
star was added after that run. In that run the feedback and cooling stages closed (no STAGE line); the integration residual was
+2.7e50 erg, a split's daughter missing from the merge/split bracket (closed before the rearrange), fixed in the trace.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command
from gizmo.conservation_trace import check_conservation_traces

NAME = "energy_step_trace_sn"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, NG, M_STAR, Z, N_H, T_DEATH = 10.0, 8, 10.0, 0.0142, 10.0, 1000.0
M_WIND, AGE_WIND = 60.0, 1.0e5
RUN_TIMEOUT = 3600  # s
UNIT_E = 1.989e33 * 1e10  # erg per code energy (Msun, km/s)


@pytest.fixture(scope="module")
def out():
    return build_and_run("run")


def build_and_run(tag, flags=()):
    """build (with extra Config flags) and run the problem in HERE/<tag>; returns its output directory"""
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME, extra_config_flags=flags)
    d = HERE / tag
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(Z)
    age = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - T_DEATH
    m_gas = N_H * 1.6726e-24 / 0.7065 * (3.085678e18 ** 3) / 1.989e33 * L ** 3 / NG ** 3
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(NG, L), [(M_STAR, age, (0.5 * L + 0.11, 0.5 * L + 0.07, 0.5 * L)),
                                                                            (M_WIND, AGE_WIND, (0.25 * L + 0.11, 0.25 * L + 0.07, 0.5 * L))],
                            gas_mass=m_gas, z=Z, T=100.0, hsml=2 * L / NG, tracabund=np.tile([0.0, 1.0e-4, 0.0], (NG ** 3, 1)))
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text().replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    (d / f"{NAME}.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=o,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=RUN_TIMEOUT)
    assert "Final time=" in (d / "run.out").read_text(), "run did not reach TimeMax"
    return d / "output"


def sync_lines(out):
    rows = []
    for line in open(out / "energy_step_trace.txt"):
        if line.startswith("SYNC"):
            rows.append({k: float(v) for k, v in (t.split("=", 1) for t in line.split()[1:]) if v not in ("nan",)})
    return rows


def test_traces_close(out):
    rep = check_conservation_traces(str(out))
    print(rep.summary())
    assert rep.ok, rep.summary()


def test_sn_and_winds_booked(out):
    s = sync_lines(out)
    assert len(s) >= 2
    e_sn = s[-1]["b_fb_sn"] * UNIT_E
    print(f"{len(s)} syncs; booked SN {e_sn:.6e} erg, winds {s[-1]['b_fb_wind'] * UNIT_E:.3e} erg; last int_res {s[-1]['int_res'] * UNIT_E:+.3e} erg")
    assert abs(e_sn / 1e51 - 1) < 1e-6 and s[-1]["b_fb_wind"] > 0


def test_heat_after_chemistry_is_reported():
    """bar (written before the run): heat added after the chemistry (TEST_COOLING_ADDS_1PCT_HEAT_AFTER_CHEMISTRY, u x 1.01 of every
    cooled cell) is unbooked, so the exact cooling stage reports it: at least one cooling STAGE finding (the default run has none,
    test_traces_close). Runs last: its build replaces the default binary (stella_dev 2758c8713)"""
    rep = check_conservation_traces(str(build_and_run("run_heat_after_chemistry", ("TEST_COOLING_ADDS_1PCT_HEAT_AFTER_CHEMISTRY",))))
    print(rep.summary())
    assert any(f["kind"] == "energy_stage" and f["stage"] == "cooling" for f in rep.findings), rep.summary()
