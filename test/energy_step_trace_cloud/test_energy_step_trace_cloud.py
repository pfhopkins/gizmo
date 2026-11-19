"""energy_step_trace_cloud: the energy and mass step traces close in a production cloud that forms single stars (the SF stage:
heat of the mass a new star takes from its parent, donors' heat and kinetic energy, the star's kinetic energy from the donors'
momentum, CR energy that leaves with the donated gas).

Problem: Stella cloud production flags and parameters (as stella_sfr_log), cloud IC from STELLA_CLOUD_IC (a MakeCloud cloud,
e.g. gmc_Z1_m1.0, 12377 cells), 2.7 Myr, 2 ranks.

Bars (the checker's, fixed in gizmo.conservation_trace): mass and metal ledger 1e-12 per checkpoint; no STAGE line of an exact stage
(feedback deposit, cooling, star formation); integration residual 1e-2 of the first sync's |E| + |E_pot| (or of the booked SN energy).
Also: stars formed, i.e. the SF stage booked heat locked into stars.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command
from gizmo.conservation_trace import check_conservation_traces

NAME = "energy_step_trace_cloud"
HERE = Path(__file__).parent
RUN_TIMEOUT = 10800  # s


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    if not os.environ.get("STELLA_CLOUD_IC"):
        pytest.skip("set STELLA_CLOUD_IC to a Stella cloud IC (HDF5)")
    T = get_stellar_evolution_tables()["StellarTablesFile"]
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    d = HERE / "run"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    ic = Path(os.environ["STELLA_CLOUD_IC"])
    os.symlink(ic, d / ic.name)
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    sub = {"InitCondFile": ic.name[:-5] if ic.name.endswith(".hdf5") else ic.name, "StellarTablesFile": T,
           "SGChemTablesPath": os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(T))}
    p = [f"{l.split()[0]} {sub[l.split()[0]]}" if l.split() and l.split()[0] in sub else l
         for l in (HERE / f"{NAME}.params").read_text().splitlines()]
    (d / f"{NAME}.params").write_text("\n".join(p) + "\n")
    with open(d / "run.out", "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=RUN_TIMEOUT)
    return d / "output"




def last_sync(out):
    rows = [l for l in open(out / "energy_step_trace.txt") if l.startswith("SYNC")]
    return {k: float(v) for k, v in (t.split("=", 1) for t in rows[-1].split()[1:]) if v != "nan"}


def test_stars_formed(run):
    s = last_sync(run)
    print(f"SF bookings: th_locked {s['b_sf_th_locked']:+.4e} donor {s['b_sf_donor']:+.4e} star_kin {s['b_sf_star_kin']:+.4e}; stage {s['s_sf']:+.4e}")
    assert s["b_sf_th_locked"] < 0 or s["b_sf_donor"] < 0


def test_traces_close(run):
    rep = check_conservation_traces(str(run))
    print(rep.summary())
    assert rep.ok, rep.summary()
