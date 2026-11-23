"""Type Ia in the Stella x FIRE-3 coupled mode (GALSF_RESOLVEDISM_HYBRID_DEFAULTS): the metals live in the FIRE slots (Z, He, C, N, O,
Ne, Mg, Si, S, Ca, Fe; no H slot), and the Ia companion share converted in place must change every slot, not only Z.

Build: the coupled-mode configuration of the hybrid tests (M_SSP = 10 Msun, so 1 Msun cells form Stella stars) plus the
isolated-feedback test hooks, STELLA_SELFCHECK and the step traces. Uniform box as test/stella_fb_conservation (n_H = 10,
1e4 K, 1 Msun cells, 16 pc). One 3 Msun star dies as an AGB star at 1 kyr and explodes as a Type Ia at 3.5 kyr (both forced).

Checks:
  - IAFE: the Ia-composition iron delivered (each receiver's Fe-slot gain plus the Fe of its gas converted in place) equals the
    table's 0.744 Msun to 1e-12;
  - no STELLA_SELFCHECK fire_slot_deposit / fire_slot_invariant_cell FAIL (every slot of every receiver gains exactly what the
    deposit books), and the check ran;
  - the mass trace closes.
"""

import os
import re
import shutil
import subprocess
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_hybrid_ia"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
CONFIG = ("HYDRO_MESHLESS_FINITE_MASS", "GALSF_RESOLVEDISM_HYBRID_DEFAULTS=(10.)", "GALSF_RESOLVEDISM_INSTANT_SN", "N_STELLAR_MASS=5",
          "METALS", "MULTIPLEDOMAINS=16", "EVALPOTENTIAL", "DO_DENSITY_AROUND_NONGAS_PARTICLES", "OUTPUT_TEMPERATURE",
          "OUTPUT_IN_DOUBLEPRECISION", "GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY", "BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "STELLA_SELFCHECK",
          "OUTPUT_ENERGY_STEP_TRACE")
BAR = 1e-12
RUN_TIMEOUT = 5400  # s


@pytest.mark.xfail(strict=True, reason="the coupled mode does not build on stella_dev: its preset defines GALSF_RESOLVEDISM_RADPRESSURE, "
                   "hence RT_NUV (precompiler_logic.h), FIRE-3 defines RT_USE_GRAVTREE_SAVE_RAD_ENERGY without RADTRANSFER, and "
                   "cooling.cc:1729-1750 then writes Lambda_RadiativeCooling_toRHDBins, which cell_data.h:451-457 declares only with "
                   "RADTRANSFER. A pass means it builds again: remove this mark.")
def test_hybrid_ia_iron():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fbc", join(REPO, "test", "stella_fb_conservation", "test_stella_fb_conservation.py"))
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    open(join(TEST_DIR, "Config.sh"), "w").write("\n".join(CONFIG) + "\n")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    d = join(TEST_DIR, "run"); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    shutil.copy(join(REPO, "cooling", "TREECOOL"), d)
    spcool = os.environ.get("SPCOOL_TABLES")
    if spcool:
        os.symlink(spcool, join(d, "spcool_tables"))
    else:
        get_cooling_tables(d)
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN)
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), d)
    with open(join(d, "run.log"), "w") as log:
        r = fbc.run_killable(mpi_launch_command(2) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"], RUN_TIMEOUT, stdout=log,
                             stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0 and "Simulation ends." in text, text[-1500:]
    rows = fbc.kv_lines(text, "IAFE")
    assert len(rows) == 1, f"expected one Ia, got {len(rows)}"
    err = abs(rows[0]["Fe"] / rows[0]["Fe_table"] - 1)
    fails = [l for l in text.splitlines() if re.match(r"STELLA_SELFCHECK fire_slot_(deposit|invariant_cell) FAIL", l)]
    print(f"Ia iron delivered {rows[0]['Fe']:.6f} Msun (table {rows[0]['Fe_table']:.6f}, rel. err {err:.3e}); {len(fails)} fire-slot selfcheck FAILs")
    assert "STELLA_SELFCHECK fire_slot_deposit RAN" in text, "the fire-slot receiver check did not run"
    assert not fails, fails[0]
    assert err < BAR
    rep = keep_and_check_traces(out, out, require_energy=False)
    assert rep.ok, rep.summary()
