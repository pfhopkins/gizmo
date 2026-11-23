"""Type Ia M_Ch matter with live dust (Config_ia_mch_dust.sh: five-element dust model, TEST_SINGLE_STARS_PRINT_IA_DUST_REMOVAL), on the
stars of single_stars_metal_origin: the 3 Msun AGB death puts dusty ejecta into its octant, then its white dwarf explodes as a Type Ia
there, so converted cells hold dust. The converted gas keeps no dust (stella_dev resolvedism_fb_thermal.cc:687).
Bars (written before the run): at least one converted cell holds dust before its conversion; every converted cell: 0 < f <= 1 and dust
metal and the dust species sum after = before x (1 - f) to 1e-14 relative; at the end no cell's dust metal exceeds its metal mass
fraction (1e-12)."""
import os
import re
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables
from test_single_stars_metal_origin import TEST_NAME, TEST_DIR, run

LINE = re.compile(r"IA_DUST_REMOVAL ID=(\d+) f=(\S+) dust_metal (\S+) -> (\S+) species (\S+) -> (\S+)")


@pytest.fixture(scope="module")
def tables_dust():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME, config_file="Config_ia_mch_dust.sh")
    return t


def test_ia_conversion_removes_the_converted_gas_dust(tables_dust):
    d = run("events_mch_dust", tables_dust, {"TimeMax": 5e-3, "TimeBetSnapshot": 5e-3})
    rows = [tuple(float(x) for x in m.groups()[1:]) for m in LINE.finditer((d / f"test_{TEST_NAME}.out").read_text())]
    print(f"converted cells: {len(rows)}, with dust before: {sum(r[1] > 0 for r in rows)}")
    assert any(r[1] > 0 for r in rows)
    for f, d0, d1, s0, s1 in rows:
        assert 0 < f <= 1
        assert abs(d1 - d0 * (1 - f)) <= 1e-14 * max(d0, 1e-300) and abs(s1 - s0 * (1 - f)) <= 1e-14 * max(s0, 1e-300), (f, d0, d1, s0, s1)
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        z = g["Metallicity"][()][:, 0]
        dz = g["DustMetallicity"][()][:, 0]
    assert np.all(dz <= z * (1 + 1e-12) + 1e-300)
