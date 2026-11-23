"""single_stars_metal_origin with GALSF_SINGLE_STARS_TYPE_IA_CHANDRASEKHAR_MATTER (Config_ia_mch.sh; the STELLA_DEFAULTS default since
GKS-100): each Type Ia is the table's M_Ch of Ia matter, M_Ch - M_WD of receiver gas converted in place with the ejecta's weights
(stella_dev 579e0bf93, STL-180). The conversion's per-slot change is written to ia_conversion_slots.txt.
Bars (written before the run): per logged source slot (SN, AGB, Ia) gas = events + that slot's conversion (1e-10 of the larger);
metals not from stars = initial + conversion of slot 0 - conversion of the origin slots (1e-12); the conversion adds Ia metals
(> 0) and clamps no receiver; every origin slot outside its star's octant stays exactly 0."""
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables
import os
from test_single_stars_metal_origin import (TEST_NAME, TEST_DIR, ORIGIN, SN, AGB, WIND, IA, KIND_OF_SLOT, ALLOWED, run, gas)


@pytest.fixture(scope="module")
def tables_mch():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME, config_file="Config_ia_mch.sh")
    return t


def test_metal_origin_with_ia_chandrasekhar_matter(tables_mch):
    d = run("events_mch", tables_mch, {"TimeMax": 5e-3, "TimeBetSnapshot": 5e-3})
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    conv_rows = np.atleast_2d(np.loadtxt(d / "output" / "ia_conversion_slots.txt"))
    n_clamped, conv = int(conv_rows[-1, 1]), conv_rows[-1, 2:]
    a, b = gas(snaps[0]), gas(snaps[-1])
    assert len(conv) == b["Z"].shape[1] == ORIGIN + 4
    M_src = (b["m"][:, None] * b["Z"][:, ORIGIN:]).sum(0)
    for s, kind in KIND_OF_SLOT.items():
        z_ev = rows[rows[:, 2] == kind, 9].sum()
        expect = z_ev + conv[ORIGIN + s]
        rel = (M_src[s] - expect) / max(abs(z_ev), abs(conv[ORIGIN + s]))
        print(f"source {s}: gas {M_src[s]:.12g} Msun, events {z_ev:.12g}, conversion {conv[ORIGIN + s]:+.6g}, rel {rel:.3g}")
        assert z_ev > 0 and abs(rel) <= 1e-10
    assert M_src[WIND] > 0
    M_init = (a["m"] * a["Z"][:, 0]).sum()
    M_rest = (b["m"] * (b["Z"][:, 0] - b["Z"][:, ORIGIN:].sum(1))).sum()
    expect_rest = M_init + conv[0] - conv[ORIGIN:].sum()
    print(f"metals not from stars: start {M_init:.15g}, end {M_rest:.15g}, expected {expect_rest:.15g}; Ia added {conv[ORIGIN + IA]:.6g} Msun, clamped {n_clamped}")
    assert abs(M_rest / expect_rest - 1) <= 1e-12
    assert conv[ORIGIN + IA] > 0 and n_clamped == 0
    octant = tuple((b["x"] // (0.5 * b["L"])).astype(int).T)
    for c in range(len(b["m"])):
        allowed = ALLOWED.get(tuple(int(o[c]) for o in octant), set())
        for s in range(4):
            if s not in allowed:
                assert b["Z"][c, ORIGIN + s] == 0, (c, s)
