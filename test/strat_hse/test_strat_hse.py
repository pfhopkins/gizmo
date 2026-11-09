"""strat_hse: an isothermal slab in hydrostatic equilibrium in a stratified shearing box stays in equilibrium.

Exercises GRAVITY_ANALYTIC_STRATIFIED_DISK (stellar disk + DM vertical gravity), GRAVITY_PERIODIC_XY (the gas slab's
own gravity from x/y shear-periodic Ewald images) and BOX_SHEARING together. The IC is the 1D HSE solution sampled in
layers, which excites a small vertical breathing mode (~5% in the rms height), so the checks use averages over the
t > 0 snapshots: the rms height, and the kernel density relative to rho_HSE(z) in |z| bins out to 2 H.
Measured: <H>/H_HSE - 1 = +0.04, worst bin +0.07. With nearest-image gravity only (GRAVITY_PERIODIC_XY
removed, an isolated 1 kpc tile) the slab under-weighs itself: +0.175 and +0.28.
"""

import numpy as np
import h5py
from pathlib import Path

from gizmo.test import build_and_run_test, default_mpi_ranks, get_final_snapshot, assert_final_time

TEST_NAME = "strat_hse"
TEST_DIR = Path(__file__).parent


def test_strat_hse():
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_strat_hse_ics import make_strat_hse_ics, profile, L, LONG_Z
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        make_strat_hse_ics(str(ic))
    build_and_run_test(TEST_NAME, min(default_mpi_ranks(), 4), 1)
    snap = get_final_snapshot(TEST_NAME)
    assert_final_time(snap, TEST_NAME)

    zg, rho = profile()
    H_ref = np.sqrt(np.trapezoid(rho * zg**2, zg) / np.trapezoid(rho, zg))
    edges = np.linspace(0, 2 * H_ref, 9)
    H, ratio, vz = [], [], []
    for fn in sorted(Path(snap).parent.glob("snapshot_*.hdf5"))[1:]:
        with h5py.File(fn, "r") as F:
            x, v, m, rh = (F[f"PartType0/{k}"][:] for k in ("Coordinates", "Velocities", "Masses", "Density"))
        z = np.abs(x[:, 2] - 0.5 * LONG_Z * L)
        H.append(np.sqrt(np.average(z**2, weights=m)))
        r, b = rh / np.interp(z, zg, rho), np.digitize(z, edges) - 1
        ratio.append([np.average(r[b == i], weights=m[b == i]) for i in range(len(edges) - 1)])
        vz.append(np.sqrt(np.average(v[:, 2] ** 2, weights=m)))
    H_err = np.mean(H) / H_ref - 1
    prof_err = np.abs(np.mean(ratio, axis=0) - 1)
    print(f"<H> = {np.mean(H):.1f} pc vs HSE {H_ref:.1f} pc ({H_err:+.3f}); <rho/rho_HSE> - 1 per bin {np.round(np.mean(ratio, axis=0) - 1, 3)}; "
          f"rms v_z {np.round(vz, 2)} km/s")
    assert abs(H_err) < 0.06
    assert prof_err.max() < 0.10
    assert max(vz) < 0.15 * 10.0  # residual vertical motions well below c_s
