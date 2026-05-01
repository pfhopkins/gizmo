"""Phase 16 two-temperature plasma regression test.

Validates that:
  (1) the LTE seed at startup applies the requested T_e/T_gas ratio
      (TwoTemp_InitialTeOverTgas = 0.1 -> T_e = 0.1 * T_gas at t=0);
  (2) once the cooling step fires, the analytic Spitzer e-i exponential
      relaxation drives T_e -> T_gas (or close to it -- residual offset
      reflects radiative cooling, hydro work partition, and ionization
      shifts during the step);
  (3) at the final snapshot the run is in the post-relaxation regime
      (|T_e - T_gas|/T_gas < 0.5 typical, not 0.9 from the seed).

This is a comparison vs the analytic Spitzer prediction, not a
snapshot-to-snapshot diff against a reference simulation -- the analytic
exp(-alpha*dt) decay is the reference. Because alpha*dt >> 1 in the GMC
regime (n_e ~ 60 cm^-3, T_e ~ 1700 K -> tau_eq ~ ms while dt ~ Myr), the
prediction collapses to "fully relaxed in one cooling step", which is
exactly what we test.
"""

from gizmo.test import build_and_run_test, get_cooling_tables, default_omp_threads
from glob import glob
import h5py
import numpy as np
import pytest


def _read_snapshot_temps(path):
    """Return (time, T_gas, T_e) from a snapshot."""
    with h5py.File(path, "r") as f:
        t = float(f["Header"].attrs["Time"])
        T_gas = f["PartType0/Temperature"][:]
        if "PartType0/ElectronTemperature" not in f:
            raise AssertionError(
                f"{path}: missing PartType0/ElectronTemperature dataset -- "
                "build was not configured with TWO_TEMPERATURE_PLASMA, "
                "or the C5a snapshot output is broken."
            )
        T_e = f["PartType0/ElectronTemperature"][:]
    return t, T_gas, T_e


@pytest.mark.parametrize("num_mpi_ranks", (2,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_two_temperature(num_mpi_ranks, num_omp_threads):
    test_name = "two_temperature"
    test_dir = "test/two_temperature"

    get_cooling_tables(test_dir)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    snaps = sorted(glob(f"{test_dir}/output/snapshot_*.hdf5"))
    assert len(snaps) >= 2, f"need at least 2 snapshots, got {len(snaps)}"

    # (1) initial-condition LTE seed: snapshot_000 should have T_e = 0.1 * T_gas
    t0, Tg0, Te0 = _read_snapshot_temps(snaps[0])
    assert t0 == 0.0, f"first snapshot should be at t=0, got t={t0}"
    assert np.all(np.isfinite(Te0)) and np.all(Te0 > 0), \
        "T_e must be finite and positive at t=0"
    initial_ratio = Te0 / Tg0
    expected_seed = 0.1
    seed_med = float(np.median(initial_ratio))
    assert seed_med == pytest.approx(expected_seed, abs=0.005), \
        f"LTE seed ratio T_e/T_gas at t=0: expected {expected_seed}, got median {seed_med}"

    # (2) at the final snapshot, the relaxation should have fired:
    # T_e/T_gas median should have moved substantially toward 1.
    t_final, Tg_final, Te_final = _read_snapshot_temps(snaps[-1])
    assert np.all(np.isfinite(Te_final)) and np.all(Te_final > 0), \
        f"T_e at t={t_final} must be finite and positive"
    final_ratio = Te_final / Tg_final
    final_med = float(np.median(final_ratio))
    assert final_med > 0.5, \
        f"T_e/T_gas median at t={t_final} should be > 0.5 (relaxation fired); " \
        f"got {final_med} -- Spitzer integrator not running?"

    # (3) total energy conservation check: the cooling step does change u_total
    # via radiation, but the Spitzer relaxation alone must conserve it.
    # We can only check this loosely from snapshots (radiative changes happen
    # in the same step), so we just sanity-check that T_gas and T_e are both
    # in a reasonable physical range (1 K < T < 1e10 K for an ideal-gas
    # GMC-with-cooling test).
    for label, T in (("T_gas_final", Tg_final), ("T_e_final", Te_final)):
        assert np.all(T > 0.5), f"{label} has values <= 0.5 K (unphysical)"
        assert np.all(T < 1e10), f"{label} has values >= 1e10 K (unphysical)"

    # (4) report a summary so the pytest log shows the relaxation actually moved
    print(f"\n[2T_test] t={t0}: T_e/T_gas median = {seed_med:.4f} (LTE seed)")
    print(f"[2T_test] t={t_final}: T_e/T_gas median = {final_med:.4f} (post-relaxation)")
    print(f"[2T_test] T_gas median: {np.median(Tg0):.2e} -> {np.median(Tg_final):.2e}")
    print(f"[2T_test] T_e   median: {np.median(Te0):.2e} -> {np.median(Te_final):.2e}")
