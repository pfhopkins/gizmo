"""Plummer cluster with a realistic binary population -- conservation under a mixed population.

test/plummer_binaries puts every star in an identical 1000 AU equal-mass circular binary: one
point in parameter space, cleanly controlled. This test draws the population from the observed
distributions instead (Kroupa IMF, Duchene & Kraus binary fraction vs primary mass, Raghavan
log-normal periods, thermal eccentricities -- see make_plummer_binaries_realistic_ics.py), so a
single run spans decades in binary hardness, mass ratio and eccentricity at once.

WHY THAT MATTERS HERE. The Hermite source prediction and the shared timestep normalization both
act on timebin structure, and timebin structure is what a heterogeneous population produces in
abundance: stars of different mass on different bins, binaries of different hardness spanning
~8 bins, and -- unlike any equal-mass test -- SINGLE stars sharing the cluster with binaries, so
mixed-eligibility neighbours are routine rather than absent.

Measured from the IO_HERMITE_SYNC datasets, and the potential is recomputed from those same
positions rather than read from the snapshot's tree-evaluated Potential field. Pairing a
last-kick velocity with a drifted-position potential gives a kinetic and a potential term
belonging to different times; at N ~ 335 the exact O(N^2) sum costs milliseconds.

Tolerances are UNCALIBRATED as of writing -- they are placeholders pending a compute-node run.
The energy assertion is deliberately loose and the test's value until then is the reported
numbers, not its pass/fail.
"""

import glob
import re
from os import path

import h5py
import numpy as np
import pytest
import matplotlib
from matplotlib import pyplot as plt

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    variant_output_dir,
)

TEST_NAME = "plummer_binaries_realistic"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"

G_CODE = 4.300917270e-3      # pc (km/s)^2 / Msun
AU_PER_PC = 206264.806
PERIODIC = False             # no BOX_PERIODIC, and the cluster is ~1 pc in a 300 pc box

N_SYSTEMS = 256
A_CLUSTER = 1.0              # pc
BOXSIZE = 300.0
A_MIN_AU = 100.0             # pericentre floor; sets the runtime, see the params file
A_MAX_AU = 5.0e3
SEED = 42

# Placeholders. |dE/E| over the run; the equal-mass test holds ~1e-3 over 10x this duration, but
# this population is harder (higher eccentricities, deeper bin spread) and shorter, so the true
# value is unknown until measured. Recalibrate from a node run before trusting a pass.
MAX_DE_OVER_E = 5e-2
MAX_COM_DRIFT = 1e-3         # |v_com| / cluster velocity dispersion


def _ensure_ic():
    if path.isfile(IC_FILE):
        return
    import importlib.util
    test_dir = path.dirname(path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "make_realistic", path.join(test_dir, "make_plummer_binaries_realistic_ics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_realistic_ics(N_SYSTEMS, A_CLUSTER, BOXSIZE, SEED,
                           path.join(test_dir, f"{TEST_NAME}_ics.hdf5"),
                           a_min_au=A_MIN_AU, a_max_au=A_MAX_AU)


def _load(snap):
    """Synced (r,v) if present; the plain datasets are a mixed state (see the module docstring)."""
    with h5py.File(snap, "r") as F:
        t = float(F["Header"].attrs["Time"])
        g = F["PartType5"]
        synced = "HermiteSyncCoordinates" in g and "HermiteSyncVelocities" in g
        pos = g["HermiteSyncCoordinates"][:] if synced else g["Coordinates"][:]
        vel = g["HermiteSyncVelocities"][:] if synced else g["Velocities"][:]
        return t, g["Masses"][:], pos, vel, g["ParticleIDs"][:], synced


def _potential_energy(pos, mass):
    d = pos[:, None, :] - pos[None, :, :]
    if PERIODIC:
        d -= BOXSIZE * np.round(d / BOXSIZE)
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)                       # drop self-pairs
    return -0.5 * G_CODE * np.sum((mass[:, None] * mass[None, :]) / np.sqrt(r2))


def _trajectory(snaps):
    t, energy, drift, bound = [], [], [], []
    e0 = p0 = None
    for s in snaps:
        ti, m, pos, vel, ids, synced = _load(s)
        assert synced, "IO_HERMITE_SYNC datasets missing -- the metrics would be mixed-state"
        ke = 0.5 * np.sum(m * np.sum(vel ** 2, axis=1))
        E = ke + _potential_energy(pos, m)
        P = (m[:, None] * vel).sum(0)
        if e0 is None:
            e0, p0, sigma = E, P, np.sqrt(2.0 * ke / m.sum())
        t.append(ti)
        energy.append(abs(E / e0 - 1.0))
        drift.append(np.linalg.norm(P - p0) / m.sum() / sigma)
    return np.array(t), np.array(energy), np.array(drift)


def _binary_elements(snap, cat):
    """Semi-major axis and eccentricity of each catalogued pair, from the synced state."""
    _, m, pos, vel, ids, _ = _load(snap)
    idx = {int(i): k for k, i in enumerate(ids)}
    a, ecc = [], []
    for pid, sid in zip(cat["PrimaryID"], cat["SecondaryID"]):
        if int(pid) not in idx or int(sid) not in idx:
            continue                                   # merged or accreted away
        i, j = idx[int(pid)], idx[int(sid)]
        M = m[i] + m[j]
        dr, dv = pos[j] - pos[i], vel[j] - vel[i]
        r = np.linalg.norm(dr)
        aa = 1.0 / (2.0 / r - (dv @ dv) / (G_CODE * M))
        h = np.cross(dr, dv)
        a.append(aa)
        ecc.append(np.sqrt(max(0.0, 1.0 - (h @ h) / (G_CODE * M * aa))) if aa > 0 else np.nan)
    return np.array(a), np.array(ecc)


def _plot(t, energy, drift, a0, a1, variant_id):
    matplotlib.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(t, energy, lw=1.0)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t [code]"); ax[0].set_ylabel("|E/E$_0$ - 1|")
    ax[1].plot(t, drift, lw=1.0)
    ax[1].set_yscale("log"); ax[1].set_xlabel("t [code]")
    ax[1].set_ylabel(r"|v$_{com}$| / $\sigma$")
    k = np.isfinite(a0) & np.isfinite(a1) & (a0 > 0) & (a1 > 0)
    ax[2].loglog(a0[k] * AU_PER_PC, a1[k] * AU_PER_PC, ".", ms=4)
    lim = [min(a0[k].min(), a1[k].min()) * AU_PER_PC, max(a0[k].max(), a1[k].max()) * AU_PER_PC]
    ax[2].plot(lim, lim, "k-", lw=0.8)
    ax[2].set_xlabel("a initial [AU]"); ax[2].set_ylabel("a final [AU]")
    fig.tight_layout()
    fig.savefig(f"{TEST_DIR}/{TEST_NAME}_{variant_id}_summary.png", dpi=120)
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (2,))
@pytest.mark.parametrize("num_omp_threads", (1,))
@pytest.mark.parametrize("extra_config_flags", [pytest.param((), id="starforge_defaults")])
def test_plummer_binaries_realistic(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    _ensure_ic()
    clean_test_outputs(TEST_NAME, extra_config_flags)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    outdir = variant_output_dir(TEST_NAME, extra_config_flags)
    snaps = sorted(glob.glob(outdir + "/snapshot_*.hdf5"),
                   key=lambda f: int(re.search(r"snapshot_(\d+)", f).group(1)))
    assert len(snaps) >= 16, f"only {len(snaps)} snapshots -- the run died early"

    t, energy, drift = _trajectory(snaps)
    assert np.all(np.diff(t) >= 0), "snapshot times are not monotonic -- file ordering is wrong"

    with h5py.File(IC_FILE, "r") as F:
        cat = {k: F["BinaryCatalog"][k][:] for k in
               ("PrimaryID", "SecondaryID", "SemiMajorAxis_AU", "Eccentricity")}
    a_first, e_first = _binary_elements(snaps[0], cat)
    a_last, e_last = _binary_elements(snaps[-1], cat)

    variant_id = request.node.callspec.id.split("-")[0]
    _plot(t, energy, drift, a_first, a_last, variant_id)
    np.savez(f"{TEST_DIR}/summary_{variant_id}.npz", t=t, energy=energy, drift=drift,
             a_first=a_first, a_last=a_last, e_first=e_first, e_last=e_last)

    k = np.isfinite(a_first) & np.isfinite(a_last) & (a_first > 0) & (a_last > 0)
    surv = int(k.sum())
    da = np.abs(a_last[k] / a_first[k] - 1.0)
    print(f"  {len(snaps)} snapshots to t={t[-1]:.3f}, {len(cat['PrimaryID'])} catalogued binaries")
    print(f"  |dE/E|     final {energy[-1]:.3e}   max {energy.max():.3e}")
    print(f"  COM drift  final {drift[-1]:.3e}   (in units of the cluster dispersion)")
    print(f"  binaries   {surv} still bound; median |da/a| = {np.median(da):.3e}, "
          f"90th pct {np.percentile(da, 90):.3e}")
    print(f"  a range    {a_first[k].min()*AU_PER_PC:.0f} - {a_first[k].max()*AU_PER_PC:.0f} AU;"
          f"  e median {np.median(e_first[np.isfinite(e_first)]):.2f}")

    assert energy[-1] < MAX_DE_OVER_E, (
        f"energy error {energy[-1]:.3e} over t={t[-1]:.2f} (tol {MAX_DE_OVER_E}). Measured on the "
        f"synced state with a direct potential, so this is the integrator, not the output "
        f"convention or the tree's opening error.")
    assert drift[-1] < MAX_COM_DRIFT, (
        f"spurious COM velocity {drift[-1]:.3e} of the cluster dispersion (tol {MAX_COM_DRIFT}). "
        f"The ICs are built in the exact COM frame, so this is entirely integration error.")
