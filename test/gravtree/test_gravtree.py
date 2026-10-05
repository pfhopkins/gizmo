"""Gravity tree accuracy: the production opening criterion against a much tighter tree walk.

Each variant is built once and run twice from the same IC: with the parameterfile's own
(production) opening criterion, and with the walk tightened towards direct summation
(ErrTolTheta=0.15, ErrTolForceAcc=1e-6). Both runs take one step; the comparison uses
snapshot_000, which holds the start-up force evaluation, so the difference is the tree error
alone, before any evolution can diverge. Particles are matched by ParticleIDs.

For the gravitational acceleration (OUTPUT_ACCELERATION with OUTPUT_HYDROACCELERATION, so the
hydro term is excluded) and the potential, the tight walk is a converged reference, and the test
bounds the median and 99th percentile of the per-particle relative error: a defect in the walk
(a missed node, a wrong softening or periodic image) shows up as a large error on some subset of
particles even when the median stays small.

Two variants also compare quantities gathered opportunistically on the same walk. For these the
tight walk is only a tighter-walk reference, not a physical truth, because the walk's openings
are chosen for force accuracy alone:
  - Sink_Distance (SINK_CALC_DISTANCES) is the distance to the nearest sink the walk encounters.
    Outside the short-range region whether a cell encounters the sink at all depends on which
    nodes are opened (the PM cull is a per-axis box test, so large accepted nodes reach further),
    so values are compared only where both walks report one, and there they must agree exactly.
  - PhotonEnergy (GALSF_FB_FIRE_RT_LONGRANGE) is a deliberately crude long-range radiation
    estimate carried on the gravity walk, with no luminosity term in the opening criterion and no
    converged solution to compare against; a heavy tail in the per-cell error is expected. Only
    the median and the energy-weighted error are gated.

Variants (extra Config flags on top of Config.sh, and the parameterfile each runs):
  default        evrard IC, fixed softening                            gravtree.params
  evalpotential  + EVALPOTENTIAL/OUTPUT_POTENTIAL: potential on the walk gravtree_evalpotential.params
  pmgrid         + BOX_PERIODIC, GRAVITY_NOT_PERIODIC, PMGRID=64       gravtree_pmgrid.params
  sinks          isodisk IC with a sink: SINK_CALC_DISTANCES            gravtree_sinks.params
  rt             isodisk IC with stars: GALSF_FB_FIRE_RT_LONGRANGE      gravtree_rt.params

Tolerances are about 3x the values measured in recorded runs (see the README).
"""

from os import chdir

import h5py
import numpy as np
import pytest

from gizmo.test import (
    build_gizmo_for_test,
    default_mpi_ranks,
    download_test_files,
    get_cooling_tables,
    parse_params,
    run_test,
)

TEST_NAME = "gravtree"

GALAXY_FLAGS = ("COOLING", "GALSF", "GALSF_SFR_CRITERION=(1+256)", "METALS",
                "BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "PMGRID=64", "ADAPTIVE_GRAVSOFT_FORGAS")

# Checks per quantity: "pointwise" bounds the (median, p99) of the per-particle relative error;
# "shared_finite" bounds the maximum relative difference where both walks report a value;
# "aggregate" bounds the (median, energy-weighted) relative error.
ACC = ("Acceleration", "pointwise")
VARIANTS = {
    "default": ((), "gravtree",
                {ACC: (4e-3, 7e-3)}),
    "evalpotential": (("EVALPOTENTIAL", "OUTPUT_POTENTIAL"), "gravtree_evalpotential",
                      {ACC: (4e-3, 7e-3),
                       ("Potential", "pointwise"): (1e-3, 2e-3)}),
    "pmgrid": (("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "PMGRID=64"), "gravtree_pmgrid",
               {ACC: (1e-2, 1.5e-2)}),
    "sinks": (GALAXY_FLAGS + ("SINK_CALC_DISTANCES", "OUTPUT_SINK_DISTANCES", "SINGLE_STAR_SINK_DYNAMICS",
                              "SINGLE_STAR_TIMESTEPPING=0", "SINGLE_STAR_FIND_BINARIES",
                              "SINGLE_STAR_FB_TIMESTEPLIMIT"), "gravtree_sinks",
              {ACC: (5e-3, 1e-2),
               ("Sink_Distance", "shared_finite"): (1e-6,)}),
    "rt": (GALAXY_FLAGS + ("GALSF_FB_FIRE_RT_LONGRANGE",), "gravtree_rt",
           {ACC: (5e-3, 1e-2),
            ("PhotonEnergy", "aggregate"): (2e-2, 7e-2)}),
}

TIGHT = {"ErrTolTheta": 0.15, "ErrTolForceAcc": 1e-6}  # ErrTolTheta must lie in (0.1, 0.9)
NO_SINK = 1e27  # Sink_Distance keeps a huge sentinel where the walk met no sink


def _load(snapfile, field, ptype="PartType0"):
    with h5py.File(snapfile, "r") as F:
        if ptype not in F or field not in F[ptype]:
            raise KeyError(f"{snapfile} has no {ptype}/{field}")
        ids = F[ptype]["ParticleIDs"][:]
        val = F[ptype][field][:]
    order = np.argsort(ids, kind="stable")
    return ids[order], val[order]


def _relative_error(a, b):
    """Per-particle |a-b|/|b| (vector norm for vector fields); |b| floored at 1e-3 of its median
    so the few particles where the reference passes through zero do not dominate."""
    num = np.linalg.norm(a - b, axis=1) if a.ndim == 2 else np.abs(a - b)
    den = np.linalg.norm(b, axis=1) if b.ndim == 2 else np.abs(b)
    floor = 1e-3 * np.median(den[den > 0]) if np.any(den > 0) else 1.0
    return num / np.maximum(den, floor)


def measure(field, kind, snap_prod, snap_ref):
    """Return (statistics, description) for one quantity; statistics line up with the limits."""
    ids_p, val_p = _load(snap_prod, field)
    ids_r, val_r = _load(snap_ref, field)
    assert np.array_equal(ids_p, ids_r), f"{field}: particle sets differ between the two runs"
    if kind == "shared_finite":
        both = (val_p < NO_SINK) & (val_r < NO_SINK)
        assert np.any(both), f"{field}: no cell saw a sink in both runs: the comparison would be vacuous"
        err = np.abs(val_p[both] - val_r[both]) / val_r[both]
        return (err.max(),), f"N_shared={both.sum()} of {len(val_r)} max={err.max():.3e}"
    assert np.any(val_r != 0), f"{field} is zero everywhere: the comparison would be vacuous"
    err = _relative_error(val_p, val_r)
    if kind == "aggregate":
        weighted = np.sum(np.abs(val_p - val_r)) / np.sum(np.abs(val_r))
        return (np.median(err), weighted), (f"N={len(err)} median={np.median(err):.3e} energy-weighted={weighted:.3e}"
                                            f" (p99={np.percentile(err, 99):.3e}, not gated)")
    return (np.median(err), np.percentile(err, 99)), (f"N={len(err)} median={np.median(err):.3e}"
                                                       f" p99={np.percentile(err, 99):.3e} max={err.max():.3e}")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_gravtree(variant):
    flags, params_name, checks = VARIANTS[variant]
    flags = ("OUTPUT_ACCELERATION", "OUTPUT_HYDROACCELERATION") + flags
    build_gizmo_for_test(TEST_NAME, 1, flags)
    chdir(f"test/{TEST_NAME}/")
    try:
        download_test_files(TEST_NAME, params_name)
        if "COOLING" in flags:
            get_cooling_tables()
        params = parse_params(f"{params_name}.params")
        outdir = params.get("OutputDir", "output")
        tight_dir = outdir + "_tight"
        dt = min(float(params["MaxSizeTimestep"]), float(params["TimeMax"]))
        one_step = {"TimeMax": dt, "TimeBetSnapshot": dt}
        run_test(TEST_NAME, default_mpi_ranks(), 1, params_name=params_name, param_overrides=one_step)
        run_test(TEST_NAME, default_mpi_ranks(), 1, params_name=params_name,
                 param_overrides=dict(one_step, OutputDir=tight_dir, **TIGHT))
    finally:
        chdir("../../")

    snap_prod = f"test/{TEST_NAME}/{outdir}/snapshot_000.hdf5"
    snap_ref = f"test/{TEST_NAME}/{tight_dir}/snapshot_000.hdf5"
    failures = []
    for (field, kind), limits in checks.items():
        stats, text = measure(field, kind, snap_prod, snap_ref)
        print(f"[{variant}] {field} ({kind}): {text}  limits {limits}")
        if any(s > lim for s, lim in zip(stats, limits)):
            failures.append(f"{field} ({kind}): {text}, limits {limits}")
    assert not failures, f"[{variant}] tree error above tolerance: " + "; ".join(failures)
