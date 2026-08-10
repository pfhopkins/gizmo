"""Shared momentum-drift diagnostics for isolated self-gravitating test problems.

Tree gravity makes O(theta^2) force errors. For a system at rest in the simulation frame each
particle sits in nearly the same place relative to the tree nodes from step to step, so those
errors are correlated in time and do not average away: the residual acts like a small net
external force and the centre of mass slowly accelerates. RANDOMIZE_GRAVTREE decorrelates
them by moving the tree geometry each rebuild, turning that secular drift into a much smaller
random walk. See Weinberger, Springel & Pakmor 2020 (arXiv:1909.04667) sec 3.1 and Grudic+
2021 (arXiv:2010.11254).

Metric -- the spurious COM velocity in units of the system's gravitational (virial) velocity:

    drift(t) = |v_com(t) - v_com(0)| / sqrt(|W|/M),    W = 0.5*sum(m*phi) at t=0

Dimensionless and unit-independent. Normalizing by |W| rather than the initial v_rms uses a
scale set by the mass distribution, so it is meaningful for cold starts too (see
momentum_trajectory). Using the change from t=0 means any net momentum in the ICs cancels.
Assumes non-cosmological runs (true of the isolated-halo/collapse tests this serves);
snapshot velocities would need a-factors otherwise.
"""

import glob
import warnings
from os import path

import h5py
import numpy as np

# Sanity ceiling, NOT a tuned tolerance: a spurious COM velocity comparable to the
# internal velocity dispersion means the system is being pushed across the box by force
# errors, which no correct configuration should do. Deliberately loose -- the point of this
# module is the measured comparison, not this bound.
DRIFT_SANITY_CEILING = 1.0

# Regression ceiling for the RANDOMIZE_GRAVTREE ("randomize*") variants only.
#
# A ceiling on the randomized drift rather than a "randomized beats flag-off" ratio: the
# flag-OFF drift is a lottery (correlated errors can happen to be small -- it spans 30x
# between plummer 9.6e-4 and hernquist 2.9e-2), so a ratio assertion would be flaky. The
# randomized drift is a bounded random walk and came out tight and repeatable.
#
# Keyed per (test, variant) because within one problem the non-periodic and periodic-TreePM
# drifts differ ~10x; one per-test number would leave 20-35x headroom on the best variants.
# Each is ~3-7x above the measured value (to tolerate random-walk scatter) and, where
# possible, below the corresponding flag-off drift so a silent revert is caught:
#
#   (test, variant)             measured   ceiling  headroom  flag-off  catches revert?
#   plummer/randomize           1.42e-4    1.0e-3     7.1x    3.25e-3   yes
#   plummer/randomize_pmgrid    1.37e-3    5.0e-3     3.6x    9.68e-4   NO -- see below
#   hernquist/randomize         2.26e-4    1.0e-3     4.4x    7.62e-3   yes
#   hernquist/randomize_pmgrid  1.13e-3    4.0e-3     3.5x    2.93e-2   yes
#   evrard/randomize            8.34e-6    1.7e-5     2.0x    3.13e-5   yes (1.8x below)
#   shu1977/randomize           5.34e-4    1.1e-3     2.1x    6.36e-4   NO (only 1.2x above)
#
# All values are under the v_grav normalization. plummer/hernquist moved <1% from the old
# v_rms(0) numbers (they are virialized, 2K/|W| ~ 1.0); the cold-start tests moved by ~80x,
# because dividing by a near-zero v_rms(0) had inflated them -- evrard was 1.8e-2 before.
#
# evrard and shu1977 are set to 2x their measured value, tighter than the 3.5-7x used above.
# That is deliberate but carries more flake risk: it rests on a SINGLE measurement each, with
# no run-to-run scatter estimate. For comparison, plummer/randomize varied 16% between 8 and
# 48 ranks, so 2x is ~6x that scatter -- but the cold collapses have not been repeated, and
# shu1977 in particular improves only 1.19x over its flag-off run, so its ceiling sits close
# to the flag-off value and cannot distinguish a revert. Loosen to 3-4x if either proves flaky.
#
# plummer/randomize_pmgrid is the one pair whose flag-OFF drift came out BELOW the randomized
# value (that run drew a low value; the same config in hernquist drifted 30x more), so it
# catches only gross regressions. Do not "fix" it by tightening below the measured value.
RANDOMIZED_DRIFT_CEILING = {
    ("plummer", "randomize"): 1.0e-3,
    ("plummer", "randomize_pmgrid"): 5.0e-3,
    # hernquist runs to TimeMax=11.8 (~1 crossing) rather than 118, so these are the
    # long-run ceilings divided by 4.6 -- the measured shrink factor, from COM drift going
    # as t^0.58 (random walk) on the baseline variant. NOT 10x: only the secular quantities
    # scale linearly with time. Other tests in this table still run long; leave them alone.
    ("hernquist", "randomize"): 2.2e-4,
    ("hernquist", "randomize_pmgrid"): 8.6e-4,
    ("evrard", "randomize"): 1.7e-5,
    ("shu1977", "randomize"): 1.1e-3,
}
# for any (test, variant) not tabulated -- loose, since there is no measurement to justify more
RANDOMIZED_DRIFT_CEILING_DEFAULT = 5.0e-2


def assert_randomized_drift(test_name, variant_id, final_drift):
    """Regression guard applied ONLY to RANDOMIZE_GRAVTREE variants: the randomized COM
    drift must stay within the calibrated ceiling for this (test, variant). No-op for
    flag-off variants, whose drift is erratic by nature and must not gate CI."""
    if "randomize" not in variant_id:
        return
    key = (test_name, variant_id)
    ceiling = RANDOMIZED_DRIFT_CEILING.get(key, RANDOMIZED_DRIFT_CEILING_DEFAULT)
    calibrated = "calibrated" if key in RANDOMIZED_DRIFT_CEILING else "default (uncalibrated)"
    assert final_drift < ceiling, (
        f"[{test_name}/{variant_id}] RANDOMIZE_GRAVTREE COM drift {final_drift:.3e} exceeds "
        f"the {calibrated} regression ceiling {ceiling:.1e}. The randomized path is supposed to "
        f"keep the spurious COM drift bounded (a random walk, not a secular drift); this "
        f"suggests the error decorrelation has regressed."
    )


def _read_snapshot_pv(snap, parttype):
    """Return (time, velocities, masses, potentials) for one snapshot; potentials is None if
    any included type lacks a Potential field.

    `parttype` may be a single "PartTypeN" string or a sequence of them, in which case the
    particles are concatenated. Summing over ALL types that carry momentum is essential for
    problems where particles change type mid-run -- e.g. shu1977, where gas is accreted into a
    sink: measuring gas alone would register the physical gas->sink momentum transfer as a huge
    spurious "drift". Types absent from a snapshot (a sink not yet formed) are skipped.
    """
    parttypes = (parttype,) if isinstance(parttype, str) else tuple(parttype)
    vels, masses, pots = [], [], []
    have_pot = True
    with h5py.File(snap, "r") as F:
        t = float(F["Header"].attrs["Time"])
        for pt in parttypes:
            if pt not in F or "Velocities" not in F[pt]:
                continue
            grp = F[pt]
            vel = np.asarray(grp["Velocities"][:], dtype=float)
            if vel.size == 0:
                continue
            if "Masses" in grp:
                mass = np.asarray(grp["Masses"][:], dtype=float)
            else:  # equal-mass particles are stored only in the header MassTable
                idx = int(pt.replace("PartType", ""))
                mass = np.full(vel.shape[0], float(F["Header"].attrs["MassTable"][idx]))
            vels.append(vel)
            masses.append(mass)
            if "Potential" in grp:
                pots.append(np.asarray(grp["Potential"][:], dtype=float))
            else:
                have_pot = False   # partial coverage is useless: W must span the same set
    if not vels:
        raise RuntimeError(f"no particles of type(s) {parttypes} found in {snap}")
    pot = np.concatenate(pots, axis=0) if (have_pot and len(pots) == len(vels)) else None
    return t, np.concatenate(vels, axis=0), np.concatenate(masses, axis=0), pot


def momentum_trajectory(snap_paths, parttype="PartType1"):
    """Compute the COM-velocity drift trajectory over a sequence of snapshots.

    Normalization is the gravitational (virial) velocity of the initial state,
    v_grav = sqrt(|W|/M) with W = 0.5*sum(m*phi). This is a fixed scale set by the mass
    distribution rather than by the initial kinetic state, which matters because the two
    coincide only for virialized systems: by the virial theorem 2K = |W| gives
    v_rms(0) = sqrt(2K/M) = v_grav (verified to 0.4% on plummer/hernquist, periodic and not).
    For a COLD start -- evrard, shu1977, where 2K/|W| ~ 0 -- v_rms(0) is near zero and would
    inflate the metric by ~80x while the absolute momentum error is actually small. Falls back
    to v_rms(0), with a warning, if Potential is not in the snapshots.

    Returns a dict with:
      times     : snapshot times
      drift     : |v_com(t) - v_com(0)| / norm   (dimensionless)
      vcom      : (N,3) COM velocity at each time
      norm      : the velocity scale actually used
      norm_kind : "v_grav" or "v_rms0" (which one that was)
      vrms0     : mass-weighted rms speed in the first snapshot
      vgrav     : sqrt(|W|/M) in the first snapshot (nan if Potential unavailable)
      virial0   : 2K/|W| in the first snapshot -- ~1 virialized, ~0 cold (nan if unavailable)
      ptot      : |total momentum| at each time (code units)
      mtot      : total mass included at each time -- should be constant; if not, the particle
                  types passed in miss something that carries momentum
    """
    times, vcoms, ptots, mtots = [], [], [], []
    vrms0 = vgrav = virial0 = None
    for s in snap_paths:
        t, vel, mass, pot = _read_snapshot_pv(s, parttype)
        mtot = mass.sum()
        p = (mass[:, None] * vel).sum(axis=0)
        vcoms.append(p / mtot)
        ptots.append(np.linalg.norm(p))
        mtots.append(mtot)
        times.append(t)
        if vrms0 is None:
            ke2 = (mass * np.sum(vel**2, axis=1)).sum()       # = 2K
            vrms0 = float(np.sqrt(ke2 / mtot))
            if pot is not None and pot.shape[0] == mass.shape[0]:
                W = 0.5 * float((mass * pot).sum())
                if W != 0:
                    vgrav = float(np.sqrt(abs(W) / mtot))
                    virial0 = float(ke2 / abs(W))
    if vgrav and vgrav > 0:
        norm, norm_kind = vgrav, "v_grav"
    else:
        norm, norm_kind = (vrms0 if vrms0 and vrms0 > 0 else 1.0), "v_rms0"
        warnings.warn(
            f"momentum_drift: no usable Potential in {snap_paths[0]}, normalizing by v_rms(0)"
            f"={norm:.4g} instead of the gravitational velocity. For a cold-start problem this "
            f"inflates the drift; enable OUTPUT_POTENTIAL for a comparable number.",
            RuntimeWarning, stacklevel=2)
    times = np.array(times)
    vcoms = np.array(vcoms)
    drift = np.linalg.norm(vcoms - vcoms[0], axis=1) / norm
    return {
        "times": times,
        "drift": drift,
        "vcom": vcoms,
        "norm": np.array(norm),
        "norm_kind": np.array(norm_kind),
        "vrms0": np.array(vrms0 if vrms0 is not None else np.nan),
        "vgrav": np.array(vgrav if vgrav is not None else np.nan),
        "virial0": np.array(virial0 if virial0 is not None else np.nan),
        "ptot": np.array(ptots),
        "mtot": np.array(mtots),
    }


def mass_bookkeeping_error(traj):
    """Max fractional deviation of the included total mass from its initial value. Nonzero
    means the chosen particle types miss something that carries momentum."""
    m = traj["mtot"]
    return float(np.abs(m - m[0]).max() / m[0]) if len(m) and m[0] != 0 else 0.0


def snapshot_list(output_dir):
    return sorted(glob.glob(path.join(output_dir, "snapshot_*.hdf5")))


def momentum_summary_path(test_dir, variant_id):
    return f"{test_dir}/momentum_{variant_id}.npz"


def save_momentum_summary(test_dir, variant_id, traj):
    np.savez(momentum_summary_path(test_dir, variant_id), **traj)


def load_momentum_summaries(test_dir):
    """Load every momentum_<variant>.npz present, newest content wins. Returns dict."""
    out = {}
    for f in sorted(glob.glob(f"{test_dir}/momentum_*.npz")):
        vid = path.basename(f)[len("momentum_"):-len(".npz")]
        try:
            with np.load(f) as d:
                out[vid] = {k: d[k] for k in d.files}
        except Exception:
            continue
    return out


def measure_and_record(test_dir, variant_id, output_dir, parttype="PartType1"):
    """Measure the momentum drift for one variant and persist it. Returns the traj dict."""
    snaps = snapshot_list(output_dir)
    if len(snaps) < 2:
        return None
    traj = momentum_trajectory(snaps, parttype=parttype)
    save_momentum_summary(test_dir, variant_id, traj)
    return traj


def plot_momentum_drift(test_dir, test_name):
    """Overlay |dv_com|/v_rms vs time for every variant with a recorded summary.

    Rebuilt from whatever momentum_*.npz files are present, so running a single variant
    still refreshes the comparison (same idiom as the plummer/hernquist profile overlays).
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    summaries = load_momentum_summaries(test_dir)
    if not summaries:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for vid, d in sorted(summaries.items()):
        t, drift = d["times"], d["drift"]
        randomized = "randomize" in vid
        ax.plot(t, drift,
                marker="o" if randomized else "s",
                markersize=4, linewidth=2.0 if randomized else 1.2,
                linestyle="-" if randomized else "--",
                alpha=0.9, label=vid)
    ax.set_yscale("log")
    ax.set_xlabel("Time (code units)")
    ax.set_ylabel(r"$|\Delta v_{\rm com}| \, / \, v_{\rm grav}$")
    ax.set_title(f"{test_name}: spurious COM drift from correlated tree-force errors")
    ax.legend(fontsize="x-small", loc="best")
    ax.grid(alpha=0.3, which="both")
    out = f"{test_dir}/momentum_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def report_momentum_drift(test_dir, test_name, variant_id, traj):
    """Print this variant's final drift plus a comparison against every other recorded
    variant. Comparison is reported, not asserted: the baseline/randomized ratio depends
    on N, run length and timestepping, so the numbers are surfaced for inspection rather
    than frozen into a threshold."""
    final = float(traj["drift"][-1])
    kind = str(traj["norm_kind"])
    vir = float(traj["virial0"])
    regime = "virialized" if vir > 0.5 else "cold"
    print(f"\n[momentum-drift] {test_name}/{variant_id}: final drift = {final:.3e}  "
          f"(norm={kind}={float(traj['norm']):.4g}, 2K/|W|={vir:.3g} -> {regime})")

    summaries = load_momentum_summaries(test_dir)
    if len(summaries) > 1:
        print(f"[momentum-drift] {test_name} recorded variants (final drift):")
        for vid, d in sorted(summaries.items(), key=lambda kv: float(kv[1]["drift"][-1])):
            print(f"    {vid:<28s} {float(d['drift'][-1]):.3e}")
        # explicit on/off comparisons for matched pairs
        for rid in [v for v in summaries if "randomize" in v]:
            bid = _baseline_counterpart(rid, summaries)
            if bid is not None:
                rd = float(summaries[rid]["drift"][-1])
                bd = float(summaries[bid]["drift"][-1])
                if rd > 0:
                    print(f"[momentum-drift] {test_name}: {bid} / {rid} = {bd / rd:.2f}x "
                          f"({'improved' if bd > rd else 'NOT improved'} by RANDOMIZE_GRAVTREE)")
    return final


def _baseline_counterpart(randomize_id, summaries):
    """Map a randomize variant id to the matching flag-off variant, so we compare like
    with like (non-periodic vs non-periodic, PM vs PM)."""
    if randomize_id == "randomize":
        return "baseline" if "baseline" in summaries else None
    if randomize_id == "randomize_pmgrid":
        return "pmgrid" if "pmgrid" in summaries else None
    if randomize_id == "randomize_ewald":
        return "ewald" if "ewald" in summaries else None
    return None
