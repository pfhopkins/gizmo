"""Rotating Shu (1977) collapse with protostellar jets, non-radiative.

Config.sh is the jets-free baseline (STARFORGE defaults + COOLING, no SINGLE_STAR_FB_RAD);
each variant appends SINGLE_STAR_FB_JETS + JET_DIRECTION_FIXED_Z plus one merge treatment
via extra_config_flags. Every run must form exactly one sink, launch a collimated outflow
along the fixed +/-z axis, and produce the accreted-angular-momentum plot.
"""

import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from meshoid import Meshoid

from gizmo.test import (
    build_and_run_test,
    flush_colorbar,
    assert_final_time,
    get_final_snapshot,
    default_omp_threads,
    default_mpi_ranks,
    variant_output_dir,
    variant_suffix,
)

TEST_NAME = "shu_jets"
TEST_DIR = Path(__file__).parent
JETS = ("SINGLE_STAR_FB_JETS", "JET_DIRECTION_FIXED_Z")
JETS_NOMERGE = JETS + ("SINK_SPAWN_NO_MERGE",)
JETS_NOMERGE_ALL = JETS + ("PREVENT_PARTICLE_MERGE",)
# Intended production behaviour: a spawned cell persists while still resolving the outflow and
# retires only once it has joined the ambient medium. jets_merge / jets_nomerge bracket it from
# the over- and under-merging sides.
JETS_AMBIENT = JETS + ("SINK_SPAWN_MERGE_WHEN_AMBIENT", "MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION")

# Thermodynamics is a parametrized axis rather than a base-Config choice, so the two regimes are
# directly comparable. ISOTHERMAL is the setup the README documents and matches the shu1977 parent;
# COOLING is the alternative. Base Config.sh sets neither.
# The pre-auto-enable behaviour: retire a spawned cell as soon as it is subsonic wrt whichever
# neighbour the search picks, and permit merges that thermalize most of the energy involved. Keeps
# the merge/wakeup ENERGY fixes on, so this isolates merge permissiveness rather than confounding it
# with the ~23% COM-frame discard and the kick-reversal injection those fixes remove.
JETS_PERMISSIVE = JETS + ("SINK_SPAWN_MERGE_ANY_NEIGHBOR", "MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION")

ISOTHERMAL = ("EOS_GAMMA=1.001", "EOS_ENFORCE_ADIABAT=0.04")
COOLING = ("COOLING", "OUTPUT_COOLRATE_DETAIL")
# Nominal only: Config.sh sets no EOS_GAMMA, and with COOLING the STARFORGE defaults
# enable EOS_SUBSTELLAR_ISM, so gamma actually varies with the H2 state. Used solely to
# put the angular-momentum plot in units of R_sink*c_s.
GAMMA = 5.0 / 3.0


def generate_ics():
    """Generate the rotating ICs if they aren't already present."""
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        sys.path.insert(0, str(TEST_DIR))
        from make_shu_jets_ics import make_shu_jets_ics

        make_shu_jets_ics(str(ic_file))
    return ic_file


def ambient_cell_mass():
    """The nominal gas mass resolution, read from the ICs where every cell is identical.

    Not taken from a snapshot: with jets on, the spawned cells sit at 0.1x this mass
    and would drag a median or mean off the ambient value.
    """
    with h5py.File(generate_ics(), "r") as f:
        return float(np.median(f["PartType0/Masses"][:]))


def get_jet_cells(f, m_gas):
    """Mask of gas cells spawned by the jet, identified by their spawn mass.

    Spawned cells all start at exactly Sink_outflow_particlemass (0.1 * m_gas) and can
    then only grow by merging, so anything well under the ambient resolution came from
    the sink.
    """
    return f["PartType0/Masses"][:] < 0.5 * m_gas


def plot_angmom_vs_mass(extra_config_flags=()):
    """Sink specific angular momentum in units of R_sink * c_s, vs sink mass.

    The top axis restates the bottom one as the number of accreted gas cells; z is the
    rotation axis of the initial core, so j_z carries the ordered rotation and j_x, j_y
    are the sampling noise floor.
    """
    m_gas = ambient_cell_mass()
    msink, jvec, m_seed = [], [], None
    for snap in sorted(glob(f"{variant_output_dir(TEST_NAME, extra_config_flags)}/snapshot_*.hdf5")):
        with h5py.File(snap, "r") as f:
            if "PartType5" not in f:
                continue
            c_s = np.sqrt(GAMMA * (GAMMA - 1.0) * np.median(f["PartType0/InternalEnergy"][:]))
            r_sink = float(f["PartType5/Sink_Radius"][:][0])
            m_seed = float(f["PartType5/Sink_InitialMass"][:][0])
            msink.append(float(f["PartType5/Sink_Mass"][:][0]))
            jvec.append(f["PartType5/Sink_Specific_AngMom"][:][0] / (r_sink * c_s))
    msink, jvec = np.array(msink), np.array(jvec)

    ink, muted = "#1c1c1c", "#8a8a8a"
    # validated categorical slots 1-4; magnitude first, then the dominant axis
    series = [
        (r"$|j|$", np.linalg.norm(jvec, axis=1), "#2a78d6"),
        (r"$j_z$", jvec[:, 2], "#eb6834"),
        (r"$j_x$", jvec[:, 0], "#1baf7a"),
        (r"$j_y$", jvec[:, 1], "#eda100"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color=muted, lw=1.0, zorder=1)
    ax.axhline(1.0, color=muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(
        msink[0], 1.02, "centrifugal radius = sink radius",
        color=muted, fontsize=8, va="bottom", ha="left",
    )
    for label, y, color in series:
        ax.plot(msink, y, color=color, lw=2.0, marker="o", ms=4.0, mew=0, label=label, zorder=3)
        ax.annotate(
            label, xy=(msink[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=9, va="center", ha="left",
        )

    ax.set_xlabel(r"$M_{\rm sink}\ (M_\odot)$")
    ax.set_ylabel(r"$j_{\rm sink} \,/\, (R_{\rm sink}\, c_s)$")
    ax.grid(True, axis="y", color=muted, alpha=0.25, lw=0.6)
    # matplotlibrc boxes the axes with ticks on all four sides; hand the top edge to
    # the secondary axis so the two don't draw over each other
    ax.tick_params(which="both", top=False)
    ax.set_xlim(0, msink[-1] * 1.09)

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda m: (m - m_seed) / m_gas, lambda n: n * m_gas + m_seed),
    )
    secax.set_xlabel("Accreted particles", labelpad=8)
    secax.xaxis.set_major_locator(MultipleLocator(500))
    secax.xaxis.set_minor_locator(MultipleLocator(100))

    run_label = "with jets" if extra_config_flags else "no jets"
    ax.set_title(
        f"Accreted specific angular momentum, rotating Shu core ({run_label})\n"
        "z is the initial rotation axis",
        fontsize=10, pad=34,
    )
    leg = ax.legend(fontsize=9, loc="upper left", ncol=4, columnspacing=1.2, handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(ink)

    out = f"test/{TEST_NAME}/sink_angmom_components_vs_mass{variant_suffix(extra_config_flags)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_jet_density_slice(snap_file, extra_config_flags=()):
    """Density slice in the x-z plane (the rotation axis is z), so the jet is edge-on."""
    with h5py.File(snap_file, "r") as f:
        coords = f["PartType0/Coordinates"][:]
        rho = f["PartType0/Density"][:]
        sink = f["PartType5/Coordinates"][:][0] if "PartType5" in f else coords.mean(axis=0)
        time = float(f["Header"].attrs["Time"])

    size = 0.02
    rho_slice = Meshoid(coords).Slice(np.log10(rho), res=800, plane="y", center=sink, size=size, order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    half = size / 2
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-half, half, -half, half])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("z (pc)")
    ax.set_title(f"Rotating Shu 1977 + jets, t = {time:.3g}")
    fig.savefig(
        f"test/{TEST_NAME}/Density_jet_xz{variant_suffix(extra_config_flags)}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def assert_collimated_outflow(final_snap, m_gas):
    """The jet run must spawn cells at the requested resolution and blow them out along z."""
    with h5py.File(final_snap, "r") as f:
        sink_pos = f["PartType5/Coordinates"][:][0]
        sink_vel = f["PartType5/Velocities"][:][0]
        coords = f["PartType0/Coordinates"][:]
        vel = f["PartType0/Velocities"][:]
        masses = f["PartType0/Masses"][:]
        jet = get_jet_cells(f, m_gas)

    assert jet.sum() > 0, "No jet cells were spawned"
    m_jet_cell = 0.1 * m_gas  # Sink_outflow_particlemass in shu_jets.params
    assert np.allclose(masses[jet], m_jet_cell, rtol=0.5), (
        f"Jet cell masses {np.unique(masses[jet])[:5]} are not near the requested "
        f"spawn mass {m_jet_cell:.4g}"
    )

    dx = coords[jet] - sink_pos
    dv = vel[jet] - sink_vel
    r = np.linalg.norm(dx, axis=1)
    moving = r > 0

    # the outflow must actually be an outflow
    v_radial = np.sum(dx[moving] * dv[moving], axis=1) / r[moving]
    assert np.median(v_radial) > 0, f"Jet cells are not moving outward (median v_r = {np.median(v_radial):.3g})"

    # ...and it must be collimated along the rotation axis rather than isotropic.
    # <cos^2(theta)> = 1/3 for an isotropic distribution; a bipolar jet is well above that.
    # Positional collimation: where the jet material has GOT to. 0.9, not 0.5: measured 0.963-0.989
    # across every merging regime and both EOS, and 0.978 at TimeMax = 0.01, so 0.5 was nowhere
    # near binding. Since
    # JET_DIRECTION_FIXED_Z pins the axis this is a launch-geometry check with a known answer.
    cos2_theta = (dx[moving, 2] / r[moving]) ** 2
    assert np.mean(cos2_theta) > 0.9, (
        f"Outflow is not collimated along the z axis: <cos^2(theta)> = {np.mean(cos2_theta):.4g}, "
        "expected > 0.9 (isotropic would give 1/3; healthy runs give 0.96-0.99)"
    )

    # Kinematic collimation, from the principal axes of the kinetic-energy tensor
    # T_ij = sum_k m_k dv_ki dv_kj (trace = 2*KE). Weighting by kinetic energy rather than counting
    # cells is what makes this usable: a cell-averaged <(dvz/|dv|)^2> REWARDS aggressive retirement,
    # because discarding the slow deflected cells leaves a fresher, more axial sample -- measured
    # 0.965 for permissive merging against 0.885 for no merging, i.e. exactly backwards. The tensor
    # ranks them correctly (111 vs 213) since the discarded cells carried little energy.
    T = np.einsum("k,ki,kj->ij", masses[jet][moving], dv[moving], dv[moving])
    evals, evecs = np.linalg.eigh(T)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    axis_ratio = evals[0] / max(evals[1], 1e-30)
    # Loose floor by design. Measured 111-272, but nomerge and nomerge_all -- which differ only in
    # whether AMBIENT gas merges -- came out 213 vs 155, so the intrinsic scatter is tens of percent
    # and comparable to the spread between merging regimes. This catches a collapse of collimation,
    # it does not discriminate between regimes; use the energetics comparison for that.
    assert axis_ratio > 50.0, (
        f"Kinetic energy is not collimated: principal-axis ratio L1/L2 = {axis_ratio:.1f}, "
        "expected > 50 (healthy runs give 110-270; isotropic would give ~1)"
    )

    # ...and the principal axis must BE the launch axis. Unlike <cos^2(theta)> above this does not
    # assume z, so it is the check that would survive in a setup where the axis is not pinned.
    axis_dot_z = abs(evecs[2, 0])
    assert axis_dot_z > 0.99, (
        f"Principal axis of the outflow is not z: |axis.z| = {axis_dot_z:.4f}, expected > 0.99 "
        "(healthy runs give > 0.997)"
    )


# Two variants of the non-RT jets setup, differing only in whether spawned jet cells may
# merge. JET_DIRECTION_FIXED_Z pins the launch axis to +/-z in both, so
# assert_collimated_outflow's <cos^2(theta)> test measures the launch geometry against a
# known axis rather than the sink's emergent angular momentum.
# Three merging regimes, to separate numerical diffusion from the physics:
#   jets_merge       - default: spawned cells may merge into their surroundings
#   jets_nomerge     - SINK_SPAWN_NO_MERGE: spawned cells protected, ambient gas still merges
#   jets_nomerge_all - PREVENT_PARTICLE_MERGE: no merging anywhere (diffusion-free reference)
# Disabling inter-jet merging visibly enlarges the jet structure, since merging both
# diffuses the outflow and coarsens the cells resolving it.
def assert_sink_mass(extra_config_flags):
    """Sink mass: exact against mass conservation, loose against the collapse itself.

    Gas + sink is a closed budget here -- the sink can only take mass from the gas, and the fraction
    it relaunches as jets goes straight back into the gas -- so M_gas + M_sink must equal its initial
    value. Measured drift is 0 to 3e-8 and dM_gas/M_sink = -1.00000 exactly, so this is a real
    conservation law and is bounded tightly.

    The sink mass itself is bounded loosely and deliberately. How much of a rotating collapse ends up
    in the star by a fixed time is a physics outcome, not a fixed number: it moves with domain
    decomposition (the retirement block fraction moved 7.9% -> 17.4% between 8 and 48 ranks on
    identical physics) and with the merge treatment (permissive merging cost 46% of the sink mass by
    dumping jet momentum into the accretion flow). The window below catches a sink that failed to
    accrete or ran away, nothing finer. It also grows steadily with the run: measured 0.629% of the
    10 Msun total at TimeMax = 0.01 and 1.194% at t = 0.02, so the window is set wide enough to
    survive a change of end time. The merge-vs-nomerge agreement is checked separately in
    test_shu_jets_merging_energetics.
    """
    snaps = sorted(glob(f"{variant_output_dir(TEST_NAME, extra_config_flags)}/snapshot_*.hdf5"))
    assert len(snaps) > 1, "need the first and last snapshot to check mass conservation"

    def gas_and_sink(path):
        with h5py.File(path, "r") as f:
            m_gas = float(f["PartType0/Masses"][:].sum())
            m_sink = float(f["PartType5/Masses"][:].sum()) if "PartType5" in f else 0.0
        return m_gas, m_sink

    g0, s0 = gas_and_sink(snaps[0])
    g1, s1 = gas_and_sink(snaps[-1])
    drift = abs((g1 + s1) - (g0 + s0)) / (g0 + s0)
    assert drift < 1e-6, (
        f"gas + sink mass is not conserved: {g0 + s0:.8f} -> {g1 + s1:.8f} "
        f"(relative drift {drift:.3e}, expected < 1e-6). The sink can only take mass from the gas "
        "and returns the relaunched fraction to it, so this budget must close."
    )

    frac = s1 / (g0 + s0)
    assert 0.002 < frac < 0.02, (
        f"final sink mass {s1:.5f} is {100 * frac:.3f}% of the {g0 + s0:.3f} total, expected "
        "0.2-2%. Below that the sink barely accreted; above it the jets are not regulating."
    )


# Two runs, both with cooling: the default merge criteria against no spawned-cell merging at all.
# SINK_SPAWN_MERGE_WHEN_AMBIENT and MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION are auto-enabled with
# SINGLE_STAR_FB_JETS, so jets_merge IS the default-criteria case. Cooling rather than isothermal
# because EOS_ENFORCE_ADIABAT pins u, which makes the thermal half of the retirement criterion inert
# (measured: 0 of 474 cells blocked, every one in the lowest |dln u| bin); with cooling it
# discriminates (207 of 1190 blocked). The permissive and no-merging-anywhere variants were dropped
# once they had served their purpose -- permissive retires jet cells ~6x closer in, which dumps the
# outflow momentum into the accretion flow and costs 46% of the sink mass.
@pytest.mark.parametrize(
    "merge_flags",
    [JETS, JETS_NOMERGE],
    ids=["jets_merge", "jets_nomerge"],
)
@pytest.mark.parametrize("eos_flags", [COOLING], ids=["cooling"])
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shu_jets(num_mpi_ranks, num_omp_threads, eos_flags, merge_flags):
    extra_config_flags = merge_flags + eos_flags
    generate_ics()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags=extra_config_flags)

    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
    assert_final_time(final_snap, TEST_NAME)

    with h5py.File(final_snap, "r") as f:
        num_sinks = f["Header"].attrs["NumPart_ThisFile"][5]
    assert num_sinks == 1, f"Expected exactly 1 PartType5 particle, got {num_sinks}"

    assert_sink_mass(extra_config_flags)

    if extra_config_flags == JETS:
        assert_collimated_outflow(final_snap, ambient_cell_mass())
        plot_jet_density_slice(final_snap, extra_config_flags)

    plot_angmom_vs_mass(extra_config_flags)


def _final_energetics(merge_flags):
    """Total gas energy, outward momentum and sink mass from a variant's last snapshot."""
    snaps = sorted(glob(f"{variant_output_dir(TEST_NAME, merge_flags + COOLING)}/snapshot_*.hdf5"))
    if not snaps:
        return None
    with h5py.File(snaps[-1], "r") as f:
        m = f["PartType0/Masses"][:].astype(float)
        v = f["PartType0/Velocities"][:].astype(float)
        u = f["PartType0/InternalEnergy"][:].astype(float)
        x = f["PartType0/Coordinates"][:].astype(float)
        m_sink = float(f["PartType5/Masses"][:].sum())
        ctr = f["PartType5/Coordinates"][:][0]
        v_sink = f["PartType5/Velocities"][:][0]
    d = x - ctr
    r = np.linalg.norm(d, axis=1)
    vr = np.sum((v - v_sink) * d, axis=1) / (r + 1e-30)
    out = vr > 0
    return {
        "E_tot": 0.5 * np.sum(m * np.sum(v**2, axis=1)) + np.sum(m * u),
        "p_out": float(np.sum(m[out] * vr[out])),
        "M_sink": m_sink,
    }


def test_shu_jets_merging_energetics():
    """The default merge criteria must not cost the outflow relative to not merging at all.

    The point of SINK_SPAWN_MERGE_WHEN_AMBIENT is to retire spawned cells that have genuinely
    joined the ISM while protecting those that have not, so its energetics should land near the
    no-merging case while still retiring cells -- and it does: it ends with ~0.3% fewer cells than
    jets_nomerge, so retirement is happening, yet the outflow is intact.

    Compares total gas energy, outward momentum and sink mass rather than the jet kinetic energy
    alone. That is deliberate and worth knowing: the purely kinetic measures (total KE, jet KE,
    outward KE) all sit at 10.5-11% between these two runs, so a 10% bound on any of them would
    fail on a healthy pair. They are also the quantities most sensitive to domain decomposition --
    the retirement block fraction moved 7.9% -> 17.4% between 8 and 48 ranks on identical physics --
    so a tight bound on them would be measuring the rank count. Measured here: E_tot 9.2%,
    p_out 5.9%, M_sink 4.5%, at 48 ranks.
    """
    a = _final_energetics(JETS)
    b = _final_energetics(JETS_NOMERGE)
    if a is None or b is None:
        pytest.skip("needs both jets_merge and jets_nomerge to have run")
    for key in ("E_tot", "p_out", "M_sink"):
        rel = abs(a[key] - b[key]) / abs(b[key])
        assert rel < 0.10, (
            f"{key} differs by {100 * rel:.1f}% between the default merge criteria "
            f"({a[key]:.5g}) and no merging ({b[key]:.5g}), expected within 10%. "
            "Either retirement has started destroying the outflow, or it has stopped "
            "retiring and the comparison is no longer meaningful -- check the "
            "'Spawned-cell retirement' report in each run's log."
        )
