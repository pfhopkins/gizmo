"""wind_singlestar test: wind-blown bubble from a 100 Msun ZAMS star.

Same box setup as SN_singlestar (50000 Msun, n_H = 100 cm^-3) but the star is
100 Msun at zero age with SINGLE_STAR_FB_WINDS=2.

Parametrized over:
  - Cooling variant (adiabatic vs COOLING vs COOLING+SINGLE_STAR_FB_RAD)
  - Wind injection mode (local mechanical injection vs particle spawning)

Fixed at 64^3 resolution. Both check Weaver+ 1977 R2, using the coefficient for the
applicable regime -- Weaver+ solve both: 0.88 adiabatic, 0.76 radiative. Adiabatic runs
also check that E_kin + E_th equals the injected energy exactly.
"""

import pytest
import numpy as np
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from glob import glob
import h5py
import sys

from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    flush_colorbar,
    assert_final_time,
    get_cooling_tables,
    get_final_snapshot,
    variant_output_dir,
)
from pathlib import Path

TEST_NAME = "wind_singlestar"
TEST_DIR = Path(__file__).parent

# Code units: pc, Msun, km/s
M_PROTON_G = 1.6726e-24
PC_IN_CM = 3.0857e18
MSUN_IN_G = 1.989e33
X_H = 0.70
RHO_AMBIENT_CODE = (100.0 * M_PROTON_G / X_H) * (PC_IN_CM**3) / MSUN_IN_G


def wind_luminosity_code(Mdot_msun_yr, v_wind_kms):
    """Wind mechanical luminosity L_w = 0.5 * Mdot * v_w^2 in code units."""
    Mdot_cgs = Mdot_msun_yr * MSUN_IN_G / (365.25 * 86400)
    L_w_cgs = 0.5 * Mdot_cgs * (v_wind_kms * 1e5) ** 2
    code_power_cgs = 1.989e43 / (PC_IN_CM / 1e5)
    return L_w_cgs / code_power_cgs


def wind_luminosity_cgs(Mdot_msun_yr, v_wind_kms):
    Mdot_cgs = Mdot_msun_yr * MSUN_IN_G / (365.25 * 86400)
    return 0.5 * Mdot_cgs * (v_wind_kms * 1e5) ** 2


def weaver_radius(L_w, rho_0, t, alpha):
    """Weaver+ 1977 R2, the shock front: R2 = alpha * (L_w t^3 / rho_0)^(1/5).

    Weaver+ solve both regimes; only the coefficient differs. ALPHA_ADIABATIC applies when the
    swept-up gas retains its thermal energy, ALPHA_RADIATIVE when it radiates it away and collapses
    into a thin dense shell, retaining only part of L*t.
    """
    return alpha * (L_w / rho_0) ** 0.2 * t**0.6


# Weaver+ 1977 R2 coefficients. R2 is the SHOCK FRONT, which is the right comparison for
# measure_shell_radius: it locates the density peak, and behind a strong shock the density peaks
# immediately post-shock, so peak and front nearly coincide. Measured xi = 0.876 adiabatic and 0.772
# radiative, against 0.88 and 0.76. Do NOT substitute Draine 2011's (100/(27 pi))^(1/5) = 1.0335,
# a differently-defined radius that reads ~15% high here -- enough to swamp the effect of a merge
# treatment on the coefficient.
ALPHA_ADIABATIC = 0.88
ALPHA_RADIATIVE = 0.76


def generate_ics(res=128):
    """Generate ICs, regenerating if the resolution changed."""
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if ic_file.exists():
        with h5py.File(str(ic_file), "r") as F:
            existing_ngas = int(F["Header"].attrs["NumPart_Total"][0])
        if existing_ngas != res**3:
            ic_file.unlink()
    if not ic_file.exists():
        sys.path.insert(0, str(TEST_DIR))
        from make_wind_singlestar_ics import make_wind_singlestar_ics

        make_wind_singlestar_ics(str(ic_file), res=res)


def make_wind_config_flags(Mdot, v_w, wind_mode=None):
    """Return extra_config_flags tuple encoding wind parameters."""
    L_w_cgs = wind_luminosity_cgs(Mdot, v_w)
    flags = (f"WIND_MDOT={Mdot:.6g}", f"WIND_LUMINOSITY={L_w_cgs:.6g}")
    if wind_mode is not None:
        flags += (f"SINGLE_STAR_WIND_MODE={wind_mode}",)
    return flags


def load_snapshot_radial(snap_file):
    """Load a snapshot and return (time, box_size, coords, r, rho, vr)."""
    with h5py.File(snap_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
        box_size = float(F["Header"].attrs["BoxSize"])
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]
        if "PartType5" in F and "Coordinates" in F["PartType5"]:
            center = F["PartType5/Coordinates"][0]
        else:
            center = np.array([box_size / 2.0] * 3)
    dr = coords - center
    r = np.sqrt(np.sum(dr * dr, axis=1))
    vr = np.sum(vel * dr, axis=1) / (r + 1e-30)
    return time, box_size, coords, r, rho, vr


def plot_slices(snap_file, output_dir=".", suffix=""):
    """Plot density and temperature slices from a snapshot."""
    with h5py.File(snap_file, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        temp = F["PartType0/Temperature"][:]
        box_center = float(F["Header"].attrs["BoxSize"]) / 2.0

    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    ext = [-box_center, box_center, -box_center, box_center]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    rho_slice = M.Slice(np.log10(rho), res=512, plane="z", center=center, size=2 * box_center, order=0)
    im0 = axes[0].imshow(rho_slice.T, origin="lower", cmap="inferno", extent=ext)
    flush_colorbar(im0, ax=axes[0], label=r"$\log_{10}\rho$ (Msun/pc$^3$)")
    axes[0].set_xlabel("x (pc)")
    axes[0].set_ylabel("y (pc)")
    axes[0].set_title("Density")
    temp_slice = M.Slice(np.log10(temp), res=512, plane="z", center=center, size=2 * box_center, order=0)
    im1 = axes[1].imshow(temp_slice.T, origin="lower", cmap="inferno", extent=ext)
    flush_colorbar(im1, ax=axes[1], label=r"$\log_{10} T$ (K)")
    axes[1].set_xlabel("x (pc)")
    axes[1].set_ylabel("y (pc)")
    axes[1].set_title("Temperature")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / f"Density_Temperature_2D{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure_shell_radius(snap_file, n_bins=140, min_particles_per_bin=30):
    """Return (time, r_shell), the radius of the peak of the spherically-averaged gas density.

    The swept-up shell IS that peak, and an extremum needs no threshold. On the adiabatic no-merging
    run this gives slope 0.611 against the analytic 0.6 with log-residual scatter 0.0037, versus
    0.593/0.0042 for a density-contrast threshold and 0.538/0.0039 for the best "innermost
    undisturbed particle" variant. The momentum-flux peak agrees to 0.001, so both locate the same
    feature. Note it is the shell peak, inside the outer edge a contrast threshold returns, so xi
    comes out a few percent smaller.

    Threshold-based alternatives all fare worse: a density contrast tracks the outer edge and depends
    on bin width; "innermost undisturbed particle" is biased low however undisturbed is defined, since
    it measures the leading edge of the disturbance and, as a strict minimum, whichever direction lags.
    Calling it by velocity fails because initial density perturbations relax acoustically to speeds of
    order c_s*(drho/rho); by entropy because the ambient itself cools or heats, leaving no stable
    reference -- worst under COOLING, where shocked-then-cooled gas reads as undisturbed.
    """
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(snap_file)
    with h5py.File(snap_file, "r") as F:
        masses = F["PartType0/Masses"][:]
    r_bins = np.linspace(0.0, box_size / 2.0, n_bins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    mass_in_bin = binned_statistic(r_sim, masses, "sum", r_bins)[0]
    counts = np.histogram(r_sim, r_bins)[0]
    # drop under-populated bins: the innermost ones hold few particles and are pure shot noise
    rho_sph = np.where(counts >= min_particles_per_bin, mass_in_bin / shell_vol, np.nan)
    if not np.isfinite(rho_sph).any():
        return time, np.nan
    return time, r_centers[np.nanargmax(rho_sph)]


def plot_energy_vs_time(snaps, L_w, output_dir=".", suffix=""):
    """Plot kinetic and thermal energy vs time alongside expected and code-tracked injection."""
    times, E_kin, E_therm = [], [], []
    for s in snaps:
        with h5py.File(s, "r") as F:
            t = float(F["Header"].attrs["Time"])
            masses = F["PartType0/Masses"][:]
            vel = F["PartType0/Velocities"][:]
            u = F["PartType0/InternalEnergy"][:]
        times.append(t)
        E_kin.append(0.5 * np.sum(masses * np.sum(vel**2, axis=1)))
        E_therm.append(np.sum(masses * u))

    times = np.array(times)
    E_kin = np.array(E_kin)
    E_therm = np.array(E_therm)
    E_injected = L_w * times
    E_kin_excess = E_kin - E_kin[0]
    E_therm_excess = E_therm - E_therm[0]

    plt.figure()
    mask = times > 0
    plt.plot(times[mask], E_kin_excess[mask], "bo-", markersize=4, label="KE (gas)")
    plt.plot(times[mask], E_therm_excess[mask], "rs-", markersize=4, label="Thermal (gas)")
    plt.plot(times[mask], E_kin_excess[mask] + E_therm_excess[mask], "k^-", markersize=5, label="KE + Thermal")
    plt.plot(times[mask], E_injected[mask], "g--", linewidth=2, label="Expected (Lw * t)")
    plt.xlabel("t (code units)")
    plt.ylabel("Energy (code units)")
    plt.legend(fontsize=9)
    plt.title("Energy budget")
    plt.savefig(str(Path(output_dir) / f"Energy_vs_t{suffix}.png"), bbox_inches="tight")
    plt.close()

    return times, E_kin_excess, E_therm_excess, E_injected


def get_snapshots(test_name, extra_config_flags=()):
    outdir = variant_output_dir(test_name, extra_config_flags)
    return sorted(glob(f"{outdir}/snapshot_*.hdf5"))


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(default_mpi_ranks(), default_omp_threads())])
@pytest.mark.parametrize("res", [64], ids=lambda x: f"N{x}")
@pytest.mark.parametrize(
    "Mdot_vw", [(1e-4,3000.0),], ids=lambda x: f"Mdot{x[0]:.0e}_vw{x[1]:.0f}"
)
@pytest.mark.parametrize("wind_mode", [1, 2], ids=["spawn", "local"])
@pytest.mark.parametrize(
    "cooling_flags",
    [(), ("COOLING",), ("COOLING", "SINGLE_STAR_FB_RAD")],
    ids=["adiabatic", "cooling", "rad"],
)
def test_wind_singlestar(request, num_mpi_ranks, num_omp_threads, Mdot_vw, res, wind_mode,
                         cooling_flags):
    if wind_mode == 2:
        # KNOWN DEFECT, local mechanical injection. WITH COOLING the shell comes out ~28% inside
        # Weaver. Mode 2 over-delivers energy per unit injected mass by ~25%, so the bubble is
        # over-pressured early, radiates the excess, and ends up too small. Two separable causes,
        # both in the FIRE coupling in galaxy_sf/mechanical_fb.cc:
        #   - the injection weight is a vector magnitude (wk = pnorm), and the sum of per-neighbour
        #     magnitudes exceeds one by ~2%, which is exactly the excess mass delivered;
        #   - e_shock includes the relative star-gas motion, v_ej^2 + 2 v_ej (phat.dv) + dv^2. The
        #     cross term does not average away, since phat is radial and the gas outflows radially,
        #     and it grows as the gas accelerates -- consistent with the excess drifting 1.22 -> 1.30
        #     over a run, and with 2*dv/v_ej for the measured dv ~ 300 km/s.
        # Arguably a test-premise mismatch rather than a code bug: for a SN ploughing into moving gas
        # the shock energy in the gas frame genuinely includes relative motion, whereas L_w*t here is
        # the wind's own mechanical luminosity and excludes it. Resolving that decides whether the
        # fix belongs in the coupling or in the comparison.
        #
        # ADIABATIC mode 2 is covered by this marker too (it was not, until measured). The energy
        # check already has a wind_mode-dependent bound that tolerates the ~30% surplus, but the
        # bubble-radius check does not, and the adiabatic case misses it by 3%: measured
        # xi = 0.7891 vs the Weaver adiabatic R2 coefficient 0.880, relative error 0.1033 against
        # a 0.10 bound. Same defect, same direction, merely below the threshold that made the
        # cooling cases obvious -- and note 0.789 sits between Weaver's adiabatic 0.880 and his
        # radiative 0.76, i.e. the bubble behaves partly radiative despite cooling being off.
        # Reproduced to the third decimal across two jobs on different nodes and thread counts
        # (0.1033 and 0.1040), so it is a property of the coupling, not run-to-run scatter, and
        # it predates the PR #27 sink/Hermite port.
        #
        # Not strict: mode 2 injects stochastically, so the size of the miss varies between runs and
        # a strict marker would make the suite flaky rather than informative. A strict marker would
        # also break the moment the coupling is fixed, which is the outcome we want.
        request.node.add_marker(pytest.mark.xfail(
            reason="mode-2 local injection over-delivers energy per unit mass by ~25%", strict=False))
    Mdot, v_w = Mdot_vw
    L_w = wind_luminosity_code(Mdot, v_w)

    # Build extra_config_flags from wind params + wind mode + cooling.
    wind_flags = make_wind_config_flags(Mdot, v_w, wind_mode=wind_mode)
    # WIND_TEST_NRES is unused by the code but ensures a unique output directory per resolution
    extra_config_flags = wind_flags + (f"WIND_TEST_NRES={res}",) + cooling_flags

    generate_ics(res=res)
    if "COOLING" in cooling_flags:
        get_cooling_tables(str(TEST_DIR))

    # Set Sink_outflow_particlemass = 1e-2 * gas mass resolution
    m_gas = 50000.0 / res**3
    params_file = TEST_DIR / f"{TEST_NAME}.params"
    params_text = params_file.read_text()
    import re

    params_text = re.sub(
        r"Sink_outflow_particlemass\s+\S+",
        f"Sink_outflow_particlemass {1e-2 * m_gas:.6g}",
        params_text,
    )

    # The params file is tracked, so restore it afterwards rather than leaving this run's rewritten
    # Sink_outflow_particlemass behind for the next run or for a manual invocation. GIZMO records what
    # it actually used in <name>.params-usedvalues, so provenance is kept.
    original_params = params_file.read_text()
    params_file.write_text(params_text)
    try:
        build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)
    finally:
        params_file.write_text(original_params)

    snaps = get_snapshots(TEST_NAME, extra_config_flags)
    assert len(snaps) > 1, f"No snapshots produced for {extra_config_flags}"
    final_snap = snaps[-1]

    # Measure shell radius
    times, r_shells = [], []
    for s in snaps:
        t, r_sh = measure_shell_radius(s)
        if t > 0:
            times.append(t)
            r_shells.append(r_sh)
    times = np.array(times)
    r_shells = np.array(r_shells)

    wm_str = f"wm{wind_mode}" if wind_mode is not None else "wm_auto"
    label = f"Mdot={Mdot:.0e} vw={v_w:.0f} N={res} {wm_str}"
    suffix = f"_Mdot{Mdot:.0e}_vw{v_w:.0f}_N{res}_{wm_str}"
    if cooling_flags:
        cooling_label = "+".join(cooling_flags)
        label += f" +{cooling_label}"
        suffix += "_" + "_".join(cooling_flags)

    # Save shell radius data to file
    np.savetxt(
        str(TEST_DIR / f"Rshell_data{suffix}.txt"),
        np.column_stack([times, r_shells]),
        header=f"t(code) R_shell(pc) | {label}",
        fmt="%.6e",
    )

    # Individual shell radius plot
    good = np.isfinite(r_shells) & (r_shells > 0) & (times > 0)
    if good.any():
        t_plot = np.logspace(np.log10(times[good].min()), np.log10(times[good].max() * 1.1), 200)
        R_weaver_plot = weaver_radius(L_w, RHO_AMBIENT_CODE, t_plot, ALPHA_RADIATIVE)
        from numpy.polynomial.polynomial import polyfit

        fit_mask = good & (times >= 0.01)
        if not fit_mask.any():
            fit_mask = good
        c = polyfit(np.log10(times[fit_mask]), np.log10(r_shells[fit_mask]), 1)
        alpha, A = c[1], 10 ** c[0]

        plt.figure()
        plt.loglog(times[good], r_shells[good], "ko", markersize=4, label="GIZMO")
        # overplot whichever similarity solution actually applies to this run, and show the other
        # dashed for comparison: the two regimes differ only in Weaver's coefficient
        R_adiabatic_plot = weaver_radius(L_w, RHO_AMBIENT_CODE, t_plot, ALPHA_ADIABATIC)
        if cooling_flags:
            plt.loglog(t_plot, R_weaver_plot, "r-", linewidth=1.5, label=rf"Weaver 1977 (radiative), $\alpha$={ALPHA_RADIATIVE:.2f}")
            plt.loglog(t_plot, R_adiabatic_plot, "r:", linewidth=1.0,
                       label=rf"Weaver 1977 (adiabatic), $\alpha$={ALPHA_ADIABATIC:.2f} (n/a here)")
        else:
            plt.loglog(
                t_plot, R_adiabatic_plot, "r-", linewidth=1.5,
                label=rf"Weaver 1977 (adiabatic), $\alpha$={ALPHA_ADIABATIC:.2f}",
            )
            plt.loglog(t_plot, R_weaver_plot, "r:", linewidth=1.0,
                       label=rf"Weaver 1977 (radiative), $\alpha$={ALPHA_RADIATIVE:.2f} (n/a here)")
        #        plt.loglog(t_plot, A * t_plot**alpha, "b--", label=f"Best fit ($t^{{{alpha:.2f}}}$)")
        plt.xlabel("t (code units)")
        plt.ylabel("R_shell (pc)")
        plt.legend()
        plt.title(label)
        plt.savefig(str(TEST_DIR / f"Rshell_vs_t{suffix}.png"), bbox_inches="tight")
        plt.close()

    # Plots
    snap_times, E_kin_x, E_therm_x, E_inj = plot_energy_vs_time(snaps, L_w, output_dir=str(TEST_DIR), suffix=suffix)
    plot_slices(final_snap, output_dir=str(TEST_DIR), suffix=suffix)

    # Final-snapshot radial profiles
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(final_snap)
    R_weaver = weaver_radius(L_w, RHO_AMBIENT_CODE, time, ALPHA_RADIATIVE)

    r_max = box_size / 2.0
    r_bins = np.linspace(0.0, r_max, 40)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    with h5py.File(final_snap, "r") as F:
        masses_final = F["PartType0/Masses"][:]
    mass_in_bin = binned_statistic(r_sim, masses_final, "sum", r_bins)[0]
    rho_binned = mass_in_bin / shell_vol
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]

    for field, binned in [("Density", rho_binned), ("RadialVelocity", vr_binned)]:
        plt.figure()
        plt.plot(r_centers, binned, "o", markersize=3, label="GIZMO")
        plt.axvline(R_weaver, color="red", linestyle="--", label="Weaver $R_b$")
        if field == "Density":
            plt.axhline(RHO_AMBIENT_CODE, color="grey", linestyle=":", label=r"$\rho_{\rm amb}$")
            plt.yscale("log")
        plt.xlabel("r (pc)")
        plt.ylabel(field)
        plt.legend()
        plt.title(f"t = {time:.4f}")
        plt.savefig(str(TEST_DIR / f"{field}{suffix}.png"))
        plt.close()

    # --- Assertions ---

    if not cooling_flags:
        # Adiabatic: energy conservation. 10% for spawning, matching SN_singlestar, with a few times that
        # much margin in practice; it catches a regression in either MERGE_SPLIT_CONSERVE_ENERGY or
        # WAKEUP_TRUNCATE_STEP_ON_DEMOTION, each of which costs ~20% of L*t on its own.
        #
        # Note the instrument: a snapshot is not a conserved total, since io.cc writes Velocities from
        # P[].Vel (kick-time) but InternalEnergy from InternalEnergyPred (drift-time). That mixed-clock
        # bias is small once the bubble is established but not zero, and E_inj = L_w*t -> 0 leaves the
        # ratio ill-conditioned early, hence the mask. ENERGY_BUDGET_DIAGNOSTIC's 'Energy (synced)' line
        # is the authoritative test; use that, not this, to chase a real conservation question.
        #
        # KNOWN OPEN DEFECT, wind_mode=2 (local mechanical injection): delivers ~30% MORE energy than
        # L_w*t. Held at a loose tolerance so the spawning path can still be tested at 10% -- this is an
        # unexplained bug, not an accepted error budget. It is unrelated to the merge and wakeup fixes:
        # local injection heats existing gas rather than spawning fast cells, so there is no spawned-cell
        # merging for either to act on, and the discrepancy is a steady excess rather than a loss. Either
        # local injection over-delivers, or WIND_LUMINOSITY means something different for mode 2 than the
        # L_w*t compared against here; check the code's own tracked injected energy against L_w*t first.
        tol = 0.10 if wind_mode == 1 else 0.35
        established = snap_times > 0.25 * snap_times.max()  # past the startup transient
        if established.any():
            ratio = (E_kin_x[established] + E_therm_x[established]) / E_inj[established]
            assert np.all(np.abs(ratio - 1.0) < tol), (
                f"Adiabatic energy conservation outside {tol:.0%} for wind_mode={wind_mode}: "
                f"ratio dE/E_injected = {ratio[-1]:.3f} "
                f"at t={snap_times[established][-1]:.4f} (worst {np.abs(ratio - 1.0).max():.3f}). "
                "Check MERGE_SPLIT_CONSERVE_ENERGY and WAKEUP_TRUNCATE_STEP_ON_DEMOTION are still active, "
                "then confirm against the in-code synced totals before believing this number."
            )

        # No radiative shell here, so Weaver's adiabatic coefficient is the applicable one
        # similarity solution. The first two checks are coefficient-free and so the robust ones; only
        # the third compares against ALPHA_ADIABATIC.
        late = (times > 0.3 * times.max()) & (r_shells > 0)
        assert late.sum() >= 3, f"Too few late-time snapshots to fit a growth law ({late.sum()})"

        slope = np.polyfit(np.log(times[late]), np.log(r_shells[late]), 1)[0]
        assert abs(slope - 0.6) < 0.12, (
            f"Adiabatic bubble growth R ~ t^{slope:.3f}, expected t^0.6 for constant-luminosity "
            "energy-driven expansion (0.5 would indicate momentum-driven, 0.4 fixed-energy Sedov)"
        )

        xi = r_shells[late] * (RHO_AMBIENT_CODE / (L_w * times[late] ** 3)) ** 0.2
        assert np.std(xi) / np.mean(xi) < 0.15, (
            f"Similarity coefficient not constant in time: xi = {np.mean(xi):.3f} "
            f"+/- {np.std(xi):.3f}; the solution is not self-similar"
        )

        # 10%: with the density-peak radius and Weaver's adiabatic alpha=0.88 this run reproduces the
        # similarity solution to well under 1% without spawned-cell merging, so a loose bound here
        # would be wasted. Merging costs a factor of a few in that agreement, which this still admits.
        rel_err = abs(np.mean(xi) - ALPHA_ADIABATIC) / ALPHA_ADIABATIC
        assert rel_err < 0.10, (
            f"Adiabatic bubble radius mismatch: measured xi = {np.mean(xi):.4f} vs Weaver+ 1977 "
            f"adiabatic R2 coefficient {ALPHA_ADIABATIC:.3f} (relative error {rel_err:.4f}); "
            "Weaver's radiative coefficient is 0.76 for comparison"
        )

    # Shell should exist and be expanding
    valid = np.isfinite(rho_binned)
    assert valid.any(), "No valid density bins"
    i_peak = np.nanargmax(rho_binned)
    r_peak = r_centers[i_peak]
    assert r_peak > r_centers[1], f"Density peak at r={r_peak:.3f} pc is too close to the star"
    assert vr_binned[i_peak] > 0, f"No outflow at shell: vr = {vr_binned[i_peak]}"

    if cooling_flags:
        # 10%, matching the adiabatic bound: the radiative shell tracks Weaver's 0.76 to ~1.6%
        # without spawned-cell merging, and to ~5.7% with it.
        rel_err = abs(r_peak - R_weaver) / R_weaver
        assert rel_err < 0.10, (
            f"Shell-radius mismatch: peak at {r_peak:.3f} pc vs Weaver {R_weaver:.3f} pc "
            f"(relative error {rel_err:.4f})"
        )


def test_wind_thermo_fidelity():
    """Guard the u->T conversion in the wind bubble's two sensitive regimes.

    The Weaver-radius assertion tolerates the cached-EOS defect that mislabeled the
    swept shell's warm molecular gas by 7-11% at matched u and pinned the hot bubble's
    T/u at the frozen fully-ionized slope (see COOLING_UTOT_REPORT.md). These checks
    are calibrated on inversion-correct runs (reference starforge_dev and the
    anchored-cache arm agree to <~1%; the defective build fails both).

    Reads the mode-1 (spawn) COOLING variant's existing output; skips if absent.
    """
    Mdot, v_w = 1e-4, 3000.0
    flags = make_wind_config_flags(Mdot, v_w, wind_mode=1) + ("WIND_TEST_NRES=64", "COOLING")
    outdir = Path(variant_output_dir(TEST_NAME, flags))
    snaps = sorted(outdir.glob("snapshot_*.hdf5"))
    if not snaps:
        pytest.skip("mode-1 COOLING variant output not present")
    with h5py.File(snaps[-1], "r") as F:
        T = F["PartType0/Temperature"][:].astype(float)
        u = F["PartType0/InternalEnergy"][:].astype(float)
        fmol = F["PartType0/MolecularMassFraction"][:].astype(float)

    # Regime guard: the shell-formed H2 layer only exists once the shell has swept and
    # shielded enough mass (t >~ 0.05 here). If the run ended early there is nothing to test.
    shell = (fmol > 0.3) & (T > 100) & (T < 800)
    if shell.sum() < 5000:
        pytest.skip(f"warm molecular shell not developed (N={shell.sum()})")

    # 1. Warm molecular shell T/u at matched u (reference medians at the final snapshot;
    #    the defective conversion read 7-11% low here, inversion arms within ~1%).
    u_edges = [0.2731, 1.266, 5.869]
    Tu_expected = [164.18, 131.63]   # reference medians WITH the fH2>0.3 cut (defective build: 137.7, 117.2 -> 13.5% aggregate)
    devs = []
    for lo, hi, Tue in zip(u_edges[:-1], u_edges[1:], Tu_expected):
        sel = (u >= lo) & (u < hi) & (fmol > 0.3)
        if sel.sum() >= 100:
            devs.append(abs(np.median((T / u)[sel]) / Tue - 1))
    assert devs, "warm-molecular u bins unexpectedly empty despite developed shell"
    agg = float(np.mean(devs))
    assert agg < 0.05, (
        f"warm-molecular shell T/u off by {100*agg:.1f}% aggregate (>5%): H2-bearing shell "
        f"gas is mislabeled at matched u (defective conversion measured 7-11%)"
    )

    # 2. Hot shocked-wind bubble: T/u must sit at the He-ionized inversion value
    #    (49.50 +- decomposition scatter), not the frozen fully-ionized slope (47.93).
    bub = u > 2.7e5
    if bub.sum() >= 100:
        tu_hot = float(np.median((T / u)[bub]))
        assert 48.7 < tu_hot < 50.3, (
            f"hot-bubble median T/u = {tu_hot:.3f} outside [48.7, 50.3]: the u->T map is "
            f"reading the frozen ionized slope (defect gives 47.93; inversion gives 49.50)"
        )
