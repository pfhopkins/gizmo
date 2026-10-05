"""Radiation energy conservation in a closed box: a static (FREEZE_HYDRO), uniform box of dusty gas (solar metallicity,
32^3 glass, 8 pc) holding a uniform radiation field in every band of the build, with no ISRF exchange and, except in the
"star" variants, no sources. Nothing enters or leaves the bands except through the channels the code models.

The runs start from a snapshot (restart flag 2) because a flag-0 start resets the photon energies. The initial field is
the same at every density (20 eV/cm^3 per band, 30 in the IR), so the radiation-equilibrium T_dust is the same (11.75 K)
and only the optical depths change with n_H. Steps are 3e-3 (RT Courant-limited at c_tilde = 1e-4 c) at every density.
Medians per cell and per step, from the runs' own rt_kappa output:

  n_H     IR tau per cell (no COOLING / COOLING)   donor tau per step: photoelectric, NUV, optical-NIR
  1e3     8e-4 / 4e-4                              0.22, 0.15, 0.055
  1e4     0.016                                    2.2, 1.5, 0.55
  1e5     0.78                                     22, 15, 5.5
  1e7     6.8e3 / 2.5                              2200, 1500, 550

Without COOLING the IR gas opacity takes x_e = 1, so free-free (proportional to rho) dominates it above n_H ~ 1e4; with
COOLING x_e ~ 1e-9 and the IR opacity is dust (0.14-0.25 cm^2/g).

Bare-RT variants (photoelectric, NUV, optical-NIR, IR; periodic box), from snapshot band totals and from the
RT_DIAG_IR_DOUBLE_COUNT ledger:
  (a) total band energy constant to 1e-6. The kick conserves it to round-off, so the floor is the float32 snapshot
      output: each cell's value is rounded to 6e-8, which bounds the error of a band total at 6e-8 of it;
  (b) IR gain = donor loss, to 1e-6 of the total band energy (same bound);
  (c) a donor band with tau per step < 3 decays at c_tilde (1 - albedo) kappa rho, kappa from rt_kappa at this Z and
      T_dust (10%); with tau per step >= 3 it is absorbed within the first output interval, so it must be gone
      (< 1e-7 of its initial energy);
  (d) the IR band's gain equals the E_abs_tot_toIR routed to it (1e-4).
Energy absorbed in a donor band reaches the IR band once, as the E_abs_tot_toIR source term of the IR update; donating
it directly as well (as the kick did before) makes (a), (b) and (d) fail with an IR gain of 2.000x the donor loss.

With COOLING the gas-opacity share f_gas (~0.5-0.75% here) of the IR absorption heats the gas, and cooling returns
radiation to the bands at c_tilde/c of the gas energy loss. At n_H = 1e3 this moves < 3e-5 of the band energy, so (a),
(b) and (d) are held to 1e-3. At n_H = 1e5 and 1e7 the IR band is optically thick and the bands drain into the gas (see
the gas-share checks below), so (a), (b) and (d) are reported there, not asserted; the IR ledger including the gas share
and the cooling return (as for STARFORGE, below) is asserted instead.

Gas-share checks (every COOLING and STARFORGE variant). Each half-kick gives the gas the share f_gas de_abs of the IR band's in-step
absorption, at the 1:1 band-to-gas convention the code uses for it. From the diagnostic:
  - the gas receives exactly that: its thermal energy change minus the cooling step's own exchange (the cooling-step
    change without the DtInternalEnergy term) equals the gas share taken from the band (1e-4 of the share); not
    asserted with the star, whose HII-region model also changes the gas energy outside these two;
  - none of it is discarded: the DtInternalEnergy the hydro pre-loop drops is zero (FREEZE_HYDRO makes the hydro part
    zero). Deposited as a rate in DtInternalEnergy (as before), the opening half-kick's share was zeroed by the hydro
    pre-loop and the closing one reached the gas once (split cooling) or twice (unsplit, over the full step in the
    cooling solve).
Reported, not asserted: the remaining band drain. The gas re-radiates what it absorbs and the cooling step returns it
at c_tilde/c, so with the 1:1 deposit the bands lose ~(1 - c_tilde/c) of the gas share; the reduced-c convention of
kicks.cc (E_band c/c_tilde + E_gas conserved) would deposit it at c/c_tilde.

STARFORGE variants: the production RT configuration (SINGLE_STAR_STARFORGE_DEFAULTS + SINGLE_STAR_FB_RAD + COOLING,
which add the ionizing band with RT_CHEM_PHOTOION, the STARFORGE cooling, dust and H2 chemistry, radiation pressure,
discrete source injection with sub-grid reprocessing, and RT_ISRF_BACKGROUND), at n_H = 1e3 and 1e7, without sources
and with a 30 Msun main-sequence star at the centre (G ~ 0 and FREEZE_HYDRO keep it from accreting or moving).
RT_ISRF_BACKGROUND=0 switches off the interstellar background the umbrella turns on, whose edge reset would otherwise
set the field in cells within 10% of the box edge every kick. Every energy flow into or out of a band is tallied by
the diagnostic, so:
  - the IR ledger, (IR change - kick source - cooling return - injection + gas share) / routed, must be 1 (1e-6), and
    no absorbed energy may be donated to the IR band directly (it was, a second time, before: 2.000);
  - the band ledger must close: total band change = kick sources + cooling returns + injection - IR gas share, to 1e-4
    of the initial band energy;
  - with the star, the energy injected into the bands over the run must equal c_tilde/c L dt (1e-6);
  - the budget bands + gas thermal - injected is reported, not asserted: it is not conserved under a reduced speed of
    light, and the printed channels name the terms that move it."""

import re
import numpy as np
import h5py
import pytest
import shutil
import sys
from glob import glob
from os import chdir, getcwd
from pathlib import Path
from gizmo.test import build_gizmo_for_test, run_test, assert_final_time, default_mpi_ranks, get_cooling_tables

TEST_NAME = "rt_closed_box"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_rt_closed_box_ics import make_rt_closed_box_ics, Z_SOLAR_SPECIES, U_BANDS_EV_IONIZING  # noqa: E402

PLAIN_RT = ("RT_M1", "RT_COMOVING", "RT_SOURCES=32", "RT_PHOTOELECTRIC", "RT_NUV", "RT_OPTICAL_NIR", "RT_INFRARED",
            "RT_DISABLE_RAD_PRESSURE", "IO_SUPPRESS_OUTPUT_EDDINGTON_TENSOR")
# what COOLING brings in a STARFORGE build (SINGLE_STAR_SINK_DYNAMICS adds the two sub-modules); a bare COOLING has
# neither the low-temperature/dust cooling nor the 11 metal species
PLAIN_COOLING = PLAIN_RT + ("COOLING", "COOL_LOW_TEMPERATURES", "COOL_METAL_LINES_BY_SPECIES")
STARFORGE = ("SINGLE_STAR_STARFORGE_DEFAULTS", "SINGLE_STAR_FB_RAD", "COOLING", "RT_ISRF_BACKGROUND=0")
# G ~ 0: the star binds no gas, so it cannot accrete; the critical density keeps sink formation off
STARFORGE_PARAMS = {"GravityConstantInternal": "1e-100", "CritPhysDensity": "1e30"}
STAR_MASS = 30.0
VARIANTS = {PLAIN_RT: "bare", PLAIN_COOLING: "cooling", STARFORGE: "starforge"}
CHANNELS = ("absorbed", "donated_in", "ir_gas_share", "cooling_to_band", "injected", "kick_source")
BANDS = ("photoelectric", "NUV", "optical-NIR", "IR")
BANDS_STARFORGE = ("ionizing",) + BANDS
IR, DONORS = 3, (0, 1, 2)
KAPPA_SOLAR = np.array([720.0, 480.0, 180.0])  # cm^2/g, rt_kappa's solar dust opacities of the donor bands
ALBEDO = 0.5  # rt_absorb_frac_albedo default for these bands
C_RATIO = 1e-4  # RT_SPEEDOFLIGHT_REDUCTION
C_TILDE_CGS = C_RATIO * 2.9979e10
DT = 3e-3  # code units: MaxSizeTimestep in the params rounds down to TimeMax / 2^6
UNIT_TIME_S, UNIT_LENGTH_CM, UNIT_MASS_G = 3.085678e18 / 1e5, 3.085678e18, 1.989e33
UNIT_RHO_CGS, UNIT_SURFDEN_CGS = UNIT_MASS_G / UNIT_LENGTH_CM**3, UNIT_MASS_G / UNIT_LENGTH_CM**2
TOL_ENERGY = 1e-6
TOL_LEDGER = 1e-4
TOL_COOLING = 1e-3
TOL_IR_LEDGER = 1e-6
TOL_BAND_LEDGER = 1e-4
TOL_INJECTION = 1e-6
TOL_GAS_SHARE = 1e-4  # of the share. The smallest miss with the share deposited as a rate is 5.6e-4 (STARFORGE, 1e7);
# the deposit itself closes to 1e-8, and at n_H = 1e3, where the cooling step moves 1e4 x the share, an untraced 6e-10 of
# the gas thermal energy (6e-6 of the share) is left
TAU_STEP_SATURATED = 3.0


def fdust(t_dust):
    """return_dust_to_metals_ratio_vs_solar under RT_INFRARED: sublimation cutoff at 1500 K"""
    x = t_dust / 1500.0
    s = 9 * (1 - x)
    return 0.5 * (1 + s / np.sqrt(1 + s * s)) * np.exp(-np.minimum(40.0, x * x / 9))


def tag(n_h):
    return f"{n_h:.0e}".replace("+0", "").replace("+", "")


def is_starforge(flags):
    return "SINGLE_STAR_STARFORGE_DEFAULTS" in flags


def ic_name(flags, n_h, star):
    kind = "_starforge" if is_starforge(flags) else ("_cooling" if "COOLING" in flags else "")
    return f"{TEST_NAME}_n{tag(n_h)}{kind}{'_star' if star else ''}_ics"


def output_dir(flags, n_h, star):
    """short on purpose: GIZMO truncates long OutputDir values"""
    return TEST_DIR / f"output_{VARIANTS[flags]}{'_star' if star else ''}_n{tag(n_h)}"


def band_history(flags, n_h, star):
    """times and per-band totals [n_snap, n_bands], IC first; the first snapshot's cells"""
    files = ([TEST_DIR / f"{ic_name(flags, n_h, star)}.hdf5"]
             + sorted(glob(f"{output_dir(flags, n_h, star)}/snapshot_*.hdf5")))
    t, e = [], []
    for fn in files:
        with h5py.File(fn, "r") as F:
            t.append(float(F["Header"].attrs["Time"]))
            e.append(F["PartType0/PhotonEnergy"][:].astype(np.float64).sum(axis=0))
    with h5py.File(files[1], "r") as F:
        g = F["PartType0"]
        first = {k: g[k][:].astype(np.float64) for k in ("Density", "Masses", "PhotonOpacity", "Dust_Temperature",
                                                          "IRBand_Radiation_Temperature")}
    return np.array(t), np.array(e), first


def diag(flags, n_h, star):
    """the RT_DIAG_IR_DOUBLE_COUNT rows, with every flow summed from the start of the run"""
    fn = f"{output_dir(flags, n_h, star)}/rt_ir_diag.txt"
    nb = int(re.search(r"E_rad\[0\.\.(\d+)\]", open(fn).readline()).group(1)) + 1
    d = np.atleast_2d(np.loadtxt(fn))
    o = {"t": d[:, 0], "routed": np.cumsum(d[:, 2]), "lum_injected": np.cumsum(d[:, 3]), "L_sun": d[:, 4],
         "E_thermal": d[:, 5], "E": d[:, 7:7 + nb]}
    for i, c in enumerate(CHANNELS):
        o[c] = np.cumsum(d[:, 7 + nb * (1 + i): 7 + nb * (2 + i)], axis=0)
    o["cooling_gas"] = np.cumsum(d[:, 7 + nb * (1 + len(CHANNELS))])
    o["cooling_offered"] = np.cumsum(d[:, 8 + nb * (1 + len(CHANNELS))])
    return o


def cool_diag(flags, n_h, star):
    """rt_cool_diag.txt (named columns, per-row sums) summed over the run"""
    fn = f"{output_dir(flags, n_h, star)}/rt_cool_diag.txt"
    names = open(fn).readline().split()[2:]
    d = np.atleast_2d(np.loadtxt(fn))
    return {k: d[:, 1 + j].sum() for j, k in enumerate(names)}


_built = {}


def run(flags, n_h, star, num_mpi_ranks):
    """Build once per flag set (its variants differ only in the IC), then run from the snapshot-format IC"""
    cooling, starforge = "COOLING" in flags, is_starforge(flags)
    if cooling:
        get_cooling_tables(str(TEST_DIR))
    kw = {}
    if cooling:
        kw["metallicity"] = Z_SOLAR_SPECIES
    if starforge:
        kw.update(u_bands_ev=U_BANDS_EV_IONIZING, neutral_hydrogen=True, star_mass=STAR_MASS if star else None)
    make_rt_closed_box_ics(str(TEST_DIR / f"{ic_name(flags, n_h, star)}.hdf5"), n_h=n_h, **kw)
    if _built.get("flags") != flags or not (TEST_DIR / "GIZMO").is_file():
        _built.clear()
        build_gizmo_for_test(TEST_NAME, 0, flags)
        _built["flags"] = flags
    out = output_dir(flags, n_h, star)
    shutil.rmtree(out, ignore_errors=True)
    overrides = {"InitCondFile": ic_name(flags, n_h, star), "OutputDir": out.name}
    if starforge:
        overrides.update(STARFORGE_PARAMS)
    cwd = getcwd()
    try:
        chdir(TEST_DIR)
        run_test(TEST_NAME, num_mpi_ranks, 0, timeout=600, restart_flag=2, param_overrides=overrides)
    finally:
        chdir(cwd)
    shutil.copy(TEST_DIR / f"test_{TEST_NAME}.out", out)
    assert_final_time(sorted(glob(f"{out}/snapshot_*.hdf5"))[-1], TEST_NAME)


def check_gas_share(flags, n_h, star):
    """the IR gas share reaches the gas exactly once, and nothing is dropped from DtInternalEnergy. With the star the
    gas also gains and loses energy outside the kick and the cooling solve (HII-region model), so only the second
    holds exactly there and the first is reported"""
    o, c = diag(flags, n_h, star), cool_diag(flags, n_h, star)
    E = o["E"]
    share = o["ir_gas_share"][-1].sum()
    d_th = o["E_thermal"][-1] - o["E_thermal"][0]
    cooling_own = c["gas_dE_cooling"] - c["gas_DtIE_term"]
    delivered = d_th - cooling_own
    band_drain = -(E[-1].sum() - E[0].sum() - o["kick_source"][-1].sum() - o["injected"][-1].sum())
    print(f"  gas share {share:.6g} (opening / closing half-kick {c['gas_IR_share_K1']:.6g} / {c['gas_IR_share_K2']:.6g}); "
          f"gas thermal change {d_th:.6g} = delivered {delivered:.6g} + cooling step {cooling_own:.6g} (incl. CR heating "
          f"{c['gas_CR']:.4g}); delivered / share = {delivered / share:.8f}; DtInternalEnergy dropped by the hydro "
          f"pre-loop {c['gas_DtIE_discarded']:.6g} = {c['gas_DtIE_discarded'] / share:.6f} x share; kick-side gas change "
          f"{c['gas_kick_dE']:.6g}")
    print(f"  band drain {band_drain:.6g} = {band_drain / E[0].sum():.4%} of the band energy = gas share - cooling return "
          f"({o['cooling_to_band'][-1].sum():.6g}); (1 - c_tilde/c) x share = {(1 - C_RATIO) * share:.6g}")
    assert share > 0, "no IR gas share: the check is not exercised"
    if not star:
        assert abs(delivered / share - 1) < TOL_GAS_SHARE, f"the gas receives {delivered / share:.6f} x the IR gas share"
    assert c["gas_DtIE_discarded"] == 0, \
        f"{c['gas_DtIE_discarded'] / share:.6f} x the gas share dropped from DtInternalEnergy by the hydro pre-loop"


def check_bare(flags, n_h, star):
    cooling = "COOLING" in flags
    thick = cooling and n_h >= 1e5
    tol = TOL_COOLING if cooling else TOL_ENERGY
    t, e, first = band_history(flags, n_h, star)
    e_tot = e.sum(axis=1)
    drift = e_tot / e_tot[0] - 1
    gain, loss = e[:, IR] - e[0, IR], e[0, DONORS].sum() - e[:, DONORS].sum(axis=1)
    o = diag(flags, n_h, star)
    routed, d_ir = o["routed"], o["E"][:, IR] - o["E"][0, IR]
    d_donor = o["E"][0, DONORS].sum() - o["E"][:, DONORS].sum(axis=1)

    rho_cgs = first["Density"] * UNIT_RHO_CGS
    dx_cm = (first["Masses"] / first["Density"]) ** (1 / 3) * UNIT_LENGTH_CM
    kappa = first["PhotonOpacity"] / UNIT_SURFDEN_CGS  # rt_kappa, cm^2/g
    tau_cell = np.median(kappa * (rho_cgs * dx_cm)[:, None], axis=0)
    t_dust = np.median(first["Dust_Temperature"])
    print(f"\n{VARIANTS[flags]}, n_H = {n_h:g}: at t = {t[1]:g} T_dust {t_dust:.4g} K, T_rad "
          f"{np.median(first['IRBand_Radiation_Temperature']):.4g} K, median kappa [cm^2/g] "
          f"{np.median(kappa, axis=0)}, tau per cell {tau_cell}")
    print("   time    " + "  ".join(f"{b:>13s}" for b in BANDS) + "   total/total0-1   IR gain/donor loss")
    for i in range(len(t)):
        ratio = gain[i] / loss[i] if loss[i] > 0 else np.nan
        print(f"  {t[i]:.4f}  " + "  ".join(f"{x:13.6e}" for x in e[i]) + f"   {drift[i]:+.3e}        {ratio:.6f}")
    sel = routed > 0
    print(f"diagnostic ledger, last row: IR gain / routed = {d_ir[-1] / routed[-1]:.6f}, "
          f"donor loss / routed = {d_donor[-1] / routed[-1]:.6f}; cumulative IR gain / routed over the rows in "
          f"[{np.min(d_ir[sel] / routed[sel]):.6f}, {np.max(d_ir[sel] / routed[sel]):.6f}]")

    # (c) a snapshot is written after the opening half-kick of the step that reaches the output time, so at the first
    # snapshot the bands have been absorbed for t - DT/2
    for k in DONORS:
        expected = C_TILDE_CGS * (1 - ALBEDO) * KAPPA_SOLAR[k] * fdust(t_dust) * np.mean(rho_cgs)
        tau_step = expected * DT * UNIT_TIME_S
        frac = e[1, k] / e[0, k]
        if tau_step < TAU_STEP_SATURATED:
            rate = -np.log(frac) / ((t[1] - DT / 2) * UNIT_TIME_S)
            print(f"{BANDS[k]}: tau per step {tau_step:.3g}, decay rate {rate:.4e} /s, c_tilde (1-albedo) kappa rho = "
                  f"{expected:.4e} /s, ratio {rate / expected:.4f}")
            assert abs(rate / expected - 1) < 0.1, f"{BANDS[k]} decays at {rate / expected:.3f}x the dust absorption rate"
        else:
            print(f"{BANDS[k]}: tau per step {tau_step:.3g}, E/E0 at t = {t[1]:g}: {frac:.3e}")
            assert frac < 1e-7, f"{BANDS[k]} (tau per step {tau_step:.3g}) keeps {frac:.3e} of its energy"

    assert loss[-1] > 0.5 * e[0, DONORS].sum(), "the donor bands did not decay: the test is not exercising absorption"
    if cooling:
        ch = {k: o[k][-1] for k in CHANNELS}
        ir_ledger = (d_ir[-1] - ch["kick_source"][IR] - ch["cooling_to_band"][IR] + ch["ir_gas_share"][IR]) / routed[-1]
        print(f"  IR ledger: (IR change - kick source - cooling return + gas share) / routed = {ir_ledger:.8f}")
        check_gas_share(flags, n_h, star)
        assert abs(ir_ledger - 1) < TOL_IR_LEDGER, f"IR gain = {ir_ledger:.6f}x the energy routed to it"
        assert ch["donated_in"][IR] == 0, "absorbed donor energy donated to the IR band directly"
    if thick:
        return
    assert np.max(np.abs(drift)) < tol, f"total band energy not conserved: max |E/E0 - 1| = {np.max(np.abs(drift)):.3e}"
    assert np.max(np.abs(gain[1:] - loss[1:]) / e_tot[0]) < tol, \
        f"IR gain / donor loss = {gain[-1] / loss[-1]:.6f} at the end"
    tol_ledger = max(tol, TOL_LEDGER)
    assert abs(d_ir[-1] / routed[-1] - 1) < tol_ledger and abs(d_donor[-1] / routed[-1] - 1) < tol_ledger, \
        f"IR gain = {d_ir[-1] / routed[-1]:.6f}x the absorbed donor energy routed to it"


def check_starforge(flags, n_h, star):
    o = diag(flags, n_h, star)
    E, routed = o["E"], o["routed"]
    nb = E.shape[1]
    ir, h0, opt = nb - 1, 0, nb - 2
    ch = {c: o[c][-1] for c in CHANNELS}
    d_ir = E[-1, ir] - E[0, ir]
    ir_ledger = (d_ir - ch["kick_source"][ir] - ch["cooling_to_band"][ir] - ch["injected"][ir] + ch["ir_gas_share"][ir]) / routed[-1]
    band_change = E[-1].sum() - E[0].sum()
    band_flows = (ch["kick_source"].sum() + ch["cooling_to_band"].sum() + ch["injected"].sum() - ch["ir_gas_share"][ir]
                  + ch["donated_in"][ir])
    residual = (band_change - band_flows) / E[0].sum()
    injected = o["injected"].sum(axis=1)
    budget = E.sum(axis=1) + o["E_thermal"] - injected

    print(f"\n{VARIANTS[flags]}{' + star' if star else ''}, n_H = {n_h:g}: star luminosity {o['L_sun'][-1]:.4g} L_sun")
    print("  band            E(0)         E(end)      absorbed   donated in   kick source  cooling ret.   injected")
    for k, b in enumerate(BANDS_STARFORGE):
        print(f"  {b:13s} {E[0, k]:11.5g} {E[-1, k]:12.5g} {ch['absorbed'][k]:12.5g} {ch['donated_in'][k]:12.5g} "
              f"{ch['kick_source'][k]:12.5g} {ch['cooling_to_band'][k]:12.5g} {ch['injected'][k]:12.5g}")
    print(f"  IR ledger: (IR change - kick source - cooling return - injection + gas share) / routed = {ir_ledger:.8f}; "
          f"direct donation to IR / routed = {ch['donated_in'][ir] / routed[-1]:.6f}; routed {routed[-1]:.6g}")
    print(f"  band ledger residual {residual:+.3e} of the initial band energy")
    print(f"  ionizing band: absorbed {ch['absorbed'][h0]:.6g}, all donated to optical-NIR ({ch['donated_in'][opt]:.6g}); "
          f"gas thermal energy peaks at {o['E_thermal'].max():.4g} (t = {o['t'][np.argmax(o['E_thermal'])]:.3g}) from "
          f"{o['E_thermal'][0]:.4g}")
    if star:
        print(f"  injected {injected[-1]:.6g} = (1 {injected[-1] / o['lum_injected'][-1] - 1:+.2e}) x c_tilde/c L dt")
    print(f"  budget bands + gas thermal - injected, and the flows that move it (cumulative; band units unless noted):")
    print("     time      budget/b0-1   IR gas share  cooling return  gas emission routed (c/c_tilde x offered)  "
          "cooling-step gas dE   kick sources")
    for t_out in np.arange(0, o["t"][-1] + 1e-9, 0.024):
        i = int(np.argmin(np.abs(o["t"] - t_out)))
        print(f"    {o['t'][i]:.4f}   {budget[i] / budget[0] - 1:+.4e}   {o['ir_gas_share'][i, ir]:12.5g}   "
              f"{o['cooling_to_band'][i].sum():12.5g}   {o['cooling_offered'][i] / C_RATIO:16.5g}"
              f"                        {o['cooling_gas'][i]:12.5g}   {o['kick_source'][i].sum():12.5g}")

    check_gas_share(flags, n_h, star)
    assert routed[-1] > 0.5 * E[0, 1:ir].sum(), "the donor bands were not absorbed: the test is not exercising them"
    assert abs(residual) < TOL_BAND_LEDGER, f"band ledger does not close: residual {residual:+.3e} of the band energy"
    if star:
        assert abs(injected[-1] / o["lum_injected"][-1] - 1) < TOL_INJECTION, "injected energy is not c_tilde/c L dt"
    assert abs(ir_ledger - 1) < TOL_IR_LEDGER, f"IR gain = {ir_ledger:.6f}x the energy routed to it"
    assert ch["donated_in"][ir] == 0, f"{ch['donated_in'][ir] / routed[-1]:.6f} x routed donated to the IR band directly"


def case(flags, n_h, *marks, star=False):
    return pytest.param(flags, n_h, star, id=f"{VARIANTS[flags]}{'_star' if star else ''}-n{tag(n_h)}", marks=marks)


# grouped by flag set, so each set is built once
CASES = ([case(PLAIN_RT, n) for n in (1e3, 1e4, 1e5, 1e7)]
         + [case(PLAIN_COOLING, n) for n in (1e3, 1e5, 1e7)]
         + [case(STARFORGE, n, star=s) for n in (1e3, 1e7) for s in (False, True)])


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=8),))
@pytest.mark.parametrize("flags,n_h,star", CASES)
def test_rt_closed_box(flags, n_h, star, num_mpi_ranks):
    run(flags, n_h, star, num_mpi_ranks)
    (check_starforge if is_starforge(flags) else check_bare)(flags, n_h, star)
