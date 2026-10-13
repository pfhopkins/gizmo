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

Point-source variant: the bare-RT bands plus an ionizing band (RT_CHEM_PHOTOION, no COOLING), with a type-4 source
at the centre (RT_SOURCES=16, no gas sources) of luminosity IonizingLuminosityPerSolarMass_cgs x its mass, injected
continuously (without RT_INJECT_PHOTONS_DISCRETELY a source's injection is a rate in its neighbours' Rad_Je, applied by
the kicks). The ionizing band starts empty; what it absorbs is donated to the optical-NIR band and on to the IR. Every
row of the diagnostic must show the bands gaining c_tilde/c L dt through their kick source terms (transport sums to zero
over the box), and the total band energy must grow by c_tilde/c L t (1e-6 for both). Rad_Je has to be zeroed before
each injection: if it is not, every injection adds to the last and the injected rate grows step by step.
With RT_ILIEV_TEST1 the source emits 5e48 ionizing photons/s whatever its mass, and photons are counted at the band's
number-weighted mean energy rt_nu_eff_eV (a blackbody at star_Teff over 13.6-500 eV, recomputed here as rt_get_sigma
does), so the same checks hold with L = 5e48 rt_nu_eff_eV; emitting 5e48 x 13.6 eV instead gives 13.6/rt_nu_eff_eV of
the photons.

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
    light, and the printed channels name the terms that move it.

Molecular variant (EOS check): the STARFORGE build plus EOS_GAMMA_PROBE at n_H = 1e4, starting fully molecular at 300 K,
so the gas cools through the H2 rotational regime and its adiabatic index changes between EOS updates. After every
update the sound speed must be the ideal-gas one for the adiabatic index of the temperature that update computed:
|SoundSpeed^2 rho / (gamma P) - 1| < 1e-12, with gamma spanning > 0.05. Taking gamma before the temperature refresh
leaves the sound speed one update behind.

STARFORGE in IR-thick gas: the STARFORGE variant also runs at n_H = 1e8 and 1e9, and from a hot start in local thermal
equilibrium (LTE: fully molecular gas, dust and the IR band at T_LTE = 200 or 500 K, the IR band at a T^4, the other bands
empty; n_H = 1e9). It reports the box in the convention a jaco model conserves, E_gas + sum_b E_b c/c_tilde, with the flows
that move it: the IR gas share (deposited 1:1, so the ledger loses c/c_tilde - 1 of it), the cooling step's net heating
that no band pays or receives (photoheating, photoelectric and cosmic-ray heating), and the return limiters.

jaco variants (JACO=<model> on the STARFORGE build, JACO_MODELS): the model's cooling step does every exchange between the
bands, the gas and the dust; the kicks only transport. A model conserves E_gas + sum_b E_b c/c_tilde (the dust a
reservoir of zero capacity) in every exchange except those it documents. cooling/jaco.cc tallies, per solved cell, the
step's change of that ledger and Delta_t R, R the model's net ledger rate at the solved state (energy row, band rows
weighted by their energy over c_tilde/c, dust row; conserving exchanges cancel in it), split by switching off one thing
at a time: cosmic rays, photoionization heating, the photoelectric band, and the reduced speed of light (what depends on
c_tilde/c once the photoelectric band is empty: in starforge_legacy_RT the IR band's gas absorption at c_tilde/c; zero in
a model whose every exchange is in the c/c_tilde convention). At n_H = 1e7, 1e8, 1e9 with and without the star, and in
the LTE boxes:
  - the step changes the ledger by Delta_t (R + PdV) to within the solver's bound (tol times its row scales) in every row,
    and to 1e-6 of the ledger over the run (measured 1e-11 to 1.5e-9);
  - end to end, the box's ledger moves only by R, the PdV work, the kicks' band sources and what the kick's positivity
    clamp creates (1e-6): nothing outside the model's step creates or destroys energy (with the star, the IR kick
    sources around it do not sum to zero, about -0.8% of the injected energy; GIZMO's own STARFORGE run shows the same).
    The clamp (kick_limiter channel) fires when the solve empties a cell's ionizing band and the closing half-kick then
    applies the flux divergence computed before the solve: 3.3e-6 of the ledger in the star box at n_H = 1e3, 99% of it
    before t = 0.03 while the initial ionizing field is absorbed, bounded at 1e-5. GIZMO's 1% band floor, which legacy
    keeps, is compiled out under JACO; the conservative fix is the transport term as a source in the solve;
  - the parts of R the model does not document, and what is left of R past its documented parts, are < 1e-6 of the
    ledger; the band ledger, injection and gas-share checks as for STARFORGE.
starforge_legacy_RT documents all four parts. Its IR gas absorption drains 26-45% of the ledger at n_H >= 1e7 over the
run, as GIZMO's kick does; test_rt_closed_box_gas_share_vs_legacy compares the two drains per unit band energy in the LTE
boxes (1e-2). GIZMO gives the gas kappa_gas / kappa(0,0) of the absorbed energy, kappa(0,0) the extinction, whose dust
part lacks the absorption factor A(T_rad) = 1 - 0.5 / (1 + 725^2 / (1 + T_rad^2)): without A(T_rad) in its kick gas
share (jaco before legacy_rt_gas_share), starforge_legacy_RT drains 1/A(T_rad) times as fast, 1.036 at 200 K and
1.190 at 500 K."""

import os
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
POINT_SOURCE = tuple(f for f in PLAIN_RT if not f.startswith("RT_SOURCES")) + ("RT_SOURCES=16", "RT_CHEM_PHOTOION=1")
ILIEV = POINT_SOURCE + ("RT_ILIEV_TEST1",)
Q_ILIEV = 5e48  # ionizing photons/s
SOURCE_MASS = 1.0  # Msun
SOURCE_PARAMS = {"IonizingLuminosityPerSolarMass_cgs": "1e38", "star_Teff": "1e5"}
U_BANDS_EV_SOURCE = (0.0, 20.0, 20.0, 20.0, 30.0)  # ionizing band empty at the start
STARFORGE = ("SINGLE_STAR_STARFORGE_DEFAULTS", "SINGLE_STAR_FB_RAD", "COOLING", "RT_ISRF_BACKGROUND=0")
MOLECULAR = STARFORGE + ("EOS_GAMMA_PROBE",)
T_MOLECULAR = 300.0  # K
TOL_GAMMA_PROBE = 1e-12
# G ~ 0: the star binds no gas, so it cannot accrete; the critical density keeps sink formation off
STARFORGE_PARAMS = {"GravityConstantInternal": "1e-100", "CritPhysDensity": "1e30"}
STAR_MASS = 30.0
# jaco models of the STARFORGE coupling on the STARFORGE build: the model's cooling step does every exchange between the bands,
# the gas and the dust, the kicks only transport. Per model, the parts of its ledger rate that it documents as sources or
# sinks (cr: cosmic-ray heating; photoion: photoionization heating; fuv: what the photoelectric band drives; rsol: what
# depends on the reduced speed of light, in starforge_legacy_RT the IR band's gas absorption at c_tilde/c); every other
# part must vanish. reproduces_legacy: the model must drain the IR band into the gas as GIZMO's kick does
JACO_MODELS = {"starforge_legacy_RT": {"documented": ("cr", "photoion", "fuv", "rsol"), "reproduces_legacy": True}}
JACO_PARTS = ("cr", "photoion", "fuv", "rsol")
JACO_DENSITIES = (1e7, 1e8, 1e9)
# hot, IR-thick start in local thermal equilibrium: fully molecular gas, dust and the IR band at T_LTE, the IR band at a T^4,
# the other bands empty (n_H = 1e9: a protostellar core's first-core gas)
T_LTE = (200.0, 500.0)
N_LTE = 1e9
A_RAD = 7.5657e-15  # erg cm^-3 K^-4


def jaco_flags(model):
    return STARFORGE + (f"JACO={model}",)


VARIANTS = {PLAIN_RT: "bare", PLAIN_COOLING: "cooling", POINT_SOURCE: "source", ILIEV: "iliev", STARFORGE: "starforge",
            MOLECULAR: "molecular", **{jaco_flags(m): f"jaco_{m}" for m in JACO_MODELS}}
CHANNELS = ("absorbed", "donated_in", "ir_gas_share", "cooling_to_band", "injected", "kick_source", "kick_limiter")
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
UNIT_LUM_CGS = UNIT_MASS_G * 1e10 / UNIT_TIME_S
EV_ERG, K_BOLTZMANN = 1.60217733e-12, 1.38066e-16  # GIZMO's ELECTRONVOLT_IN_ERGS, BOLTZMANN_CGS
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
TIME_FIRST_OUTPUT = 0.024  # TimeBetSnapshot; the restart ramp's steps all fall before it
TOL_JACO_SOLVE = 1e-6  # the jaco solver's relative tolerance (JACO_TOL_DEFAULT), on the ledger it changes per step
TOL_JACO_UNDOCUMENTED = 1e-6  # of the ledger: any part of the model's ledger rate it does not document
TOL_KICK_LIMITER = 1e-5  # of the ledger: what the kick's positivity clamp creates (3.3e-6 in the star box at n_H = 1e3, see below)
TOL_DRAIN_COEFFICIENT = 1e-2  # jaco over GIZMO, of the IR band's drain into the gas per unit band energy
OUT_ROOT = Path(os.environ.get("RT_CLOSED_BOX_OUTPUT_ROOT", TEST_DIR))  # absolute, < 100 characters (OutputDir)


def fdust(t_dust):
    """return_dust_to_metals_ratio_vs_solar under RT_INFRARED: sublimation cutoff at 1500 K"""
    x = t_dust / 1500.0
    s = 9 * (1 - x)
    return 0.5 * (1 + s / np.sqrt(1 + s * s)) * np.exp(-np.minimum(40.0, x * x / 9))


def tag(n_h):
    return f"{n_h:.0e}".replace("+0", "").replace("+", "")


def is_starforge(flags):
    return "SINGLE_STAR_STARFORGE_DEFAULTS" in flags


def is_molecular(flags):
    return "EOS_GAMMA_PROBE" in flags


def is_source(flags):
    return "RT_SOURCES=16" in flags


def jaco_model(flags):
    return next((f.split("=", 1)[1] for f in flags if f.startswith("JACO=")), None)


def setup_tag(star, lte):
    return f"{'_star' if star else ''}{f'_lte{lte:g}' if lte else ''}"


def ic_name(flags, n_h, star, lte=None):
    kind = ("_molecular" if is_molecular(flags) else f"_jaco_{jaco_model(flags)}" if jaco_model(flags) else "_starforge"
            if is_starforge(flags) else "_cooling" if "COOLING" in flags else "_source" if is_source(flags) else "")
    return f"{TEST_NAME}_n{tag(n_h)}{kind}{setup_tag(star, lte)}_ics"


def output_dir(flags, n_h, star, lte=None):
    """short on purpose: GIZMO truncates long OutputDir values"""
    return OUT_ROOT / f"output_{VARIANTS[flags]}{setup_tag(star, lte)}_n{tag(n_h)}"


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


def diag(flags, n_h, star, lte=None):
    """the RT_DIAG_IR_DOUBLE_COUNT rows, with every flow summed from the start of the run"""
    fn = f"{output_dir(flags, n_h, star, lte)}/rt_ir_diag.txt"
    nb = int(re.search(r"E_rad\[0\.\.(\d+)\]", open(fn).readline()).group(1)) + 1
    d = np.atleast_2d(np.loadtxt(fn))
    o = {"t": d[:, 0], "routed": np.cumsum(d[:, 2]), "lum_injected": np.cumsum(d[:, 3]), "L_sun": d[:, 4],
         "E_thermal": d[:, 5], "E": d[:, 7:7 + nb]}
    for i, c in enumerate(CHANNELS):
        o[c] = np.cumsum(d[:, 7 + nb * (1 + i): 7 + nb * (2 + i)], axis=0)
    o["cooling_gas"] = np.cumsum(d[:, 7 + nb * (1 + len(CHANNELS))])
    o["cooling_offered"] = np.cumsum(d[:, 8 + nb * (1 + len(CHANNELS))])
    return o


def cool_diag(flags, n_h, star, cumulative=False, lte=None):
    """rt_cool_diag.txt (named columns, per-row sums) summed over the run, or summed from the start at each row"""
    fn = f"{output_dir(flags, n_h, star, lte)}/rt_cool_diag.txt"
    names = open(fn).readline().split()[2:]
    d = np.atleast_2d(np.loadtxt(fn))
    if cumulative:
        return {k: np.cumsum(d[:, 1 + j]) for j, k in enumerate(names)}
    return {k: d[:, 1 + j].sum() for j, k in enumerate(names)}


def thermal_history(flags, n_h, star, lte=None):
    """per snapshot: time and the gas-mass-weighted gas, dust and IR radiation temperatures and IR band energy density
    [eV cm^-3], over the whole box and over the cells within 2 pc of its centre (the star's neighbourhood)"""
    rows = []
    for fn in sorted(glob(f"{output_dir(flags, n_h, star, lte)}/snapshot_*.hdf5")):
        with h5py.File(fn, "r") as F:
            g, box = F["PartType0"], F["Header"].attrs["BoxSize"]
            m, rho = g["Masses"][:].astype(np.float64), g["Density"][:].astype(np.float64)
            r = np.linalg.norm(g["Coordinates"][:] - 0.5 * box, axis=1)
            q = (g["Temperature"][:], g["Dust_Temperature"][:], g["IRBand_Radiation_Temperature"][:],
                 g["PhotonEnergy"][:, -1] * rho / m * UNIT_RHO_CGS * 1e10 / EV_ERG)
            near = r < 2.0
            rows.append((F["Header"].attrs["Time"],) + tuple(np.average(x, weights=m) for x in q)
                        + tuple(np.average(x[near], weights=m[near]) for x in q))
    return np.array(rows)


def print_thermal_history(flags, n_h, star, lte=None):
    h = thermal_history(flags, n_h, star, lte)
    print("     time    box: T_gas    T_dust    T_rad   u_IR[eV/cm^3]   r < 2 pc: T_gas    T_dust    T_rad   u_IR")
    for row in h:
        print(f"    {row[0]:.4f}   " + " ".join(f"{x:9.4g}" for x in row[1:5]) + "        " + " ".join(f"{x:9.4g}" for x in row[5:]))


_built = {}


def run(flags, n_h, star, num_mpi_ranks, lte=None):
    """Build once per flag set (its variants differ only in the IC), then run from the snapshot-format IC"""
    cooling, starforge = "COOLING" in flags, is_starforge(flags)
    if cooling:
        get_cooling_tables(str(TEST_DIR))
    kw = {}
    if cooling:
        kw["metallicity"] = Z_SOLAR_SPECIES
    if starforge:
        kw.update(u_bands_ev=U_BANDS_EV_IONIZING, neutral_hydrogen=True, star_mass=STAR_MASS if star else None)
    if is_molecular(flags):
        kw.update(t_gas=T_MOLECULAR, h2_per_neutral_h=1.0)
    if is_source(flags):
        kw.update(u_bands_ev=U_BANDS_EV_SOURCE, neutral_hydrogen=True, source_mass=SOURCE_MASS)
    if lte:
        kw.update(t_gas=lte, t_rad=lte, u_bands_ev=(0.0,) * 4 + (A_RAD * lte**4 / EV_ERG,), h2_per_neutral_h=1.0)
    make_rt_closed_box_ics(str(TEST_DIR / f"{ic_name(flags, n_h, star, lte)}.hdf5"), n_h=n_h, **kw)
    if _built.get("flags") != flags or not (TEST_DIR / "GIZMO").is_file():
        _built.clear()
        build_gizmo_for_test(TEST_NAME, 0, flags)
        _built["flags"] = flags
    out = output_dir(flags, n_h, star, lte)
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    overrides = {"InitCondFile": ic_name(flags, n_h, star, lte), "OutputDir": out.name if OUT_ROOT == TEST_DIR else str(out)}
    if starforge:
        overrides.update(STARFORGE_PARAMS)
    if is_source(flags):
        overrides.update(SOURCE_PARAMS)
    cwd = getcwd()
    try:
        chdir(TEST_DIR)
        run_test(TEST_NAME, num_mpi_ranks, 0, timeout=600, restart_flag=2, param_overrides=overrides)
    finally:
        chdir(cwd)
    shutil.copy(TEST_DIR / f"test_{TEST_NAME}.out", out)
    assert_final_time(sorted(glob(f"{out}/snapshot_*.hdf5"))[-1], TEST_NAME)


def check_gas_share(flags, n_h, star, lte=None):
    """the IR gas share reaches the gas exactly once, and nothing is dropped from DtInternalEnergy. With the star the
    gas also gains and loses energy outside the kick and the cooling solve (HII-region model), so only the second
    holds exactly there and the first is reported"""
    o, c = diag(flags, n_h, star, lte), cool_diag(flags, n_h, star, lte=lte)
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


def nu_eff_ev(t_eff):
    """rt_get_sigma's single-band rt_nu_eff_eV: number-weighted mean energy of a blackbody over 13.6-500 eV"""
    n = 10000
    e = 13.6 + np.arange(n) * ((500.0 - 13.6) / (n - 1))
    i_nu = e**3 / np.expm1(e * EV_ERG / (K_BOLTZMANN * t_eff))  # constant factors cancel in the ratio
    return i_nu.sum() / (i_nu / e).sum()


def source_luminosity(flags):
    """the source's luminosity in code units"""
    if "RT_ILIEV_TEST1" in flags:
        return Q_ILIEV * nu_eff_ev(float(SOURCE_PARAMS["star_Teff"])) * EV_ERG / UNIT_LUM_CGS
    return float(SOURCE_PARAMS["IonizingLuminosityPerSolarMass_cgs"]) * SOURCE_MASS / UNIT_LUM_CGS


def check_source(flags, n_h, star):
    o = diag(flags, n_h, star)
    l_code = source_luminosity(flags)
    t, E = o["t"], o["E"].sum(axis=1)
    injected = np.diff(o["kick_source"].sum(axis=1))  # per row: transport sums to zero over the box
    rate = injected / np.diff(t) / (C_RATIO * l_code)
    gain = (E[1:] - E[0]) / (C_RATIO * l_code * t[1:])
    if "RT_ILIEV_TEST1" in flags:
        nu = nu_eff_ev(float(SOURCE_PARAMS["star_Teff"]))
        q = injected / np.diff(t) / C_RATIO * UNIT_LUM_CGS / (nu * EV_ERG)
        print(f"\nrt_nu_eff_eV = {nu:.6f} eV; ionizing photons injected per second: first {q[0]:.6e}, last {q[-1]:.6e}")
    print(f"\n{VARIANTS[flags]}, n_H = {n_h:g}: L = {l_code:.6g} code units; per-row injection / (c_tilde/c L dt): first "
          f"{np.round(rate[:4], 8)}, last {np.round(rate[-3:], 8)}; band energy gain / (c_tilde/c L t) at the end "
          f"{gain[-1]:.8f}")
    assert np.max(np.abs(rate - 1)) < TOL_INJECTION, \
        f"injected per step / (c_tilde/c L dt) ranges over [{rate.min():.6g}, {rate.max():.6g}]"
    assert abs(gain[-1] - 1) < TOL_INJECTION, f"band energy gain = {gain[-1]:.6f} x c_tilde/c L t"


def check_starforge(flags, n_h, star, lte=None):
    o = diag(flags, n_h, star, lte)
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

    print(f"\n{VARIANTS[flags]}{' + star' if star else ''}{f', LTE at {lte:g} K' if lte else ''}, n_H = {n_h:g}: star "
          f"luminosity {o['L_sun'][-1]:.4g} L_sun")
    print("  band            E(0)         E(end)      absorbed   donated in   kick source  cooling ret.   injected")
    for k, b in enumerate(BANDS_STARFORGE):
        print(f"  {b:13s} {E[0, k]:11.5g} {E[-1, k]:12.5g} {ch['absorbed'][k]:12.5g} {ch['donated_in'][k]:12.5g} "
              f"{ch['kick_source'][k]:12.5g} {ch['cooling_to_band'][k]:12.5g} {ch['injected'][k]:12.5g}")
    if routed[-1] > 0:
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

    # the same box in the c/c_tilde convention a jaco model conserves (check_jaco), with the flows that move it
    c_cum = cool_diag(flags, n_h, star, cumulative=True, lte=lte)
    ledger = o["E_thermal"] + (E.sum(axis=1) - injected) / C_RATIO
    share_term = -(1 / C_RATIO - 1) * o["ir_gas_share"][:, ir]  # band loss at c/c_tilde, gas gain at 1
    unrouted = c_cum["gas_unrouted"]  # the cooling step's net heating that no band pays or receives
    clips = (sum(c_cum[k] for k in c_cum if k.startswith("band_clip_min")) -
             sum(c_cum[k] for k in c_cum if k.startswith("band_clip_max"))) / C_RATIO
    kick = o["kick_source"].sum(axis=1) / C_RATIO
    rest = ledger - ledger[0] - share_term - unrouted - clips - kick
    print(f"  ledger E_gas + (bands - injected) c/c_tilde, and the flows that move it (cumulative, / its start {ledger[0]:.6g}):")
    print("     time      ledger/l0-1   IR gas share   unrouted heat   return clips   kick sources   rest")
    for t_out in np.arange(0, o["t"][-1] + 1e-9, TIME_FIRST_OUTPUT):
        i = int(np.argmin(np.abs(o["t"] - t_out)))
        print(f"    {o['t'][i]:.4f}   " + "   ".join(f"{x / ledger[0]:+.4e}" for x in
                                                     (ledger[i] - ledger[0], share_term[i], unrouted[i], clips[i], kick[i], rest[i])))
    print_thermal_history(flags, n_h, star, lte)
    print(f"  IR gas-absorption drain coefficient {drain_coefficient(flags, n_h, star, lte):.6g} per code time")

    check_gas_share(flags, n_h, star, lte)
    assert abs(residual) < TOL_BAND_LEDGER, f"band ledger does not close: residual {residual:+.3e} of the band energy"
    if lte:  # nothing to donate
        return
    assert routed[-1] > 0.5 * E[0, 1:ir].sum(), "the donor bands were not absorbed: the test is not exercising them"
    if star:
        assert abs(injected[-1] / o["lum_injected"][-1] - 1) < TOL_INJECTION, "injected energy is not c_tilde/c L dt"
    assert abs(ir_ledger - 1) < TOL_IR_LEDGER, f"IR gain = {ir_ledger:.6f}x the energy routed to it"
    assert ch["donated_in"][ir] == 0, f"{ch['donated_in'][ir] / routed[-1]:.6f} x routed donated to the IR band directly"


def drain_coefficient(flags, n_h, star, lte=None):
    """The IR band's drain into the gas through its gas absorption, per unit time and per unit of the band's energy (in
    the c/c_tilde ledger, so the ratio is the same in either convention), over the run after the first output: GIZMO's
    kick moves the gas share at 1:1 (the ledger loses c/c_tilde - 1 of it), the jaco model reports it (jaco_R_rsol_dt)"""
    o, c = diag(flags, n_h, star, lte), cool_diag(flags, n_h, star, cumulative=True, lte=lte)
    drain = c["jaco_R_rsol_dt"] if jaco_model(flags) else -(1 / C_RATIO - 1) * o["ir_gas_share"][:, -1]
    t, e_ir = o["t"], o["E"][:, -1] / C_RATIO
    i = int(np.argmin(np.abs(t - TIME_FIRST_OUTPUT)))
    return -(drain[-1] - drain[i]) / np.sum(0.5 * (e_ir[i + 1:] + e_ir[i:-1]) * np.diff(t[i:]))


def check_jaco(flags, n_h, star, lte=None):
    """A jaco model's energy ledger. The model conserves E_gas + sum_b E_b c/c_tilde (the dust a reservoir of zero
    capacity) in every exchange but those it documents (JACO_MODELS); cooling/jaco.cc tallies, per solved cell, the
    step's change of that ledger, Delta_t R (R: the model's net ledger rate at the solved state, the conserving
    exchanges cancelled) with the documented parts switched off one at a time, and the PdV work."""
    model = jaco_model(flags)
    o, c = diag(flags, n_h, star, lte), cool_diag(flags, n_h, star, cumulative=True, lte=lte)
    E, inj = o["E"], o["injected"].sum(axis=1)
    e0_bands = E[0].sum()
    band_flows = o["kick_source"].sum(axis=1) + o["cooling_to_band"].sum(axis=1) + inj
    band_residual = (E.sum(axis=1) - e0_bands - band_flows) / e0_bands
    ledger = o["E_thermal"] + (E.sum(axis=1) - inj) / C_RATIO  # E_gas + sum_b E_b c/c_tilde, less the injection
    scale = ledger[0]
    R, pdv, kick = c["jaco_R_dt"], c["jaco_pdv_dt"], o["kick_source"].sum(axis=1) / C_RATIO
    parts = {k: c[f"jaco_R_{k}_dt"] for k in JACO_PARTS}
    rest = R - sum(parts.values())
    # end to end: the box's ledger against the model's own rate, the PdV work and the kicks' band sources (transport,
    # which sums to zero over the box, and the radiation work terms)
    limiter = o["kick_limiter"].sum(axis=1) / C_RATIO  # energy the kick's positivity clamp created (under JACO there is no floor; the clamp fires on an emptied band whose transport term is still negative)
    drift = ledger - ledger[0] - R - pdv - kick - limiter
    gas_outside = o["E_thermal"] - o["E_thermal"][0] - c["jaco_gas_dE"] - c["gas_kick_dE"]
    err_abs, bound = (np.diff(c[k], prepend=0) for k in ("jaco_ledger_err_abs", "jaco_ledger_bound"))  # per row
    legacy = o["E_thermal"] + E.sum(axis=1) - inj  # the 1:1 band/gas convention
    print(f"\n{VARIANTS[flags]}{' + star' if star else ''}{f', LTE at {lte:g} K' if lte else ''}, n_H = {n_h:g}: "
          f"{c['jaco_n_solves'][-1]:.0f} solves, {c['jaco_n_subcycled'][-1]:.0f} subcycled; star luminosity {o['L_sun'][-1]:.4g} L_sun")
    print(f"  ledger E_gas + sum_b E_b c/c_tilde - injected c/c_tilde: start {scale:.6g} (gas {o['E_thermal'][0]:.4g}), end "
          f"{ledger[-1]:.6g}; model rate integral {R[-1]:+.6g} = " + " + ".join(f"{k} {parts[k][-1]:+.4g}" for k in JACO_PARTS)
          + f" + rest {rest[-1]:+.4g}; PdV {pdv[-1]:+.4g}; kick sources {kick[-1]:+.4g}; kick limiter {limiter[-1]:+.4g} "
          f"({limiter[-1] / scale:+.3e})")
    print(f"  solve: step change - Delta_t (R + PdV) = {c['jaco_ledger_err'][-1]:+.4g} = {c['jaco_ledger_err'][-1] / scale:+.3e} of "
          f"the ledger (sum |.| {c['jaco_ledger_err_abs'][-1]:.4g}, at most {np.max(err_abs / np.maximum(bound, 1e-300)):.3e} of "
          f"the solver's bound in a row; subcycled {c['jaco_ledger_err_subcycled'][-1]:+.4g})")
    print(f"  band ledger residual {band_residual[-1]:+.3e} of the initial band energy; gas change outside the jaco step and "
          f"the kick {gas_outside[-1]:+.4g}; IR gas-absorption drain coefficient {drain_coefficient(flags, n_h, star, lte):.6g} "
          f"per code time")
    print("     time    ledger/l0-1   drift/l0     R/l0: total     " + "    ".join(f"{k:>8s}" for k in JACO_PARTS)
          + "       rest    1:1 budget/b0-1")
    for t_out in np.arange(0, o["t"][-1] + 1e-9, TIME_FIRST_OUTPUT):
        i = int(np.argmin(np.abs(o["t"] - t_out)))
        print(f"    {o['t'][i]:.4f}   {ledger[i] / scale - 1:+.4e}   {drift[i] / scale:+.3e}   "
              + " ".join(f"{x / scale:+.3e}" for x in [R[i]] + [parts[k][i] for k in JACO_PARTS] + [rest[i]])
              + f"   {legacy[i] / legacy[0] - 1:+.4e}")
    print_thermal_history(flags, n_h, star, lte)

    assert c["jaco_n_solves"][-1] > 0, "no jaco solve was tallied: the check is not exercised"
    assert abs(band_residual[-1]) < TOL_BAND_LEDGER, f"band ledger does not close: residual {band_residual[-1]:+.3e}"
    if star:
        assert abs(inj[-1] / o["lum_injected"][-1] - 1) < TOL_INJECTION, "injected energy is not c_tilde/c L dt"
    else:  # with the star the HII-region model may also heat the gas
        assert abs(gas_outside[-1]) < TOL_JACO_SOLVE * scale, f"gas energy changed outside the jaco step: {gas_outside[-1]:+.4g}"
    assert np.all(err_abs <= bound), \
        f"the solves change the ledger by Delta_t (R + PdV) to {np.max(err_abs / np.maximum(bound, 1e-300)):.3g}x the solver's bound"
    assert abs(c["jaco_ledger_err"][-1]) < TOL_JACO_SOLVE * scale, \
        f"the solves change the ledger by Delta_t (R + PdV) to {c['jaco_ledger_err'][-1] / scale:+.3e} of it"
    assert np.max(np.abs(drift)) < TOL_JACO_SOLVE * scale, \
        f"the box's ledger departs from the model's rate by {np.max(np.abs(drift)) / scale:.3e} of it"
    assert np.max(np.abs(limiter)) < TOL_KICK_LIMITER * scale, \
        f"the kick's positivity clamp created {np.max(np.abs(limiter)) / scale:+.3e} of the ledger"
    for k in JACO_PARTS:
        if k not in JACO_MODELS[model]["documented"]:
            assert abs(parts[k][-1]) < TOL_JACO_UNDOCUMENTED * scale, f"{model}: undocumented {k} part {parts[k][-1] / scale:+.3e}"
    assert abs(rest[-1]) < TOL_JACO_UNDOCUMENTED * scale, f"{model}: undocumented ledger rate {rest[-1] / scale:+.3e} of it"


def check_molecular(flags, n_h, star):
    rows = re.findall(r"EOS_GAMMA_PROBE t=(\S+) updates=(\d+) above_1e-12=(\d+) max_rel=(\S+) gamma_range=\[(\S+), (\S+)\]",
                      open(f"{output_dir(flags, n_h, star)}/test_{TEST_NAME}.out").read())
    a = np.array(rows, dtype=float)
    t, max_rel, g_lo, g_hi = a[:, 0], a[:, 3], a[:, 4].min(), a[:, 5].max()
    print(f"\n{VARIANTS[flags]}, n_H = {n_h:g}: {a[:, 1].sum():.0f} EOS updates, gamma in [{g_lo:.6f}, {g_hi:.6f}]; "
          f"max |SoundSpeed^2 rho / (gamma P) - 1| = {max_rel.max():.3e} (at t = {t[np.argmax(max_rel)]:.4g}), "
          f"{a[:, 2].sum():.0f} updates above 1e-12")
    assert len(rows) > 0, "no EOS_GAMMA_PROBE output: the check is not exercised"
    assert g_hi - g_lo > 0.05, f"the adiabatic index only spans [{g_lo:.4f}, {g_hi:.4f}]: the check is not exercised"
    assert max_rel.max() < TOL_GAMMA_PROBE, f"sound speed inconsistent with the refreshed adiabatic index: {max_rel.max():.3e}"


def case(flags, n_h, *marks, star=False, lte=None):
    return pytest.param(flags, n_h, star, lte, id=f"{VARIANTS[flags]}{setup_tag(star, lte)}-n{tag(n_h)}", marks=marks)


# grouped by flag set, so each set is built once
CASES = ([case(PLAIN_RT, n) for n in (1e3, 1e4, 1e5, 1e7)]
         + [case(PLAIN_COOLING, n) for n in (1e3, 1e5, 1e7)]
         + [case(POINT_SOURCE, 1e3), case(ILIEV, 1e3)]
         + [case(STARFORGE, n, star=s) for n in (1e3, 1e7) for s in (False, True)]
         + [case(STARFORGE, n) for n in (1e8, 1e9)] + [case(STARFORGE, N_LTE, lte=t) for t in T_LTE]
         + [case(MOLECULAR, 1e4)]
         + [case(jaco_flags(m), n, star=s) for m in JACO_MODELS for n in JACO_DENSITIES for s in (False, True)]
         + [case(jaco_flags(m), N_LTE, lte=t) for m in JACO_MODELS for t in T_LTE])


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=8),))
@pytest.mark.parametrize("flags,n_h,star,lte", CASES)
def test_rt_closed_box(flags, n_h, star, lte, num_mpi_ranks):
    run(flags, n_h, star, num_mpi_ranks, lte)
    if is_molecular(flags):
        check_molecular(flags, n_h, star)
    elif jaco_model(flags):
        check_jaco(flags, n_h, star, lte)
    elif is_starforge(flags):
        check_starforge(flags, n_h, star, lte)
    else:
        (check_source if is_source(flags) else check_bare)(flags, n_h, star)


@pytest.mark.parametrize("model,t_lte", [pytest.param(m, t, id=f"{m}-lte{t:g}") for m, v in JACO_MODELS.items()
                                         if v["reproduces_legacy"] for t in T_LTE])
def test_rt_closed_box_gas_share_vs_legacy(model, t_lte):
    """A jaco model that reproduces GIZMO's coupling must drain the IR band into the gas at GIZMO's rate: in the hot,
    IR-thick LTE box the drain coefficient (drain_coefficient) of the jaco run over the STARFORGE run's, within
    TOL_DRAIN_COEFFICIENT. Uses both runs' outputs, so the LTE cases of test_rt_closed_box must have run first."""
    legacy, jaco = (STARFORGE, N_LTE, False, t_lte), (jaco_flags(model), N_LTE, False, t_lte)
    for f in (legacy, jaco):
        if not (output_dir(*f) / "rt_ir_diag.txt").is_file():
            pytest.skip(f"{output_dir(*f)} missing: run the LTE cases of test_rt_closed_box first")
    k_legacy, k_jaco = drain_coefficient(*legacy), drain_coefficient(*jaco)
    a = 1 - 0.5 / (1 + 725.0**2 / (1 + t_lte**2))
    print(f"\n{model}, LTE at {t_lte:g} K: IR gas-absorption drain coefficient {k_jaco:.6g} vs GIZMO {k_legacy:.6g} per code "
          f"time, ratio {k_jaco / k_legacy:.5f} (1/A(T_rad) = {1 / a:.5f})")
    assert abs(k_jaco / k_legacy - 1) < TOL_DRAIN_COEFFICIENT, f"drain coefficient {k_jaco / k_legacy:.4f}x GIZMO's"
