"""conservative_psi_cleaning: constrained hyperbolic/parabolic div B cleaning (Tricco & Price 2012; Tricco, Price & Bate 2016, TPB16) in place of Dedner:
DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH evolves PsiZ = (psi/c_h) sqrt(V), so that E_psi = PsiZ^2/2 survives volume changes and changes of c_h, and the pair of
operators moves energy between B and psi only. An idealised numerical test (cleaning alone, no chemistry or stars to involve): a 24^3 box of static gas at
P = 100, rho = 1, B = (f(r), 0, 0) with f = (r/r0)^8 - 2 (r/r0)^4 + 1 inside r0 = 0.25 (TPB16 sec. 3.1), which has div B != 0; plasma beta >~ 200, so the gas
stays nearly static, c_h ~ c_s ~ 13, t = 0.2, 2 ranks.
E_B = sum V B^2/2, E_psi = sum V psi^2/(2 c_h^2) with c_h^2 = c_s^2 + v_A^2, from the snapshot.

Bars, written before the first run:
  test_damping_cleans (SIGMA default 1): the rms of h div B / |B| falls by at least 2 between t = 0.02 and t = 0.2.
  test_damped_energy_is_heat (SIGMA 1): total energy (heat, kinetic, B, psi) stays within 2 % of the initial E_B.
  test_no_damping_heat (_NO_DAMPING_HEAT): the same total falls, by at least 10 % of the initial E_B or by at least half of what E_B + E_psi lost.
  test_pair_conserves[pair] (SIGMA 0), [symmetric] (SIGMA 0, _SYMMETRIC_DIVB), [alternating] (SIGMA 0, TEST_DIVBCLEANING_ALTERNATING_CLEANING_SPEED:
    c_h doubles every other 0.05, TPB16 sec. 3.2; the output psi there carries the doubled c_h, so E_psi divides by 4 c_h^2). Bar from the
    budget, set before the run (reports/test_bars_2026_11_13.md): the pair moves energy between E_B and E_psi only (exact in rate form); the MHD
    fluxes book their E_B change as heat; so E_th + E_kin + E_B + E_psi changes only by the ledger's unbalanced terms (kick floor, Powell, closure
    limiter) and the E_B the u equation does not book. The rest R is time-integration error: |R| at CourantFac 0.1 over |R| at 0.05 gives a
    measured order >= 0.75 (or |R| below the roundoff floor 1e4 x 2.2e-16 x E_tot); and the pair moves energy: max E_psi >= E_div/4, E_div = 0.32
    E_B0 the curl-free part of the initial field (equipartition E_div/2, factor 2 for the damping of the cleaning waves).
    History: first E_B + E_psi within 1 %, then total within 1 % with E_psi >= 1 %, both set after runs; replaced by this one.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "conservative_psi_cleaning"
TEST_DIR = Path(__file__).parent
N, P0, GAMMA = 24, 100.0, 5.0 / 3.0
LEDGER = ("OUTPUT_MASS_METAL_ENERGY_LEDGER", "TEST_MHD_ENERGY_LEDGER_OUTPUT")  # booking only
VARIANTS = {
    "damped": (),
    "no_heat": ("DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH_NO_DAMPING_HEAT",),
    "pair": ("DIVBCLEANING_CONSERVATIVE_SIGMA=0",) + LEDGER,
    "symmetric": ("DIVBCLEANING_CONSERVATIVE_SIGMA=0", "DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH_SYMMETRIC_DIVB") + LEDGER,
    "alternating": ("DIVBCLEANING_CONSERVATIVE_SIGMA=0", "TEST_DIVBCLEANING_ALTERNATING_CLEANING_SPEED") + LEDGER,
}
PAIR = ("pair", "symmetric", "alternating")  # also run at half the Courant factor
SOURCES = ("E_kick_floor_erg", "E_mhd_powell_erg", "E_mhd_dedner_face_erg", "E_mhd_dedner_correction_erg", "E_mhd_closure_erg")
E_DIV = 0.32  # curl-free share of the initial E_B (FFT of the IC field, 24^3 and 192^3 agree)
LEDGERS = {}


def write_ic(fname):
    g = (np.arange(N) + 0.5) / N
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    r = np.sqrt(((x - 0.5) ** 2).sum(1)) / 0.25
    b = np.zeros_like(x)
    b[:, 0] = np.where(r < 1, r ** 8 - 2 * r ** 4 + 1, 0.0)
    n = len(x)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        np_ = np.array([n, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = np_; h.attrs["NumPart_Total"] = np_; h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6); h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = h.attrs["Flag_Cooling"] = h.attrs["Flag_StellarAge"] = h.attrs["Flag_Metals"] = h.attrs["Flag_Feedback"] = 0; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = x; p["Velocities"] = np.zeros((n, 3)); p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, 1.0 / n); p["InternalEnergy"] = np.full(n, P0 / (GAMMA - 1)); p["MagneticField"] = b


def launch(tag, courant=None):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    write_ic(d / "ics.hdf5")
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    if courant:
        p = re.sub(r"(?m)^CourantFac\s+\S+", f"CourantFac                         {courant}", p)
        p = re.sub(r"(?m)^TimeBetStatistics\s+\S+", "TimeBetStatistics                  0.02", p)
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params"], cwd=d, stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d


def ledger(d):
    fn = d / "output" / "ledger.txt"
    with open(fn) as f:
        head = f.readline().split()[1:]
    names = head[:next(k for k, w in enumerate(head) if w.startswith("("))]
    a = np.loadtxt(fn, ndmin=2)
    return {"_t": a[:, 0], **{n: a[:, k] for k, n in enumerate(names)}}


@pytest.fixture(scope="module")
def runs():
    out = {}
    for tag, flags in VARIANTS.items():
        build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
        d = launch(tag, 0.1 if tag in PAIR else None)
        out[tag] = [books(s, tag == "alternating") for s in sorted((d / "output").glob("snapshot_*.hdf5"))]
        if tag in PAIR:
            LEDGERS[tag] = ledger(d)
            LEDGERS[tag + "_half"] = ledger(launch(tag + "_half", 0.05))
    return out


def books(fname, alternating=False):
    with h5py.File(fname, "r") as F:
        t = F["Header"].attrs["Time"]
        g = F["PartType0"]
        m, rho, u, v, B, psi = (g[k][()] for k in ("Masses", "Density", "InternalEnergy", "Velocities", "MagneticField", "DivBcleaningFunctionPhi"))
        divB = g["DivergenceOfMagneticField"][()]
    V, b2 = m / rho, (B ** 2).sum(1)
    c2 = GAMMA * (GAMMA - 1) * u + b2 / rho
    if alternating and int(t / 0.05) % 2:
        c2 = 4 * c2  # the code's c_h is doubled in this window, and the output psi with it
    h = V ** (1.0 / 3.0)
    return dict(EB=(0.5 * V * b2).sum(), Epsi=(0.5 * V * psi ** 2 / c2).sum(), Eth=(m * u).sum(), Ek=0.5 * (m * (v ** 2).sum(1)).sum(),
                divb=np.sqrt((V * (h * divB) ** 2).sum() / max((V * b2).sum(), 1e-300)))


def total(b):
    return b["Eth"] + b["Ek"] + b["EB"] + b["Epsi"]


def test_damping_cleans(runs):
    r = runs["damped"]
    assert r[-1]["divb"] < 0.5 * r[1]["divb"], (r[1]["divb"], r[-1]["divb"])


def test_damped_energy_is_heat(runs):
    r = runs["damped"]
    dev = max(abs(total(b) - total(r[0])) for b in r) / r[0]["EB"]
    print("total energy deviation / E_B0:", dev)
    assert dev < 0.02


def test_no_damping_heat(runs):
    r = runs["no_heat"]
    lost_total = total(r[0]) - total(r[-1])
    lost_mag = r[0]["EB"] + r[0]["Epsi"] - r[-1]["EB"] - r[-1]["Epsi"]
    print("lost total", lost_total, "lost magnetic+psi", lost_mag, "E_B0", r[0]["EB"])
    assert lost_total > min(0.1 * r[0]["EB"], 0.5 * lost_mag)


def rest(L):
    """total-energy change net of the ledger's unbalanced terms and of the E_B the u equation does not book (test/mhd_energy_ledger)"""
    d = lambda n: L[n][-1] - L[n][0]
    dE = sum(d(n) for n in ("E_th_erg", "E_kin_erg", "E_B_erg", "E_psi_erg"))
    unbooked = d("E_B_erg") - (d("E_mhd_B_dVB_booked_erg") - d("E_mhd_half_B2_dV_booked_erg"))
    return dE - sum(d(n) for n in SOURCES) - unbooked, dE


@pytest.mark.parametrize("tag", list(PAIR))
def test_pair_conserves(runs, tag):
    r, L, Lh = runs[tag], LEDGERS[tag], LEDGERS[tag + "_half"]
    assert L["_t"][-1] > 0.199 and abs(L["_t"][-1] - Lh["_t"][-1]) < 1e-6
    (R, dE), (Rh, dEh) = rest(L), rest(Lh)
    EB0, Etot0 = L["E_B_erg"][0], sum(L[n][0] for n in ("E_th_erg", "E_kin_erg", "E_B_erg", "E_psi_erg"))
    floor, order = 1e4 * 2.2e-16 * Etot0, np.log2(abs(R) / abs(Rh)) if Rh != 0 else np.inf
    Epsi = max(b["Epsi"] for b in r) / r[0]["EB"]
    print(f"{tag}: total change / E_B0 {dE / EB0:+.3e} (half step {dEh / EB0:+.3e}); "
          + " ".join(f"{n} {(L[n][-1] - L[n][0]) / EB0:+.3e}" for n in SOURCES)
          + f"; rest / E_B0 {R / EB0:+.3e}, half step {Rh / EB0:+.3e}, order {order:.2f}, floor / E_B0 {floor / EB0:.1e}; max E_psi / E_B0 {Epsi:.3f}")
    assert Epsi >= E_DIV / 4
    assert abs(R) <= floor or order >= 0.75
