"""STL-356: the CR share f_CR E of a Stella SN or fast wind reaches the CR fluid in full (GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE):
the reduced-speed factor that inject_cosmic_rays applies at injection is divided out at the Stella deposit; transport and
losses keep it.

Setup of the audit's measurement (row STL-356, job 243486): the cloud configuration with the CR fluid (Config_stella_cloud.sh
without CR_BACKGROUND_IONRATE_ATTENUATED, with COSMIC_RAY_FLUID and CRFLUID_DIFFUSION_MODEL=7) plus the isolated-feedback and
energy-trace hooks; uniform molecular box, n_H = 100, 10 K, 1 Msun cells, B = 2 uG, one 12 Msun star, forced SN at 2.1 kyr.
MaxStellarTimestep is 100 yr here: with the production 1e4 yr (1.0e-5 code) the star's step exceeds the whole 5e-6 run, so
the star is active only at t = 0 and at TimeMax and the forced SN fired at 5.0 kyr in job 243486.
Quantities: every feedback call that asks for CR energy (CRSTEP: f_CR E_SN of the SN, f_CR 1/2 dp^2/dM of fast winds) against
the CR energy that arrived in the cells in that call. Bars (fixed before the run): |arrived / asked - 1| <= 1e-10 for the SN and
for every wind call; no exact-stage energy finding and no mass finding in the traces (booked = deposited, thermal + kinetic + CR).
Expected without the flag: the SN's ratio 0.075 (job 243486)."""

import os
from os.path import join

import pytest

from audit_round1_common import REPO, TEST_DIR, binary, kv, make_box, run
from gizmo.conservation_trace import keep_and_check_traces

BAR = 1e-10
CR_DROP = ("CR_BACKGROUND_IONRATE_ATTENUATED",)
CR_ADD = ("COSMIC_RAY_FLUID", "CRFLUID_DIFFUSION_MODEL=7", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE")
PARAMS = {"InitCondFile": "box_ics", "BoxSize": 0.02, "TimeMax": 5.0e-6, "TimeBetSnapshot": 5.0e-7, "TimeBetStatistics": 1.0e-7,
          "StellarTablesFile": "gizmo/galaxy_sf/stellar_tables/stellar_tables_unified.hdf5", "BiniZ": 2.0e-6, "CosmicRayIonRate": None,
          "CosmicRay_SNeFraction": 0.1, "CosmicRayDiffusionCoeff": 690, "CosmicRayInitialAmbient_eVcm3": 1.0, "CosmicRayInitialCloud_eVcm3": 0.1,
          "CosmicRayInitialCloudThreshold_nH": 10.0, "TestStarMass": 12.0, "TestStarMass2": 0, "TestStarMass3": 0, "TestNumStars": 1,
          "TestSNTime0": 2.1e-6, "MaxStellarTimestep": 100.0}


def check(arm):
    exe = binary("cr_" + arm, base="Config_stella_cloud.sh", drop=CR_DROP, add=CR_ADD + (("GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE",) if arm == "on" else ()))
    name = "cr_" + arm
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), n_H=100.0, box_pc=20.0, T=10.0, tracabund=(0.4999, 1e-4, 1.4446454e-4))
    text, out = run(name, exe, PARAMS, nranks=16, base=join(REPO, "stella_cloud.params"))[1:]
    sn = [l for l in text.splitlines() if l.startswith("RESOLVEDISM SN:")]
    det = kv(text, "FBTL_DET")
    steps = [(float(r["t"]), float(r["asked"]), float(r["got"])) for r in kv(text, "CRSTEP") if float(r["asked"]) > 0]
    assert len(sn) == 1, f"expected one SN, got {len(sn)}"
    big = max(steps, key=lambda s: s[1])  # the SN asks 1e50 erg, a wind call far less
    winds = [s for s in steps if s is not big]
    worst_w = max((abs(g / a - 1) for _, a, g in winds), default=0.0)
    rep = keep_and_check_traces(out, out)
    bad = [f for f in rep.findings if f["kind"] in ("energy_stage", "mass")]
    print(f"CR share ({arm}): SN at t={big[0]:.6e} (asked TestSNTime0 2.1e-6; {[l.split('(t=')[-1] for l in text.splitlines() if l.startswith('FBTL_DET')]}): "
          f"asked {big[1]:.10e} erg, arrived {big[2]:.10e} erg, ratio {big[2] / big[1]:.10f}; {len(winds)} wind calls, worst |ratio - 1| {worst_w:.3e}; "
          f"trace findings (energy_stage, mass): {len(bad)}")
    assert big[0] <= 2.1e-6 + 2.0e-7, "the forced SN did not fire within two star steps of TestSNTime0"
    assert abs(big[2] / big[1] - 1) <= BAR
    assert worst_w <= BAR
    assert not bad, rep.summary()


def test_cr_full_share_flag_on():
    check("on")


@pytest.mark.xfail(strict=True, reason="STL-356 without the flag: the reduced-speed factor cuts the SN's CR share to 0.075 here")
def test_cr_full_share_flag_off():
    check("off")
