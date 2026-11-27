"""STL-403: the ejecta carry the star's birth composition (GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION, net-yield form).

A grid star (Z = 0.014) born from gas whose Fe is doubled (O lowered by the same mass fraction, so slot 0 and the table
metallicity stay 0.014) dies naturally: TestStarInitAgeYr puts it 20 kyr before its table death, the wind closure then
returns its whole lifetime wind, the death flush the rest, and the SN (20 Msun) or AGB (2 Msun) the death ejecta.
Quantity: Fe delivered to the gas = gas Fe at the end - gas Fe of the edited start + X_Fe M_birth (the star took M_birth of
uniform gas), from double-precision snapshots. Reference: the table's Fe (wind cumulative at the grid star's own death +
death fraction x M_pre, read here from the HDF5 tables) + (X_star,Fe - X_init,Fe(Z)) x (M_birth - M_remnant).
Bar (fixed before the run): |Fe_delivered / reference - 1| <= 1e-6. Expected with the flag off (the table Fe only):
-17.6 % (20 Msun CCSN) and -49 % (2 Msun AGB), audit checker 06, section 3."""

import os
import shutil
from os.path import join

import h5py
import pytest

from audit_round1_common import HOOKS, ROUND1_FLAGS, TEST_DIR, Tables, UNIT_T_YR, binary, kv, last_snapshot, make_box, run
from gizmo.conservation_trace import keep_and_check_traces

BAR = 1e-6
SLOT_FE, SLOT_O = 22 + 1, 4 + 1  # MET_OF(ELEM_Fe), MET_OF(ELEM_O)
CASES = {"ccsn20": (20.0, -0.176), "agb2": (2.0, -0.49)}
ARMS = {"off": HOOKS, "on": HOOKS + ROUND1_FLAGS}


def birth_run(case, arm):
    M = CASES[case][0]
    exe = binary("round1_" + arm, add=ARMS[arm])
    name = f"birth_{case}_{arm}"
    d = join(TEST_DIR, "run_" + name)
    os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"))
    run(name, exe, {"InitCondFile": "box_ics", "TestStarMass": 0, "TimeMax": 1.0e-9, "TimeBetSnapshot": 1.0e-9}, nranks=1)
    start = join(d, "birth_000.hdf5")
    shutil.copy(join(d, "output", "snapshot_000.hdf5"), start)
    with h5py.File(start, "r+") as f:  # double Fe, lower O by the same fraction: slot 0 (Z = 0.014) unchanged
        X = f["PartType0/Metallicity"][:]
        dX = X[:, SLOT_FE].copy(); X[:, SLOT_FE] += dX; X[:, SLOT_O] -= dX
        f["PartType0/Metallicity"][...] = X
        E = f["PartType0/ElementAbundance"][:]  # read after Metallicity and written over it: edited the same way
        E[:, SLOT_FE - 1] += dX; E[:, SLOT_O - 1] -= dX
        f["PartType0/ElementAbundance"][...] = E
        X0_Fe = float(X[0, SLOT_FE]); Z0 = float(X[0, 0])
        fe_start = float((f["PartType0/Masses"][:] * X[:, SLOT_FE]).sum())
    T = Tables()
    t_death, _ = T.lifetime(M, Z0)
    t_end = 6.0e4
    text = run(name, exe, {"InitCondFile": "birth_000", "TestStarMass": M, "TestNumStars": 1, "TestStarInitAgeYr": t_death - 2.0e4,
                           "TestSNTime0": 0, "TimeMax": t_end / UNIT_T_YR, "TimeBetSnapshot": t_end / UNIT_T_YR}, nranks=2, restart_flag=2)[1]
    death = kv(text, "STARDEATH")
    assert len(death) == 1, f"{name}: expected one natural death, got {len(death)}"
    M_birth, M_pre = float(death[0]["M_birth"]), float(death[0]["M_pre"])
    snap = last_snapshot(join(d, "output"))
    with h5py.File(snap, "r") as f:
        fe_end = float((f["PartType0/Masses"][:] * f["PartType0/Metallicity"][:, SLOT_FE]).sum())
        M_rem = float(f["PartType4/Masses"][:].sum()) * 1e10 if "PartType4" in f else 0.0
    fe_del = (fe_end - fe_start) * 1e10 + X0_Fe * M_birth
    iz, im = T.node(M, Z0)
    fe_table = T.wind_total_at_own_death(iz, im)[22] + T.death_fraction(iz, im)[22] * M_pre
    ref = fe_table + (X0_Fe - T.X_init[iz, 22]) * (M_birth - M_rem)  # X0_Fe: the star's (edited) birth Fe
    clips = kv(text, "RESOLVEDISM EJECTA BIRTH CLIP:")
    print(f"{name}: M_birth {M_birth:.10f} M_pre {M_pre:.10f} M_rem {M_rem:.10f}; Fe delivered {fe_del:.10e}, table {fe_table:.10e}, "
          f"reference {ref:.10e}: delivered/reference - 1 = {fe_del / ref - 1:+.3e} (table/reference - 1 = {fe_table / ref - 1:+.3e}); "
          f"clips {len(clips)}; snapshot {snap}")
    return fe_del, fe_table, ref, clips, join(d, "output")


@pytest.mark.parametrize("case", list(CASES))
def test_birth_composition_flag_on(case):
    fe_del, fe_table, ref, clips, out = birth_run(case, "on")
    assert not clips, "a slot was clipped: the reference does not include the clip's rescale"
    rep = keep_and_check_traces(out, out)
    bad = [f for f in rep.findings if f["kind"] in ("energy_stage", "mass", "element_ledger", "metal_ledger")]
    assert not bad, rep.summary()
    assert abs(fe_del / ref - 1) <= BAR


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.xfail(strict=True, reason="STL-403 without the flag: ejecta take the table's X_init (expected -17.6 % / -49 % Fe)")
def test_birth_composition_flag_off(case):
    fe_del, fe_table, ref, clips, out = birth_run(case, "off")
    assert abs(fe_del / ref - 1) <= BAR
