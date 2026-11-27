"""STL-393: the hydrogen closure after a deposit (X_H = 1 - Z - Y) is booked in the element ledger (no flag: bookkeeping only).

REGRESSION CHECK ONLY, NOT A FAILS-BEFORE TEST. The intended setup (slot 0 raised by 1e-6 against the element slots in a
snapshot, so the closure moves 1e-6 of each receiver into H) cannot be made through a snapshot: with
GALSF_RESOLVEDISM_METALS_INDIVIDUAL the ElementAbundance read (file_io/read_ic.cc) rebuilds slot 0 from the element slots and
closes H, and a snapshot without ElementAbundance reads it as zeros (pure H, Z = 0). The edit below is therefore erased on read;
the run is a 20 Msun SN (forced at 1 kyr) in the 16 pc box, n_H = 10, 1 Msun cells, Z = 0.014.
Bar: no element_ledger, metal_ledger or mass finding at the trace bar 1e-12. A discriminating test needs a driver or a test
hook that sets the offset after the read (open, row of STL-393)."""

import os
import shutil
from os.path import join

import h5py

from audit_round1_common import HOOKS, TEST_DIR, binary, make_box, run
from gizmo.conservation_trace import keep_and_check_traces


def test_h_closure_booked():
    exe = binary("round1_off", add=HOOKS)
    name = "hclose"
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"))
    run(name, exe, {"InitCondFile": "box_ics", "TestStarMass": 0, "TimeMax": 1.0e-9, "TimeBetSnapshot": 1.0e-9}, nranks=1)
    start = join(d, "hclose_000.hdf5")
    shutil.copy(join(d, "output", "snapshot_000.hdf5"), start)
    with h5py.File(start, "r+") as f:
        X = f["PartType0/Metallicity"][:]
        X[:, 0] += 1.0e-6
        f["PartType0/Metallicity"][...] = X
    out = run(name, exe, {"InitCondFile": "hclose_000", "TestStarMass": 20.0, "TestNumStars": 1, "TestSNTime0": 1.0e-6, "TimeMax": 2.0e-6,
                          "TimeBetSnapshot": 2.0e-6}, nranks=2, restart_flag=2)[2]
    rep = keep_and_check_traces(out, out)
    bad = [f for f in rep.findings if f["kind"] in ("mass", "element_ledger", "metal_ledger")]
    print("H closure booking: " + ("no ledger finding" if not bad else "; ".join(f"{f['kind']} {f['stage']} n={f['count']} max|rel|={f['max_rel']:.3e}" for f in bad)))
    assert not bad, rep.summary()
