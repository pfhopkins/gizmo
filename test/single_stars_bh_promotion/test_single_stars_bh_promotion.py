"""Remnants of failed supernovae and pair pulsations become Type-5 sinks (GALSF_SINGLE_STARS_BH_PROMOTION): the supernova test's stars, built with the flag."""
import sys

import h5py

sys.path.insert(0, "test/single_stars_supernovae")
import test_single_stars_supernovae as sn  # noqa: E402

SINK_PARAMS = """
SeedSinkMass 1e-20
SinkAccretionFactor 0
SinkEddingtonFactor 1
SinkFeedbackFactor 0
SinkMaxAccretionRadius 0.1
SinkNgbFactor 1
SinkRadiativeEfficiency 0.1
Softening_Type5 1.0
Softening_Type5_MaxPhysLimit 1.0
"""
FLAGS = ("SINK_PARTICLES", "GALSF_SINGLE_STARS_BH_PROMOTION")


def test_bh_promotion():
    tables = sn.get_stellar_evolution_tables()
    sn.build_gizmo_for_test(sn.TEST_NAME, extra_config_flags=FLAGS)
    template = sn.TEST_DIR / f"{sn.TEST_NAME}.params"
    original = template.read_text()
    template.write_text(original + SINK_PARAMS)
    try:
        ic, snap, rows = sn.run("bh_promotion", tables)
    finally:
        template.write_text(original)
    with h5py.File(snap, "r") as f:
        n5 = f["Header"].attrs["NumPart_Total"][5]
    n_bh = sum(1 for r in rows if int(r[3]) in (3, 4, 6))
    assert n_bh > 0 and n5 == n_bh
