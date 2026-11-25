"""Static check: the Config.sh of tests that run a production (or another test's) configuration are copies, so they must not drift.
Each copy's non-comment lines must equal its parent's plus exactly the additions the test states (as sets: order and comment lines
do not matter)."""
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COPIES = {
    "ketju_pm": ("test/ketju_tide/Config.sh", {"PMGRID=128", "BOX_PERIODIC"}),
    "ketju_mass_loss": ("test/ketju_snapshot_midstep/Config.sh", {"TEST_KETJU_HALVE_STAR1_MASS_AFTER_GRAVITY=1.0e-5"}),
    "ketju_expand_binary": ("test/ketju_snapshot_midstep/Config.sh", {"OUTPUT_ENERGY_STEP_TRACE"}),
    "radp_starside_aniso": ("test/radp_starside/Config.sh", set()),
    "treecol_shielding_bounded": ("Config_stella_default.sh", set()),
    "sgchem_cooling_path_production": ("Config_stella_default.sh", set()),
    "sgchem_g0_floor": ("Config_stella_default.sh", set()),
    "stella_merger_remnant": ("Config_stella_default.sh", {"KETJU_REGULARIZATION", "KETJU_MERGE_STARS"}),
    "sgchem_unified_fractions": ("Config_stella_default.sh", {"SGCHEM_ELECTRONS_AND_HYDROGEN_FRACTION_FROM_NETWORK"}),
    "sgchem_cr_floor": ("Config_stella_cloud.sh", set()),
    "sgchem_cr_zero_before_first_star": ("Config_stella_default.sh", {"BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "CR_SCALE_WITH_G0",
                                                                      "SGCHEM_COSMIC_RAYS_ZERO_BEFORE_FIRST_STAR"}),
    "tallbox_gravity_cr": ("Config_stella_tallbox.sh", {"OUTPUT_ENERGY_STEP_TRACE", "ENERGY_BUDGET_DIAGNOSTIC"}),
}


def flags(path):
    out = set()
    for line in open(os.path.join(REPO, path)):
        s = line.split("%")[0].split("#")[0].strip()
        if s:
            out.add(s)
    return out


@pytest.mark.parametrize("test", sorted(COPIES))
def test_config_copy_matches_parent(test):
    parent, added = COPIES[test]
    copy, ref = flags(f"test/{test}/Config.sh"), flags(parent) | added
    assert copy == ref, f"{test}: extra {sorted(copy - ref)}, missing {sorted(ref - copy)}"
