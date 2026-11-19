"""KETJU star collision with the energy step trace (stella_dev b1359a14d / d42b1bc56, ported): the collision energy MSTAR reports and
the members' unsoftened vs tree-softened pair potential over each region step are integration bookings, so the time-integration
residual closes through the merger (the checker's 1e-2 of the first sync's |E| + |E_pot|; pure N-body, no mass step trace).
Head-on case of test_ketju_star_collision. Also: the collision and chain-potential bookings are non-zero (the pair merges inside a
region)."""

import subprocess
import sys

from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir
from gizmo.conservation_trace import check_conservation_traces

TEST_NAME = "ketju_star_collision"
TEST_DIR = f"test/{TEST_NAME}"
FLAGS = ("OUTPUT_ENERGY_STEP_TRACE",)


def test_ketju_collision_trace_closes():
    subprocess.run([sys.executable, "make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME, FLAGS)
    build_and_run_test(TEST_NAME, 1, 1, FLAGS, param_overrides={"InitCondFile": "ketju_star_collision_ics"})
    out = variant_output_dir(TEST_NAME, FLAGS)
    rows = [dict(t.split("=", 1) for t in l.split()[1:]) for l in open(f"{out}/energy_step_trace.txt") if l.startswith("SYNC")]
    assert len(rows) > 2, "no full syncs"
    rep = check_conservation_traces(out, require_mass=False)
    last = rows[-1]
    print(rep.summary())
    print(f"ketju_collision {float(last['b_ketju_collision']):+.6e} ketju_chain_pot {float(last['b_ketju_chain_pot']):+.6e} int_res {float(last['int_res']):+.6e} E_scale {float(last['E_scale']):.6e}")
    assert float(last["b_ketju_collision"]) != 0 and float(last["b_ketju_chain_pot"]) != 0
    assert rep.ok, rep.summary()
