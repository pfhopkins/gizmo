"""Guards of audit round 1: STL-417, 423, 427, 432, 446 (431b: build only, see the README).

Compile-time (a translation unit that includes allvars.h, compiled with the build's own flags on a scratch copy of the tree):
- STL-423: GALSF_RESOLVEDISM_BH_PROMOTION defines SINK_PARTICLES and SINK_GRAVCAPTURE_GAS at top level, before the checks that
  read them: the unit sees both defined (it compiles), and STARS_HERMITE_DIRECT_GRAVITY must hit its "not combinable with
  sinks" #error. Before: the unit compiles (the old block defined them at the end), Hermite compiles.
  (The first version required MAINTAIN_TREE_IN_REARRANGE; that rule sits inside SINGLE_STAR_SINK_DYNAMICS and never applies
  to a Stella build, so the case was wrong. It is replaced by the sinks-defined case, which catches the block landing inside
  a conditional region, as it did in the first fix, inside FIRE_PHYSICS_DEFAULTS.)
- STL-427: KETJU_REGULARIZATION + KETJU_PN_REMNANT_TAG without a merge flag, and BH promotion without KETJU_MERGE_BH, stop
  at the preprocessor with an STL-427 message. Before: no such message.
Runtime (one start each, stopped at the first sync point):
- STL-417: the cloud build (SGCHEM_KEEP_IC_ABUNDANCES) refuses an IC without SGChemTracAbund (endrun 8472) or with the wrong
  width (8473), and starts with the right one. Before: it starts with zero abundances.
- STL-432: the tall-box build refuses units other than kpc / 1e10 Msun / km/s (endrun 8862). Before: it starts.
- STL-446: the tall-box build refuses a ShieldingLength of half the x/y side or more (endrun 8863). Before: it starts.
Bar: the expected message and a non-zero exit where the guard must stop the run; exit 0 and no guard message otherwise."""

import os
import shlex
import shutil
import subprocess
from os.path import join

import h5py
import numpy as np
import pytest

from audit_round1_common import REPO, TEST_DIR, binary, config_text, make_box, run

TU = """#include "declarations/allvars.h"
#ifdef GUARD_EXPECT_SINKS
#if !defined(SINK_PARTICLES) || !defined(SINK_GRAVCAPTURE_GAS)
#error "GUARD-UNIT: GALSF_RESOLVEDISM_BH_PROMOTION did not define SINK_PARTICLES and SINK_GRAVCAPTURE_GAS"
#endif
#endif
int main() {return 0;}
"""
COMPILE_CASES = {  # name: (added flags, unit define, message that must appear (None: must compile))
    "stl423_sinks_defined": (("GALSF_RESOLVEDISM_BH_PROMOTION",), "GUARD_EXPECT_SINKS", None),
    "stl423_hermite": (("GALSF_RESOLVEDISM_BH_PROMOTION", "STARS_HERMITE_DIRECT_GRAVITY"), None, "not combinable with sinks"),
    "stl427_pn_tag": (("KETJU_REGULARIZATION", "KETJU_PN_REMNANT_TAG"), None, "STL-427"),
    "stl427_bh_promotion": (("KETJU_REGULARIZATION", "KETJU_MERGE_STARS", "GALSF_RESOLVEDISM_BH_PROMOTION"), None, "needs KETJU_MERGE_BH"),
}


def compile_unit(tree, name, flags, define):
    """Config_stella_default.sh + flags in a scratch copy of tree; returns (exit code, compiler output)."""
    d = join(TEST_DIR, "guard_tree_" + name + "_" + os.path.basename(os.path.normpath(tree)))
    if not os.path.isdir(d):
        shutil.copytree(tree, d, ignore=shutil.ignore_patterns(".git", "*.o", "test", "GIZMO*", "output*"), symlinks=True)
    open(join(d, "Config.sh"), "w").write(config_text(add=flags) if tree == REPO else open(join(tree, "Config_stella_default.sh")).read() + "\n" + "\n".join(flags) + "\n")
    subprocess.run(["make", "-f", "config-makefile"], cwd=d, env=dict(os.environ, CONFIG="Config.sh", PERL="/usr/bin/perl"), capture_output=True)
    dry = subprocess.run(["make", "-n", "-B", "core/main.o"], capture_output=True, text=True, cwd=d).stdout.splitlines()
    line = [l for l in dry if " -c core/main.cc" in l]
    if not line:
        return 1, "no compile line for core/main.cc (make -n):\n" + "\n".join(dry[-20:])
    open(join(d, "guard_unit.cc"), "w").write(TU)
    cc = shlex.split(line[-1])
    cc[cc.index("core/main.cc")] = "guard_unit.cc"
    i = cc.index("-o"); del cc[i:i + 2]
    cc += ["-fsyntax-only"] + (["-D" + define] if define else [])
    r = subprocess.run(cc, capture_output=True, text=True, cwd=d)
    return r.returncode, r.stdout + r.stderr


def compile_verdict(tree, name):
    flags, define, msg = COMPILE_CASES[name]
    rc, out = compile_unit(tree, name, flags, define)
    ok = (rc == 0) if msg is None else (rc != 0 and msg in out)
    print(f"{name}: exit {rc}; expected {'a clean compile' if msg is None else repr(msg)}: {'yes' if ok else 'no'}\n" + out[-1500:])
    return ok


@pytest.mark.parametrize("name", list(COMPILE_CASES))
def test_compile_guards(name):
    assert compile_verdict(os.environ.get("STELLA_AUDIT_GUARD_TREE", REPO), name)


def guard_run(kind, base, params_base, params, ic_kw, name, expect):
    """One start; expect: an endrun message (the run must stop with it) or None (the run must reach TimeMax)."""
    exe = binary(kind, base=base)
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "guard_ics.hdf5"), double=False, **ic_kw)  # cloud and tall-box builds read single-precision ICs
    rc, text, out = run(name, exe, {"InitCondFile": "guard_ics", **params}, nranks=2, base=params_base, expect_ok=False)
    stopped = rc != 0 and "Simulation ends." not in text
    print(f"{name}: exit {rc}, reached TimeMax {'Simulation ends.' in text}; guard message {expect!r} present: {expect in text if expect else '-'}")
    if expect is None:
        assert rc == 0 and "Simulation ends." in text, text[-2000:]
    else:
        assert stopped and expect in text, text[-2000:]


CLOUD = {"TimeMax": 1.0e-8, "TimeBetSnapshot": 1.0e-8, "BoxSize": 0.016, "StellarTablesFile": "gizmo/galaxy_sf/stellar_tables/stellar_tables_unified.hdf5"}
TALL = {"TimeMax": 1.0e-8, "TimeBetSnapshot": 1.0e-8, "StellarTablesFile": "gizmo/galaxy_sf/stellar_tables/stellar_tables_unified.hdf5"}
MOL = (0.4999, 1e-4, 1.4446454e-4)


def test_stl417_missing_tracabund():
    guard_run("cloud_off", "Config_stella_cloud.sh", join(REPO, "stella_cloud.params"), CLOUD, dict(n_H=100.0, T=10.0), "g417_missing",
              "SGCHEM_KEEP_IC_ABUNDANCES: the IC has no")


def test_stl417_wrong_width():
    guard_run("cloud_off", "Config_stella_cloud.sh", join(REPO, "stella_cloud.params"), CLOUD, dict(n_H=100.0, T=10.0, tracabund=MOL[:2]), "g417_width",
              "SGCHEM_KEEP_IC_ABUNDANCES: IC dataset")


def test_stl417_good_ic_starts():
    guard_run("cloud_off", "Config_stella_cloud.sh", join(REPO, "stella_cloud.params"), CLOUD, dict(n_H=100.0, T=10.0, tracabund=MOL), "g417_good", None)


def tall_ic():
    return dict(n_H=10.0, box_pc=20.0, m_cell=1.0, T=1.0e4)  # 1 Msun cells in a 20 pc cube at a corner of the 1 kpc box: a start-up check


def test_stl432_units():
    guard_run("tallbox_off", "Config_stella_tallbox.sh", join(REPO, "stella_tallbox.params"), {**TALL, "UnitMass_in_g": 1.989e33}, tall_ic(), "g432_units",
              "GALSF_RESOLVEDISM_TALLBOX: the analytic potential needs")


def test_stl446_shielding_length():
    guard_run("tallbox_off", "Config_stella_tallbox.sh", join(REPO, "stella_tallbox.params"), {**TALL, "ShieldingLength": 0.6}, tall_ic(), "g446_shield",
              "reaches half the shortest periodic box side")


def test_tallbox_production_starts():
    guard_run("tallbox_off", "Config_stella_tallbox.sh", join(REPO, "stella_tallbox.params"), TALL, tall_ic(), "g_tall_good", None)
