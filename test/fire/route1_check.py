#!/usr/bin/env python3
"""ROUTE1 policy gate: routed ONEWAY discovery is a property of the search mode.

POLICY (caller-agnostic): every GX_RD_TIME line with mode=ONEWAY — whatever the
caller — must satisfy:
  oneway_route=routed, route=hier, and zero fellback / geom_unavail / off
and the both-way compare oracle (when enabled) must report routed==broadcast:
no under-route (7704: broadcast ghost missing from routed) and no extra-match
(7722: routed ghost not in broadcast). At least one ONEWAY caller must actually
be exercised, or the run proves nothing.

Scope: this checks ghost-exchange requests (GX_RD_TIME emitters). Other local
neighbor-search APIs (twopoint / group_search / turb_powerspectra / merge_split)
are separate paths and are not covered by this gate.

Usage: route1_check.py <run_dir_or_gizmo.out> [--allow-geom-unavail]
Exit 0 = PASS, 1 = FAIL. Prints per-caller evidence either way.
"""
import sys, re, os
from collections import defaultdict

args = sys.argv[1:]
allow_geom = "--allow-geom-unavail" in args
args = [a for a in args if a != "--allow-geom-unavail"]
p = args[0] if args else "."
path = p if p.endswith(".out") else os.path.join(p, "gizmo.out")

# oneway_route outcome tally per ONEWAY caller (no whitelist — policy applies to all)
outcome = defaultdict(lambda: defaultdict(int))
route_kind = defaultdict(lambda: defaultdict(int))  # hier/flat tally on routed lines
oracle_ok = 0
under_route = 0     # 7704: broadcast has a ghost routed lacks (missing physics)
extra_match = 0     # 7722: routed has a ghost broadcast lacks (predicate/snapshot bug)
true_7704 = 0       # under-route controlled-stop message actually raised
true_7722 = 0       # extra-match controlled-stop message actually raised

for line in open(path, errors="replace"):
    if "GX_RD_TIME" in line and "mode=ONEWAY" in line:
        m = re.search(r'caller=(\w+).*oneway_route=(\w+)', line)
        if m: outcome[m.group(1)][m.group(2)] += 1
        # route=hier is the production constructor; flat is the slow debug
        # constructor and must never appear in a production/default run.
        if m and m.group(2) == "routed":
            rk = re.search(r' route=(\w+)', line)
            if rk: route_kind[m.group(1)][rk.group(1)] += 1
    if "GX_ROUTE_TRANSPORT" in line:
        if "OK routed==broadcast" in line: oracle_ok += 1
        if "UNDER-ROUTE" in line: under_route += 1
        if "EXTRA-MATCH" in line: extra_match += 1
    # true stop = the controlled-stop message text, NOT a number substring match
    if "broadcast ghosts missing from routed set (UNDER-ROUTE)" in line:
        true_7704 += 1
    if "routed set has ghosts broadcast lacks (EXTRA-MATCH" in line:
        true_7722 += 1

print(f"=== ROUTE1 policy gate: {path} ===")
fail = False
if not outcome:
    print("  WARNING: no ONEWAY ghost-exchange lines emitted — routed arm not tested by this run.")
    fail = True
for c in sorted(outcome):
    o = outcome[c]
    routed = o.get("routed", 0); fell = o.get("fellback", 0)
    geom = o.get("geom_unavail", 0); off = o.get("off", 0)
    rk = route_kind.get(c, {})
    hier = rk.get("hier", 0); flat = rk.get("flat", 0)
    geom_bad = (geom > 0 and not allow_geom)
    ok = (routed > 0 and fell == 0 and off == 0 and flat == 0 and not geom_bad)
    fail |= not ok
    why = ""
    if not ok:
        if flat: why = " (flat route observed; production Route1 should be hierarchical)"
        elif off: why = " (oneway_route=off on a ONEWAY line — dispatch policy violated)"
        elif fell: why = " (routed producer fell back to broadcast)"
        elif geom_bad: why = " (geometry unavailable; rerun with --allow-geom-unavail only if understood)"
        elif routed == 0: why = " (no routed installs)"
    print(f"  ROUTE1[{c:14s}] routed={routed} fellback={fell} geom_unavail={geom} off={off}  "
          f"route=hier:{hier}/flat:{flat}  {'PASS' if ok else 'FAIL' + why}")
oracle_bad = (under_route > 0 or extra_match > 0 or true_7704 > 0 or true_7722 > 0)
print(f"  oracle: OK={oracle_ok} under_route(7704)={under_route} extra_match(7722)={extra_match} "
      f"true_7704_stops={true_7704} true_7722_stops={true_7722}  "
      f"{'PASS' if not oracle_bad else 'FAIL'}")
fail |= oracle_bad
print(f"\nROUTE1 VERDICT: {'PASS — every ONEWAY ghost exchange routed hierarchically' if not fail else 'FAIL — see above'}")
sys.exit(1 if fail else 0)
