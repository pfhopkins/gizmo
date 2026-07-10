#!/usr/bin/env python3
"""Lever-1 (ONEWAY routed transport) gate check on a Frontera gizmo.out.

STEP 3 (correctness, oracle ON): every density + hydro_oneway request-driven
import must install routed with ZERO fallback, and the both-way oracle must say
routed==broadcast — no under-route (7704: broadcast ghost missing from routed)
and no extra-match (7722: routed ghost not in broadcast). This script makes the
pass/fail MECHANICAL (codex: judge by oneway_route, not wall time).

Usage: route1_check.py <run_dir_or_gizmo.out>
Exit 0 = PASS, 1 = FAIL. Prints the evidence either way.
"""
import sys, re, os
from collections import defaultdict

p = sys.argv[1] if len(sys.argv) > 1 else "."
path = p if p.endswith(".out") else os.path.join(p, "gizmo.out")

# oneway_route outcome tally per ONEWAY caller
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
        # On routed lines, the constructor route (hier=production, flat=debug/slow)
        # is reported as route=hier|flat. Flat in a default run means
        # flat is the slow debug constructor; production Route1 must be hierarchical.
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

print(f"=== ROUTE1 gate: {path} ===")
fail = False
CALLERS = ["density", "hydro_oneway"]
any_exercised = False
for c in CALLERS:
    o = outcome.get(c, {})
    routed = o.get("routed", 0); fell = o.get("fellback", 0)
    geom = o.get("geom_unavail", 0); off = o.get("off", 0)
    total = routed + fell + geom + off
    if total == 0:
        # Caller not exercised through request-driven on this problem (e.g. Mac
        # evrard routes hydro_oneway via tile_overlap). SKIP, not a failure.
        print(f"  ROUTE1[{c:14s}] not exercised (no request-driven ONEWAY lines)  SKIP")
        continue
    any_exercised = True
    rk = route_kind.get(c, {})
    hier = rk.get("hier", 0); flat = rk.get("flat", 0)
    ok = (routed > 0 and fell == 0 and geom == 0 and off == 0 and flat == 0)
    fail |= not ok
    print(f"  ROUTE1[{c:14s}] routed={routed} fellback={fell} geom_unavail={geom} off={off}  "
          f"route=hier:{hier}/flat:{flat}  {'PASS' if ok else 'FAIL' + (' (flat route observed; production Route1 should be hierarchical)' if flat else '')}")
if not any_exercised:
    print("  WARNING: no ONEWAY request-driven callers exercised — routed arm not tested by this run.")
    fail = True
# any other ONEWAY callers seen?
for c in outcome:
    if c not in CALLERS:
        print(f"  ROUTE1[{c:14s}] (other ONEWAY caller) {dict(outcome[c])}")
oracle_bad = (under_route > 0 or extra_match > 0 or true_7704 > 0 or true_7722 > 0)
print(f"  oracle: OK={oracle_ok} under_route(7704)={under_route} extra_match(7722)={extra_match} "
      f"true_7704_stops={true_7704} true_7722_stops={true_7722}  "
      f"{'PASS' if not oracle_bad else 'FAIL'}")
fail |= oracle_bad
print(f"\nSTEP-3 VERDICT: {'PASS — routed installs cleanly, oracle-proven == broadcast' if not fail else 'FAIL — see above'}")
sys.exit(1 if fail else 0)
