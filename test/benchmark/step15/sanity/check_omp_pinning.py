#!/usr/bin/env python3
"""check_omp_pinning.py — verify that a finished run was actually pinned as intended.

Reads pinning.log + gizmo.out from a run directory and confirms:
- requested OMP_NUM_THREADS matches what the binary saw (via [BENCH-PIN] in stdout
  if GIZMO_BENCHMARK_VERBOSE was on; otherwise we just confirm env consistency).
- ranks*threads <= avail_cores.
- exit_code == 0.

Exit code:
  0 = pinning consistent
  1 = pinning inconsistent (run is NOT trustworthy for benchmarking)
  2 = run crashed / incomplete

Usage:
  check_omp_pinning.py <run_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_pinning_log(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_bench_pin_stdout(p: Path) -> dict[str, str]:
    """Pick up [BENCH-PIN] lines if GIZMO_BENCHMARK_VERBOSE is on."""
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if "[BENCH-PIN]" not in line:
            continue
        for kv in re.findall(r"(\w+)\s*=\s*(\S+)", line):
            out[kv[0]] = kv[1]
    return out


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    pin_log = parse_pinning_log(rd / "pinning.log")
    pin_stdout = parse_bench_pin_stdout(rd / "gizmo.out")

    if not pin_log:
        print(f"FAIL: no pinning.log in {rd}", file=sys.stderr)
        return 2

    intended = int(pin_log.get("nthreads_intended", "-1"))
    omp_set = int(pin_log.get("OMP_NUM_THREADS", "-1"))
    avail = int(pin_log.get("avail_cores_os", "-1"))
    rxt = int(pin_log.get("ranks_x_threads", "-1"))
    rc = int(pin_log.get("exit_code", "-1"))

    issues: list[str] = []
    if intended != omp_set:
        issues.append(f"intended threads ({intended}) != OMP_NUM_THREADS ({omp_set})")
    if avail > 0 and rxt > avail:
        issues.append(f"oversubscription: ranks*threads={rxt} > cores={avail}")
    if rc != 0:
        issues.append(f"non-zero exit_code={rc}")

    # If GIZMO printed [BENCH-PIN] (Tier-2+), cross-check:
    if pin_stdout:
        kk_omp = pin_stdout.get("nthreads_omp")
        kk_conc = pin_stdout.get("kokkos_concurrency")
        if kk_omp and int(kk_omp) != intended:
            issues.append(f"[BENCH-PIN] runtime nthreads_omp={kk_omp} != intended={intended}")
        if kk_conc and int(kk_conc) != intended:
            issues.append(f"[BENCH-PIN] kokkos_concurrency={kk_conc} != intended={intended}")

    if issues:
        print(f"FAIL pinning check for {rd.name}:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 1 if rc == 0 else 2

    print(f"OK   {rd.name}: ranks={pin_log.get('nranks_intended')} threads={intended} avail={avail}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
