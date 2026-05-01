#!/usr/bin/env python3
"""parse_gizmo_out.py — extract headline numbers from a GIZMO stdout/stderr pair.

Pulls:
- particle counts (NumPart, total particles, types)
- box size, time begin/end
- step count completed
- peak RSS if reported
- [BENCH] tagged stderr lines (Tier-2+ diagnostic prints)
- termination reason (clean exit vs. wallclock vs. crash)

Output: <run_dir>/run_summary.json

Usage:
  parse_gizmo_out.py <run_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def safe_float(s: str) -> float | None:
    try: return float(s)
    except (TypeError, ValueError): return None


def parse_run(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    sout = run_dir / "gizmo.out"
    serr = run_dir / "gizmo.err"
    out_text = sout.read_text(errors="replace") if sout.exists() else ""
    err_text = serr.read_text(errors="replace") if serr.exists() else ""

    # NumPart total
    m = re.search(r"\bN_total\s*=\s*(\d+)", out_text) or \
        re.search(r"Total number of particles\s*[:=]\s*(\d+)", out_text) or \
        re.search(r"\bAllocated memory.*?N\s*=\s*(\d+)", out_text)
    if m: out["ntotal"] = int(m.group(1))

    # Per-type counts (very common GIZMO startup line)
    types = re.findall(r"Type\s+(\d+):\s+(\d+)\s+particles", out_text)
    if types:
        out["per_type"] = {int(t): int(n) for t, n in types}

    # Step count: max Step N seen in stdout
    step_ns = [int(x) for x in re.findall(r"^\s*Step\s+(\d+)\b", out_text, flags=re.MULTILINE)]
    if step_ns:
        out["max_step"] = max(step_ns)
        out["n_steps_completed"] = len(set(step_ns))

    # Peak RSS (a few common formats)
    m = re.search(r"Peak\s+RSS.*?([\d.]+)\s*(MB|GB|KB)", out_text)
    if m:
        v = float(m.group(1)); u = m.group(2).upper()
        scale = {"KB": 1/1024, "MB": 1, "GB": 1024}[u]
        out["peak_rss_mb"] = v * scale

    # Termination
    if re.search(r"endrun called", out_text):
        out["termination"] = "endrun"
    elif re.search(r"reached.*TimeMax|finished\s+main\s+loop", out_text, flags=re.IGNORECASE):
        out["termination"] = "timemax"
    elif re.search(r"CPU\s*time\s*exceeded|reached\s+TimeLimitCPU", out_text, flags=re.IGNORECASE):
        out["termination"] = "walltime"
    elif err_text.strip():
        out["termination"] = "stderr_present"
    else:
        out["termination"] = "unknown"

    # [BENCH] lines (stderr only by convention, but check both)
    bench = []
    for txt in (out_text, err_text):
        for line in txt.splitlines():
            if "[BENCH]" in line:
                bench.append(line.strip())
    if bench:
        out["bench_lines"] = bench
        out["bench_count"] = len(bench)

    out["stderr_nonempty"] = bool(err_text.strip())
    out["stderr_lines"] = err_text.count("\n") if err_text else 0
    return out


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return 2
    rd = Path(sys.argv[1])
    summary = parse_run(rd)
    (rd / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"OK  {rd.name}: {summary.get('n_steps_completed', 0)} steps, "
          f"{summary.get('termination','?')}, ntotal={summary.get('ntotal','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
