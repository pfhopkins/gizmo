#!/usr/bin/env python3
"""parse_cpu_txt.py — parse GIZMO's cpu.txt into a structured timings.json.

cpu.txt has a repeating block per timestep:
    Step N, Time: T, CPUs: K
    Nactive=A, Imbal(Max/Mean)=I
    <ascii bar>
    total         <secs>  <pct>%
    tree+gravity  <secs>  <pct>%
       treebuild  <secs>  <pct>%
       ...
    misc          <secs>  <pct>%
    <blank>

Parser keeps the indentation hierarchy: top-level entries become step["phases"]
keys; indented entries become children of the most-recent top-level entry.

Output:
{
  "ntask": <int>,
  "n_steps": <int>,
  "steps": [
    {"step": N, "sim_time": T, "ntask": K, "nactive": A, "imbalance": I,
     "phases": {
        "total":        {"sec": ..., "pct": ...},
        "tree+gravity": {"sec": ..., "pct": ..., "children": {"treebuild": {...}, ...}},
        ...
     }},
    ...
  ],
  "summary": {
     "phase_medians_sec": {phase: median_over_steps_2_to_9},
     "phase_means_sec":   {...},
     "phase_p90_sec":     {...},
     "headline": {"per_step_median_sec": ..., "misc_pct_median": ..., ...}
  }
}

Steps 0 and 1 are excluded from the summary (warmup); if fewer steps are
present, summary uses whatever is available and notes 'incomplete=true'.

Usage:
  parse_cpu_txt.py <run_dir>          # writes <run_dir>/timings.json
  parse_cpu_txt.py <run_dir> --stdout # prints JSON to stdout
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"^Step\s+(\d+),\s+Time:\s+(\S+),\s+CPUs:\s+(\d+)\s*$")
NACT_RE = re.compile(r"^Nactive=(\d+),\s+Imbal\(Max/Mean\)=(\S+)\s*$")
PHASE_RE = re.compile(r"^(\s*)([A-Za-z][\w()+\-/ ]*?)\s+([\d.]+)\s+([\d.]+)%\s*$")


def parse_cpu_txt(path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    cur_top: str | None = None
    ntask_global = -1

    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = STEP_RE.match(line)
            if m:
                if cur is not None:
                    steps.append(cur)
                step_n, sim_t, ntask = int(m.group(1)), float(m.group(2)), int(m.group(3))
                ntask_global = ntask
                cur = {
                    "step": step_n, "sim_time": sim_t, "ntask": ntask,
                    "nactive": None, "imbalance": None, "phases": {},
                }
                cur_top = None
                continue
            if cur is None:
                continue
            m = NACT_RE.match(line)
            if m:
                cur["nactive"] = int(m.group(1))
                try:
                    cur["imbalance"] = float(m.group(2))
                except ValueError:
                    cur["imbalance"] = None
                continue
            m = PHASE_RE.match(line)
            if m:
                indent, name, sec, pct = m.group(1), m.group(2).strip(), float(m.group(3)), float(m.group(4))
                if len(indent) == 0:
                    cur["phases"][name] = {"sec": sec, "pct": pct, "children": {}}
                    cur_top = name
                else:
                    if cur_top is None:
                        cur["phases"][name] = {"sec": sec, "pct": pct, "children": {}}
                    else:
                        cur["phases"][cur_top]["children"][name] = {"sec": sec, "pct": pct}
                continue
            # blank or other → ignore (but does not terminate a step; the next Step
            # header does, so multiple blanks are fine)
        if cur is not None:
            steps.append(cur)

    return {"ntask": ntask_global, "n_steps": len(steps), "steps": steps}


def summarize(parsed: dict[str, Any]) -> dict[str, Any]:
    steps = parsed["steps"]
    n = len(steps)
    if n == 0:
        return {"empty": True}

    # IMPORTANT: GIZMO writes CUMULATIVE wallclock per phase to cpu.txt, not
    # per-step. To get per-step costs we must diff consecutive Step records.
    # That gives us n-1 per-step samples; the very first step has no prior
    # reference and is excluded as warmup anyway.
    deltas: list[dict[str, float]] = []
    for prev, cur in zip(steps[:-1], steps[1:]):
        d: dict[str, float] = {}
        # Top-level phases
        all_top = set(prev["phases"]) | set(cur["phases"])
        for ph in all_top:
            p = prev["phases"].get(ph, {}).get("sec", 0.0)
            c = cur["phases"].get(ph, {}).get("sec", 0.0)
            d[ph] = max(0.0, c - p)
        # Children: flatten as 'parent/child' so they appear in phase_series
        for ph in all_top:
            pc = (prev["phases"].get(ph) or {}).get("children", {}) or {}
            cc = (cur["phases"].get(ph)  or {}).get("children", {}) or {}
            for ch in set(pc) | set(cc):
                p = pc.get(ch, {}).get("sec", 0.0)
                c = cc.get(ch, {}).get("sec", 0.0)
                d[f"{ph}/{ch}"] = max(0.0, c - p)
        deltas.append(d)

    # Skip warmup deltas (first 2 if we have enough)
    if len(deltas) >= 4:
        bench_deltas = deltas[2:]
        warmup_skipped = 2
    elif len(deltas) >= 2:
        bench_deltas = deltas[1:]
        warmup_skipped = 1
    else:
        bench_deltas = deltas
        warmup_skipped = 0

    # Collect per-step seconds for each phase
    phase_series: dict[str, list[float]] = {}
    for d in bench_deltas:
        for ph, sec in d.items():
            phase_series.setdefault(ph, []).append(sec)

    def stat(values: list[float], fn) -> float:
        if not values:
            return 0.0
        try:
            return float(fn(values))
        except statistics.StatisticsError:
            return 0.0

    medians = {ph: stat(v, statistics.median) for ph, v in phase_series.items()}
    means   = {ph: stat(v, statistics.mean)   for ph, v in phase_series.items()}
    def p90(v):
        if not v: return 0.0
        v2 = sorted(v); k = max(0, int(round(0.9 * (len(v2) - 1))))
        return v2[k]
    p90s = {ph: p90(v) for ph, v in phase_series.items()}

    misc = phase_series.get("misc", [])
    total = phase_series.get("total", [])
    misc_pct = [100.0 * m / t if t > 0 else 0.0 for m, t in zip(misc, total)]

    headline = {
        "per_step_median_sec": medians.get("total", 0.0),
        "per_step_mean_sec":   means.get("total", 0.0),
        "per_step_p90_sec":    p90s.get("total", 0.0),
        "misc_pct_median":     stat(misc_pct, statistics.median) if misc_pct else 0.0,
        "misc_pct_max":        max(misc_pct) if misc_pct else 0.0,
        "warmup_skipped":      warmup_skipped,
        "n_bench_steps":       len(bench_deltas),
        "n_total_steps":       len(steps),
    }
    return {
        "phase_medians_sec": medians,
        "phase_means_sec":   means,
        "phase_p90_sec":     p90s,
        "headline":          headline,
        "_note":              "Per-step values computed by diffing consecutive cumulative cpu.txt records.",
    }


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__); return 2
    run_dir = Path(args[0])
    to_stdout = "--stdout" in args
    cpu_txt = run_dir / "output" / "cpu.txt"
    if not cpu_txt.exists():
        cpu_txt = run_dir / "cpu.txt"
    if not cpu_txt.exists():
        print(f"ERROR: no cpu.txt in {run_dir}", file=sys.stderr); return 2
    parsed = parse_cpu_txt(cpu_txt)
    parsed["summary"] = summarize(parsed)
    out = run_dir / "timings.json"
    if to_stdout:
        json.dump(parsed, sys.stdout, indent=2); print()
    else:
        out.write_text(json.dumps(parsed, indent=2))
        h = parsed["summary"].get("headline", {})
        print(f"OK  {run_dir.name}: n_steps={parsed['n_steps']} "
              f"median={h.get('per_step_median_sec',0):.3f}s "
              f"misc_pct={h.get('misc_pct_median',0):.1f}%  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
