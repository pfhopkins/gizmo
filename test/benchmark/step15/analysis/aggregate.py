#!/usr/bin/env python3
"""aggregate.py — walk benchmarks_step15/runs/ and emit all_runs.csv + phase_breakdown.csv.

Per run, requires (in order of preference):
  - timings.json          (from parse_cpu_txt.py)
  - run_summary.json      (from parse_gizmo_out.py)
  - metadata.json         (from run_sweep.sh / run_local.sh)
  - pinning.log           (from gizmo_launch.sh)

If any of timings.json or metadata.json is missing, the run is logged as
'incomplete' but still recorded.

Outputs:
  <bench_root>/aggregated/all_runs.csv
  <bench_root>/aggregated/phase_breakdown.csv

Usage:
  aggregate.py <bench_root>
  aggregate.py /Users/phopkins/.../benchmarks_step15
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


CORE_COLUMNS = [
    "run_dir", "system", "build", "problem", "size_label", "ntotal",
    "nranks", "nthreads", "platform", "git_sha",
    "n_steps_completed", "n_bench_steps",
    "per_step_median_sec", "per_step_mean_sec", "per_step_p90_sec",
    "misc_pct_median", "misc_pct_max",
    "peak_rss_mb",
    "termination", "exit_code",
    "validation_status",
    "incomplete", "notes",
]


def load_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def parse_pinning_log(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not p.exists(): return out
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m: out[m.group(1)] = m.group(2)
    return out


def collect(bench_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs_dir = bench_root / "runs"
    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        print(f"WARN: {runs_dir} not found", file=sys.stderr)
        return rows, phase_rows
    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        meta = load_json(rd / "metadata.json")
        timings = load_json(rd / "timings.json")
        run_sum = load_json(rd / "run_summary.json")
        pin = parse_pinning_log(rd / "pinning.log")
        validation = load_json(rd / "validation.json")

        summary = timings.get("summary", {}) if timings else {}
        headline = summary.get("headline", {})

        notes_parts: list[str] = []
        incomplete = False
        if not timings: notes_parts.append("no_timings"); incomplete = True
        if not meta:    notes_parts.append("no_metadata")
        if not run_sum: notes_parts.append("no_run_summary")

        row = {
            "run_dir":              rd.name,
            "system":               meta.get("system", ""),
            "build":                meta.get("build", ""),
            "problem":              meta.get("problem", ""),
            "size_label":           meta.get("size_label", ""),
            "ntotal":               run_sum.get("ntotal", meta.get("ntotal", "")),
            "nranks":               meta.get("nranks", pin.get("nranks_intended", "")),
            "nthreads":             meta.get("nthreads", pin.get("nthreads_intended", "")),
            "platform":             meta.get("platform", ""),
            "git_sha":              meta.get("git_sha", ""),
            "n_steps_completed":    run_sum.get("n_steps_completed", ""),
            "n_bench_steps":        headline.get("n_bench_steps", ""),
            "per_step_median_sec":  headline.get("per_step_median_sec", ""),
            "per_step_mean_sec":    headline.get("per_step_mean_sec", ""),
            "per_step_p90_sec":     headline.get("per_step_p90_sec", ""),
            "misc_pct_median":      headline.get("misc_pct_median", ""),
            "misc_pct_max":         headline.get("misc_pct_max", ""),
            "peak_rss_mb":          run_sum.get("peak_rss_mb", ""),
            "termination":          run_sum.get("termination", ""),
            "exit_code":            pin.get("exit_code", ""),
            "validation_status":    validation.get("status", ""),
            "incomplete":           incomplete,
            "notes":                ";".join(notes_parts),
        }
        rows.append(row)

        # Phase breakdown
        for ph, sec in (summary.get("phase_medians_sec") or {}).items():
            phase_rows.append({
                "run_dir":  rd.name,
                "build":    row["build"],
                "problem":  row["problem"],
                "size_label": row["size_label"],
                "nranks":   row["nranks"],
                "nthreads": row["nthreads"],
                "phase":    ph,
                "median_sec": sec,
                "mean_sec": (summary.get("phase_means_sec") or {}).get(ph, ""),
                "p90_sec":  (summary.get("phase_p90_sec")   or {}).get(ph, ""),
            })
    return rows, phase_rows


def write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(",".join(cols or []) + "\n"); return
    fields = cols or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return 2
    bench_root = Path(sys.argv[1])
    rows, phase_rows = collect(bench_root)
    write_csv(bench_root / "aggregated" / "all_runs.csv", rows, CORE_COLUMNS)
    write_csv(bench_root / "aggregated" / "phase_breakdown.csv", phase_rows)
    print(f"Wrote {len(rows)} runs, {len(phase_rows)} phase rows -> "
          f"{bench_root / 'aggregated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
