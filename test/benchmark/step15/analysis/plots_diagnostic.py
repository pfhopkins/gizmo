#!/usr/bin/env python3
"""plots_diagnostic.py — toy plot from aggregated/all_runs.csv for sanity-checking.

Two plots:
  1. per-step time vs N, grouped by build (log-log), one line per build.
  2. phase breakdown bar chart for a single representative run.

Usage:
  plots_diagnostic.py <bench_root>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path) -> list[dict]:
    if not p.exists(): return []
    with p.open() as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float | None:
    try: return float(x)
    except (TypeError, ValueError): return None


def plot_time_vs_n(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    by_build: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        n = to_float(r.get("ntotal", ""))
        t = to_float(r.get("per_step_median_sec", ""))
        b = r.get("build", "?")
        if n and t:
            by_build.setdefault(b, []).append((n, t))
    for b, pts in by_build.items():
        pts.sort()
        if not pts: continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, "o-", label=f"build {b}")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("median time / step (s)")
    ax.set_title("Per-step time vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "diagnostic_time_vs_n.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close(fig)


def plot_phase_breakdown(phase_rows: list[dict], out_dir: Path) -> None:
    if not phase_rows: return
    # Pick the run with the largest N as the representative
    by_run: dict[str, list[dict]] = {}
    for r in phase_rows:
        by_run.setdefault(r["run_dir"], []).append(r)
    pick = max(by_run.keys())  # alphabetic max -> most recent timestamp
    rows = by_run[pick]
    rows = [r for r in rows if to_float(r.get("median_sec", "")) is not None]
    rows.sort(key=lambda r: -float(r["median_sec"]))
    rows = rows[:12]
    labels = [r["phase"] for r in rows]
    vals   = [float(r["median_sec"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[::-1], vals[::-1])
    ax.set_xlabel("median time / step (s)")
    ax.set_title(f"Phase breakdown — {pick}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "diagnostic_phase_breakdown.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return 2
    root = Path(sys.argv[1])
    rows = read_csv(root / "aggregated" / "all_runs.csv")
    phase_rows = read_csv(root / "aggregated" / "phase_breakdown.csv")
    out = root / "figures"
    plot_time_vs_n(rows, out)
    plot_phase_breakdown(phase_rows, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
