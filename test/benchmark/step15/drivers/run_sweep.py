#!/usr/bin/env python3
"""run_sweep.py — drive a Step-15 benchmark sweep from a JSON sweep spec.

Materializes deterministically-named run directories, generates per-run params
files (with overrides applied), copies the right binary + IC, and either
launches locally or emits SLURM scripts.

Usage:
  run_sweep.py <sweep.json> [--dry-run] [--only A,C] [--filter poisson_box]
  run_sweep.py test/benchmark/step15/sweeps/tier1_infra.json --dry-run

Requires that build_three_refs.sh has already produced binaries at
   test/benchmark/step15/build/GIZMO_<problem>_<platform>_<build>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GIZMO_ROOT = Path(__file__).resolve().parents[4]
STEP15 = GIZMO_ROOT / "test" / "benchmark" / "step15"
BUILD_DIR = STEP15 / "build"
LAUNCH = STEP15 / "drivers" / "gizmo_launch.sh"


# ---------- helpers ------------------------------------------------------

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def patch_params(src: Path, dst: Path, overrides: dict[str, str], ic_basename: str) -> None:
    """Copy GIZMO params file, replacing 'InitCondFile' and any keys in `overrides`.

    Lines starting with % or empty are passed through unchanged. For other
    lines, the first whitespace-separated token is the key.
    """
    out_lines = []
    seen: set[str] = set()
    overrides = dict(overrides)
    overrides["InitCondFile"] = ic_basename  # always
    for line in src.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            out_lines.append(line); continue
        parts = stripped.split(None, 1)
        key = parts[0]
        if key in overrides:
            out_lines.append(f"{key}    {overrides[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    # Append any overrides not present in the source params
    for k, v in overrides.items():
        if k not in seen and k != "InitCondFile":
            out_lines.append(f"{k}    {v}")
    dst.write_text("\n".join(out_lines) + "\n")


def ensure_poisson_ic(N_per_side: int, work_dir: Path) -> Path:
    """Run the IC generator if needed; return the IC file path (basename used in params)."""
    ic_name = f"poisson_box_{N_per_side}_ics.hdf5"
    cache_dir = STEP15 / "ic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ic_cached = cache_dir / ic_name
    if not ic_cached.exists():
        gen = GIZMO_ROOT / "test" / "poisson_box" / "make_poisson_box_ics.py"
        # The generator hard-codes BoxSize=100 and N_per_side via its main; we call the
        # function directly via -c to keep things simple.
        cmd = [sys.executable, "-c",
               f"import sys; sys.path.insert(0, '{gen.parent}'); "
               f"import importlib.util as u; "
               f"spec=u.spec_from_file_location('m','{gen}'); "
               f"m=u.module_from_spec(spec); spec.loader.exec_module(m); "
               f"m.make_poisson_box_ics(output_file='{ic_cached}', N_per_side={N_per_side})"]
        print(f"[ic] generating {ic_cached}")
        subprocess.run(cmd, check=True)
    target = work_dir / ic_name
    if not target.exists():
        shutil.copy2(ic_cached, target)
    return target


def ensure_gmc_cooling_ic(work_dir: Path) -> Path:
    """Download gmc_cooling IC file if not cached; return the IC file path in work_dir."""
    ic_name = "gmc_cooling_ics.hdf5"
    cache_dir = STEP15 / "ic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ic_cached = cache_dir / ic_name
    if not ic_cached.exists():
        urls = [
            f"http://www.tapir.caltech.edu/~phopkins/sims/{ic_name}",
            f"https://users.flatironinstitute.org/~mgrudic/gizmo_tests/gmc_cooling/{ic_name}",
        ]
        print(f"[ic] downloading {ic_name} ...")
        downloaded = False
        for url in urls:
            try:
                urllib.request.urlretrieve(url, ic_cached)
                downloaded = True
                print(f"[ic] downloaded from {url}")
                break
            except Exception as exc:
                print(f"[ic] failed {url}: {exc}")
        if not downloaded:
            raise FileNotFoundError(
                f"Could not download {ic_name} from any URL. "
                f"Place it manually at {ic_cached}")
    target = work_dir / ic_name
    if not target.exists():
        shutil.copy2(ic_cached, target)
    return target


def select_binary(problem: str, platform: str, build: str) -> Path:
    p = BUILD_DIR / f"GIZMO_{problem}_{platform}_{build}"
    if not p.exists():
        raise FileNotFoundError(
            f"Binary not found: {p}\n"
            f"Run: ./test/benchmark/step15/drivers/build_three_refs.sh {problem} "
            f"--platform {platform} --builds {build}")
    return p


def system_label(platform: str) -> str:
    if platform == "mac":
        return "macbook"
    if platform == "vista":
        host = os.environ.get("HOSTNAME", "vista")
        return host.split(".")[0]
    return platform


# ---------- run-dir construction ----------------------------------------

def build_run(spec: dict, problem: dict, size: dict, build: str, rt: dict, *,
              bench_root: Path, platform: str) -> Path:
    sysl = system_label(platform)
    name = (f"{utc_stamp()}_{sysl}_{build}_{problem['name']}"
            f"_{size['label']}_R{rt['nranks']}_T{rt['nthreads']}")
    run_dir = bench_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(exist_ok=True)

    # Binary
    bin_path = select_binary(problem["name"], platform, build)
    binary_in_run = run_dir / "GIZMO"
    if not binary_in_run.exists():
        shutil.copy2(bin_path, binary_in_run)
        binary_in_run.chmod(0o755)
    # Also stash Config.sh and binary meta if present
    for sfx in (".Config.sh", ".meta.json", ".buildlog"):
        s = bin_path.parent / (bin_path.name + sfx)
        if s.exists(): shutil.copy2(s, run_dir / s.name)

    # IC + params (problem-specific)
    if problem["name"] == "poisson_box":
        ic_path = ensure_poisson_ic(size["N_per_side"], run_dir)
        ic_basename = ic_path.stem  # GIZMO wants filename without .hdf5
    elif problem["name"] == "gmc_cooling":
        ic_path = ensure_gmc_cooling_ic(run_dir)
        ic_basename = ic_path.stem
    else:
        raise NotImplementedError(f"IC handling for problem '{problem['name']}' not yet wired")

    src_params = GIZMO_ROOT / problem["params_template"]
    overrides = dict(problem.get("params_overrides", {}))
    patch_params(src_params, run_dir / "params.txt", overrides, ic_basename)

    # Runtime data files (TREECOOL, spcool_tables, ewald table) — search a few
    # canonical locations. Build C (legacy gizmo-cpp) lacks the auto-download
    # feature, so spcool_tables MUST be present before launch.
    runtime_search = [
        GIZMO_ROOT / "test" / "benchmark" / "step15" / "runtime_data",
        GIZMO_ROOT / "cooling",
        GIZMO_ROOT,
        GIZMO_ROOT / "test" / "benchmark",
    ]
    for f in ("TREECOOL", "spcool_tables", "ewald_spc_table_64_dbl.dat"):
        for base in runtime_search:
            cand = base / f
            if cand.exists():
                dst = run_dir / f
                if not dst.exists():
                    if cand.is_dir(): shutil.copytree(cand, dst)
                    else:             shutil.copy2(cand, dst)
                break

    # Metadata
    is_kokkos = build in ("A", "B")
    metadata = {
        "name":         name,
        "system":       sysl,
        "platform":     platform,
        "build":        build,
        "problem":      problem["name"],
        "size_label":   size["label"],
        "size_params":  size,
        "nranks":       rt["nranks"],
        "nthreads":     rt["nthreads"],
        "is_kokkos":    is_kokkos,
        "binary":       str(bin_path),
        "params_template": str(src_params),
        "params_overrides": overrides,
        "min_steps":    spec.get("min_steps"),
        "warmup_steps": spec.get("warmup_steps"),
        "abort_on_misc_pct": spec.get("abort_on_misc_pct"),
        "git_sha":      git_sha(),
        "created_utc":  datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(GIZMO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except subprocess.CalledProcessError:
        return ""


# ---------- launching ----------------------------------------------------

def launch_local(run_dir: Path) -> int:
    md = json.loads((run_dir / "metadata.json").read_text())
    cmd = [str(LAUNCH),
           str(run_dir / "GIZMO"),
           str(run_dir / "params.txt"),
           str(md["nranks"]),
           str(md["nthreads"]),
           "1" if md["is_kokkos"] else "0"]
    print(f"[run] {run_dir.name}")
    print(f"  cwd={run_dir}")
    print(f"  cmd={' '.join(cmd)}")
    return subprocess.call(cmd, cwd=run_dir)


def emit_slurm(run_dir: Path, partition: str, account: str, walltime: str) -> Path:
    md = json.loads((run_dir / "metadata.json").read_text())
    script = run_dir / "run.sbatch"
    script.write_text(f"""#!/bin/bash
#SBATCH -J s15_{md['build']}_{md['problem']}_{md['size_label']}
#SBATCH -p {partition}
#SBATCH -N 1
#SBATCH --ntasks-per-node {md['nranks']}
#SBATCH --cpus-per-task {md['nthreads']}
#SBATCH -t {walltime}
#SBATCH -A {account}
#SBATCH -o slurm.out
#SBATCH -e slurm.err
source $HOME/.bashrc
{LAUNCH} ./GIZMO ./params.txt {md['nranks']} {md['nthreads']} {1 if md['is_kokkos'] else 0}
""")
    script.chmod(0o755)
    return script


# ---------- post-run analysis -------------------------------------------

def post_run(run_dir: Path) -> None:
    py = sys.executable
    for s in ("parse_cpu_txt.py", "parse_gizmo_out.py"):
        subprocess.call([py, str(STEP15 / "analysis" / s), str(run_dir)])
    subprocess.call([py, str(STEP15 / "sanity" / "check_omp_pinning.py"), str(run_dir)])


# ---------- main --------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only",   default="", help="comma-separated build labels to include")
    ap.add_argument("--filter", default="", help="substring problem filter")
    ap.add_argument("--no-launch", action="store_true",
                    help="materialize run dirs but do not launch")
    args = ap.parse_args()

    spec = json.loads(Path(args.sweep).read_text())
    bench_root = Path(spec["bench_root"]).expanduser()
    bench_root.mkdir(parents=True, exist_ok=True)
    (bench_root / "runs").mkdir(exist_ok=True)
    (bench_root / "aggregated").mkdir(exist_ok=True)
    platform = spec["platform"]
    submit = spec.get("submit_mode", "local")

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    builds = [b for b in spec["builds"] if (not only or b in only)]

    plan: list[tuple[dict, dict, str, dict]] = []
    for problem in spec["problems"]:
        if args.filter and args.filter not in problem["name"]:
            continue
        for size in problem["sizes"]:
            for build in builds:
                for rt in spec["rank_thread_grid"]:
                    plan.append((problem, size, build, rt))

    print(f"Sweep '{spec['name']}': {len(plan)} runs planned, platform={platform}, mode={submit}")
    if args.dry_run:
        for problem, size, build, rt in plan:
            print(f"  - {problem['name']:14s} {size['label']:10s} {build}  R={rt['nranks']} T={rt['nthreads']}")
        return 0

    # Build run dirs
    run_dirs: list[Path] = []
    for problem, size, build, rt in plan:
        try:
            rd = build_run(spec, problem, size, build, rt,
                           bench_root=bench_root, platform=platform)
            run_dirs.append(rd)
        except FileNotFoundError as e:
            print(f"SKIP: {e}", file=sys.stderr)

    if args.no_launch:
        print(f"Materialized {len(run_dirs)} run dirs in {bench_root / 'runs'}")
        return 0

    # Launch
    if submit == "local":
        for rd in run_dirs:
            rc = launch_local(rd)
            print(f"  exit={rc}")
            post_run(rd)
    elif submit == "slurm":
        partition  = spec.get("slurm", {}).get("partition", "gh-dev")
        account    = spec.get("slurm", {}).get("account", "TG-NAIRR260139")
        walltime   = spec.get("slurm", {}).get("walltime", "00:30:00")
        batch_size = spec.get("slurm", {}).get("batch_size", 0)  # 0 = unlimited
        submitted = 0
        for rd in run_dirs:
            sb = emit_slurm(rd, partition, account, walltime)
            print(f"  sbatch {sb}")
            rc = subprocess.call(["sbatch", str(sb)], cwd=rd)
            if rc == 0:
                submitted += 1
                if batch_size and submitted >= batch_size:
                    print(f"[slurm] batch_size={batch_size} reached; stopping submission.")
                    print(f"[slurm] Re-run with --no-launch to skip already-submitted runs,")
                    print(f"[slurm] or wait for queue slots and re-run without --only.")
                    break
    else:
        print(f"ERROR: unknown submit_mode {submit}", file=sys.stderr); return 2

    # Aggregate (local mode; slurm mode aggregates after jobs finish)
    if submit == "local":
        subprocess.call([sys.executable, str(STEP15 / "analysis" / "aggregate.py"),
                         str(bench_root)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
