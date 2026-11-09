"""makefile_systype_orion test: SYSTYPE=Orion_Kokkos selects the Orion build of the Kokkos port (mpicxx -std=c++20, OpenMP, the Kokkos OpenMP
install at KOKKOS_HOME_ORION). Looks at the commands `make -n` would run; nothing is built. See README.md."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KOKKOS_HOME = os.environ.get("KOKKOS_HOME_ORION", "/u/uli/sw/kokkos/5.0.0-openmp-gcc13")


def dry_run():
    src = ROOT / "src"
    config = src / "Config.sh"
    keep = config.read_text() if config.exists() else None
    config.write_text("SELFGRAVITY_OFF\n")
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "SYSTYPE")}
    try:
        r = subprocess.run(["make", "-n", "HOME_GIZMO=", "GIZMO"], cwd=src, env={**env, "SYSTYPE": "Orion_Kokkos"}, capture_output=True, text=True)
    finally:
        if keep is None:
            config.unlink()
        else:
            config.write_text(keep)
    return r.stdout


def test_orion_kokkos_compile_line():
    """bar (written before the run): the first compile command uses mpicxx with -std=c++20 and -fopenmp and the Kokkos include path of
    the Orion install; the link line has -lkokkoscore and the library path"""
    out = dry_run().splitlines()
    cc = next(l for l in out if " -c core/main.cc" in l)
    assert cc.startswith("mpicxx -std=c++20") and "-fopenmp" in cc and f"-I{KOKKOS_HOME}/include" in cc
    link = next(l for l in out if "-o GIZMO" in l)
    assert "-lkokkoscore" in link and f"-L{KOKKOS_HOME}/lib64" in link
