"""Stella parameter files leave gas smoothing lengths uncapped.

With adaptive gas softening, MinGasKernelRadiusFractional defaults to 1 (begrun.cc), which floors every gas kernel radius
at 2.8 x SofteningGas (gravtree.cc: MinKernelRadius = fraction x ForceSoftening[0]). Stella runs keep a FIRE-style tiny
floor instead, so each Stella parameter file built with adaptive gas softening must set the fraction explicitly, low
enough that the floor stays below 1e-3 pc."""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KERNEL_FAC = 2.8  # cubic-spline kernel extent over the Plummer-equivalent softening
FLOOR_MAX_KPC = 1e-6


def active_flags(config):
    return {line.split("%")[0].split("#")[0].strip().split("=")[0] for line in config.read_text().splitlines()} - {""}


def stella_params_files():
    """(params file, Config.sh) pairs for Stella builds with adaptive gas softening."""
    files = subprocess.run(["git", "ls-files", "*.params"], cwd=REPO, capture_output=True, text=True).stdout.split()
    pairs = []
    for f in map(Path, files):
        if f.parent == Path("."):
            config = REPO / ("Config_" + f.stem + ".sh")
        else:
            config = REPO / f.parent / "Config.sh"
        if not config.exists():
            continue
        flags = active_flags(config)
        if "GALSF_RESOLVEDISM" in flags and flags & {"ADAPTIVE_GRAVSOFT_FORALL", "ADAPTIVE_GRAVSOFT_FORGAS"}:
            pairs.append(f)
    return pairs


def param(text, name):
    m = re.search(rf"^\s*{name}\s+(\S+)", text, re.M)
    return float(m.group(1)) if m else None


@pytest.mark.parametrize("params", stella_params_files(), ids=str)
def test_gas_kernel_radius_uncapped(params):
    text = (REPO / params).read_text()
    frac = param(text, "MinGasKernelRadiusFractional")
    assert frac is not None, f"{params}: MinGasKernelRadiusFractional not set (defaults to 1: h floored at 2.8 x SofteningGas)"
    floor = frac * KERNEL_FAC * param(text, "SofteningGas")
    assert floor < FLOOR_MAX_KPC, f"{params}: gas kernel radius floored at {1e3 * floor:.3g} pc"
