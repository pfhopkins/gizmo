"""Initial conditions for the protostellar dust-heating tests.

Two families, both with a single radiating sink and no self-gravity, so the only thing being
exercised is the radiation scheme:

  glassbox    uniform periodic glass box at a chosen n_H. The analytic answer is known exactly in
              the optically thin limit, which makes it the quantitative check on the tree sum.
  core        centrally-condensed rho ~ r^-p core in vacuum, after Chakrabarti & McKee (2005).
              Spans thin to thick, so it exercises the emergent-photosphere and blend terms.

Variants of `core` exist to separate scheme behaviour from IC artifacts:
  relaxed     same rho(r) to machine precision, but the radial anisotropy left by MakeCloud's
              glass-stretching is relaxed away. MakeCloud maps a uniform glass through
              r_new ~ r^a with a = 3/(3+p), which elongates cells radially by exactly a (a = 2 for
              p = 1.5). This variant tests whether that matters. Measured: it does not.
  displaced   sink moved off the density peak by a few cell lengths. NOT cosmetic: the emitter's
              envelope-slope estimator uses GradRho at the sink, which vanishes by symmetry at the
              centre of the sink's own envelope, so the centred case is pathological. See README.

Requires MakeCloud (https://github.com/mikegrudic/MakeCloud) on PATH.
"""

import os
import re
import subprocess

import h5py
import numpy as np

PC = 3.085678e18
MSUN = 1.989e33
PROTONMASS = 1.6726219e-24
AU_PER_PC = 206264.806
KERNEL_FAC = 0.357  # params Softening is Plummer-equivalent; SinkRadius = Softening / KERNEL_FAC


def _run_makecloud(args):
    subprocess.run(["MakeCloud"] + args, check=True, capture_output=True, text=True)


def _fix_sink_radius(path, softening_pc=3.11e-5):
    """MakeCloud writes SinkRadius = the softening value, but GIZMO's own sink-formation path
    assigns ForceSoftening_KernelRadius = softening / 0.357. An IC-placed sink and a self-formed
    sink would otherwise get radii differing by 2.8x for identical parameters."""
    with h5py.File(path, "r+") as f:
        sr = f["PartType5"]["SinkRadius"]
        sr[...] = np.array([softening_pc / KERNEL_FAC], dtype=sr.dtype)


def _patch_params(src, dst, ic_name, time_max, n_snap=10, isrf=0.0):
    text = open(src).read()
    text = re.sub(r"(?m)^(InitCondFile\s+)\S+", r"\g<1>" + ic_name, text)
    subs = {
        "TimeMax": f"{time_max:.5e}",
        "TimeBetSnapshot": f"{time_max / n_snap:.5e}",
        "MaxSizeTimestep": f"{time_max / (30 * n_snap):.5e}",
    }
    for key, val in subs.items():
        text, n = re.subn(rf"(?m)^({key}\s+)\S+", rf"\g<1>{val}", text, count=1)
        assert n == 1, key
    # the test isolates the stellar term; the background field is switched off entirely
    if "InterstellarRadiationFieldStrength" in text:
        text = re.sub(r"(?m)^(InterstellarRadiationFieldStrength\s+)\S+", rf"\g<1>{isrf}", text)
    else:
        text += f"\nInterstellarRadiationFieldStrength   {isrf}\n"
    open(dst, "w").write(text)


def make_glassbox(n_H=1e5, box_pc=0.1, n_cells=64**3, m_star=1.0):
    """Uniform periodic glass box at exactly n_H.

    The mass is set explicitly rather than via MakeCloud's --nH, because its makebox path uses
    rho = n_H * m_p / 0.71 while GIZMO's UNIT_DENSITY_IN_NHCGS is rho / m_p with no hydrogen mass
    fraction. Passing --nH would give GIZMO a density 1/0.71 = 1.41x the requested value.

    n_cells must be a cube of a power of two, or MakeCloud's box path silently falls back to
    Poisson sampling instead of a glass.
    """
    cbrt = round(n_cells ** (1 / 3))
    assert cbrt**3 == n_cells and (cbrt & (cbrt - 1)) == 0, "n_cells must be (2^k)^3 for a glass"
    m_gas = n_H * PROTONMASS * (box_pc * PC) ** 3 / MSUN
    _run_makecloud([
        f"--M={m_gas:.6f}", f"--R={box_pc / 2}", f"--L={box_pc}", "--makebox",
        f"--N={n_cells}", f"--Mstar={m_star}",
        f"--x_star={box_pc / 2},{box_pc / 2},{box_pc / 2}",
        "--bturb=0", "--unit_system=starforge_classic", "--tmax=1", "--nsnap=10",
    ])
    return m_gas


def make_core(p=1.5, radius_pc=0.05, m_gas=1.0, n_cells=64**3, m_star=1.0, box_pc=0.5):
    """Centrally-condensed rho ~ r^-p core in vacuum, with a sink at the centre.

    --alpha_turb=0 gives a static cloud: with SELFGRAVITY_OFF the gas then stays put, so the dust
    temperature is not contaminated by dynamics. Uses MakeCloud's cloud path, which is glass-based
    (get_glass_coords) -- verified empirically, nearest-neighbour scatter 0.084 against 0.367 for
    Poisson.
    """
    _run_makecloud([
        f"--R={radius_pc}", f"--M={m_gas}", f"--N={n_cells}", f"--density_exponent=-{p}",
        "--alpha_turb=0", "--bturb=0", "--no_diffuse_gas", f"--Mstar={m_star}",
        f"--x_star={box_pc / 2},{box_pc / 2},{box_pc / 2}",
        "--unit_system=starforge_classic", "--tmax=1", "--nsnap=10",
    ])


def relax_glass(src, dst, n_iter=30, n_neighbor=12, step=0.25, center=(0.25, 0.25, 0.25)):
    """Remove the radial anisotropy left by glass-stretching, preserving rho(r) exactly.

    MakeCloud maps a uniform glass through r_new ~ r^a, which leaves cells elongated radially by
    exactly a = 3/(3+p). Here a short repulsion relaxation isotropises the local neighbourhood,
    and after each step the sorted radii are re-imposed, so the radial mass profile is unchanged to
    machine precision and only the angular arrangement moves.
    """
    from scipy.spatial import cKDTree

    import shutil
    shutil.copy(src, dst)
    c = np.asarray(center)
    with h5py.File(dst, "r+") as f:
        x = np.float64(f["PartType0"]["Coordinates"][:]) - c
        r_target = np.sort(np.linalg.norm(x, axis=1))
        for _ in range(n_iter):
            tree = cKDTree(x)
            d, idx = tree.query(x, k=n_neighbor + 1)
            eps = d[:, 1:].mean(axis=1)
            v = x[idx[:, 1:]] - x[:, None, :]
            dist = np.maximum(np.linalg.norm(v, axis=2), 1e-30)
            w = np.clip(eps[:, None] / dist - 1.0, -0.5, 2.0)
            x = x + step * (-(v / dist[:, :, None] * w[:, :, None]).sum(axis=1) / n_neighbor) * eps[:, None]
            rr = np.linalg.norm(x, axis=1)
            rnew = np.empty_like(rr)
            rnew[np.argsort(rr)] = r_target  # re-impose rho(r) exactly
            x = x * (rnew / np.maximum(rr, 1e-30))[:, None]
        f["PartType0"]["Coordinates"][...] = (x + c).astype(f["PartType0"]["Coordinates"].dtype)


def displace_star(src, dst, n_cells_offset=2.0, center=(0.25, 0.25, 0.25)):
    """Move the sink off the density peak by a few local cell lengths.

    The emitter estimates the envelope slope as n = |grad rho| * h / rho, which for rho ~ r^-n is
    n * (h/r) -- equal to n only when the kernel radius matches the distance from the density peak.
    A sink at the centre of its own envelope has r = 0, the gradient vanishes by symmetry, and the
    estimate is noise. Displacing the sink is the cleanest way to expose that.
    """
    from scipy.spatial import cKDTree

    import shutil
    shutil.copy(src, dst)
    c = np.asarray(center)
    with h5py.File(dst, "r+") as f:
        x = np.float64(f["PartType0"]["Coordinates"][:])
        r = np.linalg.norm(x - c, axis=1)
        tree = cKDTree(x)
        d, _ = tree.query(x[np.argsort(r)[:200]], k=2)
        cell = float(np.median(d[:, 1]))
        pos = f["PartType5"]["Coordinates"]
        p = np.float64(pos[:])
        p[0] += np.array([1, 1, 1]) / np.sqrt(3) * n_cells_offset * cell
        pos[...] = p.astype(pos.dtype)
    return cell * AU_PER_PC
