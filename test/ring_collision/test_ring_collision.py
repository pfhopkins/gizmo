"""Ring collision elastic/solid body test (Hopkins 2015)

Tests elastic solid body physics by colliding two rings. The rings should
deform on impact and bounce apart, preserving their structure due to
elastic restoring forces. With the Tillotson EOS parameters set here,
P = cs^2 * (rho - rho_0) with cs = rho_0 = 1.

Validates (ring_collision.params, rubber rings):
  1. The run reaches its end time with mass conserved.
  2. The rings bounce: restitution (relative velocity of the two rings' centres of mass after / before) between 0.4
     and 0.9 (measured: 0.66). Rings that weld or stick fail this.
  3. Each ring stays one body through the collision: in every snapshot its largest connected group of cells (linked
     within 1.5 lattice spacings) holds at least 99% of them. Rings that tear or fragment fail this.
  4. The rings are separate and round again at the end: a gap between them, and the ratio of the minor to major axis of
     each ring's mass distribution above 0.6 (measured: 0.83).

The same directory holds three material presets on the same rings, as starting points for other materials (marked slow;
`-m "not slow"` leaves them out): a stiffer, harder rubber (ring_collision_hard.params), a plastic clay that yields
(ring_collision_clay.params) and a brittle material that fractures (ring_collision_brittle.params, which needs
EOS_DAMAGE_POROSITY=3 added to Config.sh). Their checks are the qualitative outcome each material should give.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, flush_colorbar, assert_final_time, default_omp_threads, problem_output_dir

TEST_NAME = "ring_collision"
DP = 0.086          # lattice spacing of ring_collision_ics (cell mass 0.0074 at density 1)


def ring_metrics(snaps):
    """restitution, smallest largest-connected-group fraction of either ring over all snapshots, final axis ratios,
    final gap between the rings, final fraction of cells with damage above 0.5"""
    def load(snap):
        with h5py.File(snap, "r") as F:
            g = F["PartType0"]; order = np.argsort(g["ParticleIDs"][:])
            d = {k: g[k][:][order] for k in ("Coordinates", "Velocities", "Masses")}
            d["Damage"] = g["Damage"][:][order] if "Damage" in g else np.zeros(len(order))
        return d
    s0 = load(snaps[0])
    left = s0["Coordinates"][:, 0] < np.median(s0["Coordinates"][:, 0])
    rings, m = (left, ~left), s0["Masses"]
    v_rel = lambda s: np.average(s["Velocities"][~left, 0], weights=m[~left]) - np.average(s["Velocities"][left, 0], weights=m[left])
    connected = 1.0
    for snap in snaps:
        x = load(snap)["Coordinates"][:, :2]
        for ring in rings:
            tree = cKDTree(x[ring])
            _, labels = connected_components(tree.sparse_distance_matrix(tree, 1.5 * DP), directed=False)
            connected = min(connected, np.bincount(labels).max() / ring.sum())
    s1 = load(snaps[-1])
    x = s1["Coordinates"][:, :2]
    axis_ratio = [np.sqrt(np.divide(*np.linalg.eigvalsh(np.cov(x[ring].T)))) for ring in rings]
    return dict(restitution=-v_rel(s1) / v_rel(s0), connected=connected, axis_ratio=min(axis_ratio),
                gap=x[~left, 0].min() - x[left, 0].max(), damaged=np.mean(s1["Damage"] > 0.5))


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_ring_collision(num_mpi_ranks, num_omp_threads):
    test_name = TEST_NAME
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Plot each snapshot as a scatter of the resolution elements, coloured by
    # density. The elastic material occupies only the two thin ring bands, with
    # vacuum around them; a slice/kernel interpolation of the density onto a
    # grid smears the (sparse) particles across the surrounding empty space and
    # produces spurious "wedge" structure, so we render the particles directly.
    for i, snap in enumerate(snaps):
        with h5py.File(snap, "r") as F:
            pos = F["PartType0/Coordinates"][:]
            rho = F["PartType0/Density"][:]
            t = F["Header"].attrs["Time"]
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=rho, s=6, cmap="viridis", edgecolors="none")
        ax.set_aspect("equal")
        flush_colorbar(sc, ax=ax, label="Density")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Ring Collision t={t:.0f}")
        fig.savefig(f"test/{test_name}/density_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Mass conservation
    with h5py.File(snaps[0], "r") as F:
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        mass_f = F["PartType0/Masses"][:]
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    r = ring_metrics(snaps)
    print(f"restitution {r['restitution']:.3f}; largest connected group {r['connected']:.4f}; axis ratio {r['axis_ratio']:.3f}; gap {r['gap']:.2f}")
    assert 0.4 < r["restitution"] < 0.9, f"restitution {r['restitution']:.3f}: the rings did not bounce elastically"
    assert r["connected"] > 0.99, f"a ring tore: its largest connected group fell to {r['connected']:.3f} of its cells"
    assert r["gap"] > 0, "the rings have not separated"
    assert r["axis_ratio"] > 0.6, f"the rings did not recover their shape (axis ratio {r['axis_ratio']:.3f})"


# expected outcome of each material preset (measured values in brackets)
MATERIALS = {
    "hard": ((), "a stiffer, harder rubber (shear modulus 1, bulk 3): bounces like the rubber, slightly more elastically"),
    "clay": ((), "plastic: yield strength 0.003, far below the impact stresses: the rings flatten and stick"),
    "brittle": (("EOS_DAMAGE_POROSITY=3",), "brittle: tensile flaws (Weibull k 1e12, m 9) nucleate cracks: the rings fracture"),
}


@pytest.mark.slow
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("material", list(MATERIALS))
def test_ring_collision_material(num_mpi_ranks, num_omp_threads, material):
    flags, _ = MATERIALS[material]
    problem = f"{TEST_NAME}_{material}"
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags=flags, params_name=problem)
    snaps = sorted(glob.glob(f"{problem_output_dir(TEST_NAME, problem, flags)}/snapshot_*.hdf5"))
    assert len(snaps) >= 2, "GIZMO did not run successfully."
    assert_final_time(snaps[-1], TEST_NAME)
    r = ring_metrics(snaps)
    print(f"{material}: restitution {r['restitution']:.3f}; largest connected group {r['connected']:.4f}; axis ratio {r['axis_ratio']:.3f}; "
          f"gap {r['gap']:.2f}; damaged fraction {r['damaged']:.3f}")
    if material == "hard":              # (0.69, 1.0, 0.89)
        assert 0.4 < r["restitution"] < 0.95 and r["connected"] > 0.99 and r["axis_ratio"] > 0.6
    elif material == "clay":            # (-0.01, 0.999, 0.26)
        assert abs(r["restitution"]) < 0.1, "clay rings should stick, not bounce"
        assert r["connected"] > 0.98 and r["axis_ratio"] < 0.5, "clay rings should flatten without breaking up"
    elif material == "brittle":         # (-0.04, 0.37, 0.09 of cells damaged)
        assert r["damaged"] > 0.02, "no fracture damage"
        assert r["connected"] < 0.9, "brittle rings should break into pieces"
