"""Field loop advection test (Hopkins & Raives 2016)

Tests advection of a magnetic field loop. The loop should be preserved
after one full advection period. Checks magnetic flux conservation.

Runs three variants with different divergence-cleaning methods:
  Base — no special div(B) treatment
  CG   — MHD_CONSTRAINED_GRADIENT=1
  MG   — MHD_MODIFIED_GRADIENT
"""

import pytest
import numpy as np
import h5py
import glob
import os
import shutil
from gizmo.test import default_mpi_ranks, build_gizmo_for_test, download_test_files


# Common Config.sh flags shared by all variants
_BASE_FLAGS = [
    "HYDRO_MESHLESS_FINITE_MASS",
    "BOX_PERIODIC",
    "BOX_SPATIAL_DIMENSION=2",
    "OUTPUT_BFIELD_DIVCLEAN_INFO",
    "MAGNETIC",
    "SELFGRAVITY_OFF",
    "INPUT_IN_DOUBLEPRECISION",
    "OUTPUT_IN_DOUBLEPRECISION",
    "DEVELOPER_MODE",
]

# Per-variant extra flags
_VARIANT_EXTRA_FLAGS = {
    "Base": [],
    "CG":   ["MHD_CONSTRAINED_GRADIENT=1"],
    "MG":   ["MHD_MODIFIED_GRADIENT"],
}


def _write_config(test_dir, variant):
    """Write a Config.sh for the given variant into the test directory."""
    flags = _BASE_FLAGS + _VARIANT_EXTRA_FLAGS[variant]
    config_path = os.path.join(test_dir, "Config.sh")
    with open(config_path, "w") as f:
        for flag in flags:
            f.write(flag + "\n")


def _run_variant(test_name, variant, num_mpi_ranks, num_omp_threads):
    """Build and run a single field_loop variant with its own output directory."""
    test_dir = f"test/{test_name}"
    output_dir = os.path.join(test_dir, f"output_{variant}")

    # clean previous output for this variant
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # write variant-specific Config.sh
    _write_config(test_dir, variant)

    # patch the params file to use the variant-specific output directory
    params_src = os.path.join(test_dir, f"{test_name}.params")
    params_variant = os.path.join(test_dir, f"{test_name}_{variant}.params")
    with open(params_src, "r") as f:
        params_text = f.read()
    params_text = params_text.replace("OutputDir                          output",
                                      f"OutputDir                          output_{variant}")
    with open(params_variant, "w") as f:
        f.write(params_text)

    # build
    build_gizmo_for_test(test_name, num_omp_threads)

    # rename binary so variants don't clobber each other
    gizmo_src = os.path.join(test_dir, "GIZMO")
    gizmo_dst = os.path.join(test_dir, f"GIZMO_{variant}")
    if os.path.exists(gizmo_dst):
        os.remove(gizmo_dst)
    os.rename(gizmo_src, gizmo_dst)

    # download ICs if needed (idempotent)
    os.chdir(test_dir)
    download_test_files(test_name)

    # run
    from os import environ
    if num_omp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_omp_threads)
    params_basename = f"{test_name}_{variant}.params"
    os.system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus ./GIZMO_{variant} {params_basename} 0 "
              f"1>test_{test_name}_{variant}.out 2>test_{test_name}_{variant}.err")
    os.chdir("../../")

    return output_dir


def _check_results(output_dir, variant):
    """Run assertions on the output of a single variant."""
    snaps = sorted(glob.glob(output_dir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError(f"GIZMO did not run successfully for variant {variant}.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
        mass0 = F["PartType0/Masses"][:]
        dens0 = F["PartType0/Density"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]
        mass_f = F["PartType0/Masses"][:]
        dens_f = F["PartType0/Density"][:]
        divB = F["PartType0/DivergenceOfMagneticField"][:]

    # Total magnetic energy should be approximately conserved
    Emag0 = np.sum(np.sum(B0**2, axis=1) * mass0 / dens0)
    Emagf = np.sum(np.sum(Bf**2, axis=1) * mass_f / dens_f)

    rel_err = abs(Emagf - Emag0) / Emag0
    print(f"[{variant}] RELATIVE_ERR == {rel_err}")
    assert rel_err < 0.05, f"[{variant}] Magnetic energy not conserved: relative error {rel_err:.4f}"

    # Divergence of magnetic field should be approximately zero
    divB_med = np.median(np.abs(divB))
    print(f"[{variant}] DIVB == {divB_med}")
    assert divB_med < 1.e-5, f"[{variant}] Median |divB| is too large: {divB_med:.4e}"


@pytest.mark.parametrize("variant", ["Base", "CG", "MG"])
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(1),))
@pytest.mark.parametrize("num_omp_threads", (0,))
def test_field_loop(variant, num_mpi_ranks, num_omp_threads):
    test_name = "field_loop"
    output_dir = _run_variant(test_name, variant, num_mpi_ranks, num_omp_threads)
    _check_results(output_dir, variant)


def flout():
    test_name = "field_loop"
    for variant in ["Base", "CG", "MG"]:
        outputdir = f"./output_{variant}"
        snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
        if not snaps:
            print(f"[{variant}] No snapshots found in {outputdir}")
            continue

        with h5py.File(snaps[0], "r") as F:
            B0 = F["PartType0/MagneticField"][:]
            mass0 = F["PartType0/Masses"][:]
            dens0 = F["PartType0/Density"][:]

        for snap in snaps:
            with h5py.File(snap, "r") as F:
                Bf = F["PartType0/MagneticField"][:]
                mass_f = F["PartType0/Masses"][:]
                dens_f = F["PartType0/Density"][:]
                divB = F["PartType0/DivergenceOfMagneticField"][:]
                dx = np.sqrt(mass_f / dens_f)
                Bf_magnitude = np.sqrt(np.sum(Bf**2, axis=1))
                divB_normed = divB * dx / Bf_magnitude

            Emag0 = np.sum(np.sum(B0**2, axis=1) * mass0 / dens0)
            Ef = np.sum(Bf**2, axis=1) * mass_f / dens_f
            Emagf = np.sum(Ef)

            rel_err = abs(Emagf - Emag0) / Emag0
            print(f"[{variant}] RELATIVE_ERR == {rel_err}")

            divB_med = np.median(np.abs(divB))
            print(f"[{variant}] DIVB == {divB_med}")
            divB_med_normed = np.median(np.abs(divB_normed))
            print(f"[{variant}] DIVB_NORMED == {divB_med_normed}")
            divB_med_normed_ewt = np.sum(np.abs(divB_normed) * Ef) / np.sum(Ef)
            print(f"[{variant}] DIVB_NORMED_VWT == {divB_med_normed_ewt}")
