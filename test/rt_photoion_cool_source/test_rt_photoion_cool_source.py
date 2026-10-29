"""M1 multi-frequency photoionisation (RT_CHEM_PHOTOION=2) from a cool source (star_Teff = 1000 K).

The blackbody that rt_get_sigma integrates over each ionising band underflows to zero above ~70 eV
at this temperature, so that band has no photons. Its cross-section and heating normalisations must
stay finite (zero), otherwise the photoionisation rates turn into NaN in every cell the band reaches.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from glob import glob

from gizmo.test import build_and_run_test, get_cooling_tables, get_final_snapshot, assert_final_time

TEST_NAME = "rt_photoion_cool_source"
TEST_DIR = Path(__file__).parent


def test_rt_photoion_cool_source():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        sys.path.insert(0, str(TEST_DIR))
        from make_rt_photoion_cool_source_ics import make_ics
        make_ics(str(ic_file))
    get_cooling_tables(str(TEST_DIR))
    build_and_run_test(TEST_NAME, 2, 1)
    final = get_final_snapshot(TEST_NAME)
    assert_final_time(final, TEST_NAME)
    for snap in sorted(glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5")):
        with h5py.File(snap, "r") as f:
            for key, data in f["PartType0"].items():
                values = data[()]
                if np.issubdtype(values.dtype, np.floating):
                    assert np.all(np.isfinite(values)), f"{snap}: PartType0/{key} has non-finite values"
