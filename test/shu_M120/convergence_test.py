"""Manual convergence runner for shu_M120 subcycling experiments.

This file intentionally is not a pytest test.  Its filename matches pytest's
default ``*_test.py`` pattern, so keep the expensive run logic behind ``main()``
to make ``pytest --collect-only`` side-effect free.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

from test_shu_M120 import compute_test_statistic
from gizmo.test import build_and_run_test, get_cooling_tables
import numpy as np


def main():
    test_name = 'shu_M120'
    test_dir = 'test/shu_M120'

    configs = {
        'baseline': ('DEVELOPER_MODE',),
        'subcycle_rt': ('DEVELOPER_MODE', 'TRANSPORT_SUBCYCLE=10'),
        'subcycle_rt_cooling': ('DEVELOPER_MODE', 'TRANSPORT_SUBCYCLE=10', 'TRANSPORT_SUBCYCLE_COOLING'),
    }

    benchmark_stats = compute_test_statistic(test_dir + '/shu_M120_exact.hdf5')

    results = {}
    for name, flags in configs.items():
        print(f'\n=== Running {name} ===', flush=True)
        get_cooling_tables(test_dir)
        build_and_run_test(test_name, 8, 2, flags)
        snap = test_dir + '/output/snapshot_005.hdf5'
        stats = compute_test_statistic(snap)
        maxrd, meanrd = 0, 0
        for key in stats:
            reldiff = np.abs(stats[key] - benchmark_stats[key]) / (np.abs(benchmark_stats[key]) + 1e-30)
            maxrd = max(maxrd, np.nanmax(reldiff))
            meanrd += np.nanmean(reldiff)
            print(f'  {key:8s}  max={np.nanmax(reldiff):.4f}  mean={np.nanmean(reldiff):.4f}', flush=True)
        meanrd /= len(stats)
        results[name] = (maxrd, meanrd)
        print(f'  Overall max:  {maxrd:.4f}  mean: {meanrd:.4f}', flush=True)

    print('\n=== SUMMARY ===', flush=True)
    for name, (maxrd, meanrd) in results.items():
        print(f'  {name:25s}  max={maxrd:.4f}  mean={meanrd:.4f}', flush=True)


if __name__ == '__main__':
    main()
