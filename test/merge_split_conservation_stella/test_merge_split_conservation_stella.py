"""test/merge_split_conservation under the Stella configuration (SGCHEM net 17, RESOLVEDISM elements and dust, SN spawn)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "merge_split_conservation"))
from test_merge_split_conservation import run_merge_split_checks


def test_merge_split_conservation_stella():
    # STL-336: a merge keeps the survivor's four origin-tracer fractions (slots 44-47) instead of their mass-weighted mean
    run_merge_split_checks("merge_split_conservation_stella", expected_failures=tuple(f"merge:metal_mass[{k}]" for k in range(44, 48)))
