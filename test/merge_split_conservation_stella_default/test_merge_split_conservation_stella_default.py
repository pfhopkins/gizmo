"""test/merge_split_conservation under the Stella default configuration (Config_stella_default.sh): conservation, plus merge energy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "merge_split_conservation"))
from test_merge_split_conservation import run_merge_split_checks


def test_merge_split_conservation_stella_default():
    # STL-336: a merge keeps the survivor's four origin-tracer fractions (slots 27-30) instead of their mass-weighted mean
    run_merge_split_checks("merge_split_conservation_stella_default", expected_failures=tuple(f"merge:metal_mass[{k}]" for k in range(27, 31)))
