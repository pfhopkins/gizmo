"""Merge/split conservation with SINK_THERMALFEEDBACK and no METALS: the pending sink energy must be summed on
merge even when the metal block that once held that sum is compiled out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "merge_split_conservation"))
from test_merge_split_conservation import run_merge_split_checks  # noqa: E402


def test_merge_split_conservation_nometals_sinkfb():
    run_merge_split_checks("merge_split_conservation_nometals_sinkfb")
