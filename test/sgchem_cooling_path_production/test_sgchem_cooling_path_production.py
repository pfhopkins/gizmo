"""test/sgchem_cooling_path on the production path: Config_stella_default.sh (C++ backend, CHEMISTRYNETWORK=5, COOLING_OPERATOR_SPLIT,
no sinks). Same driver and bars; under the split the hydro heating checks become their split counterparts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sgchem_cooling_path"))
from test_sgchem_cooling_path import run_cooling_path_checks


def test_sgchem_cooling_path_production():
    run_cooling_path_checks("sgchem_cooling_path_production")
