"""test/sgchem_cooling_path with CHEMISTRYNETWORK=15 (Nelson & Langer 1999 CO network)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sgchem_cooling_path"))
from test_sgchem_cooling_path import run_cooling_path_checks


def test_sgchem_cooling_path_net15():
    run_cooling_path_checks("sgchem_cooling_path_net15")
