"""habing_band_energy_densities test: scripts/habing_band_energy_densities.py (Habing 1968 and Draine 1978 ISRF in the Stella G0 bands,
Draine 2011 eqs. 12.6-12.7) reproduces the Habing 6-13.6 eV energy density of Draine 2011 (sec. 12.5) and the chemistry's
HABING_EGY_DENSITY_* constants of cooling/sgchem/chem_interface.h (Draine 1978 / 1.7, rounded to four digits). No GIZMO build.
See README.md."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BANDS = {"HABING_EGY_DENSITY_6_13P6EV_CGS": (6.0, 13.6), "HABING_EGY_DENSITY_8_13P6EV_CGS": (8.0, 13.6),
         "HABING_EGY_DENSITY_11P2_13P6EV_CGS": (11.2, 13.6), "HABING_EGY_DENSITY_6_8EV_CGS": (6.0, 8.0)}


def script():
    path = ROOT / "scripts" / "habing_band_energy_densities.py"
    assert path.exists(), "the script is not in scripts/"
    spec = importlib.util.spec_from_file_location("habing_band_energy_densities", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_habing_6_to_13p6_ev():
    """bar (written before the run): the Habing field in 6-13.6 eV is 5.29e-14 erg/cm^3 to 1% (Draine 2011, sec. 12.5)"""
    m = script()
    assert m.band(m.habing, 6.0, 13.6) == pytest.approx(5.29e-14, rel=1e-2)


@pytest.mark.parametrize("name", BANDS)
def test_code_constants_are_draine_over_1p7(name):
    """bar (written before the run): the constant in chem_interface.h equals the script's Draine 1978 band density / 1.7 to its four digits"""
    m = script()
    text = (ROOT / "src" / "cooling" / "sgchem" / "chem_interface.h").read_text()
    value = float(re.search(rf"#define {name}\s+\(([0-9.e+-]+)\)", text).group(1))
    lo, hi = BANDS[name]
    assert m.band(m.draine, lo, hi) / 1.7 == pytest.approx(value, rel=5e-4)
