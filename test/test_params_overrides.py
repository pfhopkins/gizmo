"""write_params_with_overrides must keep every override on its own line."""

from gizmo.test import write_params_with_overrides


def test_override_after_a_last_line_without_newline(tmp_path):
    params = tmp_path / "x.params"
    params.write_text("TimeMax 1.0\nMaxSizeTimestep 0.1")      # no trailing newline
    out = write_params_with_overrides(str(params), {"NewKey": 7, "TimeMax": 2.0})
    lines = [l.split() for l in open(out) if l.strip() and not l.startswith("%")]
    assert ["MaxSizeTimestep", "0.1"] in lines
    assert ["NewKey", "7"] in lines
    assert ["TimeMax", "2.0"] in lines
