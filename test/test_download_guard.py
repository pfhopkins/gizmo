"""Regression test for download_test_files (gizmo.test).

A locally-generated IC must never be clobbered by a remote copy. Motivating incident:
the HII_region test generates its own IC (make_HII_region_ics.py, changed 10 -> 30 Msun),
but download_test_files used to unconditionally urlretrieve the IC over the generated one.
A fresh clone whose machine could still reach a stale 10 Msun HII_region_ics.hdf5 silently
ran the old star. The guard skips the IC download when a local IC already exists, while
still fetching it when absent (download-based tests rely on that)."""

import gizmo.test as gt


def _spy_urlretrieve(fetched, contents):
    def fake(url, f):
        fetched.append(f)
        with open(f, "w") as fh:
            fh.write(contents)
    return fake


def test_download_does_not_clobber_local_ic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ic = tmp_path / "HII_region_ics.hdf5"
    ic.write_text("LOCALLY_GENERATED")

    fetched = []
    monkeypatch.setattr(gt, "urlretrieve", _spy_urlretrieve(fetched, "STALE_REMOTE"))
    gt.download_test_files("HII_region")

    assert "HII_region_ics.hdf5" not in fetched, "IC download should be skipped when a local IC exists"
    assert ic.read_text() == "LOCALLY_GENERATED", "local IC must not be overwritten by a remote copy"


def test_download_fetches_ic_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fetched = []
    monkeypatch.setattr(gt, "urlretrieve", _spy_urlretrieve(fetched, "REMOTE"))
    gt.download_test_files("HII_region")

    assert "HII_region_ics.hdf5" in fetched, "IC must still be fetched when no local copy exists"
