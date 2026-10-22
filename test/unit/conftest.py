"""Shared fixture for the stellar-table checks in this directory: each test file defines TABLE_PATH and load_table()."""
import os
import pytest


@pytest.fixture(scope="module")
def tbl(request):
    mod = request.module
    if not os.path.exists(mod.TABLE_PATH):
        pytest.skip(f"stellar table not installed: {mod.TABLE_PATH}")
    return mod.load_table()
