"""tests/test_post_holdout.py  -  full-history data stays locked until the holdout has been run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402


def test_post_holdout_locked_before_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    with pytest.raises(PermissionError):
        data.load_post_holdout()
