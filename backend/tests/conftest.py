import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Tests must never read or write the user's real settings/history."""
    monkeypatch.setenv("TRANSLATOR_DATA_DIR", str(tmp_path / "data"))
