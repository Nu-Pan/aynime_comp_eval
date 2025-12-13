import pytest


def test_resolve_cwebp_missing(monkeypatch):
    from discord_webp_tuner.pipeline import resolve_cwebp

    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(FileNotFoundError):
        resolve_cwebp(None)

