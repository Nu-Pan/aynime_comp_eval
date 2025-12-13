import pytest


def test_require_pillow_webp_support_missing(monkeypatch):
    from discord_webp_tuner import pipeline

    monkeypatch.setattr(pipeline.features, "check", lambda _: False)
    with pytest.raises(RuntimeError, match="WebP support"):
        pipeline.require_pillow_webp_support()

