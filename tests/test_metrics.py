import pytest
from PIL import Image


def test_metrics_range_identical_images():
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("piq")

    from discord_webp_tuner.metrics import compute_metrics_y

    img = Image.new("RGB", (64, 64), color=(120, 130, 140))
    m = compute_metrics_y(img, img, compute_psnr=True)
    assert 0.0 <= m.ms_ssim_y <= 1.0
    assert m.gmsd >= 0.0
    assert m.psnr_y is not None
    assert m.ssimulacra2 is None


def test_ssimulacra2_identical_images_is_near_100():
    _ = pytest.importorskip("ssimulacra2")

    from discord_webp_tuner.metrics import compute_ssimulacra2_rgb

    img = Image.new("RGB", (96, 64), color=(120, 130, 140))
    val = compute_ssimulacra2_rgb(img, img)
    assert 0.0 <= val <= 100.0
    assert val > 99.9
