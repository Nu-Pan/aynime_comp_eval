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

