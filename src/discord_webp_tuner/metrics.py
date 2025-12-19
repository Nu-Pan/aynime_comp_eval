from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MetricResult:
    ms_ssim_y: float
    gmsd: float
    psnr_y: float | None
    # SSIMULACRA2: higher is better (100 is identical).
    ssimulacra2: float | None = None


def _require_torch_piq():
    try:
        import torch  # noqa: F401
        import piq  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing metrics dependencies. Install with: pip install -e \".[metrics]\" "
            "(requires torch + piq for MS-SSIM/GMSD)."
        ) from e


def _require_ssimulacra2():
    try:
        import ssimulacra2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing SSIMULACRA2 dependency. Install with: pip install -e \".[metrics]\" "
            "(requires ssimulacra2 (+ scipy) for SSIMULACRA2)."
        ) from e


def pil_to_y_tensor(img_rgb: Image.Image):
    _require_torch_piq()
    import torch

    y = img_rgb.convert("YCbCr").split()[0]
    y_np = np.asarray(y, dtype=np.float32) / 255.0
    t = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0)
    return t


def psnr_y(orig_y, recon_y) -> float:
    _require_torch_piq()
    import torch

    mse = torch.mean((orig_y - recon_y) ** 2).clamp_min(1e-12)
    val = 10.0 * torch.log10(1.0 / mse)
    return float(val.item())


def compute_ssimulacra2_rgb(orig_rgb: Image.Image, recon_rgb: Image.Image) -> float:
    _require_ssimulacra2()
    import numpy as np
    import ssimulacra2.ssimulacra2 as s2

    if orig_rgb.size != recon_rgb.size:
        raise ValueError(f"Size mismatch for SSIMULACRA2: orig={orig_rgb.size} recon={recon_rgb.size}")

    orig_img = np.asarray(orig_rgb.convert("RGB"), dtype=np.float64)
    recon_img = np.asarray(recon_rgb.convert("RGB"), dtype=np.float64)

    orig_linear = s2.srgb_to_linear(orig_img)
    recon_linear = s2.srgb_to_linear(recon_img)

    orig_xyb = s2.make_positive_xyb(s2.linear_rgb_to_xyb(orig_linear))
    recon_xyb = s2.make_positive_xyb(s2.linear_rgb_to_xyb(recon_linear))

    msssim = s2.Msssim()

    img1 = orig_xyb
    img2 = recon_xyb
    for scale in range(int(s2.kNumScales)):
        if img1.shape[0] < 8 or img1.shape[1] < 8:
            break

        mul = img1 * img1
        sigma1_sq = s2.blur_image(mul)

        mul = img2 * img2
        sigma2_sq = s2.blur_image(mul)

        mul = img1 * img2
        sigma12 = s2.blur_image(mul)

        mu1 = s2.blur_image(img1)
        mu2 = s2.blur_image(img2)

        scale_data = s2.MsssimScale()
        scale_data.avg_ssim = s2.ssim_map(mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
        scale_data.avg_edgediff = s2.edge_diff_map(img1, mu1, img2, mu2)
        msssim.scales.append(scale_data)

        if scale < int(s2.kNumScales) - 1:
            orig_linear = s2.downsample(orig_linear, 2, 2)
            recon_linear = s2.downsample(recon_linear, 2, 2)
            img1 = s2.make_positive_xyb(s2.linear_rgb_to_xyb(orig_linear))
            img2 = s2.make_positive_xyb(s2.linear_rgb_to_xyb(recon_linear))

    val = float(msssim.score())
    if not np.isfinite(val):
        raise ValueError(f"SSIMULACRA2 returned non-finite value: {val}")
    return val


def compute_metrics_y(
    orig_rgb: Image.Image,
    recon_rgb: Image.Image,
    *,
    compute_psnr: bool = True,
    compute_ssimulacra2: bool = False,
) -> MetricResult:
    _require_torch_piq()
    import piq
    import torch

    if orig_rgb.size != recon_rgb.size:
        raise ValueError(f"Size mismatch for metrics: orig={orig_rgb.size} recon={recon_rgb.size}")

    with torch.no_grad():
        orig_y = pil_to_y_tensor(orig_rgb.convert("RGB")).clamp(0, 1)
        recon_y = pil_to_y_tensor(recon_rgb.convert("RGB")).clamp(0, 1)
        ms = piq.multi_scale_ssim(orig_y, recon_y, data_range=1.0)
        gm = piq.gmsd(orig_y, recon_y, data_range=1.0)
        psnr = psnr_y(orig_y, recon_y) if compute_psnr else None
        s2 = compute_ssimulacra2_rgb(orig_rgb, recon_rgb) if compute_ssimulacra2 else None
        return MetricResult(ms_ssim_y=float(ms.item()), gmsd=float(gm.item()), psnr_y=psnr, ssimulacra2=s2)
