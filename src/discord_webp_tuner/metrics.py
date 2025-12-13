from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MetricResult:
    ms_ssim_y: float
    gmsd: float
    psnr_y: float | None


def _require_torch_piq():
    try:
        import torch  # noqa: F401
        import piq  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing metrics dependencies. Install with: pip install -e \".[metrics]\" "
            "(requires torch + piq for MS-SSIM/GMSD)."
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


def compute_metrics_y(orig_rgb: Image.Image, recon_rgb: Image.Image, *, compute_psnr: bool = True) -> MetricResult:
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
        return MetricResult(ms_ssim_y=float(ms.item()), gmsd=float(gm.item()), psnr_y=psnr)
