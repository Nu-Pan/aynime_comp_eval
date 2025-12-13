from __future__ import annotations

import datetime as dt
import io
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import PIL
from PIL import Image, features
from tqdm import tqdm

from .config import SweepConfig, TargetSpec
from .io import (
    cache_paths_for,
    composite_alpha_to_rgb,
    ensure_dir,
    list_pngs,
    load_png,
    parse_bg_color,
    safe_relpath,
    write_json,
)
from .metrics import compute_metrics_y


def require_pillow_webp_support() -> None:
    if not features.check("webp"):
        raise RuntimeError(
            "Pillow is built without WebP support. Install a Pillow build that includes WebP "
            "(e.g. upgrade Pillow / reinstall wheels) and verify: "
            "python -c \"from PIL import features; print(features.check('webp'))\""
        )


def _validate_pillow_webp_encoder(*, method: int, sharp_yuv: bool) -> None:
    require_pillow_webp_support()
    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": "WEBP", "quality": 80, "method": int(method)}
    if sharp_yuv:
        save_kwargs["use_sharp_yuv"] = True

    try:
        img.save(buf, **save_kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"Pillow WebP encoder does not support requested options: method={method}, sharp_yuv={sharp_yuv}. "
            f"Details: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Pillow WebP encoder: {e}") from e


def _encode_webp(
    *,
    out_webp: Path,
    img_rgb: Image.Image,
    q: int,
    method: int,
    sharp_yuv: bool,
) -> None:
    ensure_dir(out_webp.parent)

    try:
        save_kwargs: dict[str, Any] = {"format": "WEBP", "quality": int(q), "method": int(method)}
        if sharp_yuv:
            save_kwargs["use_sharp_yuv"] = True
        img_rgb.save(out_webp, **save_kwargs)
    except Exception as e:
        raise RuntimeError(f"WebP encode failed for {out_webp} q={q} method={method} sharp_yuv={sharp_yuv}") from e


def _decode_webp(path_webp: Path) -> Image.Image:
    try:
        with Image.open(path_webp) as img:
            img.load()
            out = img.convert("RGB")
            out.load()
            return out
    except Exception as e:
        raise RuntimeError(f"Failed to decode WebP: {path_webp}") from e


@dataclass(frozen=True)
class _WorkItem:
    path_png: Path


def _process_one_image(
    *,
    work: _WorkItem,
    in_dir: Path,
    out_dir: Path,
    config: SweepConfig,
    compute_psnr: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    require_pillow_webp_support()
    bg_rgb = parse_bg_color(config.bg_color)
    path_png = work.path_png

    img = load_png(path_png)
    w0, h0 = img.size
    base_rgb = composite_alpha_to_rgb(img, bg_rgb)

    for target in config.targets:
        cache = cache_paths_for(
            out_dir=out_dir,
            path_png=path_png,
            target_name=target.name,
            long_edge=target.long_edge,
            bg_color=config.bg_color,
        )
        if not cache.resized_png.exists():
            resized = _resize_for_target(base_rgb, target)
            ensure_dir(cache.resized_png.parent)
            resized.save(cache.resized_png, format="PNG", optimize=True)
        else:
            with Image.open(cache.resized_png) as im:
                resized = im.convert("RGB")
                resized.load()

        tw, th = resized.size
        rel = safe_relpath(path_png, in_dir)

        for q in config.q_values:
            webp_cache = (
                cache.webp_dir
                / f"{cache.resized_png.stem}_q{int(q)}_m{config.method}_sharp{int(config.sharp_yuv)}.webp"
            )
            if not webp_cache.exists():
                _encode_webp(
                    out_webp=webp_cache,
                    img_rgb=resized,
                    q=q,
                    method=config.method,
                    sharp_yuv=config.sharp_yuv,
                )
            recon = _decode_webp(webp_cache)
            m = compute_metrics_y(resized, recon, compute_psnr=compute_psnr)
            webp_bytes = int(os.path.getsize(webp_cache))
            bpp = float(webp_bytes) / float(tw * th)

            rows.append(
                {
                    "path_png": rel,
                    "w": int(w0),
                    "h": int(h0),
                    "target": target.name,
                    "target_long_edge": int(target.long_edge),
                    "target_w": int(tw),
                    "target_h": int(th),
                    "q": int(q),
                    "method": int(config.method),
                    "sharp_yuv": bool(config.sharp_yuv),
                    "bg_color": str(config.bg_color),
                    "webp_bytes": int(webp_bytes),
                    "bpp": float(bpp),
                    "ms_ssim_y": float(m.ms_ssim_y),
                    "gmsd": float(m.gmsd),
                    "psnr_y": (None if m.psnr_y is None else float(m.psnr_y)),
                    "webp_path_cache": str(webp_cache),
                }
            )

    return rows


def _resize_for_target(img_rgb: Image.Image, target: TargetSpec) -> Image.Image:
    from .io import resize_to_long_edge

    return resize_to_long_edge(img_rgb, target.long_edge)


def sweep_dir(
    *,
    in_dir: Path,
    out_dir: Path,
    config: SweepConfig,
    compute_psnr: bool = True,
) -> Path:
    in_dir = in_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    ensure_dir(out_dir / "results")

    require_pillow_webp_support()
    _validate_pillow_webp_encoder(method=config.method, sharp_yuv=config.sharp_yuv)
    pngs = list_pngs(in_dir)
    if not pngs:
        raise FileNotFoundError(f"No PNG files found under: {in_dir}")

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_cfg_path = out_dir / "results" / f"run_config_{run_ts}.json"
    write_json(
        run_cfg_path,
        {
            "timestamp": run_ts,
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "encoder": {
                "type": "pillow",
                "pillow_version": getattr(PIL, "__version__", None),
                "webp_supported": bool(features.check("webp")),
            },
            "compute_psnr": bool(compute_psnr),
            "config": config.to_jsonable(),
        },
    )

    work_items = [_WorkItem(path_png=p) for p in pngs]
    all_rows: list[dict[str, Any]] = []

    if config.jobs <= 1:
        for w in tqdm(work_items, desc="images", unit="img"):
            all_rows.extend(
                _process_one_image(
                    work=w,
                    in_dir=in_dir,
                    out_dir=out_dir,
                    config=config,
                    compute_psnr=compute_psnr,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=int(config.jobs)) as ex:
            futures = [
                ex.submit(
                    _process_one_image,
                    work=w,
                    in_dir=in_dir,
                    out_dir=out_dir,
                    config=config,
                    compute_psnr=compute_psnr,
                )
                for w in work_items
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="images", unit="img"):
                all_rows.extend(fut.result())

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_ts)

    csv_latest = out_dir / "results" / "metrics.csv"
    csv_stamped = out_dir / "results" / f"metrics_{run_ts}.csv"
    df.to_csv(csv_stamped, index=False, encoding="utf-8")
    df.to_csv(csv_latest, index=False, encoding="utf-8")

    _maybe_save_webp_outputs(df=df, in_dir=in_dir, out_dir=out_dir, config=config)
    return csv_latest


def _maybe_save_webp_outputs(*, df: pd.DataFrame, in_dir: Path, out_dir: Path, config: SweepConfig) -> None:
    mode = str(config.save_webp).lower().strip()
    if mode not in {"all", "best", "none"}:
        raise ValueError(f"Invalid save_webp: {config.save_webp!r} (expected all|best|none)")

    out_webp_dir = out_dir / "out_webp"
    ensure_dir(out_webp_dir)

    if mode == "none":
        return

    if mode == "all":
        for _, row in df.iterrows():
            src = Path(str(row["webp_path_cache"]))
            rel = str(row["path_png"]).replace("/", "_").replace("\\", "_")
            target = str(row["target"])
            q = int(row["q"])
            dst = out_webp_dir / f"{Path(rel).stem}_{target}_q{q}.webp"
            if not dst.exists():
                shutil.copy2(src, dst)
        return

    from .plot import knee_by_max_distance, pareto_front

    group_cols = ["path_png", "target"]
    for (path_png, target), g in df.groupby(group_cols, sort=False):
        front = pareto_front(g)
        knee = knee_by_max_distance(front)
        if knee is not None:
            pick = front.loc[front["q"] == knee.q].iloc[0]
        else:
            b = front["bpp"].astype(float)
            e = (1.0 - front["ms_ssim_y"].astype(float))
            b_norm = (b - b.min()) / max(1e-12, (b.max() - b.min()))
            e_norm = (e - e.min()) / max(1e-12, (e.max() - e.min()))
            dist = (b_norm * b_norm + e_norm * e_norm) ** 0.5
            pick = front.loc[dist.idxmin()]

        src = Path(str(pick["webp_path_cache"]))
        rel = str(path_png).replace("/", "_").replace("\\", "_")
        dst = out_webp_dir / f"{Path(rel).stem}_{target}_q{int(pick['q'])}.webp"
        if not dst.exists():
            shutil.copy2(src, dst)
