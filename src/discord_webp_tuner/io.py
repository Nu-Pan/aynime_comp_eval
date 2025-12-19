from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


def list_pngs(in_dir: Path) -> list[Path]:
    in_dir = in_dir.expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {in_dir}")
    return sorted([p for p in in_dir.rglob("*.png") if p.is_file()])


def parse_bg_color(bg_color: str) -> tuple[int, int, int]:
    s = bg_color.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bg_color must be 6 hex digits like 808080 (got {bg_color!r})")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid bg_color hex: {bg_color!r}") from e
    return (r, g, b)


def load_png(path_png: Path) -> Image.Image:
    try:
        img = Image.open(path_png)
        img.load()
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load PNG: {path_png}") from e


def composite_alpha_to_rgb(img: Image.Image, bg_rgb: tuple[int, int, int]) -> Image.Image:
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, color=(*bg_rgb, 255))
        out = Image.alpha_composite(bg, rgba).convert("RGB")
        return out
    return img.convert("RGB")


def resize_to_long_edge(img_rgb: Image.Image, long_edge: int) -> Image.Image:
    w, h = img_rgb.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    if max(w, h) == long_edge:
        return img_rgb
    if w >= h:
        new_w = long_edge
        new_h = max(1, round(h * (long_edge / w)))
    else:
        new_h = long_edge
        new_w = max(1, round(w * (long_edge / h)))
    return img_rgb.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CachePaths:
    resized_png: Path
    webp_dir: Path
    out_webp_dir: Path


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(b"\0")
    h.update(str(st.st_size).encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
    return h.hexdigest()[:16]


def cache_paths_for(
    *,
    out_dir: Path,
    path_png: Path,
    long_edge: int,
    bg_color: str,
) -> CachePaths:
    out_dir = out_dir.expanduser().resolve()
    cache_root = out_dir / "cache"
    resized_dir = cache_root / "resized"
    webp_dir = cache_root / "webp"
    out_webp_dir = out_dir / "out_webp"
    ensure_dir(resized_dir)
    ensure_dir(webp_dir)
    ensure_dir(out_webp_dir)

    fid = _file_fingerprint(path_png)
    resized_png = resized_dir / f"{fid}_le{int(long_edge)}_bg{bg_color}.png"
    return CachePaths(resized_png=resized_png, webp_dir=webp_dir, out_webp_dir=out_webp_dir)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_relpath(path: Path, base: Path) -> str:
    try:
        rel = path.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def iter_existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]
