from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class KneePoint:
    q: int
    bpp: float
    err: float


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["bpp", "ms_ssim_y"], ascending=[True, False]).reset_index(drop=True)
    best_ms = -1.0
    keep = []
    for i, row in d.iterrows():
        if float(row["ms_ssim_y"]) > best_ms:
            keep.append(i)
            best_ms = float(row["ms_ssim_y"])
    return d.loc[keep].sort_values("bpp", ascending=True)


def knee_by_max_distance(front: pd.DataFrame) -> KneePoint | None:
    if len(front) < 3:
        return None
    x = front["bpp"].to_numpy(dtype=float)
    y = (1.0 - front["ms_ssim_y"].to_numpy(dtype=float))

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx = x1 - x0
    dy = y1 - y0
    denom = (dx * dx + dy * dy) ** 0.5
    if denom <= 0:
        return None

    dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
    idx = int(dist.argmax())
    row = front.iloc[idx]
    return KneePoint(q=int(row["q"]), bpp=float(row["bpp"]), err=float(1.0 - float(row["ms_ssim_y"])))


def scatter_plot(
    *,
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    show_pareto: bool = True,
    show_knee: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = df["bpp"].astype(float)
    y = 1.0 - df["ms_ssim_y"].astype(float)
    q = df["q"].astype(int)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x, y, c=q, cmap="viridis", s=14, alpha=0.85)
    plt.xlabel("bpp (webp_bytes / (w*h))")
    plt.ylabel("1 - MS-SSIM (Y)")
    plt.title(title)
    plt.colorbar(sc, label="q")
    plt.grid(True, alpha=0.25)

    if show_pareto:
        front = pareto_front(df)
        plt.plot(front["bpp"], 1.0 - front["ms_ssim_y"], color="black", linewidth=1.2, alpha=0.8, label="pareto")
        if show_knee:
            knee = knee_by_max_distance(front)
            if knee is not None:
                plt.scatter([knee.bpp], [knee.err], marker="*", s=180, color="red", label=f"knee q={knee.q}")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

