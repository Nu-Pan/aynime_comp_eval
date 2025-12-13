from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


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


def summarize_by_q(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bpp"] = d["bpp"].astype(float)
    d["ms_ssim_y"] = d["ms_ssim_y"].astype(float)
    d["q"] = d["q"].astype(int)
    d["err"] = 1.0 - d["ms_ssim_y"]

    def qtile(p: float):
        return lambda s: float(np.quantile(s.to_numpy(dtype=float), p))

    s = (
        d.groupby("q", as_index=False)
        .agg(
            bpp=("bpp", "median"),
            ms_ssim_y=("ms_ssim_y", "median"),
            err=("err", "median"),
            n=("err", "size"),
            bpp_p10=("bpp", qtile(0.10)),
            bpp_p90=("bpp", qtile(0.90)),
            err_p10=("err", qtile(0.10)),
            err_p90=("err", qtile(0.90)),
        )
        .sort_values("q", ascending=True)
        .reset_index(drop=True)
    )
    return s


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
    summary = summarize_by_q(df)
    q_min = int(summary["q"].min()) if len(summary) else int(df["q"].min())
    q_max = int(summary["q"].max()) if len(summary) else int(df["q"].max())
    norm = Normalize(vmin=q_min, vmax=q_max)

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.hexbin(
        x.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        gridsize=140,
        bins="log",
        mincnt=1,
        linewidths=0.0,
        cmap="Greys",
        alpha=0.10,
        zorder=1,
        rasterized=True,
    )
    ax.hexbin(
        x.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        C=df["q"].to_numpy(dtype=float),
        reduce_C_function=np.median,
        gridsize=140,
        mincnt=5,
        linewidths=0.0,
        cmap="viridis",
        norm=norm,
        alpha=0.95,
        zorder=2,
        rasterized=True,
    )
    sc = ax.scatter(
        summary["bpp"].astype(float),
        summary["err"].astype(float),
        c=summary["q"].astype(int),
        cmap="viridis",
        norm=norm,
        s=40,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.25,
        zorder=4,
    )
    ax.vlines(
        summary["bpp"].astype(float),
        summary["err_p10"].astype(float),
        summary["err_p90"].astype(float),
        colors="black",
        alpha=0.18,
        linewidth=0.8,
        zorder=3,
    )
    ax.hlines(
        summary["err"].astype(float),
        summary["bpp_p10"].astype(float),
        summary["bpp_p90"].astype(float),
        colors="black",
        alpha=0.10,
        linewidth=0.8,
        zorder=3,
    )
    ax.plot(
        summary["bpp"].astype(float),
        summary["err"].astype(float),
        color="black",
        linewidth=1.0,
        alpha=0.25,
        zorder=3,
    )
    plt.xlabel("bpp (webp_bytes / (w*h))")
    plt.ylabel("1 - MS-SSIM (Y)")
    plt.title(title)
    plt.colorbar(sc, label="q")
    plt.grid(True, alpha=0.25)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    try:
        x_hi = float(np.quantile(x.to_numpy(dtype=float), 0.995))
        y_hi = float(np.quantile(y.to_numpy(dtype=float), 0.995))
        if x_hi > 0:
            plt.xlim(0.0, x_hi * 1.02)
        if y_hi > 0:
            plt.ylim(0.0, y_hi * 1.10)
    except Exception:
        pass

    knee: KneePoint | None = None
    if show_pareto:
        front = pareto_front(summary)
        plt.plot(
            front["bpp"],
            1.0 - front["ms_ssim_y"],
            color="black",
            linewidth=1.2,
            alpha=0.8,
            label="pareto (q median)",
        )
        if show_knee:
            knee = knee_by_max_distance(front)
            if knee is not None:
                plt.scatter([knee.bpp], [knee.err], marker="*", s=180, color="red", label=f"knee q={knee.q}")

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black", alpha=0.25, markersize=8, label="density (per-image)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=plt.cm.viridis(0.65),
            alpha=0.45,
            markersize=8,
            label="q in bins (median)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=7,
            label="q median (color=q)",
        ),
    ]
    labels += ["density (per-image)", "q in bins (median)", "q median (color=q)"]
    ax.legend(handles=handles, labels=labels, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
