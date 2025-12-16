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


@dataclass(frozen=True)
class KneePointXY:
    q: int
    bpp: float
    y: float


@dataclass(frozen=True)
class SaturationSummary:
    q: int
    bpp_median: float
    ms_ssim_y_p10: float
    gmsd_p90: float
    delta_ms_ssim_y_p10: float | None
    delta_gmsd_p90_improve: float | None
    delta_bpp_median: float | None


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


def knee_index_by_max_distance_xy(x: np.ndarray, y: np.ndarray) -> int | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape (got x={x.shape}, y={y.shape})")

    mask = np.isfinite(x) & np.isfinite(y)
    idxs = np.flatnonzero(mask)
    if idxs.size < 3:
        return None

    xs = x[idxs]
    ys = y[idxs]

    x0, y0 = float(xs[0]), float(ys[0])
    x1, y1 = float(xs[-1]), float(ys[-1])
    dx = x1 - x0
    dy = y1 - y0
    denom = float((dx * dx + dy * dy) ** 0.5)
    if denom <= 0.0:
        return None

    dist = np.abs(dy * xs - dx * ys + x1 * y0 - y1 * x0) / denom
    i_local = int(dist.argmax())
    return int(idxs[i_local])


def summarize_by_q(df: pd.DataFrame, *, ms_ssim_y_quantile: float = 0.50) -> pd.DataFrame:
    d = df.copy()
    d["bpp"] = d["bpp"].astype(float)
    d["ms_ssim_y"] = d["ms_ssim_y"].astype(float)
    d["q"] = d["q"].astype(int)
    if not (0.0 <= float(ms_ssim_y_quantile) <= 1.0):
        raise ValueError(f"ms_ssim_y_quantile must be in [0,1] (got {ms_ssim_y_quantile})")

    def qtile(p: float):
        return lambda s: float(np.quantile(s.to_numpy(dtype=float), p))

    s = (
        d.groupby("q", as_index=False)
        .agg(
            bpp=("bpp", "median"),
            ms_ssim_y=("ms_ssim_y", qtile(float(ms_ssim_y_quantile))),
            ms_ssim_y_p10=("ms_ssim_y", qtile(0.10)),
            ms_ssim_y_p90=("ms_ssim_y", qtile(0.90)),
            n=("ms_ssim_y", "size"),
            bpp_p10=("bpp", qtile(0.10)),
            bpp_p90=("bpp", qtile(0.90)),
        )
        .sort_values("q", ascending=True)
        .reset_index(drop=True)
    )
    s["err"] = 1.0 - s["ms_ssim_y"].astype(float)
    s["err_p10"] = 1.0 - s["ms_ssim_y_p90"].astype(float)
    s["err_p90"] = 1.0 - s["ms_ssim_y_p10"].astype(float)
    return s


def summarize_saturation_by_q(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bpp"] = d["bpp"].astype(float)
    d["ms_ssim_y"] = d["ms_ssim_y"].astype(float)
    d["gmsd"] = d["gmsd"].astype(float)
    d["q"] = d["q"].astype(int)

    def qtile(p: float):
        return lambda s: float(np.quantile(s.to_numpy(dtype=float), p))

    s = (
        d.groupby("q", as_index=False)
        .agg(
            bpp_median=("bpp", "median"),
            ms_ssim_y_p10=("ms_ssim_y", qtile(0.10)),
            gmsd_p90=("gmsd", qtile(0.90)),
            n=("q", "size"),
        )
        .sort_values("q", ascending=True)
        .reset_index(drop=True)
    )

    # Deltas: ms_ssim_y higher is better; gmsd lower is better.
    s["delta_q"] = s["q"].diff()
    s["delta_ms_ssim_y_p10"] = s["ms_ssim_y_p10"].diff()
    s["delta_gmsd_p90_improve"] = (-s["gmsd_p90"]).diff()
    s["delta_bpp_median"] = s["bpp_median"].diff()

    dq = s["delta_q"].to_numpy(dtype=float)
    s["delta_ms_ssim_y_p10_per_q"] = np.where(dq > 0, s["delta_ms_ssim_y_p10"].to_numpy(dtype=float) / dq, np.nan)
    s["delta_gmsd_p90_improve_per_q"] = np.where(dq > 0, s["delta_gmsd_p90_improve"].to_numpy(dtype=float) / dq, np.nan)
    s["delta_bpp_median_per_q"] = np.where(dq > 0, s["delta_bpp_median"].to_numpy(dtype=float) / dq, np.nan)

    bpp_m = s["bpp_median"].to_numpy(dtype=float)
    s["ms_ssim_y_p10_over_bpp_median"] = np.where(bpp_m > 0, s["ms_ssim_y_p10"].to_numpy(dtype=float) / bpp_m, np.nan)
    s["gmsd_p90_over_bpp_median"] = np.where(bpp_m > 0, s["gmsd_p90"].to_numpy(dtype=float) / bpp_m, np.nan)
    s["delta_ms_ssim_y_p10_per_bpp"] = np.where(
        s["delta_bpp_median"].to_numpy(dtype=float) > 0,
        s["delta_ms_ssim_y_p10"].to_numpy(dtype=float) / s["delta_bpp_median"].to_numpy(dtype=float),
        np.nan,
    )
    s["delta_gmsd_p90_improve_per_bpp"] = np.where(
        s["delta_bpp_median"].to_numpy(dtype=float) > 0,
        s["delta_gmsd_p90_improve"].to_numpy(dtype=float) / s["delta_bpp_median"].to_numpy(dtype=float),
        np.nan,
    )
    return s


def bpp_hist_by_q_plot(
    *,
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    bins: int = 50,
    max_cols: int = 5,
    x_max_quantile: float = 0.995,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bins <= 0:
        raise ValueError(f"bins must be > 0 (got {bins})")
    if max_cols <= 0:
        raise ValueError(f"max_cols must be > 0 (got {max_cols})")
    if not (0.0 < float(x_max_quantile) <= 1.0):
        raise ValueError(f"x_max_quantile must be in (0,1] (got {x_max_quantile})")

    d = df.copy()
    if "q" not in d.columns or "bpp" not in d.columns:
        raise ValueError("bpp_hist_by_q_plot requires columns: q, bpp")

    d["q"] = d["q"].astype(int)
    d["bpp"] = d["bpp"].astype(float)
    d = d[np.isfinite(d["bpp"].to_numpy(dtype=float))]

    q_vals = sorted(d["q"].unique().tolist())
    if not q_vals:
        raise ValueError("No rows to plot for bpp histogram")

    all_bpp = d["bpp"].to_numpy(dtype=float)
    try:
        x_max = float(np.quantile(all_bpp, float(x_max_quantile)))
    except Exception:
        x_max = float(np.max(all_bpp)) if len(all_bpp) else 0.0
    x_max = max(0.0, x_max)
    if x_max <= 0.0:
        x_max = float(np.max(all_bpp)) if len(all_bpp) else 0.0

    # Use shared bin edges so shapes are comparable across q.
    if x_max > 0:
        edges = np.linspace(0.0, x_max, int(bins) + 1)
    else:
        edges = int(bins)

    n = len(q_vals)
    ncols = int(min(max_cols, n))
    nrows = int((n + ncols - 1) // ncols)

    fig_w = 3.2 * ncols
    fig_h = 2.6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes_arr.ravel():
        ax.grid(True, alpha=0.20)

    for i, q in enumerate(q_vals):
        ax = axes_arr.ravel()[i]
        bpp_q = d.loc[d["q"] == int(q), "bpp"].to_numpy(dtype=float)
        if x_max > 0:
            mask = bpp_q <= x_max
            clipped = bpp_q[mask]
            over = int((~mask).sum())
        else:
            clipped = bpp_q
            over = 0

        ax.hist(clipped, bins=edges, color="#4c78a8", alpha=0.85, edgecolor="none")
        med = float(np.median(bpp_q)) if len(bpp_q) else float("nan")
        if np.isfinite(med):
            ax.axvline(min(med, x_max) if x_max > 0 else med, color="black", linewidth=1.0, alpha=0.75)
        ax.set_title(f"q={int(q)} (n={len(bpp_q)})", fontsize=10)
        if over > 0 and x_max > 0:
            ax.annotate(
                f"+{over} > {x_max:.4f}",
                xy=(0.98, 0.90),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                alpha=0.75,
            )

    # Turn off any unused axes.
    for j in range(n, nrows * ncols):
        axes_arr.ravel()[j].axis("off")

    for ax in axes_arr[-1, :]:
        ax.set_xlabel("bpp")
    for ax in axes_arr[:, 0]:
        ax.set_ylabel("count")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def saturation_plot(*, df: pd.DataFrame, out_path: Path, title: str) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = summarize_saturation_by_q(df)
    if len(s) == 0:
        raise ValueError("No rows to summarize for saturation plot")

    q = s["q"].to_numpy(dtype=int)
    ms_p10 = s["ms_ssim_y_p10"].to_numpy(dtype=float)
    gms_p90 = s["gmsd_p90"].to_numpy(dtype=float)
    d_ms = s["delta_ms_ssim_y_p10_per_q"].to_numpy(dtype=float)
    d_gms = s["delta_gmsd_p90_improve_per_q"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex="col")
    ax_ms, ax_dms = axes[0, 0], axes[0, 1]
    ax_gm, ax_dgm = axes[1, 0], axes[1, 1]

    ax_ms.plot(q, ms_p10, color="black", linewidth=1.4)
    ax_ms.set_title("p10(ms_ssim_y) (higher=better)")
    ax_ms.set_ylabel("p10(ms_ssim_y)")
    ax_ms.grid(True, alpha=0.25)

    ax_dms.bar(q[1:], d_ms[1:], color="#4c78a8", alpha=0.85)
    ax_dms.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_dms.set_title("d p10(ms_ssim_y) / dq (higher=better)")
    ax_dms.set_ylabel("d p10(ms_ssim_y) / dq")
    ax_dms.grid(True, alpha=0.25)

    ax_gm.plot(q, gms_p90, color="black", linewidth=1.4)
    ax_gm.set_title("p90(gmsd) (lower=better)")
    ax_gm.set_xlabel("q")
    ax_gm.set_ylabel("p90(gmsd)")
    ax_gm.grid(True, alpha=0.25)

    ax_dgm.bar(q[1:], d_gms[1:], color="#f58518", alpha=0.85)
    ax_dgm.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_dgm.set_title("d p90(gmsd) improvement / dq (positive=better)")
    ax_dgm.set_xlabel("q")
    ax_dgm.set_ylabel("d (-p90(gmsd)) / dq")
    ax_dgm.grid(True, alpha=0.25)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return s


def saturation_bpp_plot(*, df: pd.DataFrame, out_path: Path, title: str) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = summarize_saturation_by_q(df)
    if len(s) == 0:
        raise ValueError("No rows to summarize for saturation plot")

    q = s["q"].to_numpy(dtype=int)
    bpp = s["bpp_median"].to_numpy(dtype=float)
    ms_p10 = s["ms_ssim_y_p10"].to_numpy(dtype=float)
    gms_p90 = s["gmsd_p90"].to_numpy(dtype=float)
    d_ms_per_bpp = s["delta_ms_ssim_y_p10_per_bpp"].to_numpy(dtype=float)
    d_gms_per_bpp = s["delta_gmsd_p90_improve_per_bpp"].to_numpy(dtype=float)
    d_bpp = s["delta_bpp_median"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex="col")
    ax_ms, ax_dms = axes[0, 0], axes[0, 1]
    ax_gm, ax_dgm = axes[1, 0], axes[1, 1]

    ax_ms.plot(bpp, ms_p10, color="black", linewidth=1.4, marker="o", markersize=3.8)
    ax_ms.set_title("p10(ms_ssim_y) vs bpp (higher=better)")
    ax_ms.set_ylabel("p10(ms_ssim_y)")
    ax_ms.grid(True, alpha=0.25)

    widths = np.maximum(d_bpp[1:], 0.0)
    ax_dms.bar(bpp[:-1], d_ms_per_bpp[1:], color="#4c78a8", alpha=0.85, width=widths, align="edge")
    ax_dms.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_dms.set_title("d p10(ms_ssim_y) / d bpp (higher=better)")
    ax_dms.set_ylabel("d p10(ms_ssim_y) / d bpp")
    ax_dms.grid(True, alpha=0.25)

    ax_gm.plot(bpp, gms_p90, color="black", linewidth=1.4, marker="o", markersize=3.8)
    ax_gm.set_title("p90(gmsd) vs bpp (lower=better)")
    ax_gm.set_xlabel("bpp (median by q)")
    ax_gm.set_ylabel("p90(gmsd)")
    ax_gm.grid(True, alpha=0.25)

    ax_dgm.bar(bpp[:-1], d_gms_per_bpp[1:], color="#f58518", alpha=0.85, width=widths, align="edge")
    ax_dgm.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_dgm.set_title("d p90(gmsd) improvement / d bpp (higher=better)")
    ax_dgm.set_xlabel("bpp (median by q)")
    ax_dgm.set_ylabel("d (-p90(gmsd)) / d bpp")
    ax_dgm.grid(True, alpha=0.25)

    knee_ms_idx = knee_index_by_max_distance_xy(bpp, ms_p10)
    knee_gm_idx = knee_index_by_max_distance_xy(bpp, gms_p90)
    knee_ms: KneePointXY | None = None
    knee_gm: KneePointXY | None = None
    if knee_ms_idx is not None:
        knee_ms = KneePointXY(q=int(q[knee_ms_idx]), bpp=float(bpp[knee_ms_idx]), y=float(ms_p10[knee_ms_idx]))
    if knee_gm_idx is not None:
        knee_gm = KneePointXY(q=int(q[knee_gm_idx]), bpp=float(bpp[knee_gm_idx]), y=float(gms_p90[knee_gm_idx]))

    if knee_ms is not None:
        for ax in (ax_ms, ax_dms, ax_gm, ax_dgm):
            ax.axvline(knee_ms.bpp, color="#e45756", linewidth=1.1, alpha=0.75, linestyle="--")
        ax_ms.scatter([knee_ms.bpp], [knee_ms.y], marker="*", s=180, color="#e45756", zorder=5)
        ax_ms.annotate(f"knee q={knee_ms.q}", (knee_ms.bpp, knee_ms.y), textcoords="offset points", xytext=(6, -10), fontsize=9, color="#e45756")

    if knee_gm is not None:
        for ax in (ax_ms, ax_dms, ax_gm, ax_dgm):
            ax.axvline(knee_gm.bpp, color="#b279a2", linewidth=1.1, alpha=0.75, linestyle=":")
        ax_gm.scatter([knee_gm.bpp], [knee_gm.y], marker="*", s=180, color="#b279a2", zorder=5)
        ax_gm.annotate(f"knee q={knee_gm.q}", (knee_gm.bpp, knee_gm.y), textcoords="offset points", xytext=(6, -10), fontsize=9, color="#b279a2")

    for ax in (ax_ms, ax_gm):
        for bx, by, qq in zip(bpp, ax.lines[0].get_ydata(), q, strict=True):
            ax.annotate(str(int(qq)), (float(bx), float(by)), textcoords="offset points", xytext=(4, 2), fontsize=8, alpha=0.65)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
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
    summary = summarize_by_q(df, ms_ssim_y_quantile=0.10)
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
            label="pareto (q p10)",
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
            label="q p10 (color=q)",
        ),
    ]
    labels += ["density (per-image)", "q in bins (median)", "q p10 (color=q)"]
    ax.legend(handles=handles, labels=labels, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
