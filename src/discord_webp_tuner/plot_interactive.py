from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plot import knee_by_max_distance, pareto_front, summarize_by_q


def _require_plotly() -> tuple[object, object]:
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        import plotly.io as pio  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Plotly is required for interactive plots. Install with: pip install -e '.[interactive]'"
        ) from e
    return go, pio


def interactive_scatter_plot(
    *,
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    show_pareto: bool = True,
    show_knee: bool = True,
    sample_per_q: int = 0,
    random_seed: int = 0,
) -> None:
    go, pio = _require_plotly()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["bpp"] = d["bpp"].astype(float)
    d["ms_ssim_y"] = d["ms_ssim_y"].astype(float)
    d["q"] = d["q"].astype(int)
    d["err"] = 1.0 - d["ms_ssim_y"]

    if sample_per_q and sample_per_q > 0:
        sizes = d.groupby("q")["q"].size()
        big = sizes[sizes > sample_per_q].index
        d_big = d[d["q"].isin(big)].groupby("q", group_keys=False).sample(n=sample_per_q, random_state=random_seed)
        d_small = d[~d["q"].isin(big)]
        d = pd.concat([d_big, d_small], ignore_index=True)

    summary = summarize_by_q(df)
    q_vals = sorted(d["q"].unique().tolist())
    q_min = int(min(q_vals)) if q_vals else int(summary["q"].min())
    q_max = int(max(q_vals)) if q_vals else int(summary["q"].max())

    try:
        x_hi = float(np.quantile(d["bpp"].to_numpy(dtype=float), 0.995))
        y_hi = float(np.quantile(d["err"].to_numpy(dtype=float), 0.995))
    except Exception:
        x_hi, y_hi = 0.0, 0.0

    colorscale = "Viridis"

    fig = go.Figure()

    # Per-q traces: legend toggles show/hide
    for q in q_vals:
        dq = d[d["q"] == q]
        fig.add_trace(
            go.Scattergl(
                x=dq["bpp"],
                y=dq["err"],
                mode="markers",
                name=f"q={q}",
                legendgroup=f"q={q}",
                marker=dict(
                    size=6,
                    opacity=0.60,
                    color=q,
                    colorscale=colorscale,
                    cmin=q_min,
                    cmax=q_max,
                    line=dict(width=0.2, color="rgba(0,0,0,0.15)"),
                ),
                hovertemplate="bpp=%{x:.4f}<br>err=%{y:.4f}<br>q=%{marker.color}<extra></extra>",
            )
        )

    # Summary (median) with percentile ranges
    err_minus = (summary["err"] - summary["err_p10"]).astype(float)
    err_plus = (summary["err_p90"] - summary["err"]).astype(float)
    bpp_minus = (summary["bpp"] - summary["bpp_p10"]).astype(float)
    bpp_plus = (summary["bpp_p90"] - summary["bpp"]).astype(float)
    fig.add_trace(
        go.Scatter(
            x=summary["bpp"].astype(float),
            y=summary["err"].astype(float),
            mode="lines+markers",
            name="q median",
            marker=dict(
                size=8,
                color=summary["q"].astype(int),
                colorscale=colorscale,
                cmin=q_min,
                cmax=q_max,
                line=dict(color="black", width=0.6),
            ),
            line=dict(color="rgba(0,0,0,0.35)", width=2),
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, thickness=1, width=0),
            error_x=dict(type="data", symmetric=False, array=bpp_plus, arrayminus=bpp_minus, thickness=1, width=0),
            hovertemplate="bpp=%{x:.4f}<br>err=%{y:.4f}<br>q=%{marker.color}<extra></extra>",
        )
    )

    knee = None
    if show_pareto:
        front = pareto_front(summary)
        fig.add_trace(
            go.Scatter(
                x=front["bpp"].astype(float),
                y=(1.0 - front["ms_ssim_y"].astype(float)),
                mode="lines",
                name="pareto (q median)",
                line=dict(color="black", width=2),
                hoverinfo="skip",
            )
        )
        if show_knee:
            knee = knee_by_max_distance(front)
            if knee is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[knee.bpp],
                        y=[knee.err],
                        mode="markers",
                        name=f"knee q={knee.q}",
                        marker=dict(symbol="star", size=16, color="red", line=dict(color="black", width=0.7)),
                        hovertemplate="bpp=%{x:.4f}<br>err=%{y:.4f}<extra></extra>",
                    )
                )

    per_q_count = len(q_vals)
    median_idx = per_q_count
    pareto_idx = per_q_count + 1
    knee_idx = per_q_count + 2

    vis_all = [True] * len(fig.data)
    vis_median_only = [False] * len(fig.data)
    if len(fig.data) > median_idx:
        vis_median_only[median_idx] = True
    if show_pareto and len(fig.data) > pareto_idx:
        vis_median_only[pareto_idx] = True
    if show_pareto and show_knee and knee is not None and len(fig.data) > knee_idx:
        vis_median_only[knee_idx] = True

    fig.update_layout(
        title=title,
        xaxis_title="bpp (webp_bytes / (w*h))",
        yaxis_title="1 - MS-SSIM (Y)",
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=60, r=20, t=60, b=55),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="All q", method="update", args=[{"visible": vis_all}]),
                    dict(label="Median/Pareto only", method="update", args=[{"visible": vis_median_only}]),
                ],
            )
        ],
    )
    fig.update_xaxes(rangemode="tozero", range=[0, x_hi * 1.02] if x_hi > 0 else None)
    fig.update_yaxes(rangemode="tozero", range=[0, y_hi * 1.10] if y_hi > 0 else None)

    pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn", full_html=True, auto_open=False)
