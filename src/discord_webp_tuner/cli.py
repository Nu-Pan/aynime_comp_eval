from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from .config import SweepConfig
from .pipeline import sweep_dir
from .plot import (
    bpp_hist_by_q_plot,
    saturation_bpp_plot_gmsd_y,
    saturation_bpp_plot_ms_ssim_y,
    saturation_bpp_plot_ssimulacra2,
    saturation_plot_gmsd_y,
    saturation_plot_ms_ssim_y,
    saturation_plot_ssimulacra2,
    scatter_plot,
    scatter_plot_gmsd,
    scatter_plot_ssimulacra2,
)
from .plot_interactive import interactive_scatter_plot

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_metrics_csvs(*, csv_or_dir: Path, glob: str, dedupe: bool) -> pd.DataFrame:
    if csv_or_dir.is_dir():
        csv_paths = sorted([p for p in csv_or_dir.glob(glob) if p.is_file()])
        if not csv_paths:
            raise typer.BadParameter(f"No CSV files matched: dir={csv_or_dir} glob={glob!r}")
    else:
        csv_paths = [csv_or_dir]

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise typer.BadParameter(f"Failed to read CSV: {p} ({e})") from e
        df = df.copy()
        df["_source_csv"] = str(p.name)
        df["_source_mtime_ns"] = int(p.stat().st_mtime_ns)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True, sort=False) if len(dfs) > 1 else dfs[0]
    if not dedupe:
        return out

    # Prefer "newer" rows when the same (image,q,encoder) combo appears in multiple runs.
    key_cols = [
        "path_png",
        "long_edge",
        "eval_w",
        "eval_h",
        "q",
        "method",
        "sharp_yuv",
        "bg_color",
    ]
    subset = [c for c in key_cols if c in out.columns]
    sort_cols = [c for c in ["run_id", "_source_mtime_ns"] if c in out.columns]
    if subset:
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=True, na_position="first")
        out = out.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    return out


@app.command()
def sweep(
    in_dir: Annotated[Path, typer.Option("--in-dir", exists=True, file_okay=False, dir_okay=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    q_min: Annotated[int, typer.Option("--q-min")] = 60,
    q_max: Annotated[int, typer.Option("--q-max")] = 100,
    q_step: Annotated[int, typer.Option("--q-step")] = 5,
    long_edge: Annotated[int, typer.Option("--long-edge")] = 1280,
    bg_color: Annotated[str, typer.Option("--bg-color")] = "808080",
    sharp_yuv: Annotated[bool, typer.Option("--sharp-yuv")] = False,
    save_webp: Annotated[str, typer.Option("--save-webp")] = "best",
    jobs: Annotated[int, typer.Option("--jobs")] = 1,
    psnr: Annotated[bool, typer.Option("--psnr/--no-psnr")] = True,
    ssimulacra2: Annotated[bool, typer.Option("--ssimulacra2/--no-ssimulacra2")] = False,
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = True,
    bpp_hist: Annotated[bool, typer.Option("--bpp-hist/--no-bpp-hist")] = True,
    bpp_hist_bins: Annotated[int, typer.Option("--bpp-hist-bins")] = 50,
    bpp_hist_cols: Annotated[int, typer.Option("--bpp-hist-cols")] = 5,
    bpp_hist_x_quantile: Annotated[float, typer.Option("--bpp-hist-x-quantile")] = 0.995,
) -> None:
    config = SweepConfig(
        q_min=q_min,
        q_max=q_max,
        q_step=q_step,
        bg_color=bg_color,
        sharp_yuv=sharp_yuv,
        long_edge=int(long_edge),
        save_webp=save_webp,
        jobs=jobs,
    )

    csv_path = sweep_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        config=config,
        compute_psnr=psnr,
        compute_ssimulacra2=ssimulacra2,
    )
    typer.echo(f"wrote: {csv_path}")

    if plot:
        df = pd.read_csv(csv_path)
        if "long_edge" in df.columns:
            groups = [(int(le), df[df["long_edge"].astype(int) == int(le)]) for le in sorted(df["long_edge"].unique())]
        else:
            groups = [(int(long_edge), df)]

        for le, dfg in groups:
            if len(dfg) == 0:
                continue
            scope = f"le{le}"
            out_png = out_dir / "results" / f"scatter_ms_ssim_y_le{le}.png"
            out_gm_png = out_dir / "results" / f"scatter_gmsd_y_le{le}.png"
            out_s2_png = out_dir / "results" / f"scatter_ssimulacra2_le{le}.png"
            out_sat_ms_png = out_dir / "results" / f"saturation_q_ms_ssim_y_{scope}.png"
            out_sat_gm_png = out_dir / "results" / f"saturation_q_gmsd_y_{scope}.png"
            out_sat_s2_png = out_dir / "results" / f"saturation_q_ssimulacra2_{scope}.png"
            out_sat_bpp_ms_png = out_dir / "results" / f"saturation_bpp_ms_ssim_y_{scope}.png"
            out_sat_bpp_gm_png = out_dir / "results" / f"saturation_bpp_gmsd_y_{scope}.png"
            out_sat_bpp_s2_png = out_dir / "results" / f"saturation_bpp_ssimulacra2_{scope}.png"
            out_bpp_hist_png = out_dir / "results" / f"bpp_hist_by_q_le{le}.png"
            out_sat_csv = out_dir / "results" / f"saturation_ms_ssim_y_gmsd_y_le{le}_by_q.csv"
            scatter_plot(df=dfg, out_path=out_png, title=f"long_edge={le}", show_pareto=True, show_knee=True)
            typer.echo(f"wrote: {out_png}")
            if "gmsd" in dfg.columns:
                scatter_plot_gmsd(
                    df=dfg,
                    out_path=out_gm_png,
                    title=f"long_edge={le} GMSD",
                    show_pareto=True,
                    show_knee=True,
                )
                typer.echo(f"wrote: {out_gm_png}")
            if "ssimulacra2" in dfg.columns:
                scatter_plot_ssimulacra2(
                    df=dfg,
                    out_path=out_s2_png,
                    title=f"long_edge={le} SSIMULACRA2",
                    show_pareto=True,
                    show_knee=True,
                )
                typer.echo(f"wrote: {out_s2_png}")
            if bpp_hist:
                bpp_hist_by_q_plot(
                    df=dfg,
                    out_path=out_bpp_hist_png,
                    title=f"bpp histogram by q (long_edge={le})",
                    bins=int(bpp_hist_bins),
                    max_cols=int(bpp_hist_cols),
                    x_max_quantile=float(bpp_hist_x_quantile),
                )
                typer.echo(f"wrote: {out_bpp_hist_png}")
            sat = saturation_plot_ms_ssim_y(df=dfg, out_path=out_sat_ms_png, title=f"saturation MS-SSIM(Y) long_edge={le} (p10 + deltas)")
            sat.to_csv(out_sat_csv, index=False, encoding="utf-8")
            saturation_plot_gmsd_y(df=dfg, out_path=out_sat_gm_png, title=f"saturation GMSD(Y) long_edge={le} (p90 + deltas)")
            if "ssimulacra2" in dfg.columns:
                saturation_plot_ssimulacra2(
                    df=dfg,
                    out_path=out_sat_s2_png,
                    title=f"saturation SSIMULACRA2 long_edge={le} (p10 + deltas)",
                )
            saturation_bpp_plot_ms_ssim_y(
                df=dfg,
                out_path=out_sat_bpp_ms_png,
                title=f"saturation (bpp) MS-SSIM(Y) long_edge={le} (p10 + knee)",
            )
            saturation_bpp_plot_gmsd_y(
                df=dfg,
                out_path=out_sat_bpp_gm_png,
                title=f"saturation (bpp) GMSD(Y) long_edge={le} (p90 + knee)",
            )
            if "ssimulacra2" in dfg.columns:
                saturation_bpp_plot_ssimulacra2(
                    df=dfg,
                    out_path=out_sat_bpp_s2_png,
                    title=f"saturation (bpp) SSIMULACRA2 long_edge={le} (p10 + knee)",
                )
            typer.echo(f"wrote: {out_sat_ms_png}")
            typer.echo(f"wrote: {out_sat_gm_png}")
            if "ssimulacra2" in dfg.columns:
                typer.echo(f"wrote: {out_sat_s2_png}")
            typer.echo(f"wrote: {out_sat_bpp_ms_png}")
            typer.echo(f"wrote: {out_sat_bpp_gm_png}")
            if "ssimulacra2" in dfg.columns:
                typer.echo(f"wrote: {out_sat_bpp_s2_png}")
            typer.echo(f"wrote: {out_sat_csv}")


@app.command()
def plot(
    csv: Annotated[Path, typer.Option("--csv", exists=True, file_okay=True, dir_okay=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    show_pareto: Annotated[bool, typer.Option("--pareto/--no-pareto")] = True,
    show_knee: Annotated[bool, typer.Option("--knee/--no-knee")] = True,
    glob: Annotated[str, typer.Option("--glob")] = "metrics*.csv",
    dedupe: Annotated[bool, typer.Option("--dedupe/--no-dedupe")] = True,
    q_min: Annotated[Optional[int], typer.Option("--q-min")] = None,
    q_max: Annotated[Optional[int], typer.Option("--q-max")] = None,
    long_edge: Annotated[Optional[int], typer.Option("--long-edge")] = None,
    method: Annotated[Optional[int], typer.Option("--method")] = None,
    sharp_yuv: Annotated[Optional[int], typer.Option("--sharp-yuv")] = None,
    bg_color: Annotated[Optional[str], typer.Option("--bg-color")] = None,
    bpp_hist: Annotated[bool, typer.Option("--bpp-hist/--no-bpp-hist")] = True,
    bpp_hist_bins: Annotated[int, typer.Option("--bpp-hist-bins")] = 50,
    bpp_hist_cols: Annotated[int, typer.Option("--bpp-hist-cols")] = 5,
    bpp_hist_x_quantile: Annotated[float, typer.Option("--bpp-hist-x-quantile")] = 0.995,
) -> None:
    df = _load_metrics_csvs(csv_or_dir=csv, glob=glob, dedupe=dedupe)
    if q_min is not None:
        if "q" not in df.columns:
            raise typer.BadParameter(f"--q-min requires 'q' column in {csv}")
        df = df[df["q"].astype(int) >= int(q_min)]
    if q_max is not None:
        if "q" not in df.columns:
            raise typer.BadParameter(f"--q-max requires 'q' column in {csv}")
        df = df[df["q"].astype(int) <= int(q_max)]
    if long_edge is not None:
        if "long_edge" not in df.columns:
            raise typer.BadParameter(f"--long-edge requires 'long_edge' column in {csv}")
        df = df[df["long_edge"].astype(int) == int(long_edge)]
    if method is not None:
        if "method" not in df.columns:
            raise typer.BadParameter(f"--method requires 'method' column in {csv}")
        df = df[df["method"].astype(int) == int(method)]
    if sharp_yuv is not None:
        if "sharp_yuv" not in df.columns:
            raise typer.BadParameter(f"--sharp-yuv requires 'sharp_yuv' column in {csv}")
        df = df[df["sharp_yuv"].astype(int) == int(sharp_yuv)]
    if bg_color is not None:
        if "bg_color" not in df.columns:
            raise typer.BadParameter(f"--bg-color requires 'bg_color' column in {csv}")
        df = df[df["bg_color"].astype(str) == str(bg_color).strip().lstrip("#")]

    if len(df) == 0:
        raise typer.BadParameter(f"No rows after filtering in {csv}")

    if "long_edge" in df.columns:
        groups = [(int(le), df[df["long_edge"].astype(int) == int(le)]) for le in sorted(df["long_edge"].unique())]
    else:
        groups = [(0, df)]

    for le, dfg in groups:
        if len(dfg) == 0:
            continue
        label = (f"le{le}" if le else "all")
        out_png = out_dir / f"scatter_ms_ssim_y_{label}.png"
        scatter_plot(df=dfg, out_path=out_png, title=label, show_pareto=show_pareto, show_knee=show_knee)
        typer.echo(f"wrote: {out_png}")
        if "gmsd" in dfg.columns:
            out_gm_png = out_dir / f"scatter_gmsd_y_{label}.png"
            scatter_plot_gmsd(df=dfg, out_path=out_gm_png, title=f"{label} GMSD", show_pareto=show_pareto, show_knee=show_knee)
            typer.echo(f"wrote: {out_gm_png}")
        if "ssimulacra2" in dfg.columns:
            out_s2_png = out_dir / f"scatter_ssimulacra2_{label}.png"
            scatter_plot_ssimulacra2(
                df=dfg,
                out_path=out_s2_png,
                title=f"{label} SSIMULACRA2",
                show_pareto=show_pareto,
                show_knee=show_knee,
            )
            typer.echo(f"wrote: {out_s2_png}")

        if bpp_hist:
            out_bpp_hist_png = out_dir / f"bpp_hist_by_q_{label}.png"
            bpp_hist_by_q_plot(
                df=dfg,
                out_path=out_bpp_hist_png,
                title=f"bpp histogram by q ({label})",
                bins=int(bpp_hist_bins),
                max_cols=int(bpp_hist_cols),
                x_max_quantile=float(bpp_hist_x_quantile),
            )
            typer.echo(f"wrote: {out_bpp_hist_png}")

        out_sat_csv = out_dir / f"saturation_ms_ssim_y_gmsd_y_{label}_by_q.csv"
        out_sat_ms_png = out_dir / f"saturation_q_ms_ssim_y_{label}.png"
        out_sat_gm_png = out_dir / f"saturation_q_gmsd_y_{label}.png"
        out_sat_s2_png = out_dir / f"saturation_q_ssimulacra2_{label}.png"
        sat = saturation_plot_ms_ssim_y(df=dfg, out_path=out_sat_ms_png, title=f"saturation MS-SSIM(Y) {label} (p10 + deltas)")
        sat.to_csv(out_sat_csv, index=False, encoding="utf-8")
        saturation_plot_gmsd_y(df=dfg, out_path=out_sat_gm_png, title=f"saturation GMSD(Y) {label} (p90 + deltas)")
        if "ssimulacra2" in dfg.columns:
            saturation_plot_ssimulacra2(df=dfg, out_path=out_sat_s2_png, title=f"saturation SSIMULACRA2 {label} (p10 + deltas)")
        out_sat_bpp_ms_png = out_dir / f"saturation_bpp_ms_ssim_y_{label}.png"
        out_sat_bpp_gm_png = out_dir / f"saturation_bpp_gmsd_y_{label}.png"
        out_sat_bpp_s2_png = out_dir / f"saturation_bpp_ssimulacra2_{label}.png"
        saturation_bpp_plot_ms_ssim_y(df=dfg, out_path=out_sat_bpp_ms_png, title=f"saturation (bpp) MS-SSIM(Y) {label} (p10 + knee)")
        saturation_bpp_plot_gmsd_y(df=dfg, out_path=out_sat_bpp_gm_png, title=f"saturation (bpp) GMSD(Y) {label} (p90 + knee)")
        if "ssimulacra2" in dfg.columns:
            saturation_bpp_plot_ssimulacra2(df=dfg, out_path=out_sat_bpp_s2_png, title=f"saturation (bpp) SSIMULACRA2 {label} (p10 + knee)")
        typer.echo(f"wrote: {out_sat_ms_png}")
        typer.echo(f"wrote: {out_sat_gm_png}")
        if "ssimulacra2" in dfg.columns:
            typer.echo(f"wrote: {out_sat_s2_png}")
        typer.echo(f"wrote: {out_sat_bpp_ms_png}")
        typer.echo(f"wrote: {out_sat_bpp_gm_png}")
        if "ssimulacra2" in dfg.columns:
            typer.echo(f"wrote: {out_sat_bpp_s2_png}")
        typer.echo(f"wrote: {out_sat_csv}")


@app.command("plot-interactive")
def plot_interactive(
    csv: Annotated[Path, typer.Option("--csv", exists=True, file_okay=True, dir_okay=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    show_pareto: Annotated[bool, typer.Option("--pareto/--no-pareto")] = True,
    show_knee: Annotated[bool, typer.Option("--knee/--no-knee")] = True,
    sample_per_q: Annotated[int, typer.Option("--sample-per-q")] = 0,
    glob: Annotated[str, typer.Option("--glob")] = "metrics*.csv",
    dedupe: Annotated[bool, typer.Option("--dedupe/--no-dedupe")] = True,
) -> None:
    df = _load_metrics_csvs(csv_or_dir=csv, glob=glob, dedupe=dedupe)
    if len(df) == 0:
        raise typer.BadParameter(f"No rows in {csv}")

    if "long_edge" in df.columns:
        groups = [(int(le), df[df["long_edge"].astype(int) == int(le)]) for le in sorted(df["long_edge"].unique())]
    else:
        groups = [(0, df)]

    for le, dfg in groups:
        if len(dfg) == 0:
            continue
        label = (f"le{le}" if le else "all")
        out_html = out_dir / f"scatter_ms_ssim_y_{label}.html"
        try:
            interactive_scatter_plot(
                df=dfg,
                out_path=out_html,
                title=label,
                show_pareto=show_pareto,
                show_knee=show_knee,
                sample_per_q=sample_per_q,
            )
        except ModuleNotFoundError as e:
            raise typer.BadParameter(f"{e} (csv={csv})") from e
        typer.echo(f"wrote: {out_html}")


if __name__ == "__main__":
    app()
