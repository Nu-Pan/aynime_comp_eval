from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from .config import SweepConfig, parse_target_long_edges
from .pipeline import sweep_dir
from .plot import scatter_plot
from .plot_interactive import interactive_scatter_plot

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def sweep(
    in_dir: Annotated[Path, typer.Option("--in-dir", exists=True, file_okay=False, dir_okay=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    q_min: Annotated[int, typer.Option("--q-min")] = 30,
    q_max: Annotated[int, typer.Option("--q-max")] = 80,
    q_step: Annotated[int, typer.Option("--q-step")] = 5,
    target_long_edge: Annotated[Optional[list[str]], typer.Option("--target-long-edge")] = None,
    bg_color: Annotated[str, typer.Option("--bg-color")] = "808080",
    sharp_yuv: Annotated[bool, typer.Option("--sharp-yuv")] = False,
    save_webp: Annotated[str, typer.Option("--save-webp")] = "best",
    jobs: Annotated[int, typer.Option("--jobs")] = 1,
    psnr: Annotated[bool, typer.Option("--psnr/--no-psnr")] = True,
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = True,
) -> None:
    targets = parse_target_long_edges(target_long_edge)
    config = SweepConfig(
        q_min=q_min,
        q_max=q_max,
        q_step=q_step,
        bg_color=bg_color,
        sharp_yuv=sharp_yuv,
        targets=targets,
        save_webp=save_webp,
        jobs=jobs,
    )

    csv_path = sweep_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        config=config,
        compute_psnr=psnr,
    )
    typer.echo(f"wrote: {csv_path}")

    if plot:
        df = pd.read_csv(csv_path)
        for t in targets:
            dft = df[df["target"] == t.name]
            if len(dft) == 0:
                continue
            out_png = out_dir / "results" / f"scatter_target{t.name}.png"
            scatter_plot(
                df=dft,
                out_path=out_png,
                title=f"target {t.name} (long_edge={t.long_edge})",
                show_pareto=True,
                show_knee=True,
            )
            typer.echo(f"wrote: {out_png}")


@app.command()
def plot(
    csv: Annotated[Path, typer.Option("--csv", exists=True, file_okay=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    target: Annotated[str, typer.Option("--target")] = "A",
    show_pareto: Annotated[bool, typer.Option("--pareto/--no-pareto")] = True,
    show_knee: Annotated[bool, typer.Option("--knee/--no-knee")] = True,
) -> None:
    df = pd.read_csv(csv)
    dft = df[df["target"] == target]
    if len(dft) == 0:
        raise typer.BadParameter(f"No rows for target={target!r} in {csv}")
    out_png = out_dir / f"scatter_target{target}.png"
    scatter_plot(df=dft, out_path=out_png, title=f"target {target}", show_pareto=show_pareto, show_knee=show_knee)
    typer.echo(f"wrote: {out_png}")


@app.command("plot-interactive")
def plot_interactive(
    csv: Annotated[Path, typer.Option("--csv", exists=True, file_okay=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option("--out-dir")],
    target: Annotated[str, typer.Option("--target")] = "A",
    show_pareto: Annotated[bool, typer.Option("--pareto/--no-pareto")] = True,
    show_knee: Annotated[bool, typer.Option("--knee/--no-knee")] = True,
    sample_per_q: Annotated[int, typer.Option("--sample-per-q")] = 0,
) -> None:
    df = pd.read_csv(csv)
    dft = df[df["target"] == target]
    if len(dft) == 0:
        raise typer.BadParameter(f"No rows for target={target!r} in {csv}")
    out_html = out_dir / f"scatter_target{target}.html"
    try:
        interactive_scatter_plot(
            df=dft,
            out_path=out_html,
            title=f"target {target}",
            show_pareto=show_pareto,
            show_knee=show_knee,
            sample_per_q=sample_per_q,
        )
    except ModuleNotFoundError as e:
        raise typer.BadParameter(f"{e} (csv={csv})") from e
    typer.echo(f"wrote: {out_html}")


if __name__ == "__main__":
    app()
