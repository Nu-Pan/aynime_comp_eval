import os

import numpy as np
import pandas as pd


def test_summarize_by_q_accepts_quantile():
    from discord_webp_tuner.plot import summarize_by_q

    df = pd.DataFrame(
        {
            "q": [70, 70, 70, 80, 80, 80],
            "bpp": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "ms_ssim_y": [0.90, 0.95, 0.99, 0.91, 0.96, 0.98],
        }
    )
    s50 = summarize_by_q(df, ms_ssim_y_quantile=0.50)
    s10 = summarize_by_q(df, ms_ssim_y_quantile=0.10)

    assert list(s50["q"].astype(int)) == [70, 80]
    assert list(s10["q"].astype(int)) == [70, 80]

    # With 3 samples, p10 is close to the minimum; p50 is the median.
    assert float(s50.loc[0, "ms_ssim_y"]) == 0.95
    assert float(s50.loc[1, "ms_ssim_y"]) == 0.96

    assert float(s10.loc[0, "ms_ssim_y"]) < float(s50.loc[0, "ms_ssim_y"])
    assert float(s10.loc[1, "ms_ssim_y"]) < float(s50.loc[1, "ms_ssim_y"])

    assert np.isclose(float(s10.loc[0, "err"]), 1.0 - float(s10.loc[0, "ms_ssim_y"]))
    assert np.isclose(float(s10.loc[1, "err"]), 1.0 - float(s10.loc[1, "ms_ssim_y"]))


def test_summarize_saturation_by_q_adds_efficiency_columns():
    os.environ.setdefault("MPLBACKEND", "Agg")

    from discord_webp_tuner.plot import summarize_saturation_by_q

    df = pd.DataFrame(
        {
            "q": [70, 70, 80, 80, 90, 90],
            "bpp": [0.030, 0.031, 0.040, 0.041, 0.060, 0.061],
            "ms_ssim_y": [0.990, 0.995, 0.992, 0.996, 0.994, 0.997],
            "gmsd": [0.020, 0.018, 0.016, 0.015, 0.012, 0.011],
        }
    )

    s = summarize_saturation_by_q(df)
    assert "delta_q" in s.columns
    assert "delta_ms_ssim_y_p10_per_bpp" in s.columns
    assert "delta_gmsd_p90_improve_per_bpp" in s.columns
    assert "delta_ms_ssim_y_p10_per_q" in s.columns
    assert "delta_gmsd_p90_improve_per_q" in s.columns
    assert "delta_bpp_median_per_q" in s.columns
    assert "ms_ssim_y_p10_over_bpp_median" in s.columns
    assert "gmsd_p90_over_bpp_median" in s.columns

    assert np.isnan(float(s.loc[0, "delta_ms_ssim_y_p10_per_bpp"]))
    assert np.isnan(float(s.loc[0, "delta_gmsd_p90_improve_per_bpp"]))
    assert np.isnan(float(s.loc[0, "delta_ms_ssim_y_p10_per_q"]))
    assert np.isnan(float(s.loc[0, "delta_gmsd_p90_improve_per_q"]))
    assert np.isnan(float(s.loc[0, "delta_bpp_median_per_q"]))

    for i in range(1, len(s)):
        dq = float(s.loc[i, "delta_q"])
        assert dq > 0.0

        delta_bpp = float(s.loc[i, "delta_bpp_median"])
        assert delta_bpp > 0.0

        delta_ms = float(s.loc[i, "delta_ms_ssim_y_p10"])
        got_ms = float(s.loc[i, "delta_ms_ssim_y_p10_per_bpp"])
        assert np.isfinite(got_ms)
        assert np.isclose(got_ms, delta_ms / delta_bpp)
        got_ms_per_q = float(s.loc[i, "delta_ms_ssim_y_p10_per_q"])
        assert np.isfinite(got_ms_per_q)
        assert np.isclose(got_ms_per_q, delta_ms / dq)

        delta_gms = float(s.loc[i, "delta_gmsd_p90_improve"])
        got_gms = float(s.loc[i, "delta_gmsd_p90_improve_per_bpp"])
        assert np.isfinite(got_gms)
        assert np.isclose(got_gms, delta_gms / delta_bpp)
        got_gms_per_q = float(s.loc[i, "delta_gmsd_p90_improve_per_q"])
        assert np.isfinite(got_gms_per_q)
        assert np.isclose(got_gms_per_q, delta_gms / dq)

    assert np.isfinite(float(s.loc[0, "ms_ssim_y_p10_over_bpp_median"]))
    assert np.isfinite(float(s.loc[0, "gmsd_p90_over_bpp_median"]))


def test_summarize_by_q_err_computes_quantiles():
    from discord_webp_tuner.plot import summarize_by_q_err

    df = pd.DataFrame(
        {
            "q": [70, 70, 70, 80, 80, 80],
            "bpp": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "err": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        }
    )
    s = summarize_by_q_err(df, err_col="err", err_quantile=0.90)
    assert list(s["q"].astype(int)) == [70, 80]
    assert float(s.loc[0, "err_p10"]) <= float(s.loc[0, "err"]) <= float(s.loc[0, "err_p90"])
    assert float(s.loc[1, "err_p10"]) <= float(s.loc[1, "err"]) <= float(s.loc[1, "err_p90"])


def test_pareto_front_err_is_monotone():
    from discord_webp_tuner.plot import pareto_front_err

    df = pd.DataFrame(
        {
            "q": [70, 80, 90, 95],
            "bpp": [0.02, 0.03, 0.04, 0.05],
            "err": [0.50, 0.40, 0.45, 0.35],
        }
    )
    front = pareto_front_err(df)
    assert list(front["bpp"].astype(float)) == sorted(front["bpp"].astype(float).tolist())
    errs = front["err"].astype(float).tolist()
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))


def test_bpp_hist_by_q_plot_writes_png(tmp_path):
    os.environ.setdefault("MPLBACKEND", "Agg")

    from discord_webp_tuner.plot import bpp_hist_by_q_plot

    df = pd.DataFrame(
        {
            "q": [70] * 10 + [80] * 10 + [90] * 10,
            "bpp": np.concatenate(
                [
                    np.linspace(0.020, 0.030, 10),
                    np.linspace(0.030, 0.045, 10),
                    np.linspace(0.045, 0.070, 10),
                ]
            ),
        }
    )
    out_png = tmp_path / "bpp_hist.png"
    bpp_hist_by_q_plot(df=df, out_path=out_png, title="bpp hist", bins=15, max_cols=3, x_max_quantile=1.0)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_saturation_bpp_split_plots_write_png(tmp_path):
    os.environ.setdefault("MPLBACKEND", "Agg")

    from discord_webp_tuner.plot import (
        saturation_bpp_plot_gmsd_y,
        saturation_bpp_plot_ms_ssim_y,
        saturation_bpp_plot_ssimulacra2,
    )

    df = pd.DataFrame(
        {
            "q": [70, 70, 80, 80, 90, 90],
            "bpp": [0.030, 0.031, 0.040, 0.041, 0.060, 0.061],
            "ms_ssim_y": [0.990, 0.995, 0.992, 0.996, 0.994, 0.997],
            "gmsd": [0.020, 0.018, 0.016, 0.015, 0.012, 0.011],
            "ssimulacra2": [95.0, 96.0, 97.0, 98.0, 98.5, 99.0],
        }
    )

    out_ms = tmp_path / "sat_bpp_ms.png"
    out_gm = tmp_path / "sat_bpp_gm.png"
    out_s2 = tmp_path / "sat_bpp_s2.png"
    saturation_bpp_plot_ms_ssim_y(df=df, out_path=out_ms, title="ms")
    saturation_bpp_plot_gmsd_y(df=df, out_path=out_gm, title="gm")
    saturation_bpp_plot_ssimulacra2(df=df, out_path=out_s2, title="s2")
    assert out_ms.exists() and out_ms.stat().st_size > 0
    assert out_gm.exists() and out_gm.stat().st_size > 0
    assert out_s2.exists() and out_s2.stat().st_size > 0


def test_saturation_split_plots_write_png(tmp_path):
    os.environ.setdefault("MPLBACKEND", "Agg")

    from discord_webp_tuner.plot import saturation_plot_gmsd_y, saturation_plot_ms_ssim_y, saturation_plot_ssimulacra2

    df = pd.DataFrame(
        {
            "q": [70, 70, 80, 80, 90, 90],
            "bpp": [0.030, 0.031, 0.040, 0.041, 0.060, 0.061],
            "ms_ssim_y": [0.990, 0.995, 0.992, 0.996, 0.994, 0.997],
            "gmsd": [0.020, 0.018, 0.016, 0.015, 0.012, 0.011],
            "ssimulacra2": [95.0, 96.0, 97.0, 98.0, 98.5, 99.0],
        }
    )

    out_ms = tmp_path / "sat_ms.png"
    out_gm = tmp_path / "sat_gm.png"
    out_s2 = tmp_path / "sat_s2.png"
    saturation_plot_ms_ssim_y(df=df, out_path=out_ms, title="ms")
    saturation_plot_gmsd_y(df=df, out_path=out_gm, title="gm")
    saturation_plot_ssimulacra2(df=df, out_path=out_s2, title="s2")
    assert out_ms.exists() and out_ms.stat().st_size > 0
    assert out_gm.exists() and out_gm.stat().st_size > 0
    assert out_s2.exists() and out_s2.stat().st_size > 0
