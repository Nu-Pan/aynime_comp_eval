# discord-webp-tuner

Discordでの縮小表示を想定して、PNG→縮小→WebP(cwebp)の `-q` をスイープし、サイズ vs 画質（MS-SSIM(Y) / GMSD(Y)）をCSVと散布図で出力します。

## Requirements

- `cwebp` がPATH上で実行できること（libwebp）
- Python dependencies:
  - base: `pip install -e .`
  - metrics: `pip install -e ".[metrics]"`

## Usage

スイープ:

```powershell
discord-webp-tuner sweep --in-dir data/input_png --out-dir data --q-min 30 --q-max 80 --q-step 5 --target-long-edge A=1080 B=1600 --bg-color 808080 --sharp-yuv
```

散布図（CSVから再生成）:

```powershell
discord-webp-tuner plot --csv data/results/metrics.csv --out-dir data/results --target A
```

