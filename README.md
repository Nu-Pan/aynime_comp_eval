# discord-webp-tuner

Discordでの縮小表示を想定して、PNG→縮小→WebP（Pillow）を `quality` スイープし、サイズ vs 画質（MS-SSIM(Y) / GMSD(Y)）をCSVと散布図で出力します。

## Requirements

- Python dependencies:
  - base: `pip install -e .`
  - metrics: `pip install -e ".[metrics]"`
- PillowがWebPをサポートしていること
  - 確認：`python -c "from PIL import features; print(features.check('webp'))"`

## Usage

スイープ：

```powershell
discord-webp-tuner sweep --in-dir data/input_png --out-dir data --q-min 60 --q-max 100 --q-step 5 --target-long-edge A=1280 --bg-color 808080 --sharp-yuv
```

散布図（CSVから生成）：

```powershell
discord-webp-tuner plot --csv data/results/metrics.csv --out-dir data/results --target A
```
