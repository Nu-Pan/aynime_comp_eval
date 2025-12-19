# agents.md

## 目的
Discordへ大量投下するアニメ静止画を、見た目の破綻を抑えつつ最小サイズに寄せるためのWebP圧縮パラメータ探索パイプラインを作るニャン。  
評価は「Discord上で縮小表示される見え方」を主戦場にして、WebPの `quality` をスイープして Pareto（サイズ vs 画質）の“膝”を狙うニャン。

## スコープ
- 入力：既存PNG（基本RGB、場合によりRGBA）ニャン。
- 出力：WebP（Pillowでエンコード）＋評価結果CSV＋散布図（matplotlib）ニャン。
- 評価指標：主に **MS-SSIM（輝度Y）**、補助で **GMSD**、監視でPSNR（任意、デフォルトON）＋SSIMULACRA2（任意、デフォルトOFF）ニャン。
- 評価解像度：縮小表示を想定した **long_edge=1280（デフォルト）** を主に使うニャン（長辺が1280になるようリサイズ。16:9素材なら 1280x720 になるニャン）。必要なら `--long-edge` で変更できるニャン。

## 非スコープ
- Discordクライアントやサーバ側の帯域制御の再現・測定はしないニャン。
- 画質指標を「唯一の真理」にせず、最終判断は散布図とサンプル目視で決めるニャン。
- “綺麗な保存版PNG”の代替は狙わないニャン（あくまでDiscord用の軽量版）ニャン。

## 方針（重要）
1. **評価は縮小後**に行うニャン（Discordの体験に寄せるニャン）。
2. まず **縮小→WebP** を基本とし、後段で `quality` を詰めるニャン。
3. WebPは **Pillow** を利用し、探索は `quality` を中心にするニャン。
4. エンコードは基本 `method=6` 固定（時間と引き換えにサイズを詰める）ニャン。
5. αがあるPNGは、比較前に **固定背景へ合成**してから評価するニャン（例：背景 #000/#fff/#808080 のいずれかを選択可能にするニャン）。

---

## 推奨リポジトリ構成
- `.venv/`（python 仮想環境）
- `src/discord_webp_tuner/`
  - `cli.py`（Typer推奨、なければargparseでもOK）
  - `pipeline.py`（探索の主処理）
  - `metrics.py`（MS-SSIM/GMSD/PSNR等）
  - `io.py`（入出力、キャッシュ）
  - `plot.py`（散布図、Pareto、膝点検出）
  - `plot_interactive.py`（Plotlyでのインタラクティブ散布図、任意）
  - `config.py`（探索レンジ・解像度・背景等）
- `scripts/`
  - `run_sweep.ps1` / `run_sweep.sh`（バッチ用）
- `data/`（ローカル専用、gitignore）
  - `input_png/`
  - `out_webp/`
  - `results/`
- `tests/`（最小でOK）
- `pyproject.toml`

---

## セットアップ

### 必須ツール
- Pythonの **Pillow が WebP をサポートしていること**ニャン。
  - 確認例：`from PIL import features; features.check("webp")` が `True` になることニャン。
  - Windowsのホイールで通常はOKだけど、環境によってはWebPが無効の場合があるので注意ニャン。

### Python依存（例）
- Pillow
- numpy
- torch（CPUでOK、あるならCUDA可）
- piq（MS-SSIM/GMSD用）
- ssimulacra2（SSIMULACRA2用、任意）
- pandas（集計）
- matplotlib（散布図）
- tqdm（進捗）
- typer（CLI）
- plotly（インタラクティブ散布図用、任意）

---

## 入出力仕様

### 入力
- PNG（sRGB想定）ニャン。
- 画像サイズは混在してよいが、評価は `long_edge`（デフォルト 1280）に揃えるニャン。

### 出力
- `results/metrics.csv`（最新）＋ `results/metrics_YYYYmmdd_HHMMSS.csv`（タイムスタンプ付き）ニャン。
- `results/run_config_YYYYmmdd_HHMMSS.json`（実行設定ログ）ニャン。
- `results/metrics.csv` のカラム例：
    - `run_id`（YYYYmmdd_HHMMSS）
    - `path_png`
    - `w`, `h`
    - `long_edge`（例：1280）
    - `eval_w`, `eval_h`（縮小後の評価解像度）
    - `q`（quality）
    - `method`（固定6）
    - `sharp_yuv`（0/1）
    - `bg_color`（RGBA合成背景、6桁hex）
    - `webp_bytes`
    - `bpp`（webp_bytes / (`eval_w`*`eval_h`)）
    - `ms_ssim_y`
    - `gmsd`
    - `psnr_y`（任意、デフォルトON）
    - `ssimulacra2`（任意、`--ssimulacra2` で計算）
    - `webp_path_cache`（キャッシュ上のWebPパス）
- `results/scatter_ms_ssim_y_le1280.png`（散布図：MS-SSIM(Y)）ニャン。
- `results/scatter_gmsd_y_le1280.png`（散布図：GMSD(Y)）ニャン。
- `results/scatter_ssimulacra2_le1280.png`（散布図：SSIMULACRA2、ある場合）ニャン。
- `results/saturation_ms_ssim_y_gmsd_y_le1280_by_q.csv`（`q`集約＋Δの表）ニャン。
- `results/saturation_q_ms_ssim_y_le1280.png` / `results/saturation_q_gmsd_y_le1280.png`（`q`-vs-分位＋Δ）ニャン。
- `results/saturation_bpp_ms_ssim_y_le1280.png` / `results/saturation_bpp_gmsd_y_le1280.png`（bpp軸＋膝点）ニャン。
- `results/bpp_hist_by_q_le1280.png`（q別bppヒストグラム、任意）ニャン。
- `out_webp/` に軽量WebPを保存（必要ならベストのみ）ニャン。

---

## パイプライン詳細

### 1) 前処理
- PNGを読み込み（Pillow）→必要ならRGB化ニャン。
- RGBAの場合：
  - 背景色を `bg_color` で選び合成しRGB化ニャン。
- 縮小：
  - アスペクト比を維持して **長辺が `long_edge`** になるようにリサイズするニャン（16:9素材なら 1280x720 になるニャン）。
  - リサンプラは `LANCZOS` を推奨し、全工程で固定して再現性を担保するニャン。

### 2) WebPエンコード（Pillow）
- 方針：縮小後の画像を `Image.save(..., format="WEBP")` でエンコードするニャン。
- 代表パラメータ：
  - `quality={q}`（探索対象、0..100）
  - `method=6`（固定）
  - `use_sharp_yuv` をON/OFFして比較できる設計にするニャン（輪郭の色にじみ改善に効く場合があるニャン）。
- 実装イメージ（例）：
  - `img.save(out_path, "WEBP", quality=q, method=6, use_sharp_yuv=sharp_yuv)`
- 探索レンジ（例）：
  - まず粗く：`q = 60..100 step 5`
  - 絞り込み：膝付近を `step 1` で再探索ニャン。

### 3) デコード
- PillowでWebPを読み、RGBへ変換ニャン。
- 色管理の厳密さより「同じ手順で一貫」させることを優先するニャン。

### 4) 指標計算（targetごと）
- `orig_small`（縮小後の基準）と `webp_small`（デコード後）を比較するニャン。
- 推奨：
  - `MS-SSIM` は Y（輝度）で計算ニャン。
    - 簡易で良いので `YCbCr` に変換して Y を取り出す方針で統一するニャン。
  - `GMSD` は同じくYで計算するニャン。
- 値域・型：
  - torch tensorは `[N,C,H,W]`、float32、値域 `[0,1]` に正規化ニャン。

### 5) 可視化と意思決定

#### 5-1) 散布図（俯瞰）
- 横軸：`bpp` ニャン。
- 縦軸：`1 - ms_ssim_y`（小さいほど良い）ニャン。
- 色：`q` ニャン。
- Paretoフロント抽出（任意）：
  - “サイズが小さくて画質も良い”点を抽出して可視化ニャン。

#### 5-2) “飽和域の下端”を選ぶ（推奨）
狙い：**画質が飽和している範囲内で最も低い `q`** を選び、Discord上での破綻を抑えつつサイズを詰めるニャン。

- 前提：CSVは `per-image × q` が揃っているので、`q` ごとに集約して評価するニャン。
- 見る統計（例）：
  - `p10(ms_ssim_y)`（低いほど危険、ワースト寄りの安全性）
  - `p90(gmsd)`（高いほど危険、ワースト寄りの安全性）
  - 併用で `median(bpp)`（サイズの代表値）
- “飽和”の判定に使う増分（Δ）の例：
  - `Δ(p10(ms_ssim_y))`（`q` を上げた時の改善量）
  - `Δ(p90(gmsd))`（`q` を上げた時の改善量）
  - 追加で `Δmedian(bpp)`（サイズ増分）も見ると費用対効果が分かるニャン。
- 選び方（機械的に決める手順の例）：
  1. targetごとに `q` を昇順で並べ、上の統計とΔをプロットするニャン。
  2. ある `q` 以降で品質側のΔがほぼ0（改善が頭打ち）になったら、その区間を“飽和域”とみなすニャン。
  3. “飽和域”に入った最初の `q`（最小 `q`）を候補にするニャン。
  4. 最終確認として、その候補 `q` 付近のワースト画像を目視するニャン（アニメは同点でも見た目差が出ることがあるニャン）。

#### 5-3) 膝点（任意）
- 近似曲線＋曲率最大、またはkneedle的手法で候補を出すニャン。
- ただし `ms_ssim_y` が飽和しやすいので、最終採用は「飽和域の下端」＋目視を優先するニャン。

---

## パフォーマンス設計
- 画像枚数が多い前提なので、以下を入れるニャン：
  - キャッシュ（同一PNG×targetの縮小結果を保存して使い回す）ニャン。
  - WebP生成物も `q` ごとにキャッシュし、指標計算だけを再実行可能にするニャン。
    - 既定の配置：`cache/resized/`（縮小PNG）・`cache/webp/`（WebP）ニャン。
  - 並列化：
    - 画像単位で `ProcessPoolExecutor` を検討ニャン。
    - torch/PIQはプロセスごと初期化コストがあるので、粒度は「画像1枚×q全探索」を1タスクにするのが無難ニャン。

---

## 目視サンプルの選び方（運用）
- 全画像を評価しても良いが、まずは代表サンプルを選びやすくするニャン：
  - 輪郭が多いカット（髪・瞳・服の皺）ニャン。
  - ベタ面が広いカット（空・壁・単色背景）ニャン。
  - 暗部があるカット（バンディングが出やすい）ニャン。
- 代表で候補 `q` を決めたら、全体に適用して “外れ値” だけ追加で調整するニャン。

---

## CLI要件（最低限）
- `sweep`：
  - 入力ディレクトリ、出力ディレクトリ
  - 例：Discord想定に寄せるなら `--long-edge 1280`（長辺指定）ニャン
  - `--q-min/--q-max/--q-step`
  - `--bg-color`（RGBA用）
  - `--sharp-yuv`（Pillowの `use_sharp_yuv` 切替）
  - `--save-webp`（全保存/ベストのみ/保存なし。`best` は「画像ごと」に Pareto→膝点（max-distance）で1枚選ぶニャン）
  - `--jobs`（画像単位の並列数）
  - `--psnr/--no-psnr`（PSNR(Y)のON/OFF、デフォルトON）
  - `--ssimulacra2/--no-ssimulacra2`（SSIMULACRA2のON/OFF、デフォルトOFF）
  - `--plot/--no-plot`（スイープ後に図を生成、デフォルトON）
  - `--bpp-hist/--no-bpp-hist`（q別bppヒストグラム、デフォルトON）
- `plot`：
  - `metrics.csv`（または `metrics*.csv` があるディレクトリ）から図と集約CSVを再生成ニャン。
  - `--glob`（ディレクトリ指定時の読み込みパターン）、`--dedupe`（同一条件の重複を新しいrun優先で除去）
  - `--q-min/--q-max` / `--long-edge` / `--method` / `--sharp-yuv` / `--bg-color`（フィルタ）
  - `--pareto/--no-pareto` と `--knee/--no-knee`（散布図のオプション）
  - “飽和域の下端”用に `q`-vs-quantile と Δ のプロット＋bpp軸版（膝点）も出すニャン。
- `plot-interactive`（任意）：
  - Plotlyでインタラクティブ散布図（HTML）を出すニャン（`pip install -e '.[interactive]'` が必要）ニャン。

---

## テスト（最小でOK）
- 1枚のPNGを使い、同一入力に対して：
  - 縮小サイズが期待通り
  - 指標が範囲内（MS-SSIMは0..1、GMSDは>=0）
  - PillowのWebPサポートが無い場合に分かりやすく失敗する
  - RGBA合成が再現できる（背景固定）

---

## コーディング規約
- 例外メッセージは「どのファイルで何が起きたか」を必ず入れるニャン。
- パラメータ探索の再現性のため、リサイズ方式・色変換手順・背景合成手順は固定し、設定は全部ログ/CSVに書くニャン。
- 1回の実験結果は「CSV＋図＋設定（json）」で残せるようにするニャン。

---

## 完了条件（Definition of Done）
- 指定したPNG群に対し、long_edge=1280（デフォルト）で `q` スイープを回し、散布図とCSVが出るニャン。
- `q` ごとの `p10(ms_ssim_y)` / `p90(gmsd)` とその Δ をプロットでき、飽和域の下端の候補 `q` を機械的に出せるニャン。
- 複数画像での候補 `q` が目視でも納得でき、Discord用の推奨設定として固定できるニャン。
