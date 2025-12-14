param(
  [Parameter(Mandatory=$true)][string]$InDir,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [int]$QMin = 60,
  [int]$QMax = 100,
  [int]$QStep = 5,
  [string]$TargetA = "A=1280",
  [string]$BgColor = "808080",
  [switch]$SharpYuv,
  [ValidateSet("all","best","none")][string]$SaveWebp = "best",
  [int]$Jobs = 1
)

$args = @(
  "sweep",
  "--in-dir", $InDir,
  "--out-dir", $OutDir,
  "--q-min", $QMin,
  "--q-max", $QMax,
  "--q-step", $QStep,
  "--target-long-edge", $TargetA,
  "--bg-color", $BgColor,
  "--save-webp", $SaveWebp,
  "--jobs", $Jobs
)

if ($SharpYuv) { $args += "--sharp-yuv" }

python -m discord_webp_tuner.cli @args
