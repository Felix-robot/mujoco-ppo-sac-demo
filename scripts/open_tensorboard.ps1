$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& ".\.venv\Scripts\tensorboard.exe" --logdir runs
