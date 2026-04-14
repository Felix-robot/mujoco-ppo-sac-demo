$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& ".\.venv\Scripts\python.exe" train.py `
    --config configs/ppo_halfcheetah.yaml `
    --smoke-test `
    --run-name smoke_ppo_halfcheetah
