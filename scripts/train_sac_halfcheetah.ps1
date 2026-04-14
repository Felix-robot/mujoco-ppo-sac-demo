param(
    [int]$TotalTimesteps = 200000,
    [string]$RunName = "sac_halfcheetah_200k"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Starting SAC HalfCheetah training."
Write-Host "This may make the computer slower until it finishes."
Write-Host "Run name: $RunName"
Write-Host "Total timesteps: $TotalTimesteps"

& ".\.venv\Scripts\python.exe" train.py `
    --config configs/sac_halfcheetah.yaml `
    --total-timesteps $TotalTimesteps `
    --run-name $RunName
