param(
    [int]$TotalTimesteps = 200000,
    [string]$RunName = "ppo_halfcheetah_200k"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Starting PPO HalfCheetah training."
Write-Host "This may make the computer slower until it finishes."
Write-Host "Run name: $RunName"
Write-Host "Total timesteps: $TotalTimesteps"

& ".\.venv\Scripts\python.exe" train.py `
    --config configs/ppo_halfcheetah.yaml `
    --total-timesteps $TotalTimesteps `
    --run-name $RunName
