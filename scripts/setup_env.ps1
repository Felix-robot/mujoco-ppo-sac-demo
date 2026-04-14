param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Environment is ready."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
