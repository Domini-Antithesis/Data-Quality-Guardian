# One-command launcher for Windows (PowerShell).
#   Right-click -> "Run with PowerShell", or in a terminal:  .\start.ps1
# Creates a virtual environment on first run, installs the app, starts the
# server and opens the browser. Press Ctrl+C in this window to stop.

param([switch]$NoBrowser)   # -NoBrowser: start the server without opening a browser tab

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py -3.12", "py -3", "python", "python3")) {
        try {
            $exe, $rest = $candidate.Split(" ")
            $version = & $exe @rest -c "import sys; print(sys.version_info[:2] >= (3, 10))" 2>$null
            if ($version -eq "True") { return $candidate }
        } catch {}
    }
    return $null
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $python = Find-Python
    if (-not $python) {
        Write-Host "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and run this script again." -ForegroundColor Red
        exit 1
    }
    Write-Host "Creating virtual environment (first run only)..." -ForegroundColor Cyan
    $exe, $rest = $python.Split(" ")
    & $exe @rest -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
$installed = & $venvPython -c "import importlib.util as u; print(bool(u.find_spec('dq_agent') and u.find_spec('langgraph')))" 2>$null
if ($installed -ne "True") {
    Write-Host "Installing the app and its dependencies (first run only, 1-3 minutes)..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -e ".[all-providers]" --quiet
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example (optional - you can also use the Settings page)." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Starting Data Quality Agent. Press Ctrl+C here to stop." -ForegroundColor Green
Write-Host "(If port 8100 is busy it picks the next free one and prints the address.)" -ForegroundColor DarkGray
Write-Host ""
$serveArgs = @("-m", "dq_agent.cli", "serve")
if (-not $NoBrowser) { $serveArgs += "--open" }
& $venvPython @serveArgs
