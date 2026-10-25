# One-command setup for Windows: .\setup.ps1
# Makes .venv, installs the pinned requirements, and starts the app. Safe to run again.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv ..."
    try { py -3.12 -m venv .venv } catch { }
    if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m streamlit run app.py
