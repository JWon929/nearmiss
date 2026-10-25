# One-command setup for Windows:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Makes .venv, installs the pinned requirements, and starts the app. Safe to run again.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    # A real Python 3.12-3.14. "python" in WindowsApps is only a shortcut to the Microsoft Store, so it is skipped.
    foreach ($v in "3.12", "3.13", "3.14") {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    foreach ($name in "python3.12", "python3.13", "python3.14", "python3", "python") {
        foreach ($cmd in Get-Command $name -All -ErrorAction SilentlyContinue) {
            if ($cmd.Source -like "*\WindowsApps\*") { continue }
            $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -in "3.12", "3.13", "3.14") { return $cmd.Source }
        }
    }
    return $null
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $py = Find-Python
    if (-not $py) {
        Write-Host ""
        Write-Host "No Python 3.12-3.14 found (the 'python' command here only opens the Microsoft Store)." -ForegroundColor Red
        Write-Host "Install Python 3.12 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
        Write-Host "Then close this terminal, open a new one, and run this script again."
        exit 1
    }
    Write-Host "Creating .venv with $py ..."
    & $py -m venv .venv
    if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "Could not create .venv." -ForegroundColor Red; exit 1 }
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Installing the packages failed; see the messages above." -ForegroundColor Red; exit 1 }
& .venv\Scripts\python.exe -m streamlit run app.py
