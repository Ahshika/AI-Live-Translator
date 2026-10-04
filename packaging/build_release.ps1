# Build the full release: capture helper -> app folder -> installer.
#   powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Version 1.0.0
param([string]$Version = "1.0.0")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "1/4 tests" -ForegroundColor Cyan
& "$root\backend\.venv\Scripts\python.exe" -m pytest -q "$root\backend\tests"
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Write-Host "2/4 meeting-audio capture helper (.NET)" -ForegroundColor Cyan
dotnet publish "$root\native\ProcessLoopback" -c Release -o "$root\native\ProcessLoopback\publish"
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }

Write-Host "3/4 app folder (PyInstaller)" -ForegroundColor Cyan
& "$root\backend\.venv\Scripts\pyinstaller.exe" "$root\packaging\translator.spec" --noconfirm `
    --distpath "$root\build\dist" --workpath "$root\build\work"
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

Write-Host "4/4 installer (Inno Setup)" -ForegroundColor Cyan
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 is not installed" }
& $iscc "/DAppVersion=$Version" "$root\packaging\installer.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

Get-ChildItem "$root\build\installer\*.exe" | ForEach-Object {
    Write-Host ("done: {0}  ({1:N0} MB)" -f $_.FullName, ($_.Length / 1MB)) -ForegroundColor Green
}
