# Daily local run of the leaderboard update (for Windows Task Scheduler).
# Uses the cookies in scraper\.firefox-profile\chesscom-cookies.json (log in with
# `python -m scraper.game_scraping login` first). Writes leaderboards\data only.
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File <repo>\leaderboards\run_local.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $env:LOCALAPPDATA "chaturaji-leaderboard"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("run-" + (Get-Date -Format "yyyy-MM-dd") + ".log")

python -m leaderboards.update *>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) {
    Write-Error "leaderboard update failed (see $log)"
    exit $LASTEXITCODE
}
Write-Output "leaderboards\data updated; review and commit it yourself"
