# Lipflow and QuietBridge for Windows: install dependencies, download the models (~1.2 GB) and add
# Start menu shortcuts (Lipflow, and QuietBridge, which opens its page in the browser).
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Samples] [-StrictSamples] [-NoShortcut]
#
#   -Samples        also fetch the public-domain test clips (best effort: a failure warns, setup goes on)
#   -StrictSamples  fetch the clips and exit non-zero if any could not be downloaded
#   -NoShortcut     don't create the Start menu shortcuts
#
# Downloads come from scripts/models.txt and scripts/samples.txt (shared with setup.sh). Each is retried
# with backoff (honouring Retry-After) and resumed from its .part file. Tunables for CI:
#   LIPFLOW_DOWNLOAD_RETRIES (default 6), LIPFLOW_DOWNLOAD_BACKOFF (first wait in seconds, default 5),
#   LIPFLOW_DOWNLOAD_MAX_WAIT (longest single wait in seconds, default 60, also caps Retry-After).
param([switch]$Samples, [switch]$StrictSamples, [switch]$NoShortcut)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is 10x slower with the progress bar
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
if ($StrictSamples) { $Samples = $true }

# A generic, descriptive client name: Wikimedia throttles default library user agents.
$UA = "lipflow-setup/0.1 (model and sample downloader; curl)"
$Retries = if ($env:LIPFLOW_DOWNLOAD_RETRIES) { [int]$env:LIPFLOW_DOWNLOAD_RETRIES } else { 6 }
$Backoff = if ($env:LIPFLOW_DOWNLOAD_BACKOFF) { [int]$env:LIPFLOW_DOWNLOAD_BACKOFF } else { 5 }
$MaxWait = if ($env:LIPFLOW_DOWNLOAD_MAX_WAIT) { [int]$env:LIPFLOW_DOWNLOAD_MAX_WAIT } else { 60 }

function Write-SetupWarning($msg) {
    Write-Warning $msg
    if ($env:GITHUB_ACTIONS) { Write-Host "::warning::$msg" }
}

function Get-RetryAfter($headers) {
    # Seconds from the last Retry-After header (numeric form only), or $null.
    if (-not (Test-Path $headers)) { return $null }
    $value = $null
    foreach ($line in Get-Content $headers) {
        if ($line -match '^\s*Retry-After:\s*(\d+)\s*$') { $value = [int]$Matches[1] }
    }
    return $value
}

function Get-File([string]$url, [string]$dest, [long]$min = 1) {
    # $true on success; on failure prints the URL and returns $false.
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) { Write-Host "ok  $dest"; return $true }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    $part = "$dest.part"
    $headers = "$dest.headers"
    $wait = $Backoff
    for ($attempt = 1; ; $attempt++) {
        Write-Host "get $dest (attempt $attempt/$Retries)"
        # -f: an HTTP error writes nothing into the .part file; -C -: resume what an earlier try saved.
        $code = & curl.exe -fL -sS -A $UA -C - --connect-timeout 30 --speed-limit 1024 --speed-time 60 `
            -D $headers -o $part -w '%{http_code}' $url
        $rc = $LASTEXITCODE
        if ($rc -eq 0) {
            $size = (Get-Item $part).Length
            if ($size -ge $min) {
                Move-Item -Force $part $dest
                Remove-Item -Force -ErrorAction SilentlyContinue $headers
                return $true
            }
            Write-Host "  got only $size bytes (expected at least $min); starting over"
            Remove-Item -Force $part
        } elseif ($code -eq "416" -or $rc -eq 33) {
            Write-Host "  the server can't resume this file; starting over"  # stale or unsupported range
            Remove-Item -Force -ErrorAction SilentlyContinue $part
        } elseif ($rc -eq 22 -and @("408", "425", "429", "500", "502", "503", "504") -notcontains $code) {
            Write-Host "  HTTP $code is not temporary; giving up on $url"
            Remove-Item -Force -ErrorAction SilentlyContinue $headers
            return $false
        } else {
            Write-Host "  failed (curl exit $rc, HTTP $code)"
        }
        if ($attempt -ge $Retries) {
            Write-Host "  giving up after $attempt attempts: $url"
            Remove-Item -Force -ErrorAction SilentlyContinue $headers
            return $false
        }
        $after = Get-RetryAfter $headers
        $sleepFor = if ($null -ne $after) { $after } else { $wait }
        $sleepFor = [Math]::Min($sleepFor, $MaxWait)
        if ($null -ne $after) { Write-Host "  retrying in ${sleepFor}s (server asked: Retry-After $after)" }
        else { Write-Host "  retrying in ${sleepFor}s" }
        Start-Sleep -Seconds $sleepFor
        $wait = [Math]::Min($wait * 2, $MaxWait)
    }
}

function Get-List([string]$manifest) {
    # Downloads every entry; returns the destinations that failed.
    $failed = @()
    foreach ($line in Get-Content $manifest) {
        $line = $line.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $url, $dest, $min = -split $line
        if (-not $min) { $min = 1 }
        if (-not (Get-File $url $dest ([long]$min))) {
            Write-Host "FAILED: $dest from $url"
            $failed += $dest
        }
    }
    return ,$failed
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

# Required: without these nothing can be lip-read, so a failure stops setup.
$failedModels = Get-List "scripts/models.txt"
if ($failedModels.Count -gt 0) {
    throw "Model download failed: $($failedModels -join ', '). Re-run setup.ps1 to resume."
}

$sampleProblem = $false
if ($Samples) {
    $failedSamples = Get-List "scripts/samples.txt"
    if ($failedSamples.Count -gt 0) {
        # Only tests use the clips; a rate-limited clip must not stop a working install.
        Write-SetupWarning ("Sample clips could not be downloaded ($($failedSamples -join ', ')); tests that " +
            "need them will be skipped. Re-run setup.ps1 -Samples later.")
        $sampleProblem = $true
    }
}

if (-not $NoShortcut) {
    # Start menu entry that runs the tray app without a console window
    $data = Join-Path $env:APPDATA "Lipflow"
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    $ico = Join-Path $data "Lipflow.ico"
    uv run python -c "import sys; from lipflow.win.hud import tray_image; tray_image(size=256).save(sys.argv[1], sizes=[(16,16),(32,32),(48,48),(256,256)])" $ico
    if ($LASTEXITCODE -ne 0) { throw "could not write the shortcut icon $ico" }
    $lnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Lipflow.lnk"
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sh.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $sh.Arguments = "-X utf8 -m lipflow"
    $sh.WorkingDirectory = $PSScriptRoot
    $sh.IconLocation = $ico
    $sh.Description = "Silent dictation by lip reading"
    $sh.Save()
    Write-Host "ok  Start menu shortcut: $lnk"

    # QuietBridge without a console window: the page opens in the default browser, Quit is in its
    # Settings. Starting it again while it runs just opens the running one's page.
    $qlnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\QuietBridge.lnk"
    $qs = (New-Object -ComObject WScript.Shell).CreateShortcut($qlnk)
    $qs.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $qs.Arguments = "-X utf8 -m quietbridge.server"
    $qs.WorkingDirectory = $PSScriptRoot
    $qs.Description = "QuietBridge: typing, lip movements and gestures become text you review, then speech"
    $qs.Save()
    Write-Host "ok  Start menu shortcut: $qlnk"
}

Write-Host ""
Write-Host "Done. Open Lipflow from the Start menu (it lives in the system tray, next to the clock)."
Write-Host "The first launch walks you through your Wispr Flow words and ~24 practice sentences."
Write-Host "QuietBridge: open it from the Start menu (or: uv run quietbridge); the page opens by itself."
Write-Host "Optional stock gesture model: download it from Setup in the QuietBridge page (or: uv run python -m quietbridge.signs.assets --download)"
Write-Host "Check everything with:  uv run lipflow doctor"
if ($sampleProblem -and $StrictSamples) { exit 1 }
