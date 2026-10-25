# Start the Cocktail demo: hub (GPU), USB tunnel, app launch with 2 mic channels, keep the headset awake.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File quest/tools/run_demo.ps1
# Needs: headset on USB with USB debugging allowed, the APK installed (see quest/README.md), uv, the hub GPU env.
# Stop everything with Ctrl+C; undo keep-awake with:  adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
param(
    [string]$Adb = "C:\Program Files\Unity\Hub\Editor\6000.0.80f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe",
    [switch]$NoKeepAwake
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pkg = "com.DefaultCompany.quest"

function Adb([string]$a, [int]$sec = 20) {
    $o = [IO.Path]::GetTempFileName()
    $p = Start-Process $Adb -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardOutput $o
    if (-not $p.WaitForExit($sec * 1000)) { $p | Stop-Process -Force; Write-Warning "adb timeout: $a"; return "" }
    return (Get-Content $o -Raw)
}

if ((Adb "devices") -notmatch "\sdevice") { throw "No headset on USB (adb devices is empty). Replug the cable, put the headset on, accept USB debugging." }

# 1. hub: offline (the models are cached), voice memory for stable identities. Launch config of record.
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
$hub = Start-Process uv -ArgumentList "run", "--group", "gpu", "python", "-m", "separate", "--voice-memory" `
    -WorkingDirectory (Join-Path $repo "hub") -PassThru -NoNewWindow
Write-Host "hub starting (pid $($hub.Id)); first start loads the models for ~20 s"
$deadline = (Get-Date).AddSeconds(120)
do { Start-Sleep -Seconds 2; try { $up = (Invoke-WebRequest http://127.0.0.1:8765/ingest/status -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { $up = $false } } until ($up -or (Get-Date) -gt $deadline)
if (-not $up) { throw "hub did not come up" }

# 2. tunnel + app (2 mic channels: the Quest's other two are empty padding)
Adb "reverse tcp:8765 tcp:8765" | Out-Null
Adb "shell ""echo 127.0.0.1 > /sdcard/Android/data/$pkg/files/hub_host.txt""" | Out-Null
Adb "shell am force-stop $pkg" | Out-Null
Adb "shell am start -n $pkg/com.unity3d.player.UnityPlayerGameActivity -e micch 2" | Out-Null

# 3. keep the headset awake as if worn
if (-not $NoKeepAwake) { Adb "shell am broadcast -a com.oculus.vrpowermanager.prox_close" | Out-Null }

Write-Host "Demo up. Status: http://localhost:8765/ingest/status  /speakers  /person-box/status  /selection"
Write-Host "Scene: both people in view and facing the headset; each talks ALONE ~3 s first so the voice-to-face match can learn."
while (-not $hub.HasExited) {
    Start-Sleep -Seconds 30
    if (-not $NoKeepAwake) { Adb "shell am broadcast -a com.oculus.vrpowermanager.prox_close" | Out-Null }
}
