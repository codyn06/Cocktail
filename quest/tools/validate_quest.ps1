<#
Human-checkable validation of the Quest streaming client (PR #18) against a running hub.
Every line prints PASS or FAIL with the measured number and the threshold. Nothing to trust.

Prereqs: Quest plugged in over USB, app installed, hub running:
    uv run --project hub python quest/tools/mock_hub.py     (in another window)
Run from the repo root:
    powershell -ExecutionPolicy Bypass -File quest/tools/validate_quest.ps1
Add -SkipInput to skip the controller step (the part that needs you to press a button).
#>
param(
    [string]$Adb = "C:\Program Files\Unity\Hub\Editor\6000.0.80f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe",
    [int]$Port = 8765,
    [switch]$SkipInput
)

$pkg = "com.DefaultCompany.quest"
$act = "$pkg/com.unity3d.player.UnityPlayerGameActivity"
$results = @()
function Check($name, $ok, $measured, $threshold) {
    $script:results += [pscustomobject]@{ Check = $name; Result = $(if ($ok) { "PASS" } else { "FAIL" }); Measured = $measured; Threshold = $threshold }
}
function Health { try { Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 3 } catch { $null } }

$h0 = Health
Check "hub reachable" ($null -ne $h0) $(if ($h0) { "ok" } else { "no response" }) "GET /health answers"
$devs = & $Adb devices | Select-String "\tdevice$"
Check "headset connected (adb)" ($null -ne $devs) $(if ($devs) { "device" } else { "none" }) "adb lists a 'device'"
if ($null -eq $h0 -or $null -eq $devs) { $results | Format-Table -AutoSize; Write-Host "Cannot continue: fix the FAIL lines above." -ForegroundColor Red; exit 1 }

& $Adb reverse tcp:$Port tcp:$Port | Out-Null
& $Adb shell am force-stop $pkg | Out-Null
& $Adb logcat -c | Out-Null
& $Adb shell am start -n $act -e hub_host 127.0.0.1 -e micsrc unprocessed -e micch 2 | Out-Null
Write-Host "App launched. Put the headset on (accept permission prompts). Measuring in 15 s..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

$a = Health; Start-Sleep -Seconds 10; $b = Health
$dt = 10.0
function Rate($k) { [math]::Round((($b.$k) - ($a.$k)) / $dt, 1) }
$ra = Rate "audio"; $rp = Rate "pose"; $rv = Rate "video"
Check "audio chunks per second" ($ra -ge 40) $ra ">= 40 (expected ~50)"
Check "pose samples per second" ($rp -ge 40) $rp ">= 40 (expected ~50)"
Check "camera frames per second" ($rv -ge 8) $rv ">= 8 (target 15)"

$log = & $Adb logcat -d -s Unity
$rep = $log | Select-String "\[MicStreamer\] REPORT" | Select-Object -Last 1
$raw = $null -ne $rep -and $rep.Line -match "raw=True"
Check "mic is raw (UNPROCESSED)" $raw $(if ($rep) { ($rep.Line -replace '.*REPORT ','' -replace ' status=.*','') } else { "no REPORT line" }) "raw=True"
$live = $log | Select-String "\[MicArray\].*levels dB" | Select-Object -Last 1
$lvl = if ($live -and $live.Line -match "levels dB (-?\d+)/(-?\d+)") { [int]$Matches[1] } else { -999 }
Check "mic channel 0 has signal" ($lvl -gt -100) "$lvl dB" "> -100 dB (floor is -120)"

if (-not $SkipInput) {
    $i0 = (Health).input
    Write-Host "Pull the RIGHT FRONT TRIGGER a few times in the next 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    $i1 = (Health).input
    Check "controller presses reach hub" (($i1 - $i0) -ge 1) ($i1 - $i0) ">= 1 new input events"
}

$results | Format-Table -AutoSize
$fails = ($results | Where-Object Result -eq "FAIL").Count
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 } else { Write-Host "$fails FAIL" -ForegroundColor Red; exit 1 }
