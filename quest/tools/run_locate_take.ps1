<#
One-command real-world test of the head-movement locator (PR #11), made for a noisy room.

You keep the headset ON and stay seated. The LAPTOP is the sound source and stays put in
front of you (about 1 m, speakers facing you, volume up). For each take you turn your CHAIR so
the laptop is at a different angle from where you face; a voice tells you what to do.
Each take: ~4 s of room noise (source off, used to subtract the room), the laptop plays
noise, then two beeps in the headset (and a voice) = turn your head slowly left/right about 40 degrees
each way, three beeps = stop. Then it scores the take (PASS/FAIL per check).

Run from the repo root:
    powershell -ExecutionPolicy Bypass -File quest/tools/run_locate_take.ps1
Options: -Truths 0,90,180,-90   -Quiet (no spoken prompts; just wait)   -Seconds 28
Coverage-grid labels (stored in meta.json for eval/dataset_check.py; all optional):
    -Distance 1.0 (source distance in meters)   -Motion small|normal|large   -Stimulus pink|speech
    -Stimulus speech needs -StimulusWav <PCM .wav> (the script itself only generates pink noise).
Angles: 0 = laptop straight ahead, 90 = directly on your right, 180 = behind you, -90 = on your left.
#>
param(
    [double[]]$Truths = @(0, 90, 180, -90),
    [switch]$Quiet,
    [int]$VoiceVolume = 35,      # spoken prompts, 0-100
    [int]$NoiseUntil = 4,
    [int]$GoAt = 8,
    [int]$StopAt = 28,
    [string]$Adb = "C:\Program Files\Unity\Hub\Editor\6000.0.80f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe",
    [int]$Port = 8765,
    [double]$Distance = [double]::NaN,                      # meters; NaN = not labeled
    [ValidateSet("", "small", "normal", "large")][string]$Motion = "",
    [ValidateSet("", "pink", "speech")][string]$Stimulus = "",
    [string]$StimulusWav = ""                                # PCM wav to loop instead of the generated pink noise
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$pkg = "com.DefaultCompany.quest"
$act = "$pkg/com.unity3d.player.UnityPlayerGameActivity"
$stamp = Get-Date -Format "MMdd-HHmm"
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Volume = $VoiceVolume
function Say($t) { Write-Host ">> $t" -ForegroundColor Cyan; if (-not $Quiet) { $voice.Speak($t) } else { Start-Sleep -Seconds 4 } }

# stimulus: 40 s of pink-ish noise as a WAV the laptop can loop
$noiseWav = Join-Path $env:TEMP "cocktail_pink.wav"
$gen = @'
import numpy as np, wave, sys
rng = np.random.default_rng(0); n = 44100 * 40
w = rng.standard_normal(n); W = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1/44100); f[0] = 1
x = np.fft.irfft(W / np.sqrt(f), n); x = np.clip(x / x.std() * 0.30, -0.98, 0.98)  # loud: RMS about -10 dBFS
with wave.open(sys.argv[1], "wb") as o:
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(44100); o.writeframes((x * 32767).astype("<i2").tobytes())
'@
$gen | Set-Content -Encoding ascii "$env:TEMP\gen_pink.py"
uv run --project hub python "$env:TEMP\gen_pink.py" $noiseWav
if ($Stimulus -eq "speech" -and -not $StimulusWav) { throw "-Stimulus speech needs -StimulusWav <PCM .wav>: this script only generates pink noise." }
if ($StimulusWav) { $noiseWav = (Resolve-Path $StimulusWav).Path }
$player = New-Object System.Media.SoundPlayer $noiseWav
$labelArgs = @()
if (-not [double]::IsNaN($Distance)) { $labelArgs += @("--distance-m", $Distance) }
if ($Motion) { $labelArgs += @("--motion", $Motion) }
if ($Stimulus) { $labelArgs += @("--stimulus", $Stimulus) }

# adb can hang when two adb servers fight over port 5037; run it with a timeout and restart the server once.
function AdbRun([string[]]$a, [int]$timeoutSec = 15) {
    for ($try = 0; $try -lt 2; $try++) {
        $of = Join-Path $env:TEMP "adbrun.out"
        $p = Start-Process $Adb -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $of
        if ($p.WaitForExit($timeoutSec * 1000)) { return (Get-Content $of -ErrorAction SilentlyContinue) }
        Write-Host "adb hung; restarting the adb server..." -ForegroundColor Yellow
        Get-Process adb -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
        Start-Process $Adb -ArgumentList "start-server" -NoNewWindow -Wait | Out-Null
    }
    throw "adb keeps hanging. Close Unity/Meta Quest Developer Hub, run 'taskkill /f /im adb.exe', and try again."
}

AdbRun @("reverse","tcp:$Port","tcp:$Port") | Out-Null
$results = @()
Say "Locator test. Keep the headset on and stay seated. Put the laptop about one meter in front of you with the volume up. Starting in ten seconds."
Start-Sleep -Seconds 10

foreach ($truth in $Truths) {
    $name = "t$($truth -replace '-','m')-$stamp"
    $out = "hub/recordings/$name"
    switch ($truth) {
        0   { $msg = "Turn your chair so the laptop is straight ahead of you." }
        90  { $msg = "Turn your chair left by ninety degrees, so the laptop is directly on your right." }
        180 { $msg = "Turn your chair around, so the laptop is directly behind you." }
        -90 { $msg = "Turn your chair right by ninety degrees, so the laptop is directly on your left." }
        default { $msg = "Turn so the laptop is $truth degrees to your right." }
    }
    Say $msg
    Start-Sleep -Seconds 5
    Say "Sit still, face front. Recording starts now. Wait for two beeps, then turn only your head slowly left and right. Three beeps means stop."

    $recArgs = @("run","--project","hub","python","quest/tools/record_session.py","--out",$out,"--truth",$truth,"--noise-until",$NoiseUntil,"--go-at",$GoAt,"--stop-at",$StopAt) + $labelArgs
    $rec = Start-Process uv -ArgumentList $recArgs -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 5
    # start the headset app if it is not running (if it is, it reconnects to the recorder by itself;
    # restarting it would reset its clock mid-recording)
    AdbRun @("shell","am","start","-n",$act,"-e","hub_host","127.0.0.1","-e","micsrc","unprocessed","-e","micch","2") | Out-Null
    $t0 = $null
    for ($i = 0; $i -lt 60 -and -not $t0; $i++) {
        Start-Sleep -Milliseconds 500
        try { $h = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 2; if ($h.audio_frames -gt 4800) { $t0 = (Get-Date).AddSeconds(-0.1) } } catch {}
    }
    if (-not $t0) { Write-Host "No audio reached the recorder in 30 s. Is the headset on and the app running?" -ForegroundColor Red; $rec | Stop-Process -Force -ErrorAction SilentlyContinue; continue }
    Start-Sleep -Seconds $NoiseUntil
    $player.PlayLooping()
    # spoken cue exactly when the two beeps sound (the scorer ignores this stretch)
    Start-Sleep -Milliseconds (($GoAt - $NoiseUntil - 0.4) * 1000)
    if (-not $Quiet) { $null = $voice.SpeakAsync("Now. Turn your head slowly left and right, and keep going.") }
    $rec.WaitForExit(($StopAt - $GoAt + 12) * 1000) | Out-Null
    $player.Stop()
    if (-not $rec.HasExited) { $rec | Stop-Process -Force }
    Say "Take finished."
    Write-Host "`n===== TAKE truth=$truth =====" -ForegroundColor Yellow
    Push-Location hub
    $txt = uv run python -m locate.real_eval "recordings/$name" 2>&1
    Pop-Location
    $txt | ForEach-Object { Write-Host $_ }
    $res = if ($txt -match "RESULT: PASS") { "PASS" } else { "FAIL" }
    $results += [pscustomobject]@{ TruthDeg = $truth; Result = $res; Take = $name }
    Start-Sleep -Seconds 3
}

Write-Host "`n===== SUMMARY =====" -ForegroundColor Green
$results | Format-Table -AutoSize
$fails = ($results | Where-Object Result -eq "FAIL").Count
if ($results.Count -eq 0) { exit 1 }
if ($fails -eq 0) { Write-Host "ALL TAKES PASS" -ForegroundColor Green; exit 0 } else { Write-Host "$fails of $($results.Count) takes FAIL" -ForegroundColor Red; exit 1 }
