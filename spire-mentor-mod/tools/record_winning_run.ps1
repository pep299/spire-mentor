# record_winning_run.ps1 -- keep playing runs until one is WON, and keep the
# full-length video of exactly that run.
#
# Purpose: produce an uploadable demo. A demo of a loss is worthless, so this
# enforces the property mechanically instead of hoping the timing works out.
#
# ONE GAME PROCESS PER RUN. An earlier version tried to keep a single game
# alive across many runs on the assumption that num_runs > 1 makes
# SimAutoSlayer chain them. It does not, in this lane: when a run ends the
# advisor logs "Auto-play off: the run has ended" and disarms, and nothing
# starts the next run -- the driver then sat forever recording an idle main
# menu. So each attempt relaunches the game.
#
# That is affordable only because Start-Capture retries: Game Bar refuses a
# capture for the first ~10-25 s of a process's life while a run embarks at
# ~25-30 s, so a single early attempt loses the race but a loop that keeps
# trying every ~10 s wins it. Measured: launch 20:27:55, first toggle refused,
# capture running 20:28:19, run embarked seconds later.
#
# Run 1 IS recorded, best-effort. An earlier version skipped it because it can
# only ever be partial -- and the very first run it skipped was the only win of
# the session. Wins are the scarce resource; a partial video of a win beats no
# video of a win, and it costs nothing to try.
#
# Whether a capture is running is decided by an open write handle on the mp4,
# never by file size: Game Bar buffers and writes the bulk on stop, so a live
# capture can sit at a constant size for 15 s and read as "stopped".
#
# OUTCOME DETECTION IS DELIBERATELY ASYMMETRIC. A video is deleted ONLY on a
# confirmed death ("has died to a MONSTER"). Everything else -- victory, an
# ambiguous ending, a crash, the game exiting -- is KEPT for inspection.
# `[SimAutoSlayer] Victory!` is NOT reliable here: with advisor_autoplay the
# SimAutoSlayer loop has usually already returned by the time the run ends, so
# nothing logs it. The 2026-08-10 win produced no victory line whatsoever, so
# treating "no victory marker" as "not a win" would silently bin the exact
# file we are trying to produce.
#
# Progress is appended to logs/record_winning_run.log so a stalled attempt is
# visible without attaching to anything.

param(
    [int]$MaxAttempts = 12,
    [int]$MaxHours = 4,
    # Per-attempt cap. Without one, a wedged run sits in the wait loop until
    # the GLOBAL deadline and then reports TIMEOUT -- which the
    # keep-on-ambiguity rule below promotes to a "win", ending the session
    # with a video of a hung game. A run that has not ended in this many
    # minutes is abandoned and retried instead. Generous on purpose: the
    # 2026-08-10 A1 win took 4 minutes end to end.
    [int]$AttemptMinutes = 20,
    [string]$Character = "Ironclad",
    [int]$Ascension = 1,
    # Fixed seed to replay. A seed does NOT guarantee the same result -- it
    # fixes the CONTENT (map, offers, HP rolls) while live snapshot timing can
    # still shift a decision and diverge the run -- but a seed already won by
    # this agent is strictly better information than a random one.
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"
$repo     = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gameDir  = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
$modDir   = "$gameDir\mods\SpireMentor"
$capDir   = "$env:USERPROFILE\Videos\Captures"
$godotLog = "$env:APPDATA\SlayTheSpire2\logs\godot.log"
$outLog   = "$repo\logs\record_winning_run.log"
New-Item -ItemType Directory -Force -Path (Split-Path $outLog) | Out-Null

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $outLog -Value $line
}

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Cap {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool f);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern void keybd_event(byte b, byte s, uint f, UIntPtr e);
    const uint UP = 2;
    // Windows refuses SetForegroundWindow from a background process and
    // returns success anyway -- that silently pointed one capture at the
    // terminal. Tap ALT and borrow the foreground thread's input queue.
    public static bool Focus(IntPtr h) {
        keybd_event(0x12,0,0,UIntPtr.Zero); keybd_event(0x12,0,UP,UIntPtr.Zero);
        uint fg = GetWindowThreadProcessId(GetForegroundWindow(), IntPtr.Zero);
        uint me = GetCurrentThreadId();
        if (fg != me) AttachThreadInput(me, fg, true);
        ShowWindow(h, 9); BringWindowToTop(h); SetForegroundWindow(h);
        if (fg != me) AttachThreadInput(me, fg, false);
        System.Threading.Thread.Sleep(400);
        return GetForegroundWindow() == h;
    }
    public static void WinAltR() {
        keybd_event(0x5B,0,0,UIntPtr.Zero); keybd_event(0x12,0,0,UIntPtr.Zero); keybd_event(0x52,0,0,UIntPtr.Zero);
        System.Threading.Thread.Sleep(60);
        keybd_event(0x52,0,UP,UIntPtr.Zero); keybd_event(0x12,0,UP,UIntPtr.Zero); keybd_event(0x5B,0,UP,UIntPtr.Zero);
    }
}
'@

function Get-GameWindow {
    Get-Process -Name SlayTheSpire2 -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
}
function Get-NewestCapture {
    Get-ChildItem $capDir -Filter *.mp4 -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
function Test-Recording($file) {
    if (-not $file) { return $false }
    $f = Get-Item $file.FullName -ErrorAction SilentlyContinue
    if (-not $f) { return $false }
    try { $s = [IO.File]::Open($f.FullName, 'Open', 'ReadWrite', 'None'); $s.Close(); return $false }
    catch { return $true }
}
function Get-LogLength {
    if (-not (Test-Path $godotLog)) { return 0 }
    try { return (Get-Item $godotLog).Length } catch { return 0 }
}
function Read-LogSince($offset) {
    if (-not (Test-Path $godotLog)) { return "" }
    $fs = [IO.File]::Open($godotLog, 'Open', 'Read', 'ReadWrite')
    if ($offset -gt $fs.Length) { $offset = 0 }   # log rotated on relaunch
    # NEVER ReadToEnd() an unbounded tail. The game has a render-loop wedge
    # that spits `Parameter "particles" is null` at ~4 MB/s; godot.log hit
    # 1.96 GB in nine minutes, and this function was slurping the whole thing
    # into memory every 10 s. A normal full run's log is ~300 KB, so a 4 MB
    # window never truncates anything real -- it only bounds the pathological
    # case.
    $maxTail = 4MB
    if (($fs.Length - $offset) -gt $maxTail) { $offset = $fs.Length - $maxTail }
    $fs.Seek($offset, 'Begin') | Out-Null
    $sr = New-Object IO.StreamReader($fs)
    $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close()
    return $t
}
function Toggle-Capture {
    $p = Get-GameWindow
    if (-not $p) { return $false }
    for ($i = 1; $i -le 5; $i++) { if ([Cap]::Focus($p.MainWindowHandle)) { break }; Start-Sleep -Milliseconds 500 }
    if ([Cap]::GetForegroundWindow() -ne $p.MainWindowHandle) {
        Say "  WARN: could not focus the game; refusing to toggle capture"
        return $false
    }
    [Cap]::WinAltR()
    return $true
}
# Start a capture and PROVE it is running AND pointed at the game.
function Start-Capture {
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        $before = Get-NewestCapture
        if (-not (Toggle-Capture)) { Start-Sleep -Seconds 3; continue }
        Start-Sleep -Seconds 6
        $now = Get-NewestCapture
        if ($now -and (-not $before -or $now.Name -ne $before.Name) -and (Test-Recording $now)) {
            if ($now.Name -notlike "Slay the Spire 2*") {
                Say "  WARN: captured the WRONG window ($($now.Name)); stopping and retrying"
                [void](Toggle-Capture); Start-Sleep -Seconds 6
                Remove-Item $now.FullName -Force -ErrorAction SilentlyContinue
                continue
            }
            Say "  recording -> $($now.Name)"
            return $now
        }
        Say "  Game Bar not ready yet (attempt $attempt); retrying"
        Start-Sleep -Seconds 4
    }
    return $null
}
function Stop-Capture($file) {
    if (-not $file) { return }
    # If the game is gone, Game Bar finalizes on its own -- just wait for the
    # write handle to be released rather than trying to focus a dead window.
    if (-not (Get-GameWindow)) {
        for ($i = 1; $i -le 15 -and (Test-Recording $file); $i++) { Start-Sleep -Seconds 2 }
    } else {
        for ($i = 1; $i -le 4 -and (Test-Recording $file); $i++) {
            [void](Toggle-Capture); Start-Sleep -Seconds 8
        }
    }
    $f = Get-Item $file.FullName -ErrorAction SilentlyContinue
    if ($f) { Say ("  stopped -> {0:N1} MB" -f ($f.Length/1MB)) }
}

# ---- config: one game process, many runs -------------------------------
$seedJson = if ($Seed) { "`"$Seed`"" } else { "null" }
Say "=== record_winning_run: $Character A$Ascension seed=$(if($Seed){$Seed}else{'random'}) max $MaxAttempts ==="
Copy-Item "$modDir\SpireMentor.conf" "$modDir\SpireMentor.conf.bak-recorder" -Force
@"
{
  "port": 9002,
  "character": "$Character",
  "ascension": $Ascension,
  "seed": $seedJson,
  "num_runs": 99,
  "speed_patches": true,
  "agent_timeout_seconds": 0,
  "auto_start": true,
  "advisor_autoplay": true,
  "pause_on": "",
  "resume_run": false,
  "fast_mode": "instant",
  "time_scale": 4.0,
  "client_run_config": false,
  "debug_player_hp": 0,
  "overlay": true,
  "stall_timeout_seconds": 300,
  "think_fps": 10,
  "run_timeout_minutes": 360
}
"@ | Set-Content "$modDir\SpireMentor.conf" -Encoding UTF8

Get-Process -Name SlayTheSpire2 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
Get-Process -Name python, uv -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$deadline = (Get-Date).AddHours($MaxHours)
$attempt = 0
$won = $false

while ($attempt -lt $MaxAttempts -and (Get-Date) -lt $deadline -and -not $won) {
    $attempt++
    if (-not (Get-GameWindow)) {
        Say "launching game (attempt $attempt)"
        Start-Process -FilePath "$gameDir\SlayTheSpire2.exe" -WorkingDirectory $gameDir `
            -RedirectStandardOutput "$repo\sts2-pycli\logs\game_stdout.log" `
            -RedirectStandardError  "$repo\sts2-pycli\logs\game_stdout.log.err"
        for ($i = 0; $i -lt 90 -and -not (Get-GameWindow); $i++) { Start-Sleep -Seconds 1 }
        if (-not (Get-GameWindow)) { Say "FATAL: no game window"; break }
        Start-Sleep -Seconds 2
    }
    $offset = if (Test-Path $godotLog) { (Get-Item $godotLog).Length } else { 0 }

    Say "attempt $attempt/$MaxAttempts : starting capture"
    $recording = Start-Capture
    if (-not $recording) { Say "  capture could not start; recording this run is not possible" }

    # --- wait for this run to end -----------------------------------------
    # A death line is the ONLY negative signal. The positive one cannot be
    # `Victory!`: the 2026-08-10 win logged no victory line at all, because
    # with advisor_autoplay the SimAutoSlayer loop has already returned by the
    # time the run ends. What every ending does emit is the advisor disarming
    # itself ("Auto-play off: the run has ended") and the game-over screen. So
    # run-end is detected from those, and the outcome is decided by whether a
    # death was seen -- no death at run-end means the run was completed.
    $outcome = $null
    $attemptEnd = (Get-Date).AddMinutes($AttemptMinutes)
    $lastLen = Get-LogLength
    while (-not $outcome -and (Get-Date) -lt $deadline -and (Get-Date) -lt $attemptEnd) {
        Start-Sleep -Seconds 10
        # Runaway-log wedge, checked BEFORE reading: the game's own render loop
        # can lock onto a room and emit `Parameter "particles" is null`
        # forever (~4 MB/s -> 1.96 GB in nine minutes, floor 13, 2026-08-06).
        # It never advances, so waiting out the attempt cap costs 20 minutes
        # and gigabytes of disk. A healthy run writes a few hundred KB in
        # total, so 50 MB in one 10 s poll is unambiguous.
        $len = Get-LogLength
        if (($len - $lastLen) -gt 50MB) {
            $outcome = "WEDGED (log grew {0:N0} MB in 10s -- game render loop)" -f (($len - $lastLen)/1MB)
            break
        }
        $lastLen = $len
        $chunk = Read-LogSince $offset
        if ($chunk -match "has died to a MONSTER\.([A-Z_]+)") { $outcome = "LOSS ($($Matches[1]))"; break }
        if ($chunk -match "SimAutoSlayer\] Victory!")          { $outcome = "WIN (victory line)"; break }
        if ($chunk -match "Auto-play off: the run has ended" -or $chunk -match "NGameOverScreen") {
            $outcome = "ENDED, no death seen -> PROBABLE WIN"; break
        }
        # Checked LAST, and NOT a win: a process that died without emitting
        # run-end did not finish a run. This branch used to fall through to
        # the keep-on-ambiguity arm, so killing a wedged game by hand ended
        # the session "successfully" with a video of the wedge.
        if (-not (Get-GameWindow))                             { $outcome = "CRASHED (game exited before run end)"; break }
    }
    # STALLED is a third class, deliberately NOT folded into the
    # keep-on-ambiguity branch: a run that never ended is not a win, and every
    # real ending so far has emitted run-end within a few minutes.
    if (-not $outcome) {
        $outcome = if ((Get-Date) -ge $attemptEnd) { "STALLED (no run end in $AttemptMinutes min)" } else { "TIMEOUT" }
    }
    Say "attempt ${attempt}: $outcome"

    Stop-Capture $recording

    if ($recording) {
        $f = Get-Item $recording.FullName -ErrorAction SilentlyContinue
        # Everything that is not a completed run is discarded. Ambiguity must
        # not cost a win, but "the run never ended" is not ambiguity.
        $notARun = @("LOSS*", "STALLED*", "WEDGED*", "CRASHED*") | Where-Object { $outcome -like $_ }
        if ($notARun) {
            Say "  $(($outcome -split ' ')[0].ToLower()) -> deleting $($recording.Name)"
            Remove-Item $recording.FullName -Force -ErrorAction SilentlyContinue
        } else {
            # Not a confirmed death: keep it. Ambiguity must never cost a win.
            Say "*** KEEPING $($f.FullName) ($([math]::Round($f.Length/1MB,1)) MB) -- outcome: $outcome ***"
            $won = $true
        }
    }

    if (-not $won) {
        # Autoplay has disarmed and nothing will start another run in this
        # process, so the next attempt needs a fresh game. Kill it here rather
        # than at the top of the loop so the "did the game exit?" outcome check
        # above still means what it says.
        Say "  closing the game so the next attempt starts clean"
        Get-Process -Name SlayTheSpire2 -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 3
        # Godot rotates godot.log on the next launch and keeps four of them.
        # After a wedge that log is gigabytes, so drop it rather than let it
        # be preserved as a rotation.
        if ((Get-LogLength) -gt 100MB) {
            Say ("  discarding {0:N0} MB runaway godot.log" -f ((Get-LogLength)/1MB))
            Remove-Item $godotLog -Force -ErrorAction SilentlyContinue
        }
        Get-Process -Name python, uv -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }
}

if (-not $won) { Say "no winning run captured (attempts=$attempt)" }
Say "=== done ==="
