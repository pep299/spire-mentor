#!/usr/bin/env bash
# coop_watch.sh -- watch a coop_test.ps1 run unattended; one line per event.
#
# Usage: coop_watch.sh <log-dir> [stall-seconds]
#
# Emits, from every instance's godot log (host, dummy, dummy2, dummy3) and
# the advisors' stdout logs:
#   - run-level picks (map / event / rest / treasure / shop) as a heartbeat
#   - errors, refusals, desync / checksum mismatches, failed saves
#   - advisor failures (a reply with no move: the run waits on it), from
#     advisor_stdout.log and each dummy's advisor_stdout.<port>.log
#   - STALL when no instance executed an AI move for stall-seconds
#     (default 240); repeated while it lasts
#   - GAME EXITED when fewer game processes remain than the run started
#     with (one per godot log)
# Silence is never "fine": a hung run produces STALL lines.
set -uo pipefail
dir="$1"
stall="${2:-240}"
advisor_dir="${LOCALAPPDATA:-$HOME/AppData/Local}/SpireMentor"

# The instances start one by one and each creates its log when it starts.
# coop_test.ps1 writes the run's player count to players.txt; wait (up to
# 3 minutes) until that many logs exist, else go with what is there.
want=$(tr -dc '0-9' < "$dir/players.txt" 2>/dev/null)
want=${want:-2}
for _ in $(seq 180); do
  shopt -s nullglob
  logs=("$dir"/*.godot.log)
  shopt -u nullglob
  (( ${#logs[@]} >= want )) && break
  sleep 1
done
players=${#logs[@]}
if (( players < want )); then
  echo "WARNING: $players of $want instance logs after 3 min; watching those"
fi
echo "watching $players instance(s): ${logs[*]##*/}"

# `--pid=$$`: the tail exits with this script, even when the script is killed
# outright and no trap runs (a TaskStop'd watcher once left six tails holding
# the logs open for hours, 2026-10-29). Process substitution makes $! the
# tail itself, not the last grep.
tail -q -F -n0 --pid=$$ "${logs[@]}" "$advisor_dir/advisor_stdout.log" "$advisor_dir"/advisor_stdout.9*.log 2>/dev/null \
  > >(grep -v --line-buffered "Generating checksum" \
      | grep -E --line-buffered \
          "Takeover executed on N(Map|EventRoom|RestSite|Treasure|MerchantRoom)|refuses|refused|without a move|Exception|\[ERROR\]|Failed to save|Disconnect|[Dd]esync|[Dd]ivergen|[Mm]ismatch|has lost|Victory|runaway|disarm|not available in co-op|advice_request failed|UnsupportedStateError") &
tail_pid=$!
trap 'kill $tail_pid 2>/dev/null' EXIT

last_count=-1
idle=0
while sleep 30; do
  n=$(cat "${logs[@]}" 2>/dev/null | grep -ac "Takeover executed")
  if [[ "$n" != "$last_count" ]]; then
    last_count=$n
    idle=0
  else
    idle=$((idle + 30))
    if (( idle >= stall && idle % stall < 30 )); then
      echo "STALL: no AI move in any instance for ${idle}s (moves so far: $n)"
    fi
  fi
  procs=$(tasklist 2>/dev/null | grep -ci "SlayTheSpire2")
  if (( procs < players )); then
    echo "GAME EXITED: $procs of $players game process(es) left"
    exit 0
  fi
done
