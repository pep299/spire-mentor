#!/bin/bash
# tests/test_wheel_venv_layout.sh — GH #506: wheel-venv.sh must see a venv in
# either layout.
#
# A POSIX venv is `bin/python` + `lib/python3.X/site-packages/sts2_rs/*.so`; a
# native Windows venv (the script under Git Bash) is `Scripts/python.exe` +
# `Lib/site-packages/sts2_rs/*.pyd`; MSYS2/mingw Python mixes the two. The
# script used to probe only the POSIX
# paths, so on Windows `ensure_venv` thought an existing venv was missing,
# `uv venv` refused to overwrite it, and the build aborted before the
# stale-wheel gate could run.
#
# Hermetic: fabricates both layouts under a temp dir and asks the
# `venv-layout` subcommand what the helpers resolve. Nothing is built.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WV="$REPO_DIR/scripts/wheel-venv.sh"

[ -x "$WV" ] || { echo "SKIP: wheel-venv.sh not found at $WV"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# A here-string, not `printf | grep -q`: grep -q exits at the first match,
# and under pipefail the SIGPIPE it leaves printf with (once $out outgrows
# the pipe buffer, as case 2b's 300 modules do) fails the pipeline (#1093).
expect_line() {
  local label="$1" out="$2" want="$3"
  if grep -qxF -- "$want" <<<"$out"; then
    echo "ok: $label"
  else
    echo "FAIL: $label — wanted line '$want' in:"; printf '   %s\n' "$out"; fail=1
  fi
}

# 1) POSIX layout.
posix="$TMP/posix"
mkdir -p "$posix/bin" "$posix/lib/python3.13/site-packages/sts2_rs"
printf '#!/bin/sh\n' > "$posix/bin/python"; chmod +x "$posix/bin/python"
touch "$posix/lib/python3.13/site-packages/sts2_rs/sts2_rs.cpython-313-x86_64-linux-gnu.so"
out="$("$WV" venv-layout "$posix" 2>&1)"
expect_line "posix python" "$out" "python=$posix/bin/python"
expect_line "posix bin" "$out" "bin=$posix/bin"
expect_line "posix ext" "$out" "ext=$posix/lib/python3.13/site-packages/sts2_rs/sts2_rs.cpython-313-x86_64-linux-gnu.so"

# 2) Windows layout — the case the old probes could not see.
win="$TMP/win"
mkdir -p "$win/Scripts" "$win/Lib/site-packages/sts2_rs"
printf '#!/bin/sh\n' > "$win/Scripts/python.exe"; chmod +x "$win/Scripts/python.exe"
touch "$win/Lib/site-packages/sts2_rs/sts2_rs.cp313-win_amd64.pyd"
out="$("$WV" venv-layout "$win" 2>&1)"
expect_line "windows python" "$out" "python=$win/Scripts/python.exe"
expect_line "windows bin" "$out" "bin=$win/Scripts"
expect_line "windows ext" "$out" "ext=$win/Lib/site-packages/sts2_rs/sts2_rs.cp313-win_amd64.pyd"

# 2b) MSYS2/mingw Python: POSIX dirs, Windows suffix. The extension must
#     still be found (banner + stale-wheel self-heal).
msys="$TMP/msys"
mkdir -p "$msys/bin" "$msys/lib/python3.12/site-packages/sts2_rs"
printf '#!/bin/sh\n' > "$msys/bin/python"; chmod +x "$msys/bin/python"
touch "$msys/lib/python3.12/site-packages/sts2_rs/sts2_rs.cp312-mingw_x86_64_msvcrt_gnu.pyd"
out="$("$WV" venv-layout "$msys" 2>&1)"
expect_line "msys python" "$out" "python=$msys/bin/python"
expect_line "msys ext" "$out" "ext=$msys/lib/python3.12/site-packages/sts2_rs/sts2_rs.cp312-mingw_x86_64_msvcrt_gnu.pyd"

# 2c) Both interpreters present: `bin/python` wins, on every platform.
both="$TMP/both"
mkdir -p "$both/bin" "$both/Scripts"
printf '#!/bin/sh\n' > "$both/bin/python"; chmod +x "$both/bin/python"
printf '#!/bin/sh\n' > "$both/Scripts/python.exe"; chmod +x "$both/Scripts/python.exe"
out="$("$WV" venv-layout "$both" 2>&1)"
expect_line "both-layouts python prefers bin/" "$out" "python=$both/bin/python"

# 2d) Many extension modules: the banner's first-module lookup must not die of
#     SIGPIPE under `pipefail` (it was `venv_ext_modules | head -1`).
many="$TMP/many"
mkdir -p "$many/bin" "$many/lib/python3.13/site-packages/sts2_rs"
printf '#!/bin/sh\n' > "$many/bin/python"; chmod +x "$many/bin/python"
for n in $(seq -w 1 300); do
  touch "$many/lib/python3.13/site-packages/sts2_rs/sts2_rs.m$n.so"
done
out="$("$WV" venv-layout "$many" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL: venv-layout with 300 extension modules exited $rc"; fail=1
fi
expect_line "many-modules first" "$out" "first=$many/lib/python3.13/site-packages/sts2_rs/sts2_rs.m001.so"

# 3) No venv yet, no extension: the platform default, and no ext lines (the
#    banner's "(not built yet)" path must not trip `set -e`).
missing="$TMP/missing"
out="$("$WV" venv-layout "$missing" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL: venv-layout on a missing venv exited $rc:"; printf '   %s\n' "$out"; fail=1
fi
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*) want_py="$missing/Scripts/python.exe" ;;
  *) want_py="$missing/bin/python" ;;
esac
expect_line "missing-venv default python" "$out" "python=$want_py"
expect_line "missing-venv no first module" "$out" "first="
if grep -q '^ext=' <<<"$out"; then
  echo "FAIL: a missing venv reported an extension module:"; printf '   %s\n' "$out"; fail=1
else
  echo "ok: missing venv has no ext lines"
fi

# 4) A worktree build points `.venv` at its `.venv-h/<hash>` venv and re-points
#    it when the hash moves. It must be a LINK: on Windows, Git Bash's `ln -s`
#    deep-copied the venv instead, and every later build left that real dir
#    as-is, so `.venv` stayed stale.
pd="$TMP/pycli"
mkdir -p "$pd/.venv-h/aaa" "$pd/.venv-h/bbb"
touch "$pd/.venv-h/aaa/A" "$pd/.venv-h/bbb/B"
"$WV" relink-dot-venv "$pd" aaa >/dev/null 2>&1
"$WV" relink-dot-venv "$pd" bbb >/dev/null 2>&1
if [ -L "$pd/.venv" ] && [ -e "$pd/.venv/B" ] && [ ! -e "$pd/.venv/A" ] \
   && [ -e "$pd/.venv-h/aaa/A" ]; then
  echo "ok: .venv is a link, re-pointed, old venv intact"
else
  echo "FAIL: .venv after relink:"; ls -la "$pd" "$pd/.venv" 2>&1 | sed 's/^/   /'; fail=1
fi

if [ "$fail" -eq 0 ]; then echo "▶ test_wheel_venv_layout: PASS"; else
  echo "▶ test_wheel_venv_layout: FAIL"; exit 1; fi
