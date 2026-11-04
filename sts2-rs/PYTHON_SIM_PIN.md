# Python sim pin — HISTORICAL

> **Status (2026-08): the pin regime is retired.** The `sts2-rl-agent` Python
> sim is archived and **no longer checked out** anywhere on disk; the
> pin-verification and fixture-recording scripts that depended on it have
> been deleted (this repo has no GitHub Actions CI — local
> nextest/clippy/fmt are the only gates).
> The parity target today is the C# decompilation at
> `../sts2-wiki/decompiled/v0.107.1/` via the C# oracle harness in
> `../sts2-pycli` (see `docs/parity-suite.md` there).
>
> **2026-10-30: the per-file port-pin headers are gone.** The
> `// Ported from sts2-rl-agent@…` / `Port-pin:` header lines, the
> `// port-pin-skip` markers and `scripts/check-port-pins.sh` were removed:
> the engine has been rewritten against the C# decomp, and only its early
> combat-loop scaffolding followed the Python sim. Inline doc comments that
> cite a `sts2-rl-agent@01c094d:…` path remain as historical
> cross-references; they are not a statement of where the current code
> comes from, and the C# decomp wins wherever the two disagree.

The Rust port originally parity-tested against the Python simulator at
`sts2-rl-agent/sts2_env/`. The commit was pinned so port targets didn't
drift under the port.

## Final pin (historical provenance)

- Repo: `sts2-rl-agent` (archived, not checked out)
- Commit: `01c094d87056dfca351d002fe2af231c54f5d163`
- Short sha: `01c094d`
- Pinned date: 2026-05-06
- Commit subject: "Expand parity coverage and align gameplay semantics"

The pin was never bumped and never will be — do not "fix" the remaining
inline citations to a newer sha, do not add new ones, and do not resurrect
the old bump rule (which predated the
nextest-only rule and instructed `cargo test`).

## Retired verification (tooling deleted)

`scripts/check-python-sim-pin.sh` (and its `.fish` wrapper) verified the
sibling `../sts2-rl-agent` checkout was at the pinned sha. With the checkout
gone that check became permanently unsatisfiable, so the script and its
callers — the fixture recorder `scripts/record_fixture.py`, the Cultist fuzz
loop `scripts/run_fuzz_cultist.sh` / `scripts/fuzz_cultist.py`, and
`scripts/verify_mcts_2sigma.py` — were deleted in 2026-08. The parity
fixtures they recorded (`tests/parity/scripted/`, `tests/parity/fuzz/`) are
**frozen**: they remain valid regression pins of the recorded ground truth,
but there is no way to re-record them and none is planned.
