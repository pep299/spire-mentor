# Integration Hub

Where the real game's code, the Rust engine, the training loop and the
workbench meet. The most entangled context, by design.

Terms shared with the rest of the monorepo live in
[../CONTEXT-MAP.md](../CONTEXT-MAP.md#shared-language).

## Language

### Running the real game

**Headless oracle**:
The vendored, IL-patched game DLLs hosted with Godot stubbed out, so the real game code runs scriptably and deterministically. The *reference* in every parity comparison.
_Avoid_: bare "oracle" — the word is overloaded across this repo; also avoid the sim, the C# sim

**Godot stubs**:
The clean-room replacement for GodotSharp, supplying no-op types so game code loads without an engine or a display.
_Avoid_: mocks, shims, fakes

### Parity

Bare **"parity"** is banned. It names three different harnesses that fail in
different ways, and an unqualified claim about "parity" cannot be acted on.
Always pick one:

**Combat parity**:
A single scripted fight compared against a frozen fixture. The gate every new mechanic crosses.

**Run parity**:
A whole run driven through both implementations in lockstep from one seed, checking that the RNG streams and state stay aligned.

**State parity**:
A full-state diff taken at each decision point across a range of runs, rather than at the end.

**State trace**:
The C# oracle's record of every write to HP, block, powers, piles, energy and the named RNG streams, each tagged with the model method that made it. `act23_state_parity --trace-split` uses it to explain a run's first full-state split with the C# change history of each diverging field. Only C# is traced; the Rust side contributes its end value.

**Residual**:
A divergence that survives a parity sweep and still needs explaining. A count of residuals is a work list, not a bug count.
_Avoid_: failure, diff, mismatch

**REAL_BUG**:
A residual where the engine is wrong and the game is right. Actionable.

**ORACLE_ARTIFACT**:
A residual caused by the harness or the headless oracle's own limits, not by an engine defect. Not actionable against the engine.
_Avoid_: false positive, noise

### Measurement

**Cell**:
One actually-played unit of an evaluation grid — a single (scenario, knob level) fight or run. Never carried forward or interpolated. Quote the cell count beside every result.
_Avoid_: sample, data point, trial

**Arm**:
One side of a comparison — the configuration being tested, or the control it is tested against.
_Avoid_: variant, branch, condition

**Paired**:
Both arms measured on identical seeds, so the difference between them isn't seed luck. Whether a comparison is paired must be stated; an unpaired difference means much less.
_Avoid_: matched, controlled

**pp**:
Percentage points of win rate. Always qualified by character, ascension, panel, cell count, agent and search budget — an unqualified pp figure is not a result.
_Avoid_: percent, %

**Sweep**:
A grid of cells played across a knob's levels to find where it stops paying.
_Avoid_: scan, search, experiment

### Training and promotion

**Feature probe**:
An offline check, taking minutes, of whether a proposed input carries signal at all. May never promote anything.
_Avoid_: ablation, pilot

**Screen**:
A paired two-arm run at reduced combat budget, taking about an hour, that decides whether an idea is worth a campaign. May never promote anything.
_Avoid_: trial, pilot, quick test

**Campaign**:
A full training run, 31–82 wall-clock hours, that can produce a promotable candidate.
_Avoid_: run, job, experiment

**Champion**:
The promoted run policy currently serving a character. `data/ppo/champions.toml` is the single source of truth; a net that isn't named there is not a champion no matter what's on disk.
_Avoid_: best model, current net, production model

**Promotion**:
The act of making a candidate the champion, which regenerates every downstream serving lane from `champions.toml`.
_Avoid_: deploy, release, ship

### Surfaces

**Workbench**:
The web app for playing, replaying and inspecting runs. Deployed as the
browser-only build on Cloudflare Pages; the Cloud Run deployment was retired
2026-09-08.
_Avoid_: the UI, the app, the viewer

**Advisor**:
The frozen executable that serves recommendations to the live game through the bridge.
_Avoid_: helper, assistant, hint server
