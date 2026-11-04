# Live-Game Bridge

The C# mod loaded into the running game, exposing its state and accepting
commands over TCP. Compiles with zero sibling files; couples to the
Integration Hub only when deploying or packaging.

Terms shared with the rest of the monorepo live in
[../CONTEXT-MAP.md](../CONTEXT-MAP.md#shared-language).

## Language

**Spire Mentor**:
The mod itself, running inside the real game process. Identifier `SpireMentor`
(mod id, DLL, namespace, conf file, log prefix). Called **Spire Coach** in
0.3.0 and **SimBridge** before that, in a folder named `sts2-bridge/`;
historical logs and experiment records keep those names.
_Avoid_: the mod, the plugin, the server, SimBridge, Spire Coach

**Protocol**:
The TCP message contract between the mod and its clients, specified in `PROTOCOL.md`. The contract is the specification — the client and the mod ship separately and cannot assume a shared version.
_Avoid_: API, interface, wire format

**Autoplay**:
The mode where the bridge acts on the advisor's recommendations itself, without a human in the loop.
_Avoid_: bot mode, auto, AI mode

**Bug report**:
A frozen moment sent from the game (`Ctrl+Shift+B`) or a workbench (🐞 Report):
the position, the start-of-room save, the run log and log tails, in the
workbench's schema-1 dump. One store for all of them, keyed `wb-<16 hex>`.
_Avoid_: crash report, feedback (the game's own F2 feedback screen goes to Mega Crit)

**Agent gate**:
The check governing whether an automated client may act on the live game at all.
_Avoid_: guard, lock, permission
