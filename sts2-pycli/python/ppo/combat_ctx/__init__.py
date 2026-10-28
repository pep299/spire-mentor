"""Context-aware combat student, step 0 of GH #830 (DESIGN.md / PREREG.md).

Is model context the bottleneck for learning what the deployed search knows?
Today's pilot (honest BeliefDraw MCTS, 1000/4000 sims, rl2 playouts) labels
human-bank fights (`gen`); three students learn the same targets on the same
split: rl2's own architecture over rl2's 32 + 24 features (`S_rl2arch`), a
transformer over those same features only (`S_ctxarch_rl2in`), and the same
transformer over the full honest combat state as tokens (`S_ctx`). They are
scored by the teacher's root-Q regret against the teacher's seed-to-seed noise
(`metrics`), by a fair re-search of their disagreements (`referee`), and by
playing whole held-out boss fights with no search (`play`).

Modules: `snapshot` (the honest view), `intent_log` (displayed-intent
history), `static_tables` (decomp-pinned card types / ShowCounter relics),
`tokens` (token schemas), `gen` (teacher data), `power` (split-scheme
projection), `data` (token cache), `model` (`CombatCtxNet`), `train`
(every student), `metrics` (regret / agreement / CIs / sealing), `referee`,
`play` (play-alone driver and the rl2 anchor). Nothing here changes a default:
it is a new package with its own CLI (`python -m ppo.combat_ctx`).
"""
