// Combatant's pure layout and display math, kept out of Combatant.tsx so that
// file exports only the component (react-refresh/only-export-components).
import type { Power } from "../api";
import type { DissolveEffect } from "../effects/manifest";
import { instanceScale, renderedSize } from "../scene";

/** What a nameplate shows for `name` with board label `label`: the bare name
 *  — debug size suffixes ("Twig Slime (M)") stay OFF the scene — plus the
 *  label's duplicate number (" #2"), so the scene tells same-named enemies
 *  apart. `full` (the whole label, size tag included) is the tooltip and the
 *  accessible name whenever the visible text dropped or added anything. */
export function nameplateText(name: string, label?: string): { text: string; full?: string } {
  const full = label ?? name;
  const dup = label && label !== name && label.startsWith(name) ? label.slice(name.length) : "";
  const m = /\s*\((?:XS|S|M|L|XL|B)\)\s*$/.exec(name);
  const text = (m ? name.slice(0, m.index) : name) + dup;
  return { text, full: text !== full ? full : undefined };
}

/** Global combat sprite scale. The r1 visual gauntlet read the combatants
 *  ~30% under the game's reference framing; multiplying the authored world
 *  scale at the call sites keeps the shared SpineSprite renderer untouched.
 *  (Board.tsx applies the same factor to the shared-rig boss groups.)
 *
 *  r5 MEASURED it instead of eyeballing. In refs/game-combat-1.png the
 *  Ironclad's body is 279px of a 1005px-tall scene (27.8%) and the slime 136px
 *  (13.5%); at 1.28 our own capture read 25.5% / 12.3% of the stage — and,
 *  because our stage is only ~62% of the game's scene height, barely half the
 *  game's ABSOLUTE on-screen size, which is what both r4 critics graded
 *  ("enemies roughly half the game's size", two rounds running). 1.68 lands the
 *  hero at 27.4% of the stage against the ref's 27.2% of its scene, measured
 *  the same way both times (topmost row whose tan-armour pixel count clears
 *  30% of the body's widest row — the raised sword's hilt is thin enough to
 *  fall under that, so an idle-loop pose difference cannot move the number).
 *  1.7+ clipped the raised sword at 1366x768. */

/**
 * The per-hit randomness `NDamageNumVfx._Ready` rolls: a spawn jitter, a
 * sideways kick, a slightly varied throw and a small tilt. Without it every
 * number of a multi-hit attack traces the identical path and they stack into
 * one illegible blob.
 *
 * Seeded off the effect nonce rather than `Math.random()` so a re-render
 * mid-flight doesn't teleport a number that is already in the air — the CSS
 * animation keeps running, but its custom properties must not move under it.
 * Values are the game's, scaled to our shorter board (see game-skin.css).
 */
export function damageNumberJitter(nonce: number): React.CSSProperties {
  // xorshift on the nonce — any cheap decorrelating hash would do; consecutive
  // nonces must not produce visibly similar throws.
  let h = (nonce * 2654435761) >>> 0;
  const next = () => {
    h ^= h << 13; h >>>= 0;
    h ^= h >> 17;
    h ^= h << 5; h >>>= 0;
    return h / 4294967296;
  };
  const span = (a: number, b: number) => a + next() * (b - a);
  return {
    "--dmg-jx": `${span(-6, 6).toFixed(1)}px`,
    "--dmg-jy": `${(-55 + span(-3, 3)).toFixed(1)}px`,
    "--dmg-dx": `${span(-66, 66).toFixed(1)}px`,
    "--dmg-vy": span(0.93, 1.07).toFixed(3),
    "--dmg-rot": `${span(-5, 5).toFixed(1)}deg`,
  } as React.CSSProperties;
}

/** How far the rig reaches above its root, in SCENE px — the height of the box
 *  the combatant column reserves for the sprite, whose bottom is the ground the
 *  creature stands on.
 *
 *  `rig_bounds_h` (the game's authored `Visuals.Bounds`) is the FALLBACK, not
 *  the answer. It is the creature's hitbox, and a hitbox drawn a little roomier
 *  than the body — a median 1.07x the drawn height since GH #611 corrected it
 *  from the ~2x that once hung the intent chip a body-length above the monster.
 *  Close is not the thing the overlays pin to, though, and the rig's own box is
 *  exact and equally free: it comes from the generated table, so it is known
 *  before the atlas pages have decoded and the slot never resizes on load. */
export function spriteBodyH(
  c: {
    rig?: string | null; rig_scale?: number | null; rig_bounds_h?: number | null;
    size_jitter?: number | null;
  },
  isEnemy: boolean,
): number {
  const size = renderedSize({ rig: c.rig, rigScale: c.rig_scale, boundsH: c.rig_bounds_h });
  const jitter = instanceScale(c);
  if (size) return size.top * jitter;
  return (c.rig_bounds_h ?? (isEnemy ? FALLBACK_BODY_H.enemy : FALLBACK_BODY_H.player)) * jitter;
}

/** Fallback body height (SCENE px) for a combatant whose rig has no authored
 *  scale — a rig staged without a `creature_scales.json` entry. Real creatures
 *  never take this path; it only keeps such a rig from rendering at 1:1 rig
 *  units, which would be several screens tall. */
export const FALLBACK_BODY_H = { enemy: 300, player: 440 };

/** HP-bar length in px for a given max HP. The game sizes the bar to the
 *  creature rather than to a number, but a fixed width (r3's 150px for
 *  everything) reads as a web progress bar the moment two enemies of different
 *  sizes stand side by side. sqrt gives visible separation at the low end
 *  (13 HP -> 106, 27 -> 123) without a boss's bar running off the stage. */
export function hpBarWidth(maxHp: number): number {
  const w = 66 + Math.sqrt(Math.max(1, maxHp)) * 11;
  return Math.round(Math.max(96, Math.min(196, w)));
}

/** The creature's Doom counter, or 0. Doom kills its owner at the end of their
 *  side's turn once `CurrentHp <= Amount` (decomp ``DoomPower.IsOwnerDoomed``),
 *  and the game paints that threshold ONTO the health bar (``NHealthBar``'s
 *  `%DoomForeground`) rather than leaving it to be read off a status pip — the
 *  bar is where you see how close the kill line is.
 *
 *  Matched on the engine id first: `name` is a localized display title, so a
 *  name comparison is the same trap that froze the Test Subject in form 1. */
export function doomAmount(powers?: Power[]): number {
  const p = powers?.find(
    (x) => (x.id ?? "").toUpperCase() === "DOOM" || x.name.trim().toUpperCase() === "DOOM",
  );
  return Math.max(0, p?.amount ?? 0);
}

/** How much the creature's BODY is drawn at, from its powers. Shrink is the
 *  one power that resizes a creature: `ShrinkPower.AfterApplied` calls
 *  `NCreature.ScaleTo(0.5f, 0.75)` (and `AfterRemoved` scales back to 1) —
 *  players included, since `ScaleTo` only refuses a monster whose
 *  `CanChangeScale` is false (`DecimillipedeSegment`, which is drawn as a
 *  `noSprite` overlay here anyway). `ScaleTo` tweens `Visuals.Scale`, i.e. the
 *  sprite subtree only — the health bar, nameplate and power row keep their
 *  size and place. */
export function bodyScale(powers?: Power[]): number {
  const shrunk = powers?.some(
    (x) => (x.id ?? "").toUpperCase() === "SHRINK" || x.name.trim().toUpperCase() === "SHRINK",
  );
  return shrunk ? 0.5 : 1;
}

/** Chrome (HP bar / name / powers) fade duration for a dying enemy's
 *  `.gs-dying` wrapper, before the manifest has loaded (or when it lacks
 *  the `vfx_monster_death` entry) — the pre-Task-14 CSS timing (0.9s
 *  animation + its 0.1s delay), and the SAME fallback `Board.tsx`'s
 *  corpse-removal timer uses, so a manifest-less board removes a corpse
 *  exactly when its chrome finishes fading.
 *
 *  Not to be confused with `game-skin.css`'s `var(--gs-dying-ms, 2.5s)`,
 *  which is a different fallback for a different failure: that one covers
 *  the custom property being unset at all (nothing here rendered the
 *  corpse), and 2.5s is the shipped manifest's real
 *  `vfx_monster_death.dissolveSeconds`. This 0.9s covers the property being
 *  set from a `dissolve` of `null` — no dissolve is running, so the old
 *  chrome-only timing is what should play. The numbers differ on purpose. */
export const DEFAULT_DYING_FADE_SECONDS = 0.9;

/** The `--gs-dying-ms` custom property `game-skin.css`'s `gs-chrome-fade`
 *  keyframe reads (Task 14 fix round 1): the WebGL dissolve owns the sprite
 *  now (`SpineSprite`'s `dissolve` prop), so `.gs-dying`'s own CSS animation
 *  is scoped to the surrounding DOM chrome only (`:not(.gs-sprite-wrap)` in
 *  the stylesheet) — but it still needs to run for the SAME duration as the
 *  dissolve, or the chrome and the sprite finish at different times. Matches
 *  `dissolve.effect.dissolveSeconds` exactly (the same value `SpineSprite`
 *  feeds `dissolveThreshold`), falling back to `DEFAULT_DYING_FADE_SECONDS`
 *  when `dissolve` is `null`/`undefined` (dying but the manifest hasn't
 *  resolved `vfx_monster_death` yet, or at all).
 *
 *  `deathAnimSeconds` is the third case, and it is not a fallback for a
 *  failure: a monster whose `ShouldFadeAfterDeath` is false gets NO dissolve
 *  from the game at all (`NCreature.cs:1106`), so `dissolve` is correctly
 *  `null` — but its corpse still lives for `DeathAnimLengthOverride`
 *  (`:1093`), which is what the chrome must fade over. Ordered above the 0.9s
 *  fallback and below a real dissolve, so a monster that both dissolves and
 *  sets an override (none do at v0.107.1) would follow the dissolve, matching
 *  the C#, where the override only applies on the `_spineAnimator == null`
 *  branch. */
export function dyingChromeFadeSeconds(
  dissolve?: { effect: DissolveEffect } | null,
  deathAnimSeconds?: number | null,
): number {
  return dissolve?.effect.dissolveSeconds ?? deathAnimSeconds ?? DEFAULT_DYING_FADE_SECONDS;
}
