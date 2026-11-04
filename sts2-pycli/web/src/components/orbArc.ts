// Where the orbs sit around a Defect, in the game's own geometry — kept out of
// the component file so it exports only the component (react-refresh/only-
// export-components).

// Belt geometry, ported from `NOrbManager.TweenLayout` (v0.107.1). The game
// sweeps the queue across a 125° arc that stops 25° above the horizontal on
// each side, so slot 0 — the next orb to evoke — sits out on the creature's
// RIGHT, the last slot on its LEFT, and the middle of the queue rides over the
// crown. The radius GROWS with the queue's capacity, so a 10-slot belt is a
// wider ring than the starting 3 rather than a more crowded one.
const GAME_ARC_DEG = 125; // NOrbManager._range
const GAME_ARC_END_DEG = -25; // NOrbManager._angleOffset
const GAME_R_MIN = 225; // NOrbManager._minRadius, at capacity 3
const GAME_R_MAX = 300; // NOrbManager._maxRadius, at capacity 10

/** The player's body height in the same game px those radii are in: the
 *  Ironclad's crown to his boots, measured off `refs/game-combat-1.png` (the
 *  game at 1080p). The belt hangs around a CREATURE, so it is scaled to our
 *  creature and not to our stage — the workbench draws combatants ~1.6× larger
 *  relative to the frame than the game does, and a stage-scaled belt would
 *  float clear of the character it belongs to. */
export const GAME_BODY_H = 245;

/** How big an orb is drawn, in the same game px the radii above are in.
 *
 *  Read off `scenes/orbs/orb.tscn`, NOT off the art: the `%Outline` ring and
 *  the `Bounds` hitbox are both a 60x60 rect, and `expand_mode = 1` means the
 *  100x100 `empty_orb.png` (which is what `orb_atlas/*.png` mirrors) is SCALED
 *  DOWN into it. `NOrb._Ready` then multiplies the whole control by 0.85 for
 *  the local player. Taking the 100px art at face value — which is what an
 *  earlier pass did — draws every orb 1.67x too big, big enough to sit on the
 *  character's head instead of orbiting it. */
const GAME_ORB_PX = 60 * 0.85;

/** …and as a fraction of the body, because the belt is scaled to our creature
 *  rather than to our stage (see `GAME_BODY_H`). */
export const ORB_TO_BODY = GAME_ORB_PX / GAME_BODY_H;

/** How far ABOVE the creature's root the arc is CENTRED, as a fraction of the
 *  drawn body.
 *
 *  The game does not sweep the belt around the creature's root. `NOrbManager`
 *  lays its orbs out around ITSELF, and `NCreature.SetOrbManagerPosition` hangs
 *  the manager on a marker: `Visuals.OrbPosition`, which
 *  `NCreatureVisuals._Ready` resolves to the creature scene's `%OrbPos` — or,
 *  for a scene that authors none, to `%IntentPos`, the marker the intent icon
 *  hangs on just over the crown. Centring the sweep on the root instead, which
 *  is what this did, is what left the queue's ENDS — slot 0, the next orb to
 *  evoke, and the last slot — out beside the character's HIPS, with only the
 *  middle of the queue as high as its head.
 *
 *  The markers themselves live in `scenes/creature_visuals/*.tscn` inside the
 *  PCK, which is not vendored here (`creature_scales.json` carries the `%Bounds`
 *  box, not the markers), so this is the one number in the belt that is
 *  calibrated rather than ported. 0.6 puts the arc's low ends level with the
 *  crown and its middle half a body above that — the halo the game draws over
 *  the Defect — where the game's own `%IntentPos` fallback (1.0, i.e. the crown)
 *  lifts the whole belt clear of the character with a gap under it. Replace it
 *  with the authored marker if the PCK is ever extracted for it. */
export const ORB_CENTRE_TO_BODY = 0.6;

/** Where each of `n` slots sits, in px relative to the creature's FEET (the
 *  origin `.gs-orbs` is pinned to), for a creature drawn `bodyH` px tall.
 *  `+y` is down (Godot and CSS agree), so every slot comes out negative. */
export function orbArc(n: number, bodyH: number): { x: number; y: number }[] {
  // Godot's Mathf.Lerp doesn't clamp and neither does this: a degenerate
  // capacity tightens the ring rather than snapping to the 3-slot radius.
  const r = (GAME_R_MIN + (GAME_R_MAX - GAME_R_MIN) * ((n - 3) / 7)) * (bodyH / GAME_BODY_H);
  const cy = -bodyH * ORB_CENTRE_TO_BODY;
  return Array.from({ length: n }, (_, i) => {
    // The game's sweep counter runs 125° -> 0° across the queue. A one-slot
    // belt has no step to take (it divides by `capacity - 1`); park it over the
    // crown instead of at the game's degenerate far end.
    const swept = n > 1 ? GAME_ARC_DEG * (1 - i / (n - 1)) : GAME_ARC_DEG / 2;
    const a = ((GAME_ARC_END_DEG - swept) * Math.PI) / 180;
    return { x: -Math.cos(a) * r, y: cy + Math.sin(a) * r };
  });
}
