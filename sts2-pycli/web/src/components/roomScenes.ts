// Which combat room a fight happens in, and where each of its painted layers
// sits on the shared canvas.
//
// The game does NOT key the backdrop on the act alone. `EncounterModel` lets an
// encounter declare `HasCustomBackground`, and thirteen of them do — so the
// Insatiable fights on a sand floor among stumps, the Waterfall Giant in its
// own gorge, Vantom in an oil-black cavern, and the Kin in front of a lit brazier
// that the plain Overgrowth room does not have. Keying on the act only, as this
// module's predecessor did, drew the same four rooms for every fight in the game.
//
// The manifest itself is derived from the PCK, not hand-written — see
// `roomScenes.data.ts` and `sts2-wiki/scripts/extract_bg_layers.py`.
import { ROOM_MANIFEST } from "./roomScenes.data";
import { backgroundRng } from "../megaRng";
import type { CSSProperties } from "react";
import type { Enemy } from "../api";

/** One painted layer: an art file and where its scene puts it on the 1920x900
 *  canvas.
 *
 *  `left`/`top` is the art's OWN origin and `w`/`h` its two edge lengths, so a
 *  layer with a `t` is not a box at (`left`, `top`) — it is that box turned by
 *  `t` about its own top-left corner. That split is the game's: a `Control` has
 *  no flip, so these scenes mirror a prop with `rotation = 3.14159` and the art
 *  lands on the far side of the origin. See `roomLayerStyle`. */
export interface RoomLayer {
  /** Path under the served `rooms/` tree, e.g. `hive/the_insatiable/stump1.png`. */
  src: string;
  left: number;
  top: number;
  w: number;
  h: number;
  /** `[a, b, c, d]` — the rotation/mirror left once `w`/`h` carry the scale.
   *  Absent on an axis-aligned layer, which is nearly all of them. */
  t?: number[];
  /** Inherited `modulate` alpha. Absent at full opacity. */
  alpha?: number;
  /** Inherited `modulate` tint, multiplied over the art. Absent at white. */
  tint?: number[];
  /** A static stand-in for a particle emitter that hangs room art (Glory's
   *  cloud bank, the fog every act room drifts), rather than a painted
   *  layer — see `_particle_layers`. */
  vfx?: boolean;
  /** The emitter's `CanvasItemMaterial.blend_mode`. `"add"` is the additive
   *  fog and cloud work: it LIFTS what is behind it and covers nothing. */
  blend?: string;
  /** `[index, count]` for a sheet cut into `particles_anim_v_frames` frames:
   *  the art is one frame of a vertical strip, not the whole file. */
  frame?: number[];
}

/** One of the variants the game rolls between for a single layer slot.
 *  A variant can paint MORE than one file — the Underdocks' ruin band comes
 *  with its own drop shadow, the Kin's brazier with its flame. */
export interface RoomVariant {
  id: string;
  layers: RoomLayer[];
}

/** A slot in the room scene, in draw order.
 *
 *  `bucket` and `fg` are filled by a roll; `prop` is art the room scene paints
 *  itself, and it must stay in ITS place in the order — the Insatiable's sand
 *  floor covers the Hive backdrop but the stumps stand on the floor. */
export type RoomSlot =
  | { kind: "bucket"; index: string }
  | { kind: "fg" }
  | { kind: "prop"; name: string; layers: RoomLayer[] };

/** How much this room's air colours what is behind it.
 *
 *  Combat haze is pure particle work in the game, so a straight layer stack
 *  renders these rooms with the weather taken out: the Insatiable's fight is
 *  brown dunes under brown canopy in a game screenshot and dark GREEN canopy
 *  without its 16k sand motes. `rgb` is the tint the room hangs in the air,
 *  `strength` how thickly. Only rooms the artists actually filled with weather
 *  have one — see `atmosphere()` in the extractor. */
export interface RoomAtmosphere {
  rgb: [number, number, number] | number[];
  strength: number;
  particles: number;
}

export interface RoomScene {
  /** Scene px per canvas px — the size THIS room draws its canvas at.
   *
   *  The game ships two: 1.44 for the Overgrowth/Underdocks family and 1.50078
   *  for the Hive/Glory one. Both keep the canvas's 1920:900 aspect, so the
   *  layers still land on one canvas; it is only how big that canvas is drawn
   *  that differs. A single fitted 1.44 for every room drew the Hive and Glory
   *  families 4.2% small. */
  scale: number;
  atmosphere: RoomAtmosphere | null;
  slots: RoomSlot[];
  /** Layer variants by `_bg_NN` index — the game picks exactly one per index. */
  buckets: Record<string, RoomVariant[]>;
  /** Foreground variants — the game picks exactly one. */
  fg: RoomVariant[];
}

export interface RoomManifest {
  canvas: { w: number; h: number };
  scenes: Record<string, RoomScene>;
}

export const CANVAS_W = ROOM_MANIFEST.canvas.w;
export const CANVAS_H = ROOM_MANIFEST.canvas.h;

/** The act rooms, keyed by the act name the engine sends (loc titles match the
 *  PCK's scene directories). Anything else falls back to the act-1 jungle ruin. */
const ACT_SCENES: Record<string, string> = {
  overgrowth: "overgrowth",
  hive: "hive",
  glory: "glory",
  underdocks: "underdocks",
};
export const DEFAULT_SCENE = "overgrowth";

/** Encounters that bring their own room, keyed by the art slug of a monster
 *  that only appears in that encounter.
 *
 *  The game keys this on the encounter, which the combat wire does not carry —
 *  but every `HasCustomBackground` encounter has a signature monster, so the
 *  roster identifies it exactly. Slugs are the ART slugs (what `rig` / `image`
 *  resolve to), so the Kaiser Crab's two arms both arrive as `kaiser_crab` and
 *  the Merchant??? as `fake_merchant_monster` — see `_MONSTER_ALIASES` in
 *  `workbench/assets.py`, which this deliberately shares a vocabulary with.
 *
 *  `AeonglassBoss` is the one boss that declares `HasCustomBackground => false`
 *  and so is absent here on purpose: it fights in the plain act room. */
const BOSS_SCENES: Record<string, string> = {
  ceremonial_beast: "ceremonial_beast_boss",
  fake_merchant_monster: "fake_merchant_event_encounter",
  kaiser_crab: "kaiser_crab_boss",
  knowledge_demon: "knowledge_demon_boss",
  lagavulin_matriarch: "lagavulin_matriarch_boss",
  queen: "queen_boss",
  soul_fysh: "soul_fysh_boss",
  test_subject: "test_subject_boss",
  // NOT mapped: `architect` -> `the_architect_event_encounter`. The Architect
  // is a no-op dummy (a NOTHING self-loop behind a hidden intent, filed with
  // BigDummy and the Battle Friends in `monster_dispatch.rs`) reached only as
  // the canonical encounter of a scripted ancient event, and its background is
  // a victory cutscene plate the scene blows up to 2.2x for a push-in. Faithful
  // to render, wrong to frame as a combat room — and no run ever fights it.
  the_insatiable: "the_insatiable_boss",
  kin_priest: "the_kin_boss",
  kin_follower: "the_kin_boss",
  vantom: "vantom_boss",
  waterfall_giant: "waterfall_giant_boss",
};

/** Art slugs this enemy could be filed under, most authoritative first.
 *
 *  `rig` (`monster_rigs/<slug>/<slug>`) and `image` (`monsters/<slug>.png`)
 *  already carry the resolved art slug, alias table applied. Both are nullable
 *  — a monster with no staged rig AND no sprite falls back to its display name,
 *  which needs both slug forms because `slugify` drops a leading "the" while
 *  the art tree keeps it ("The Insatiable" -> `the_insatiable`). */
export function artSlugs(enemy: Enemy): string[] {
  const out: string[] = [];
  const rig = /monster_rigs\/([a-z0-9_]+)\//.exec(enemy.rig ?? "");
  if (rig) out.push(rig[1]);
  const image = /monsters\/([a-z0-9_]+)\.png$/.exec(enemy.image ?? "");
  if (image) out.push(image[1]);
  if (enemy.rig_group) out.push(enemy.rig_group);
  const raw = enemy.name.trim().toLowerCase().replace(/['’]/g, "").replace(/[^a-z0-9]+/g, "_");
  out.push(raw.replace(/^_|_$/g, ""));
  out.push(raw.replace(/^the_/, "").replace(/^_|_$/g, ""));
  return out;
}

/** Which map node a fight stands on, plus the run's master RNG seed.
 *
 *  `NCombatRoom.GenerateBackgroundRngForCurrentPoint` is
 *  `new Rng(runState.Rng.Seed + row + col * 747)`, so these three numbers fix a
 *  fight's room exactly — and both producers already have them: SpireMentor sends
 *  `rng.run_seed` and `map.current`, and the engine knows its own seed and node. */
export interface BgRoll {
  /** `RunState.Rng.Seed` — the run's master seed, NOT the string seed. A
   *  `uint` on v0.107.1, a `ulong` on v0.111.0; a decimal string past 2^53. */
  seed: number | string;
  row: number;
  col: number;
  /** The run's game build; absent = v0.107.1. Decides the seed's width. */
  gameVersion?: string | null;
}

/** The scene key for this fight: a boss's own room if one of them is on the
 *  board, else the act's room, else the act-1 default. */
export function pickScene(actName?: string | null, enemies?: Enemy[]): string {
  for (const enemy of enemies ?? []) {
    for (const slug of artSlugs(enemy)) {
      const scene = BOSS_SCENES[slug];
      if (scene && paintsAnything(scene)) return scene;
    }
  }
  return ACT_SCENES[(actName ?? "").trim().toLowerCase()] ?? DEFAULT_SCENE;
}

/** Does this scene put any servable art on the canvas?
 *
 *  Not every background scene does. `fake_merchant_event_encounter` is all
 *  particles and shader specks: the Merchant??? ambush composites over the shop
 *  room it springs in, so its own scene carries no room texture at all.
 *  Selecting it would render that fight — a real one — on the bare fallback
 *  gradient, so an empty scene falls through to the act room instead. */
function paintsAnything(sceneKey: string): boolean {
  const scene = ROOM_MANIFEST.scenes[sceneKey];
  if (!scene) return false;
  // Counting VARIANTS is not the same as counting art: a variant that paints
  // nothing is still in the roll (`underdocks_fg_b` anchors its plate to the
  // zero-sized `Foreground` slot, so the game shows no foreground when it comes
  // up), and the Merchant??? ambush's whole scene is like that.
  const painted = (v: RoomVariant) => v.layers.length > 0;
  return (
    scene.fg.some(painted) ||
    Object.values(scene.buckets).some((b) => b.some(painted)) ||
    scene.slots.some((s) => s.kind === "prop" && s.layers.length > 0)
  );
}

/** FNV-1a over a string — a stable 32-bit hash with no dependencies.
 *  Only ever used to pick a variant, so its quality bar is "spreads". */
function hash(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** The layer stack for one fight, split at the room's `Foreground` slot.
 *
 *  Mirrors `BackgroundAssets`: the layer scenes are bucketed by their `_bg_NN`
 *  index and exactly ONE variant of each is drawn, plus one `_fg_`. The game
 *  rolls those from its combat `Rng`; we roll them from the room and the floor,
 *  which the wire does carry — so a fight looks the same every time you replay
 *  it and different from the fight on the floor below, without needing the
 *  game's RNG stream.
 *
 *  Both halves come back back-to-front. `near` is still drawn BEHIND the
 *  combatants — see `.gs-arena-fg`; nothing in a room ever covers a fighter. */
export function resolveScene(
  sceneKey: string,
  act?: number | null,
  floor?: number | null,
  bg?: BgRoll | null,
): { far: RoomLayer[]; near: RoomLayer[]; picks: Record<string, string> } {
  const scene = ROOM_MANIFEST.scenes[sceneKey] ?? ROOM_MANIFEST.scenes[DEFAULT_SCENE];
  // THE GAME'S OWN ROLL when the fight tells us which map node it stands on.
  //
  // `BackgroundAssets` takes one `rng.NextItem` per `_bg_NN` bucket in ascending
  // key order and then one for `_fg_`, off
  // `NCombatRoom.GenerateBackgroundRngForCurrentPoint` — a pure function of the
  // run seed and the node. The manifest's slot list is already in that order
  // (props carry no draw), so walking it IS the C# loop. Reproducing this is
  // what puts the barrel, the netting and the lit window where the real fight
  // had them instead of on a different variant of the same room.
  //
  // Without a node (a workbench-native run, an old log) it falls back to the
  // hash below: stable per room+floor, so a fight still looks the same every
  // time you replay it and different from the one on the floor below.
  const rng = bg ? backgroundRng(bg.seed, bg.row, bg.col, bg.gameVersion) : null;
  const seed = `${sceneKey}|${act ?? 0}|${floor ?? 0}`;
  const roll = (variants: readonly RoomVariant[], slot: string): RoomVariant | undefined => {
    if (rng) return rng.nextItem(variants);
    return variants.length ? variants[hash(`${seed}|${slot}`) % variants.length] : undefined;
  };

  const far: RoomLayer[] = [];
  const near: RoomLayer[] = [];
  // Which variant each slot drew. Two variants of one slot often SHARE art
  // (Glory's three ruin variants all hang the same torch), so the picks are the
  // only reliable way to tell what was rolled — a src that came out of the
  // stack does not identify a variant.
  const picks: Record<string, string> = {};
  let bin = far;
  for (const slot of scene.slots) {
    if (slot.kind === "prop") {
      bin.push(...slot.layers);
      continue;
    }
    // Everything from the foreground slot onward is the room's near half.
    if (slot.kind === "fg") bin = near;
    const key = slot.kind === "fg" ? "fg" : slot.index;
    const pick = roll(slot.kind === "fg" ? scene.fg : scene.buckets[slot.index] ?? [], key);
    if (pick) {
      picks[key] = pick.id;
      bin.push(...pick.layers);
    }
  }
  return { far, near, picks };
}

/** The size this room draws its canvas at — `RoomScene.scale`, falling back to
 *  the Overgrowth family's 1.44 for a key the manifest does not carry. */
export function sceneScale(sceneKey: string): number {
  return ROOM_MANIFEST.scenes[sceneKey]?.scale ?? 1.44;
}

export function sceneAtmosphere(sceneKey: string): RoomAtmosphere | null {
  return ROOM_MANIFEST.scenes[sceneKey]?.atmosphere ?? null;
}

/** The CSS custom properties that give a room its weather.
 *
 *  `--gs-dust-rgb` / `--gs-dust-a` drive the dust wash — the room's OWN
 *  particles (the Insatiable's 16k sand motes, the Waterfall Giant's spray),
 *  which the game really does hang between the camera and the backdrop. A room
 *  whose artists hung nothing in the air sets neither and the layer is inert.
 *
 *  It used to damp a `--gs-haze-k` as well: a global green depth wash over
 *  every room. That wash is gone (GH #610) — the game paints its rooms with the
 *  layer stack and nothing else — so the dust now sits on the raw art its
 *  alphas were solved against instead of on a graded, hazed one.
 *
 *  The wash sits BEHIND the combatants, like the game's particles, so it thickens
 *  the room without fogging the fighters standing in it. */
export function atmosphereVars(sceneKey: string): CSSProperties {
  const air = sceneAtmosphere(sceneKey);
  if (!air) return {};
  return {
    "--gs-dust-rgb": air.rgb.slice(0, 3).join(" "),
    "--gs-dust-a": air.strength,
  } as CSSProperties;
}

/** Where one layer's element goes, in CANVAS px.
 *
 *  Each layer is its own element now, not a `background-image` in a shared
 *  multi-background stack. That stack could only ever say "this file, this
 *  size, this position" — one alpha and no orientation for the whole element —
 *  and the scenes need three things it cannot express: the half-turn these
 *  rooms mirror a prop with (`t`), the inherited `modulate` alpha (Glory's
 *  `03_c` plate at 0.83, the Insatiable's floor at 0.47) and its tint (the
 *  Queen's teal war-historian mural, the Insatiable's sand piles).
 *
 *  The element is placed in an inner box that IS the 1920x900 canvas at 1:1,
 *  which the caller scales as a whole — so these are the manifest's own
 *  numbers, unconverted, and `transform-origin: 0 0` makes `t` turn the art
 *  about exactly the corner its scene turns it about.
 *
 *  The tint is a `<feColorMatrix>` filter rather than a CSS one: no `filter`
 *  shorthand multiplies per channel, and multiplying is what `modulate` does.
 *  `tintFilterId` names the filter; `RoomTintFilters` renders the defs — except
 *  on a blended layer, where it moves to a child: see `roomLayerPaintStyle`. */
export function roomLayerStyle(layer: RoomLayer): CSSProperties {
  const t = layer.t;
  const style: CSSProperties = {
    position: "absolute",
    left: `${layer.left}px`,
    top: `${layer.top}px`,
    width: `${layer.w}px`,
    height: `${layer.h}px`,
  };
  if (t) {
    style.transformOrigin = "0 0";
    style.transform = `matrix(${t[0]}, ${t[1]}, ${t[2]}, ${t[3]}, 0, 0)`;
  }
  if (layer.blend === "add") {
    // `plus-lighter` IS Godot's `BLEND_MODE_ADD` (`screen` is a different curve
    // and would under-light the fog it stands in for). It blends against the
    // backdrop inside `.gs-room-canvas`, which is a stacking context because it
    // carries a transform — so the fog lifts the room's own layers and nothing
    // outside them.
    style.mixBlendMode = "plus-lighter";
  }
  return roomLayerPaintStyle(layer) ? style : { ...style, ...layerPaint(layer) };
}

/** The paint — background sizing, `modulate`, the sheet's frame — when it has
 *  to move off the layer's own element onto a child that fills it.
 *
 *  That is exactly the tinted, additively blended layer: a `filter` composites
 *  its element as a group BEFORE blending, and Chromium then ignores a
 *  `mix-blend-mode` on that same element. The Overgrowth's fog rendered as
 *  `normal` — and its sheet is black outside the puff, so it rendered as a
 *  black rectangle across the far wall. Splitting them composites in the order
 *  the game does. Null when nothing needs moving, which is every other layer.
 */
export function roomLayerPaintStyle(layer: RoomLayer): CSSProperties | null {
  if (layer.blend !== "add" || !layer.tint) return null;
  return { position: "absolute", inset: 0, ...layerPaint(layer) };
}

function layerPaint(layer: RoomLayer): CSSProperties {
  const paint: CSSProperties = {
    backgroundSize: "100% 100%",
    backgroundRepeat: "no-repeat",
  };
  if (layer.frame) {
    // An animated sheet is a vertical strip of `count` frames and the sprite
    // draws ONE. The element is already the frame's box, so blow the image up
    // by the frame count and slide the wanted one into it.
    const [index, count] = layer.frame;
    paint.backgroundSize = `100% ${count * 100}%`;
    paint.backgroundPosition = count > 1 ? `0 ${(index / (count - 1)) * 100}%` : "0 0";
  }
  if (layer.alpha != null) paint.opacity = layer.alpha;
  if (layer.tint) paint.filter = `url(#${tintFilterId(layer.tint)})`;
  return paint;
}

/** A stable id for one `modulate` tint, so identical tints share one filter. */
export function tintFilterId(tint: number[]): string {
  return `gs-tint-${tint.map((c) => Math.round(c * 1000)).join("-")}`;
}

/** Every distinct tint in a layer list, for `RoomTintFilters`. */
export function layerTints(...lists: RoomLayer[][]): number[][] {
  const seen = new Map<string, number[]>();
  for (const list of lists) {
    for (const layer of list) {
      if (layer.tint) seen.set(tintFilterId(layer.tint), layer.tint);
    }
  }
  return [...seen.values()];
}
