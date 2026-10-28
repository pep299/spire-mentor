// Archetype surfacing for the workbench (spec §10).
//
// Loads the per-character archetype file (`/archetypes_<char>.json`) and answers
// two questions: "which archetypes does my deck belong to, and where am I on
// each dose-response curve?" (DeckAnalysis) and "does this offered card advance
// an archetype I'm already building?" (CardReward chip).
//
// Only `confirmed`/`undiscovered` archetypes chip; `folklore` is
// catalog-browser-only — the same treatment `myth` pairs get.

import { bareToken } from "./synergy";

export type ArchetypeTag = "confirmed" | "folklore" | "undiscovered";

export interface ArchetypeCard {
  token: string;
  weight: number;
}
export interface ArchetypeCurvePoint {
  k: number;
  v: number;
  additive: number;
}
export interface ArchetypeCurve {
  targetK: number;
  points: ArchetypeCurvePoint[];
}
export interface Archetype {
  id: string;
  label: string;
  tag: ArchetypeTag;
  cards: ArchetypeCard[];
  curve: ArchetypeCurve;
}
export interface ArchetypeDoc {
  character: string;
  archetypes: Archetype[];
}

export interface ArchetypeMembership {
  archetype: Archetype;
  /** Bare tokens of held member cards (distinct). */
  held: string[];
  /** Number of held member pieces (copies counted). */
  k: number;
  /** Would the next piece put the deck into the super-additive part of the curve? */
  superAdditiveNext: boolean;
  /** Curve gain from k → k+1. */
  marginalNext: number;
}

export interface ArchetypeChip {
  id: string;
  label: string;
  tag: ArchetypeTag;
  /** Which piece the offered card would be (held + 1). */
  piece: number;
  /** Curve gain from held → held+1. */
  marginal: number;
  superAdditive: boolean;
}

// Per-character promise cache so concurrent mounts share one fetch (mirrors
// synergy.ts loadEdges). Like the synergy catalogs these are the v0.107
// corpus's tables for every game build (GH #877, see synergy.ts): a card new
// in v0.111.0 is simply unclassified.
const cache = new Map<string, Promise<ArchetypeDoc>>();

/** Load `/archetypes_<char>.json`; a missing/failed/off-schema fetch → an empty doc.
 *
 * The shape check is not paranoia: the served file is easy to mistake for the
 * raw `emit-archetypes` discovery output (`{character, candidates}`), which
 * would fetch 200 and then throw inside `deckMembership` — a swallowed error
 * that silently blanks the panel forever. An off-schema doc degrades to the
 * same empty-doc path as a 404 instead. */
export function loadArchetypes(character: string): Promise<ArchetypeDoc> {
  const hit = cache.get(character);
  if (hit) return hit;
  const p = fetch(`/archetypes_${character}.json`)
    .then((r) =>
      r.ok ? (r.json() as Promise<ArchetypeDoc>) : Promise.reject(new Error(`HTTP ${r.status}`)),
    )
    .then((doc) => {
      if (!Array.isArray(doc?.archetypes)) {
        throw new Error(`/archetypes_${character}.json: missing "archetypes" array`);
      }
      return doc;
    })
    .catch(() => ({ character, archetypes: [] }) as ArchetypeDoc);
  cache.set(character, p);
  return p;
}

/** Test-only: drop the per-character fetch cache (mirrors nothing in prod). */
export function __clearArchetypeCache(): void {
  cache.clear();
}

/** Curve value at dose k, clamped to the measured range. */
export function curveV(curve: ArchetypeCurve, k: number): number {
  const pts = curve.points;
  if (pts.length === 0) return 0;
  const clamped = Math.max(pts[0].k, Math.min(pts[pts.length - 1].k, k));
  let best = pts[0];
  for (const p of pts) if (p.k <= clamped) best = p;
  return best.v;
}

/** Curve gain from fromK → fromK+1 (0 past the measured range). */
export function curveMarginal(curve: ArchetypeCurve, fromK: number): number {
  return curveV(curve, fromK + 1) - curveV(curve, fromK);
}

/** Is the measured curve above its additive prediction at k (super-additive)? */
export function isSuperAdditive(curve: ArchetypeCurve, k: number): boolean {
  const p = curve.points.find((q) => q.k === k);
  return p ? p.v > p.additive : false;
}

function deckCounts(deckTokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const t of deckTokens) {
    const bt = bareToken(t);
    counts.set(bt, (counts.get(bt) ?? 0) + 1);
  }
  return counts;
}

const TAG_RANK: Record<ArchetypeTag, number> = { confirmed: 0, undiscovered: 1, folklore: 2 };

/** Archetypes the deck holds >= 1 piece of, with dose-curve position. */
export function deckMembership(doc: ArchetypeDoc, deckTokens: string[]): ArchetypeMembership[] {
  const counts = deckCounts(deckTokens);
  const out: ArchetypeMembership[] = [];
  for (const a of doc.archetypes) {
    const held: string[] = [];
    let k = 0;
    for (const c of a.cards) {
      const n = counts.get(c.token) ?? 0;
      if (n > 0) {
        held.push(c.token);
        k += n;
      }
    }
    if (k === 0) continue;
    out.push({
      archetype: a,
      held,
      k,
      superAdditiveNext: isSuperAdditive(a.curve, k + 1),
      marginalNext: curveMarginal(a.curve, k),
    });
  }
  out.sort((x, y) => TAG_RANK[x.archetype.tag] - TAG_RANK[y.archetype.tag] || y.k - x.k);
  return out;
}

/** Archetype chips for an offered card: the deck already holds >= 1 piece and the
 *  offered card is a weighty member of a confirmed/undiscovered archetype. */
export function archetypeChipsFor(
  doc: ArchetypeDoc,
  offeredId: string,
  deckTokens: string[],
  weightThreshold = 0.5,
): ArchetypeChip[] {
  const offered = bareToken(offeredId);
  const counts = deckCounts(deckTokens);
  const chips: ArchetypeChip[] = [];
  for (const a of doc.archetypes) {
    if (a.tag === "folklore") continue; // catalog-browser only (spec §10)
    const member = a.cards.find((c) => c.token === offered);
    if (!member || member.weight < weightThreshold) continue;
    let held = 0;
    for (const c of a.cards) if (c.token !== offered) held += counts.get(c.token) ?? 0;
    if (held < 1) continue; // only when the deck already commits to the archetype
    chips.push({
      id: a.id,
      label: a.label,
      tag: a.tag,
      piece: held + 1,
      marginal: curveMarginal(a.curve, held),
      superAdditive: isSuperAdditive(a.curve, held + 1),
    });
  }
  chips.sort((x, y) => y.marginal - x.marginal);
  return chips;
}
