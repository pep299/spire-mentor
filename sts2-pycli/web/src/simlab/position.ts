/**
 * The sandbox position (`simlab-position/1`) as the editor holds it, plus the
 * pure helpers around it: an empty one, immutable updates, the URL-hash codec
 * that carries a position between tabs, and card-row grouping for display.
 *
 * The server (`python/simlab/position.py`) owns the mapping to the engine
 * wire; this file never touches engine field names.
 */

import { encodePositionHash } from "./positionHash";

export const SCHEMA = "simlab-position/1";

export type PileName = "hand" | "draw" | "discard" | "exhaust";
export const PILES: PileName[] = ["hand", "draw", "discard", "exhaust"];

export interface PowerRow {
  id: string;
  amount: number;
  counter?: number | null;
}
export interface RelicRow {
  id: string;
  counter?: number;
}
export interface OrbRow {
  type: string;
  passive?: number;
  evoke?: number;
}
export interface CardRow {
  card: string;
  upgrade: number;
  cost: number | null;
}
export interface MonsterRow {
  monster: string;
  hp: number;
  maxHp: number;
  block: number;
  powers: PowerRow[];
  /** Engine move-state id; null = let the engine roll the opener. */
  intent: string | null;
}
export interface PlayerBlock {
  hp: number;
  maxHp: number;
  block: number;
  energy: number;
  maxEnergy: number;
  stars: number;
  gold: number;
  powers: PowerRow[];
  relics: RelicRow[];
  potions: string[];
  orbSlots: number;
  orbs: OrbRow[];
  osty: { alive: boolean; hp: number; maxHp: number; block: number } | null;
  cardsPlayedThisTurn: number[] | null;
}
export interface Position {
  schema: string;
  character: string;
  ascension: number;
  turn: number;
  isElite: boolean;
  encounter: string | null;
  seed: number;
  player: PlayerBlock;
  piles: Record<PileName, CardRow[]>;
  monsters: MonsterRow[];
  /** The game build this fight plays (GH #877); absent = the default build.
   *  The server writes it only for a non-default build. */
  game_version?: string;
}

export function emptyPosition(character = "Ironclad", hp = 80): Position {
  return {
    schema: SCHEMA,
    character,
    ascension: 0,
    turn: 1,
    isElite: false,
    encounter: null,
    seed: 1,
    player: {
      hp,
      maxHp: hp,
      block: 0,
      energy: 3,
      maxEnergy: 3,
      stars: 0,
      gold: 99,
      powers: [],
      relics: [],
      potions: [],
      orbSlots: 0,
      orbs: [],
      osty: null,
      cardsPlayedThisTurn: null,
    },
    piles: { hand: [], draw: [], discard: [], exhaust: [] },
    monsters: [],
  };
}

/** Fill in every field a pasted / older document may be missing. */
export function normalizePosition(raw: unknown): Position {
  const base = emptyPosition();
  const r = (raw ?? {}) as Partial<Position> & { player?: Partial<PlayerBlock>; piles?: Partial<Record<PileName, CardRow[]>> };
  const piles = { ...base.piles };
  for (const p of PILES) {
    const rows = r.piles?.[p];
    piles[p] = Array.isArray(rows)
      ? rows.map((c) => ({
          card: String((c as CardRow).card ?? ""),
          upgrade: Number((c as CardRow).upgrade ?? 0) || 0,
          cost: (c as CardRow).cost === undefined || (c as CardRow).cost === null ? null : Number((c as CardRow).cost),
        }))
      : [];
  }
  return {
    ...base,
    ...r,
    schema: SCHEMA,
    character: String(r.character ?? base.character),
    ascension: Number(r.ascension ?? 0) || 0,
    turn: Number(r.turn ?? 1) || 1,
    isElite: Boolean(r.isElite),
    encounter: r.encounter ?? null,
    seed: Number(r.seed ?? 1) || 1,
    player: {
      ...base.player,
      ...(r.player ?? {}),
      powers: Array.isArray(r.player?.powers) ? r.player.powers : [],
      relics: Array.isArray(r.player?.relics)
        ? r.player.relics.map((x) => (typeof x === "string" ? { id: x, counter: 0 } : x))
        : [],
      potions: Array.isArray(r.player?.potions) ? r.player.potions.filter((x): x is string => typeof x === "string" && !!x) : [],
      orbs: Array.isArray(r.player?.orbs) ? r.player.orbs : [],
    },
    piles,
    monsters: Array.isArray(r.monsters)
      ? r.monsters.map((m) => ({
          monster: String(m.monster ?? ""),
          hp: Number(m.hp ?? 1) || 1,
          maxHp: Number(m.maxHp ?? m.hp ?? 1) || 1,
          block: Number(m.block ?? 0) || 0,
          powers: Array.isArray(m.powers) ? m.powers : [],
          intent: m.intent ?? null,
        }))
      : [],
  };
}

// ---------------------------------------------------------------------------
// Immutable updates
// ---------------------------------------------------------------------------

export function setPlayer(pos: Position, patch: Partial<PlayerBlock>): Position {
  return { ...pos, player: { ...pos.player, ...patch } };
}

export function setPile(pos: Position, pile: PileName, rows: CardRow[]): Position {
  return { ...pos, piles: { ...pos.piles, [pile]: rows } };
}

export function addCard(pos: Position, pile: PileName, card: string): Position {
  return setPile(pos, pile, [...pos.piles[pile], { card, upgrade: 0, cost: null }]);
}

export function updateCard(pos: Position, pile: PileName, index: number, patch: Partial<CardRow>): Position {
  return setPile(
    pos,
    pile,
    pos.piles[pile].map((c, i) => (i === index ? { ...c, ...patch } : c)),
  );
}

export function removeCard(pos: Position, pile: PileName, index: number): Position {
  return setPile(pos, pile, pos.piles[pile].filter((_, i) => i !== index));
}

/** Move one card row to the end of another pile (draw order = list order). */
export function moveCard(pos: Position, from: PileName, index: number, to: PileName): Position {
  if (from === to) return pos;
  const row = pos.piles[from][index];
  if (!row) return pos;
  const without = removeCard(pos, from, index);
  return setPile(without, to, [...without.piles[to], row]);
}

export function setMonster(pos: Position, index: number, patch: Partial<MonsterRow>): Position {
  return { ...pos, monsters: pos.monsters.map((m, i) => (i === index ? { ...m, ...patch } : m)) };
}

export function addMonster(pos: Position, monster: string, hp = 1): Position {
  return { ...pos, monsters: [...pos.monsters, { monster, hp, maxHp: hp, block: 0, powers: [], intent: null }] };
}

export function removeMonster(pos: Position, index: number): Position {
  return { ...pos, monsters: pos.monsters.filter((_, i) => i !== index) };
}

export function totalCards(pos: Position): number {
  return PILES.reduce((n, p) => n + pos.piles[p].length, 0);
}

// ---------------------------------------------------------------------------
// Grouped card rows: identical consecutive cards show as one row with "× N".
// ---------------------------------------------------------------------------

export interface CardGroup {
  card: CardRow;
  /** Indices in the pile this group covers (consecutive). */
  indices: number[];
}

export function groupCards(rows: CardRow[]): CardGroup[] {
  const out: CardGroup[] = [];
  rows.forEach((row, i) => {
    const last = out[out.length - 1];
    if (last && sameCard(last.card, row)) {
      last.indices.push(i);
    } else {
      out.push({ card: row, indices: [i] });
    }
  });
  return out;
}

function sameCard(a: CardRow, b: CardRow): boolean {
  return a.card === b.card && a.upgrade === b.upgrade && (a.cost ?? null) === (b.cost ?? null);
}

/** Resize a group to `count` copies, keeping the group's position in the pile. */
export function setGroupCount(rows: CardRow[], group: CardGroup, count: number): CardRow[] {
  const n = Math.max(0, Math.floor(count));
  const first = group.indices[0];
  const last = group.indices[group.indices.length - 1];
  const before = rows.slice(0, first);
  const after = rows.slice(last + 1);
  return [...before, ...Array.from({ length: n }, () => ({ ...group.card })), ...after];
}

// ---------------------------------------------------------------------------
// URL hash codec — `#p=<base64url(JSON)>` carries a position between tabs.
// ---------------------------------------------------------------------------

function base64UrlToUtf8(b64: string): string {
  const padded = b64.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (b64.length % 4)) % 4);
  const bin = atob(padded);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function encodeHash(pos: Position): string {
  return "p=" + encodePositionHash(pos);
}

/** The position in a `#p=…` hash, or null when the hash carries none / is junk. */
export function decodeHash(hash: string): Position | null {
  const m = /(?:^|[#&])p=([A-Za-z0-9_-]+)/.exec(hash);
  if (!m) return null;
  try {
    return normalizePosition(JSON.parse(base64UrlToUtf8(m[1])));
  } catch {
    return null;
  }
}

/** Sweep id in a `?sweep=<id>` query string, if any. */
export function sweepFromSearch(search: string): string | null {
  return new URLSearchParams(search).get("sweep");
}

/** Pretty JSON for the Position JSON box. */
export function formatPosition(pos: Position): string {
  return JSON.stringify(pos, null, 2);
}
