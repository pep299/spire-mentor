import { vi } from "vitest";
import type { Catalog, MctsRow, Results, SweepListEntry, SweepSpec } from "./api";
import { emptyPosition, type Position } from "./position";

/** The `/api/meta` payload the Studies form is driven by. */
export const META = {
  deck_presets: {
    ironclad_starter: [...Array<string>(5).fill("STRIKE_IRONCLAD"), ...Array<string>(4).fill("DEFEND_IRONCLAD"), "BASH"],
    aggro_rush: ["STRIKE_IRONCLAD", "BASH"],
  },
  strategies: ["random", "block_first", "min_overblock"],
  encounters: ["MAWLER_NORMAL", "VANTOM_BOSS"],
  cards: ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
  monsters: ["Nibbit", "Mawler"],
};

/** A small `/api/simlab/catalog`. */
export const CATALOG: Catalog = {
  characters: [
    { id: "Ironclad", name: "The Ironclad", hp: 80, relic: "BURNING_BLOOD", deck: ["STRIKE_IRONCLAD"], orbSlots: 0 },
    { id: "Defect", name: "The Defect", hp: 75, relic: "CRACKED_CORE", deck: ["STRIKE_DEFECT"], orbSlots: 3 },
  ],
  cards: [
    { id: "STRIKE_IRONCLAD", name: "Strike", type: "Attack", rarity: "Basic", cost: 1, character: "Ironclad" },
    { id: "DEFEND_IRONCLAD", name: "Defend", type: "Skill", rarity: "Basic", cost: 1, character: "Ironclad" },
    { id: "BASH", name: "Bash", type: "Attack", rarity: "Basic", cost: 2, character: "Ironclad" },
    { id: "DEMON_FORM_CARD", name: "Demon Form", type: "Power", rarity: "Rare", cost: 3, character: "Ironclad" },
  ],
  relics: [{ id: "BURNING_BLOOD", name: "Burning Blood" }],
  potions: [{ id: "FIRE_POTION", name: "Fire Potion" }],
  powers: [
    { id: "STRENGTH", name: "Strength" },
    { id: "RAVENOUS", name: "Ravenous" },
  ],
  orbs: [{ id: "Lightning", name: "Lightning", passive: 3, evoke: 8 }],
  monsters: [
    {
      id: "FuzzyWurmCrawler",
      name: "Fuzzy Wurm Crawler",
      minHp: 58,
      maxHp: 59,
      moves: [
        { id: "FIRST_ACID_GOOP", title: "First Acid Goop", initial: true, intent: { kind: "Attack", amount: 4 } },
        { id: "INHALE", title: "Inhale", initial: false, intent: { kind: "Buff", amount: 7 } },
      ],
    },
    { id: "Nibbit", name: "Nibbit", minHp: 20, maxHp: 24, moves: [{ id: "BUTT", title: "Butt", initial: true, intent: { kind: "Attack", amount: 12 } }] },
  ],
  encounters: [
    { id: "FUZZY_WURM_CRAWLER_WEAK", display: "Fuzzy Wurm Crawler", tier: "Weak", act: 1, actName: "Overgrowth", monsters: ["FuzzyWurmCrawler"] },
    { id: "VANTOM_BOSS", display: "Vantom", tier: "Boss", act: 1, actName: "Overgrowth", monsters: ["Vantom"] },
  ],
  strategies: ["block_first", "mcts", "min_overblock", "random"],
};

/** A built-looking position: the crawler fight on turn 1. */
export function samplePosition(): Position {
  const p = emptyPosition("Ironclad", 80);
  p.encounter = "FUZZY_WURM_CRAWLER_WEAK";
  p.player.hp = 64;
  p.player.relics = [{ id: "BURNING_BLOOD", counter: 0 }];
  p.piles.hand = [
    { card: "STRIKE_IRONCLAD", upgrade: 0, cost: 1 },
    { card: "STRIKE_IRONCLAD", upgrade: 0, cost: 1 },
    { card: "DEFEND_IRONCLAD", upgrade: 0, cost: 1 },
    { card: "BASH", upgrade: 0, cost: 2 },
  ];
  p.piles.draw = [
    { card: "DEFEND_IRONCLAD", upgrade: 0, cost: 1 },
    { card: "STRIKE_IRONCLAD", upgrade: 0, cost: 1 },
  ];
  p.monsters = [{ monster: "FuzzyWurmCrawler", hp: 58, maxHp: 58, block: 0, powers: [], intent: "FIRST_ACID_GOOP" }];
  return p;
}

/**
 * Every `api` function as a `vi.fn` with a sane default, so a test file can
 * `vi.mock("./api", () => ({ api: mocks.api }))` and override only what it
 * cares about. Built inside `vi.hoisted` by the caller so the mock factory
 * can reach it.
 */
export function makeApiMocks() {
  const sweeps: SweepListEntry[] = [];
  return {
    meta: vi.fn(async () => META),
    capabilities: vi.fn(async () => ({ game_versions: ["v0.107.1"], default_game_version: "v0.107.1" })),
    catalog: vi.fn<(gameVersion?: string | null) => Promise<Catalog>>(async () => CATALOG),
    submit: vi.fn<(spec: SweepSpec) => Promise<{ sweep_id: string }>>(async () => ({ sweep_id: "sw1" })),
    list: vi.fn(async () => ({ sweeps })),
    status: vi.fn<(id: string) => Promise<{ status: string; done: number; total: number }>>(async () => ({ status: "done", done: 1, total: 1 })),
    results: vi.fn<(id: string, opts?: unknown) => Promise<Results>>(async () => ({ buckets: {}, errors_total: 0 })),
    stop: vi.fn<(id: string) => Promise<{ status: string }>>(async () => ({ status: "stopping" })),
    buildPosition: vi.fn(async (position: Position) => ({ position, wire: { wire_version: 1 }, notes: [] })),
    newPosition: vi.fn<(body: unknown) => Promise<{ position: Position; wire: Record<string, unknown>; notes: string[] }>>(async () => ({ position: samplePosition(), wire: { wire_version: 1 }, notes: [] })),
    importPosition: vi.fn<(text: string) => Promise<{ position: Position; notes: string[]; source: string }>>(async () => ({ position: samplePosition(), notes: [], source: "advice_request" })),
    firstMoves: vi.fn(async (position: Position) => ({
      moves: [
        { label: "Play Strike → Fuzzy Wurm Crawler", action: {}, terminal: false, note: null, position, wire: { m: 1 } },
        { label: "Play Bash → Fuzzy Wurm Crawler", action: {}, terminal: false, note: null, position, wire: { m: 2 } },
        { label: "End turn", action: {}, terminal: false, note: null, position, wire: { m: 3 } },
      ],
      wire: { wire_version: 1 },
    })),
    mcts: vi.fn<(position: Position, opts: { sims: number }) => Promise<{ rows: MctsRow[]; sims: number; seed: number }>>(async () => ({
      rows: [
        { label: "Play Bash → Fuzzy Wurm Crawler", action: {}, win_prob: 0.91, q: 0.82, visits: 900, is_best: true, low_confidence: false },
        { label: "End turn", action: {}, win_prob: 0.4, q: 0.1, visits: 100, is_best: false, low_confidence: false },
      ],
      sims: 1000,
      seed: 0,
    })),
  };
}
