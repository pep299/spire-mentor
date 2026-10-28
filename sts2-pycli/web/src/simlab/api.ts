import { apiFetch } from "../transport";

export interface SweepSpec {
  name: string;
  encounters: string[];
  ascensions: number[];
  strategies: string[];
  n_seeds: number;
  /** Deck variants as [name, [card ids...]] pairs (replaces `deck_preset`). */
  decks: [string, string[]][];
  /** Legacy field on historical specs only; new specs use `decks`. */
  deck_preset?: string;
  mcts_sims?: number;
  mcts_depth?: number;
  /** Sweep FROM these boards (label, engine wire JSON) instead of building fights. */
  start_states?: [string, string][];
  /** The sandbox positions behind `start_states` (label, position JSON) — display only. */
  positions?: [string, string][];
}

export interface Meta {
  deck_presets: Record<string, string[]>;
  strategies: string[];
  encounters: string[];
  /** All known card ids (base, no upgrade suffix). */
  cards: string[];
  /** All known monster names (for custom enemy lineups). */
  monsters: string[];
}

export interface SweepStatus {
  status: "running" | "done" | "error" | "unknown" | "cancelled";
  done?: number;
  total?: number;
  error?: string;
  /** Worker count for this sweep (present on running/finished web sweeps). */
  jobs?: number;
  /** Seconds elapsed since the run started. */
  elapsed_s?: number;
  /** Live wall-clock combats/second (done / elapsed). */
  combats_per_s?: number;
}

/** Per-core throughput aggregated over a sweep's rows. */
export interface Throughput {
  combats: number;
  actions: number;
  turns: number;
  core_seconds: number;
  combats_per_core_s: number;
  actions_per_core_s: number;
  turns_per_core_s: number;
  /** Sweep-wide wall-clock view, injected by run_sweep (summary.json only). */
  wall?: {
    elapsed_s: number;
    jobs: number;
    combats_per_s: number;
    actions_per_s: number;
    turns_per_s: number;
    parallel_efficiency: number;
  };
}

export interface SweepListEntry extends SweepStatus {
  sweep_id: string;
  spec: SweepSpec;
  /** Epoch seconds of the sweep's spec.json; list is server-sorted by this desc. */
  mtime: number;
}

export interface Bucket {
  n: number;
  mean_dmg: number;
  median_dmg: number;
  p95_dmg: number;
  max_dmg: number;
  win_rate: number;
  mean_turns: number;
  errors: number;
  dmg_values: number[];
  kde: { x: number[]; y: number[] };
  /** Most-frequent damage value (histogram mode). */
  mode_dmg: number;
  /** Worst (highest) observed damage. */
  worst: number;
  /** Probability of the worst-damage tail (e.g. 0.00025 = "1 in 4,000"). */
  worst_p: number;
  /** Precomputed box-plot stats: q1/median/q3 box, whiskers, outliers. */
  box: { q1: number; median: number; q3: number; lo: number; hi: number; outliers: number[] };
  delta_vs_baseline?: { mean: number; values: number[] };
  /** Per-bucket per-core throughput (mcts-vs-heuristic speed per cell). */
  throughput?: Throughput;
}

export interface Results {
  buckets: Record<string, Bucket>;
  errors_total: number;
}

const j = async (r: Response) => {
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}${body ? `: ${body}` : ""}`);
  }
  return r.json();
};

/** The `/api/simlab/catalog` payload — ids + display names the editor offers. */
export interface CatalogCard {
  id: string;
  name: string;
  type: string;
  rarity: string;
  cost: number | null;
  character: string;
}
export interface CatalogNamed {
  id: string;
  name: string;
  rarity?: string;
  description?: string;
}
export interface CatalogMove {
  id: string;
  title: string;
  initial: boolean;
  intent: { kind: string; amount?: number; times?: number } | null;
}
export interface CatalogMonster {
  id: string;
  name: string;
  moves: CatalogMove[];
  minHp?: number;
  maxHp?: number;
}
export interface CatalogEncounter {
  id: string;
  display: string;
  tier: string;
  act: number;
  actName: string;
  monsters: string[];
}
export interface CatalogCharacter {
  id: string;
  name: string;
  hp: number;
  relic: string;
  deck: string[];
  orbSlots: number;
}
export interface Catalog {
  characters: CatalogCharacter[];
  cards: CatalogCard[];
  relics: CatalogNamed[];
  potions: CatalogNamed[];
  powers: CatalogNamed[];
  orbs: { id: string; name: string; passive: number; evoke: number }[];
  monsters: CatalogMonster[];
  encounters: CatalogEncounter[];
  strategies: string[];
}

/** `POST /api/simlab/position/build` and friends. */
export interface BuiltPosition {
  position: Position;
  wire: Record<string, unknown>;
  notes: string[];
}
export interface FirstMove {
  label: string;
  action: Record<string, unknown>;
  terminal: boolean;
  note: string | null;
  position?: Position | null;
  wire?: Record<string, unknown>;
}
export interface MctsRow {
  label: string;
  action: Record<string, unknown>;
  win_prob: number;
  q: number;
  visits: number;
  is_best: boolean;
  low_confidence: boolean;
}

import type { Position } from "./position";
import { gameVersionBody } from "../gameVersion";

export const api = {
  /** The workbench's `/api/capabilities` — which game builds this server
   *  runs (GH #801), so the sandbox offers the same builds as the launcher. */
  capabilities: (): Promise<{ game_versions?: string[]; default_game_version?: string }> =>
    apiFetch("/api/capabilities").then(j),
  /** The catalog named in game build `gameVersion`'s text (GH #877). */
  catalog: (gameVersion?: string | null): Promise<Catalog> => {
    const v = gameVersionBody(gameVersion).game_version;
    return apiFetch(`/api/simlab/catalog${v ? `?game_version=${encodeURIComponent(v)}` : ""}`).then(j);
  },
  buildPosition: (position: Position): Promise<BuiltPosition> =>
    apiFetch("/api/simlab/position/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ position }),
    }).then(j),
  newPosition: (body: {
    character: string;
    encounter?: string | null;
    monsters?: string[];
    ascension: number;
    seed: number;
    hp?: number | null;
    maxHp?: number | null;
    relics?: string[];
    potions?: string[];
    deck?: string[];
    /** The build to deal it in (GH #877); omitted for the default build. */
    game_version?: string;
  }): Promise<BuiltPosition> =>
    apiFetch("/api/simlab/position/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j),
  importPosition: (text: string): Promise<{ position: Position; notes: string[]; source: string }> =>
    apiFetch("/api/simlab/position/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then(j),
  firstMoves: (position: Position): Promise<{ moves: FirstMove[]; wire: Record<string, unknown> }> =>
    apiFetch("/api/simlab/position/moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ position }),
    }).then(j),
  mcts: (position: Position, opts: { sims: number; seed?: number; threads?: number }): Promise<{ rows: MctsRow[]; sims: number; seed: number }> =>
    apiFetch("/api/simlab/position/mcts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ position, ...opts }),
    }).then(j),
  meta: (): Promise<Meta> => apiFetch("/api/meta").then(j),
  submit: (spec: SweepSpec): Promise<{ sweep_id: string }> =>
    apiFetch("/api/sweeps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then(j),
  list: (): Promise<{ sweeps: SweepListEntry[] }> => apiFetch("/api/sweeps").then(j),
  status: (id: string): Promise<SweepStatus> => apiFetch(`/api/sweeps/${id}`).then(j),
  /** Request a hard-stop of a running sweep (404 if it isn't running). */
  stop: (id: string): Promise<{ status: string }> =>
    apiFetch(`/api/sweeps/${id}/stop`, { method: "POST" }).then(j),
  results: (
    id: string,
    opts?: { baseline?: string; baselineDeck?: string },
  ): Promise<Results> => {
    // `baseline` and `baseline_deck` are mutually exclusive server-side (422 if
    // both); the caller is responsible for never setting both at once.
    const params = new URLSearchParams();
    if (opts?.baseline) params.set("baseline", opts.baseline);
    if (opts?.baselineDeck) params.set("baseline_deck", opts.baselineDeck);
    const qs = params.toString();
    return apiFetch(`/api/sweeps/${id}/results${qs ? `?${qs}` : ""}`).then(j);
  },
};
