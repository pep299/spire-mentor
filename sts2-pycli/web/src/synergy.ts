// Synergy lookup for the card-choice helper.
//
// Loads the per-character synergy catalog (`/catalog_<char>.json`, the same data
// the standalone catalog page uses) and answers: "for this offered card, which
// cards already in my deck does it synergize with?"  Used by CardReward to put a
// synergy chip under each offered card so the player can compare picks.
//
// A pair counts as a real synergy when the catalog tagged it `confirmed`
// (co-picked AND causal) or `undiscovered` (causal but under-picked). `myth`
// (co-picked but causal≤0) and `noise` are not shown — the whole point of the
// catalog is to separate those out.

export type Quadrant = "confirmed" | "myth" | "undiscovered" | "noise";

interface CatalogEdge {
  pair: [string, string];
  pmi: number;
  econ_S: number | null; // now the PPO value-head probe (spec §6)
  combat_S: number | null;
  /** Run-level injection A/B effect size (spec §8); absent on pre-Slice-8 catalogs. */
  runlevel_S?: number | null;
  quadrant: Quadrant;
}
interface Catalog {
  character: string;
  edges: CatalogEdge[];
}

/** A deck card that the offered card synergizes with. */
export interface PartnerHint {
  /** Bare engine token of the deck partner, e.g. `WHIRLWIND`. */
  partner: string;
  /** Display label, e.g. `Whirlwind`. */
  label: string;
  /** How many copies of the partner are in the deck. */
  count: number;
  quadrant: Quadrant;
  /** Effect size used for ranking; source given by `metric` (see `edgeScore`). */
  score: number;
  /** Which metric `score` came from, most-trusted first: runlevel > combat > ppo > pmi. */
  metric: "runlevel" | "combat" | "ppo" | "pmi";
}

const CHAR_BY_BASIC: Record<string, string> = {
  IRONCLAD: "ironclad",
  SILENT: "silent",
  DEFECT: "defect",
  REGENT: "regent",
  NECROBINDER: "necrobinder",
};

/** Strip a `CARD.`/`RELIC.` namespace prefix and any `+N` upgrade marker. */
export function bareToken(id: string): string {
  const noNs = id.includes(".") ? id.slice(id.indexOf(".") + 1) : id;
  const plus = noNs.indexOf("+");
  return plus >= 0 ? noNs.slice(0, plus) : noNs;
}

const CHARACTER_SUFFIX = /_(IRONCLAD|SILENT|DEFECT|REGENT|NECROBINDER)$/;

/** Human label from an engine token: drop the character suffix, title-case. */
export function labelFor(token: string): string {
  const base = token.replace(CHARACTER_SUFFIX, "");
  return base
    .split("_")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/** Detect the run's character from its deck tokens via the `STRIKE_<X>` /
 *  `DEFEND_<X>` basic that every character carries. Returns null if none seen. */
export function detectCharacter(deckTokens: string[]): string | null {
  for (const t of deckTokens) {
    const m = /^(?:STRIKE|DEFEND)_([A-Z]+)$/.exec(t);
    if (m && CHAR_BY_BASIC[m[1]]) return CHAR_BY_BASIC[m[1]];
  }
  return null;
}

function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

// Per-character edge map cache: character → ("A|B" → edge). Promise-cached so
// concurrent CardReward mounts share one fetch.
//
// One catalog per character, NOT per game build (GH #877): the catalogs are
// built from a scraped v0.107 run corpus and cannot be regenerated for
// v0.111.0 until a corpus of that build exists. A v0.111.0 run therefore
// reads the v0.107 tables -- a card new in v0.111.0 has no pairs, so it simply
// gets no synergy chip, and a changed card is judged by its v0.107 pairs.
const cache = new Map<string, Promise<Map<string, CatalogEdge>>>();

function loadEdges(character: string): Promise<Map<string, CatalogEdge>> {
  const hit = cache.get(character);
  if (hit) return hit;
  const p = fetch(`/catalog_${character}.json`)
    .then((r) => (r.ok ? (r.json() as Promise<Catalog>) : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((cat) => {
      const m = new Map<string, CatalogEdge>();
      for (const e of cat.edges) m.set(pairKey(e.pair[0], e.pair[1]), e);
      return m;
    })
    .catch(() => new Map<string, CatalogEdge>()); // missing catalog → no hints
  cache.set(character, p);
  return p;
}

const QUADRANT_RANK: Record<Quadrant, number> = {
  confirmed: 0,
  undiscovered: 1,
  myth: 2,
  noise: 3,
};

// Ranking metric order (spec §10): runlevel > combat > ppo > pmi — most-trusted
// causal signal available for the pair wins.
function edgeScore(e: CatalogEdge): { score: number; metric: PartnerHint["metric"] } {
  if (e.runlevel_S != null) return { score: e.runlevel_S, metric: "runlevel" };
  if (e.combat_S != null) return { score: e.combat_S, metric: "combat" };
  if (e.econ_S != null) return { score: e.econ_S, metric: "ppo" };
  return { score: e.pmi, metric: "pmi" };
}

/** Synergy partners for `offeredId` among the cards in `deckTokens`.
 *
 * Only `confirmed`/`undiscovered` pairs are returned (real causal synergy),
 * ranked confirmed-first then by effect size, deduped by partner with copy
 * counts.  Returns `[]` when the catalog is missing or nothing synergizes. */
export async function synergyFor(
  character: string,
  offeredId: string,
  deckTokens: string[],
  limit = 5,
): Promise<PartnerHint[]> {
  const edges = await loadEdges(character);
  if (edges.size === 0) return [];

  const offered = bareToken(offeredId);
  // Count deck copies by bare token.
  const counts = new Map<string, number>();
  for (const t of deckTokens) {
    const bt = bareToken(t);
    counts.set(bt, (counts.get(bt) ?? 0) + 1);
  }

  const hints: PartnerHint[] = [];
  for (const [partner, count] of counts) {
    if (partner === offered) continue;
    const e = edges.get(pairKey(offered, partner));
    if (!e) continue;
    if (e.quadrant !== "confirmed" && e.quadrant !== "undiscovered") continue;
    const { score, metric } = edgeScore(e);
    hints.push({ partner, label: labelFor(partner), count, quadrant: e.quadrant, score, metric });
  }

  hints.sort((a, b) => {
    const r = QUADRANT_RANK[a.quadrant] - QUADRANT_RANK[b.quadrant];
    return r !== 0 ? r : b.score - a.score;
  });
  return hints.slice(0, limit);
}
