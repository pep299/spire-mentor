import type { Enemy } from "./api";

/**
 * Map an enemy slot index to something a human recognises.
 *
 * The reasoning panel and the search-tree view both used to render a target as
 * its raw index ("Strike → 1"), which is only meaningful if you already know
 * the encounter's slot order. Names are NOT unique (two Cultists, three
 * Louses), so a duplicated name is disambiguated by its 1-based position among
 * the enemies sharing it — "Louse #2" — rather than by the slot index, which
 * is the thing that was unreadable in the first place.
 *
 * The grouping name is the enemy's `name`, else its `id`, else "enemy <i>"
 * (its list position) — the same rule as `python/enemy_labels.py` and the
 * Rust `sts2-workbench` `enemy_labels.rs`. The view lists live enemies only,
 * so every entry is counted.
 *
 * Falls back to "enemy N" when the caller has no enemy list, e.g. a search
 * tree opened from a step recorded without one.
 */
export function enemyLabeller(
  enemies: (Pick<Enemy, "index" | "name"> & { id?: string | null })[] | undefined,
): (index: number) => string {
  const byIndex = new Map<number, string>();
  if (enemies && enemies.length > 0) {
    const names = enemies.map((e, i) => e.name || e.id || `enemy ${i}`);
    const total = new Map<string, number>();
    for (const n of names) total.set(n, (total.get(n) ?? 0) + 1);
    const seen = new Map<string, number>();
    enemies.forEach((e, i) => {
      const name = names[i];
      const nth = (seen.get(name) ?? 0) + 1;
      seen.set(name, nth);
      byIndex.set(e.index, (total.get(name) ?? 0) > 1 ? `${name} #${nth}` : name);
    });
  }
  return (index: number) => byIndex.get(index) ?? `enemy ${index}`;
}
