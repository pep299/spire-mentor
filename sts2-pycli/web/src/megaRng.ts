/**
 * MegaCrit's `Rng` / `MegaRandom`, ported from the v0.107.1 decomp.
 *
 * `MegaRandom` is Xoshiro256** seeded through four Splitmix64 rounds, and `Rng`
 * wraps it: `NextInt(max)` is `(int)(NextDouble() * max)` and `NextItem(list)`
 * is `list[NextInt(list.length)]`. That chain is what picks a combat room's
 * layer variants (`BackgroundAssets`), so reproducing it exactly is the only way
 * a workbench fight can show the barrel, netting and window the real fight did.
 *
 * 64-bit through BigInt, which is exact where doubles are not — the state words
 * routinely exceed 2^53 and a Number port drifts within a handful of draws.
 * Cost is a few dozen BigInt ops per fight, once.
 *
 * Verified end to end, not just against itself: for run seed 44815703 at map
 * node (col 5, row 5) this picks `underdocks_01_c`, which is the variant the
 * real game drew in that fight — the one carrying the lit window.
 */

const MASK = (1n << 64n) - 1n;
const U32 = 0xffffffffn;
const TWO_POW_53 = 9007199254740992; // 2^53, the divisor behind `_incrDouble`

function rotl(x: bigint, k: bigint): bigint {
  return ((x << k) | (x >> (64n - k))) & MASK;
}

/** Splitmix64, the seeding PRNG (`MegaRandom.Splitmix64`). Returns the advanced
 *  state alongside its output, since the C# takes the state by ref. */
function splitmix64(state: bigint): [bigint, bigint] {
  const next = (state + 0x9e3779b97f4a7c15n) & MASK;
  let z = next;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK;
  return [next, z ^ (z >> 31n)];
}

/** `MegaCrit.Sts2.Core.Random.Rng`, with only the draws the room roll needs. */
export class MegaRng {
  private s0: bigint;
  private s1: bigint;
  private s2: bigint;
  private s3: bigint;

  /** `new Rng(seed)`. A `number` is a v0.107.1 `uint` seed, taken modulo
   *  2^32 as the C# type is; a `bigint` is a v0.111.0 `ulong`, taken modulo 2^64. */
  constructor(seed: number | bigint) {
    let state = typeof seed === "bigint" ? seed & MASK : BigInt(Math.floor(seed) >>> 0);
    const words: bigint[] = [];
    for (let i = 0; i < 4; i++) {
      const [next, out] = splitmix64(state);
      state = next;
      words.push(out);
    }
    [this.s0, this.s1, this.s2, this.s3] = words;
  }

  /** `MegaRandom.NextULongInner` — one Xoshiro256** step. */
  private nextULong(): bigint {
    const { s0, s1, s2, s3 } = this;
    const result = (rotl((s1 * 5n) & MASK, 7n) * 9n) & MASK;
    const t = (s1 << 17n) & MASK;
    let a = s0;
    let b = s1;
    let c = s2 ^ s0;
    let d = s3 ^ s1;
    b = b ^ c;
    a = a ^ d;
    c = c ^ t;
    d = rotl(d, 45n);
    this.s0 = a;
    this.s1 = b;
    this.s2 = c;
    this.s3 = d;
    return result;
  }

  /** `MegaRandom.NextDouble` — the top 53 bits, in [0, 1). */
  nextDouble(): number {
    return Number(this.nextULong() >> 11n) / TWO_POW_53;
  }

  /** `Rng.NextInt(maxExclusive)` — truncation, NOT modulo. */
  nextInt(maxExclusive: number): number {
    return Math.floor(this.nextDouble() * maxExclusive);
  }

  /** `Rng.NextItem` — `default(T)` (undefined here) for an empty list. */
  nextItem<T>(items: readonly T[]): T | undefined {
    if (items.length === 0) return undefined;
    return items[this.nextInt(items.length)];
  }
}

/**
 * `NCombatRoom.GenerateBackgroundRngForCurrentPoint`: the run's master seed
 * offset by the map node the fight stands on, so every node in a run draws its
 * own room and re-entering one draws the same room again.
 *
 * `new Rng(state.Rng.Seed + (uint)(row + col * 747))`. On v0.107.1 `Seed` is a
 * `uint`, so the sum wraps at 2^32; on v0.111.0 it is a `ulong` (XxHash64
 * seeding) and wraps at 2^64 (`v0.111.0/…/NCombatRoom.cs:385`). The width is the
 * build's, not the seed's: a v0.107.1 seed near 2^32 must still wrap.
 *
 * `runSeed` is a decimal string when the producer's seed is past 2^53, where a
 * JSON number would already have been rounded (GH #878).
 */
export function backgroundRng(
  runSeed: number | string,
  row: number,
  col: number,
  gameVersion?: string | null,
): MegaRng {
  const seed = BigInt(runSeed);
  const sum = seed + (BigInt(row + col * 747) & U32);
  // A seed past 2^32 can only be a `ulong` one, which covers a replayed log
  // that never named its build.
  const wide = (gameVersion != null && gameVersion !== "v0.107.1") || seed > U32;
  return wide ? new MegaRng(sum & MASK) : new MegaRng(Number(sum & U32));
}
