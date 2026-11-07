// Autobattle.cs -- the in-game twin of the workbench's autobattle mode
// (sts2-pycli/python/workbench/autobattle.py): the AI plays the fights, and
// every RUN-level decision -- card reward, map, shop, rest site, event, card
// removal / upgrade / transform, boss relic -- stops for the player.
//
// Why: drilling run-level decisions as cold flashcards does not work, because
// reading a stranger's deck costs far more than deciding does. In real play
// context accrues one card at a time. An autobattle keeps that: the run is
// yours, only the combats are skipped, and at each stop a chip says what moved
// since the last one ("Added Inflame · HP 52/75 (−8) · 14 decisions
// auto-played"), so you read only the diff. The verdict comes AFTER you
// commit, from the coach's choice chip (agreement with the AI, never a win
// rate) -- in the Coach style nothing is shown before you act.
//
// This file is the pure core: which decisions stop, and the diff between two
// stops. Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link
// it; SimAdvisor reads the run and drives autoplay.

using System;
using System.Collections.Generic;
using System.Linq;

namespace SpireMentor;

/// <summary>The run-level state at one autobattle stop. Every field may be
/// missing (null / empty) rather than breaking the autobattle.</summary>
internal sealed record AutobattleSnapshot(
    int? Floor,
    int? Hp,
    int? MaxHp,
    int? Gold,
    IReadOnlyList<(string Name, bool Upgraded)> Deck,
    IReadOnlyList<string> Relics,
    IReadOnlyList<string> Potions);

internal static class Autobattle
{
    /// <summary><see cref="ChipText.Kind"/> of the "since your last stop"
    /// chip. SimCoachChip keeps it apart from the coach's own chips, so the
    /// verdict on the last pick and the diff for this one show together.</summary>
    public const string ChipKind = "autobattle";

    private const string Minus = "−"; // real minus glyph, as the workbench

    /// <summary>
    /// True when autobattle hands this decision to the player.
    /// <paramref name="adviceFor"/> is the decision's wire `advice_for`
    /// ("combat_action" for a combat move); <paramref name="local"/> is a
    /// decision the mod answers itself (SimScreenDecision.Local).
    ///
    /// Everything inside a fight is the AI's, card selections included (a
    /// Headbutt pick is part of the fight, not of the run). Outside a fight
    /// every real decision stops. The mod's one-button steps (claim the
    /// loot, open the chest, proceed, leave a shop with nothing affordable)
    /// are not decisions and play on -- except a card-select confirm, which
    /// only exists because the player has started picking cards on that
    /// screen: confirming it early would cut a multi-card pick short.
    /// </summary>
    public static bool IsStop(string? adviceFor, bool local, bool inCombat)
    {
        if (adviceFor is null or "game_over" or "combat_action" || inCombat)
        {
            return false;
        }
        return !local || adviceFor == "card_select";
    }

    /// <summary>The chip for this stop. <paramref name="prev"/> is the last
    /// stop's state, null at the first stop since autobattle was switched on
    /// (absolutes only: diffing against nothing would list the whole deck as
    /// "Added").</summary>
    public static ChipText Chip(string id, AutobattleSnapshot? prev, AutobattleSnapshot cur, int autoplayed)
    {
        string floor = cur.Floor is int f ? $" — floor {f}" : "";
        return new ChipText
        {
            Id = id,
            Kind = ChipKind,
            Title = prev == null ? $"Autobattle — your call{floor}" : $"Since your last stop{floor}",
            Lines = Lines(prev, cur, autoplayed),
            Tone = ChipTone.Info,
        };
    }

    /// <summary>
    /// "Since your last stop" lines, most context-carrying first: deck
    /// changes lead (they are what the next pick depends on), then relics and
    /// potions, then one resources line. Unchanged things are left out.
    /// </summary>
    public static List<string> Lines(AutobattleSnapshot? prev, AutobattleSnapshot cur, int autoplayed)
    {
        var lines = new List<string>();
        if (prev != null)
        {
            (List<string> added, List<string> removed, List<string> upgraded) = DeckDelta(prev.Deck, cur.Deck);
            (List<string> relicsGained, List<string> relicsLost) = GainedLost(prev.Relics, cur.Relics);
            (List<string> potionsGained, List<string> potionsLost) = GainedLost(prev.Potions, cur.Potions);
            Add(lines, "Upgraded ", upgraded);
            Add(lines, "Added ", added);
            Add(lines, "Removed ", removed);
            Add(lines, "Relic ", relicsGained);
            Add(lines, "Lost relic ", relicsLost);
            Add(lines, "Potion ", potionsGained);
            Add(lines, "Used potion ", potionsLost);
            if (!Moved(prev, cur))
            {
                lines.Add("Nothing changed.");
            }
        }

        var resources = new List<string>();
        if (cur.Hp is int hp)
        {
            string text = $"HP {hp}" + (cur.MaxHp is int max ? $"/{max}" : "");
            resources.Add(text + Signed(Delta(prev?.Hp, cur.Hp)));
        }
        if (cur.Gold is int gold)
        {
            resources.Add($"Gold {gold}" + Signed(Delta(prev?.Gold, cur.Gold)));
        }
        resources.Add($"Deck {cur.Deck.Count}");
        lines.Add(string.Join(" · ", resources));

        if (autoplayed > 0)
        {
            lines.Add($"{autoplayed} decision{(autoplayed == 1 ? "" : "s")} auto-played");
        }
        return lines;
    }

    /// <summary>Anything a stop's chip would report: the deck, relics,
    /// potions, HP or gold. (Max HP and the floor alone do not count.)</summary>
    public static bool Moved(AutobattleSnapshot prev, AutobattleSnapshot cur)
    {
        (List<string> added, List<string> removed, List<string> upgraded) = DeckDelta(prev.Deck, cur.Deck);
        return added.Count + removed.Count + upgraded.Count > 0 ||
               GainedLost(prev.Relics, cur.Relics) is not ({ Count: 0 }, { Count: 0 }) ||
               GainedLost(prev.Potions, cur.Potions) is not ({ Count: 0 }, { Count: 0 }) ||
               Delta(prev.Hp, cur.Hp) is not (null or 0) ||
               Delta(prev.Gold, cur.Gold) is not (null or 0);
    }

    /// <summary>
    /// Multiset diff of two master decks. A card whose unupgraded copy left
    /// while an upgraded copy of the same name arrived is an upgrade (a rest
    /// site, an event), not a removal plus an addition; only surplus upgraded
    /// copies count as added ("Strike+").
    /// </summary>
    public static (List<string> Added, List<string> Removed, List<string> Upgraded) DeckDelta(
        IReadOnlyList<(string Name, bool Upgraded)> prev, IReadOnlyList<(string Name, bool Upgraded)> cur)
    {
        Dictionary<(string, bool), int> gained = Subtract(Count(cur), Count(prev));
        Dictionary<(string, bool), int> lost = Subtract(Count(prev), Count(cur));

        var upgraded = new List<string>();
        foreach (((string name, bool isUpgraded), int count) in gained.ToList())
        {
            if (!isUpgraded)
            {
                continue;
            }
            int paired = Math.Min(count, lost.GetValueOrDefault((name, false)));
            if (paired > 0)
            {
                upgraded.AddRange(Enumerable.Repeat(name, paired));
                gained[(name, true)] -= paired;
                lost[(name, false)] -= paired;
            }
        }

        static List<string> Names(Dictionary<(string, bool), int> counts) =>
            counts.Where(kv => kv.Value > 0)
                .SelectMany(kv => Enumerable.Repeat(kv.Key.Item2 ? kv.Key.Item1 + "+" : kv.Key.Item1, kv.Value))
                .OrderBy(n => n, StringComparer.Ordinal)
                .ToList();

        return (Names(gained), Names(lost), upgraded.OrderBy(n => n, StringComparer.Ordinal).ToList());
    }

    private static (List<string> Gained, List<string> Lost) GainedLost(
        IReadOnlyList<string> prev, IReadOnlyList<string> cur)
    {
        Dictionary<string, int> before = Count(prev);
        Dictionary<string, int> after = Count(cur);
        static List<string> Names(Dictionary<string, int> counts) =>
            counts.Where(kv => kv.Value > 0)
                .SelectMany(kv => Enumerable.Repeat(kv.Key, kv.Value))
                .OrderBy(n => n, StringComparer.Ordinal)
                .ToList();
        return (Names(Subtract(after, before)), Names(Subtract(before, after)));
    }

    private static Dictionary<T, int> Count<T>(IEnumerable<T> items) where T : notnull
    {
        var counts = new Dictionary<T, int>();
        foreach (T item in items)
        {
            counts[item] = counts.GetValueOrDefault(item) + 1;
        }
        return counts;
    }

    /// <summary>Counter subtraction: what <paramref name="a"/> has beyond <paramref name="b"/>.</summary>
    private static Dictionary<T, int> Subtract<T>(Dictionary<T, int> a, Dictionary<T, int> b) where T : notnull =>
        a.Select(kv => (kv.Key, Count: kv.Value - b.GetValueOrDefault(kv.Key)))
            .Where(x => x.Count > 0)
            .ToDictionary(x => x.Key, x => x.Count);

    private static void Add(List<string> lines, string label, List<string> names)
    {
        if (names.Count > 0)
        {
            lines.Add(label + string.Join(", ", names));
        }
    }

    private static int? Delta(int? before, int? after) =>
        before is int b && after is int a ? a - b : null;

    private static string Signed(int? delta) =>
        delta is int d && d != 0 ? $" ({(d < 0 ? Minus : "+")}{Math.Abs(d)})" : "";
}
