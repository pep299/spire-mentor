// EnemyTagText.cs -- which of several same-named enemies is which.
//
// Three Two-Tailed Rats all read "Two-Tailed Rat", so the advisor's
// "Strike → Two-Tailed Rat" did not say which rat. The advisor now numbers a
// shared name ("Two-Tailed Rat #2", sts2-pycli/python/enemy_labels.py) and the
// HUD puts the same "#2" beside that rat's health bar. Both count over the
// combat's enemy list in order -- the list the wire serializes -- so the two
// always agree. A dead enemy takes no number (the game keeps a dying monster
// in that list until its move resolves; the advisor's labels skip
// `is_alive: false` the same way), so the numbers close up when one dies;
// they are right at any moment, not fixed per fight.
//
// Game-free so tests/unit can compile it.

using System.Collections.Generic;

namespace SpireMentor;

internal static class EnemyTagText
{
    /// <summary>
    /// One tag per name, in order: "#k" (1-based rank among the enemies
    /// sharing that name) for a shared name, null for a unique one. A null
    /// name is an enemy off the board (dead): it is not counted and gets no tag.
    /// </summary>
    public static string?[] Tags(IReadOnlyList<string?> names)
    {
        var total = new Dictionary<string, int>();
        foreach (string? name in names)
        {
            if (name != null)
            {
                total[name] = total.GetValueOrDefault(name) + 1;
            }
        }
        var seen = new Dictionary<string, int>();
        var tags = new string?[names.Count];
        for (int i = 0; i < names.Count; i++)
        {
            string? name = names[i];
            if (name == null)
            {
                continue;
            }
            int nth = seen.GetValueOrDefault(name) + 1;
            seen[name] = nth;
            tags[i] = total[name] > 1 ? $"#{nth}" : null;
        }
        return tags;
    }
}
