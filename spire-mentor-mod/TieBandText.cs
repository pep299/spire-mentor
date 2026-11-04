// TieBandText.cs -- the HUD's "about equal" lines, from the engine's own tie
// band (`advice.pick` + `options[].in_band`, PROTOCOL.md section 5; GH #1022).
// Game-free so tests/unit can pin the copy; SimHud only lays the lines out.
//
// The engine decides which moves it cannot tell apart and why it picked the
// one it did; this file only says so. It holds no threshold of its own: the
// HUD used to call any top moves within 0.01 Q "equivalent", a third near-tie
// rule beside the engine's band (0.02) and the workbench's (0.004).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace SpireMentor;

internal static class TieBandText
{
    /// <summary>Most in-band moves named before the rest are counted.</summary>
    private const int MaxNamed = 3;

    /// <summary>
    /// The lines to show under the ranked list, or none when the band holds
    /// only the pick (a decided move needs no comment).
    /// </summary>
    /// <param name="inBand">Labels of the in-band options, the pick first,
    /// then in the engine's preference order.</param>
    /// <param name="reason">`advice.pick.reason`: why the engine took the
    /// pick over the rest of the band.</param>
    /// <param name="gapHp">`advice.pick.gap_hp`: mean final HP of the pick
    /// minus the runner-up's, null when either has no resolved playout.</param>
    public static List<string> Lines(IReadOnlyList<string> inBand, string? reason, double? gapHp)
    {
        var lines = new List<string>();
        if (inBand.Count < 2)
        {
            return lines;
        }
        string moves = string.Join(" ≈ ", inBand.Take(MaxNamed));
        if (inBand.Count > MaxNamed)
        {
            moves += $" (+{inBand.Count - MaxNamed})";
        }
        string line = "about equal: " + moves;
        if (gapHp is double hp && Math.Abs(hp) >= 0.5)
        {
            line += $" (~{Math.Abs(hp).ToString("0", CultureInfo.InvariantCulture)} HP)";
        }
        lines.Add(line);
        if (!string.IsNullOrEmpty(reason))
        {
            lines.Add($"→ {inBand[0]}: {reason}");
        }
        return lines;
    }
}
