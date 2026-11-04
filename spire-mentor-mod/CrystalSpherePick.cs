// CrystalSpherePick.cs -- which Crystal Sphere cell to divine (GH #872).
//
// The minigame (CrystalSphereMinigame.cs) hides fifteen items on an 11x11 grid
// and reveals an item only once EVERY cell it covers is clear. The mod selects
// the Big tool (the game's default) before every reveal, so each divination
// clears the clicked cell and its eight neighbours. The proceed button only enables once the
// divinations are spent (NCrystalSphereScreen: `if (_entity.IsFinished)`), so
// "when to stop" is never a choice -- only which cells to clear.
//
// The board's contents are not on the wire, only which cells are still hidden,
// so the rule is geometric. The relic is 4x4: only adjacent, non-overlapping
// 3x3 blocks can uncover it. The fixed order below clears a 3-wide column
// first (the 3-divination event option) and then the column beside it (the
// 6-divination option, a 6x9 rectangle). Monte Carlo over the decomp's
// placement rules (n=4000, item values relic 150 ... curse -60) put this
// order at EV ~46 / ~167 for 3 / 6 divinations, against ~34 / ~69 for a
// uniformly random hidden cell and ~21 / ~50 for the first hidden cell (what
// F7 clicked before).
//
// A centre already cleared (the player divined by hand before handing over)
// is skipped; past the list, the cell clearing the most still-hidden cells
// wins, nearest the grid centre on a tie.
//
// Twin of sts2-pycli/python/crystal_sphere_policy.py (the driver lane). Keep
// the two orders identical. Game-free so tests/unit can link it.

using System.Collections.Generic;

namespace SpireMentor;

public static class CrystalSpherePick
{
    /// <summary>`_defaultWidth` / `_defaultHeight` (CrystalSphereMinigame.cs).</summary>
    public const int GridSize = 11;

    /// <summary>Big-tool centres, in order: column x 2..4, then x 5..7.</summary>
    public static readonly (int X, int Y)[] Centers =
    {
        (3, 2), (3, 5), (3, 8), (6, 2), (6, 5), (6, 8),
    };

    /// <summary>
    /// The index into <paramref name="hidden"/> of the cell to divine, or -1
    /// when nothing is hidden.
    /// </summary>
    public static int Pick(IReadOnlyList<(int X, int Y)> hidden)
    {
        if (hidden.Count == 0)
        {
            return -1;
        }
        foreach ((int X, int Y) center in Centers)
        {
            for (int i = 0; i < hidden.Count; i++)
            {
                if (hidden[i] == center)
                {
                    return i;
                }
            }
        }

        var set = new HashSet<(int, int)>(hidden);
        int best = 0;
        int bestCover = -1;
        int bestDist = int.MaxValue;
        for (int i = 0; i < hidden.Count; i++)
        {
            (int x, int y) = hidden[i];
            int cover = 0;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    if (set.Contains((x + dx, y + dy)))
                    {
                        cover++;
                    }
                }
            }
            int c = GridSize / 2;
            int dist = (x - c) * (x - c) + (y - c) * (y - c);
            if (cover > bestCover || (cover == bestCover && dist < bestDist))
            {
                best = i;
                bestCover = cover;
                bestDist = dist;
            }
        }
        return best;
    }
}
