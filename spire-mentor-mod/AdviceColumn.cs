// AdviceColumn.cs -- where the advice panel and the coach chips sit once the
// player has dragged the toolbar off its dock: at the strip's left edge,
// right under it, so the advice travels with the strip. (They used to keep
// their own slot at the top-left, where the moved strip was no longer above
// them -- or where the player had just dropped it.)
//
// The panel and the chip stack go as ONE block, panel on top and chips under
// it, so they never overlap each other or the toolbar: below the strip when
// the block fits, else on whichever side has more room; never past the left
// or right screen edge.
//
// Docked, SimHud keeps its old placement (under the game's top bar).
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

using System;

namespace SpireMentor;

internal static class AdviceColumn
{
    /// <summary>Between the toolbar and the block, and between the panel and
    /// the chips -- the 8 px SimHud already used.</summary>
    public const float Gap = 8f;

    /// <summary>Height of the block: the panel when it is showing, with the
    /// chip stack under it.</summary>
    public static float BlockHeight(float panelHeight, bool panelShowing, float chipsHeight)
    {
        float height = panelShowing ? Math.Max(0f, panelHeight) : 0f;
        if (chipsHeight > 0f)
        {
            height += (height > 0f ? Gap : 0f) + chipsHeight;
        }
        return height;
    }

    /// <summary>
    /// Top-left of a <paramref name="blockWidth"/> x <paramref name="blockHeight"/>
    /// block following a toolbar whose top-left is (<paramref name="toolbarX"/>,
    /// <paramref name="toolbarY"/>), in a viewport of the given size (pixels).
    /// </summary>
    public static (float X, float Y) BlockTopLeft(
        float toolbarX, float toolbarY, float toolbarHeight,
        float blockWidth, float blockHeight, float viewWidth, float viewHeight)
    {
        float x = Math.Clamp(toolbarX, 0f, Math.Max(0f, viewWidth - blockWidth));
        float below = toolbarY + toolbarHeight + Gap;
        float roomBelow = viewHeight - below;
        float roomAbove = toolbarY - Gap;
        if (blockHeight <= roomBelow || roomBelow >= roomAbove)
        {
            return (x, below);
        }
        return (x, Math.Max(0f, roomAbove - blockHeight));
    }
}
