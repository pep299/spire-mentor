// LogPanelPlacement.cs -- where the action-log window (Ctrl+L, SimActionLog)
// goes.
//
// Until the player drags it, it sits under the advice column (the advice
// panel and the coach chips, SimHud.ChipSlot) and follows it as the column
// grows and shrinks. It used to open at a fixed 30% down the left edge,
// which is where that column ends up most of the time, so the chips -- and
// autobattle's "since your last stop" chip, which stays up for the whole
// decision -- landed on top of the log.
//
// Once dragged it stays where it was put, across launches too (persisted as
// "x,y" in the mod config); right-clicking the header returns it to the
// column.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

using System;
using System.Globalization;

namespace SpireMentor;

internal static class LogPanelPlacement
{
    /// <summary>Column bottom to log top.</summary>
    public const float Gap = 8f;

    /// <summary>The log is never squeezed below this (header plus a few
    /// lines); with less room than this under the column it overlaps the
    /// column rather than vanish.</summary>
    public const float MinHeight = 140f;

    /// <summary>Kept clear above the screen's bottom edge.</summary>
    public const float BottomMargin = 8f;

    /// <summary>
    /// Top and height of the log under a column ending at
    /// <paramref name="columnBottom"/>: its own height when it fits, else
    /// shortened to the room left, never below <see cref="MinHeight"/>.
    /// </summary>
    public static (float Top, float Height) BelowColumn(float columnBottom, float wantHeight, float viewHeight)
    {
        float top = columnBottom + Gap;
        float room = viewHeight - BottomMargin - top;
        float minHeight = Math.Min(MinHeight, wantHeight);
        if (room >= wantHeight)
        {
            return (top, wantHeight);
        }
        if (room >= minHeight)
        {
            return (top, room);
        }
        return (Math.Max(0f, viewHeight - BottomMargin - minHeight), minHeight);
    }

    /// <summary>A dragged position as the config stores it ("x,y").</summary>
    public static string Format(float x, float y) =>
        string.Create(CultureInfo.InvariantCulture, $"{x:0},{y:0}");

    /// <summary>The stored position, or false for "follow the column"
    /// (blank or unreadable).</summary>
    public static bool TryParse(string? text, out float x, out float y)
    {
        x = 0f;
        y = 0f;
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }
        string[] parts = text.Split(',');
        return parts.Length == 2 &&
               float.TryParse(parts[0].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out x) &&
               float.TryParse(parts[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out y) &&
               float.IsFinite(x) && float.IsFinite(y);
    }
}
