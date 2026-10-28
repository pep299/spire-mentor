// CoachChipText.cs -- the words on the coach chip, from one `coach` record
// (PROTOCOL.md section 5, v5 "coach"). Game-free so tests/unit can pin the
// copy; SimCoachChip only lays these strings out.
//
// The advisor decides WHAT is worth saying (thresholds, toss-ups, silence on
// uncertain turns); this file decides only HOW it reads, plus the one
// player-side filter: good turns and agreed choices stay silent unless the
// player asked for a check mark (settings, default off).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json.Nodes;

namespace SpireMentor;

internal enum ChipTone
{
    Good,
    Slip,
    Blunder,
    Info,
}

/// <summary>What a chip says: a title in the tone's colour, then lines.</summary>
internal sealed class ChipText
{
    public string Id = "";
    public string Kind = "";
    public string Title = "";
    public List<string> Lines = new();
    public ChipTone Tone;
    /// <summary>The record's run log and replay step (v5.1), for "open in
    /// workbench"; null when the advisor sent none.</summary>
    public string? LogName;
    public int? LogIdx;
}

internal static class CoachChipText
{
    /// <summary>
    /// The chip for <paramref name="rec"/>, or null to render nothing: a good
    /// turn / agreed choice without <paramref name="showGood"/>, a
    /// <c>withdraw</c> (not text -- see SimCoachChip.Withdraw), an unknown or
    /// malformed record.
    /// </summary>
    public static ChipText? Format(JsonObject rec, bool showGood)
    {
        try
        {
            ChipText? chip = Str(rec, "kind") switch
            {
                "turn" => Turn(rec, showGood),
                "choice" => Choice(rec, showGood),
                _ => null,
            };
            if (chip != null)
            {
                (chip.LogName, chip.LogIdx) = LogPosition(rec);
            }
            return chip;
        }
        catch (Exception)
        {
            return null; // a malformed record is silence, never a crash
        }
    }

    /// <summary>1st, 2nd, 3rd, 4th ... 11th-13th, 21st (plan 2's rule).</summary>
    public static string Ordinal(int n)
    {
        int mod100 = Math.Abs(n) % 100;
        string suffix = mod100 is >= 11 and <= 13
            ? "th"
            : (Math.Abs(n) % 10) switch
            {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            };
        return n.ToString(CultureInfo.InvariantCulture) + suffix;
    }

    /// <summary>A record's <c>log_name</c> / <c>log_idx</c> (v5.1); a null
    /// or negative step means "no position".</summary>
    internal static (string? Name, int? Idx) LogPosition(JsonObject rec)
    {
        string name = Str(rec, "log_name");
        int? idx = Num(rec, "log_idx") is double d && d >= 0 ? (int)d : null;
        return (name.Length > 0 ? name : null, idx);
    }

    // ------------------------------------------------------------------

    private static ChipText? Turn(JsonObject rec, bool showGood)
    {
        string verdict = Str(rec, "verdict");
        int turn = (int)(Num(rec, "turn") ?? 0);
        bool peeked = Bool(rec, "peeked");
        string peekedSuffix = peeked ? " (peeked)" : "";
        var chip = new ChipText { Id = Str(rec, "id"), Kind = "turn" };

        if (verdict is "best" or "tie")
        {
            if (!showGood)
            {
                return null;
            }
            chip.Tone = ChipTone.Good;
            chip.Title = verdict == "tie" ? $"✓ Turn {turn} · toss-up" : $"✓ Turn {turn}";
            return chip;
        }
        if (verdict is not ("slip" or "blunder"))
        {
            return null;
        }

        chip.Tone = verdict == "blunder" ? ChipTone.Blunder : ChipTone.Slip;
        chip.Title = verdict == "blunder" ? "⚠ blunder" : "▽ suboptimal";
        chip.Lines.Add($"Turn {turn} · you: {string.Join(" → ", StrList(rec, "you"))}");
        chip.Lines.Add($"AI:  {string.Join(" → ", StrList(rec, "ai"))}");

        string hp = "";
        if (Num(rec, "hp_drop") is double drop)
        {
            long rounded = (long)Math.Round(drop, MidpointRounding.AwayFromZero);
            if (rounded >= 1)
            {
                hp = $"≈ {rounded.ToString(CultureInfo.InvariantCulture)} HP more lost · ";
            }
        }
        chip.Lines.Add($"{hp}win {Pct(Num(rec, "win_before"))} → {Pct(Num(rec, "win_after"))}{peekedSuffix}");
        string why = Str(rec, "why");
        if (why.Length > 0)
        {
            chip.Lines.Add(why);
        }
        return chip;
    }

    internal static ChipText? Choice(JsonObject rec, bool showGood)
    {
        string verdict = Str(rec, "verdict");
        var chip = new ChipText { Id = Str(rec, "id"), Kind = "choice" };
        if (verdict is "agree" or "tossup")
        {
            if (!showGood)
            {
                return null;
            }
            chip.Tone = ChipTone.Good;
            chip.Title = verdict == "agree" ? "✓ AI agrees" : "✓ toss-up";
            return chip;
        }
        if (verdict != "flag")
        {
            return null;
        }

        JsonObject? played = rec["played"] as JsonObject;
        JsonObject? top = rec["top"] as JsonObject;
        if (played == null || top == null)
        {
            return null;
        }
        string playedLabel = Str(played, "label");
        int rank = (int)(Num(played, "rank") ?? 0);
        string peekedSuffix = Bool(rec, "peeked") ? " (peeked)" : "";

        chip.Tone = ChipTone.Info;
        chip.Title = $"{Str(rec, "label_kind")} · you: {playedLabel}";
        string rankText = rank > 0 ? $" · {playedLabel} was its {Ordinal(rank)} choice" : "";
        chip.Lines.Add($"AI would take: {Str(top, "label")} ({Str(rec, "strength")}){rankText}{peekedSuffix}");

        if (rec["path"] is JsonObject path &&
            RouteHp(path["you"]) is double youHp && RouteHp(path["ai"]) is double aiHp)
        {
            chip.Lines.Add($"route HP: you ≈{Whole(youHp)} · AI ≈{Whole(aiHp)}");
        }
        return chip;
    }

    /// <summary>The projected HP at the far end of a route: the last node
    /// that carries one.</summary>
    private static double? RouteHp(JsonNode? route)
    {
        if (route is not JsonArray steps)
        {
            return null;
        }
        double? hp = null;
        foreach (JsonNode? step in steps)
        {
            if (step is JsonObject o && Num(o, "hp") is double h)
            {
                hp = h;
            }
        }
        return hp;
    }

    internal static string Whole(double v) =>
        ((long)Math.Round(v, MidpointRounding.AwayFromZero)).ToString(CultureInfo.InvariantCulture);

    internal static string Pct(double? p) =>
        p is double v ? Whole(v * 100.0) + "%" : "?";

    internal static string Str(JsonObject o, string key) =>
        o[key] is JsonValue v && v.TryGetValue(out string? s) ? s ?? "" : "";

    internal static bool Bool(JsonObject o, string key) =>
        o[key] is JsonValue v && v.TryGetValue(out bool b) && b;

    internal static double? Num(JsonObject o, string key)
    {
        if (o[key] is not JsonValue v)
        {
            return null;
        }
        if (v.TryGetValue(out double d))
        {
            return d;
        }
        if (v.TryGetValue(out long l))
        {
            return l;
        }
        if (v.TryGetValue(out int i))
        {
            return i;
        }
        return null;
    }

    internal static List<string> StrList(JsonObject o, string key) =>
        o[key] is JsonArray a
            ? a.Select(n => n is JsonValue v && v.TryGetValue(out string? s) ? s ?? "" : "")
               .Where(s => s.Length > 0)
               .ToList()
            : new List<string>();
}
