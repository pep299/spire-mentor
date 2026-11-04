// BugReportDump.cs -- the bug report the mod sends, assembled from parts.
//
// Game-free so tests/unit can pin it; SimBugReport gathers the parts from the
// running game (on the main thread) and uploads the result.
//
// The shape is the workbench's schema-1 dump (sts2-pycli/web/src/bugreport.ts,
// workbench/bugreport.py): {schema_version, meta, repro, state, ui, note}, so
// the same store takes it (the hosted workbench's Pages Function,
// web/functions/api/_bugreport.ts), the same `?report=<id>` view opens it and
// `just bug-report <id>` pulls it down. What only a mod report carries:
//
//   meta.source = "mod"   tells every reader which kind this is
//   state.game_state      the Copy-state payload (SimStateCopy): the
//                         workbench's Import state rebuilds the position
//   repro.run_save        the game's own save at the start of this room
//                         (SimSnapshots), which a rewind can restore
//   repro.run_log         the advisor's run log, when the advisor is up and
//                         the log is small enough: the workbench replays it
//   logs                  tails of the mod's action log, godot.log and the
//                         advisor's stdout -- user paths scrubbed

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SpireMentor;

/// <summary>Everything a report is built from. Every field is optional:
/// a report taken at the main menu has no state, a report with the advisor
/// off has no run log -- it still goes out.</summary>
internal sealed class BugReportParts
{
    public string? Note { get; init; }
    public string ModVersion { get; init; } = "";
    public string GameVersion { get; init; } = "";
    public string Os { get; init; } = "";
    /// <summary>SimRuntime.Mode / Style, as their names.</summary>
    public string CoachMode { get; init; } = "";
    public string CoachStyle { get; init; } = "";
    public bool AdvisorConnected { get; init; }
    /// <summary>When the report was taken, local time with offset.</summary>
    public DateTimeOffset CapturedAt { get; init; } = DateTimeOffset.Now;

    /// <summary>The Copy-state payload (an <c>advice_request</c> or a
    /// <c>run_snapshot</c>), or null with no run.</summary>
    public object? State { get; init; }

    /// <summary>The room-entry snapshot's metadata and its run-save JSON text.</summary>
    public IReadOnlyDictionary<string, object?>? RunSaveMeta { get; init; }
    public string? RunSaveJson { get; init; }

    /// <summary>The advisor's run log (its control plane's <c>GET /log</c>),
    /// or null. <see cref="RunLogOmitted"/> says why it is absent when the
    /// advisor had one.</summary>
    public string? RunLogName { get; init; }
    public string? RunLog { get; init; }
    public string? RunLogOmitted { get; init; }

    public IReadOnlyList<string>? ModLog { get; init; }
    public string? GameLog { get; init; }
    public string? AdvisorLog { get; init; }

    /// <summary>The mod's settings worth knowing about (no secrets live there).</summary>
    public IReadOnlyDictionary<string, object?>? Settings { get; init; }

    /// <summary>Who to scrub out of the logs (Environment.UserName).</summary>
    public string? UserName { get; init; }
}

internal static class BugReportDump
{
    /// <summary>The store's schema (web/functions/api/_bugreport.ts validateDump).</summary>
    public const int SchemaVersion = 1;

    /// <summary>The Pages Function caps a report at 20 MiB; a run log that
    /// would push it near that is left out (its size is still reported).</summary>
    public const int MaxRunLogChars = 8 * 1024 * 1024;

    /// <summary>Log tails: enough for the lead-up to most bugs, small next to
    /// the 20 MiB cap.</summary>
    public const int MaxGameLogChars = 256 * 1024;
    public const int MaxAdvisorLogChars = 128 * 1024;
    public const int MaxModLogLines = 400;

    /// <summary>The user's note is a description, not a document.</summary>
    public const int MaxNoteChars = 4000;

    /// <summary>The dump, ready for <see cref="JsonSerializer"/>.</summary>
    public static Dictionary<string, object?> Build(BugReportParts p)
    {
        IReadOnlyDictionary<string, object?>? state = AsMap(p.State);
        IReadOnlyDictionary<string, object?>? runState = AsMap(Get(state, "run_state"));

        var meta = new Dictionary<string, object?>
        {
            ["source"] = "mod",
            ["mode"] = "game",
            ["mod_version"] = p.ModVersion,
            // The workbench's viewer and listing read `engine_build` as "which
            // game rules": for a game-side report that is the game's build.
            ["engine_build"] = p.GameVersion,
            ["game_version"] = p.GameVersion,
            ["character"] = Get(runState, "character"),
            ["seed"] = NonEmpty(Get(state, "seed")),
            ["ascension"] = Get(runState, "ascension"),
            ["act"] = Get(state, "act"),
            ["floor"] = Get(state, "floor"),
            ["coach"] = new Dictionary<string, object?>
            {
                ["mode"] = p.CoachMode,
                ["style"] = p.CoachStyle,
                ["advisor_connected"] = p.AdvisorConnected,
            },
            ["os"] = p.Os,
            ["captured_local"] = p.CapturedAt.ToString("yyyy-MM-ddTHH:mm:sszzz", CultureInfo.InvariantCulture),
            ["user_agent"] = UserAgent(p.ModVersion),
        };

        var repro = new Dictionary<string, object?>();
        if (p.RunSaveJson != null)
        {
            repro["run_save"] = new Dictionary<string, object?>
            {
                ["snapshot"] = p.RunSaveMeta,
                // Parsed, not a string: a reader sees the save, not an escaped blob.
                ["save"] = ParseOrText(p.RunSaveJson),
            };
        }
        if (p.RunLog != null)
        {
            repro["run_file"] = p.RunLogName;
            repro["run_log"] = p.RunLog;
        }
        else if (p.RunLogOmitted != null)
        {
            repro["run_file"] = p.RunLogName;
            repro["run_log_omitted"] = p.RunLogOmitted;
        }

        var logs = new Dictionary<string, object?>
        {
            ["mod"] = TailLines(p.ModLog, MaxModLogLines, p.UserName),
            ["game"] = p.GameLog == null ? null : Scrub(TailChars(p.GameLog, MaxGameLogChars), p.UserName),
            ["advisor"] = p.AdvisorLog == null ? null : Scrub(TailChars(p.AdvisorLog, MaxAdvisorLogChars), p.UserName),
        };

        return new Dictionary<string, object?>
        {
            ["schema_version"] = SchemaVersion,
            ["meta"] = meta,
            ["repro"] = repro,
            ["state"] = new Dictionary<string, object?> { ["game_state"] = p.State, ["map"] = null },
            ["ui"] = new Dictionary<string, object?> { ["screen"] = ScreenOf(state), ["overlay"] = null },
            ["logs"] = logs,
            ["settings"] = p.Settings,
            ["note"] = CleanNote(p.Note),
        };
    }

    /// <summary>The User-Agent every upload sends. Cloudflare's bot check
    /// answers 403 to a client that sends none.</summary>
    public static string UserAgent(string modVersion) =>
        $"SpireMentor/{(modVersion.Length > 0 ? modVersion : "dev")} (Slay the Spire 2 mod)";

    /// <summary>What the report was taken on: the decision the payload is
    /// for (<c>combat_action</c>, <c>card_reward</c>, ...), <c>run</c> for a
    /// bare run snapshot, null with no run.</summary>
    public static string? ScreenOf(IReadOnlyDictionary<string, object?>? state)
    {
        if (state == null)
        {
            return null;
        }
        if (Get(state, "advice_for") is string adviceFor && adviceFor.Length > 0)
        {
            return adviceFor;
        }
        return Get(state, "type") is "run_snapshot" ? "run" : null;
    }

    /// <summary>Trimmed, capped, null when blank.</summary>
    public static string? CleanNote(string? note)
    {
        if (string.IsNullOrWhiteSpace(note))
        {
            return null;
        }
        string t = note.Trim();
        return t.Length > MaxNoteChars ? t.Substring(0, MaxNoteChars) : t;
    }

    /// <summary>
    /// The last <paramref name="maxChars"/> of <paramref name="text"/>, cut
    /// forward to the next line start so the first line is whole. Text that
    /// fits comes back unchanged.
    /// </summary>
    public static string TailChars(string text, int maxChars)
    {
        if (text.Length <= maxChars)
        {
            return text;
        }
        int start = text.Length - maxChars;
        int nl = text.IndexOf('\n', start);
        return nl >= 0 && nl + 1 < text.Length ? text.Substring(nl + 1) : text.Substring(start);
    }

    /// <summary>The last <paramref name="max"/> lines, scrubbed; null for none.</summary>
    public static List<string>? TailLines(IReadOnlyList<string>? lines, int max, string? userName)
    {
        if (lines == null)
        {
            return null;
        }
        var result = new List<string>(Math.Min(lines.Count, max));
        for (int i = Math.Max(0, lines.Count - max); i < lines.Count; i++)
        {
            result.Add(Scrub(lines[i], userName));
        }
        return result;
    }

    /// <summary>
    /// Replace the user's account name where it is a path segment under
    /// <c>Users</c> / <c>home</c> (<c>C:\Users\alice\AppData</c> becomes
    /// <c>C:\Users\&lt;user&gt;\AppData</c>). Only as a path segment: an
    /// account called "steam" must not blank every mention of Steam.
    /// </summary>
    public static string Scrub(string text, string? userName)
    {
        if (string.IsNullOrEmpty(userName) || text.Length == 0)
        {
            return text;
        }
        string pattern = @"([\\/](?:Users|home)[\\/]+)" + Regex.Escape(userName) + @"(?=[\\/""'\s]|$)";
        return Regex.Replace(text, pattern, "$1<user>", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    /// <summary>
    /// The last <paramref name="maxBytes"/> of a file another process may be
    /// writing (godot.log, the advisor's stdout), starting at a whole line.
    /// Null when it cannot be read; never throws.
    /// </summary>
    public static string? ReadFileTail(string path, int maxBytes)
    {
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            // One byte more than asked: when it is a newline, the tail starts
            // on a whole line and nothing is cut.
            long start = fs.Length > maxBytes ? fs.Length - maxBytes - 1 : 0;
            fs.Seek(start, SeekOrigin.Begin);
            var buf = new byte[fs.Length - start];
            int read = 0;
            while (read < buf.Length)
            {
                int n = fs.Read(buf, read, buf.Length - read);
                if (n <= 0)
                {
                    break;
                }
                read += n;
            }
            string text = Encoding.UTF8.GetString(buf, 0, read);
            if (start > 0)
            {
                // Mid-file: drop through the first newline -- a partial line
                // (maybe a split UTF-8 sequence), or the extra byte alone.
                int nl = text.IndexOf('\n');
                text = nl >= 0 ? text.Substring(nl + 1) : text;
            }
            return text;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>The local copy's file name: sortable, one per second.</summary>
    public static string LocalFileName(DateTimeOffset at) =>
        $"bugreport-{at.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)}.json";

    /// <summary>
    /// The store's reply to an upload: <c>{id, url}</c>. Null when it is not
    /// that (an HTML error page, a proxy's answer, a changed contract).
    /// </summary>
    public static (string Id, string Url)? ParseUploadReply(string body)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            JsonElement root = doc.RootElement;
            if (root.ValueKind == JsonValueKind.Object &&
                root.TryGetProperty("id", out JsonElement id) && id.ValueKind == JsonValueKind.String &&
                root.TryGetProperty("url", out JsonElement url) && url.ValueKind == JsonValueKind.String)
            {
                return (id.GetString()!, url.GetString()!);
            }
        }
        catch (JsonException)
        {
        }
        return null;
    }

    /// <summary>
    /// A redirect where the store should have answered: on the hosted site
    /// that is the sign-in page (Cloudflare Access) -- the intake is not
    /// exempt from it. Uploads never follow redirects: a followed POST would
    /// arrive at the login page as a GET and come back as HTML.
    /// </summary>
    public static string RedirectDetail(int status, string? location)
    {
        string host = "";
        if (location != null && Uri.TryCreate(location, UriKind.Absolute, out Uri? uri))
        {
            host = uri.Host;
        }
        return host.Length > 0
            ? $"HTTP {status}: the report store sent us to {host} (a sign-in page?) -- its /intake/bugreport is not exempt from the site's sign-in"
            : $"HTTP {status}: the report store redirected instead of answering";
    }

    /// <summary>The store's error text (<c>{detail}</c>), else the body's start.</summary>
    public static string FailureDetail(int status, string body)
    {
        string detail = body;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                doc.RootElement.TryGetProperty("detail", out JsonElement d) && d.ValueKind == JsonValueKind.String)
            {
                detail = d.GetString()!;
            }
        }
        catch (JsonException)
        {
        }
        detail = detail.Trim();
        if (detail.Length > 200)
        {
            detail = detail.Substring(0, 200) + "...";
        }
        return detail.Length > 0 ? $"HTTP {status}: {detail}" : $"HTTP {status}";
    }

    private static object? ParseOrText(string json)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            return doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            return json;
        }
    }

    private static IReadOnlyDictionary<string, object?>? AsMap(object? value) => value switch
    {
        IReadOnlyDictionary<string, object?> map => map,
        IDictionary<string, object?> map => new Dictionary<string, object?>(map),
        _ => null,
    };

    private static object? Get(IReadOnlyDictionary<string, object?>? map, string key) =>
        map != null && map.TryGetValue(key, out object? v) ? v : null;

    private static object? NonEmpty(object? value) => value is string s && s.Length == 0 ? null : value;
}

/// <summary>Where reports go: the hosted workbench's origin.</summary>
internal static class BugReportSite
{
    /// <summary>The <see cref="System.Reflection.AssemblyMetadataAttribute"/>
    /// key a release build carries the site under (SpireMentor.csproj, set by
    /// package.ps1). The site is private, so it lives in no source file.</summary>
    public const string MetadataKey = "SpireMentorBugReportSite";

    /// <summary>
    /// The site to upload to: the setting when it is set, else the one baked
    /// into the build, as <c>scheme://host[:port][/path]</c> without a
    /// trailing slash. Null when neither is an http(s) URL -- the report is
    /// then saved to a file only.
    /// </summary>
    public static string? Resolve(string? setting, string? baked)
    {
        foreach (string? candidate in new[] { setting, baked })
        {
            string t = (candidate ?? "").Trim();
            if (t.Length == 0)
            {
                continue;
            }
            return Normalize(t);
        }
        return null;
    }

    /// <summary>The upload route on <paramref name="site"/>: the store's
    /// intake, the one path the site's sign-in (Cloudflare Access) exempts.
    /// Not <c>/api/bugreport</c>: that one is the signed-in page's, and
    /// exempting it would expose the reports under it
    /// (sts2-pycli/web/functions/api/_bugreport.ts).</summary>
    public static string Endpoint(string site) => site + "/intake/bugreport";

    private static string? Normalize(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) ||
            (uri.Scheme != Uri.UriSchemeHttps && uri.Scheme != Uri.UriSchemeHttp) ||
            !string.IsNullOrEmpty(uri.Query) || !string.IsNullOrEmpty(uri.Fragment))
        {
            return null;
        }
        return uri.GetLeftPart(UriPartial.Path).TrimEnd('/');
    }
}
