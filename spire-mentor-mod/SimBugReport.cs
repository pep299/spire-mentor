// SimBugReport.cs -- "Report a bug" from inside the game.
//
// Ctrl+Shift+B, the toolbar's Panels menu or `spirementor bugreport` opens a
// small dialog: a line for what went wrong, then Send or Save to file only.
// The report is the workbench's schema-1 dump (BugReportDump.cs has the
// shape) and carries the moment the dialog OPENED -- the position the player
// was looking at when they saw the bug, not wherever the run went while they
// typed:
//
//   * the Copy-state payload (SimStateCopy), which the workbench's Import
//     state rebuilds;
//   * the game's save at the start of this room (SimSnapshots);
//   * the advisor's run log, through its control plane's GET /log, when the
//     advisor is up;
//   * tails of the action log, godot.log and the advisor's stdout;
//   * mod / game version, OS, coach mode and the mod's settings.
//
// Where it goes. Every report is saved under
// %LOCALAPPDATA%\SpireMentor\bugreports\ first. Send then posts it to the
// hosted workbench's bug-report store -- the same one the workbench's own
// Report button fills, through its intake `<site>/intake/bugreport`, the one
// path the site's sign-in leaves open; the reply's id/URL goes on the
// clipboard and into the file name (reading a report needs the sign-in).
// The site is private, so no source file names it: a release build carries
// it as assembly metadata (package.ps1 -> SpireMentor.csproj), and the
// "Bug report site" setting overrides it. With neither, Send is not offered
// and the dialog says to attach the file instead.
//
// Threading: the dialog and the capture run on the main thread (the state
// builders read live game objects); file reads, the run-log fetch and the
// upload run on a worker; results come back to the main thread for the
// clipboard and the dialog. Only Godot's own API is used for that (and for
// adding the dialog to the tree), so this file names no game namespace.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;

namespace SpireMentor;

internal static class SimBugReport
{
    private const float PanelWidth = 560f;

    private static CanvasLayer? _layer;
    private static PanelContainer? _panel;
    private static LineEdit? _note;
    private static Label? _status;
    private static Button? _send;
    private static Button? _save;
    private static bool _busy;
    private static bool _openDeferred;

    /// <summary>What the dialog captured when it opened.</summary>
    private static Captured? _captured;

    private sealed class Captured
    {
        public DateTimeOffset At { get; init; }
        public Dictionary<string, object>? State { get; init; }
        public IReadOnlyDictionary<string, object?>? RunSaveMeta { get; init; }
        public string? RunSaveJson { get; init; }
        public List<string> ModLog { get; init; } = new();
        public string CoachMode { get; init; } = "";
        public string CoachStyle { get; init; } = "";
        public bool AdvisorConnected { get; init; }
        public IReadOnlyDictionary<string, object?> Settings { get; init; } = new Dictionary<string, object?>();
        public string? GameLogPath { get; init; }
        public string AdvisorLogPath { get; init; } = "";
        public int ControlPort { get; init; }
    }

    /// <summary>The upload site, or null (BugReportSite.Resolve).</summary>
    public static string? Site =>
        BugReportSite.Resolve(SpireMentorModConfig.BugReportSite, BakedSite());

    /// <summary>The site a release build was packaged with, or null.</summary>
    public static string? BakedSite()
    {
        try
        {
            return typeof(SimBugReport).Assembly
                .GetCustomAttributes<AssemblyMetadataAttribute>()
                .FirstOrDefault(a => a.Key == BugReportSite.MetadataKey)?.Value;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Folder every report is saved to before anything is sent.</summary>
    public static string ReportsDir =>
        System.IO.Path.Combine(SimAdvisorProcess.AppDataDir(), "bugreports");

    /// <summary>Open the dialog (capturing this moment), or close it if it is
    /// open. Safe from any thread.</summary>
    public static void Toggle() => RunOnMainThread(() =>
    {
        if (_panel != null && GodotObject.IsInstanceValid(_panel) && _panel.Visible)
        {
            Close();
            return;
        }
        Open();
    });

    /// <summary>
    /// Capture and send without the dialog (the console verb, which already
    /// names its note). Falls back to a file when there is no site.
    /// </summary>
    public static void SendNow(string? note) => RunOnMainThread(() =>
    {
        if (_busy)
        {
            SimActionLog.Note("A bug report is already being sent.");
            return;
        }
        Captured captured = Capture();
        _busy = true;
        Start(captured, note, upload: Site != null);
    });

    private static void Open()
    {
        if (!EnsureBuilt() || _panel == null || _note == null)
        {
            return;
        }
        if (!_panel.IsInsideTree())
        {
            // The layer was added deferred (the root was busy): focus and
            // the viewport size need the tree, so open on the next idle --
            // once; a layer that never arrives is logged, not retried forever.
            if (!_openDeferred)
            {
                _openDeferred = true;
                Callable.From(() => Guarded(Open)).CallDeferred();
            }
            else
            {
                _openDeferred = false;
                Logger.Log("[SimBugReport] Dialog is not in the scene tree; not opening.");
            }
            return;
        }
        _openDeferred = false;
        _captured = Capture();
        _note.Text = "";
        string? site = Site;
        bool siteSettingBad = site == null && SpireMentorModConfig.BugReportSite.Trim().Length > 0;
        if (_send != null)
        {
            _send.Visible = site != null;
        }
        SetStatus(site != null
            ? ""
            : siteSettingBad
                ? "The Bug report site setting is not an http(s) address, so this build cannot send reports. " +
                  "Save it to a file and attach that instead."
                : "This build has no bug-report address. Save the report to a file and attach it to your " +
                  "post (Steam Workshop comments / GitHub).",
            site != null ? Colors.White : new Color(1f, 0.8f, 0.55f));
        SetBusy(false);
        float top = (SimToolbar.Instance.BottomY() ?? 60f) + 8f;
        Vector2 screen = _panel.GetViewportRect().Size;
        _panel.Position = new Vector2(Mathf.Max(16f, (screen.X - PanelWidth) / 2f), top);
        _panel.Visible = true;
        _note.GrabFocus();
        // Editing (not just focus) is what makes NHotkeyManager stand down
        // while the player types: the game's own hotkeys and ours.
        _note.Edit();
    }

    private static void Close()
    {
        if (_panel != null && GodotObject.IsInstanceValid(_panel))
        {
            _panel.GetViewport()?.GuiReleaseFocus();
            _panel.Visible = false;
        }
        _captured = null;
    }

    private static void OnSubmit(bool upload)
    {
        if (_busy)
        {
            return;
        }
        Captured captured = _captured ?? Capture();
        string note = _note?.Text ?? "";
        _note?.Unedit();
        _panel?.GetViewport()?.GuiReleaseFocus();
        SetBusy(true);
        SetStatus(upload ? "Sending…" : "Saving…", Colors.White);
        _busy = true;
        Start(captured, note, upload);
    }

    /// <summary>
    /// Run <see cref="Finish"/> on the thread pool. Not a bare call: an async
    /// method started on the main thread resumes there after each await
    /// (Godot installs a SynchronizationContext), and reading megabytes of
    /// logs and serializing the dump there would hitch the game.
    /// </summary>
    private static void Start(Captured captured, string? note, bool upload) =>
        _ = Task.Run(() => Finish(captured, note, upload));

    /// <summary>The main-thread half: everything that reads the live game.</summary>
    private static Captured Capture()
    {
        Dictionary<string, object>? state = null;
        try
        {
            state = SimStateCopy.BuildState();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] State not captured: {ex.Message}");
        }

        IReadOnlyDictionary<string, object?>? saveMeta = null;
        string? saveJson = null;
        try
        {
            SimSnapshotMeta? room = SimSnapshots.CurrentRoomEntry();
            if (room != null)
            {
                saveJson = SimSnapshots.GetJson(room.Id);
                saveMeta = new Dictionary<string, object?>
                {
                    ["id"] = room.Id,
                    ["floor"] = room.Floor,
                    ["act"] = room.Act,
                    ["post_combat"] = room.PostCombat,
                    ["ts"] = room.Ts,
                };
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Run save not captured: {ex.Message}");
        }

        string? gameLog = null;
        try
        {
            gameLog = System.IO.Path.Combine(OS.GetUserDataDir(), "logs", "godot.log");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] No godot.log path: {ex.Message}");
        }

        int bridgePort = BridgeServer.Instance.Port;
        return new Captured
        {
            At = DateTimeOffset.Now,
            State = state,
            RunSaveMeta = saveMeta,
            RunSaveJson = saveJson,
            ModLog = SimActionLog.Snapshot(null),
            CoachMode = SimRuntime.Mode.ToString(),
            CoachStyle = SimRuntime.Style.ToString(),
            AdvisorConnected = BridgeServer.Instance.IsClientConnected,
            Settings = SettingsSnapshot(),
            GameLogPath = gameLog,
            AdvisorLogPath = SimAdvisorProcess.StdoutLogPath(bridgePort),
            ControlPort = SimAdvisorProcess.ControlPort(bridgePort),
        };
    }

    /// <summary>The worker half: logs, run log, file, upload.</summary>
    private static async Task Finish(Captured c, string? note, bool upload)
    {
        string? site = upload ? Site : null;
        try
        {
            string user = System.Environment.UserName;
            (string? logName, string? logText, string? logOmitted) = await FetchRunLog(c.ControlPort);
            Dictionary<string, object?> dump = BugReportDump.Build(new BugReportParts
            {
                Note = note,
                ModVersion = MainFile.ModVersion,
                GameVersion = SafeGameVersion(),
                Os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                CoachMode = c.CoachMode,
                CoachStyle = c.CoachStyle,
                AdvisorConnected = c.AdvisorConnected,
                CapturedAt = c.At,
                State = c.State,
                RunSaveMeta = c.RunSaveMeta,
                RunSaveJson = c.RunSaveJson,
                RunLogName = logName,
                RunLog = logText,
                RunLogOmitted = logOmitted,
                ModLog = c.ModLog,
                GameLog = c.GameLogPath == null ? null : BugReportDump.ReadFileTail(c.GameLogPath, BugReportDump.MaxGameLogChars),
                AdvisorLog = BugReportDump.ReadFileTail(c.AdvisorLogPath, BugReportDump.MaxAdvisorLogChars),
                Settings = c.Settings,
                UserName = user,
            });
            string json = JsonSerializer.Serialize(dump);

            string? path = WriteLocal(json, c.At);
            if (site == null)
            {
                string msg = path != null
                    ? $"Bug report saved to {path}. Attach that file to your report."
                    : "Could not save the bug report (see godot.log).";
                Done(msg, ok: path != null, sent: false, copy: path);
                return;
            }

            (string? id, string? url, string? error) = await Upload(site, json);
            if (id != null && url != null)
            {
                string? renamed = path == null ? null : RenameWithId(path, id);
                Logger.Log($"[SimBugReport] Sent {id} ({json.Length} chars): {url}");
                Done($"Bug report sent: {id}. The link is on your clipboard -- paste it into your post." +
                     (renamed != null ? $" (Saved too: {renamed})" : ""), ok: true, sent: true, copy: url);
            }
            else
            {
                Logger.Log($"[SimBugReport] Upload to {site} failed: {error}");
                Done(path != null
                        ? $"Could not send the bug report ({error}). It is saved to {path} -- attach that file instead."
                        : $"Could not send or save the bug report ({error}).",
                    ok: false, sent: false, copy: path);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Report failed: {ex}");
            Done($"Bug report failed: {ex.Message}", ok: false, sent: false, copy: null);
        }
    }

    /// <summary>Report the outcome on the main thread: action log, dialog,
    /// and the clipboard (the URL, or the saved file's path).</summary>
    private static void Done(string message, bool ok, bool sent, string? copy)
    {
        SimActionLog.Note(message);
        RunOnMainThread(() =>
        {
            _busy = false;
            if (copy != null)
            {
                try { DisplayServer.ClipboardSet(copy); }
                catch (Exception ex) { Logger.Log($"[SimBugReport] Clipboard: {ex.Message}"); }
            }
            if (_panel != null && GodotObject.IsInstanceValid(_panel) && _panel.Visible)
            {
                SetBusy(false);
                SetStatus(message, ok ? new Color(0.6f, 1f, 0.6f) : new Color(1f, 0.7f, 0.5f));
                if (sent && _send != null)
                {
                    // Sent: a second press would file the same report twice.
                    _send.Visible = false;
                }
            }
        });
    }

    /// <summary>
    /// The advisor's run log via its control plane (<c>GET /log</c>), when
    /// the advisor is up: (name, text, why-omitted). The size is asked first
    /// so a huge log is never pulled just to be dropped.
    /// </summary>
    private static async Task<(string? Name, string? Text, string? Omitted)> FetchRunLog(int controlPort)
    {
        try
        {
            using var http = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            string baseUrl = $"http://127.0.0.1:{controlPort}/log";
            using HttpResponseMessage statResp = await http.GetAsync(baseUrl + "?stat=1");
            if (!statResp.IsSuccessStatusCode)
            {
                return (null, null, null);
            }
            using JsonDocument stat = JsonDocument.Parse(await statResp.Content.ReadAsStringAsync());
            string? name = stat.RootElement.TryGetProperty("name", out JsonElement n) ? n.GetString() : null;
            long size = stat.RootElement.TryGetProperty("size", out JsonElement s) && s.TryGetInt64(out long v) ? v : 0;
            if (size > BugReportDump.MaxRunLogChars)
            {
                return (name, null, $"{size} bytes, over the {BugReportDump.MaxRunLogChars}-byte cap");
            }
            using HttpResponseMessage resp = await http.GetAsync(baseUrl);
            if (!resp.IsSuccessStatusCode)
            {
                return (name, null, $"GET /log answered {(int)resp.StatusCode}");
            }
            using JsonDocument doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
            string? text = doc.RootElement.TryGetProperty("text", out JsonElement t) ? t.GetString() : null;
            return (name, text, text == null ? "GET /log carried no text" : null);
        }
        catch (Exception)
        {
            // No advisor (coach off), or not this build's control plane: the
            // report goes out without a run log.
            return (null, null, null);
        }
    }

    private static async Task<(string? Id, string? Url, string? Error)> Upload(string site, string json)
    {
        try
        {
            // No redirects: on the hosted site a redirect is the sign-in page,
            // and a followed POST would come back from it as HTML.
            using var handler = new HttpClientHandler { AllowAutoRedirect = false };
            using var http = new System.Net.Http.HttpClient(handler) { Timeout = TimeSpan.FromSeconds(60) };
            using var req = new HttpRequestMessage(HttpMethod.Post, BugReportSite.Endpoint(site))
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json"),
            };
            req.Headers.TryAddWithoutValidation("User-Agent", BugReportDump.UserAgent(MainFile.ModVersion));
            using HttpResponseMessage resp = await http.SendAsync(req);
            string body = await resp.Content.ReadAsStringAsync();
            int code = (int)resp.StatusCode;
            if (code >= 300 && code < 400)
            {
                return (null, null, BugReportDump.RedirectDetail(code, resp.Headers.Location?.ToString()));
            }
            if (!resp.IsSuccessStatusCode)
            {
                return (null, null, BugReportDump.FailureDetail((int)resp.StatusCode, body));
            }
            (string Id, string Url)? reply = BugReportDump.ParseUploadReply(body);
            return reply is { } r ? (r.Id, r.Url, null) : (null, null, "the store's reply was not {id, url}");
        }
        catch (TaskCanceledException)
        {
            return (null, null, "timed out");
        }
        catch (Exception ex)
        {
            return (null, null, ex.Message);
        }
    }

    private static string? WriteLocal(string json, DateTimeOffset at)
    {
        try
        {
            Directory.CreateDirectory(ReportsDir);
            string path = System.IO.Path.Combine(ReportsDir, BugReportDump.LocalFileName(at));
            File.WriteAllText(path, json, new UTF8Encoding(false));
            return path;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Local save failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>`bugreport-<ts>.json` -> `bugreport-<ts>-<id>.json`, so the
    /// file says which report it is. Returns the path it ends up at.</summary>
    private static string? RenameWithId(string path, string id)
    {
        try
        {
            string target = System.IO.Path.Combine(
                System.IO.Path.GetDirectoryName(path) ?? ReportsDir,
                System.IO.Path.GetFileNameWithoutExtension(path) + "-" + id + ".json");
            File.Move(path, target, overwrite: true);
            return target;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Rename failed: {ex.Message}");
            return path;
        }
    }

    /// <summary>The mod's own settings (simple values only) plus the conf
    /// keys that change behavior. Read on the main thread.</summary>
    private static IReadOnlyDictionary<string, object?> SettingsSnapshot()
    {
        var result = new Dictionary<string, object?>();
        try
        {
            foreach (PropertyInfo prop in typeof(SpireMentorModConfig).GetProperties(BindingFlags.Public | BindingFlags.Static))
            {
                Type t = prop.PropertyType;
                if (!(t.IsPrimitive || t.IsEnum || t == typeof(string) || t == typeof(decimal)))
                {
                    continue;
                }
                object? v = prop.GetValue(null);
                result[prop.Name] = v is string s
                    ? BugReportDump.Scrub(s, System.Environment.UserName)
                    : t.IsEnum ? v?.ToString() : v;
            }
            SimConfig conf = SimConfig.Instance;
            result["conf.port"] = conf.Port;
            result["conf.auto_start"] = conf.AutoStart;
            result["conf.advisor_autoplay"] = conf.AdvisorAutoplay;
            result["conf.fast_mode"] = conf.FastMode;
            result["bridge.port"] = BridgeServer.Instance.Port;
        }
        catch (Exception ex)
        {
            result["error"] = ex.Message;
        }
        return result;
    }

    private static string SafeGameVersion()
    {
        try { return SimGameVersion.Wire; }
        catch { return ""; }
    }

    // ------------------------------------------------------------------
    // Dialog
    // ------------------------------------------------------------------

    private static bool EnsureBuilt()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return false;
            }
            var layer = new CanvasLayer { Name = "SpireMentorBugReportLayer", Layer = 103 };

            // An explicitly opened dialog: it may block the game beneath it
            // (same contract as the settings panel).
            var panel = new PanelContainer
            {
                Name = "SpireMentorBugReportPanel",
                CustomMinimumSize = new Vector2(PanelWidth, 0f),
                MouseFilter = Control.MouseFilterEnum.Stop,
                Visible = false,
            };
            panel.AddThemeStyleboxOverride("panel", SimUiKit.PanelStyle());

            var rows = new VBoxContainer();
            rows.AddThemeConstantOverride("separation", 6);

            var title = new HBoxContainer();
            Label titleLabel = SimUiKit.SmallLabel("Report a bug", Colors.White, 16);
            titleLabel.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
            title.AddChild(titleLabel);
            title.AddChild(SimUiKit.MakeButton("✕", "Close", Close));
            rows.AddChild(title);

            rows.AddChild(SimUiKit.SmallLabel("What went wrong? (optional, Enter sends)", new Color(0.85f, 0.88f, 0.95f)));

            var note = new LineEdit
            {
                PlaceholderText = "e.g. the advisor recommended a card I could not play",
                MaxLength = BugReportDump.MaxNoteChars,
                SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
                MouseFilter = Control.MouseFilterEnum.Stop,
            };
            note.AddThemeFontSizeOverride("font_size", 15);
            var noteStyle = SimUiKit.PanelStyle();
            noteStyle.BgColor = new Color(0.08f, 0.09f, 0.12f, 0.95f);
            note.AddThemeStyleboxOverride("normal", noteStyle);
            note.AddThemeStyleboxOverride("focus", noteStyle);
            note.AddThemeColorOverride("font_color", Colors.White);
            note.Connect(LineEdit.SignalName.TextSubmitted, Callable.From((string _) =>
                Guarded(() => OnSubmit(upload: Site != null))));
            rows.AddChild(note);

            Label what = SimUiKit.SmallLabel(
                "The report carries this position (as when this box opened), the start-of-room save, " +
                "the advisor's run log, the last lines of the mod, game and advisor logs (your Windows " +
                "user name is taken out of file paths), and the mod's settings. Nothing else.",
                new Color(0.75f, 0.78f, 0.85f), 13);
            what.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            what.CustomMinimumSize = new Vector2(PanelWidth - 24f, 0f);
            rows.AddChild(what);

            var buttons = new HBoxContainer();
            buttons.AddThemeConstantOverride("separation", 8);
            Button send = SimUiKit.MakeButton("Send report", "Send it to the Spire Mentor developers", () => OnSubmit(upload: true));
            Button save = SimUiKit.MakeButton("Save to file only", "Write it to a file and send nothing", () => OnSubmit(upload: false));
            buttons.AddChild(send);
            buttons.AddChild(save);
            buttons.AddChild(SimUiKit.MakeButton("Open folder", "Show the saved reports",
                () => OpenFolder(ReportsDir)));
            rows.AddChild(buttons);

            Label status = SimUiKit.SmallLabel("", Colors.White, 14);
            status.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            status.CustomMinimumSize = new Vector2(PanelWidth - 24f, 0f);
            status.Visible = false;
            rows.AddChild(status);

            panel.AddChild(rows);
            layer.AddChild(panel);
            if (tree.Root.IsNodeReady())
            {
                tree.Root.AddChild(layer);
            }
            else
            {
                tree.Root.CallDeferred(Node.MethodName.AddChild, layer);
            }

            _layer = layer;
            _panel = panel;
            _note = note;
            _status = status;
            _send = send;
            _save = save;
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Dialog failed: {ex.Message}");
            _layer = null;
            _panel = null;
            return false;
        }
    }

    private static void SetStatus(string text, Color color)
    {
        if (_status == null || !GodotObject.IsInstanceValid(_status))
        {
            return;
        }
        _status.Text = text;
        _status.AddThemeColorOverride("font_color", color);
        _status.Visible = text.Length > 0;
        _panel?.ResetSize();
    }

    private static void SetBusy(bool busy)
    {
        if (_send != null && GodotObject.IsInstanceValid(_send)) _send.Disabled = busy;
        if (_save != null && GodotObject.IsInstanceValid(_save)) _save.Disabled = busy;
        if (_note != null && GodotObject.IsInstanceValid(_note)) _note.Editable = !busy;
    }

    /// <summary>Explorer on the folder -- through the shell, not as a child
    /// of the game (SimWorkbenchLink has why).</summary>
    private static void OpenFolder(string dir)
    {
        try
        {
            Directory.CreateDirectory(dir);
            using var p = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = "\"" + dir + "\"",
                UseShellExecute = false,
            });
        }
        catch (Exception ex)
        {
            SimActionLog.Note($"Bug reports are in {dir}");
            Logger.Log($"[SimBugReport] Open folder failed: {ex.Message}");
        }
    }

    private static void RunOnMainThread(Action action)
    {
        try
        {
            if (OS.GetThreadCallerId() == OS.GetMainThreadId())
            {
                Guarded(action);
            }
            else
            {
                Callable.From(() => Guarded(action)).CallDeferred();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Dispatch failed: {ex.Message}");
        }
    }

    private static void Guarded(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimBugReport] Ignoring UI error (never breaks gameplay): {ex.Message}");
        }
    }
}
