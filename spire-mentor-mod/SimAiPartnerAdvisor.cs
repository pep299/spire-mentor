// SimAiPartnerAdvisor.cs -- an AI partner's own advisor
// (docs/plans/2026-11-05-ai-coop-partners.md, phase 2).
//
// Each partner gets the advisor the human gets -- the same Python process
// (MCTS for combat, the PPO run policy for screens), launched the same way
// (SimAdvisorProcess.ResolveLaunch) -- on its own secondary bridge
// (BridgeServer.CreateSecondary: advice lane only, a free port). The
// advisor plans for whoever the board's `player` is, with the other players
// in `allies`, so it needs nothing partner-specific: the partner sends its
// own board (built from its own Player, with the game's "me" pointed at it
// for the span of the build: SimAiPartnerBrain.BuildAsMe) and executes the
// reply's `best` as network messages.
//
// The human's advisor machinery (SimAdvisor, SimAdvisorProcess, SimHud) is
// a singleton wired to the local player and the HUD; none of it is used.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace SpireMentor;

internal sealed class SimAiPartnerAdvisor
{
    /// <summary>Times the advisor is relaunched after it exits on its own,
    /// per session; past that the partner keeps its first-legal fallback.</summary>
    private const int MaxRelaunches = 2;

    private readonly SimAiPartner _partner;
    private readonly object _lock = new();
    private BridgeServer? _bridge;
    private Process? _process;
    private bool _stopping;
    private int _relaunches;

    public SimAiPartnerAdvisor(SimAiPartner partner)
    {
        _partner = partner;
    }

    /// <summary>An advisor is attached and can take requests.</summary>
    public bool Connected => _bridge?.IsClientConnected == true;

    public void Start()
    {
        try
        {
            _bridge = BridgeServer.CreateSecondary(_partner.Label);
            _bridge.Start(0);
            Launch();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: could not start ({ex.Message}); it plays without an advisor.");
        }
    }

    public void Stop()
    {
        lock (_lock)
        {
            _stopping = true;
            try
            {
                if (_process is { HasExited: false })
                {
                    _process.Kill(entireProcessTree: true);
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: stop: {ex.Message}");
            }
            _process = null;
        }
        try
        {
            _bridge?.Stop();
        }
        catch (Exception)
        {
            // Already down.
        }
        _bridge = null;
    }

    /// <summary>Sends one advice_request; <paramref name="onReply"/> runs on
    /// the socket thread. False when there is no advisor to ask.</summary>
    public bool Request(Dictionary<string, object> state, Action<JsonElement> onReply)
    {
        BridgeServer? bridge = _bridge;
        if (bridge == null || !bridge.IsClientConnected)
        {
            return false;
        }
        string json;
        try
        {
            json = JsonSerializer.Serialize(state);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: could not serialize the request: {ex.Message}");
            return false;
        }
        return bridge.SendAdviceRequest(json, onReply) != null;
    }

    /// <summary>A fire-and-forget message (the human_action that keeps the
    /// advisor's combat mirror in step).</summary>
    public void Report(Dictionary<string, object?> message)
    {
        try
        {
            _bridge?.SendState(JsonSerializer.Serialize(message));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: could not report: {ex.Message}");
        }
    }

    private void Launch()
    {
        if (!OperatingSystem.IsWindows())
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: the bundled advisor is Windows-only; no advisor.");
            return;
        }
        int port = _bridge?.Port ?? 0;
        if (port == 0)
        {
            return;
        }
        // Its own control plane on a free port (0), no workbench server, and
        // a lighter search than the human's: up to three of these share the
        // machine with the human's advisor and the game.
        string args = $"--port {port} --control-port 0 --workbench off " +
                      $"--sims {SimMp.AiPartnerSims} --boss-sims {SimMp.AiPartnerSims * 4} --threads 2";
        string logDir = Path.Combine(SimAdvisorProcess.AppDataDir(), "logs", $"realgame_advisor_partner{_partner.Seat + 1}");
        ProcessStartInfo? psi = SimAdvisorProcess.ResolveLaunch(args, logDir);
        if (psi == null)
        {
            return;
        }
        string stdoutPath = Path.Combine(SimAdvisorProcess.AppDataDir(), $"advisor_stdout.partner{_partner.Seat + 1}.log");
        TextWriter? log = null;
        WindowsChildProcess? child = null;
        try
        {
            SimAdvisorProcess.RotateOversizedLog(stdoutPath);
            log = TextWriter.Synchronized(new StreamWriter(new FileStream(
                stdoutPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete)));
            log.WriteLine($"--- {_partner.Label} advisor launched {DateTime.Now:yyyy-MM-dd HH:mm:ss} (port {port}) ---");
            child = WindowsChildProcess.Start(psi.FileName, psi.Arguments, psi.WorkingDirectory,
                process => SimJobObject.Adopt(process));
            lock (_lock)
            {
                _process = child.Process;
            }
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: advisor launched (pid {child.Process.Id}, port {port}); output -> {stdoutPath}");
            _ = MonitorAsync(child, log);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: launch failed: {ex.Message}");
            try
            {
                log?.Dispose();
            }
            catch (IOException)
            {
            }
            child?.Dispose();
        }
    }

    private async Task MonitorAsync(WindowsChildProcess child, TextWriter log)
    {
        int code = -1;
        try
        {
            await Task.WhenAll(
                Pump(child.StandardOutput, log),
                Pump(child.StandardError, log),
                child.Process.WaitForExitAsync()).ConfigureAwait(false);
            code = child.Process.ExitCode;
            SimAdvisorProcess.TryWrite(log, $"--- advisor exited ({code}) ---");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: monitoring failed: {ex.Message}");
        }
        finally
        {
            try
            {
                log.Dispose();
            }
            catch (IOException)
            {
            }
            child.Dispose();
        }
        bool relaunch;
        lock (_lock)
        {
            relaunch = !_stopping && _relaunches < MaxRelaunches;
            if (relaunch)
            {
                _relaunches++;
            }
        }
        if (relaunch)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: advisor exited ({code}); relaunching ({_relaunches}/{MaxRelaunches}).");
            Launch();
        }
        else if (!_stopping)
        {
            Logger.Log($"[SimAiPartnerAdvisor] {_partner.Label}: advisor exited ({code}); the partner plays without it.");
        }
    }

    private static async Task Pump(StreamReader reader, TextWriter log)
    {
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            SimAdvisorProcess.TryWrite(log, line);
        }
    }
}
