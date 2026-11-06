// SimAiPartnerContinue.cs -- continuing a saved co-op run that has AI
// partners (docs/plans/2026-11-05-ai-coop-partners.md, phase 4).
//
// The multiplayer menu's "Load" hosts the saved run the game's way (Steam, or
// ENet under fastmp) in a load lobby (NMultiplayerLoadGameScreen,
// LoadRunLobby), which waits for every saved player to rejoin. A partner's
// seat is in the save under its net id (SimAiPartner.NetIds), so when the
// lobby opens, the partners of those seats join it next to the host's game
// host (SimAiPartnerSession.ForLobby), through the game's own JoinFlow (which
// answers a load lobby with ClientLoadJoinRequestMessage), and ready up.
// Human players of the save rejoin as usual.
//
// `--spire-mentor-ai-partner-continue` (the test lane) presses "Load" from
// the main menu; with `--spire-mentor-autoplay` the host also embarks once
// every saved player is back.

using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace SpireMentor;

internal static class SimAiPartnerContinue
{
    private const string MainMenuNodePath = "/root/Game/RootSceneContainer/MainMenu";
    private static readonly TimeSpan MainMenuWait = TimeSpan.FromSeconds(90);

    private static readonly FieldInfo? RunLobbyField =
        AccessTools.Field(typeof(NMultiplayerLoadGameScreen), "_runLobby");

    public static void Install(Harmony patcher)
    {
        MethodInfo? init = AccessTools.Method(typeof(NMultiplayerLoadGameScreen), nameof(NMultiplayerLoadGameScreen.InitializeAsHost));
        if (init == null)
        {
            Logger.Log("  patch skipped: SimAiPartnerContinue (NMultiplayerLoadGameScreen.InitializeAsHost not found)");
            return;
        }
        patcher.Patch(init, postfix: new HarmonyMethod(typeof(SimAiPartnerContinue), nameof(InitializeAsHostPostfix)));
        Logger.Log("  patch applied: SimAiPartnerContinue (load lobby)");
    }

    /// <summary>The seat of a saved player that is an AI partner, else -1.</summary>
    private static int SeatOf(ulong netId) => Array.IndexOf(SimAiPartner.NetIds, netId);

    private static void InitializeAsHostPostfix(NMultiplayerLoadGameScreen __instance, INetGameService gameService, SerializableRun run)
    {
        try
        {
            var seats = run.Players.Where(p => SeatOf(p.NetId) >= 0).ToList();
            if (seats.Count == 0 || gameService is not NetHostGameService host)
            {
                return;
            }
            SimAiPartnerSession? session = SimAiPartnerSession.ForLobby(host);
            if (session == null)
            {
                Logger.Log("[SimAiPartner] The saved run has AI partners, but this load lobby has no game host to join; they cannot rejoin.");
                return;
            }
            foreach (var saved in seats)
            {
                CharacterModel? character = ModelDb.AllCharacters.FirstOrDefault(c => saved.CharacterId is { } id && c.Id.Equals(id));
                SimAiPartner? partner = session.Add(null, __instance, SeatOf(saved.NetId), character ?? ModelDb.AllCharacters.First());
                Logger.Log($"[SimAiPartner] Continuing a saved run: {partner?.Label ?? "a partner"} rejoins as {character?.Id.Entry ?? "?"}.");
            }
            if (SimMp.AiPartnerContinue && SimMp.AutoplayOverride && Engine.GetMainLoop() is SceneTree tree)
            {
                TaskHelper.RunSafely(EmbarkWhenAllBackAsync(tree, __instance, run.Players.Count));
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] Continuing a saved run: {ex}");
        }
    }

    /// <summary>The lane's host: embark once every saved player is back in
    /// the load lobby.</summary>
    private static async Task EmbarkWhenAllBackAsync(SceneTree tree, NMultiplayerLoadGameScreen screen, int players)
    {
        DateTime deadline = DateTime.UtcNow + TimeSpan.FromMinutes(3);
        while (DateTime.UtcNow < deadline && GodotObject.IsInstanceValid(screen))
        {
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            if (RunLobbyField?.GetValue(screen) is not LoadRunLobby lobby
                || (lobby.GetType().GetProperty("Players")?.GetValue(lobby) as ICollection)?.Count < players)
            {
                continue;
            }
            if (screen.GetNodeOrNull<NClickableControl>("ConfirmButton") is { IsEnabled: true } confirm && confirm.IsVisibleInTree())
            {
                await Task.Delay(1000);
                Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: every saved player is back; embarking.");
                await UiHelper.Click(confirm);
                return;
            }
        }
        Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: not every saved player came back.");
    }

    /// <summary><c>--spire-mentor-ai-partner-continue</c>: press the
    /// multiplayer menu's "Load" from the main menu.</summary>
    public static async Task ContinueFromCommandLineAsync(SceneTree tree)
    {
        DateTime deadline = DateTime.UtcNow + MainMenuWait;
        NMainMenu? menu = null;
        while (DateTime.UtcNow < deadline)
        {
            menu = tree.Root.GetNodeOrNull<NMainMenu>(MainMenuNodePath);
            if (menu != null && menu.IsVisibleInTree() && menu.SubmenuStack != null)
            {
                break;
            }
            menu = null;
            await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }
        if (menu == null)
        {
            Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: the main menu never showed.");
            return;
        }
        await Task.Delay(1500);
        if (!SaveManager.Instance.HasMultiplayerRunSave)
        {
            Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: no saved co-op run.");
            return;
        }
        NMultiplayerSubmenu submenu = menu.SubmenuStack.GetSubmenuType<NMultiplayerSubmenu>();
        menu.SubmenuStack.Push(submenu);
        await Task.Delay(1500);
        MethodInfo? load = AccessTools.Method(typeof(NMultiplayerSubmenu), "StartLoad");
        if (load == null)
        {
            Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: NMultiplayerSubmenu.StartLoad not found.");
            return;
        }
        Logger.Log("[SimAiPartner] --spire-mentor-ai-partner-continue: pressing Load.");
        load.Invoke(submenu, new object?[] { null });
    }
}
