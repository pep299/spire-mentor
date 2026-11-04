// SimRewardsScreenHandler.cs -- post-combat/event rewards screen, decided by the agent.
// Originally ported from zhiyue/sts2-rl-agent (STS2BridgeMod); rewritten clean-room.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

public class SimRewardsScreenHandler : IScreenHandler, IHandler
{
    private const string LogName = nameof(NRewardsScreen);
    private const string UnknownRewardLabel = "unknown";
    private const string GoldLabelPrefix = "gold:";

    private const int RewardClickSettleMs = 500;
    private static readonly TimeSpan ProceedCloseWait = TimeSpan.FromSeconds(10);

    public Type ScreenType => typeof(NRewardsScreen);

    public TimeSpan Timeout => TimeSpan.FromHours(24);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.EnterScreen(LogName);
        NRewardsScreen screen = AutoSlayer.GetCurrentScreen<NRewardsScreen>();
        var claimed = new HashSet<NRewardButton>(ReferenceEqualityComparer.Instance);

        while (true)
        {
            ct.ThrowIfCancellationRequested();

            List<NRewardButton> claimable = ClaimableRewards(screen, claimed);
            List<NRewardButton> blocked = FullBeltPotionRewards(screen, claimed);
            NProceedButton? proceed = UiHelper.FindFirst<NProceedButton>(screen);

            // Nothing left to take: leaving is the only move, so it is not asked.
            if (claimable.Count == 0 && blocked.Count == 0)
            {
                if (proceed != null)
                    await ProceedAndCloseAsync(screen, proceed, ct);
                AutoSlayLog.ExitScreen(LogName);
                return;
            }

            RewardOptionSlot choice = await ChooseRewardScreenOption(claimable, blocked, proceed, ct);

            if (choice.Kind == RewardOptionKind.Proceed)
            {
                if (proceed != null)
                    await ProceedAndCloseAsync(screen, proceed, ct);
                AutoSlayLog.ExitScreen(LogName);
                return;
            }

            NRewardButton button;
            if (choice.Kind == RewardOptionKind.Pick)
            {
                button = claimable[choice.Button];
            }
            else
            {
                button = blocked[choice.Button];
                if (choice.Kind == RewardOptionKind.SwapDecline)
                {
                    // Left on the screen, as a player who walks past it. Not
                    // offered again on this screen.
                    claimed.Add(button);
                    AutoSlayLog.Action($"Rewards: leaving {RewardLabel(button)} (belt full)");
                    continue;
                }
                await DiscardForSwapAsync(choice.DiscardSlot, ct);
            }

            claimed.Add(button);
            AutoSlayLog.Action($"Clicking reward button: {RewardLabel(button)}");
            await UiHelper.Click(button);
            await Task.Delay(RewardClickSettleMs, ct);

            if (!IsTopScreen(screen))
            {
                AutoSlayLog.Info("Child screen opened, returning to drain loop");
                AutoSlayLog.ExitScreen(LogName);
                return;
            }
        }
    }

    /// <summary>
    /// Frees belt slot <paramref name="slot"/> for a full-belt potion reward:
    /// the discard the game's potion popup offers on the reward screen
    /// (PotionCmd.Discard). The reward's click then procures into that slot,
    /// the first free one.
    /// </summary>
    private static async Task DiscardForSwapAsync(int slot, CancellationToken ct)
    {
        Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        if (me == null)
            return;
        AutoSlayLog.Action($"Rewards: discarding potion slot {slot} to make room");
        await SimCombatHandler.DiscardPotionAndWaitAsync(me, slot, ct);
    }

    /// <summary>
    /// Enabled reward buttons not yet clicked by this handler. Potions are
    /// only claimable when the player can take them (free slot, no Sozu).
    /// </summary>
    private static List<NRewardButton> ClaimableRewards(
        NRewardsScreen screen, HashSet<NRewardButton> claimed)
    {
        Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        return UiHelper.FindAll<NRewardButton>(screen)
            .Where(b => b.IsEnabled && !claimed.Contains(b))
            .Where(b => b.Reward is not PotionReward pr || (me != null && SimPotionRules.CanProcure(me, pr.Potion)))
            .ToList();
    }

    /// <summary>
    /// Potion rewards that only a full belt blocks (GH #1069). Discarding a
    /// held potion makes them claimable, so they are offered as a swap. A
    /// potion a hook refuses (Sozu) stays hidden: no discard helps.
    /// </summary>
    private static List<NRewardButton> FullBeltPotionRewards(
        NRewardsScreen screen, HashSet<NRewardButton> claimed)
    {
        Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        if (me == null)
            return new List<NRewardButton>();
        return UiHelper.FindAll<NRewardButton>(screen)
            .Where(b => b.IsEnabled && !claimed.Contains(b))
            .Where(b => b.Reward is PotionReward pr && SimPotionRules.BlockedOnlyByFullBelt(me, pr.Potion))
            .ToList();
    }

    private static bool IsTopScreen(NRewardsScreen screen)
    {
        NOverlayStack? stack = NOverlayStack.Instance;
        return stack != null && stack.ScreenCount > 0 && ReferenceEquals(stack.Peek(), screen);
    }

    private static bool MapOpen() => NMapScreen.Instance?.IsOpen ?? false;

    private static async Task ProceedAndCloseAsync(
        NRewardsScreen screen, NProceedButton proceed, CancellationToken ct)
    {
        AutoSlayLog.Action("Rewards: clicking proceed");
        await UiHelper.Click(proceed);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen) || !IsTopScreen(screen) || MapOpen(),
            ct, ProceedCloseWait, "Rewards screen did not close after proceed");
    }

    /// <summary>
    /// Asks the agent what to do next. The options follow
    /// <see cref="RewardSwapLayout"/>: each claimable reward, then a decline
    /// option and one discard option per belt slot for every full-belt potion
    /// reward, then the trailing "proceed", which is only offered while the
    /// proceed button is enabled. Proceed is re-checked for every ask and
    /// again when the reply arrives.
    /// </summary>
    private static async Task<RewardOptionSlot> ChooseRewardScreenOption(
        List<NRewardButton> buttons, List<NRewardButton> blocked, NProceedButton? proceed, CancellationToken ct)
    {
        Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        List<string?> belt = me?.PotionSlots.Select(p => p?.Id.Entry).ToList() ?? new List<string?>();
        List<RewardOptionSlot> layout = RewardSwapLayout.Build(
            buttons.Count, blocked.Select(_ => belt.Count).ToList(), proceed != null);
        bool proceedOffered = false;

        int index = await SimAgentChoice.AskAsync(
            (_, _) =>
            {
                var options = new List<object>();
                for (int i = 0; i < layout.Count; i++)
                {
                    RewardOptionSlot slot = layout[i];
                    switch (slot.Kind)
                    {
                        case RewardOptionKind.Pick:
                            options.Add(RewardOption(buttons[slot.Button], i));
                            break;
                        case RewardOptionKind.SwapDecline:
                        case RewardOptionKind.Swap:
                            options.Add(SwapOption(blocked[slot.Button], slot, belt, i));
                            break;
                        case RewardOptionKind.Proceed:
                            proceedOffered = ProceedEnabled(proceed);
                            if (proceedOffered)
                            {
                                options.Add(new Dictionary<string, object>
                                {
                                    ["index"] = i,
                                    ["id"] = NonCombatBridgeProtocol.ProceedAction,
                                    ["action"] = NonCombatBridgeProtocol.ProceedAction,
                                    ["label"] = NonCombatBridgeProtocol.ProceedAction,
                                    ["enabled"] = true,
                                });
                            }
                            break;
                    }
                }
                return new Dictionary<string, object>
                {
                    [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.RewardScreenState,
                    ["options"] = options,
                };
            },
            layout.Count,
            NonCombatBridgeProtocol.RewardScreenState, "SimRewards", ct,
            rejectAtReply: i => layout[i].Kind == RewardOptionKind.Proceed && !(proceedOffered && ProceedEnabled(proceed))
                ? $"index {i} (proceed) is not enabled"
                : null);

        return layout[index];
    }

    /// <summary>
    /// One option of a full-belt potion's swap group. Shaped like the sim's
    /// `potion_swap` decision option: `discard_slot` null declines, slot k
    /// discards the held potion there. `reward_button` tells groups apart
    /// when two potion rewards are blocked.
    /// </summary>
    private static Dictionary<string, object?> SwapOption(
        NRewardButton button, RewardOptionSlot slot, List<string?> belt, int index)
    {
        bool decline = slot.Kind == RewardOptionKind.SwapDecline;
        string label = decline
            ? "decline"
            : $"discard {belt[slot.DiscardSlot] ?? "empty"}";
        return new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = RewardLabel(button),
            ["action"] = NonCombatBridgeProtocol.PotionSwapAction,
            ["label"] = label,
            ["description"] = RewardDescription(button),
            ["enabled"] = true,
            ["reward"] = RewardDetail(button),
            ["reward_button"] = slot.Button,
            ["discard_slot"] = decline ? null : slot.DiscardSlot,
        };
    }

    private static bool ProceedEnabled(NProceedButton? proceed) =>
        proceed != null && GodotObject.IsInstanceValid(proceed) && proceed.IsEnabled;

    private static Dictionary<string, object> RewardOption(NRewardButton button, int index)
    {
        string label = RewardLabel(button);
        var option = new Dictionary<string, object>
        {
            ["index"] = index,
            ["id"] = label,
            ["action"] = NonCombatBridgeProtocol.PickRewardAction,
            ["label"] = label,
            ["description"] = RewardDescription(button),
            ["enabled"] = button.IsEnabled,
        };
        object? detail = RewardDetail(button);
        if (detail != null)
        {
            option["reward"] = detail;
        }
        return option;
    }

    private static object? RewardDetail(NRewardButton button)
    {
        try
        {
            return button.Reward switch
            {
                CardReward cardReward => cardReward.Cards.Select(c => SimStateExtractor.SerializeCard(c)).ToList(),
                PotionReward { Potion: not null } potionReward => SimStateExtractor.SerializePotionSlot(potionReward.Potion, -1),
                RelicReward { Relic: not null } relicReward => SimStateExtractor.SerializeRelic(relicReward.Relic),
                GoldReward goldReward => new Dictionary<string, object?> { ["amount"] = goldReward.Amount },
                _ => null,
            };
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn($"[SimRewards] Error building reward detail: {ex.Message}");
            return null;
        }
    }


    /// <summary>
    /// Wire id of a reward: "gold:N" (no space) for gold, otherwise the
    /// reward's runtime type name (CardReward, PotionReward, RelicReward, ...).
    /// The Python driver matches these literally.
    /// </summary>
    private static string RewardLabel(NRewardButton button) => button.Reward switch
    {
        null => UnknownRewardLabel,
        GoldReward gold => GoldLabelPrefix + gold.Amount,
        Reward reward => reward.GetType().Name,
    };

    private static string RewardDescription(NRewardButton button)
    {
        try
        {
            return button.Reward?.Description.GetFormattedText() ?? "";
        }
        catch (Exception)
        {
            return "";
        }
    }
}
