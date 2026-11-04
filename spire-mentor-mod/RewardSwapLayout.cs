// RewardSwapLayout.cs -- the option index layout of a reward_screen ask (GH #1069).
//
// A potion reward that finds a full belt is not lost in the real game: the
// player can discard a held potion from the belt (NPotionPopup ->
// DiscardPotionGameAction -> PotionCmd.Discard) and then claim it, or leave
// it on the screen. The sim engines surface that as a `potion_swap` decision
// (decline, or discard slot k). The reward screen offers the same choice: one
// decline option plus one option per belt slot for every potion reward that
// only a full belt blocks.
//
// Wire order, one entry per option index: every claimable reward, then each
// blocked potion's group (decline first, then slot 0..n-1, the order the
// engines list them in), then the trailing proceed. Game-free so tests/unit
// can link it.

using System.Collections.Generic;

namespace SpireMentor;

public enum RewardOptionKind
{
    /// <summary>Click a claimable reward button.</summary>
    Pick,

    /// <summary>Leave a full-belt potion reward on the screen.</summary>
    SwapDecline,

    /// <summary>Discard the potion in <see cref="RewardOptionSlot.DiscardSlot"/>, then claim the reward.</summary>
    Swap,

    /// <summary>Click proceed and leave the screen.</summary>
    Proceed,
}

/// <param name="Kind">What the option does.</param>
/// <param name="Button">Index into the claimable list (Pick) or the blocked-potion list (SwapDecline/Swap); -1 for Proceed.</param>
/// <param name="DiscardSlot">Belt slot to discard (Swap only); -1 otherwise.</param>
public readonly record struct RewardOptionSlot(RewardOptionKind Kind, int Button, int DiscardSlot);

public static class RewardSwapLayout
{
    /// <summary>
    /// The options of one ask, in wire index order. <paramref name="beltSizes"/>
    /// holds, per blocked potion reward, how many belt slots it could replace.
    /// </summary>
    public static List<RewardOptionSlot> Build(int pickCount, IReadOnlyList<int> beltSizes, bool proceed)
    {
        var slots = new List<RewardOptionSlot>();
        for (int i = 0; i < pickCount; i++)
            slots.Add(new RewardOptionSlot(RewardOptionKind.Pick, i, -1));
        for (int b = 0; b < beltSizes.Count; b++)
        {
            slots.Add(new RewardOptionSlot(RewardOptionKind.SwapDecline, b, -1));
            for (int s = 0; s < beltSizes[b]; s++)
                slots.Add(new RewardOptionSlot(RewardOptionKind.Swap, b, s));
        }
        if (proceed)
            slots.Add(new RewardOptionSlot(RewardOptionKind.Proceed, -1, -1));
        return slots;
    }
}
