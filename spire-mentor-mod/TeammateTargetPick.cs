// TeammateTargetPick.cs -- which teammate an AnyAlly card or potion goes to.
//
// With three or four players the advisor names the teammate (GH #1017): a
// play's `ally_index` is an index into the board's `allies[]`, which lists
// every other player in the run's order (SimCombatHandler.SerializeAllies),
// dead ones included, and never drops one. The engine's
// `Action::PlayCardAtAlly` carries the same index. Without one (-1) the first
// living teammate, the only one there is with two players.
//
// The game takes only a living teammate (CardModel.IsValidTarget,
// NTargetManager), so a named dead one picks nobody and the play is refused,
// unless exactly one teammate is alive: they are the only possible target
// whatever was named, so two-player clients keep working as before.
//
// Game-free so tests/unit can compile it.

using System.Collections.Generic;

namespace SpireMentor;

internal static class TeammateTargetPick
{
    /// <summary>
    /// The index into <paramref name="players"/> (the run's players in order,
    /// each flagged as us and as alive) of the teammate
    /// <paramref name="allyIndex"/> names, counting only the others, as
    /// <c>allies[]</c> does; the first living teammate for a negative index,
    /// or whenever only one is alive. -1 when that is nobody.
    /// </summary>
    internal static int Pick(IReadOnlyList<(bool IsMe, bool IsAlive)> players, int allyIndex)
    {
        int first = -1, living = 0, named = -1, ally = 0;
        for (int i = 0; i < players.Count; i++)
        {
            if (players[i].IsMe)
                continue;
            if (players[i].IsAlive)
            {
                living++;
                if (first < 0)
                    first = i;
            }
            if (ally == allyIndex)
                named = i;
            ally++;
        }
        if (allyIndex < 0 || living == 1)
            return first;
        return named >= 0 && players[named].IsAlive ? named : -1;
    }

    /// <summary>
    /// The reverse of <see cref="Pick"/>: the <c>allies[]</c> index of the
    /// player at <paramref name="player"/> in the run's order, as seen by the
    /// player flagged in <paramref name="isMe"/>. -1 for us or out of range.
    /// What a human's AnyAlly play reports as its <c>ally_index</c>, so the
    /// advisor's mirror aims it where the human did.
    /// </summary>
    internal static int AllyIndex(IReadOnlyList<bool> isMe, int player)
    {
        if (player < 0 || player >= isMe.Count || isMe[player])
            return -1;
        int ally = 0;
        for (int i = 0; i < player; i++)
        {
            if (!isMe[i])
                ally++;
        }
        return ally;
    }
}
