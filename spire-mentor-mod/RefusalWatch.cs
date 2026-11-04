// RefusalWatch.cs -- what the combat lane's refused-action watchdog concludes
// when it fires (SimAdvisor.WatchForRefusedAction).
//
// A potion or card that asks the player to pick something (Skill Potion,
// Colorless Potion, ...) leaves no combat-history entry until the pick lands,
// and the pick goes through the screen lane, which can take longer than the
// watchdog's 5 s. Read as "changed nothing on the board", that counted as a
// refusal and re-asked the advisor while the pick was still on its way: on
// 2026-10-28 an A10 win logged two such refusals and answered each pick twice.
// A decision screen that is up, or came up since the action, means the action
// was taken and is waiting on that pick -- look again later instead.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

namespace SpireMentor;

internal enum RefusalVerdict
{
    /// <summary>The board moved: the action resolved.</summary>
    Accepted,
    /// <summary>A pick the action asked for is open or just closed; look again later.</summary>
    WaitForPick,
    /// <summary>Nothing moved and nothing is pending: the game dropped it.</summary>
    Refused,
}

internal static class RefusalWatch
{
    public static RefusalVerdict Judge(
        int markAtAction, int markNow, bool decisionScreenUp, int screensAtAction, int screensNow)
    {
        if (markNow != markAtAction)
        {
            return RefusalVerdict.Accepted;
        }
        if (decisionScreenUp || screensNow != screensAtAction)
        {
            return RefusalVerdict.WaitForPick;
        }
        return RefusalVerdict.Refused;
    }
}
