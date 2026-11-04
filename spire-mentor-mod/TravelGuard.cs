// TravelGuard.cs -- whether a map move is still in flight, so no screen is
// described or clicked mid-travel (GH #987).
//
// The game's own flag (NMapScreen.IsTraveling) is set when TravelToMapCoord
// starts and cleared only after the next room is entered, so it covers the
// whole window in which the room being left is still on screen. The one risk
// of trusting it is a flag that never clears (a travel that threw before
// RunManager reset it): hiding every decision would stall autoplay for good.
// So a flag held through one move for longer than the limit is ignored for
// the rest of that move -- and only that move: the next move (a new visited
// count) re-arms the guard even if the flag never dropped in between.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

namespace SpireMentor;

internal sealed class TravelGuard
{
    /// <summary>How long one move may hold the flag before it counts as
    /// stuck. Generous: a co-op peer's sync and a cold-disk room preload are
    /// both inside the window, and giving up early reopens the race.</summary>
    public const ulong StuckAfterMs = 120_000;

    private ulong? _since;
    private int _move = int.MinValue;
    private bool _stuck;

    /// <summary>True when this move's flag has just been declared stuck;
    /// cleared by the caller once it has logged it.</summary>
    public bool JustGaveUp { get; private set; }

    /// <summary>
    /// Should describing be suppressed? <paramref name="traveling"/> is the
    /// game's flag, <paramref name="move"/> identifies the move (the visited
    /// map-coord count), <paramref name="nowMs"/> a monotonic clock.
    /// </summary>
    public bool Suppress(bool traveling, int move, ulong nowMs)
    {
        JustGaveUp = false;
        if (!traveling)
        {
            _since = null;
            _stuck = false;
            return false;
        }
        if (move != _move)
        {
            _move = move;
            _since = nowMs;
            _stuck = false;
        }
        _since ??= nowMs;
        if (_stuck)
        {
            return false;
        }
        if (nowMs - _since.Value < StuckAfterMs)
        {
            return true;
        }
        _stuck = true;
        JustGaveUp = true;
        return false;
    }
}
