// SpeedSteps.cs -- the speed hotkeys' steps and autobattle's speed rule.
//
// F9 / F10 and the toolbar's -/+ step in whole speeds (1x, 2x, 3x, 4x):
// tenths were too fine to be worth a key press. The settings slider and the
// `spirementor speed` console command still set any speed SimSpeed allows.
//
// Autobattle (Autobattle.cs) at 1x means watching every fight at normal
// speed, so switching it on raises 1x to 2x. Any other speed is the player's
// own choice and is left alone; switching autobattle off puts 1x back only
// if the speed is still the 2x autobattle set.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it;
// SimSpeed applies the result.

using System;

namespace SpireMentor;

internal static class SpeedSteps
{
    /// <summary>The slowest and fastest whole step (SimSpeed.Max is 4).</summary>
    public const double MinStep = 1.0;
    public const double MaxStep = 4.0;

    /// <summary>The speed autobattle raises 1x to.</summary>
    public const double Autobattle = 2.0;

    private const double Eps = 1e-6;

    /// <summary>
    /// The speed one step up (<paramref name="direction"/> &gt; 0) or down
    /// from <paramref name="current"/>: the next whole speed in that
    /// direction, within 1x..4x. A speed between steps (1.5x from the
    /// slider) goes to the nearest one in that direction (2x up, 1x down). A
    /// speed below 1x is never sped up by "down"; "up" takes it to 1x.
    /// </summary>
    public static double Step(double current, int direction)
    {
        if (direction > 0)
        {
            return Math.Min(MaxStep, Math.Max(MinStep, Math.Floor(current + Eps) + 1.0));
        }
        if (current <= MinStep + Eps)
        {
            return current;
        }
        return Math.Max(MinStep, Math.Min(MaxStep, Math.Ceiling(current - Eps) - 1.0));
    }

    /// <summary>The speed to switch to as autobattle turns on, or null to
    /// leave <paramref name="current"/> alone (anything but 1x).</summary>
    public static double? AutobattleOn(double current) =>
        Math.Abs(current - 1.0) < Eps ? Autobattle : null;

    /// <summary>As autobattle turns off, having raised the speed: whether to
    /// put 1x back, i.e. the player has not changed it since.</summary>
    public static bool RestoreOnOff(double current) => Math.Abs(current - Autobattle) < Eps;

    /// <summary>The speed as the toolbar shows it: "2x", "1.5x".</summary>
    public static string Label(double speed) =>
        speed.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) + "x";
}
