using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;

namespace Sts2Headless;

/// <summary>
/// Combat state-mutation trace for the act23 parity driver (GH #1088).
///
/// When on, Harmony patches on the game's lowest-level writers record every
/// change to creature HP / max HP / block, power amounts, combat piles,
/// energy / stars and the named RNG streams. The writers are private property
/// setters and <c>*Internal</c> methods, so no write can bypass them. Each
/// event carries <c>src</c>: the chain of model methods (hook overrides, card
/// <c>OnPlay</c>, monster moves) active when it happened, e.g.
/// <c>Hellraiser.AfterCardDrawnEarly&gt;StrikeIronclad.OnPlay</c>.
///
/// Events are grouped in segments, one per <c>action</c> command, numbered by
/// <see cref="Seq"/>; the last <see cref="MaxSegments"/> are kept. The driver
/// remembers <c>get_state</c>'s <c>trace_seq</c> at each clean full-state check,
/// and on a split asks <c>get_trace</c> for the segments after it.
///
/// Attribution is approximate where model methods truly suspend (a frame stays
/// on the stack until its Task completes), but headless awaits complete
/// synchronously almost everywhere, so frames nest like a call stack.
/// Patches are applied on the first <c>trace on</c> only, so untraced runs pay
/// nothing.
/// </summary>
internal static class StateTrace
{
    public const int MaxSegments = 16;
    public const int MaxEventsPerSegment = 4000;

    private static readonly object Lock = new();
    private static readonly LinkedList<Dictionary<string, object?>> Segments = new();
    private static List<Dictionary<string, object?>>? _current;
    private static readonly List<(int Id, string Label)> Frames = new();
    private static int _nextFrameId;
    private static bool _patched;

    public static bool Enabled { get; private set; }
    public static int Seq { get; private set; }

    /// <summary>Maps an <see cref="Rng"/> instance to its stream name (the
    /// <c>rng_counters</c> key), or null for an unnamed stream. Set by
    /// <see cref="RunSimulator"/>, refreshed at each action.</summary>
    public static Func<Dictionary<Rng, string>>? RngStreams;
    private static Dictionary<Rng, string> _rngNames = new(ReferenceEqualityComparer.Instance);
    private static readonly Dictionary<string, int> LastRngCounter = new();

    public static Dictionary<string, object?> SetEnabled(bool on)
    {
        string? patchInfo = null;
        if (on && !_patched)
        {
            patchInfo = ApplyPatches();
            _patched = true;
        }
        lock (Lock)
        {
            Enabled = on;
            Segments.Clear();
            _current = null;
            Frames.Clear();
        }
        return new Dictionary<string, object?>
        {
            ["type"] = "ok",
            ["enabled"] = on,
            ["seq"] = Seq,
            ["patches"] = patchInfo,
        };
    }

    /// <summary>Open a new segment for one <c>action</c> command.</summary>
    public static void BeginAction(string action, Dictionary<string, object?>? args)
    {
        if (!Enabled) return;
        var names = RngStreams?.Invoke();
        lock (Lock)
        {
            Seq++;
            _current = new List<Dictionary<string, object?>>();
            Segments.AddLast(new Dictionary<string, object?>
            {
                ["seq"] = Seq,
                ["action"] = action,
                ["args"] = args == null ? "" : string.Join(",", args.Select(kv => $"{kv.Key}={kv.Value}")),
                ["events"] = _current,
            });
            while (Segments.Count > MaxSegments) Segments.RemoveFirst();
            Frames.Clear();
            if (names != null) _rngNames = names;
        }
    }

    public static Dictionary<string, object?> GetTrace(int since)
    {
        lock (Lock)
        {
            return new Dictionary<string, object?>
            {
                ["type"] = "ok",
                ["seq"] = Seq,
                ["segments"] = Segments.Where(s => (int)s["seq"]! > since).ToList(),
            };
        }
    }

    private static string Src()
    {
        // Innermost three frames, outermost first.
        int n = Frames.Count;
        if (n == 0) return "";
        int from = Math.Max(0, n - 3);
        return string.Join(">", Frames.Skip(from).Select(f => f.Label));
    }

    private static void Record(Dictionary<string, object?> ev)
    {
        lock (Lock)
        {
            if (_current == null || _current.Count >= MaxEventsPerSegment) return;
            ev["src"] = Src();
            _current.Add(ev);
        }
    }

    internal static string Who(Creature c)
    {
        if (c.IsMonster)
        {
            var enemies = CombatManager.Instance?.DebugOnlyGetState()?.Enemies;
            if (enemies != null)
                for (int i = 0; i < enemies.Count; i++)
                    if (ReferenceEquals(enemies[i], c)) return $"monsters[{i}]";
            return $"monster:{c.Monster?.Id.Entry}";
        }
        if (c.IsPet) return "osty";
        return "player";
    }

    private static void Change(string who, string field, int old, int now, string? id = null)
    {
        if (old == now) return;
        var ev = new Dictionary<string, object?> { ["k"] = field, ["who"] = who, ["old"] = old, ["new"] = now };
        if (id != null) ev["id"] = id;
        Record(ev);
    }

    // ---------------------------------------------------------------- patches

    private static string ApplyPatches()
    {
        var h = new Harmony("sts2headless.statetrace");
        var p = typeof(StateTrace);
        int ok = 0, failed = 0;
        void Patch(MethodBase? target, string? prefix, string? postfix)
        {
            if (target == null) { failed++; return; }
            try
            {
                h.Patch(
                    target,
                    prefix: prefix == null ? null : new HarmonyMethod(p.GetMethod(prefix, BindingFlags.NonPublic | BindingFlags.Static)),
                    postfix: postfix == null ? null : new HarmonyMethod(p.GetMethod(postfix, BindingFlags.NonPublic | BindingFlags.Static)));
                ok++;
            }
            catch (Exception ex)
            {
                failed++;
                Console.Error.WriteLine($"[WARN] state trace: patch {target.DeclaringType?.Name}.{target.Name}: {ex.Message}");
            }
        }

        var creature = typeof(Creature);
        Patch(AccessTools.PropertySetter(creature, "CurrentHp"), nameof(HpPre), nameof(HpPost));
        Patch(AccessTools.PropertySetter(creature, "MaxHp"), nameof(MaxHpPre), nameof(MaxHpPost));
        Patch(AccessTools.PropertySetter(creature, "Block"), nameof(BlockPre), nameof(BlockPost));
        var pcs = typeof(PlayerCombatState);
        Patch(AccessTools.PropertySetter(pcs, "Energy"), nameof(EnergyPre), nameof(EnergyPost));
        Patch(AccessTools.PropertySetter(pcs, "Stars"), nameof(StarsPre), nameof(StarsPost));
        var power = typeof(PowerModel);
        Patch(AccessTools.Method(power, "SetAmount"), nameof(PowerAmountPre), nameof(PowerAmountPost));
        Patch(AccessTools.Method(power, "RemoveInternal"), nameof(PowerRemovePre), null);
        var pile = typeof(CardPile);
        Patch(AccessTools.Method(pile, "AddInternal"), null, nameof(PileAddPost));
        Patch(AccessTools.Method(pile, "RemoveInternal"), nameof(PileRemovePre), null);
        Patch(AccessTools.Method(pile, "Clear"), nameof(PileClearPre), null);
        var rng = typeof(Rng);
        foreach (var sig in new[]
                 {
                     ("NextInt", new[] { typeof(int) }), ("NextInt", new[] { typeof(int), typeof(int) }),
                     ("NextBool", Type.EmptyTypes), ("NextDouble", Type.EmptyTypes),
                     ("NextDouble", new[] { typeof(double), typeof(double) }),
                     ("NextFloat", new[] { typeof(float) }), ("NextFloat", new[] { typeof(float), typeof(float) }),
                 })
            Patch(rng.GetMethod(sig.Item1, sig.Item2), null, nameof(RngPost));
        int core = ok;

        // Attribution frames: every Task-returning hook override on a model
        // (AfterCardDrawnEarly, OnPlay, AfterTurnEnd, ...) plus every
        // Task-returning method a monster declares (its move bodies).
        int frames = 0;
        foreach (var t in AbstractModelSubtypes.All)
        {
            bool monster = typeof(MonsterModel).IsAssignableFrom(t);
            foreach (var m in t.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
            {
                if (m.IsAbstract || m.ContainsGenericParameters || m.Name.Contains('<')) continue;
                if (!typeof(Task).IsAssignableFrom(m.ReturnType)) continue;
                bool isOverride = m.GetBaseDefinition().DeclaringType != t;
                if (!isOverride && !monster) continue;
                Patch(m, nameof(FramePre), nameof(FramePost));
                frames++;
            }
        }
        string info = $"core={core} frames={frames} ok={ok} failed={failed}";
        Console.Error.WriteLine($"[INFO] state trace patches: {info}");
        return info;
    }

    private static void HpPre(Creature __instance, out int __state) => __state = __instance.CurrentHp;
    private static void HpPost(Creature __instance, int __state) { if (Enabled) Change(Who(__instance), "hp", __state, __instance.CurrentHp); }
    private static void MaxHpPre(Creature __instance, out int __state) => __state = __instance.MaxHp;
    private static void MaxHpPost(Creature __instance, int __state) { if (Enabled) Change(Who(__instance), "max_hp", __state, __instance.MaxHp); }
    private static void BlockPre(Creature __instance, out int __state) => __state = __instance.Block;
    private static void BlockPost(Creature __instance, int __state) { if (Enabled) Change(Who(__instance), "block", __state, __instance.Block); }
    private static void EnergyPre(PlayerCombatState __instance, out int __state) => __state = __instance.Energy;
    private static void EnergyPost(PlayerCombatState __instance, int __state) { if (Enabled) Change("player", "energy", __state, __instance.Energy); }
    private static void StarsPre(PlayerCombatState __instance, out int __state) => __state = __instance.Stars;
    private static void StarsPost(PlayerCombatState __instance, int __state) { if (Enabled) Change("player", "stars", __state, __instance.Stars); }

    // `Owner`'s getter asserts the model is mutable, so read the field.
    private static readonly AccessTools.FieldRef<PowerModel, Creature?> PowerOwner =
        AccessTools.FieldRefAccess<PowerModel, Creature?>("_owner");
    private static readonly AccessTools.FieldRef<PowerModel, int> PowerAmount =
        AccessTools.FieldRefAccess<PowerModel, int>("_amount");

    private static void PowerAmountPre(PowerModel __instance, out int __state) => __state = PowerAmount(__instance);
    private static void PowerAmountPost(PowerModel __instance, int __state)
    {
        if (!Enabled) return;
        var owner = PowerOwner(__instance);
        if (owner == null) return;
        Change(Who(owner), "power", __state, PowerAmount(__instance), __instance.Id.Entry);
    }

    private static void PowerRemovePre(PowerModel __instance)
    {
        if (!Enabled) return;
        var owner = PowerOwner(__instance);
        if (owner == null) return;
        Record(new Dictionary<string, object?>
        {
            ["k"] = "power_remove", ["who"] = Who(owner), ["id"] = __instance.Id.Entry, ["old"] = PowerAmount(__instance),
        });
    }

    private static int IndexIn(CardPile pile, CardModel card)
    {
        int i = 0;
        foreach (var c in pile.Cards)
        {
            if (ReferenceEquals(c, card)) return i;
            i++;
        }
        return -1;
    }

    private static void PileAddPost(CardPile __instance, CardModel card)
    {
        if (!Enabled) return;
        Record(new Dictionary<string, object?>
        {
            ["k"] = "pile_add", ["pile"] = __instance.Type.ToString(), ["id"] = card.Id.ToString(), ["idx"] = IndexIn(__instance, card),
        });
    }

    private static void PileRemovePre(CardPile __instance, CardModel card)
    {
        if (!Enabled) return;
        Record(new Dictionary<string, object?>
        {
            ["k"] = "pile_remove", ["pile"] = __instance.Type.ToString(), ["id"] = card.Id.ToString(), ["idx"] = IndexIn(__instance, card),
        });
    }

    private static void PileClearPre(CardPile __instance)
    {
        if (!Enabled || __instance.Cards.Count == 0) return;
        Record(new Dictionary<string, object?>
        {
            ["k"] = "pile_clear", ["pile"] = __instance.Type.ToString(), ["old"] = __instance.Cards.Count,
        });
    }

    private static void RngPost(Rng __instance)
    {
        if (!Enabled) return;
        string? name;
        lock (Lock)
        {
            if (!_rngNames.TryGetValue(__instance, out name)) return;
        }
#if GAME_V0111
        int ctr = RngCompat.Counter(__instance);
#else
        int ctr = __instance.Counter;
#endif
        // Nested draws (NextInt(a, b) -> NextInt(n)) report the same cursor
        // twice; keep one event per cursor value.
        lock (Lock)
        {
            if (LastRngCounter.TryGetValue(name, out var last) && last == ctr) return;
            LastRngCounter[name] = ctr;
        }
        Record(new Dictionary<string, object?> { ["k"] = "rng", ["who"] = name, ["new"] = ctr });
    }

    private static void FramePre(MethodBase __originalMethod, out int __state)
    {
        __state = -1;
        if (!Enabled) return;
        lock (Lock)
        {
            __state = ++_nextFrameId;
            Frames.Add((__state, $"{__originalMethod.DeclaringType?.Name}.{__originalMethod.Name}"));
        }
    }

    private static void FramePost(object __result, int __state)
    {
        if (__state < 0) return;
        int id = __state;
        if (__result is Task task && !task.IsCompleted)
        {
            task.ContinueWith(_ => PopFrame(id), TaskContinuationOptions.ExecuteSynchronously);
            return;
        }
        PopFrame(id);
    }

    private static void PopFrame(int id)
    {
        lock (Lock)
        {
            for (int i = Frames.Count - 1; i >= 0; i--)
                if (Frames[i].Id == id)
                {
                    Frames.RemoveAt(i);
                    return;
                }
        }
    }
}
