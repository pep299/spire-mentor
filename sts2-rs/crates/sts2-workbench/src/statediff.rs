//! Human-readable diff between two consecutive game states.
//!
//! Port of `sts2-pycli/python/workbench/statediff.py`.
//!
//! The workbench event log (Play and Replay) shows what an action *did* —
//! damage dealt/taken, block, buffs, deaths, and resource changes — by
//! comparing the state before an action with the state after it. This module
//! is the single place those effect lines are derived. Functions are pure and
//! defensive: a missing key skips a line rather than raising.

use serde_json::Value;

use crate::render::py::{self, Num};

/// Real minus glyph, for aligned signs.
pub const MINUS: &str = "−";
const DEAD: &str = "☠ defeated";

/// A Python `dict` keyed by JSON values under Python `==` (so `1`, `1.0` and
/// `True` share a slot): insertion-ordered, an update keeps the key's
/// original position. Also serves as `collections.Counter`.
#[derive(Debug)]
struct PyDict<'a, V> {
    entries: Vec<(&'a Value, V)>,
}

impl<'a, V> PyDict<'a, V> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn position(&self, key: &Value) -> Option<usize> {
        self.entries.iter().position(|(k, _)| py::eq(k, key))
    }

    fn get(&self, key: &Value) -> Option<&V> {
        self.position(key).map(|i| &self.entries[i].1)
    }

    fn contains(&self, key: &Value) -> bool {
        self.position(key).is_some()
    }

    fn insert(&mut self, key: &'a Value, value: V) {
        match self.position(key) {
            Some(i) => self.entries[i].1 = value,
            None => self.entries.push((key, value)),
        }
    }

    fn iter(&self) -> impl Iterator<Item = (&'a Value, &V)> {
        self.entries.iter().map(|(k, v)| (*k, v))
    }
}

impl<'a> PyDict<'a, i64> {
    /// `Counter(keys)`.
    fn counter(keys: impl Iterator<Item = &'a Value>) -> Self {
        let mut out = Self::new();
        for k in keys {
            let n = out.get(k).copied().unwrap_or(0);
            out.insert(k, n + 1);
        }
        out
    }

    /// `Counter.__sub__`: keeps only positive counts, in `self`'s key order.
    fn sub(&self, other: &Self) -> Vec<(&'a Value, i64)> {
        self.iter()
            .filter_map(|(k, n)| {
                let d = n - other.get(k).copied().unwrap_or(0);
                (d > 0).then_some((k, d))
            })
            .collect()
    }
}

/// `f"({sign}{abs(delta)})"`: signed magnitude like `(-6)` / `(+16)` using a
/// real minus glyph.
fn signed(delta: Num) -> String {
    let sign = if delta.is_negative() { MINUS } else { "+" };
    format!("({sign}{})", delta.abs().to_str())
}

/// `x.get("index")` — the raw slot, `Null` when absent (Python's `None`
/// key).
fn index_of(x: &Value) -> &Value {
    static NULL: Value = Value::Null;
    py::get_raw(x, "index").unwrap_or(&NULL)
}

/// The display name of `enemies[pos]`: its board label — the name plus
/// `" #k"` when several enemies in `enemies` share it
/// ([`crate::enemy_labels`]) — or `fallback` when it carries no `name`.
///
/// A diff spans two states, so each enemy is labelled from ONE of them: an
/// enemy present before the action by its label in the BEFORE state (the
/// board the player acted on — so a kill still names the rat that was hit,
/// even though the after-board's numbers close up), a newly spawned one by
/// its label in the AFTER state (the only board it is on).
///
/// `labels` is `enemy_labels(enemies)`, computed once per state by the caller
/// rather than once per enemy.
fn enemy_name_at(
    enemies: &[Value],
    labels: &[String],
    pos: usize,
    fallback: impl FnOnce() -> String,
) -> String {
    if py::get_raw(&enemies[pos], "name").is_none() {
        return fallback();
    }
    labels[pos].clone()
}

/// `enemy_name_at` with `e.get("name", f"enemy {idx}")`'s fallback.
fn enemy_name(enemies: &[Value], labels: &[String], pos: usize) -> String {
    enemy_name_at(enemies, labels, pos, || {
        format!("enemy {}", py::to_str(index_of(&enemies[pos])))
    })
}

/// `d.get(k, 0)`.
fn get_or_zero<'a>(d: &'a Value, k: &str, zero: &'a Value) -> &'a Value {
    py::get_or(d, k, zero)
}

fn player_hp_block(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    let p0 = py::get_obj(prev, "player");
    let p1 = py::get_obj(nxt, "player");
    let (h0, h1) = (py::get(p0, "hp"), py::get(p1, "hp"));
    if let (Some(h0), Some(h1)) = (h0, h1) {
        if let (Some(i0), Some(i1)) = (py::as_int(h0), py::as_int(h1)) {
            if i0 != i1 {
                let delta = Num::Int(i1).sub(Num::Int(i0));
                out.push(format!(
                    "You HP {}→{} {}",
                    py::to_str(h0),
                    py::to_str(h1),
                    signed(delta)
                ));
            }
        }
    }
    let zero = Value::from(0);
    let (b0, b1) = (
        get_or_zero(p0, "block", &zero),
        get_or_zero(p1, "block", &zero),
    );
    if !py::eq(b0, b1) {
        out.push(format!("You Block {}→{}", py::to_str(b0), py::to_str(b1)));
    }
}

fn enemy_lines(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    let zero = Value::from(0);
    let mut nxt_by_idx = PyDict::new();
    for e in py::get_list(nxt, "enemies") {
        nxt_by_idx.insert(index_of(e), e);
    }
    let prev_es = py::get_list(prev, "enemies");
    let prev_labels = crate::enemy_labels::enemy_labels(prev_es);
    for (pos, e0) in prev_es.iter().enumerate() {
        let name = enemy_name(prev_es, &prev_labels, pos);
        let h0 = get_or_zero(e0, "hp", &zero);
        let Some(e1) = nxt_by_idx.get(index_of(e0)) else {
            // A live enemy vanishing = defeated; an already-0 corpse being
            // cleaned up is silent.
            if py::num(h0).is_some_and(Num::is_positive) {
                out.push(format!("{name} HP {}→0  {DEAD}", py::to_str(h0)));
            }
            continue;
        };
        let h1 = get_or_zero(e1, "hp", &zero);
        if !py::eq(h0, h1) {
            let mut line = match (py::num(h0), py::num(h1)) {
                (Some(n0), Some(n1)) => format!(
                    "{name} HP {}→{} {}",
                    py::to_str(h0),
                    py::to_str(h1),
                    signed(n1.sub(n0))
                ),
                // Python would raise on non-numeric HP; keep the line readable.
                _ => format!("{name} HP {}→{}", py::to_str(h0), py::to_str(h1)),
            };
            if py::num(h1).is_some_and(|n| !n.is_positive()) {
                line.push_str(&format!("  {DEAD}"));
            }
            out.push(line);
        }
        let (b0, b1) = (
            get_or_zero(e0, "block", &zero),
            get_or_zero(e1, "block", &zero),
        );
        if !py::eq(b0, b1) {
            out.push(format!(
                "{name} Block {}→{}",
                py::to_str(b0),
                py::to_str(b1)
            ));
        }
    }
}

/// `{p.get("name"): p.get("amount", 0) for p in powers or []}`.
fn power_map<'a>(powers: Option<&'a Value>, zero: &'a Value) -> PyDict<'a, &'a Value> {
    static NULL: Value = Value::Null;
    let mut out = PyDict::new();
    for p in py::list(powers) {
        let name = py::get_raw(p, "name").unwrap_or(&NULL);
        out.insert(name, get_or_zero(p, "amount", zero));
    }
    out
}

fn diff_powers(
    who: &str,
    prev_powers: Option<&Value>,
    nxt_powers: Option<&Value>,
    out: &mut Vec<String>,
) {
    let zero = Value::from(0);
    let p0 = power_map(prev_powers, &zero);
    let p1 = power_map(nxt_powers, &zero);
    for (name, amt) in p1.iter() {
        let name_s = py::to_str(name);
        match p0.get(name) {
            None => {
                if py::truthy(amt) {
                    // Render the magnitude's sign properly so a debuff gained at
                    // a negative amount reads "You −1 Shrink", not "You +-1
                    // Shrink".
                    match py::num(amt) {
                        Some(n) => {
                            let sign = if n.is_positive() { "+" } else { MINUS };
                            out.push(format!("{who} {sign}{} {name_s}", n.abs().to_str()));
                        }
                        // Python would raise on a non-numeric amount.
                        None => out.push(format!("{who} +{} {name_s}", py::to_str(amt))),
                    }
                } else {
                    out.push(format!("{who} +{name_s}"));
                }
            }
            Some(old) if !py::eq(amt, old) => {
                out.push(format!(
                    "{who} {name_s} {}→{}",
                    py::to_str(old),
                    py::to_str(amt)
                ));
            }
            Some(_) => {}
        }
    }
    for (name, _) in p0.iter() {
        if !p1.contains(name) {
            out.push(format!("{who} {MINUS}{}", py::to_str(name)));
        }
    }
}

fn power_lines(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    diff_powers(
        "You",
        py::get(prev, "player_powers"),
        py::get(nxt, "player_powers"),
        out,
    );
    let prev_list = py::get_list(prev, "enemies");
    let prev_labels = crate::enemy_labels::enemy_labels(prev_list);
    let mut prev_es = PyDict::new();
    for (pos, e) in prev_list.iter().enumerate() {
        prev_es.insert(index_of(e), (pos, e));
    }
    for e1 in py::get_list(nxt, "enemies") {
        let Some(&(pos, e0)) = prev_es.get(index_of(e1)) else {
            continue;
        };
        // Labelled from the BEFORE state, like the HP lines (`enemy_name_at`).
        let name = enemy_name(prev_list, &prev_labels, pos);
        diff_powers(&name, py::get(e0, "powers"), py::get(e1, "powers"), out);
    }
}

fn spawn_lines(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    let mut prev_idx = PyDict::new();
    for e in py::get_list(prev, "enemies") {
        prev_idx.insert(index_of(e), ());
    }
    let nxt_es = py::get_list(nxt, "enemies");
    let nxt_labels = crate::enemy_labels::enemy_labels(nxt_es);
    for (pos, e) in nxt_es.iter().enumerate() {
        if !prev_idx.contains(index_of(e)) {
            let name = enemy_name_at(nxt_es, &nxt_labels, pos, || "enemy".to_string());
            out.push(format!("Spawned: {name}"));
        }
    }
}

/// `(pt.get("name") for pt in player.get("potions") or [])`.
fn potion_names(player: &Value) -> impl Iterator<Item = &Value> {
    static NULL: Value = Value::Null;
    py::get_list(player, "potions")
        .iter()
        .map(|pt| py::get_raw(pt, "name").unwrap_or(&NULL))
}

fn resource_lines(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    static NULL: Value = Value::Null;
    let p0 = py::get_obj(prev, "player");
    let p1 = py::get_obj(nxt, "player");
    if let (Some(g0), Some(g1)) = (py::get(p0, "gold"), py::get(p1, "gold")) {
        if let (Some(i0), Some(i1)) = (py::as_int(g0), py::as_int(g1)) {
            if i0 != i1 {
                out.push(format!("Gold {}→{}", py::to_str(g0), py::to_str(g1)));
            }
        }
    }
    let mut r0 = PyDict::new();
    for r in py::get_list(p0, "relics") {
        r0.insert(py::get_raw(r, "name").unwrap_or(&NULL), ());
    }
    for r in py::get_list(p1, "relics") {
        let name = py::get_raw(r, "name").unwrap_or(&NULL);
        if !r0.contains(name) {
            out.push(format!("+Relic {}", py::to_str(name)));
        }
    }
    let pot0 = PyDict::counter(potion_names(p0));
    let pot1 = PyDict::counter(potion_names(p1));
    for (name, n) in pot1.sub(&pot0) {
        for _ in 0..n {
            out.push(format!("+Potion {}", py::to_str(name)));
        }
    }
    for (name, n) in pot0.sub(&pot1) {
        for _ in 0..n {
            out.push(format!("Used {}", py::to_str(name)));
        }
    }
}

/// Attribute end-of-turn effects to the Defect's orb passives.
///
/// Orb passives fire when the player ends their turn; a `round` increment
/// marks that boundary. The net HP/block diff above bundles enemy-turn effects
/// with the orb passives, so this annotates *what each orb did* (using the
/// pre-turn-end queue) to make the numbers traceable. Mechanics follow the C#
/// decomp (`…/Models.Orbs`): Lightning hits one random enemy, Glass hits all,
/// Frost grants block, Plasma grants energy, Dark stores its evoke value.
fn orb_passive_lines(prev: &Value, nxt: &Value, out: &mut Vec<String>) {
    let (Some(r0), Some(r1)) = (
        py::get(prev, "round").and_then(py::as_int),
        py::get(nxt, "round").and_then(py::as_int),
    ) else {
        return;
    };
    if r1 <= r0 {
        return;
    }
    let orbs = py::get_list(prev, "orbs");
    if orbs.is_empty() {
        return;
    }
    let zero = Value::from(0);
    let mut descs: Vec<String> = Vec::new();
    for o in orbs {
        let kind = py::get(o, "type")
            .filter(|t| py::truthy(t))
            .map_or_else(String::new, |t| py::to_str(t).to_lowercase());
        let passive = py::to_str(get_or_zero(o, "passive", &zero));
        descs.push(match kind.as_str() {
            "lightning" => format!("Lightning orb passive: {passive} damage to a random enemy"),
            "frost" => format!("Frost orb passive: +{passive} block"),
            "dark" => format!("Dark orb passive: +{passive} stored damage"),
            "plasma" => format!("Plasma orb passive: +{passive} energy"),
            "glass" => format!("Glass orb passive: {passive} damage to all enemies"),
            _ => format!(
                "{} orb passive",
                py::get_raw(o, "type").map_or_else(|| "Orb".to_string(), py::to_str)
            ),
        });
    }
    // Collapse identical lines (e.g. two Lightning orbs) into a "(×N)" suffix.
    let mut counted: Vec<(&str, usize)> = Vec::new();
    for d in &descs {
        match counted.iter_mut().find(|(k, _)| *k == d) {
            Some((_, n)) => *n += 1,
            None => counted.push((d, 1)),
        }
    }
    for (desc, n) in counted {
        out.push(if n == 1 {
            desc.to_string()
        } else {
            format!("{desc} (×{n})")
        });
    }
}

/// Return ordered, human-readable effect lines for the `prev`→`nxt` transition.
///
/// * `prev`: the state an action was taken from.
/// * `nxt`: the state resulting from that action.
///
/// Returns lines like `"Nibbit HP 50→44 (−6)"`; empty when nothing notable
/// changed.
#[must_use]
pub fn diff_states(prev: &Value, nxt: &Value) -> Vec<String> {
    let mut out = Vec::new();
    player_hp_block(prev, nxt, &mut out);
    enemy_lines(prev, nxt, &mut out);
    orb_passive_lines(prev, nxt, &mut out);
    power_lines(prev, nxt, &mut out);
    spawn_lines(prev, nxt, &mut out);
    resource_lines(prev, nxt, &mut out);
    out
}

#[cfg(test)]
mod tests {
    //! Port of `tests/workbench/test_statediff.py` (every case, same names)
    //! plus the diff half of `test_timeline_details.py`.

    use super::*;
    use serde_json::json;

    fn combat(player_hp: i64, player_block: i64, enemies: Vec<Value>) -> Value {
        json!({
            "decision": "combat_play",
            "player": {"name": "Ironclad", "hp": player_hp, "max_hp": 80, "block": player_block},
            "enemies": enemies,
        })
    }

    fn enemy(index: i64, name: &str, hp: i64) -> Value {
        json!({"index": index, "name": name, "hp": hp, "max_hp": 50, "block": 0})
    }

    fn enemy_with_powers(index: i64, name: &str, hp: i64, powers: Vec<Value>) -> Value {
        let mut e = enemy(index, name, hp);
        e["powers"] = Value::Array(powers);
        e
    }

    fn pow(name: &str, amount: i64) -> Value {
        json!({"name": name, "amount": amount})
    }

    fn lines(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_no_change_is_empty() {
        let s = combat(70, 0, vec![enemy(0, "Nibbit", 50)]);
        assert!(diff_states(&s, &s).is_empty());
    }

    #[test]
    fn test_player_damage_and_heal() {
        let a = combat(70, 0, vec![]);
        let b = combat(64, 0, vec![]);
        assert_eq!(diff_states(&a, &b), lines(&["You HP 70→64 (−6)"]));
        assert_eq!(
            diff_states(&b, &combat(80, 0, vec![])),
            lines(&["You HP 64→80 (+16)"])
        );
    }

    #[test]
    fn test_player_block_change() {
        let a = combat(70, 0, vec![]);
        let b = combat(70, 5, vec![]);
        assert_eq!(diff_states(&a, &b), lines(&["You Block 0→5"]));
    }

    #[test]
    fn test_enemy_damage() {
        let a = combat(70, 0, vec![enemy(0, "Nibbit", 50)]);
        let b = combat(70, 0, vec![enemy(0, "Nibbit", 44)]);
        assert_eq!(diff_states(&a, &b), lines(&["Nibbit HP 50→44 (−6)"]));
    }

    #[test]
    fn test_enemy_death_when_hp_zero() {
        let a = combat(70, 0, vec![enemy(0, "Looter", 30)]);
        let b = combat(70, 0, vec![enemy(0, "Looter", 0)]);
        assert_eq!(
            diff_states(&a, &b),
            lines(&["Looter HP 30→0 (−30)  ☠ defeated"])
        );
    }

    #[test]
    fn test_enemy_death_when_removed() {
        let a = combat(70, 0, vec![enemy(0, "Looter", 30)]);
        let b = combat(70, 0, vec![]);
        assert_eq!(diff_states(&a, &b), lines(&["Looter HP 30→0  ☠ defeated"]));
    }

    #[test]
    fn test_removed_corpse_already_dead_is_silent() {
        let a = combat(70, 0, vec![enemy(0, "Looter", 0)]);
        let b = combat(70, 0, vec![]);
        assert!(diff_states(&a, &b).is_empty());
    }

    #[test]
    fn test_enemy_spawn() {
        let a = combat(70, 0, vec![enemy(0, "Nibbit", 50)]);
        let b = combat(
            70,
            0,
            vec![enemy(0, "Nibbit", 50), enemy(1, "Spire Growth", 60)],
        );
        assert_eq!(diff_states(&a, &b), lines(&["Spawned: Spire Growth"]));
    }

    #[test]
    fn test_duplicated_enemy_names_are_numbered_from_the_board_they_are_on() {
        let rat = "Two-Tailed Rat";
        let a = combat(
            70,
            0,
            vec![
                enemy(0, rat, 20),
                enemy(1, rat, 20),
                enemy_with_powers(2, rat, 20, vec![pow("Weak", 1)]),
            ],
        );
        // Rat #1 dies and leaves the list, rat #2 loses HP, rat #3 its Weak,
        // and a fourth rat spawns into slot 3.
        let b = combat(
            70,
            0,
            vec![enemy(1, rat, 14), enemy(2, rat, 20), enemy(3, rat, 20)],
        );
        assert_eq!(
            diff_states(&a, &b),
            lines(&[
                "Two-Tailed Rat #1 HP 20→0  ☠ defeated",
                "Two-Tailed Rat #2 HP 20→14 (−6)",
                "Two-Tailed Rat #3 −Weak",
                // On the AFTER board it is the third rat.
                "Spawned: Two-Tailed Rat #3",
            ])
        );
        // A unique name stays bare.
        let a = combat(70, 0, vec![enemy(0, rat, 20), enemy(1, "Louse", 9)]);
        let b = combat(70, 0, vec![enemy(0, rat, 20), enemy(1, "Louse", 4)]);
        assert_eq!(diff_states(&a, &b), lines(&["Louse HP 9→4 (−5)"]));
    }

    #[test]
    fn test_player_power_gained_with_amount() {
        let a = json!({"player": {"hp": 70, "block": 0}, "player_powers": []});
        let b = json!({"player": {"hp": 70, "block": 0}, "player_powers": [pow("Strength", 2)]});
        assert_eq!(diff_states(&a, &b), lines(&["You +2 Strength"]));
    }

    #[test]
    fn test_power_gained_without_amount() {
        let a = json!({"player": {"hp": 70, "block": 0}, "player_powers": []});
        let b = json!({"player": {"hp": 70, "block": 0}, "player_powers": [pow("Intangible", 0)]});
        assert_eq!(diff_states(&a, &b), lines(&["You +Intangible"]));
    }

    #[test]
    fn test_power_gained_with_negative_amount_renders_signed() {
        // A debuff gained at a negative amount must read "You −1 Shrink", not
        // the old "You +-1 Shrink".
        let a = json!({"player": {"hp": 70, "block": 0}, "player_powers": []});
        let b = json!({"player": {"hp": 70, "block": 0}, "player_powers": [pow("Shrink", -1)]});
        assert_eq!(diff_states(&a, &b), vec![format!("You {MINUS}1 Shrink")]);
    }

    #[test]
    fn test_enemy_power_amount_changed() {
        let a = combat(
            70,
            0,
            vec![enemy_with_powers(
                0,
                "Nibbit",
                50,
                vec![pow("Vulnerable", 3)],
            )],
        );
        let b = combat(
            70,
            0,
            vec![enemy_with_powers(
                0,
                "Nibbit",
                50,
                vec![pow("Vulnerable", 1)],
            )],
        );
        assert_eq!(diff_states(&a, &b), lines(&["Nibbit Vulnerable 3→1"]));
    }

    #[test]
    fn test_power_lost() {
        let a = json!({"player": {"hp": 70, "block": 0}, "player_powers": [pow("Strength", 2)]});
        let b = json!({"player": {"hp": 70, "block": 0}, "player_powers": []});
        assert_eq!(diff_states(&a, &b), vec![format!("You {MINUS}Strength")]);
    }

    #[test]
    fn test_gold_change() {
        let a = json!({"player": {"hp": 70, "block": 0, "gold": 120}});
        let b = json!({"player": {"hp": 70, "block": 0, "gold": 70}});
        assert_eq!(diff_states(&a, &b), lines(&["Gold 120→70"]));
    }

    #[test]
    fn test_relic_gained() {
        let a = json!({"player": {"hp": 70, "block": 0, "relics": []}});
        let b = json!({"player": {"hp": 70, "block": 0, "relics": [{"name": "Burning Blood"}]}});
        assert_eq!(diff_states(&a, &b), lines(&["+Relic Burning Blood"]));
    }

    #[test]
    fn test_potion_gained_and_used() {
        let none = json!({"player": {"hp": 70, "block": 0, "potions": []}});
        let one = json!({"player": {"hp": 70, "block": 0, "potions": [{"index": 0, "name": "Fire Potion"}]}});
        assert_eq!(diff_states(&none, &one), lines(&["+Potion Fire Potion"]));
        assert_eq!(diff_states(&one, &none), lines(&["Used Fire Potion"]));
    }

    fn defect_turn(round_no: i64, enemies: Vec<Value>, orbs: Vec<Value>) -> Value {
        json!({
            "decision": "combat_play",
            "round": round_no,
            "player": {"name": "Defect", "hp": 75, "max_hp": 75, "block": 0},
            "enemies": enemies,
            "orbs": orbs,
        })
    }

    fn orb(index: i64, kind: &str, passive: i64) -> Value {
        json!({"index": index, "type": kind, "name": kind, "passive": passive, "evoke": passive})
    }

    #[test]
    fn test_lightning_orb_passive_attributed_on_turn_end() {
        let pre = defect_turn(
            1,
            vec![enemy(0, "Shrinker Beetle", 38)],
            vec![orb(0, "Lightning", 3)],
        );
        let post = defect_turn(
            2,
            vec![enemy(0, "Shrinker Beetle", 35)],
            vec![orb(0, "Lightning", 3)],
        );
        let lines = diff_states(&pre, &post);
        assert!(lines.contains(&"Shrinker Beetle HP 38→35 (−3)".to_string()));
        assert!(lines.contains(&"Lightning orb passive: 3 damage to a random enemy".to_string()));
    }

    #[test]
    fn test_orb_passive_only_fires_on_round_increment() {
        // Same round (e.g. a card play) → no orb-passive annotation.
        let a = defect_turn(
            1,
            vec![enemy(0, "Beetle", 38)],
            vec![orb(0, "Lightning", 3)],
        );
        let b = defect_turn(
            1,
            vec![enemy(0, "Beetle", 35)],
            vec![orb(0, "Lightning", 3)],
        );
        assert!(diff_states(&a, &b)
            .iter()
            .all(|l| !l.contains("orb passive")));
    }

    #[test]
    fn test_orb_passive_per_type_descriptions_and_collapse() {
        let orbs = vec![
            orb(0, "Lightning", 3),
            orb(1, "Lightning", 3),
            orb(2, "Frost", 2),
            orb(3, "Plasma", 1),
            orb(4, "Dark", 6),
            orb(5, "Glass", 4),
        ];
        let pre = defect_turn(1, vec![enemy(0, "Beetle", 38)], orbs.clone());
        let post = defect_turn(2, vec![enemy(0, "Beetle", 38)], orbs);
        let lines = diff_states(&pre, &post);
        for expected in [
            "Lightning orb passive: 3 damage to a random enemy (×2)",
            "Frost orb passive: +2 block",
            "Plasma orb passive: +1 energy",
            "Dark orb passive: +6 stored damage",
            "Glass orb passive: 4 damage to all enemies",
        ] {
            assert!(
                lines.contains(&expected.to_string()),
                "{expected:?} not in {lines:?}"
            );
        }
    }

    #[test]
    fn test_no_orb_passive_for_non_defect_turn() {
        let pre =
            json!({"decision": "combat_play", "round": 1, "player": {"hp": 70}, "enemies": []});
        let post =
            json!({"decision": "combat_play", "round": 2, "player": {"hp": 70}, "enemies": []});
        assert!(diff_states(&pre, &post)
            .iter()
            .all(|l| !l.contains("orb passive")));
    }

    // ── test_timeline_details.py (the diff part) ────────────────────────────

    fn timeline_state(enemy_hp: i64) -> Value {
        json!({
            "decision": "combat_play",
            "round": 1,
            "player": {"name": "Ironclad", "hp": 70, "max_hp": 80, "block": 0},
            "enemies": [{"index": 0, "name": "Nibbit", "hp": enemy_hp, "max_hp": 50, "block": 0}],
            "hand": [],
        })
    }

    #[test]
    fn test_timeline_includes_details() {
        // Step 0's action result is step 1's state -> Nibbit took 6 damage.
        assert_eq!(
            diff_states(&timeline_state(50), &timeline_state(44)),
            lines(&["Nibbit HP 50→44 (−6)"])
        );
    }

    // ── Python-semantics pins ───────────────────────────────────────────────

    #[test]
    fn float_hp_is_skipped_for_the_player_but_diffed_for_enemies() {
        // `isinstance(hp, int)` excludes floats on the player side …
        let a = json!({"player": {"hp": 70.0, "block": 0}});
        let b = json!({"player": {"hp": 64.0, "block": 0}});
        assert!(diff_states(&a, &b).is_empty());
        // … while the enemy path subtracts whatever it finds (`abs(-6.5)` →
        // "6.5", `abs(-6.0)` → "6.0").
        let a = combat(70, 0, vec![json!({"index": 0, "name": "N", "hp": 50.5})]);
        let b = combat(70, 0, vec![json!({"index": 0, "name": "N", "hp": 44})]);
        assert_eq!(diff_states(&a, &b), lines(&["N HP 50.5→44 (−6.5)"]));
    }

    #[test]
    fn potion_counter_keeps_first_occurrence_order_and_multiplicity() {
        let a = json!({"player": {"potions": [{"name": "B"}]}});
        let b = json!({"player": {"potions": [{"name": "A"}, {"name": "B"}, {"name": "A"}, {"name": "B"}]}});
        assert_eq!(
            diff_states(&a, &b),
            lines(&["+Potion A", "+Potion A", "+Potion B"])
        );
    }
}
