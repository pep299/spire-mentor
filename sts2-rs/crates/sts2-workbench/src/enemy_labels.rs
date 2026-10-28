//! Name enemies so a duplicated name still says WHICH one.
//!
//! Three Two-Tailed Rats all render as "Two-Tailed Rat", so "Play Strike →
//! Two-Tailed Rat" does not say which rat was hit. A name shared by several
//! enemies gets its 1-based rank among the enemies sharing it, in list order
//! (left to right on the board): "Two-Tailed Rat #2". A unique name stays
//! bare.
//!
//! The ONE Rust home of the rule. It is the same rule as the SPA's
//! `enemyLabeller` (`sts2-pycli/web/src/enemyLabels.ts`) and the Python
//! `enemy_labels.enemy_labels` (`sts2-pycli/python/enemy_labels.py`), so every
//! surface numbers the same enemy the same way. Only enemies the board shows
//! are counted: an entry marked `is_dead` / `is_escaped` / `is_alive: false`
//! keeps its bare name and takes no number (the combat view already skips
//! `is_dead`; the mod's wire keeps a dying monster until its move resolves).
//! So the numbers close up when one dies — consistent at any moment, not
//! across the fight.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::render::py;

/// One label per name, same order: a shared name gets `" #k"`.
#[must_use]
pub fn labels_from_names<S: AsRef<str>>(names: &[S]) -> Vec<String> {
    labels_counting(names, |_| true)
}

/// [`labels_from_names`] counting only the names where `counted(i)` holds;
/// an uncounted one keeps its bare name.
fn labels_counting<S: AsRef<str>>(names: &[S], counted: impl Fn(usize) -> bool) -> Vec<String> {
    let mut total: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        if counted(i) {
            *total.entry(name.as_ref()).or_default() += 1;
        }
    }
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let name = name.as_ref();
            if counted(i) && total.get(name).copied().unwrap_or(0) > 1 {
                let rank = seen.entry(name).or_default();
                *rank += 1;
                format!("{name} #{rank}")
            } else {
                name.to_string()
            }
        })
        .collect()
}

/// True for an entry the board no longer shows: dead or escaped.
#[must_use]
pub fn off_board(enemy: &Value) -> bool {
    py::get_raw(enemy, "is_dead").is_some_and(py::truthy)
        || py::get_raw(enemy, "is_escaped").is_some_and(py::truthy)
        || enemy.get("is_alive") == Some(&Value::Bool(false))
}

/// One display label per entry of `enemies` (the view's enemy dicts), in the
/// same order. The grouping name is Python's `str(e.get("name") or
/// e.get("id") or f"enemy {i}")`.
#[must_use]
pub fn enemy_labels(enemies: &[Value]) -> Vec<String> {
    let names: Vec<String> = enemies
        .iter()
        .enumerate()
        .map(|(i, e)| {
            ["name", "id"]
                .iter()
                .find_map(|k| py::get_raw(e, k).filter(|v| py::truthy(v)))
                .map_or_else(|| format!("enemy {i}"), py::to_str)
        })
        .collect();
    labels_counting(&names, |i| !off_board(&enemies[i]))
}

/// The label of the enemy whose stable `index` field equals `index`, or
/// `None` when no enemy carries it.
#[must_use]
pub fn enemy_label_by_index(enemies: &[Value], index: i64) -> Option<String> {
    let pos = enemies
        .iter()
        .position(|e| e.get("index").and_then(Value::as_i64) == Some(index))?;
    enemy_labels(enemies).into_iter().nth(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn duplicated_names_are_numbered_unique_stay_bare() {
        let got = labels_from_names(&["Rat", "Louse", "Rat", "Rat"]);
        assert_eq!(got, ["Rat #1", "Louse", "Rat #2", "Rat #3"]);
    }

    #[test]
    fn enemy_labels_reads_name_then_id_then_position() {
        let es = vec![
            json!({"index": 0, "name": "Rat"}),
            json!({"index": 2, "name": "Rat"}),
            json!({"index": 3, "id": "CULTIST"}),
            json!({"index": 4}),
        ];
        assert_eq!(
            enemy_labels(&es),
            ["Rat #1", "Rat #2", "CULTIST", "enemy 3"]
        );
    }

    #[test]
    fn off_board_entries_take_no_number() {
        let es = vec![
            json!({"name": "Rat"}),
            json!({"name": "Rat", "is_alive": false}),
            json!({"name": "Rat", "is_escaped": true}),
            json!({"name": "Rat"}),
        ];
        assert_eq!(enemy_labels(&es), ["Rat #1", "Rat", "Rat", "Rat #2"]);
    }

    #[test]
    fn lookup_is_by_stable_index_not_position() {
        // Slot 1 died: the view lists slots 0 and 2, which are rats #1 and #2.
        let es = vec![
            json!({"index": 0, "name": "Rat"}),
            json!({"index": 2, "name": "Rat"}),
        ];
        assert_eq!(enemy_label_by_index(&es, 2).as_deref(), Some("Rat #2"));
        assert_eq!(enemy_label_by_index(&es, 1), None);
    }
}
