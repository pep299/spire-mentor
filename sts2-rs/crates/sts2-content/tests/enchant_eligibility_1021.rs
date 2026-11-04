//! `can_enchant_card` agrees with C# `EnchantmentModel.CanEnchant` for every
//! registry card and every v0.107.1 enchantment (GH #1021).
//!
//! The fixture is the C# oracle's own answer: `python
//! python/dump_enchant_eligibility.py` (sts2-pycli) puts each card in a deck
//! and enchants it through `CardCmd.Enchant`. Regenerate it when the card
//! registry or the oracle's game build changes.

use std::collections::BTreeSet;

use sts2_content::cards::spec_index::{
    card_pod_from_id, COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, NECROBINDER_SPECS,
    REGENT_SPECS, SILENT_SPECS, STATUS_SPECS,
};
use sts2_content::enchantments::can_enchant_card;
use sts2_core::cards::EnchantmentId;

const FIXTURE: &str = include_str!("fixtures/enchant_eligibility_v0107.json");

#[test]
fn can_enchant_card_matches_the_csharp_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let enchantments: Vec<(String, EnchantmentId)> = fixture["enchantments"]
        .as_array()
        .expect("enchantments")
        .iter()
        .map(|e| {
            let name = e.as_str().expect("name").to_owned();
            let id = EnchantmentId::from_wire_name(&name).expect("known enchantment");
            (name, id)
        })
        .collect();
    let eligible = fixture["eligible"].as_object().expect("eligible");

    let mut checked = 0;
    let mut mismatches = Vec::new();
    for table in [
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        REGENT_SPECS,
        NECROBINDER_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
    ] {
        for factory in table {
            let spec = factory();
            let Some(row) = eligible.get(spec.name) else {
                continue;
            };
            let truth: BTreeSet<&str> = row
                .as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_str().expect("enchantment"))
                .collect();
            let card = card_pod_from_id(spec.id, false).expect("registry card");
            for (name, id) in &enchantments {
                let rust = can_enchant_card(&spec, &card, *id, sts2_core::GameVersion::V0_107_1);
                let csharp = truth.contains(name.as_str());
                if rust != csharp {
                    mismatches.push(format!("{}@{name}: rust {rust}, C# {csharp}", spec.name));
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000, "only {checked} pairs checked");
    assert!(
        mismatches.is_empty(),
        "{} of {checked} card x enchantment pairs disagree with C#:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
