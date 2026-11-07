//! GH #1129: a select that opens during the enemy turn blocks it.
//!
//! The enemy's hit triggers Centennial Puzzle, whose first draw is Seeker
//! Strike, which Hellraiser auto-plays. C# monster attacks run their hooks
//! under a `BlockingPlayerChoiceContext`, so Seeker's draw-pile select halts
//! the rest of the enemy turn until the player picks, the Puzzle's other two
//! draws included (`BlockingPlayerChoiceContext.cs:7-15` names this exact
//! case; `AttackCommand.cs:670`, `CentennialPuzzle.cs:47-50`).
//!
//! Rust used to run the rest of the enemy turn and the next turn's hand draw
//! with the select open, and the answer then left Seeker Strike parked on
//! the Play pile with no choice to resume it: a hang where only a no-op
//! `EndTurn` was legal (e2e lane seed 1130202794).
//!
//! The fixture is that run's combat state at the end of player turn 1.

use sts2_core::cards::CardId;
use sts2_core::{Action, ChoiceKind, CombatPhase, CombatState};
use sts2_wire::state_from_json;

fn fixture() -> CombatState {
    state_from_json(&fixture_json()).expect("builds")
}

fn fixture_json() -> serde_json::Value {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gh1129_hellraiser_seeker_pre.json"),
    )
    .expect("fixture");
    serde_json::from_str(&text).expect("json")
}

fn name(st: &CombatState, u: sts2_core::CardUid) -> CardId {
    st.cards[u.0.get() as usize - 1].card_id
}

#[test]
fn enemy_turn_waits_for_the_seeker_select() {
    let mut st = fixture();
    let deck = st.cards.len();
    let _ = sts2_core::apply(&mut st, Action::EndTurn);

    let pc = st.pending_choice.as_ref().expect("Seeker's select prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromRandomDrawSubset);
    assert_eq!(pc.options.len(), 3);
    assert_eq!(st.phase, CombatPhase::EnemyTurnExec, "mid enemy turn");
    assert!(st.player.hand.is_empty(), "the Puzzle's other draws wait");
    assert_eq!(st.player.play.len(), 1);
    assert_eq!(name(&st, st.player.play[0]), CardId::SeekerStrike);
    let picked = pc.options[2].card_uid.expect("a draw-pile card");
    assert!(st.player.draw.contains(&picked));

    let _ = sts2_core::apply(
        &mut st,
        Action::ResolveChoice {
            selection: smallvec::smallvec![2],
        },
    );
    assert!(st.pending_choice.is_none());
    assert_eq!(st.phase, CombatPhase::PlayerPlayPhase);
    assert!(st.player.play.is_empty(), "play: {:?}", st.player.play);
    assert!(st.pending_autoplay_resume.is_none());
    assert!(st.pending_draw_resume.is_none());
    assert!(!st.pending_turn_setup_resume);
    assert!(st
        .player
        .discard
        .iter()
        .any(|&u| name(&st, u) == CardId::SeekerStrike));
    assert!(st.player.hand.contains(&picked), "the pick stays in hand");
    assert_eq!(st.cards.len(), deck, "no card created or lost");
    sts2_core::invariants::check_combat_invariants(&st, "gh1129").expect("invariants");
    assert!(sts2_core::legal_actions(&st)
        .iter()
        .any(|a| matches!(a, Action::PlayCard { .. })));
}

/// A second Seeker Strike under the first: the enemy turn needs two answers,
/// each replayed in order, before it goes on.
#[test]
fn two_selects_in_one_enemy_turn_prompt_in_order() {
    let mut st = fixture();
    let first = st.player.draw[0];
    assert_eq!(name(&st, first), CardId::SeekerStrike);
    let mut second = st.cards[first.0.get() as usize - 1];
    let uid = sts2_core::CardUid::new(u32::try_from(st.cards.len() + 1).unwrap()).unwrap();
    second.uid = uid;
    st.cards.push(second);
    st.player.draw.insert(1, uid);

    let _ = sts2_core::apply(&mut st, Action::EndTurn);
    let pc = st.pending_choice.as_ref().expect("the first select");
    assert_eq!(pc.kind, ChoiceKind::AddFromRandomDrawSubset);
    assert_eq!(st.player.play.as_slice(), &[first]);
    let pick1 = pc.options[0].card_uid.unwrap();

    let _ = sts2_core::apply(
        &mut st,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pc = st.pending_choice.as_ref().expect("the second select");
    assert_eq!(pc.kind, ChoiceKind::AddFromRandomDrawSubset);
    assert_eq!(st.phase, CombatPhase::EnemyTurnExec);
    assert_eq!(
        st.player.play.as_slice(),
        &[uid],
        "the first Seeker finished"
    );
    assert_eq!(st.player.hand.as_slice(), &[pick1], "the first pick landed");
    assert!(!pc.options.iter().any(|o| o.card_uid == Some(pick1)));

    let _ = sts2_core::apply(
        &mut st,
        Action::ResolveChoice {
            selection: smallvec::smallvec![1],
        },
    );
    assert!(st.pending_choice.is_none());
    assert_eq!(st.phase, CombatPhase::PlayerPlayPhase);
    assert!(st.player.play.is_empty());
    assert!(st.player.hand.contains(&pick1));
    assert!(st.player.discard.contains(&first));
    assert!(st.player.discard.contains(&uid));
    sts2_core::invariants::check_combat_invariants(&st, "gh1129").expect("invariants");
}

/// An illegal answer is refused and the select stays open.
#[test]
fn an_illegal_answer_keeps_the_select_open() {
    let mut st = fixture();
    let _ = sts2_core::apply(&mut st, Action::EndTurn);
    let before = st.clone();
    let r = sts2_core::apply(
        &mut st,
        Action::ResolveChoice {
            selection: smallvec::smallvec![7],
        },
    );
    assert_eq!(r, sts2_core::StepResult::IllegalPlay);
    assert!(st == before, "nothing changed");
}

/// The Puzzle's draw finds an empty draw pile: its reshuffle's Stratagem grid
/// prompts mid enemy turn too, and the enemy turn waits for the pick.
#[test]
fn stratagem_reshuffle_in_the_enemy_turn_prompts() {
    let mut v = fixture_json();
    let powers = v["player"]["powers"].as_array_mut().expect("powers");
    powers.retain(|p| p["name"] != "HELLRAISER");
    powers.push(
        serde_json::json!({"amount": 1, "counter": 0, "display_amount": 1, "name": "STRATAGEM"}),
    );
    let draw = std::mem::take(v["deck"]["draw_pile"].as_array_mut().expect("draw"));
    v["deck"]["discard_pile"]
        .as_array_mut()
        .expect("discard")
        .extend(draw);
    let mut st = state_from_json(&v).expect("builds");
    assert!(st.player.draw.is_empty());

    let _ = sts2_core::apply(&mut st, Action::EndTurn);
    let pc = st.pending_choice.as_ref().expect("the Stratagem grid");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);
    assert_eq!(st.phase, CombatPhase::EnemyTurnExec, "mid enemy turn");
    assert!(st.player.hand.is_empty(), "nothing drawn yet");
    let picked = pc.options[1].card_uid.expect("a draw-pile card");

    let _ = sts2_core::apply(
        &mut st,
        Action::ResolveChoice {
            selection: smallvec::smallvec![1],
        },
    );
    assert!(st.pending_choice.is_none());
    assert_eq!(st.phase, CombatPhase::PlayerPlayPhase);
    assert!(st.player.hand.contains(&picked), "the pick stays in hand");
    sts2_core::invariants::check_combat_invariants(&st, "gh1129").expect("invariants");
}
