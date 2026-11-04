"""GH #545: a card token must say WHICH deck op it belongs to.

Only `shop_remove_card` set `cat[6]`; remove / upgrade / transform / enchant /
grant all emitted a bare `card_tok`. The only separator was `state_cat[8]`,
and the wb-27d55ac0 sweep showed that surface integer has ZERO authority over
the champion's unconditional anti-upgraded prior.
"""

from ppo import schema, vocab
from ppo.encoder import featurize_row


def _row(kind: str, extra_obs: dict | None = None):
    obs = {
        "deck": [["STRIKE", False], ["DEFEND", False]],
        "card_removal_options": [["STRIKE", False]],
        "card_grant_options": [["BASH", False]],
        "upgrade_select_deck_indices": [0],
        "transform_select_deck_indices": [0],
        "enchant_select_deck_indices": [0],
        "max_hp": 80,
        **(extra_obs or {}),
    }
    return featurize_row(obs, [{"action": kind, "idx": 0}], {})


def test_each_deck_op_gets_its_own_cat10_row():
    seen = {}
    for kind, op in [
        ("remove_deck_card", "remove"),
        ("upgrade_deck_card", "upgrade"),
        ("transform_deck_card", "transform"),
        ("enchant_deck_card", "enchant"),
        ("grant_deck_card", "grant"),
    ]:
        feat = _row(kind)
        seen[op] = int(feat.act_cat[0][schema.CAT_DECK_OP])
        assert seen[op] == vocab.deck_op_index(op)
    # Five distinct rows: the whole point is that they stop colliding.
    assert len(set(seen.values())) == 5


def test_n_cat_is_fourteen_and_the_appended_pads_are_last():
    assert schema.N_CAT == 14
    pad = schema.empty_cat()
    assert pad[schema.CAT_DECK_OP] == vocab.N_DECK_OP
    assert pad[schema.CAT_ENCHANT] == vocab.enchantments().size
    assert pad[schema.CAT_CARD_TYPE] == vocab.N_CARD_TYPE
    assert pad[schema.CAT_CARD_RARITY] == vocab.N_CARD_RARITY


def test_the_deck_op_pad_stays_at_row_eleven():
    # The table was defined with one spare row; "bundle" (GH #978) took it.
    # Every committed champion's `emb_deck_op` has 12 rows with the pad at
    # 11, so the pad must not move: the next op needs a widened table.
    assert vocab.N_DECK_OP == 11
    assert len(vocab.DECK_OPS) == vocab.N_DECK_OP
    assert vocab.deck_op_index("bundle") == 10
