from sts2_hint.display_maps import loc_id, monster_title
from workbench.assets import slugify


def test_loc_id_normalises_pascal_and_snake():
    assert loc_id("StrikeIronclad") == "STRIKE_IRONCLAD"
    assert loc_id("STRIKE_IRONCLAD") == "STRIKE_IRONCLAD"
    assert loc_id("RELIC.WHETSTONE") == "WHETSTONE"


def test_loc_id_splits_capital_runs_like_game_slugify():
    # PascalCase ids with a one-letter "word" abutting the next word (a run of
    # capitals) must split between every capital, matching the game's
    # ``StringHelper.Slugify``. The old lower->upper-only boundary produced
    # ``EXPECT_AFIGHT`` / ``IAM_INVINCIBLE``, so these cards rendered the raw id
    # and found no art in the deck / reward / shop views.
    assert loc_id("ExpectAFight") == "EXPECT_A_FIGHT"
    assert loc_id("IAmInvincible") == "I_AM_INVINCIBLE"
    # The combat SCREAMING_SNAKE form already matches and is left as-is.
    assert loc_id("EXPECT_A_FIGHT") == "EXPECT_A_FIGHT"


def test_loc_id_strips_card_suffix_to_match_loc_key():
    # A subset of sts2_rs card ids carry a disambiguating ``_CARD`` suffix
    # (``CardId::StampedeCard`` -> name "STAMPEDE_CARD") that the localization
    # keys and art slugs omit (``STAMPEDE.title`` / ``stampede.png``). Collapse
    # it so names resolve and art is found.
    assert loc_id("StampedeCard") == "STAMPEDE"
    assert loc_id("STAMPEDE_CARD") == "STAMPEDE"
    assert loc_id("INFERNO_CARD") == "INFERNO"
    assert loc_id("FOCUSED_STRIKE_CARD") == "FOCUSED_STRIKE"
    # A real ``_IRONCLAD`` suffix (or any non-_CARD tail) is untouched.
    assert loc_id("DEFEND_IRONCLAD") == "DEFEND_IRONCLAD"


def test_loc_id_keeps_relic_card_suffix():
    # The Membership Card relic's loc key IS ``MEMBERSHIP_CARD.title``; the
    # card-only ``_CARD`` strip must not turn it into ``MEMBERSHIP``.
    assert loc_id("RELIC.MEMBERSHIP_CARD") == "MEMBERSHIP_CARD"


def test_loc_id_strips_status_suffix_to_match_loc_key():
    # The Sloth status card's Rust id is ``SLOTH_STATUS`` (CardId::SlothStatus),
    # but the C# class is plain ``Sloth`` so the loc key / art stem is ``SLOTH``.
    # Without the strip the card-select screen shows the raw ``SLOTH_STATUS.title``
    # / ``SLOTH_STATUS.description`` keys and finds no art (Knowledge Daemon pick).
    assert loc_id("SLOTH_STATUS") == "SLOTH"
    assert loc_id("SlothStatus") == "SLOTH"


def test_monster_title_strips_unresolved_specimen_template():
    # Test Subject's loc name is the raw template "Test Subject #C{Count}"
    # (the game fills {Count} from cross-run save progress the workbench doesn't
    # model). Left verbatim it both leaks the {Count} into the UI and breaks art
    # lookup (slug "test_subject_c_count" misses monsters/test_subject.png). It
    # must collapse to the bare species name, whose slug finds the sprite/rig.
    title = monster_title("TestSubject")
    assert title == "Test Subject"
    assert "{" not in title and "#" not in title
    assert slugify(title) == "test_subject"


def test_monster_title_leaves_plain_names_untouched():
    assert monster_title("Nibbit") == "Nibbit"
    assert monster_title("TheForgotten") == "The Forgotten"
    assert monster_title("AssassinRubyRaider") == "Assassin Raider"
    # Uncurated id falls back to CamelCase spacing.
    assert monster_title("SomeNewMonster") == "Some New Monster"


def test_monster_title_decimillipede_segments_share_base_name():
    # The engine spawns three directional segment kinds, but the game gives all
    # three the single localized name "Decimillipede" (DECIMILLIPEDE_SEGMENT.name
    # on the shared base class). The directional variants are excluded from the
    # title->id map (their titles collide), so without an explicit reverse-map
    # entry monster_title falls back to CamelCase spacing ("Decimillipede Segment
    # Front"), whose slug finds no sprite -> the combatant renders as a blank box
    # in the workbench (#188). Map all three to the base name so they show the
    # correct label AND resolve to monsters/decimillipede.png.
    for kind in (
        "DecimillipedeSegmentFront",
        "DecimillipedeSegmentMiddle",
        "DecimillipedeSegmentBack",
    ):
        title = monster_title(kind)
        assert title == "Decimillipede", kind
        assert slugify(title) == "decimillipede", kind
