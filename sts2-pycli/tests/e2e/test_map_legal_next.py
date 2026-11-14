import json
from pathlib import Path

from ppo.e2e import tokens as K


def test_all_legal_first_moves_have_next_tokens_after_ancient():
    path = Path(__file__).parent / "fixtures" / "map_ancient_before_first_move.json"
    full_map = json.loads(path.read_text())
    moves = [{"action": "move", "col": col} for col in [1, 3, 6]]
    buf = K._Buf()
    K._map_body(buf, full_map, moves)
    assert int(buf._cont[: buf.n, K.XF["N_NEXT"]].sum()) == len(moves)
