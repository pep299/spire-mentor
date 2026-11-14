# fork固有の接続契約

上流の比較基準は `tfriedel/spire-mentor@66979787`。
ゲーム挙動の修正と、既存クライアントの入力形式を受ける互換処理を区別する。

## 現行Labへの互換処理

以下は `pep299/sts2-agent-lab` の現在の呼び出しを維持するための差分であり、
上流よりゲーム規則が正しいことを理由に維持するものではない。

| 契約 | Labの利用箇所 | 削除条件 |
|---|---|---|
| `local_cost` のデコード・出力・正規化 | `agents/mentor/combat.py::wire_cards`、`nested_combat.py::public_card` | 入力と公開カード照合を上流の `cost_mods` へ移す。Issue #37で追跡 |
| 開始・完了プレイの履歴、Block獲得元の公開Play ID、発動元カードの参照 | `combat_history.py::apply_public_history`、`nested_combat.py::checked_snapshot` / `public_projection` | 上流の履歴形式と継続ミラーへ接続を移し、途中選択・再使用後の状態を実ゲームと照合する |
| `condition_observed_draws` / `observed_draws_consumed` | `nested_combat.py::condition_draws` / `resume_choice` | 公開されたドローを継続ミラーへ反映し、以前の局面からの行動再生を不要にする |
| `normalize_public_cards` / `open_generated_choice` | 公開された候補の属性照合・生成候補の投入 | 上流APIで費用・強化・付与効果を含む同じ候補を表現できる接続へ移す |
| `resolve_public_choice` | 公開された複数選択の一括確定 | 上流の選択継続APIへ移し、枚数・重複・範囲の検査を維持する |
| `open_relic_start_choice` | `policy.py` の戦闘開始時のレリック選択 | 上流の開始処理の保持・再開へ移し、選択後の残りの開始効果が一度だけ実行されることを確認する |
| `add_from_random_draw_subset` | `nested_combat.py::align_choice` のSeeker Strike候補照合 | 上流の選択種別・候補位置表現で、公開された部分集合を照合する |

移行前にAPIだけを削除すると、現行Labの呼び出しは動作しない。
上流そのものを現行Labの106テストへ接続すると、API不足と候補置換のエラー時契約の違いで16件が失敗する。
これは上流の戦闘処理が誤っていることを示す結果ではない。

費用形式の移行:
<https://github.com/pep299/sts2-agent-lab/issues/37>

## 再現可能なマップ入力の修正

E2Eのトークン生成で、Ancient直後の行判定と合法な移動先を反映する。
回帰テストは `sts2-pycli/tests/e2e/test_map_legal_next.py`。
保存した公開マップには合法な次の移動先が3つあるが、上流は `N_NEXT` を1つしか生成せず、修正後は3つになる。
同じ入力計算をPythonとRustで使う。

## 運用上の差分

GitHub Pagesワークフローの削除は既存のfork方針として維持する。
配布PPOモデルの固定先と重みは変更しない。

## 維持の判断基準

上流と重複するゲーム処理は上流へ戻す。
ゲーム挙動の独自修正を残す場合は、同じゲーム版・局面・行動で上流が実ゲームと食い違い、修正後に一致する再現ケースを添える。
接続互換は呼び出し元と削除条件を記載し、移行後も無条件に維持しない。
