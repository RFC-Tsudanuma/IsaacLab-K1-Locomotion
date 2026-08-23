# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""カリキュラムの終値固定 — fine-tune 段 (``--load_pretrained``) の共有部品。

なぜ walk_kick 側の共有モジュールなのか
---------------------------------------
初出は :mod:`..walk_inside_kick.walk_inside_kick_env_cfg` の
``_pin_curricula_at_end`` (2026-08-23)。「収束済み checkpoint から fine-tune 段を
積むときは、全カリキュラムを終値に固定してから積む」という手順は
**walk_kick 系のどのタスクでも同じ**で、inside 固有の要素は 1 つも無い
(判定は ``func`` の identity、値は全て term 自身の params から読む)。

2 つ目の利用者 (:mod:`..walk_lob_plant.walk_lob_plant_env_cfg` の stage 3) が
現れた時点でここへ切り出した。**inside 側の挙動は 1 ビットも変えていない** —
あちらは名前を ``_pin_curricula_at_end`` のまま別名 import して使い続けており、
モジュール docstring の ``:func:`_pin_curricula_at_end``` 参照もそのまま生きる。

置き場所が :mod:`..walk_kick` なのは、キック系タスクの共通土台
(:class:`~.walk_kick_env_cfg.K1WalkKickEnvCfg` と :mod:`.mdp`) がここにあるため。
この関数が知っているカリキュラム関数は全て :mod:`.mdp.curriculums` のもので、
除外リストだけが locomotion 側 (:class:`~..locomotion.flat_env_cfg.K1FlatCurriculumCfg`)
を参照する。

使い方 (fine-tune 段の ``__post_init__``)::

    super().__post_init__()      # 報酬・コマンド・イベントが全部揃うまで待つ
    pin_curricula_at_end(self)   # ← その後で固定する
    _apply_rough_terrain(self)   # 地形や DR の差分はこの後

**必ず ``super()`` の後**。まだ登録されていない curriculum 項は固定できないし、
固定した後に新しいランプを足すと、その 1 本だけが巻き戻る側に残る。
"""

from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from ..locomotion.mdp.curriculums import (
    lin_vel_command_curriculum,
    modify_command_resampling_time_range,
    modify_push_robot,
)
from . import mdp

if TYPE_CHECKING:
    # 型注釈のためだけの import。実行時に読むと walk_kick_env_cfg (重い) を
    # 巻き込むうえ、あちらがこのモジュールを import したときに循環する。
    from .walk_kick_env_cfg import K1WalkKickEnvCfg

# --------------------------------------------------------------------------- #
# :func:`pin_curricula_at_end` が **固定せずそのまま残す** カリキュラム項。
#
# 3 つとも locomotion 側 (:class:`~..locomotion.flat_env_cfg.K1FlatCurriculumCfg`) が
# 全 K1 タスクへ配っている **環境 DR のスケジュール** で、報酬のランプではない:
#
#   * modify_command_resampling_time_range … base_velocity の再サンプリング間隔
#   * lin_vel_command_curriculum           … 線速度コマンド範囲の段階拡大
#   * modify_push_robot                    … 外乱プッシュの強さ / 間隔
#
# 残す理由が 3 つある:
#
# 1. **巻き戻る向きが「易しい方」**。フェードイン系が巻き戻ると「蹴らない方が得」に
#    なるのが問題なのに対し、こちらが巻き戻ると外乱が弱く・コマンド帯が狭くなる
#    だけで、収支が逆転する経路が無い。
# 2. **窓が短い**。num_steps は raw step で 6000-14000 = 250-583 iteration
#    (steps_per_iteration = 24)。fine-tune の既定 3000 iteration のごく序盤で
#    終値に着く。
# 3. ``lin_vel_command_curriculum`` はそもそも壁時計のランプではなく **追従誤差で
#    段が進むゲート**。「終値を書き込む」という操作が意味を持たない。
#
# また、このリポジトリの全 PLAY cfg が既にこの 3 項を生かしたまま回している
# (CurriculumManager は PLAY でも step 0 から走る) ので、ここだけ挙動を変えると
# 他タスクの PLAY と比較できなくなる。
#
# **func の identity で判定する** (名前ではなく)。名前で書くと、将来 cfg 側で項名を
# 変えたときに黙って NotImplementedError 側へ落ちる。
# --------------------------------------------------------------------------- #
UNPINNED_CURRICULUM_FUNCS = (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
)


def pin_curricula_at_end(cfg: "K1WalkKickEnvCfg", *, expansion_alpha: float = 1.0) -> list[str]:
    """全カリキュラム項の **終値を対象へ直接書き込み、項そのものを None にする**。

    なぜ必要か
    ----------
    stage 2/3 は ``--load_pretrained`` で **収束済み**の checkpoint から始める
    (基準 run 2026-08-22_11-56-42、3600 iteration、カリキュラムは全て 3000 で終点に
    到達済み)。``--load_pretrained`` は ``--resume`` と違って ``common_step_counter`` を
    引き継がず 0 から数え直すので、カリキュラムを生かしたままだと **全部のランプが
    巻き戻る**:

    * キック報酬 4 項 (direction / scaled / inside_contact / foot_ceiling) が
      weight 0 からフェードインし直す。この間 ``kick_finished`` は「残りの歩行報酬を
      捨てるコスト」だけを課すので、**最初の 500 iteration は蹴らない方が得**が
      明示的に成立する
      (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg._freeze_fade_in_curricula` の
      docstring にある実測。500 iteration 後に weight が戻っても、そのときには
      蹴らなくなっているので払われる先が無い)。
    * 拡大ゲートの α が 0 に戻り、ボールが ±60°・0.5-0.8 m の限定レンジに縮む。
      収束済みポリシーには易しすぎるうえ、``ball_avoidance`` が 0 に落ちて
      ``approach_penalty`` が復活するので、回り込みの構えを壊す方向に更新される。
    * ``sigma_velocity`` が 0.5 → 1.0 に戻り、速度の採点が緩む
      (「指令どおりに蹴る」の圧が消える)。
    * ``kick_foot_ceiling`` の weight が 0 に戻る。接触点をボール中心より下へ
      置かせる圧は実機の「巻き込んで転ぶ」への直接の対策なので、ここが緩むのが
      いちばん困る (:data:`~..walk_inside_kick.walk_inside_kick_env_cfg._FOOT_LOW_H_TARGET`)。
    * ``kick_velocity_strong`` が満額 (=「速く蹴るほど得」) で復活する。この項は
      **トーキックを名指しで要求する**項で、退場させたのが inside の肝
      (:data:`~..walk_inside_kick.walk_inside_kick_env_cfg._INSIDE_STRONG_KNOTS`)。

    なぜ ``_freeze_fade_in_curricula`` を使い回さないのか
    ----------------------------------------------------
    あちらは (1) ``func`` が ``linear_reward_weight`` の項しか見ず、(2) ``end_step`` が
    ``before_iter`` 以下のものだけを対象にし、(3) 項は残したまま
    ``start_weight = end_weight`` に潰す、という作り。fine-tune 段で足りないのは
    (1) と (2) の方:

    * ``piecewise_reward_weight`` (strong の折れ線) と ``linear_reward_param``
      (σ_velocity) と ``piecewise_reward_param`` と
      ``kick_rate_gated_expansion`` (拡大ゲート) は対象外なので、そのまま巻き戻る。
    * ``kick_velocity_overshoot_weight`` の窓は 1500 → 3000 なので、
      ``before_iter = 500`` では拾えない。基準 run では完走しているので凍結が正しい。

    (3) の「項を残す」は、まだ動く窓が後ろに残っている段では利点 (今いくつなのかが
    ``Curriculum/...`` に出続ける) だが、**全部の窓が既に閉じているこの段では
    ただのノイズ**。項ごと ``None`` にすると:

    * 「カリキュラムはもう 1 本も無い」を関数の最後に検査できる (下の assert)。
      定数化しただけだと、新しい項が足されたときに黙って巻き戻る側へ回る。
    * PLAY cfg が自動的に正しくなる。CurriculumManager は PLAY でも
      ``common_step_counter`` 0 から走るので、項が生きていると PLAY で見る値が
      学習終盤と食い違う (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.hold_sigma_direction`
      の「2. アニールが入っている段の PLAY」と同じ機序)。継承だけで直る。

    Args:
        cfg: ``__post_init__`` を通した後の env cfg。
        expansion_alpha: 拡大ゲートを固定する α [0, 1]。既定 1.0 = 全方位。
            基準 run 2026-08-22_11-56-42 の ``Curriculum/kick_expansion/alpha`` は
            3643 iteration 時点で **1.0** (kick_rate_ema 0.996) なので既定でよい。
            別の checkpoint から始めるときは、その run の同じタグを見て合わせること
            (実力より広い範囲を固定すると、ゲートが本来やる「崩れたら戻る」が
            効かない状態で難易度だけ据え置かれる)。

    Returns:
        固定した curriculum 項の名前 (呼び出し側の検証・表示用)。

    Raises:
        NotImplementedError: 終値の意味が分からない ``func`` の項が残っていたとき。
            **黙って巻き戻らせないため、握り潰さずに落とす。** 新しいカリキュラムを
            足したら、この関数にも固定の仕方を書くこと。
    """
    pinned: list[str] = []

    for name in sorted(dir(cfg.curriculum)):
        if name.startswith("_"):
            continue
        term = getattr(cfg.curriculum, name, None)
        # configclass のメソッド (to_dict / replace など) も dir() に出るので、
        # CurrTerm であることを型で確かめてから触る。
        if not isinstance(term, CurrTerm):
            continue

        func = term.func
        params = term.params

        if func in UNPINNED_CURRICULUM_FUNCS:
            # 報酬のランプではない環境 DR のスケジュール。理由は
            # :data:`UNPINNED_CURRICULUM_FUNCS` のコメント。
            continue

        if func is mdp.linear_reward_weight:
            reward_term(cfg, params["term_name"], name).weight = params["end_weight"]

        elif func is mdp.piecewise_reward_weight:
            # 折れ線は最後の knot の weight で頭打ちになる (piecewise_reward_weight の
            # 実装: step >= knots[-1][0] なら knots[-1][1])。inside / lob_plant の strong は
            # :data:`~..walk_inside_kick.walk_inside_kick_env_cfg._INSIDE_STRONG_KNOTS` の最終 knot が (1200, 0.0) なので 0.0。
            reward_term(cfg, params["term_name"], name).weight = params["knots"][-1][1]

        elif func is mdp.linear_reward_param:
            # σ_velocity (1.0 → 0.5)。
            reward_term(cfg, params["term_name"], name).params[params["param_name"]] = params["end_value"]

        elif func is mdp.piecewise_reward_param:
            # 折れ線の params 版。piecewise_reward_weight と同じく最後の knot で
            # 頭打ちになる (実装: step >= knots[-1][0] なら knots[-1][1])。
            #
            # NOTE: **現在どの呼び出し元にも 1 つも無い。** 初出は inside / lob_plant の
            #       ``kick_plant_lon`` の ``lon_span`` (3 段の折れ線) だったが、
            #       軸足 2 項は 2026-08-24 に両タスクから撤去された。この分岐は
            #       将来の折れ線 param のために残してある (無いと NotImplementedError)。
            reward_term(cfg, params["term_name"], name).params[params["param_name"]] = params["knots"][-1][1]

        elif func is mdp.window_reward_weight:
            # 「start_step < step <= end_step の間だけ weight、外は 0」。窓の外 =
            # end_step より後が終状態なので 0。**現在どの呼び出し元にも 1 つも無い**が、
            # weak/middle 側に足されたときに黙って巻き戻らないよう先に書いてある
            # (窓が生きていると fine-tune の序盤だけ罰/報酬が復活する)。
            reward_term(cfg, params["term_name"], name).weight = 0.0

        elif func is mdp.kick_rate_gated_expansion:
            pin_expansion_gate(cfg, params, expansion_alpha)

        else:
            raise NotImplementedError(
                f"curriculum.{name} (func={getattr(func, '__name__', func)}) の終値の固定方法が "
                "pin_curricula_at_end に書かれていません。"
                "stage 2/3 は収束済み checkpoint からの fine-tune なので、"
                "巻き戻るランプが 1 本でも残っていると型が壊れます。"
                "固定の仕方をこの関数に足すか、巻き戻ってよい理由を "
                "UNPINNED_CURRICULUM_FUNCS に書いて除外してください。"
            )

        setattr(cfg.curriculum, name, None)
        pinned.append(name)

    # -- 検算: ランプが 1 本も残っていないこと ----------------------------- #
    #
    # 「新しいカリキュラム項を足したのにこの関数を直し忘れる」を起動時に落とすための
    # 検査。上のループが NotImplementedError で守っているので通常は到達しないが、
    # 除外リストの誤用 (報酬ランプを間違って入れる) はここでしか捕まらない。
    remaining = [
        n
        for n in sorted(dir(cfg.curriculum))
        if not n.startswith("_")
        and isinstance(getattr(cfg.curriculum, n, None), CurrTerm)
        and getattr(cfg.curriculum, n).func not in UNPINNED_CURRICULUM_FUNCS
    ]
    if remaining:
        raise AssertionError(f"固定されなかった curriculum 項が残っています: {remaining}")

    return pinned


def reward_term(cfg: "K1WalkKickEnvCfg", term_name: str, curr_name: str):
    """カリキュラムの ``term_name`` が指す報酬項を取り出す (無ければ落とす)。

    ``None`` の報酬項に weight を書いても ``AttributeError`` になるだけで
    「なぜ壊れたか」が読めないので、curriculum 項の名前を添えて先に落とす。
    """
    term = getattr(cfg.rewards, term_name, None)
    if term is None:
        raise AssertionError(
            f"curriculum.{curr_name} が指す報酬項 rewards.{term_name} がありません "
            "(報酬項だけ None にして curriculum 項を消し忘れている可能性)。"
        )
    return term


def pin_expansion_gate(cfg: "K1WalkKickEnvCfg", params: dict, alpha: float) -> None:
    """:func:`~..walk_kick.mdp.curriculums.kick_rate_gated_expansion` が α で動かす
    5 つの対象に、α を固定した値を直接書き込む。

    **値は全て term 自身の params から読む** (このモジュールの定数を再参照しない)。
    ゲートの設定を呼び出し元のレシピ関数
    (:func:`~..walk_inside_kick.walk_inside_kick_env_cfg._apply_inside_kick_recipe` など)
    で変えたときに、固定側だけ古い値のまま残る事故を構造的に防ぐため。

    α = 1 のとき ``approach_penalty`` の weight は **0** になる (``end_weight`` では
    ない)。あちらは ``approach_end_weight × fade × (1 − α)`` というクロスフェードで、
    全方位に届いた時点で「ボールに寄れ」の圧は ``ball_avoidance`` (寄るな) に
    完全に置き換わるため。``end_weight`` を書き込むと、収束済みポリシーに対して
    **互いに打ち消し合う 2 つの罰を同時に掛ける**ことになる。
    """
    def lerp(a: float, b: float) -> float:
        return a + (b - a) * alpha

    half_angle_range = params["half_angle_range"]
    dist_start, dist_end = params["dist_range_start"], params["dist_range_end"]
    heading_range = params["heading_halfwidth_range"]

    ball_event = getattr(cfg.events, params["ball_event_name"])
    ball_event.params["half_angle"] = lerp(*half_angle_range)
    ball_event.params["dist_range"] = (lerp(dist_start[0], dist_end[0]), lerp(dist_start[1], dist_end[1]))

    heading_half = lerp(*heading_range)
    getattr(cfg.commands, params["command_name"]).ranges.heading = (-heading_half, heading_half)

    # approach (寄れ) → avoidance (寄るな) のクロスフェード。fade は
    # min(now / approach_fade_iterations, 1) で、固定する時点では既に 1。
    # α = 1 では積が -0.0 になる。値は 0.0 と等価だがログ表示が紛らわしいので +0.0 で均す。
    getattr(cfg.rewards, params["approach_term_name"]).weight = params["approach_end_weight"] * (1.0 - alpha) + 0.0
    getattr(cfg.rewards, params["avoidance_term_name"]).weight = params["avoidance_end_weight"] * alpha
