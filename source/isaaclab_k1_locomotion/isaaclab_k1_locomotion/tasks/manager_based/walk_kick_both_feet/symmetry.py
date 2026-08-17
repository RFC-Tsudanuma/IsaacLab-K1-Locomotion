# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick_both_feet 系 (55/58 次元観測) の左右対称写像。rsl_rl の symmetry 用。

:mod:`..locomotion.mdp.symmetry` の walk_kick 系版。あちらは歩行タスクの 49 次元観測
専用で、こちらは **both_feet の 55 次元 policy / 58 次元 critic / 12 次元 action** を
扱う。dual 3 ファミリー (:mod:`..walk_kick_dual` / :mod:`..walk_weak_kick_dual` /
:mod:`..walk_middle_kick_dual`) の agents からも import される **共有層**なので、
鏡像規約はここ 1 箇所で管理すること。

mirror loss が有効な場合、PPO は以下を最小化する::

    || policy(mirror(obs)) - mirror(policy(obs)) ||^2

すなわち「観測を左右反転して与えたら、行動も左右反転して返す」対称な方策を促す。

なぜ walk_kick 系で mirror loss が定義できるようになったのか
------------------------------------------------------------
walk_kick 本体の観測はスロット 3 が **左足裏** (``sole_pos``) で、左右非対称な項が
policy 観測に入っていた。:mod:`.walk_kick_both_feet` がここを **ボール 3D 位置**
(原典 B-Human の観測表どおり) に差し替えたことで、55 次元すべてが左右対称な量に
なった。これで「右足でしか蹴らない」の残る 3 点目 (mirror loss が無効) に手を
付けられるようになった、というのがこのモジュールの位置付け。

**既存 checkpoint は使えない。** ``kick_foot_right_frac`` が run ごとに 0.99 / 0.01 へ
張り付く = 既に片足に収束しているので、対称化の出発点として不適。mirror loss を
入れたら **walk phase (stage 1) から回し直すこと**。

規約の出所
----------
関節の左右対応 permutation と符号は :mod:`..locomotion.mdp.symmetry` の
``_LEFT_JOINT_IDX`` / ``_RIGHT_JOINT_IDX`` / ``_JOINT_MIRROR_SIGN`` を **import して
再利用** している (書き写さない)。歩行タスクで実績のある規約と食い違わせないため。
非関節量の符号も同じ流儀:

* ``projected_gravity`` (x, y, z) → (x, −y, z)      [矢状面反転で横成分だけ反転]
* ``base_ang_vel`` (wx, wy, wz) → (−wx, wy, −wz)    [面外の軸まわりが反転]
* ``base_lin_vel`` (vx, vy, vz) → (vx, −vy, vz)     [critic のみ]

``gait_phase`` だけは walk_kick 系固有
--------------------------------------
歩行タスクの観測は左右両脚の位相 4 次元 ``[sin_L, cos_L, sin_R, cos_R]`` なので、
locomotion 側は **並べ替え** (45+[2,3,0,1]) で左右を入れ替えている。walk_kick 系は
左脚位相だけの 2 次元 ``[sin, cos]`` で、右脚位相は左 + π で一意に決まる
(:func:`~..walk_kick.mdp.observations.gait_phase_sincos` の docstring)。したがって
左右入れ替えは **位相 π シフト** = ``(sin, cos) → (−sin, −cos)`` になる。
停止時のゼロ埋め (``cmd_threshold`` 未満) は符号を掛けてもゼロのままなので整合する。

walk phase (stage 1) でも同じ写像で正しい
-----------------------------------------
walk phase はボール由来のスロットを歩行コマンドに差し替えるが、写像は変わらない:

* スロット 3 ``walk_command_xyz`` = (vx, vy, 0) → 速度指令の鏡像 (vx, −vy, 0)
* スロット 9 ``walk_command_yaw_dir`` = (cos wz, sin wz) → wz → −wz が
  (cos wz, −sin wz) = (x, −y) と一致
* スロット 12 ``walk_command_xy`` = (vx, vy) → (vx, −vy)
* スロット 10/11 ``zero_obs`` → ゼロのまま

つまり「スロット 3/9/11/12 の中身が何であれ、その量の鏡像規約は同じ」なので、
段ごとに写像を分ける必要は無い。

leading 次元非依存
------------------
:func:`mirror_last_dim` は **最終軸だけ**に permutation と符号を適用する。dual の
policy 観測は (N, H=100, 55) の 3 次元なので、1 フレーム版 (N, 55) と同じ関数が
そのまま効く。最終軸が 55 / 58 / 12 のどれでもなければ **即例外** にして、観測
レイアウトのドリフトを黙って通さない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# NOTE: private 名を跨いで import している。歩行タスクで実績のある符号規約を
#       **書き写さずに** 共有するのが目的で、複製すると片方だけ直されたときに
#       静かにずれる (ずれても学習が落ちず、対称性だけが壊れるので気づけない)。
#       locomotion 側でこの 3 定数の名前を変えるときは、ここも一緒に直すこと。
from ..locomotion.mdp.symmetry import (
    _JOINT_MIRROR_SIGN as _LOCO_JOINT_MIRROR_SIGN,
    _LEFT_JOINT_IDX as _LOCO_LEFT_JOINT_IDX,
    _RIGHT_JOINT_IDX as _LOCO_RIGHT_JOINT_IDX,
)

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "ACTION_DIM",
    "CRITIC_OBS_DIM",
    "POLICY_OBS_DIM",
    "compute_symmetric_states",
    "mirror_last_dim",
]


# --------------------------------------------------------------------------- #
# 1. スロットの種類ごとの局所写像 (純 Python)
#
# 各要素は (perm, sign)。``out[i] = in[perm[i]] * sign[i]`` という約束で、
# 最終軸のスロット内だけで閉じている。
#
# NOTE: この節と次節は **torch を一切使わない**。対合性・オフセットの検証を
#       torch 無しの環境 (CI / 手元) でも回せるようにするため。
# --------------------------------------------------------------------------- #
def _joint_mirror_map() -> tuple[list[int], list[float]]:
    """脚 12 関節の (perm, sign) を locomotion の定数から組み立てる。

    locomotion 側は ``out[LEFT] = data[RIGHT]`` / ``out[RIGHT] = data[LEFT]`` の後に
    ``_JOINT_MIRROR_SIGN`` を掛ける実装なので、``perm[left] = right`` /
    ``perm[right] = left`` と読み替えれば同じ写像になる。

    JOINT_NAMES_K1 の並び (``preserve_order=True`` で観測・行動とも同順):
    0-5 = Left の Hip_Pitch / Hip_Roll / Hip_Yaw / Knee_Pitch / Ankle_Pitch /
    Ankle_Roll、6-11 = Right の同順。符号は矢状面内で動く pitch / knee がそのまま、
    面外成分を持つ roll / yaw が反転。
    """
    num_joints = len(_LOCO_JOINT_MIRROR_SIGN)
    perm = [-1] * num_joints
    for left, right in zip(_LOCO_LEFT_JOINT_IDX, _LOCO_RIGHT_JOINT_IDX):
        perm[left] = right
        perm[right] = left
    if -1 in perm:
        raise ValueError(
            "関節 permutation に穴があります。locomotion.mdp.symmetry の"
            " _LEFT_JOINT_IDX / _RIGHT_JOINT_IDX が 12 関節を覆っていません。"
        )
    return perm, [float(s) for s in _LOCO_JOINT_MIRROR_SIGN]


_JOINT_PERM, _JOINT_SIGN = _joint_mirror_map()

# スロットの種類 → (perm, sign)。
_SLOT_MAPS: dict[str, tuple[list[int], list[float]]] = {
    # 位置・並進速度・向きベクトルの類。横 (y) 成分だけ反転。
    "vec3": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "vec2": ([0, 1], [1.0, -1.0]),
    # 角速度。矢状面反転では roll(x) と yaw(z) が反転し pitch(y) は不変。
    "ang_vel3": ([0, 1, 2], [-1.0, 1.0, -1.0]),
    # 左脚位相 (sin, cos)。左右入れ替え = 位相 π シフト = 両成分の符号反転。
    "gait_phase2": ([0, 1], [-1.0, -1.0]),
    # 左右に無関係なスカラー。
    "scalar": ([0], [1.0]),
    # 脚 12 関節 (joint_pos / joint_vel / prev_joint_request / action)。
    "joints12": (_JOINT_PERM, _JOINT_SIGN),
}


# --------------------------------------------------------------------------- #
# 2. 観測スロットの並び (純 Python)
#
# **K1WalkKickBothFeetPolicyCfg / K1WalkKickBothFeetCriticCfg の宣言順と 1 対 1**。
# ObsGroup は configclass のフィールド順に連結されるので、あちらの項を足す・消す・
# 並べ替えるときは必ずここも直すこと (下の _EXPECTED_OFFSETS が食い違いを検出する)。
#
# 要素は (項名, 次元, スロットの種類)。
# --------------------------------------------------------------------------- #
_POLICY_SLOTS: tuple[tuple[str, int, str], ...] = (
    ("projected_gravity", 3, "vec3"),
    ("base_ang_vel", 3, "ang_vel3"),
    ("ball_pos", 3, "vec3"),
    ("gait_phase", 2, "gait_phase2"),
    ("joint_pos", 12, "joints12"),
    ("joint_vel", 12, "joints12"),
    ("prev_joint_request", 12, "joints12"),
    ("gait_phase_factor_offset", 1, "scalar"),
    ("kick_direction", 2, "vec2"),
    ("target_kick_velocity", 1, "scalar"),
    ("ball_vel", 2, "vec2"),
    ("prev_ball_pos", 2, "vec2"),
)

# critic は policy と同じ 12 スロットの後ろに特権情報が付く
# (walk_kick_both_feet の critic は ``base_lin_vel`` 1 つだけ。ball_pos_rel は
#  スロット 3 を delay_steps=0 にして畳んである。あちらの定数ブロックの NOTE 参照)。
_CRITIC_EXTRA_SLOTS: tuple[tuple[str, int, str], ...] = (("base_lin_vel", 3, "vec3"),)
_CRITIC_SLOTS = _POLICY_SLOTS + _CRITIC_EXTRA_SLOTS

# 行動は脚 12 関節の目標角 (JointPositionActionCfg, JOINT_NAMES_K1, preserve_order=True)。
# scale はスカラー 0.5、use_default_offset=True で default 姿勢が左右対称なので、
# 関節量の写像がそのまま行動の写像になる。
_ACTION_SLOTS: tuple[tuple[str, int, str], ...] = (("joint_pos", 12, "joints12"),)


def _build_mirror_map(slots) -> tuple[dict[str, int], list[int], list[float]]:
    """スロット表から (オフセット表, 全体 perm, 全体 sign) を組み立てる。

    ``out[..., i] = x[..., perm[i]] * sign[i]``。
    """
    offsets: dict[str, int] = {}
    perm: list[int] = []
    sign: list[float] = []
    for name, dim, kind in slots:
        local_perm, local_sign = _SLOT_MAPS[kind]
        if len(local_perm) != dim or len(local_sign) != dim:
            raise ValueError(
                f"スロット '{name}' の次元 {dim} と写像 '{kind}' の長さ"
                f" {len(local_perm)} が食い違っています。"
            )
        offsets[name] = len(perm)
        base = len(perm)
        perm.extend(base + p for p in local_perm)
        sign.extend(local_sign)
    return offsets, perm, sign


_POLICY_OFFSETS, _POLICY_PERM, _POLICY_SIGN = _build_mirror_map(_POLICY_SLOTS)
_CRITIC_OFFSETS, _CRITIC_PERM, _CRITIC_SIGN = _build_mirror_map(_CRITIC_SLOTS)
_ACTION_OFFSETS, _ACTION_PERM, _ACTION_SIGN = _build_mirror_map(_ACTION_SLOTS)

POLICY_OBS_DIM = len(_POLICY_PERM)
"""Actor 観測の次元 (55)。dual では (N, H, 55) の最終軸。"""

CRITIC_OBS_DIM = len(_CRITIC_PERM)
"""Critic 観測の次元 (58 = 55 + base_lin_vel)。"""

ACTION_DIM = len(_ACTION_PERM)
"""行動の次元 (12)。"""

# --------------------------------------------------------------------------- #
# 手書きのスロット境界。上の表から機械導出した値との一致を import 時に検査する。
#
# 二重に書いているのは冗長に見えるが、これが **観測レイアウトのドリフト検出器**。
# 表の側だけを直して env cfg の宣言順を直し忘れる (あるいはその逆) と、次元の合計は
# 55 のままなのに写像だけが静かにずれる。ずれても学習は落ちず、対称性が効かなく
# なるだけなので、数値を見ても気づけない。
# --------------------------------------------------------------------------- #
_EXPECTED_POLICY_OFFSETS = {
    "projected_gravity": 0,
    "base_ang_vel": 3,
    "ball_pos": 6,
    "gait_phase": 9,
    "joint_pos": 11,
    "joint_vel": 23,
    "prev_joint_request": 35,
    "gait_phase_factor_offset": 47,
    "kick_direction": 48,
    "target_kick_velocity": 50,
    "ball_vel": 51,
    "prev_ball_pos": 53,
}
_EXPECTED_CRITIC_EXTRA_OFFSETS = {"base_lin_vel": 55}
_EXPECTED_POLICY_OBS_DIM = 55
_EXPECTED_CRITIC_OBS_DIM = 58
_EXPECTED_ACTION_DIM = 12


def _check_layout() -> None:
    """スロット表と手書きのオフセット・次元が一致しているかを検査する。"""
    if _POLICY_OFFSETS != _EXPECTED_POLICY_OFFSETS:
        raise ValueError(
            "policy 観測のスロット境界が想定と違います。"
            f" 表から導出: {_POLICY_OFFSETS} / 手書き: {_EXPECTED_POLICY_OFFSETS}。"
            " K1WalkKickBothFeetPolicyCfg の宣言順に合わせて両方を直してください。"
        )
    expected_critic = {**_EXPECTED_POLICY_OFFSETS, **_EXPECTED_CRITIC_EXTRA_OFFSETS}
    if _CRITIC_OFFSETS != expected_critic:
        raise ValueError(
            "critic 観測のスロット境界が想定と違います。"
            f" 表から導出: {_CRITIC_OFFSETS} / 手書き: {expected_critic}。"
        )
    for label, actual, expected in (
        ("policy", POLICY_OBS_DIM, _EXPECTED_POLICY_OBS_DIM),
        ("critic", CRITIC_OBS_DIM, _EXPECTED_CRITIC_OBS_DIM),
        ("action", ACTION_DIM, _EXPECTED_ACTION_DIM),
    ):
        if actual != expected:
            raise ValueError(f"{label} の次元が {actual} で、想定の {expected} と違います。")


def _check_involution(perm: list[int], sign: list[float], label: str) -> None:
    """鏡像写像が **対合** (2 回掛けると恒等) であることを全次元で検査する。

    鏡は 2 回掛ければ元に戻らなければならない。``out[i] = x[perm[i]] * sign[i]`` を
    2 回適用すると ``x[perm[perm[i]]] * sign[i] * sign[perm[i]]`` になるので、
    条件は **perm[perm[i]] == i** かつ **sign[i] * sign[perm[i]] == 1**。

    これを満たさない写像は「左右を入れ替えたつもりが元に戻らない」ので、mirror loss
    が矛盾した目標を与える (しかも損失は下がらないだけで学習は走ってしまう)。
    符号を 1 つ間違えただけでも落ちるように、import 時に必ず通す。
    """
    for i, p in enumerate(perm):
        if perm[p] != i:
            raise ValueError(
                f"{label}: perm が対合ではありません (perm[perm[{i}]] = {perm[p]} != {i})。"
            )
        if sign[i] * sign[p] != 1.0:
            raise ValueError(
                f"{label}: 符号が対合ではありません"
                f" (sign[{i}] * sign[{p}] = {sign[i] * sign[p]} != 1)。"
            )


_check_layout()
_check_involution(_POLICY_PERM, _POLICY_SIGN, "policy")
_check_involution(_CRITIC_PERM, _CRITIC_SIGN, "critic")
_check_involution(_ACTION_PERM, _ACTION_SIGN, "action")


# --------------------------------------------------------------------------- #
# 3. torch 側
#
# 最終軸の次元で写像を引く。55 / 58 / 12 のどれでもなければ例外。
# --------------------------------------------------------------------------- #
_MIRROR_MAPS: dict[int, tuple[list[int], list[float]]] = {
    POLICY_OBS_DIM: (_POLICY_PERM, _POLICY_SIGN),
    CRITIC_OBS_DIM: (_CRITIC_PERM, _CRITIC_SIGN),
    ACTION_DIM: (_ACTION_PERM, _ACTION_SIGN),
}
# 次元で写像を引く以上、3 つは相異なっていなければならない。同じ値になると dict の
# 後勝ちで別スロット構成の写像が黙って使われる (例: critic から特権情報を全部外して
# 55 次元にすると policy の写像が使われるが、そのときは偶然それが正しい……という
# 判断を無言でしてはいけない)。
if len(_MIRROR_MAPS) != 3:
    raise ValueError(
        "policy / critic / action の次元が重複しています"
        f" ({POLICY_OBS_DIM} / {CRITIC_OBS_DIM} / {ACTION_DIM})。"
        " 最終軸の次元で写像を引く設計が成り立たないので、グループ名で引く形に"
        " 作り替えてください。"
    )

# 鏡像を掛ける観測グループ。
#
# both_feet 系の ObservationsCfg はこの 2 つだけ (基底 locomotion の ObservationsCfg は
# policy のみ、both_feet が critic を足す)。**知らないグループが増えたら
# compute_symmetric_states が例外を投げる**。「複製したまま素通し」を既定にすると、
# 左右非対称な量を持つグループが足されたときに黙って壊れた拡張を学び続けることになる。
_MIRRORED_OBS_GROUPS = {"policy", "critic"}

# 反転に使う定数を (dim, device, dtype) ごとに一度だけ生成してキャッシュする。
# mirror loss は学習の epoch × mini_batch 回呼ばれるので、毎回 Python リストから
# torch.tensor(...) を作ると host→device コピーが効いてくる
# (locomotion.mdp.symmetry の _CONST_CACHE と同じ理由・同じ作り)。
_CONST_CACHE: dict = {}


def _mirror_consts(dim: int, device: torch.device, dtype: torch.dtype):
    key = (dim, device, dtype)
    consts = _CONST_CACHE.get(key)
    if consts is None:
        perm, sign = _MIRROR_MAPS[dim]
        consts = (
            torch.tensor(perm, device=device, dtype=torch.long),
            torch.tensor(sign, device=device, dtype=dtype),
        )
        _CONST_CACHE[key] = consts
    return consts


def mirror_last_dim(x: torch.Tensor) -> torch.Tensor:
    """最終軸を左右反転する。leading 次元は問わない。

    ``(N, 55)`` / ``(N, 100, 55)`` / ``(N, 58)`` / ``(N, 12)`` のいずれも同じ関数で
    扱える (dual の履歴観測がそのまま通る)。写像は最終軸の次元だけで決まる。

    Args:
        x: 最終軸が 55 (policy 観測) / 58 (critic 観測) / 12 (行動・関節量) の
            テンソル。

    Returns:
        左右反転した同じ形のテンソル。

    Raises:
        ValueError: 最終軸が想定のどれでもないとき。観測レイアウトが変わった
            (あるいは別のタスクから呼ばれた) サインなので、黙って通さない。
    """
    dim = int(x.shape[-1])
    if dim not in _MIRROR_MAPS:
        raise ValueError(
            f"walk_kick_both_feet.symmetry: 最終軸の次元 {dim} に対応する鏡像写像が"
            f" ありません (対応: {sorted(_MIRROR_MAPS)} = policy 55 / critic 58 /"
            " action 12)。観測グループの構成が変わっていないか確認し、変わっている"
            " なら _POLICY_SLOTS / _CRITIC_EXTRA_SLOTS を追随させてください。"
        )
    perm, sign = _mirror_consts(dim, x.device, x.dtype)
    return x[..., perm] * sign


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """観測・行動に左右対称変換を適用して拡張する (rsl_rl 用)。

    返すバッチは ``[元のサンプル, 左右反転したサンプル]`` の順に連結され、バッチ
    サイズが 2 倍になる。rsl_rl の PPO は

    * mirror loss: ``policy(mirror(obs)) ≈ mirror(policy(obs))`` を促す MSE 損失
    * data augmentation: 反転サンプルを学習バッチに追加

    のいずれ (または両方) にこの関数を使う。本リポジトリの walk_kick 系では
    **mirror loss のみ**を有効化している (設定は各 agents/rsl_rl_ppo_cfg.py)。

    rsl_rl 側の呼ばれ方 (rsl-rl-lib 3.1.2 = IsaacLab 2.3.2 の pin)
    --------------------------------------------------------------
    ``data_augmentation_func(env=..., obs=..., actions=...)`` の **キーワード呼び出し**。
    ``obs_type`` / ``is_critic`` のような引数は無く (2.3.1 までの旧 API にはあった)、
    obs は **全観測グループを含む TensorDict** で 1 回だけ渡る。obs だけ / actions だけ
    で呼ばれる経路があるので、両方 None 許容にしておくこと。シグネチャは
    :func:`..locomotion.mdp.symmetry.compute_symmetric_states` と同一に揃えてある。

    **観測は正規化前の生の値**で渡る。3.x では観測正規化がモデルの内側
    (``actor_obs_normalizer``) にあり、RolloutStorage には生の観測が入るため。
    したがってここでの符号反転は物理量に対する反転として正しい。

    **critic グループも鏡像にする** (locomotion 版・IsaacLab の参照実装との差)
    ------------------------------------------------------------------------
    locomotion 版と IsaacLab 同梱の参照実装 (anymal / cartpole) は
    ``obs.repeat(N)`` で全グループを複製したあと **policy だけ**を反転し、critic は
    複製のまま据え置く。ここでは 58 次元の写像を持っているので critic も反転する:

    * 現行設定 (``use_data_augmentation=False`` + ``use_mirror_loss=True``) では
      **どちらでも結果は同じ**。mirror loss は actor しか評価しないので、拡張バッチの
      critic グループは 1 度も読まれない (rsl_rl のソースで確認済み)。つまり
      切り替えのリスクが無い。
    * ``use_data_augmentation=True`` にすると拡張バッチがそのまま value loss に回る
      (critic は ``N × num_aug`` 行すべてで評価され、``values`` / ``returns`` /
      ``advantages`` が repeat される)。critic だけ複製だと「policy 観測は反転、
      critic 観測は元のまま」という **状態としてあり得ない行**を学ぶことになる。
      反転しておけば ``V(mirror(s)) → 同じ return`` という正しい対応になり、
      価値の対称性も一緒に学べる。
    * critic 観測に非対称な特権情報を足したら :func:`mirror_last_dim` が次元の変化で
      落ちる。黙って間違った拡張を続けるより落ちる方がよい。

    グループごとの分岐は名前ではなく **最終軸の次元**で写像を引くので、policy が
    (N, H=100, 55) の履歴になっている dual でもそのまま動く。**知らないグループが
    増えたら例外**にする (対称化の要否はその項を足した人にしか決められないため)。

    Args:
        env: 環境インスタンス (本関数では未使用だが rsl_rl の規約に合わせて受け取る)。
        obs: 観測の TensorDict。None なら観測を変換しない。
        actions: 行動テンソル (N, 12)。None なら行動を変換しない。

    Returns:
        ``(obs_aug, actions_aug)``。入力が None だった側は None を返す。
    """
    # -- 観測
    if obs is not None:
        groups = set(obs.keys())
        unknown = groups - _MIRRORED_OBS_GROUPS
        if unknown:
            raise ValueError(
                f"walk_kick_both_feet.symmetry: 未知の観測グループ {sorted(unknown)} が"
                f" あります (対応: {sorted(_MIRRORED_OBS_GROUPS)})。左右反転してよい量か"
                " どうかを判断したうえで _MIRRORED_OBS_GROUPS に足し、必要ならスロット表も"
                " 追加してください。"
            )
        batch_size = obs.batch_size[0]
        # 左右対称は 1 種類なのでバッチを 2 倍に拡張する
        obs_aug = obs.repeat(2)
        for group in groups:
            obs_aug[group][:batch_size] = obs[group][:]
            obs_aug[group][batch_size:] = mirror_last_dim(obs[group])
    else:
        obs_aug = None

    # -- 行動
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = mirror_last_dim(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
