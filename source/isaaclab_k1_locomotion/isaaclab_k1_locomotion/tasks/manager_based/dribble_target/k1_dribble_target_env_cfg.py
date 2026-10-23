
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""目標座標へ向けた「継続的なドリブル」タスク。

``dribble_env_cfg.py`` (階層型: 凍結歩行方策 + 専用上位方策、ワールド座標系の
固定方向へキックし続けるタスク) とは異なり、こちらは ``walk_kick`` タスク
(``fewa/walk_kick_dual_encoder`` ブランチ) と同じ **直接 fine-tune** 方式を採る:

ネットワークは ``K1FlatEnvCfg`` (Isaac-Velocity-Flat-K1-v0) で学習した歩行方策と
同一のアーキテクチャ (History CNN+MLP dual encoder)。単一ネットワークをそのまま
ドリブル報酬で fine-tune する (階層型の凍結低レベル方策は使わない)。
Stage 1 (``K1DribbleTargetWalkPhaseEnvCfg``, walk-phase): 観測ベクトルの形状は
Stage 2 と完全に同一に保ったまま、ボール/目標座標に依存するスロットを
:func:`mdp.observations.zero_obs` でゼロ埋めし、ドリブル関連の報酬・終了条件は
全て無効化して歩行のみを学習する。
Stage 2 (``K1DribbleTargetEnvCfg``): ボール・目標座標コマンドを有効化し、
ドリブル関連報酬を ``modify_reward_weight_linear`` カリキュラムで
weight=0 → 本番値へフェードインする。Stage 1 の checkpoint を
``--checkpoint`` で読み込んで再開する前提 (観測形状が同一なのでそのまま載る)。

目標は「目標座標 (ワールド xy, 定期的に再サンプリング)」であり、``dribble_env_cfg.py``
の ``kick_direction`` (固定方向) とは異なる。1回蹴って終わりではなく、ボールを
半径 ``safe_radius`` [m] 以内のコントロール下に置きながら目標座標へ進み続ける
連続タスクとして設計する (目標到達によるエピソード終了はしない)。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import (
    _COMMAND_THRESHOLD,
    K1FlatCommandCfg,
    K1FlatCriticHistoryCfg,
    K1FlatPolicyHistoryCfg,
    K1FlatStancePlaneCfg,
)
from .history_layout import DRIBBLE_TARGET_POLICY_EXTRA_TERMS
from .velocity_env_cfg import MySceneCfg
from .mdp.curriculums import (
    dribble_target_range_curriculum,
    log_ball_front_ratio,
    modify_gait_phase_cmd_threshold_linear,
    modify_reward_weight_linear,
)
from .mdp.commands import WaypointPose2dCommandCfg
from .mdp.events import replan_dribble_path, track_ball_front_ratio
from .mdp.obs_noise_models import SensorArtifactNoiseCfg
from .mdp.observations import (
    ball_pos_rel,
    ball_pos_rel_vision,
    ball_vel,
    ball_vel_rel_vision,
    foot_to_ball_pos_rel,
    target_pos_rel,
    zero_obs,
)
from .mdp.rewards import (
    approach_ball_progress,
    ball_control_radius_penalty,
    ball_progress_to_target,
    ball_proximity_to_target,
    ball_trajectory_deviation_penalty,
    ball_velocity_toward_target,
    impact_kick_reward,
    penalize_ball_clamping,
    push_ball_toward_target,
    robot_facing_ball,
    robot_progress_to_target,
    touch_ball,
)
from .mdp.terminations import ball_reached_target, ball_too_far, dribble_time_out


# NOTE (2026-10-22): 障害物 (旧 Level6〜8) は廃止した。障害物の回避は戦略側の経路計画 (pathplan_lib) に任せる。


# ---------------------------------------------------------------------------
# Observations: K1Flat の履歴観測 + 目標座標相対位置 (+ critic はボール相対位置/速度)
# ---------------------------------------------------------------------------
# 末尾に追加する項の並び順は history_layout.py の DRIBBLE_TARGET_POLICY_TERM_SPECS /
# DRIBBLE_TARGET_CRITIC_TERM_SPECS と正確に一致させること。
@configclass
class K1DribbleTargetPolicyHistoryCfg(K1FlatPolicyHistoryCfg):
    """K1FlatPolicyHistoryCfg (歩行履歴観測) + 目標座標相対位置・ボール相対位置・ボール速度 (末尾)。

    NOTE (2026-09-12): ball_pos_rel / ball_vel は当初 critic 専用 (特権情報) にしていたが、
    単一ネットワークの直接 fine-tune では歩行を決める Actor 自身がボール位置を見られないと
    ボールに向かって歩くことを学習できない (実際に touch_ball 報酬が全く伸びない不具合として
    発現した)。policy にも同じ2項を追加する。
    """

    # 2026-10-06: 実機の /localization_result 相当 (10〜20Hzの更新・遅延あり) を
    # 模すため、白色ノイズに加えてフレームホールド+遅延 (SensorArtifactNoiseCfg) を
    # 追加した (docs/dribble_deploy_handoff.md §3)。
    # NOTE (2026-10-07): 19時間学習が停滞した件の原因切り分け (Run A) で一時的に
    # hold_prob_range/delay_range を (0,0) にしていたが、真因は別 (obstacle_pos_rel
    # 除去の移行スクリプトのバグ) と判明したため、本来の値に戻した。
    target_pos_rel = ObsTerm(
        func=target_pos_rel,
        params={"command_name": "dribble_target"},
        noise=SensorArtifactNoiseCfg(
            noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
            hold_prob_range=(0.6, 0.8),
            delay_range=(0.0, 0.1),
        ),
    )
    # 2026-10-06: 実機 vision_filter の「ボールが足元に近いほど見失い/hold が増える、
    # 速度推定不能tickがある」挙動を模した観測関数に差し替え (docs/dribble_deploy_handoff.md
    # §6-2)。Criticは特権情報として素の ball_pos_rel/ball_vel のままにする (下記参照)。
    # NOTE (2026-10-07): 同じ理由で near_hold_prob/near_invalid_prob も一時的に 0 にして
    # いたが、真因が別 (移行スクリプトのバグ) と判明したため、observations.py の
    # デフォルト値 (near_hold_prob=0.3, near_invalid_prob=0.15) をそのまま使う
    # (params で上書きしない)。
    # NOTE (2026-10-11-d): 一時的な診断用。この近距離劣化 (near_distance=0.35m 以内で
    # ホールド確率30%・速度無効化確率15%) は「ボールは普段遠く、近いのは一瞬」という
    # 想定で設計されていたが、本日ボールのスポーン位置をロボットの進路上・近距離
    # (0.7〜0.9m) に集中させたことで、接触を試みる瞬間がまさにこの劣化が最も強く効く
    # 距離帯 (0.35m 以内) と重なってしまっている可能性がある。push_ball_toward_target
    # の方向がバラバラ・ball_progress_to_target が正味マイナスという症状と整合するため、
    # near_hold_prob/near_invalid_prob を far 側の値まで下げて距離依存の劣化を一時的に
    # 平坦化し、影響を確認する。確認が終わったら元の値 (デフォルト) に戻すこと。
    # NOTE (2026-10-18 再構築): 白色ノイズ (±0.05 / ±0.1) は ObsTerm の noise ではなく観測関数の
    # noise_range で掛ける。ObsTerm の noise は関数の戻り値の後に掛かるため、ホールドした値や
    # 「速度が推定できず厳密に 0」の値にもノイズが乗り、実機 vision_filter (同じ値の繰り返し /
    # ちょうど 0) と食い違うため (mdp/observations.py の _BallVisionTerm 参照)。
    # NOTE (2026-10-18 再構築): 10-11-d の診断用の平坦化 (近距離も遠距離と同じ 0.05 / 0.02) は
    # 「確認が終わったら戻す」ものだったので外し、仕様 (最新ラン) の値に合わせる:
    #   ホールド確率   近距離 0.6  → 遠距離 0.05
    #   速度欠損確率   近距離 0.4  → 遠距離 0.02
    # 距離に応じて near_distance〜far_distance の間を線形補間する。値は env.yaml で追えるよう全て明示する。
    # TODO: near_distance=0.35m / far_distance=2.0m は仮の値 (元の値が失われたため 2026-10-18 に確定扱い)。
    #       実機 vision_filter のログ (ボール距離ごとの mask / vel_valid の割合) が取れたら合わせ込むこと。
    ball_pos_rel = ObsTerm(
        func=ball_pos_rel_vision,
        params={
            "near_distance": 0.35, "far_distance": 2.0,
            "near_hold_prob": 0.6, "far_hold_prob": 0.05,
            "noise_range": 0.05,
        },
    )
    ball_vel = ObsTerm(
        func=ball_vel_rel_vision,
        params={
            "near_distance": 0.35, "far_distance": 2.0,
            "near_hold_prob": 0.6, "far_hold_prob": 0.05,
            "near_invalid_prob": 0.4, "far_invalid_prob": 0.02,
            "noise_range": 0.1,
        },
    )
    # 2026-10-06: obstacle_pos_rel は Policy から削除した (Critic専用に戻す)。
    # 実機に障害物検出センサが無く、最初のデプロイ版では Policy の入力に含めない方針
    # (docs/dribble_deploy_handoff.md §6-1、history_layout.py 参照)。
    # 2026-10-01: 「両足でボールを挟み込んで運ぶ」局所解対策。片足ずつボールとの
    # 相対位置を見せることで、挟み込みではなく個別の足によるボールコントロールを
    # 学習しやすくする。obstacle_pos_rel と同じく末尾追記、移行作業が必要。
    foot_left_ball_pos_rel = ObsTerm(
        func=foot_to_ball_pos_rel, params={"foot_body_name": "left_foot_link"}, noise=Unoise(n_min=-0.05, n_max=0.05)
    )
    foot_right_ball_pos_rel = ObsTerm(
        func=foot_to_ball_pos_rel, params={"foot_body_name": "right_foot_link"}, noise=Unoise(n_min=-0.05, n_max=0.05)
    )

@configclass
class K1DribbleTargetCriticHistoryCfg(K1FlatCriticHistoryCfg):
    """K1FlatCriticHistoryCfg + 目標座標相対位置・ボール相対位置・ボール速度 (特権情報)。"""

    target_pos_rel = ObsTerm(func=target_pos_rel, params={"command_name": "dribble_target"})
    ball_pos_rel = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)
    foot_left_ball_pos_rel = ObsTerm(func=foot_to_ball_pos_rel, params={"foot_body_name": "left_foot_link"})
    foot_right_ball_pos_rel = ObsTerm(func=foot_to_ball_pos_rel, params={"foot_body_name": "right_foot_link"})

@configclass
class K1DribbleTargetObservationsCfg:
    policy: K1DribbleTargetPolicyHistoryCfg = K1DribbleTargetPolicyHistoryCfg()
    critic: K1DribbleTargetCriticHistoryCfg = K1DribbleTargetCriticHistoryCfg()
    command: K1FlatCommandCfg = K1FlatCommandCfg()


# ---------------------------------------------------------------------------
# Scene: K1Flat のシーン + サッカーボール + 足-ボール接触センサ
# (dribble_env_cfg.py の K1DribbleSceneCfg と同一定義)
# ---------------------------------------------------------------------------
@configclass
class K1DribbleTargetSceneCfg(MySceneCfg):
    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # 2026-09-26: PhysX は「転がり抵抗 (rolling resistance)」を摩擦係数からは
                # 一切計算しない (滑り無しで転がる球は接触点で相対速度ゼロ = 摩擦力ゼロのため、
                # static/dynamic_friction をいくら上げても無関係)。そのため
                # physics_material の摩擦だけでは動画で見られた「一度触れると氷上のように
                # 転がり続ける」挙動を止められない。linear/angular_damping で明示的に
                # 減衰を与えて芝生・人工芝相当の「転がると徐々に失速する」挙動にする。
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            # dynamic_friction を明示しないと RigidBodyMaterialCfg のデフォルト値 0.5 が
            # 使われ、static_friction=1.0 と食い違っていた (滑り接触時の摩擦だけ弱くなる)。
            # 両方 1.0 に揃える (床側 velocity_env_cfg.py の physics_material も 1.0/1.0)。
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.11)),
    )

    contact_balls_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    contact_balls_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )

    # NOTE (2026-10-22): 障害物 (旧 Level6〜8) は廃止した。障害物の回避は戦略側の経路計画 (pathplan_lib) に任せる。


# ---------------------------------------------------------------------------
# Stage 2: フルタスク (ボールあり、ドリブル報酬フェードイン)
# ---------------------------------------------------------------------------
# 歩行の親 cfg。2026-10-18: K1FlatEnvCfg → K1FlatStancePlaneCfg に切り替えた。
# 59 次元に広げた best_walk (model_52500) を推論だけで動かした比較 (512 env × 600 step × 2 seed) で、
# StancePlane 系の方が追従誤差 (並進 0.25 vs 0.31 m/s、旋回 0.19 vs 0.24 rad/s) と両足空中率
# (0.3% vs 1.4%) が小さく、転倒率は同等 (2.7% vs 2.7〜3.3%) だったため。best_walk はこの系で
# 学習されている (frozen_policies/best_walk/README.md も StancePlane 系からの派生を指示)。
_WALK_BASE_CFG = K1FlatStancePlaneCfg


@configclass
class K1DribbleTargetEnvCfg(_WALK_BASE_CFG):
    scene: K1DribbleTargetSceneCfg = K1DribbleTargetSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1DribbleTargetObservationsCfg = K1DribbleTargetObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 2026-09-23: 全方位ドリブルを廃止し、レベルが上がるほど目標距離が遠ざかる
        # (最大10m) 前進特化型カリキュラムへ変更。長距離を運ぶには「物理的に move
        # しきれる制限時間」と「運んでいる途中で目標が再サンプリングされて振り出しに
        # 戻らないこと」の両方が要るため、episode_length_s と resampling_time_range を
        # セットで延長する (20s/1000step, 8-15s 再サンプリングのままだと Level5 の
        # 8-10m 到達前にエピソードが尽きるか、目標が入れ替わってしまう)。
        # 2026-10-23: 制限時間はエピソードごとに経路の長さから決める: max(30 秒, 長さ ÷ 0.5m/s + 5 秒)、上限 60 秒
        # (WaypointPose2dCommandCfg.time_limit_*、terminations.dribble_time_out)。time_out は進級の判定で失敗に数えるので、
        # 遅くても完走できる時間にする。曲がるコースでのドリブル中の平均の速さは 0.5〜0.6m/s (model_24600 の C 字・L 字で測定)。
        # episode_length_s は制限時間の上限にしておき、実際の打ち切りは time_out の終了条件が行う。
        self.episode_length_s = 60.0
        self.terminations.time_out = DoneTerm(
            func=dribble_time_out, time_out=True, params={"command_name": "dribble_target"}
        )

        # 目標座標コマンド (ワールド xy)。継続タスクなので定期的に再サンプリングし続ける
        # (目標到達によるエピソード終了はしない)。初期レンジは Level1 相当だが、
        # dribble_target_range_curriculum が構築時に即座に上書きするのでここは仮値でよい。
        # 2026-10-09: 直線一点ゴールだけでなく、経路形状 (path_shape、カリキュラムの
        # 各レベルが指定) に応じて L字・Z字(ジグザグ)等の中間ウェイポイントを自動生成
        # する WaypointPose2dCommand に差し替えた (mdp/commands.py 参照)。既存の
        # target_pos_rel/ball_progress_to_target/ball_reached_target 等は
        # pos_command_w (=現在のウェイポイント) を見るだけの実装なので変更不要。
        self.commands.dribble_target = WaypointPose2dCommandCfg(
            asset_name="robot",
            ranges=WaypointPose2dCommandCfg.Ranges(
                pos_x=(2.0, 2.5), pos_y=(0.0, 0.0),
            ),
            waypoint_reach_radius=0.3,
            debug_vis=True,
            # 2026-10-22: エピソードの途中で目標を全面的に引き直すのはやめた (O 字・8 の字を途中で打ち切らないため)。
            # 代わりに replan_dribble_path イベントが、未通過のウェイポイントだけを横にずらす。
            resampling_time_range=(1.0e6, 1.0e6),
            # 各レベルのエピソードの 20% は、同じ長さのランダムな折れ線 (実機の経路計画の出力に近い形) にする。
            random_polyline_prob=0.2,
            # 通過判定: 0.3m 以内に入る、または「入ってきた区間に垂直な線」を越える (行き過ぎて引き返す転倒の対策)。
            # 線を越えた方は、ウェイポイントから 1.0m 以内に限る (O 字・8 の字でループの内側を突っ切ったとき、
            # 離れたウェイポイントの線を次々に越えて「通過」になるのを防ぐ。幾何のテストで、制限なしでは最大 4.6m
            # 離れた通過が出て、1.0m に制限すると 0 になった。コースに沿って 0.5m 外を回る通過は 0.5〜0.9m で認められる)。
            pass_line_max_distance=1.0,
        )

        # NOTE (2026-09-20): base_velocity を BallFollowVelocityCommand (ボール追従の強制コマンド)
        # に差し替える方式は 2026-09-15〜09-18 に試したが、ロボットが Staging Point 近傍で
        # 速度指令がゼロへ収束し「寸止め」してしまう問題が残り (動的オフセットで軽減を試みたが
        # 解消せず)、ball_progress_to_target は改善しなかった。また「常にボール方向へ進め」を
        # 数式で強制する設計は、将来の障害物回避 (目の前に障害物があっても構わず直進してしまう)
        # とも本質的に相性が悪い。「どこへ歩くか」の決定権はコマンドではなく方策 (Policy) に
        # 委ねる。BallFollowVelocityCommand/Cfg の実装自体は mdp/commands.py に残してある。
        #
        # 2026-10-06: ただし base_velocity を K1FlatEnvCfg 由来の乱数コマンド (ExtremeVelocityCommand)
        # のまま放置すると、実機デプロイ上の問題が生じることが判明した
        # (docs/dribble_deploy_handoff.md §4): track_lin_vel_* 報酬が _phase2 でフェードアウトした
        # 後、このコマンドは報酬と無関係な乱数になる一方、観測フレームには毎ステップ
        # velocity_commands (乱数) として入り続け、かつ gait_phase の周波数もこの乱数コマンドの
        # 大きさに依存してしまう。実機では「ドリブル中に何を base_velocity として入れるか」が
        # 自明でなくなる。そこで Stage2 では base_velocity を定数ゼロに固定し、カリキュラムに
        # よる再ランダム化も止める。gait_phase を「常時オン (ゼロ化しない)」にする変更は
        # 下の curriculum.gait_phase_cmd_threshold で _phase2 にあわせて徐々に行う (後述の
        # NOTE 2026-10-06-b 参照、ここで即時切り替えると重大な不安定化を起こした)。
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None

        # ボールの reset (dribble_env_cfg.py の reset_ball と同一パターン)。
        # 2026-09-26: x上限を 1.0→0.5 に縮小 (下の Level1 距離レンジのコメント参照)。
        # ball_reached_target (半径1m以内で成功) を追加した際、reset_root_state_uniform は
        # デフォルト位置 (0.30) + env_origin + このオフセットでボールを配置するため、
        # 旧上限 1.0 だとボールが env_origin から最大 1.30m の位置に出現しうる一方、
        # dribble_target の Level1 目標は同じ env_origin 基準で 1.5m から始まっていた。
        # この2つの範囲は 0.2m しか離れておらず、ボール-目標のy成分が0に近い spawn では
        # 何もせずとも距離が1m未満になり「reset直後に即成功」してしまう
        # (実際、無学習のランダム方策で smoke test したところ ball_reached_target による
        # 終了が33.8%も発生し、これが原因と判明した)。これでは成功判定にも
        # dribble_target_range カリキュラムの成功率ゲートにも使えないため、x上限を
        # 0.5 (ボール最大 x=0.80m) に縮め、Level1 の pos_x 下限を 2.0 に上げることで
        # 最悪ケースでも 2.0-0.80=1.2m > 1.0m の余裕を確保する。
        # NOTE (2026-10-11): safe_radius緩和 (0.6→1.0m) と push_ball_toward_target 復活
        # (weight 0→1.0) を順に試したが、いずれも touch_ball/push_ball_toward_target の
        # 生値が数時間 (1000+ iteration) ほぼゼロのままで、接触そのものが全く発生しない
        # ("安全に見つめ続ける" 局所解への収束) ことが判明した。原因として
        # (a) robot の reset_base yaw が完全ランダム (-π〜π) でボールに背を向けて
        # 出現しうる、(b) ボールの y 方向の散らばり (±0.8m) が大きく、前進しても
        # 必ずしも進路上に無い、の2点が疑われる。(b) への対策として y の散らばりを
        # ほぼゼロに絞り、ボールを常に「最初の一歩の進路上」に集中させる。
        # x は 0.4〜0.6 (実位置 0.70〜0.90m) とし、(1) robot 自身の reset_base も
        # env_origin から ±0.5m ランダムなので、ロボットが robot_x=+0.5 付近に出現
        # してもボールと食い込まない十分な余裕 (最低0.2m) を確保しつつ、
        # (2) Level1 の目標下限 2.0m との差を最低 1.10m (2.0-0.90) に保ち、
        # 2026-09-26 に修正した「reset直後に即成功」バグを再発させない、という
        # 両方の安全マージンを満たす値を選んだ。
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "pose_range": {
                    "x": (0.4, 0.6), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("soccer_ball"),
            },
        )
        # NOTE (2026-10-11-b): ボールを原点付近の進路上に集中させただけでは効果が
        # 出なかった (50 iteration 経っても push_ball_toward_target が完全にゼロ)。
        # 原因は robot 自身の reset_base が K1FlatEnvCfg 由来のまま env_origin から
        # ±0.5m・yaw 完全ランダム (-π〜π) だったこと — ボールを原点付近に固定しても、
        # ロボット自身がそこから離れた位置・ランダムな向きで出現すれば、ボールが
        # 「進路上」にあるとは限らない。ボール位置の集中と対になる形で、ロボットの
        # 出現位置・向きもここで絞り、両者の相対ジオメトリを実際に固定する
        # (上の reset_ball の変更だけでは不完全だった、という自己訂正)。
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        }

        # 2026-10-22: エピソードの途中の経路の引き直し。実機の戦略は毎周期経路を計画し直すので、ウェイポイントが
        # 途中で動くことがある。env ごとに 2〜5 秒間隔で、確率 0.3 で未通過の途中のウェイポイントを横に 0.2〜0.8m ずらす
        # (最終目標は動かさない。mdp/commands.py の WaypointPose2dCommand.replan)。
        self.events.replan_dribble_path = EventTerm(
            func=replan_dribble_path,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            params={"command_name": "dribble_target", "probability": 0.3, "offset_range": (0.2, 0.8)},
        )

        # ドリブル報酬。Stage 1→2 のフェードインに合わせて weight=0 で登録する。
        self.rewards.ball_progress_to_target = RewTerm(
            func=ball_progress_to_target,
            weight=0.0,
            params={"command_name": "dribble_target"},
        )
        # 2026-10-12: 動画で確認した「その場足踏み」局所解への対策。base_velocity 固定
        # (定数ゼロ) + gait_phase 常時オンの組み合わせにより、ロボットは並進せずとも
        # feet_phase 等の歩行品質報酬を安全に稼ぎ続けられ、ロボット自身の前進を直接
        # 評価する報酬が一つも無かった (ball_progress_to_target/approach_ball_progress は
        # いずれもボールに関する量で、ロボットの並進そのものは見ていない)。
        # ball_progress_to_target と全く同じポテンシャル形式をロボット自身→目標座標の
        # 距離に適用する (mdp/rewards.py 参照)。
        self.rewards.robot_progress_to_target = RewTerm(
            func=robot_progress_to_target,
            weight=0.0,
            params={"command_name": "dribble_target"},
        )
        # 2026-10-09: 「強く蹴り飛ばして追いかける」局所解対策。safe_radius を2.0→0.6mに
        # 大幅縮小し、ボールから離れすぎることを強く罰する (メッシのような細かいタッチで
        # 足元に置き続けるドリブルを促す狙い)。
        # NOTE (2026-10-11): 0.6m は逆に「触れて少しでも転がると即座に半径を超えて
        # 罰せられる」恐怖を生み、touch_ball が30イテレーション連続で完全にゼロ
        # (一度も接触しない) という「寸止め」局所解を誘発していたと判明した
        # (approach_ball_progress だけが微増していた)。1.0m に緩和し、「突いて少し
        # 転がってもすぐ追いつけば罰を受けない」安全マージンを確保する。
        self.rewards.ball_control_radius_penalty = RewTerm(
            func=ball_control_radius_penalty,
            weight=0.0,
            params={"safe_radius": 1.0},
        )
        # ロボット→ボール距離の1ステップ減少量 (ポテンシャル形式) を密な接近報酬として与える。
        # ball_control_radius_penalty は 2m を超えるまで無反応 (勾配ゼロ) なので、単独では
        # 「積極的にボールへ寄る」動機が生まれない。この項が接近の主動力 (アクセル)、
        # ball_control_radius_penalty は離れすぎ防止の安全装置 (ブレーキ) として機能する。
        # ポテンシャル形式のため、エピソード総和は d_初期 - d_最終 に telescope する
        # (居座って往復しても差し引きゼロ、近接後は縮める余地が無いので farm 不可)。
        self.rewards.approach_ball_progress = RewTerm(
            func=approach_ball_progress,
            weight=0.0,
            params={"max_progress_per_step": 0.1},
        )
        self.rewards.touch_ball = RewTerm(func=touch_ball, weight=0.0)
        # 2026-10-06: 動画で「ロボットが後ろ向きで歩いてドリブルする」挙動が確認された。
        # track_ang_vel_z_exp/track_lin_vel_* が _phase2 でフェードアウトした後、
        # このタスクには「ロボットがどちらを向いて歩くか」を評価する報酬が一切無く
        # (ball_progress_to_target 等はボール自体のワールド座標速度/位置だけを見ており、
        # ロボットの向きは無関係)、後ろ向き歩行を抑止する圧力が存在しなかったことが
        # 原因と判断した。既存の dribble_env_cfg.py (階層型キックタスク) で実績のある
        # robot_facing_ball (ロボット Trunk 正面がボール方向を向いているほど大きい
        # [0,1] の密な報酬) をそのまま再利用して追加する (weight はキックタスクの
        # 0.4 よりやや強め、他の密な報酬と同等のオーダーにするため 2.0)。
        self.rewards.robot_facing_ball = RewTerm(
            func=robot_facing_ball,
            weight=0.0,
            params={"min_distance": 0.15},
        )
        # 2026-09-25: 「ボールの目の前で安全に足踏みし続け、押しに行かない」局所解への対策の
        # 第3弾 (termination_penalty 緩和・touch_ball 増量だけでは解消しなかったため)。
        # touch_ball (触れるだけで一律報酬、向き不問) と ball_progress_to_target
        # (ポテンシャル形式、押した結果が遅れて反映される) の間を埋める、接触した瞬間
        # その場で「目標方向へ押せているか」を評価する密な信号。速度ではなく接触力の向きを
        # 見るため、ball_progress_to_target の旧バグ (生の速度を報酬にして強打を誘発) の
        # 再発はしない (max_reward=1.0 で出力自体をクランプ)。
        self.rewards.push_ball_toward_target = RewTerm(
            func=push_ball_toward_target, weight=0.0, params={"command_name": "dribble_target", "max_reward": 1.0}
        )
        # 2026-09-25 追加: ball_progress_to_target (差分のみ評価) を補う絶対位置ベースの
        # 密な報酬。近づくほど、また近づいた状態を維持するほど毎ステップ報酬が積み上がる
        # ([0,1] に指数減衰でクランプ済みなので farm/暴走のリスクは低い)。
        self.rewards.ball_proximity_to_target = RewTerm(
            func=ball_proximity_to_target,
            weight=0.0,
            params={"command_name": "dribble_target", "max_reward": 1.0, "distance_scale": 2.0},
        )
        # 2026-09-26: 「目標半径1m以内へ運べたら成功」を明示的な終了条件+報酬として追加。
        # DoneTerm と RewTerm の両方が同じ ball_reached_target 関数・同じ
        # distance_threshold を参照するため、判定基準がズレることはない
        # (mdp/terminations.py 参照)。weight は他の密な報酬 (ball_progress_to_target
        # 等、1エピソードでの累積が概ね数十〜百程度) に対して圧倒的すぎない
        # 100.0 に設定 (外部AI提案の1000.0は他項に対して3〜10倍のスケール差になり、
        # PPOのvalue推定を不安定化させるリスクがあると判断して縮小)。他の Stage2
        # 報酬と同じくフェードイン方式にする (Stage1 checkpoint 引き継ぎ直後に
        # いきなり満額が乗らないようにするため)。
        # NOTE (2026-10-10): 2026-10-09 に成功判定を半径1.0m→0.3mに厳格化したが、
        # base_velocity固定+フルノイズ等と組み合わせた学習で11時間以上Level1から
        # 全く進まない停滞が発生したため、半径1.0mに戻した (0.3mでは成功が
        # ほぼ発生せず、成功率ベースのカリキュラム進級判定にも ball_reached_target_bonus
        # (weight=300) にも学習信号が入らなくなっていた可能性がある)。
        # NOTE (2026-10-11): ball_reached_target をボールのみの判定からボール+ロボット
        # 両方が半径内に入る判定へ変更 (mdp/terminations.py 参照)。ボールを蹴って
        # 目標へ転がすだけで成功していた「キック&放置」局所解を防ぎ、ロボット自身も
        # 目標まで移動することを要求する (ドリブルとして成立させるため)。
        self.rewards.ball_reached_target_bonus = RewTerm(
            func=ball_reached_target,
            weight=0.0,
            params={"command_name": "dribble_target", "distance_threshold": 1.0},
        )
        self.terminations.ball_reached_target = DoneTerm(
            func=ball_reached_target,
            params={"command_name": "dribble_target", "distance_threshold": 1.0},
        )

        # NOTE (2026-10-22): 障害物 (旧 Level6〜8) は廃止した。障害物の回避は戦略側の経路計画 (pathplan_lib) に任せる。

        # 2026-10-01: 「両足でボールを挟み込んでペンギン歩きのように運ぶ」局所解への対策。
        # touch_ball (片足でも満額) だけでは挟み込みを防げなかった (挟めば ball_too_far も
        # 取りこぼしもほぼ起きず、綺麗に蹴るより安全に稼げてしまうため)。
        self.rewards.penalize_ball_clamping = RewTerm(func=penalize_ball_clamping, weight=0.0)
        # ボールの目標方向への速度成分 (旧 kick タスクの既存関数を dribble_target 用に
        # 再利用)。ball_progress_to_target (ポテンシャル形式、1ステップ最大 ±2.0) を
        # 補い、「結果として前に進んだか」だけでなく「今ちゃんと転がっているか」を見る。
        # 外部提案の weight=10.0 は、このタスクで過去に何度も問題になった
        # 「生の速度ベースの報酬を大きくしすぎて暴走する」パターン (crab walk 事件等) を
        # 再発しかねないため、同程度の密な信号である push_ball_toward_target と同じ
        # 3.0 に抑える。
        # 2026-10-09: max_speed を1.5→0.8に縮小。この報酬は v_along/max_speed を[0,1]に
        # クランプするだけなので、max_speedを超える速度を出しても追加の得にはならない
        # (= 強く蹴るインセンティブを除去し、ゆっくり転がす細かいタッチを促す)。
        self.rewards.ball_velocity_toward_target = RewTerm(
            func=ball_velocity_toward_target,
            weight=0.0,
            params={"command_name": "dribble_target", "max_speed": 0.8},
        )
        # 接触中の足速度とボール速度の内積 (キックのインパクトの質)。[0,1] にクランプ済み
        # なので、速度がほぼゼロの静的な挟み込みでは発火せず、動的なキック動作だけを
        # 後押しする。
        self.rewards.impact_kick_reward = RewTerm(func=impact_kick_reward, weight=0.0)
        # ボールが目標方向から横に逸れていく速度成分 (横ブレの軌道) を罰する。
        self.rewards.ball_trajectory_deviation_penalty = RewTerm(
            func=ball_trajectory_deviation_penalty,
            weight=0.0,
            params={"command_name": "dribble_target", "min_speed": 0.5},
        )

        # 2026-09-23: 「強く蹴って即転倒しリセットした方が地道に運ぶより得」という
        # reward hacking (動画で「ボールに触れた瞬間に膝から崩れ落ちる」形で観測) への対策。
        # ball_progress_to_target をポテンシャル形式に変えて瞬間速度への旨味を無くすのに
        # 加えて、転倒ペナルティ自体も引き上げて「わざと転ぶ」の期待値を下げ、姿勢の乱れ
        # (突進的な体幹の傾き) への抑制も強化する二段構え。
        # 2026-09-24 再調整: -500 は「強く蹴って転倒レリーズ」ハックの抑止には効いたが
        # (base_height 転倒率 ~2% まで低下、episode の97%超が time_out まで完走)、効きすぎて
        # 今度は逆方向の局所解 — 「ボールの目の前で安全に足踏みし続け、押しに行かない」—
        # を誘発していた。feet_phase/feet_parallel_to_ground/zmp_stability 等の姿勢系報酬は
        # ボールに触れなくても足踏みだけで確実に稼げる一方、ball_progress_to_target
        # (最大 +2.0/step) の不確かな見返りと -500 のリスクを比べると「押しに行かない」方が
        # 期待値で勝ってしまっていたと考えられる。-300 まで戻し (元の -200 よりは重く、
        # reward hacking 再発は防ぎつつ)、touch_ball も 0.3→1.0 に引き上げてボールへ
        # 踏み込む動機を後押しする (ball_progress_to_target 自体はポテンシャル形式なので
        # 「触れるだけで押さない」を単独で得にはしない = 旧来のtouch_ball局所解は再発しにくい)。
        #
        # 2026-09-27 バグ修正: termination_penalty は元々 mdp.is_terminated (rough_env_cfg.py)
        # を使っており、これは time_out 以外の「全ての」termination で無条件に発火する。
        # 昨日追加した ball_reached_target (成功) も time_out=False で登録したため、
        # 成功した瞬間に ball_reached_target_bonus (+100、フェード中はさらに小) と同時に
        # この -300 も発火し、成功の正味報酬が実質マイナスになっていた
        # (「ボールを運んでも損」という致命的な符号ミス)。mdp.is_terminated_term に切り替え、
        # term_keys で「本当の失敗」系 termination だけを対象にし、ball_reached_target を除外する。
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-300.0, # 元 -200 (rough_env_cfg.py) → -500 → -300
            params={"term_keys": ["base_contact", "base_height", "ball_too_far"]},
        )
        self.rewards.flat_orientation_l2.weight = -40.0 # 元 -20 (rough_env_cfg.py)

        # 終了条件: ボールを見失った (蹴り飛ばした/遠くに逸れた) 場合のみ失敗として打ち切る。
        self.terminations.ball_too_far = DoneTerm(func=ball_too_far, params={"max_distance": 4.0})

        # 2026-10-18: 評価指標「ボールが body yaw 座標で方位角 ±45° 以内にある時間の割合」
        # (robot_facing_ball の扱いを決めるための監視用。報酬には使わない)。毎ステップの集計を
        # interval イベントで行い、エピソード終了時に Curriculum/ball_front_ratio としてログに出す。
        self.events.track_ball_front_ratio = EventTerm(
            func=track_ball_front_ratio,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"max_azimuth_deg": 45.0},
        )
        self.curriculum.ball_front_ratio = CurrTerm(func=log_ball_front_ratio)

        # Stage 1→2 フェードイン。num_steps_per_env=48 (rsl_rl_ppo_cfg.py) を基準に
        # 500 iteration かけて 0 → 本番 weight へ線形補間する (walk_kick の _phase2 を踏襲)。
        _spi = 48
        _phase2 = {"start_step": 0, "end_step": 500 * _spi}

        # NOTE (2026-10-06-b): gait_phase の cmd_threshold を Stage1 の値 (_COMMAND_THRESHOLD、
        # コマンドゼロで位相もゼロ化=歩行ポリシーが実際に経験した挙動) から「常時オン」相当の
        # 到達不能な値へ、_phase2 と同じ期間で徐々に遷移させる。base_velocity を定数ゼロに
        # 固定した直後にこれを cmd_threshold=-1.0 へ即時切り替えたところ、Stage1 の
        # ポリシーが一度も経験していない「位相≠0 かつ velocity_commands=0」の組み合わせに
        # いきなり晒され、base_height 転倒率が ~99.5% に張り付く重大な不安定化が起きた
        # (track_lin_vel_* の重み修正だけでは解消しなかった)。段階的に慣らすことで、
        # Stage1 直後の挙動をそのまま引き継いでから新しい gait_phase 挙動へ移行させる。
        self.curriculum.gait_phase_cmd_threshold = CurrTerm(
            func=modify_gait_phase_cmd_threshold_linear,
            params={"start_threshold": _COMMAND_THRESHOLD, "end_threshold": -1.0, **_phase2},
        )

        # NOTE (2026-09-20): 「base_velocity をボール方向へ強制する」方式 (BallFollowVelocityCommand,
        # 2026-09-15〜09-18) は Staging Point 近傍で速度指令がゼロへ収束し「寸止め」する問題が
        # 解消できず、かつ将来の障害物回避とも相性が悪いため廃止した。「どこへ歩くか」の決定権を
        # 方策 (Policy) に返すため、base_velocity の追従報酬 (track_lin_vel_*/track_ang_vel_z_exp)
        # を再びフェードアウトさせる。Stage1 で「転ばずに歩く」スキルは習得済みなので、
        # 追従報酬が消えた後は ball_pos_rel/target_pos_rel 観測とボール関連報酬だけを頼りに
        # 「ボールを運ぶと報酬が入り続ける」ことを自力で発見させる。
        #
        # 2026-10-06 修正 (重大): base_velocity を定数ゼロに固定した (上の §4 対応) ことで、
        # start_weight を 1.5/2.4/4.8 のまま残すと「ゼロ速度を追従=直立静止」を強く奨励する
        # 状態になり、これが常時回り続ける gait_phase (1.8Hz, 「今まさに一歩踏み出すべき」を
        # 示唆する信号) と正面から矛盾する。実際にこの状態で学習を再開した結果、
        # base_height 転倒率が ~99.5% に張り付いて全く回復しない重大な不安定化が発生した
        # (このプロセス再起動のたびに _phase2 のフェードが 0 からやり直しになる関係で、
        # 複数回の再起動によってこの「矛盾ゾーン」に繰り返し晒されたことも悪化要因)。
        # base_velocity はもはや無意味な定数なので、その追従報酬をフェードさせる意味自体が
        # 無い。start_weight も 0.0 にして、Stage2 では最初から完全に無効化する。
        # NOTE (2026-10-07): Run B の原因切り分けで一時的に 1.5/2.4/4.8 からのフェードに
        # 戻していたが、真因は別 (移行スクリプトのバグ) と判明し base_velocity 固定も
        # 復元したため、これらも start_weight=0.0 (完全無効化) に戻す。
        self.curriculum.track_lin_vel_sharp_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "track_lin_vel_xy_exp", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )
        self.curriculum.track_lin_vel_coarse_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "track_lin_vel_xy_coarse", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )
        self.curriculum.track_ang_vel_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "track_ang_vel_z_exp", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )

        # weight目安 (2026-09-23 修正): 2026-09-20 の「weight=40」判定は、生の速度 [m/s] を
        # ±max_speed=1.5 でクランプしただけの量 (1ステップ最大 1.5×40=60) を、距離差分を
        # ±0.1m にクランプした approach_ball_progress のポテンシャル量 (1ステップ最大
        # 0.1×40=4.0) と同列に扱ってしまった単位の見落としだった (2026-09-22 に weight=12
        # へ緊急downしたが、根本原因である「瞬間速度をそのまま報酬にする」設計自体は
        # 残っていたため、動画で「ボールに触れた瞬間に膝から崩れ落ちる」reward hacking
        # (強く蹴り飛ばして即転倒レリーズした方が地道に運ぶより得になる) が観測された)。
        # 2026-09-23: ball_progress_to_target 自体を approach_ball_progress と同じ
        # ポテンシャル形式 (距離差分、±0.05m/step クランプ) に書き換えたので、
        # 今度こそ単位が揃った状態で weight=40.0 (approach_ball_progress と同格) に戻す。
        self.curriculum.ball_progress_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "ball_progress_to_target", "start_weight": 0.0, "end_weight": 40.0, **_phase2},
        )
        # NOTE (2026-10-11-e / 2026-10-12 訂正): 「報酬の断捨離」実験で接近・接触・
        # 姿勢・速度系を一旦全てゼロにしたが、6時間回しても変化せず、動画で確認した
        # ところロボットが一歩も並進していなかった (その場足踏みのみ)。真因は
        # ball_progress_to_target 以外の報酬が多すぎたことではなく、ロボット自身の
        # 並進を直接評価する報酬が一つも無かったこと (robot_progress_to_target を
        # 新設、下記 robot_progress_weight 参照) だったため、断捨離した5項目は
        # 全て元の値に戻す。
        self.curriculum.ball_control_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "ball_control_radius_penalty", "start_weight": 0.0, "end_weight": -2.0, **_phase2},
        )
        # weight目安: 1ステップの進捗は dt=0.02s・接近速度~0.9m/s で ~0.018m 程度しかなく、
        # weight=1 だと他の報酬項 (feet_phase≈1.3 等) に対して桁違いに弱く実質無視される。
        # 旧キックタスクでの同関数の実績値 (weight=40.0) に合わせる
        # (2026-09-14: weight=1.0 で20000iter回した結果、この項がノイズレベル (~1e-4) の
        # まま全く成長せず、ボールへの接近行動が学習されなかったことを踏まえた修正)。
        self.curriculum.approach_ball_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "approach_ball_progress", "start_weight": 0.0, "end_weight": 40.0, **_phase2},
        )
        # 2026-10-06: 「後ろ向きドリブル」対策 (robot_facing_ball の登録箇所を参照)。
        # dribble_env_cfg.py (キックタスク) での実績値 0.4 よりやや強め、他の密な報酬
        # (push_ball_toward_target 等) と同等のオーダーにする。
        # 2026-10-07: 診断の結果 (Run A〜D)、19時間の学習停滞の真因は robot_facing_ball
        # ではなく、obstacle_pos_rel 除去の移行スクリプトのバグ (レイアウト誤認識で
        # 正規化統計・mlp重みの大半を誤った位置で破壊していた) と判明したため、
        # end_weight を本来の 2.0 に戻す (migrate_checkpoint_remove_policy_obstacle_obs.py
        # 参照、修正・再検証済み)。
        self.curriculum.facing_ball_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "robot_facing_ball", "start_weight": 0.0, "end_weight": 2.0, **_phase2},
        )
        # touch_ball は元々「ボールに触れ続けること」自体が目的化する局所解 (立ち止まって
        # 交互に小突いて稼ぐ) を誘発しやすいため小さく (0.3) 留めていたが、2026-09-24 に
        # termination_penalty を強化した結果、逆に「ボールへ踏み込むリスクを避けて手前で
        # 足踏みし続ける」別の局所解を誘発してしまった。ball_progress_to_target が既に
        # ポテンシャル形式 (押して進めない限り報酬が出ない) なので「触れるだけ」を単独で
        # 得にしてしまう心配は薄いと判断し、ボールへ踏み込む動機付けとして 1.0 に引き上げる。
        # 2026-10-03: 「片足ずつ超高速で交互にボールへ触り続けるペンギン歩き」局所解への
        # 対策。penalize_ball_clamping は両足同時接触のみを検知するため、片足ずつ触れば
        # 罰則を回避しつつ touch_ball/push_ball_toward_target/impact_kick_reward という
        # 接触系の「お小遣い」を毎ステップ満額で稼ぎ続けられてしまっていた。ボール自体の
        # 前進を評価する ball_progress_to_target (40.0, ポテンシャル形式) /
        # ball_velocity_toward_target (3.0) / approach_ball_progress (40.0, ポテンシャル
        # 形式) だけで「蹴って離れて追いかける」動機付けは十分と判断し、接触系3項
        # (touch_ball, push_ball_toward_target, impact_kick_reward) の end_weight を
        # 0.0 にして完全に無効化する。
        # NOTE (2026-10-11-c): 診断用に一時的に weight=0.05 にして生の接触有無を確認した
        # 結果 (30 iteration): touch_ball は約25%のイテレーションで非ゼロ (ごく短時間・
        # 微弱な接触は起きている) が、push_ball_toward_target (方向フィルタ付き) は
        # 約6%でしか非ゼロにならず、接触の大半が目標方向への意図的な押しになっていない
        # ことを確認した。確認が終わったので 0.0 に戻す (ペンギン歩き局所解の再発リスクを
        # 避けるため、touch_ball 自体は引き続き無効化のまま)。
        self.curriculum.touch_ball_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "touch_ball", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )
        # weight目安: push_ball_toward_target は [0, max_reward=1.0] にクランプ済みの密な
        # 信号なので、touch_ball (weight=1.0) と同格〜やや強めに置き、ball_progress_to_target
        # (weight=40、ただしポテンシャル形式で疎) / approach_ball_progress (weight=40、疎) に
        # 対して「押しに行く」きっかけを作る程度の中間的な強さに留める。
        # NOTE (2026-10-11): safe_radius 緩和 (0.6→1.0m) だけでは touch_ball 等が
        # 全て weight=0 のままで「触れる直接的な得」が無く、約1時間半回しても
        # ball_progress_to_target/ball_velocity_toward_target/成功率のいずれも横ばいの
        # ままだった (base_height は健全、mean_noise_std も正常、feet_phase はむしろ
        # 向上しており、ネットワーク飽和ではなく「安全に見つめ続ける」局所解への収束と
        # 判断)。「押しに行く」ことそのものに報酬を与えるため、end_weight を 0.0→1.0 に
        # 復活させる。ただし一度に変える箇所は1つに絞るため、他の接触系
        # (touch_ball, impact_kick_reward) は今回は 0.0 のまま据え置く
        # (ペンギン歩き再発の監視がしやすいよう、原因を切り分けられる形にする)。
        self.curriculum.push_ball_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "push_ball_toward_target", "start_weight": 0.0, "end_weight": 1.0, **_phase2},
        )
        # 2026-09-29 バグ修正 (reward hacking): 2026-09-27_03-46-13 の学習 (成功率33%で
        # 頭打ち) を分析した結果、ball_proximity_to_target (絶対距離ベースの密な報酬、
        # ポテンシャル形式ではない) が「目標半径1mのすぐ手前 (1.1m 付近) に留まり続ける」
        # ことを実質タダで稼げてしまう局所解を誘発していたと判明した。ball_reached_target
        # で成功終了すると、その先本来得られたはずの proximity 報酬 (残り時間 × 密な値) を
        # 丸ごと失うため、「ゴールに入れず寸止めする」方がエピソード内の期待報酬で上回って
        # しまっていた (touch_ball が0.6〜0.7%しかなく、ball_progress_to_target が正味ゼロ
        # という実測値と整合)。ball_progress_to_target (距離差分のポテンシャル形式) だけで
        # 「近づくほど得」は理論上十分カバーできるため、この項は完全に無効化する
        # (weight=0 のまま固定。start/end とも 0.0 なのでフェードも実質無意味)。
        self.curriculum.ball_proximity_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "ball_proximity_to_target", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )
        # 2026-09-29: proximity 廃止でエピソード終盤の「タダ稼ぎ」手段が消えるため、成功
        # ボーナス自体の相対的な重みは自然に増す。それでも念のため 100→300 に引き上げ、
        # 「早くゴールする」ことの魅力をさらに後押しする。500 も検討したが、one-time の
        # 巨大な報酬スパイクは PPO の value 推定を不安定化させるリスクがある
        # (2026-09-26 に 1000→100 へ縮小した際と同じ懸念) ため、中間的な値に留める。
        self.curriculum.ball_reached_bonus_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "ball_reached_target_bonus", "start_weight": 0.0, "end_weight": 300.0, **_phase2},
        )
        # 2026-10-01: 挟み込み局所解対策の4項。他の密な報酬 (push_ball_toward_target=3.0,
        # ball_proximity 系と同程度) とスケールを揃えて暴走リスクを抑える。
        self.curriculum.clamping_penalty_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "penalize_ball_clamping", "start_weight": 0.0, "end_weight": -10.0, **_phase2},
        )
        self.curriculum.ball_velocity_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "ball_velocity_toward_target", "start_weight": 0.0, "end_weight": 3.0, **_phase2},
        )
        # 2026-10-12: ロボット自身の並進に直接報酬を与える新項。「その場足踏み」局所解
        # 対策 (robot_progress_to_target 登録箇所・mdp/rewards.py 参照)。weight は
        # feet_phase の実測値 (定常的に ~1.3〜1.4/step相当) に対して明確に優位になる
        # よう、approach_ball_progress の実績値 (weight=40 では歩行速度0.9m/s相当でも
        # ~1程度にしかならず不十分だった) より大きく、150 に設定する。この checkpoint
        # が一度も経験していない全く新しい報酬項なので facing_ball_weight と同様
        # _phase2 でフェードインさせるが、「その場足踏み」という強固な局所解を崩すには
        # 通常の500iterationでは緩すぎる懸念があるため、150iteration (_phase2の3割) の
        # 短縮フェード専用ウィンドウを使う。
        self.curriculum.robot_progress_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={
                "term_name": "robot_progress_to_target", "start_weight": 0.0, "end_weight": 150.0,
                "start_step": 0, "end_step": 150 * _spi,
            },
        )
        self.curriculum.impact_kick_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={"term_name": "impact_kick_reward", "start_weight": 0.0, "end_weight": 0.0, **_phase2},
        )
        self.curriculum.trajectory_deviation_weight = CurrTerm(
            func=modify_reward_weight_linear,
            params={
                "term_name": "ball_trajectory_deviation_penalty", "start_weight": 0.0, "end_weight": -5.0, **_phase2,
            },
        )

        # 2026-09-23: 全方位ドリブルを廃止し、レベル上昇とともに目標距離が遠ざかる
        # (最大10m) 前進特化型カリキュラムへ変更。
        #
        # 2026-09-26: ゲート指標を「ボールと目標座標の距離」の生値EMAから、
        # 「完走したエピソードのうち ball_reached_target (半径1m以内) で成功できた
        # 割合」の失敗率EMAに変更した。前者は成功即リセット方式と併用すると、大半の
        # 環境が常に移動中 (距離>1m) のまま定常化してしまい閾値を割れなくなるため
        # (詳細は mdp/curriculums.py の dribble_target_range_curriculum docstring 参照)。
        # lin_vel_command_curriculum と同じ EMA ゲート方式 (min_updates で温め、
        # ステージ切替直後は hold して一気の連続遷移を防ぐ) は維持。
        #
        # NOTE: heading レンジは pos_x/pos_y (目標位置) とは独立にサンプリングされる
        # 「目標地点でのロボット最終向き」であり、目標の配置角度そのものには影響しない。
        # さらにこのタスクでは heading/error_heading を消費する報酬が一切登録されていない
        # ため (track_ang_vel_z_exp はフェードアウト済み)、heading レンジの値自体は現状
        # 学習挙動に対して不活性 (2026-09-23 の調査で確認済み)。実際に各レベルの「角度の
        # 広がり」を作っているのは pos_y (横方向レンジ) の拡大だけである。
        self.curriculum.dribble_target_range = CurrTerm(
            func=dribble_target_range_curriculum,
            params={
                "command_name": "dribble_target",
                "success_term_name": "ball_reached_target",
                # 2026-10-09: heading を全ステージから削除。当時の成功判定 (ball_reached_target)
                # はボール-目標の2次元距離のみで判定しており、heading/error_heading を消費する
                # 報酬・終了条件が一つも無かった (学習挙動に対して完全に不活性) ため、
                # 設定自体を無くした (dribble_target_range_curriculum 参照)。
                # (2026-10-11 追記: 成功判定自体はその後 ball_reached_target がボール+ロボット
                # 両方の距離を見るように変更されたが、heading 不使用の結論には影響しない)。
                # 2026-10-09: 「前進ドリブルしか学習しない」対策として、Level3-5 を
                # 直線以外の経路形状 (path_shape) に差し替えた (外部AI提案の
                # Level1-2=直線/Level3=カーブ/Level4=L字/Level5=Z字という対応を採用)。
                # Level7 も「障害物10個ランダム散布」から「Z字経路のコーナーに障害物を
                # 置いてジグザグに回避させる」設計に変更した (corner_obstacles=True)。
                # 2026-10-22: 障害物のレベル (旧 Level6〜8) をやめ、コースの形で難しくする 10 レベルに作り直した。
                # 障害物の回避は戦略側の経路計画 (pathplan_lib) に任せ、ポリシーは渡されたウェイポイント列を
                # たどることに専念する。レベルの並び:
                #   Level1〜2   直線                     (実装済みの straight)
                #   Level3〜5   連続的な旋回と重心移動 (止まらず曲がる): C 字 (curve) / O 字 (oval) / S 字 (s_curve)
                #   Level6〜8   減速・ピボット・再加速を伴う方向転換: L 字 (l_shape) / V 字 (v_turn) / U ターン (u_turn)
                #   Level9〜10  複合軌道: Z 字 (z_shape) / 8 の字 (figure8)
                # straight / curve / l_shape / z_shape は目標のオフセット (pos_x, pos_y、ワールド軸) から作る従来の形。
                # oval 以降はローカル座標 (始点 = ボール、x = 進行方向) で shape_params の範囲から作る
                # (mdp/commands.py の _local_path)。左右はランダムに反転し、omni_prob の割合で進行方向を全方位にする。
                "stages": [
                    {"pos_x": (2.0, 2.5), "pos_y": (0.0, 0.0)},  # Level1: 近距離・直線 (2.0〜2.5m)
                    {"pos_x": (2.5, 4.0), "pos_y": (-0.5, 0.5)},  # Level2: 中距離・直線 (2.5〜4.0m)
                    {"pos_x": (4.0, 6.0), "pos_y": (-1.0, 1.0), "path_shape": "curve"},  # Level3: 緩やかなカーブ (C 字)
                    {  # Level4: 大きな円軌道 (O 字、楕円を 1 周、周長 約 11〜17m)
                        "path_shape": "oval",
                        "shape_params": {"a": (2.0, 3.0), "b": (1.5, 2.5), "n_points": 16},
                    },
                    {  # Level5: S 字スラローム (振幅 0.5〜1.0m、波長 3〜4m、1.5〜2 周期 = 長さ 4.5〜8m)
                        "path_shape": "s_curve",
                        "shape_params": {"amplitude": (0.5, 1.0), "wavelength": (3.0, 4.0), "periods": (1.5, 2.0), "n_points": 12},
                    },
                    {"pos_x": (6.0, 8.0), "pos_y": (-1.5, 1.5), "path_shape": "l_shape"},  # Level6: L 字 (直角 1 回)
                    {  # Level7: V 字の鋭角ターン (120〜150° 向きを変える)
                        "path_shape": "v_turn",
                        "shape_params": {"leg1": (2.0, 4.0), "leg2": (2.0, 4.0), "turn_deg": (120.0, 150.0)},
                    },
                    {  # Level8: U ターン (幅 1.0〜2.0m で折り返す)
                        "path_shape": "u_turn",
                        "shape_params": {"leg1": (2.0, 4.0), "leg2": (2.0, 4.0), "width": (1.0, 2.0)},
                    },
                    {"pos_x": (8.0, 10.0), "pos_y": (-2.0, 2.0), "path_shape": "z_shape"},  # Level9: Z 字 (切り返し 2 回)
                    {  # Level10: 8 の字 (半径 1.2〜1.8m の円を左回り・右回りで 1 周ずつ、周長 約 15〜23m)
                        "path_shape": "figure8",
                        "shape_params": {"radius": (1.2, 1.8), "n_points_per_loop": 12},
                    },
                ],
                # 進級に必要な失敗率 EMA (Level10 の値は進級に使わない)。
                "error_threshold": [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.55, 0.6, 0.6, 0.6],
                # 2026-10-22: 進級が速すぎた (旧 Level6 は約 2 iteration で通過) ので、各レベルに最低 300 iteration 留まり、
                # EMA は終わったエピソード 1 つごとに重み 1/2000 で更新する。エピソードの 25% は通過済みのレベルから選ぶ
                # (曲がり方を忘れた対策)。混合・ランダムな折れ線のエピソードは進級の判定に数えない。
                "ema_episode_weight": 1.0 / 2000.0,
                "min_iterations": 300,
                "steps_per_iteration": 48,
                "prev_level_mix_prob": 0.25,
            },
        )

@configclass
class K1DribbleTargetEnvCfg_ResumeMature(K1DribbleTargetEnvCfg):
    """Stage2 で既に成熟済み (Level5 到達済み) の checkpoint を、新規プロセスとして
    ``--resume`` する専用 variant。

    ``modify_reward_weight_linear`` は ``env.common_step_counter`` (このプロセスの
    env.step 呼び出し回数) を基準に 0 → ``_phase2`` の距離で線形補間する。これは
    プロセスを新しく起動するたびに 0 から始まるため、``K1DribbleTargetEnvCfg`` の
    まま成熟済み checkpoint を resume すると、track_lin_vel_*/ball_progress_to_target
    等のフェードが最初からやり直しになり、ロード直後の数イテレーションだけ
    「歩行速度追従は満額、ボール関連報酬はほぼゼロ」という、この checkpoint が
    何万イテレーションも前に卒業した Stage1 相当の状態に一瞬で引き戻されてしまう。
    2026-09-22 の再学習 (ball_progress_to_target の weight 修正 40→12 を適用して
    model_39998.pt から resume) で、Optimizer state をリセットしても base_height
    転倒率が数イテレーションで 99% 超まで急落した根本原因はこれだった
    (Optimizer state の話は無関係だった)。

    このクラスは該当するフェード系カリキュラムの区間を ``(start_step=-1, end_step=0)``
    にして、1ステップ目から即座に最終値を適用する (フェードなし)。これにより
    resume 直後から checkpoint が実際に学習されていた設定のまま学習を継続できる。
    Stage1 から新規にフルカリキュラムを回す通常の ``K1DribbleTargetEnvCfg`` 自体は
    変更しない (将来また Stage1 スクラッチから回す場合はフェードが必要なため)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _instant = {"start_step": -1, "end_step": 0}
        for _term_name in (
            "track_lin_vel_sharp_weight",
            "track_lin_vel_coarse_weight",
            "track_ang_vel_weight",
            "ball_progress_weight",
            "ball_control_weight",
            "approach_ball_weight",
            "touch_ball_weight",
            "ball_proximity_weight",
            "ball_reached_bonus_weight",
            "clamping_penalty_weight",
            "ball_velocity_weight",
            "impact_kick_weight",
            "trajectory_deviation_weight",
            "gait_phase_cmd_threshold",
            # NOTE (2026-10-06): facing_ball_weight はここに加えない。この checkpoint が
            # 一度も経験していない全く新しい報酬項なので、他の成熟済み項と違って
            # 「即座に本番値へ戻す」のではなく、通常の _phase2 (500 iteration) で
            # 0 から滑らかにフェードインさせる (Stage1→2 遷移時と同じ「新しい報酬は
            # 急に満額を当てない」という方針を踏襲)。
            # NOTE (2026-10-11): push_ball_weight も同じ理由でここから除いた。
            # end_weight を 0.0→1.0 に復活させたが (push_ball_toward_target 登録箇所
            # 参照)、resume 対象のこの checkpoint は weight=0.0 のまま学習されており
            # この報酬を一度も経験していないため、facing_ball_weight と同様に通常の
            # _phase2 で滑らかにフェードインさせる。
        ):
            _term = getattr(self.curriculum, _term_name)
            _term.params = {**_term.params, **_instant}

@configclass
class K1DribbleTargetEnvCfg_PLAY(K1DribbleTargetEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # NOTE (2026-09-18/20): origin_type="asset_root" (ロボット追従) は毎レンダーフレーム
        # 発火するコールバックとして登録され、動画保存/エピソードリセットのタイミングで
        # 内部データが一時的に破棄されると ReferenceError: weakly-referenced object no
        # longer exists を無限に吐き続けてプロセスがハングする不具合がある。これは
        # video_length がエピソード長 (episode_length_s=20s ≒ 1000 step) を跨ぐと
        # ほぼ確実に発生するため、**録画コマンド側で video_length をエピソード長未満に
        # 抑えて使うこと** (例: --video_length 900 以下、--num_envs 1 推奨)。
        # ロボットが遠くに移動して画角から外れる問題 (固定視点 world だと起こる) より、
        # 追従できてロボットが常に見える方を優先する。
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, 2.5, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# Stage 1: walk-phase (ボールなし、観測形状は Stage 2 と同一のまま関連スロットをゼロ埋め)
# ---------------------------------------------------------------------------
@configclass
class K1DribbleTargetWalkPhaseEnvCfg(K1DribbleTargetEnvCfg):
    """Stage 1: 歩行のみを学習する。観測ベクトルの形状は Stage 2 と完全に同一に保つ
    (walk_kick の walk-phase パターンと同じ slot-reuse)。ここで得た checkpoint は
    そのまま ``Isaac-Velocity-Flat-K1-Dribble-Target-v0`` へ ``--checkpoint`` で
    引き継げる。
    """

    def __post_init__(self):
        super().__post_init__()

        # シーンからボール/障害物関連要素を取り除く (walk-phase では不要)。
        self.scene.soccer_ball = None
        self.scene.contact_balls_left = None
        self.scene.contact_balls_right = None
        self.events.reset_ball = None
        self.events.replan_dribble_path = None
        self.events.track_ball_front_ratio = None
        self.curriculum.ball_front_ratio = None
        # ボールが存在しないので目標座標コマンドも不要 (計算コスト削減)。
        # target_pos_rel obs は下で zero_obs に差し替え済みなのでコマンド削除に副作用はない。
        self.commands.dribble_target = None
        # Stage1 は学習済みの設定 (エピソード 25 秒、通常の time_out) のまま
        self.episode_length_s = 25.0
        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # 2026-10-06: K1DribbleTargetEnvCfg.__post_init__ (super() 経由で継承される) が
        # Stage2 向けに base_velocity を定数ゼロへ固定し、lin_vel_command/extreme_commands
        # カリキュラムを無効化し、gait_phase のゼロ化閾値も無効化している (docs/
        # dribble_deploy_handoff.md §4)。Stage1 (walk-phase) は本来の歩行学習
        # (速度コマンド追従そのものを学ぶ) が目的なので、これを K1FlatEnvCfg 相当の
        # 元の設定に戻す (NOTE 2026-09-20 の「戻す処理は不要」は Stage2 がこの固定を
        # 導入する前の記述であり、現在は不要ではなくなった)。
        # NOTE: ここで self.commands.base_velocity.ranges を読み直して再利用しないこと。
        # 親の __post_init__ が既に (0.0, 0.0) へ固定済みのため、代わりに元の
        # K1FlatEnvCfg 相当の初期値 (velocity_env_cfg.py の基本レンジ) を直接書く。
        # lin_vel_x/lin_vel_y はどのみち lin_vel_command カリキュラムが stage0 で
        # 即座に上書きする。
        # NOTE (2026-10-18 再構築): 歩行の親を K1FlatStancePlaneCfg に切り替えたので、ここで戻す
        # 「元の設定」も K1FlatEnvCfg の直書きの値ではなく、親 (_WALK_BASE_CFG) の値をそのまま
        # 引き継ぐ (best_walk はこの分布で学習されている: x ±1.5→1.8→2.0、extreme 3000/1500、
        # 再サンプリング 1〜7 秒)。親の __post_init__ で作られる値を、親の cfg を 1 つ作って取り出す。
        _walk = _WALK_BASE_CFG()
        self.commands.base_velocity.ranges = _walk.commands.base_velocity.ranges
        self.commands.base_velocity.resampling_time_range = _walk.commands.base_velocity.resampling_time_range
        self.curriculum.lin_vel_command = _walk.curriculum.lin_vel_command
        self.curriculum.extreme_commands = _walk.curriculum.extreme_commands
        self.curriculum.command_resampling_time_range = _walk.curriculum.command_resampling_time_range
        self.curriculum.gait_phase_cmd_threshold = None
        self.observations.policy.gait_phase.params = {
            **self.observations.policy.gait_phase.params, "cmd_threshold": _COMMAND_THRESHOLD,
        }
        self.observations.critic.gait_phase.params = {
            **self.observations.critic.gait_phase.params, "cmd_threshold": _COMMAND_THRESHOLD,
        }

        # 観測スロットの次元・並びは変えず、中身だけゼロ埋めする。
        self.observations.policy.target_pos_rel.func = zero_obs
        self.observations.policy.target_pos_rel.params = {"dim": 2}
        self.observations.policy.ball_pos_rel.func = zero_obs
        self.observations.policy.ball_pos_rel.params = {"dim": 2}
        self.observations.policy.ball_vel.func = zero_obs
        self.observations.policy.ball_vel.params = {"dim": 2}
        self.observations.critic.target_pos_rel.func = zero_obs
        self.observations.critic.target_pos_rel.params = {"dim": 2}
        self.observations.critic.ball_pos_rel.func = zero_obs
        self.observations.critic.ball_pos_rel.params = {"dim": 2}
        self.observations.critic.ball_vel.func = zero_obs
        self.observations.critic.ball_vel.params = {"dim": 2}
        # 2026-10-06: obstacle_pos_rel は Policy から削除済み (Critic専用)。
        self.observations.policy.foot_left_ball_pos_rel.func = zero_obs
        self.observations.policy.foot_left_ball_pos_rel.params = {"dim": 2}
        self.observations.policy.foot_right_ball_pos_rel.func = zero_obs
        self.observations.policy.foot_right_ball_pos_rel.params = {"dim": 2}
        self.observations.critic.foot_left_ball_pos_rel.func = zero_obs
        self.observations.critic.foot_left_ball_pos_rel.params = {"dim": 2}
        self.observations.critic.foot_right_ball_pos_rel.func = zero_obs
        self.observations.critic.foot_right_ball_pos_rel.params = {"dim": 2}
        # 2026-10-18 (§9-2 の構造的バグ修正): zero_obs に差し替えた項のノイズ設定を外す。
        # 残したままだと「0 + ノイズ」が観測され、EmpiricalNormalization の std が
        # ノイズ幅 (0.03〜0.06) で記録されて、Stage2 で実際の値が入ると 15〜25 倍に
        # 増幅されていた。Stage1 ではこの 5 項は厳密に 0 にする。
        for _term_name in DRIBBLE_TARGET_POLICY_EXTRA_TERMS:
            getattr(self.observations.policy, _term_name).noise = None

        # ドリブル関連報酬・終了条件・カリキュラムを無効化する。
        self.rewards.ball_progress_to_target = None
        self.rewards.robot_progress_to_target = None
        self.rewards.ball_control_radius_penalty = None
        self.rewards.approach_ball_progress = None
        self.rewards.touch_ball = None
        self.rewards.robot_facing_ball = None
        self.rewards.push_ball_toward_target = None
        self.rewards.ball_proximity_to_target = None
        self.rewards.ball_reached_target_bonus = None
        self.rewards.penalize_ball_clamping = None
        self.rewards.ball_velocity_toward_target = None
        self.rewards.impact_kick_reward = None
        self.rewards.ball_trajectory_deviation_penalty = None
        self.terminations.ball_too_far = None
        self.terminations.ball_reached_target = None
        # termination_penalty の term_keys が walk-phase に存在しない termination term
        # (ball_too_far/ball_reached_target/obstacle_collision) を含んだままだと
        # mdp.is_terminated_term の find_terms が "見つからない regex" で ValueError を出す。
        # walk-phase に実在する termination だけに絞り直す。
        self.rewards.termination_penalty.params = {"term_keys": ["base_contact", "base_height"]}
        self.curriculum.ball_progress_weight = None
        self.curriculum.robot_progress_weight = None
        self.curriculum.ball_control_weight = None
        self.curriculum.approach_ball_weight = None
        self.curriculum.touch_ball_weight = None
        self.curriculum.facing_ball_weight = None
        self.curriculum.push_ball_weight = None
        self.curriculum.ball_proximity_weight = None
        self.curriculum.ball_reached_bonus_weight = None
        self.curriculum.clamping_penalty_weight = None
        self.curriculum.ball_velocity_weight = None
        self.curriculum.impact_kick_weight = None
        self.curriculum.trajectory_deviation_weight = None
        self.curriculum.dribble_target_range = None
        # Stage2 で再導入した速度追従報酬のフェードアウトも無効化する。walk-phase では
        # K1FlatEnvCfg 由来の元の重み (track_lin_vel_xy_exp=1.5, track_lin_vel_xy_coarse=2.4,
        # track_ang_vel_z_exp=4.8) のまま、そのランダムコマンドに正しく追従できることを学ぶ。
        self.curriculum.track_lin_vel_sharp_weight = None
        self.curriculum.track_lin_vel_coarse_weight = None
        self.curriculum.track_ang_vel_weight = None

@configclass
class K1DribbleTargetWalkPhaseEnvCfg_PLAY(K1DribbleTargetWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
