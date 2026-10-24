#!/usr/bin/env python3
"""bhuman_walk_kick (Isaac-K1-BHuman-Walk-Kick-v0) の方策を MuJoCo (CPU) で動かして確かめる。

IsaacLab を使わない。checkpoint (.pt) を渡すと ONNX に書き出してから回す (``<run>/exported/``)。

    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py \
        --checkpoint logs/rsl_rl/k1_bhuman_walk_kick/<run>/model_XXXX.pt [--video_dir out/] [--json out.json]

ONNX の入出力 (歩行 policy.onnx と同じ形で、1 ステップの観測とアクションが増えただけ):

    入力  command      [1, 3]         (vx, vy, wz)  … 下の「歩行コマンド」で作る
          obs_history  [1, 100, 61]   古い → 新しい。1 ステップ = 歩行の 49 + キック用 12
    出力  actions      [1, 13]        脚 12 (scale 0.5、既定角まわり) + 歩行周波数のオフセット [Hz]

学習環境との対応 (値は bhuman_walk_kick_env_cfg.py / locomotion/rough_env_cfg.py から写したもの):

* ロボット: 学習と同じ K1_locomotion.urdf (腕・頭は固定、脚 12 関節)。自己衝突なし。
* 関節 PD: 股・膝 kp 160 / kd 4、足首 kp 50 / kd 2.5、トルク上限 68/76/38.3/112/38.3/38.3 Nm、
  armature は股・膝が URDF 変換時の値、足首 0.0565。目標は物理ステップ (5 ms) で ``--delay`` だけ遅らせる
  (学習: 2〜7 の一様乱数)。
* 制御: 5 ms × 4 = 50 Hz。
* 位相: 周波数 = 速度コマンドから決まる値 (|v| ≤ 1.0 で 1.8 Hz、1.8 m/s で 2.0 Hz) + clip(action[12], ±0.5)。
  左足 sin/cos, 右足 (+π) sin/cos。コマンドの大きさが 0.05 未満なら位相の観測は 0。
* ボール: 半径 0.11 m、0.45 kg、PhysX の線形/回転ダンピング (0.75 / 0.5) は外力で再現する。
* キック用の観測・歩行コマンドは学習環境 (mdp/kick_command.py, mdp/walk_command.py, mdp/rewards._walk_target)
  の式を 1 env 分 numpy に写したもの (ノイズなし)。ボール位置は両足裏の中点・yaw フレーム、
  ``--ball_delay`` ステップ前の位置 (学習: 1〜3 ステップの一様乱数)。

各シナリオで「転倒したか」「蹴ったか (ボールが動き出した時刻)」「ボールの最高速度と目標初速」
「ボールの進んだ向きと指令方向のずれ」を出す。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf"

JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
DEFAULT_Q = np.array([-0.26, 0.0, 0.0, 0.52, -0.26, 0.0] * 2)
KP = np.array([160.0, 160.0, 160.0, 160.0, 50.0, 50.0] * 2)
KD = np.array([4.0, 4.0, 4.0, 4.0, 2.5, 2.5] * 2)
EFFORT = np.array([68.0, 76.0, 38.3, 112.0, 38.3, 38.3] * 2)
ARMATURE = np.array([0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0565, 0.0565] * 2)
SIM_DT, DECIMATION = 0.005, 4
STEP_DT = SIM_DT * DECIMATION
ACTION_SCALE = 0.5
HISTORY = 100
# 1 ステップの観測の項の次元 (locomotion/history_layout.py の KICK_POLICY_TERM_SPECS と同じ並び)
TERM_DIMS = [3, 3, 3, 12, 12, 12, 4, 2, 1, 2, 3, 2, 1, 1]
STEP_DIM = sum(TERM_DIMS)
# 位相 (rough_env_cfg._PHASE_FREQ_PARAMS)
PH_LOW_SPEED, PH_HIGH_SPEED, PH_LOW_F, PH_HIGH_F = 1.0, 1.8, 1.8, 2.0
CMD_THRESHOLD = 0.05
GAIT_LIMIT = (-0.5, 0.5)
# ボール (bhuman_walk_kick_env_cfg.py)
BALL_R, BALL_M = 0.11, 0.45
BALL_LIN_DAMP, BALL_ANG_DAMP = 0.75, 0.5
# kick_command.BHumanKickCommandCfg (Stage 1 の値。--stage_cfg で上書き)
KCFG = dict(
    ball_friction=0.3, kick_strong_range=20.0, kick_pose_x=(0.0, -0.35), kick_pose_y_off=0.06,
    ball_y_dis_ratio=(2.0, 2.0), ball_distance_offset=0.3, ball_rotation_offset=0.1, ball_avoidance_range=0.55,
)
# walk_command.BallWalkVelocityCommandCfg
WCFG = dict(gain_xy=1.5, vx_range=(-0.5, 1.0), max_vy=0.5, gain_yaw=1.5, max_wz=1.0, near_dist=0.6, blend_dist=0.4,
            kick_through_speed=0.5)
SOLE_OFFSET = np.array([0.0, 0.0, -0.038])
BASE_INIT = np.array([0.0, 0.0, 0.6])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# --------------------------------------------------------------------------- #
# checkpoint -> ONNX (HistoryActorCritic の actor + 観測の正規化)
# --------------------------------------------------------------------------- #
def build_torch_policy(ckpt: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    assert sd["actor.cnn.0.weight"].shape[1] == STEP_DIM, "1 ステップ 61 次元のキック方策ではない"

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = nn.Parameter(sd["actor_obs_normalizer._mean"].clone(), requires_grad=False)
            self.std = nn.Parameter(sd["actor_obs_normalizer._std"].clone(), requires_grad=False)
            self.c0 = nn.Conv1d(STEP_DIM, 32, 8, 4)
            self.c1 = nn.Conv1d(32, 16, 4, 2)
            self.c0.weight.data.copy_(sd["actor.cnn.0.weight"]); self.c0.bias.data.copy_(sd["actor.cnn.0.bias"])
            self.c1.weight.data.copy_(sd["actor.cnn.2.weight"]); self.c1.bias.data.copy_(sd["actor.cnn.2.bias"])
            layers = []
            i = 0
            while f"actor.mlp.{i}.weight" in sd:
                w = sd[f"actor.mlp.{i}.weight"]
                lin = nn.Linear(w.shape[1], w.shape[0])
                lin.weight.data.copy_(w); lin.bias.data.copy_(sd[f"actor.mlp.{i}.bias"])
                layers.append(lin)
                if f"actor.mlp.{i + 2}.weight" in sd:
                    layers.append(nn.ELU())
                i += 2
            self.mlp = nn.Sequential(*layers)

        def forward(self, command, obs_history):
            # 学習時の並び: [command, 項ごとの (100 × dim) ブロック (古い → 新しい)]
            parts, ch = [command], 0
            for d in TERM_DIMS:
                parts.append(obs_history[:, :, ch:ch + d].reshape(obs_history.shape[0], -1))
                ch += d
            x = (torch.cat(parts, dim=-1) - self.mean) / (self.std + 1e-2)
            direct, hist = x[:, :3], x[:, 3:]
            blocks = torch.split(hist, [HISTORY * d for d in TERM_DIMS], dim=-1)
            frames3 = [b.reshape(-1, HISTORY, d) for b, d in zip(blocks, TERM_DIMS)]
            recent = torch.cat([f[:, -4:].reshape(f.shape[0], -1) for f in frames3], dim=-1)
            frames = torch.cat(frames3, dim=-1).transpose(1, 2)
            h = F.elu(self.c1(F.elu(self.c0(frames)))).flatten(1)
            return self.mlp(torch.cat((direct, recent, h), dim=-1))

    return Policy().eval()


def export_onnx(ckpt: Path) -> Path:
    import torch

    pol = build_torch_policy(ckpt)
    out = ckpt.parent / "exported" / f"{ckpt.parent.name}_{ckpt.stem}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    ex = (torch.zeros(1, 3), torch.zeros(1, HISTORY, STEP_DIM))
    torch.onnx.export(pol, ex, str(out), input_names=["command", "obs_history"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    return out


# --------------------------------------------------------------------------- #
# MuJoCo model
# --------------------------------------------------------------------------- #
def build_model(friction: float, ball_friction: float = 1.0, ball_foot_friction: float | None = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.body("Trunk").add_freejoint()
    robot_geoms = []
    for i, g in enumerate(spec.geoms):
        # ロボット同士は当たらない (学習: enabled_self_collisions=False)、床とボールには当たる
        g.contype, g.conaffinity = 1, 0
        g.friction = [friction, 0.005, 0.0001]
        if not g.name:
            g.name = f"robot_geom_{i}"
        robot_geoms.append(g.name)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=0, conaffinity=1,
                   friction=[friction, 0.005, 0.0001], rgba=[0.35, 0.55, 0.35, 1])
    world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    ball = world.add_body(name="ball", pos=[1.0, 0.0, BALL_R])
    ball.add_freejoint()
    ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_M, contype=1,
                  conaffinity=1, condim=3, friction=[ball_friction, 0.005, 0.0001], rgba=[1.0, 0.45, 0.0, 1],
                  priority=1)  # ボールの接触はボールの摩擦を使う (MuJoCo 既定は大きい方)
    if ball_foot_friction is not None:
        # 学習 (PhysX) では ボール-床 = ボール 1.0 × 床 1.0 (multiply)、ボール-足 = (ボール 1.0 + 足 0.3〜1.0) / 2 (average)
        # で、足との摩擦は 0.65〜1.0。ボールとロボットの接触だけ明示の pair にして摩擦を別に決める。
        bodies = {spec.geoms[i].parent.name for i in range(len(spec.geoms)) if spec.geoms[i].name in robot_geoms}
        for b in bodies:
            spec.add_exclude(bodyname1="ball", bodyname2=b)
        for gname in robot_geoms:
            spec.add_pair(geomname1="ball", geomname2=gname, condim=3,
                          friction=[ball_foot_friction, ball_foot_friction, 0.005, 0.0001, 0.0001])
    target = world.add_body(name="target_marker", mocap=True, pos=[0, 0, 0.002])
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.002, 0], contype=0, conaffinity=0,
                    rgba=[1.0, 0.0, 0.0, 0.8])
    for i, name in enumerate(JOINTS):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP[i]
        a.biasprm[:3] = [0.0, -KP[i], -KD[i]]
        a.forcelimited = True
        a.forcerange = [-EFFORT[i], EFFORT[i]]
    model = spec.compile()
    for i, name in enumerate(JOINTS):
        adr = model.joint(name).dofadr[0]
        model.dof_armature[adr] = ARMATURE[i]
        model.dof_damping[adr] = 0.0
        model.dof_frictionloss[adr] = 0.0
    return model


@dataclass
class Scenario:
    name: str
    ball_xy: tuple[float, float]  # ロボット初期姿勢 (原点・+x 向き) からの位置 [m]
    kick_deg: float  # キック方向 (ロボットの初期向き基準) [deg]
    strong: bool
    range_m: float = 3.0  # 距離指定キックの距離 [m] (strong のときは 20 m)


DEFAULT_SCENARIOS = [
    Scenario("front_strong", (1.0, 0.0), 0.0, True),
    Scenario("front_normal_2m", (1.0, 0.0), 0.0, False, 2.0),
    Scenario("front_normal_5m", (1.0, 0.0), 0.0, False, 5.0),
    Scenario("left30_strong", (1.0, 0.3), 30.0, True),
    Scenario("right30_strong", (1.0, -0.3), -30.0, True),
    Scenario("left60_normal_4m", (0.8, 0.4), 60.0, False, 4.0),
    Scenario("right90_strong", (1.2, 0.0), -90.0, True),
    Scenario("left90_strong", (1.2, 0.0), 90.0, True),
    Scenario("behind_strong", (-0.8, 0.3), 180.0, True),
    Scenario("far_front_strong", (2.5, 0.5), 0.0, True),
]


class Sim:
    def __init__(self, onnx_path: Path, friction: float, delay: int, ball_delay: int, ball_friction: float = 1.0,
                 ball_foot_friction: float | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.m = build_model(friction, ball_friction, ball_foot_friction)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in JOINTS])
        self.act = np.array([self.m.actuator(n).id for n in JOINTS])
        self.trunk = self.m.body("Trunk").id
        self.feet = [self.m.body("left_foot_link").id, self.m.body("right_foot_link").id]
        self.ball = self.m.body("ball").id
        self.ball_q = self.m.joint(self.m.body_jntadr[self.ball]).qposadr[0]
        self.ball_v = self.m.joint(self.m.body_jntadr[self.ball]).dofadr[0]
        self.target_mocap = self.m.body("target_marker").mocapid[0]
        self.floor = self.m.geom("floor").id
        self.trunk_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.trunk}
        self.robot_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] not in (0, self.ball)}
        self.ball_geom = self.m.geom("ball").id
        self.delay = int(delay)
        self.ball_delay = int(ball_delay)
        self.ball_inertia = self.m.body_inertia[self.ball].copy()

    # ------------------------------------------------------------------ #
    def reset(self, sc: Scenario):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = BASE_INIT
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = DEFAULT_Q
        d.qpos[self.ball_q:self.ball_q + 3] = [sc.ball_xy[0], sc.ball_xy[1], BALL_R]
        d.qpos[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros(13)
        self.phase = 0.0
        self.target_buf = deque([DEFAULT_Q.copy()] * (self.delay + 1), maxlen=self.delay + 1)
        bp = d.xpos[self.ball].copy()
        self.ball_buf = deque([bp.copy()] * 4, maxlen=4)
        # 学習の _resample_kick_ball: 基準向き = 今の yaw、目標点 = ボール + range * (cos, sin)
        self.ref_yaw = self.yaw()
        self.range0 = KCFG["kick_strong_range"] if sc.strong else sc.range_m
        gdir = wrap(self.ref_yaw + math.radians(sc.kick_deg))
        self.global_dir0 = gdir
        self.target = bp[:2] + self.range0 * np.array([math.cos(gdir), math.sin(gdir)])
        d.mocap_pos[self.target_mocap] = [bp[0] + 0.6 * math.cos(gdir), bp[1] + 0.6 * math.sin(gdir), 0.002]
        self.strong = sc.strong
        # キック姿勢の状態 (kick_command._update の 6, 8)
        self.ball_close_to_kick_pose = 1.0
        self.ball_pos = self.ball_rel(bp)
        dac = wrap(gdir - self.yaw())
        self.filtered_ball_pos = rot2(self.ball_pos, -dac)
        self.kick_pose_translation_ratio = 0.0
        self.ball_kick_pose = bp[:2] + np.array([math.cos(gdir), math.sin(gdir)]) * KCFG["kick_pose_x"][1]
        self.update_kick_state()
        self.command = self.walk_command()
        self.history = None

    def yaw(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def origin(self):
        pts = []
        for b in self.feet:
            R = self.d.xmat[b].reshape(3, 3)
            pts.append(self.d.xpos[b] + R @ SOLE_OFFSET)
        return np.mean(pts, axis=0)

    def ball_rel(self, ball_w):
        return rot2(ball_w[:2] - self.origin()[:2], -self.yaw())

    # ------------------------------------------------------------------ #
    # 学習環境の式 (1 env 分)
    # ------------------------------------------------------------------ #
    def update_kick_state(self):
        """kick_command._update の 3.3, 4, 6, 8 (観測と歩行コマンドに要る部分)。"""
        k = KCFG
        d = self.d
        yaw = self.yaw()
        ball_w = d.xpos[self.ball].copy()
        delayed = self.ball_buf[len(self.ball_buf) - 1 - self.ball_delay]
        self.ball_pos = self.ball_rel(delayed)
        bv = d.qvel[self.ball_v:self.ball_v + 3]
        self.ball_vel = rot2(bv[:2], -yaw)
        to_target = self.target - ball_w[:2]
        self.global_direction = math.atan2(to_target[1], to_target[0])
        self.range = min(float(np.linalg.norm(to_target)), k["kick_strong_range"])
        self.dac = wrap(self.global_direction - yaw)
        # 6) キック方向に回したボール位置
        rot_ratio = np.clip((abs(self.dac) - k["ball_rotation_offset"]) / 0.4, 0.0, 1.0)
        bik = rot2(self.ball_pos, -self.dac)
        by = bik[1]
        dist = float(np.linalg.norm(self.ball_pos))
        # 8) キック姿勢
        y_ratio = np.clip((abs(by) - k["ball_y_dis_ratio"][0] * k["kick_pose_y_off"]) / (k["ball_y_dis_ratio"][1] * BALL_R),
                          0.0, 1.0)
        self.kick_pose_translation_ratio = float(np.clip((abs(math.atan2(by, bik[0])) - 0.3) / 1.2, 0.0, 1.0))
        dist_ratio = np.clip((dist - k["ball_distance_offset"]) / 0.4, 0.0, 1.0)
        mr = max(y_ratio, dist_ratio, rot_ratio, self.kick_pose_translation_ratio)
        self.ball_close_to_kick_pose = float(np.clip(-1.0 + 2.0 * mr, self.ball_close_to_kick_pose - STEP_DT * 2,
                                                     self.ball_close_to_kick_pose + STEP_DT * 2))
        lp = 0.9 if mr == 1.0 else 0.2
        self.filtered_ball_pos = (1.0 - lp) * self.filtered_ball_pos + lp * bik
        kp0, kp1 = k["kick_pose_x"]
        kp = kp0 + (kp1 - kp0) * max(self.ball_close_to_kick_pose, 0.0)
        self.ball_kick_pose = ball_w[:2] + np.array([math.cos(self.global_direction), math.sin(self.global_direction)]) * kp

    def walk_target(self):
        """rewards._walk_target (ロボットの yaw フレーム、胴体原点基準)。"""
        k = KCFG
        d = self.d
        yaw = self.yaw()
        base = d.qpos[0:3].copy()
        ball_w = d.xpos[self.ball].copy()
        kpir = rot2(self.ball_kick_pose - base[:2], -yaw)
        extra = max((self.kick_pose_translation_ratio - 0.5) / 0.5, 0.0) * (BALL_R + 0.1)
        x_off = -math.sin(self.dac) * (k["kick_pose_y_off"] + extra)
        y_off = math.cos(self.dac) * (k["kick_pose_y_off"] + extra)
        if self.filtered_ball_pos[1] < 0:
            wt = kpir + np.array([x_off, y_off])
        else:
            wt = kpir - np.array([x_off, y_off])
        gdir = self.global_direction
        bir = rot2(ball_w[:2] - base[:2], -gdir)
        ratio = 0.0 if self.ball_close_to_kick_pose < 1 else float(np.clip(-bir[0] / 0.3, 0.0, 1.0))
        bir = bir.copy()
        bir[0] += np.clip(abs(bir[1]) / 0.3, 0.0, 1.0) * -0.2 + np.clip(max(-bir[0], 1.0 - abs(bir[1]) / 0.3), 0.0, 1.0) * 0.3
        bir[1] += (1.0 if self.filtered_ball_pos[1] < 0 else -1.0) * k["ball_avoidance_range"]
        side = rot2(rot2(bir, gdir), -yaw)
        return wt * (1.0 - ratio) + side * ratio

    def walk_command(self):
        """walk_command.BallWalkVelocityCommand._update_command。"""
        w = WCFG
        t = self.walk_target()
        head_far = math.atan2(t[1], t[0])
        head_near = wrap(self.dac)
        bd = float(np.linalg.norm(self.ball_pos))
        bl = float(np.clip((bd - w["near_dist"]) / w["blend_dist"], 0.0, 1.0))
        err = wrap(head_near + bl * wrap(head_far - head_near))
        vx = float(np.clip(w["gain_xy"] * t[0], *w["vx_range"]))
        vx *= (1.0 - bl) + bl * max(math.cos(head_far), 0.0)
        through = float(np.clip(-self.ball_close_to_kick_pose, 0.0, 1.0)) * w["kick_through_speed"]
        through *= max(math.cos(head_near), 0.0) * (1.0 if self.ball_pos[0] > 0 else 0.0)
        vx = max(vx, through)
        vy = float(np.clip(w["gain_xy"] * t[1], -w["max_vy"], w["max_vy"]))
        wz = float(np.clip(w["gain_yaw"] * err, -w["max_wz"], w["max_wz"]))
        return np.array([vx, vy, wz])

    def phase_freq(self, cmd, freq_action):
        s = float(np.linalg.norm(cmd[:2]))
        slope = (PH_HIGH_F - PH_LOW_F) / (PH_HIGH_SPEED - PH_LOW_SPEED)
        return PH_LOW_F + max(s - PH_LOW_SPEED, 0.0) * slope + float(np.clip(freq_action, *GAIT_LIMIT))

    def frame(self) -> np.ndarray:
        """1 ステップの観測 (61)。"""
        d = self.d
        R = d.xmat[self.trunk].reshape(3, 3)
        cmd = self.command
        pl = self.phase
        ph = np.array([math.sin(pl), math.cos(pl), math.sin(pl + math.pi), math.cos(pl + math.pi)])
        if np.linalg.norm(cmd) < CMD_THRESHOLD:
            ph[:] = 0.0
        exp_v = math.sqrt(2.0) * math.sqrt(KCFG["ball_friction"] * self.range)
        return np.concatenate([
            d.qvel[3:6],  # free joint の角速度は胴体フレーム
            R.T @ np.array([0.0, 0.0, -1.0]),
            cmd,
            d.qpos[self.qadr] - DEFAULT_Q,
            d.qvel[self.dadr],
            self.last_action[:12],
            ph,
            self.ball_pos * 0.1,
            [self.dac / math.pi],
            [math.sin(self.dac), math.cos(self.dac)],
            [1.0 if self.strong else 0.0, 0.0, 0.0],
            self.ball_vel * 0.1,
            [exp_v * 0.1],
            [float(np.clip(self.last_action[12], -2.0, 2.0))],
        ]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def trunk_floor_contact(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 == self.floor and c.geom2 in self.trunk_geoms) or (
                c.geom2 == self.floor and c.geom1 in self.trunk_geoms
            ):
                return True
        return False

    def robot_ball_contact(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 == self.ball_geom and c.geom2 in self.robot_geoms) or (
                c.geom2 == self.ball_geom and c.geom1 in self.robot_geoms
            ):
                return True
        return False

    def ball_damping(self):
        v = self.d.qvel[self.ball_v:self.ball_v + 6]
        f = np.zeros(6)
        f[:3] = -BALL_M * BALL_LIN_DAMP * v[:3]
        R = self.d.xmat[self.ball].reshape(3, 3)
        w_local = R.T @ v[3:6]  # free joint の角速度はボール座標系
        f[3:] = R @ (-self.ball_inertia * BALL_ANG_DAMP * w_local)
        self.d.xfrc_applied[self.ball] = f

    # ------------------------------------------------------------------ #
    def run(self, sc: Scenario, duration: float, after_kick: float = 1.5, video: Path | None = None, fps: int = 25):
        self.reset(sc)
        m, d = self.m, self.d
        renderer = None
        frames = []
        if video is not None:
            renderer = mujoco.Renderer(m, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.azimuth = 3.0, -25, 135
        ball0 = d.xpos[self.ball].copy()
        exp_vel = math.sqrt(2.0 * KCFG["ball_friction"] * self.range0)
        rec = dict(fell=False, fell_t=None, reason="", touch_t=None, kick_t=None, kick_speed=0.0,
                   kick_dir_err_deg=None, min_base_z=1.0, max_abs_freq_offset=0.0)
        n_steps = int(round(duration / STEP_DT))
        kick_window = []
        ball_at_kick = None
        end_step = n_steps
        for step in range(n_steps):
            f = self.frame()
            if self.history is None:  # CircularBuffer と同じく、最初のフレームで全履歴を埋める
                self.history = np.repeat(f[None], HISTORY, axis=0)
            else:
                self.history = np.concatenate((self.history[1:], f[None]), axis=0)
            a = self.sess.run(None, {"command": self.command[None].astype(np.float32),
                                     "obs_history": self.history[None]})[0][0].astype(np.float64)
            self.last_action = a.copy()
            rec["max_abs_freq_offset"] = max(rec["max_abs_freq_offset"], abs(float(np.clip(a[12], *GAIT_LIMIT))))
            target_q = DEFAULT_Q + ACTION_SCALE * a[:12]
            # 位相は「前ステップのコマンド + 今の周波数 action」で進む (学習: 報酬計算時に積分)
            self.phase = math.fmod(self.phase + 2 * math.pi * self.phase_freq(self.command, a[12]) * STEP_DT,
                                   2 * math.pi)
            for _ in range(DECIMATION):
                self.target_buf.append(target_q)
                d.ctrl[self.act] = self.target_buf[0]
                self.ball_damping()
                mujoco.mj_step(m, d)
                if rec["touch_t"] is None and self.robot_ball_contact():
                    rec["touch_t"] = step * STEP_DT
            self.ball_buf.append(d.xpos[self.ball].copy())
            self.update_kick_state()
            self.command = self.walk_command()
            t = (step + 1) * STEP_DT
            bv = d.qvel[self.ball_v:self.ball_v + 3]
            speed = float(np.linalg.norm(bv[:2]))
            rec["min_base_z"] = min(rec["min_base_z"], float(d.qpos[2]))
            if rec["kick_t"] is None and speed > 0.3 and np.linalg.norm(d.xpos[self.ball][:2] - ball0[:2]) > 0.02:
                rec["kick_t"] = t
                ball_at_kick = d.xpos[self.ball][:2].copy()
                end_step = min(n_steps, step + int(round(after_kick / STEP_DT)))
            if rec["kick_t"] is not None and t - rec["kick_t"] <= 0.4:
                kick_window.append((speed, math.atan2(bv[1], bv[0])))
            # 蹴った瞬間の速さは足に押された一瞬の値で、すぐ転がりに変わって落ちる。
            # 「転がっていく速さ」として 0.3 秒後の速さと、1 秒間に進んだ距離も記録する。
            if rec["kick_t"] is not None and "speed_0p3" not in rec and t - rec["kick_t"] >= 0.3 - 1e-6:
                rec["speed_0p3"] = speed
            if rec["kick_t"] is not None and "disp_1s" not in rec and t - rec["kick_t"] >= 1.0 - 1e-6:
                rec["disp_1s"] = float(np.linalg.norm(d.xpos[self.ball][:2] - ball_at_kick))
            if renderer is not None and step % max(1, int(round(1.0 / fps / STEP_DT))) == 0:
                cam.lookat[:] = [(d.qpos[0] + d.xpos[self.ball][0]) / 2, (d.qpos[1] + d.xpos[self.ball][1]) / 2, 0.3]
                renderer.update_scene(d, cam)
                frames.append(renderer.render().copy())
            if d.qpos[2] < 0.30:
                rec.update(fell=True, fell_t=t, reason=f"base height {d.qpos[2]:.2f} < 0.30")
                break
            if self.trunk_floor_contact():
                rec.update(fell=True, fell_t=t, reason="trunk touched ground")
                break
            if step + 1 >= end_step:
                break
        if kick_window:
            sp, ang = max(kick_window, key=lambda x: x[0])
            rec["kick_speed"] = sp
            rec["kick_dir_err_deg"] = math.degrees(wrap(ang - self.global_dir0))
        ball1 = d.xpos[self.ball].copy()
        ref = ball_at_kick if ball_at_kick is not None else ball0[:2]
        disp = ball1[:2] - ref
        rec["ball_disp_m"] = float(np.linalg.norm(ball1[:2] - ball0[:2]))
        rec["ball_disp_dir_err_deg"] = (
            math.degrees(wrap(math.atan2(disp[1], disp[0]) - self.global_dir0)) if np.linalg.norm(disp) > 0.05 else None
        )
        rec["expected_kick_speed"] = exp_vel
        rec["robot_disp_m"] = float(np.linalg.norm(d.qpos[0:2] - BASE_INIT[:2]))
        rec["sim_time_s"] = (step + 1) * STEP_DT
        if frames:
            import imageio

            imageio.mimsave(str(video), frames, fps=fps, macro_block_size=1)
        return rec


def fmt(sc: Scenario, r: dict) -> str:
    kick = "no kick" if r["kick_t"] is None else (
        f"kick at {r['kick_t']:.2f}s, ball {r['kick_speed']:.2f} m/s (target {r['expected_kick_speed']:.2f}), "
        f"dir err {r['kick_dir_err_deg']:+.0f} deg"
    )
    fell = f"FELL at {r['fell_t']:.2f}s ({r['reason']})" if r["fell"] else "no fall"
    disp_err = "" if r["ball_disp_dir_err_deg"] is None else f", disp dir err {r['ball_disp_dir_err_deg']:+.0f} deg"
    return (f"{sc.name:18s} | {fell:34s} | {kick} | ball moved {r['ball_disp_m']:.2f} m{disp_err} | "
            f"freq off max {r['max_abs_freq_offset']:.2f} Hz, min base z {r['min_base_z']:.2f}")


def summarize(scenarios, results) -> dict:
    kicked = [r for r in results.values() if r["kick_t"] is not None and not r["fell"]]
    strong = [r for s, r in zip(scenarios, results.values()) if s.strong and r["kick_t"] is not None]
    normal = [r for s, r in zip(scenarios, results.values()) if not s.strong and r["kick_t"] is not None]
    errs = [abs(r["ball_disp_dir_err_deg"]) for r in kicked if r["ball_disp_dir_err_deg"] is not None]
    return {
        "n": len(results),
        "falls": sum(r["fell"] for r in results.values()),
        "kicked": len(kicked),
        "strong_speed_mean": float(np.mean([r["kick_speed"] for r in strong])) if strong else None,
        "strong_speed_0p3_mean": float(np.mean([r.get("speed_0p3", 0.0) for r in strong])) if strong else None,
        "strong_disp_1s_mean": float(np.mean([r.get("disp_1s", 0.0) for r in strong])) if strong else None,
        "normal_speed_abs_err_mean": float(np.mean([abs(r["kick_speed"] - r["expected_kick_speed"]) for r in normal]))
        if normal else None,
        "dir_err_abs_mean_deg": float(np.mean(errs)) if errs else None,
        "dir_err_abs_max_deg": float(np.max(errs)) if errs else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="rsl_rl の model_*.pt (ONNX に書き出してから使う)")
    src.add_argument("--onnx", type=Path)
    ap.add_argument("--export_only", action="store_true")
    ap.add_argument("--duration", type=float, default=15.0, help="1 シナリオの上限 [s]")
    ap.add_argument("--after_kick", type=float, default=1.5, help="蹴ってからこの時間で打ち切る [s]")
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--ball_friction", type=float, default=1.0, help="ボールと床の摩擦 (学習: 1.0)")
    ap.add_argument("--ball_foot_friction", type=float, default=0.8,
                    help="ボールとロボットの摩擦 (学習: 0.65〜1.0)。負なら pair を作らずボールの摩擦を使う")
    ap.add_argument("--delay", type=int, default=4, help="PD 目標の遅れ [物理ステップ] (学習: 2〜7)")
    ap.add_argument("--ball_delay", type=int, default=1, help="観測するボール位置の遅れ [制御ステップ] (学習: 1〜3)")
    ap.add_argument("--scenarios", type=str, default=None, help="カンマ区切りで既定シナリオを絞る")
    ap.add_argument("--random", type=int, default=0,
                    help="既定シナリオの代わりに N 個のランダムなシナリオ (ボール 0.5〜2 m・全方向、キック方向 全方向、強/距離指定 半々)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage_cfg", type=str, default=None,
                    help='KCFG の上書き (JSON)。例: \'{"ball_distance_offset": 0.7, "ball_rotation_offset": 0.2}\'')
    ap.add_argument("--kick_through_speed", type=float, default=0.5, help="学習の walk_command と合わせる (s1/s1b は 0)")
    ap.add_argument("--video_dir", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    WCFG["kick_through_speed"] = a.kick_through_speed
    if a.stage_cfg:
        KCFG.update(json.loads(a.stage_cfg))
    onnx_path = a.onnx or export_onnx(a.checkpoint)
    print(f"[sim2sim] policy: {onnx_path}")
    if a.export_only:
        return 0
    if a.video_dir is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        a.video_dir.mkdir(parents=True, exist_ok=True)
    scenarios = DEFAULT_SCENARIOS
    if a.random > 0:
        rng = np.random.default_rng(a.seed)
        scenarios = []
        for i in range(a.random):
            dist, ang = rng.uniform(0.5, 2.0), rng.uniform(-math.pi, math.pi)
            kick = rng.uniform(-180.0, 180.0)
            strong = bool(rng.random() < 0.5)
            rng_m = float(np.exp(rng.uniform(math.log(1.0), math.log(8.0))))
            scenarios.append(Scenario(f"rand{i:02d}_{'S' if strong else 'N'}{kick:+04.0f}",
                                      (dist * math.cos(ang), dist * math.sin(ang)), kick, strong, rng_m))
    if a.scenarios:
        names = set(a.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in names]
    bff = None if a.ball_foot_friction < 0 else a.ball_foot_friction
    sim = Sim(onnx_path, a.friction, a.delay, a.ball_delay, a.ball_friction, bff)
    results = {}
    for sc in scenarios:
        video = a.video_dir / f"{sc.name}.mp4" if a.video_dir else None
        r = sim.run(sc, a.duration, a.after_kick, video)
        results[sc.name] = r
        print(fmt(sc, r), flush=True)
    summ = summarize(scenarios, results)
    print("[summary]", json.dumps(summ), flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"policy": str(onnx_path), "summary": summ, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
