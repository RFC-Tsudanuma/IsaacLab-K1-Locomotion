#!/usr/bin/env python3
"""bhuman_walk_kick (Isaac-K1-BHuman-Walk-Kick-*) の方策を MuJoCo (CPU) で動かし、仕様の表の各項目を測る。

IsaacLab を使わない。checkpoint (.pt) を渡すと ONNX に書き出してから回す (``<run>/exported/``)。

    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py \
        --checkpoint logs/rsl_rl/k1_bhuman_walk_kick/<run>/model_XXXX.pt --random 100 [--video_dir out/] [--json out.json]
    # 画面で見る (実時間。蹴った後も次のボールが出て続く。--onnx で書き出し済みの ONNX も渡せる)
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --checkpoint <model_XXXX.pt> --view --scene left90_strong
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_walk_kick.py --checkpoint <model_XXXX.pt> --view \
        --ball_x 1.0 --ball_y 0.3 --kick_deg 45 --strong

場面の進め方と測り方 (蹴り足の向き・当て方・止まる・軸足・振り・つながり など) は kick_metrics.py にまとめてあり、
B-Human の配布方策を回す sim2sim_bhuman_reference.py と同じ。仕様は docs/bhuman_walk_kick_spec.md。

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
* ボール: 既定は B-Human と同じ 半径 0.095 m・0.29 kg (``--ball_radius/--ball_mass``)。PhysX の線形/回転ダンピング
  (0.75 / 0.5) は外力で再現する。
* キック用の観測・歩行コマンドは学習環境 (mdp/kick_command.py, mdp/walk_command.py, mdp/rewards._walk_target)
  の式を 1 env 分 numpy に写したもの (ノイズなし)。ボール位置は両足裏の中点・yaw フレーム、
  ``--ball_delay`` ステップ前の位置 (学習: 1〜3 ステップの一様乱数)。
* ``--kick_through_speed``: 歩行コマンドの「キック姿勢に入った後の最低前進速度」。Stage 2 以降は 0 (既定)。
  Stage 1 (s1c) 〜 前回の s6b/s7 は 0.5 で学習したので、それらを回すときは 0.5 を渡す。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kick_metrics as km  # noqa: E402

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
# kick_command.BHumanKickCommandCfg (Stage 1 の値。--stage_cfg で上書き)
KCFG = dict(
    ball_friction=0.3, kick_strong_range=20.0, kick_pose_x=(0.0, -0.35), kick_pose_y_off=0.06,
    ball_y_dis_ratio=(2.0, 2.0), ball_distance_offset=0.3, ball_rotation_offset=0.1, ball_avoidance_range=0.55,
)
# walk_command.BallWalkVelocityCommandCfg
WCFG = dict(gain_xy=1.5, vx_range=(-0.5, 1.0), max_vy=0.5, gain_yaw=1.5, max_wz=1.0, near_dist=0.6, blend_dist=0.4,
            kick_through_speed=0.0)
BASE_INIT = np.array([0.0, 0.0, 0.6])
wrap, rot2 = km.wrap, km.rot2


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
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(out.parent, os.W_OK):
            raise PermissionError(out.parent)
    except PermissionError:
        # 学習の run ディレクトリはコンテナ (root) が作るので、ホストから書けないことがある
        out = Path("/tmp") / f"bwk_onnx_{os.getuid()}" / out.name
        out.parent.mkdir(parents=True, exist_ok=True)
    ex = (torch.zeros(1, 3), torch.zeros(1, HISTORY, STEP_DIM))
    torch.onnx.export(pol, ex, str(out), input_names=["command", "obs_history"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    return out


# --------------------------------------------------------------------------- #
# MuJoCo model
# --------------------------------------------------------------------------- #
def build_model(friction: float, ball_friction: float, ball_foot_friction: float | None, ball_r: float,
                ball_m: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.body("Trunk").add_freejoint()
    robot_geoms = []
    for i, g in enumerate(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            continue  # 見た目だけのメッシュ (URDF の visual) は当てない (2026-10-26 まではこれも当たっていた)
        # ロボット同士は当たらない (学習: enabled_self_collisions=False)、床とボールには当たる
        g.contype, g.conaffinity = 1, 0
        g.friction = [friction, 0.005, 0.0001]
        if not g.name:
            g.name = f"robot_geom_{i}"
        robot_geoms.append(g.name)
    km.add_world(spec, robot_geoms, friction, ball_friction, ball_foot_friction, ball_r, ball_m)
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


class Sim:
    def __init__(self, onnx_path: Path, friction: float, delay: int, ball_delay: int, ball_friction: float = 1.0,
                 ball_foot_friction: float | None = None, ball_r: float = km.BALL_R_DEFAULT,
                 ball_m: float = km.BALL_M_DEFAULT):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.m = build_model(friction, ball_friction, ball_foot_friction, ball_r, ball_m)
        self.d = mujoco.MjData(self.m)
        self.w = km.World(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in JOINTS])
        self.act = np.array([self.m.actuator(n).id for n in JOINTS])
        self.trunk = self.w.trunk
        self.feet = self.w.feet
        self.delay = int(delay)
        self.ball_delay = int(ball_delay)
        self.ball_r = ball_r
        self.ctrl_dt = STEP_DT
        self.cur = 0

    # ------------------------------------------------------------------ #
    def ball_w(self):
        return self.d.xpos[self.w.balls[self.cur]].copy()

    def ball_v(self):
        v = self.w.ball_v[self.cur]
        return self.d.qvel[v:v + 3]

    def reset(self, sc: km.Scenario):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = BASE_INIT
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = DEFAULT_Q
        self.w.place_ball(d, 0, sc.ball_xy, self.ball_r)
        self.w.place_ball(d, 1, km.PARK_XY[1], self.ball_r)
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros(13)
        self.phase = 0.0
        self.target_buf = deque([DEFAULT_Q.copy()] * (self.delay + 1), maxlen=self.delay + 1)
        gdir = wrap(self.yaw() + math.radians(sc.kick_deg))
        range0 = KCFG["kick_strong_range"] if sc.strong else sc.range_m
        self.set_task(0, gdir, sc.strong, range0)
        self.expected_vel0 = math.sqrt(2.0 * KCFG["ball_friction"] * range0)
        self.history = None

    def set_task(self, idx: int, gdir: float, strong: bool, range_m: float):
        """学習の _reset_ball_idx / _resample_kick_ball: 見るボールとキック指令を変える。"""
        self.cur = idx
        bp = self.ball_w()
        self.ball_buf = deque([bp.copy()] * 4, maxlen=4)
        self.ref_yaw = self.yaw()
        self.target = bp[:2] + min(range_m, KCFG["kick_strong_range"]) * np.array([math.cos(gdir), math.sin(gdir)])
        self.d.mocap_pos[self.w.target_mocap] = [bp[0] + 0.6 * math.cos(gdir), bp[1] + 0.6 * math.sin(gdir), 0.002]
        self.strong = strong
        self.ball_close_to_kick_pose = 1.0
        self.ball_pos = self.ball_rel(bp)
        dac = wrap(gdir - self.yaw())
        self.filtered_ball_pos = rot2(self.ball_pos, -dac)
        self.kick_pose_translation_ratio = 0.0
        self.ball_kick_pose = bp[:2] + np.array([math.cos(gdir), math.sin(gdir)]) * KCFG["kick_pose_x"][1]
        self.update_kick_state()
        self.command = self.walk_command()

    def kick_direction(self) -> float:
        return self.gdir

    def yaw(self):
        return km.body_yaw(self.d, self.trunk)

    def origin(self):
        pts = []
        for b in self.feet:
            R = self.d.xmat[b].reshape(3, 3)
            pts.append(self.d.xpos[b] + R @ km.SOLE_OFFSET)
        return np.mean(pts, axis=0)

    def ball_rel(self, ball_w):
        return rot2(ball_w[:2] - self.origin()[:2], -self.yaw())

    # ------------------------------------------------------------------ #
    # 学習環境の式 (1 env 分)
    # ------------------------------------------------------------------ #
    def update_kick_state(self):
        """kick_command._update の 3.3, 4, 6, 8 (観測と歩行コマンドに要る部分)。"""
        k = KCFG
        yaw = self.yaw()
        ball_w = self.ball_w()
        delayed = self.ball_buf[len(self.ball_buf) - 1 - self.ball_delay]
        self.ball_pos = self.ball_rel(delayed)
        self.ball_vel = rot2(self.ball_v()[:2], -yaw)
        to_target = self.target - ball_w[:2]
        self.gdir = math.atan2(to_target[1], to_target[0])
        self.range = min(float(np.linalg.norm(to_target)), k["kick_strong_range"])
        self.dac = wrap(self.gdir - yaw)
        # 6) キック方向に回したボール位置
        rot_ratio = np.clip((abs(self.dac) - k["ball_rotation_offset"]) / 0.4, 0.0, 1.0)
        bik = rot2(self.ball_pos, -self.dac)
        by = bik[1]
        dist = float(np.linalg.norm(self.ball_pos))
        # 8) キック姿勢
        y_ratio = np.clip((abs(by) - k["ball_y_dis_ratio"][0] * k["kick_pose_y_off"]) / (k["ball_y_dis_ratio"][1] * self.ball_r),
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
        self.ball_kick_pose = ball_w[:2] + np.array([math.cos(self.gdir), math.sin(self.gdir)]) * kp

    def walk_target(self):
        """rewards._walk_target (ロボットの yaw フレーム、胴体原点基準)。"""
        k = KCFG
        d = self.d
        yaw = self.yaw()
        base = d.qpos[0:3].copy()
        ball_w = self.ball_w()
        kpir = rot2(self.ball_kick_pose - base[:2], -yaw)
        extra = max((self.kick_pose_translation_ratio - 0.5) / 0.5, 0.0) * (self.ball_r + 0.1)
        x_off = -math.sin(self.dac) * (k["kick_pose_y_off"] + extra)
        y_off = math.cos(self.dac) * (k["kick_pose_y_off"] + extra)
        if self.filtered_ball_pos[1] < 0:
            wt = kpir + np.array([x_off, y_off])
        else:
            wt = kpir - np.array([x_off, y_off])
        gdir = self.gdir
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
    def step(self, hook=None):
        """制御 1 周期 (観測 → 方策 → 物理 4 ステップ)。hook(物理ステップの終わりの時刻 [s, 周期の頭から]) を呼ぶ。"""
        m, d = self.m, self.d
        f = self.frame()
        if self.history is None:  # CircularBuffer と同じく、最初のフレームで全履歴を埋める
            self.history = np.repeat(f[None], HISTORY, axis=0)
        else:
            self.history = np.concatenate((self.history[1:], f[None]), axis=0)
        a = self.sess.run(None, {"command": self.command[None].astype(np.float32),
                                 "obs_history": self.history[None]})[0][0].astype(np.float64)
        self.last_action = a.copy()
        target_q = DEFAULT_Q + ACTION_SCALE * a[:12]
        # 位相は「前ステップのコマンド + 今の周波数 action」で進む (学習: 報酬計算時に積分)
        self.phase = math.fmod(self.phase + 2 * math.pi * self.phase_freq(self.command, a[12]) * STEP_DT, 2 * math.pi)
        for i in range(DECIMATION):
            self.target_buf.append(target_q)
            d.ctrl[self.act] = self.target_buf[0]
            self.w.ball_damping(d)
            mujoco.mj_step(m, d)
            if hook is not None:
                hook((i + 1) * SIM_DT)
        self.ball_buf.append(self.ball_w())
        self.update_kick_state()
        self.command = self.walk_command()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="rsl_rl の model_*.pt (ONNX に書き出してから使う)")
    src.add_argument("--onnx", type=Path)
    ap.add_argument("--export_only", action="store_true")
    ap.add_argument("--duration", type=float, default=15.0, help="蹴るまでの上限 [s]")
    ap.add_argument("--post_kick", type=float, default=3.0, help="蹴ってからこの時間で打ち切る [s]")
    ap.add_argument("--next_ball_delay", type=float, default=0.4, help="蹴ってからこの時間で次のボールを置く [s]")
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--ball_friction", type=float, default=1.0, help="ボールと床の摩擦 (学習: 1.0)")
    ap.add_argument("--ball_foot_friction", type=float, default=0.8,
                    help="ボールとロボットの摩擦 (学習: 0.65〜1.0)。負なら pair を作らずボールの摩擦を使う")
    ap.add_argument("--ball_radius", type=float, default=km.BALL_R_DEFAULT)
    ap.add_argument("--ball_mass", type=float, default=km.BALL_M_DEFAULT)
    ap.add_argument("--delay", type=int, default=4, help="PD 目標の遅れ [物理ステップ] (学習: 2〜7)")
    ap.add_argument("--ball_delay", type=int, default=1, help="観測するボール位置の遅れ [制御ステップ] (学習: 1〜3)")
    ap.add_argument("--scenarios", type=str, default=None, help="カンマ区切りで場面を絞る")
    ap.add_argument("--random", type=int, default=0,
                    help="既定の 10 場面の代わりに N 個のランダムな場面 (kick_metrics.random_scenarios)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage_cfg", type=str, default=None,
                    help='KCFG の上書き (JSON)。例: \'{"ball_distance_offset": 0.7, "ball_rotation_offset": 0.2}\'')
    ap.add_argument("--kick_through_speed", type=float, default=0.0,
                    help="学習の walk_command と合わせる (Stage 2 以降 0、s1c〜前回の s6b/s7 は 0.5)")
    ap.add_argument("--video_dir", type=Path)
    ap.add_argument("--json", type=Path)
    km.add_scene_args(ap)
    a = ap.parse_args(argv)

    WCFG["kick_through_speed"] = a.kick_through_speed
    if a.stage_cfg:
        KCFG.update(json.loads(a.stage_cfg))
    onnx_path = a.onnx or export_onnx(a.checkpoint)
    print(f"[sim2sim] policy: {onnx_path}")
    if a.export_only:
        return 0
    bff = None if a.ball_foot_friction < 0 else a.ball_foot_friction
    if a.view or a.view_seconds > 0:
        sim = Sim(onnx_path, a.friction, a.delay, a.ball_delay, a.ball_friction, bff, a.ball_radius, a.ball_mass)
        km.run_viewer(sim, km.scene_from_args(a), a.next_ball_delay, a.view_seconds, a.speed)
        return 0
    if a.video_dir is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        a.video_dir.mkdir(parents=True, exist_ok=True)
    scenarios = km.random_scenarios(a.random, a.seed) if a.random > 0 else km.DEFAULT_SCENARIOS
    if a.scenarios:
        names = set(a.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in names]
    sim = Sim(onnx_path, a.friction, a.delay, a.ball_delay, a.ball_friction, bff, a.ball_radius, a.ball_mass)
    results = {}
    for sc in scenarios:
        video = a.video_dir / f"{sc.name}.mp4" if a.video_dir else None
        r = km.run_scene(sim, sc, a.duration, a.next_ball_delay, a.post_kick, video)
        results[sc.name] = r
        print(km.fmt(sc, r), flush=True)
    summ = km.summarize(scenarios, results)
    print("[summary]", json.dumps(summ, ensure_ascii=False, default=float), flush=True)
    print(km.spec_table(summ), flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"policy": str(onnx_path), "args": {k: str(v) for k, v in vars(a).items()},
                                      "summary": summ, "results": results}, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
