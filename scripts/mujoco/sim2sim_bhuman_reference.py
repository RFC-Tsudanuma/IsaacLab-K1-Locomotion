#!/usr/bin/env python3
"""B-Human の配布方策 K1_kick_inner.onnx (インサイドキック) を MuJoCo (CPU) で動かし、仕様の表の各項目を測る。

仕様 (docs/bhuman_walk_kick_spec.md) の合格条件の数字を、本物の B-Human の方策で較正するためのもの。
場面の進め方・測り方は kick_metrics.py (うちの方策の sim2sim_bhuman_walk_kick.py と同じ)。

    # ランダム 100 場面 + 既定の 10 場面 (動画つき) を測る
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_reference.py --random 100 \
        --json logs/rsl_rl/k1_bhuman_walk_kick/bhuman_reference/random100.json
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_reference.py \
        --video_dir logs/rsl_rl/k1_bhuman_walk_kick/bhuman_reference/videos
    # 画面で見る (蹴った後も次のボールが出て続く)
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_reference.py --view --scene left90_strong
    ~/k1_deploy_env/bin/python scripts/mujoco/sim2sim_bhuman_reference.py --view --ball_x 1.0 --ball_y 0.3 --kick_deg 45 --strong

B-Human の学習環境 (IsaacGymRL/envs/t1_ball.py, K1_Ball.yaml) に合わせたところ:

* ロボット: B-Human の resources/K1/K1_locomotion.urdf (学習で使った 3 種類のロボットのうち足がメッシュのもの)。
  うちの K1_locomotion.urdf とは股関節の取り付け高さ (z -0.077 / うち -0.062)、すねの当たり判定 (メッシュ 0.8 倍 /
  うち円柱+箱)、腕の当たり判定 (なし / うちあり)、関節の範囲 (股 yaw ±1.03 / うち ±1.0 など) が違う。
* 関節: トルク指令。物理ステップ (2 ms) ごとに python で PD を計算する
  (トルク = kp (目標 - 角度) - kd × 関節速度 (差分を 0.22 / 0.78 で平滑化したもの))。
  kp 股 80 / 膝 80 / 足首 pitch 50 / 足首 roll 60、kd 4/4/2/2 (yaml の IsaacGym 用の値)、トルク上限 50/50/30/60/30/30、
  armature は yaml の値。目標は制御周期の中で ``--delay`` 物理ステップ目に切り替わる (学習: 0〜9 の一様乱数)。
* 制御: 2 ms × 10 = 50 Hz。action は ±2 で切り、目標角 = 既定角 (股 pitch -0.2, 膝 0.4, 足首 pitch -0.25) + action
  (scale 1.0) を関節の範囲で切る。13 番目は歩行周波数 1.5 Hz への足し分 (±0.5 で切る、前の周期の action を使う)。
* 観測 59 (t1_ball.py _compute_observations の並び、ノイズなし):
  重力 3, 角速度 3, ボール位置 (両足裏の中点・胴体の yaw 基準) ×0.1 2, キック方向/π 1, cos/sin(2π 位相) 2,
  関節角 - 既定角 12, 関節速度 (最後の物理ステップの差分) ×0.1 12, 前の action 13,
  強キック 1, 0, 0 (正確に蹴る), ボール速度 ×0.1 2, 0, sin/cos(キック方向) 2, 必要な初速 ×0.1 1, 前のボール位置 ×0.1 2。
  ボール位置は ``--ball_delay`` 制御周期前のもの (学習: 1〜3)。
* ボール: 半径 0.095 m、0.29 kg、ダンピング 0.75 / 0.5。床・摩擦はうちの方策の確認と同じ値。
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

BH = Path.home() / "MachineLearning/IsaacGymRL"
URDF_BH = BH / "resources/K1/K1_locomotion.urdf"
ONNX_BH = BH / "pre-trained/WalkAndKick/K1_kick_inner.onnx"

JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
DEFAULT_Q = np.array([-0.2, 0.0, 0.0, 0.4, -0.25, 0.0] * 2)
KP = np.array([80.0, 80.0, 80.0, 80.0, 50.0, 60.0] * 2)
KD = np.array([4.0, 4.0, 4.0, 4.0, 2.0, 2.0] * 2)
TORQUE_CLIP = np.array([50.0, 50.0, 30.0, 60.0, 30.0, 30.0] * 2)
ARMATURE = np.array([0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0565056, 0.0565056] * 2)
SIM_DT, DECIMATION = 0.002, 10
CTRL_DT = SIM_DT * DECIMATION
CLIP_ACTIONS = 2.0
FREQ_LIMIT = (-0.5, 0.5)
BASE_FREQ = 1.5
BALL_FRICTION_FORMULA = 0.3  # 必要な初速 = sqrt(2 × 0.3 × 距離) (B-Human BallPhysics)
STRONG_RANGE = 20.0
wrap, rot2 = km.wrap, km.rot2


def build_model(friction: float, ball_friction: float, ball_foot_friction: float | None, ball_r: float,
                ball_m: float, urdf: Path = URDF_BH) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = ""  # URDF の <mujoco><compiler meshdir="meshes/"> とファイル名の meshes/ が二重になるため
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.body("Trunk").add_freejoint()
    robot_geoms = []
    for i, g in enumerate(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            continue  # 見た目だけのメッシュは当てない
        g.contype, g.conaffinity = 1, 0  # ロボット同士は当たらない、床とボールには当たる
        g.friction = [friction, 0.005, 0.0001]
        if not g.name:
            g.name = f"robot_geom_{i}"
        robot_geoms.append(g.name)
    km.add_world(spec, robot_geoms, friction, ball_friction, ball_foot_friction, ball_r, ball_m)
    for i, name in enumerate(JOINTS):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = 1.0
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.ctrllimited = True
        a.ctrlrange = [-TORQUE_CLIP[i], TORQUE_CLIP[i]]
    model = spec.compile()
    for i, name in enumerate(JOINTS):
        adr = model.joint(name).dofadr[0]
        model.dof_armature[adr] = ARMATURE[i]
        model.dof_damping[adr] = 0.0
        model.dof_frictionloss[adr] = 0.0
    return model


class RefSim:
    def __init__(self, onnx_path: Path, friction: float, delay: int, ball_delay: int, ball_friction: float = 1.0,
                 ball_foot_friction: float | None = None, ball_r: float = km.BALL_R_DEFAULT,
                 ball_m: float = km.BALL_M_DEFAULT, urdf: Path = URDF_BH):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.m = build_model(friction, ball_friction, ball_foot_friction, ball_r, ball_m, urdf)
        self.d = mujoco.MjData(self.m)
        self.w = km.World(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in JOINTS])
        self.act = np.array([self.m.actuator(n).id for n in JOINTS])
        self.jrange = np.array([self.m.jnt_range[self.m.joint(n).id] for n in JOINTS])
        self.trunk = self.w.trunk
        self.feet = self.w.feet
        self.delay = int(delay)
        self.ball_delay = int(ball_delay)
        self.ball_r = ball_r
        self.ctrl_dt = CTRL_DT
        self.cur = 0
        self.base_z0 = self._standing_height()

    def _standing_height(self) -> float:
        """既定角で両足裏が床から 2 mm 浮く胴体の高さ。"""
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[0:3] = [0.0, 0.0, 0.6]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = DEFAULT_Q
        mujoco.mj_forward(self.m, d)
        low = min(self.w.foot_edges(d, k)[:, 2].min() for k in range(2))
        return 0.6 - low + 0.002

    # ------------------------------------------------------------------ #
    def ball_w(self):
        return self.d.xpos[self.w.balls[self.cur]].copy()

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

    def reset(self, sc: km.Scenario):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, self.base_z0]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = DEFAULT_Q
        self.w.place_ball(d, 0, sc.ball_xy, self.ball_r)
        self.w.place_ball(d, 1, km.PARK_XY[1], self.ball_r)
        mujoco.mj_forward(m, d)
        self.actions = np.zeros(13)
        self.gait = 0.0
        self.targets = DEFAULT_Q.copy()
        self.prev_q = d.qpos[self.qadr].copy()
        self.filt_v = np.zeros(12)
        self.cur_v = np.zeros(12)
        gdir = wrap(self.yaw() + math.radians(sc.kick_deg))
        range0 = STRONG_RANGE if sc.strong else sc.range_m
        self.set_task(0, gdir, sc.strong, range0)
        self.expected_vel0 = math.sqrt(2.0 * BALL_FRICTION_FORMULA * min(range0, STRONG_RANGE))

    def set_task(self, idx: int, gdir: float, strong: bool, range_m: float):
        """t1_ball.py の _reset_ball_idx / _resample_kick_ball: 見るボールとキック指令を変える。"""
        self.cur = idx
        bp = self.ball_w()
        self.ball_buf = deque([bp.copy()] * 4, maxlen=4)
        self.target = bp[:2] + min(range_m, STRONG_RANGE) * np.array([math.cos(gdir), math.sin(gdir)])
        self.d.mocap_pos[self.w.target_mocap] = [bp[0] + 0.6 * math.cos(gdir), bp[1] + 0.6 * math.sin(gdir), 0.002]
        self.strong = strong
        self.update_kick_state()
        self.last_ball_pos = self.ball_pos.copy()

    def kick_direction(self) -> float:
        return self.gdir

    def update_kick_state(self):
        """_update_kick_tensors の 3.3 と 4 (観測に要る部分)。"""
        yaw = self.yaw()
        ball_w = self.ball_w()
        delayed = self.ball_buf[len(self.ball_buf) - 1 - self.ball_delay]
        self.ball_pos = self.ball_rel(delayed)
        v = self.w.ball_v[self.cur]
        self.ball_vel = rot2(self.d.qvel[v:v + 2], -yaw)
        to_target = self.target - ball_w[:2]
        self.gdir = math.atan2(to_target[1], to_target[0])
        self.range = min(float(np.linalg.norm(to_target)), STRONG_RANGE)
        self.dac = wrap(self.gdir - yaw)

    def obs(self) -> np.ndarray:
        d = self.d
        R = d.xmat[self.trunk].reshape(3, 3)
        exp_v = math.sqrt(2.0 * BALL_FRICTION_FORMULA * self.range)
        return np.concatenate([
            R.T @ np.array([0.0, 0.0, -1.0]),
            d.qvel[3:6],  # free joint の角速度は胴体フレーム
            self.ball_pos * 0.1,
            [self.dac / math.pi],
            [math.cos(2 * math.pi * self.gait), math.sin(2 * math.pi * self.gait)],
            d.qpos[self.qadr] - DEFAULT_Q,
            self.cur_v * 0.1,
            self.actions,
            [1.0 if self.strong else 0.0, 0.0, 0.0],
            self.ball_vel * 0.1,
            [0.0],
            [math.sin(self.dac), math.cos(self.dac)],
            [exp_v * 0.1],
            self.last_ball_pos * 0.1,
        ]).astype(np.float32)

    def step(self, hook=None):
        """t1_ball.py step(): action → 物理 10 ステップ (PD をここで計算) → 位相 → ボールの状態。"""
        m, d = self.m, self.d
        o = self.obs()
        a = self.sess.run(None, {self.in_name: o[None]})[0][0].astype(np.float64)
        gait_freq = float(np.clip(self.actions[-1], *FREQ_LIMIT)) + BASE_FREQ  # 前の周期の action
        self.actions = np.clip(a, -CLIP_ACTIONS, CLIP_ACTIONS)
        tgt = np.clip(DEFAULT_Q + self.actions[:12], self.jrange[:, 0], self.jrange[:, 1])
        for i in range(DECIMATION):
            if i == self.delay:
                self.targets = tgt
            q = d.qpos[self.qadr]
            d.ctrl[self.act] = np.clip(KP * (self.targets - q) - KD * self.filt_v, -TORQUE_CLIP, TORQUE_CLIP)
            self.w.ball_damping(d)
            mujoco.mj_step(m, d)
            qn = d.qpos[self.qadr].copy()
            self.cur_v = (qn - self.prev_q) / SIM_DT
            self.filt_v = self.filt_v * 0.78 + self.cur_v * 0.22
            self.prev_q = qn
            if hook is not None:
                hook((i + 1) * SIM_DT)
        self.gait = math.fmod(self.gait + CTRL_DT * gait_freq, 1.0)
        self.last_ball_pos = self.ball_pos.copy()
        self.ball_buf.append(self.ball_w())
        self.update_kick_state()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=ONNX_BH)
    ap.add_argument("--urdf", type=Path, default=URDF_BH)
    ap.add_argument("--duration", type=float, default=15.0, help="蹴るまでの上限 [s]")
    ap.add_argument("--post_kick", type=float, default=3.0, help="蹴ってからこの時間で打ち切る [s]")
    ap.add_argument("--next_ball_delay", type=float, default=0.4, help="蹴ってからこの時間で次のボールを置く [s]")
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--ball_friction", type=float, default=1.0)
    ap.add_argument("--ball_foot_friction", type=float, default=0.8)
    ap.add_argument("--ball_radius", type=float, default=km.BALL_R_DEFAULT)
    ap.add_argument("--ball_mass", type=float, default=km.BALL_M_DEFAULT)
    ap.add_argument("--delay", type=int, default=4, help="PD 目標の切り替え [物理ステップ目] (学習: 0〜9)")
    ap.add_argument("--ball_delay", type=int, default=1, help="観測するボール位置の遅れ [制御周期] (学習: 1〜3)")
    ap.add_argument("--scenarios", type=str, default=None, help="カンマ区切りで場面を絞る")
    ap.add_argument("--random", type=int, default=0, help="既定の 10 場面の代わりに N 個のランダムな場面")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_dir", type=Path)
    ap.add_argument("--json", type=Path)
    km.add_scene_args(ap)
    a = ap.parse_args(argv)

    bff = None if a.ball_foot_friction < 0 else a.ball_foot_friction
    sim = RefSim(a.onnx, a.friction, a.delay, a.ball_delay, a.ball_friction, bff, a.ball_radius, a.ball_mass, a.urdf)
    print(f"[reference] policy: {a.onnx}  robot: {a.urdf}", flush=True)
    if a.view or a.view_seconds > 0:
        km.run_viewer(sim, km.scene_from_args(a), a.next_ball_delay, a.view_seconds, a.speed)
        return 0
    if a.video_dir is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        a.video_dir.mkdir(parents=True, exist_ok=True)
    scenarios = km.random_scenarios(a.random, a.seed) if a.random > 0 else km.DEFAULT_SCENARIOS
    if a.scenarios:
        names = set(a.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in names]
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
        a.json.write_text(json.dumps({"policy": str(a.onnx), "args": {k: str(v) for k, v in vars(a).items()},
                                      "summary": summ, "results": results}, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
