"""MuJoCo でのキック確認に共通の部品: 場面、ボール 2 個の世界、仕様の各項目の測り方と集計。

仕様: docs/bhuman_walk_kick_spec.md の「合格条件」の表。うちの方策 (sim2sim_bhuman_walk_kick.py) と
B-Human の配布方策 (sim2sim_bhuman_reference.py) を同じ測り方で比べるため、場面の進め方と測定はここにまとめる。

1 場面の流れ (run_scene):

1. ロボットは原点で +x を向いて立つ。ボール 0 を場面の位置に置き、キック方向・強さを指令する。
2. ボール 0 が動き出したら (水平速度 0.3 m/s 超・2 cm 以上移動) 「キックした」とする。
3. キックから ``next_ball_delay`` 秒後 (既定 0.4 秒 = B-Human の time_after_kick 20 ステップ) に、ボール 1 を
   ロボットの前方 (胴体の向き ±60°) 1.0〜1.5 m に置き、新しいキック方向を指令する (B-Human の学習でボールを
   置き直すのと同じ)。蹴ったボール 0 はそのまま転がしておく (向き・距離はボール 0 で測る)。
4. キックから ``post_kick`` 秒 (既定 3 秒) で終わる。転倒したらその時点で終わる。

測る値 (キックした場面ごと):

* 当たった瞬間 = ボール 0 とロボットの接触が始まった物理ステップのうち、キック判定の直前のもの。
* 蹴り足 = 当たった瞬間にボールに近い方の足 (足裏の 4 隅からボール中心までの水平距離の最小で比べる)。
* 蹴り足の向き = |足の前方向の yaw - キック方向| (0〜180°)。90° でインサイド (横向き)、0° でつま先を向けている。
  符号付きの値 (左足は +、右足は - を掛ける) が +90° なら足の内側がキック方向を向いている、-90° なら外側。
* ボールの位置 (蹴り足の足首 = foot_link 原点から、足の yaw で回した座標): 前 x と 内側 y_in
  (左足は -y、右足は +y が内側)。y_in > |x| (足首から見て内側 ±45° の扇の中) を「内側」とする。
* 止まる (減速): 当たる前 1.5 秒 (ただし場面開始 0.5 秒以降) の胴体の水平速度 (制御周期ごとの値) の最低値が
  0.6 m/s 以下なら合格。参考に 0.3 m/s 以下が続いた一番長い区間 (stop_s) も出す。
* 軸足: 当たった瞬間に蹴っていない方の足が床に触れていて、その接地が 0.1 秒以上前から続いていれば合格
  (制御周期ごとの「その周期の中で一度でも床に触れたか」で数える)。
* 振り: 当たった瞬間に蹴り足が床に触れておらず、蹴り足 (足首) の水平速度が 1.0 m/s 以上なら合格。
* 向きのずれ: ボール 0 のキックから 1 秒間の移動の向きと指令方向の差。平均と、30° を超えた場面の割合。
* ボール速度: キックから 0.4 秒間のボール 0 の水平速度の最高値 (強キックはこの平均、距離指定は必要な初速との差)。
* つながり: ボール 1 を置いてからキック後 3 秒までに、胴体の水平速度 (0.2 秒平均) が 0.3 m/s 以上になり、かつ
  ボール 1 までの距離が 0.1 m 以上縮んでいれば合格。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

BALL_R_DEFAULT, BALL_M_DEFAULT = 0.095, 0.29  # B-Human K1_Ball.yaml asset.ball
BALL_LIN_DAMP, BALL_ANG_DAMP = 0.75, 0.5  # B-Human _create_ball_asset (PhysX のダンピングを外力で再現)
FEET_EDGES = np.array([[0.1185, 0.035, -0.038], [0.1185, -0.035, -0.038], [-0.0665, 0.035, -0.038], [-0.0665, -0.035, -0.038]])
SOLE_OFFSET = np.array([0.0, 0.0, -0.038])
PARK_XY = ((40.0, 40.0), (40.0, 45.0))  # 使っていないボールの置き場

# 仕様の閾値 (docs/bhuman_walk_kick_spec.md)
SPEC = dict(
    yaw_center_deg=90.0, yaw_tol_deg=25.0, frac=0.8,
    # 止まる・軸足は B-Human の K1_kick_inner.onnx で較正した値 (2026-10-26、docs/bhuman_walk_kick_spec.md)。
    # 最初の案 (0.3 m/s 以下が 0.4 秒続く / 軸足 0.3 秒 / 向きのずれ最大 30°) は本物が 5% / 2% / 6 回中 4 回しか
    # 通らなかった。
    stop_speed=0.6, stop_window_s=1.5, stop_ignore_first_s=0.5, slow_run_speed=0.3,
    plant_min_s=0.1, swing_foot_speed=1.0,
    dir_mean_deg=10.0, dir_big_deg=30.0, dir_big_frac=0.03, strong_speed=3.0, normal_err=0.8, max_falls=2,
    walkoff_window_s=3.0, walkoff_speed=0.3, walkoff_avg_s=0.2, walkoff_closer_m=0.1, walkoff_frac=0.9,
)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def body_yaw(d, body_id) -> float:
    R = d.xmat[body_id].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


# --------------------------------------------------------------------------- #
# 場面
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str
    ball_xy: tuple[float, float]  # ロボット初期姿勢 (原点・+x 向き) からの位置 [m]
    kick_deg: float  # キック方向 (ロボットの初期向き基準) [deg]
    strong: bool
    range_m: float = 3.0  # 距離指定キックの距離 [m] (strong のときは 20 m)
    seed: int = 0  # 次のボールの置き場所の乱数


DEFAULT_SCENARIOS = [
    Scenario("front_strong", (1.0, 0.0), 0.0, True, seed=100),
    Scenario("front_normal_2m", (1.0, 0.0), 0.0, False, 2.0, seed=101),
    Scenario("front_normal_5m", (1.0, 0.0), 0.0, False, 5.0, seed=102),
    Scenario("left30_strong", (1.0, 0.3), 30.0, True, seed=103),
    Scenario("right30_strong", (1.0, -0.3), -30.0, True, seed=104),
    Scenario("left60_normal_4m", (0.8, 0.4), 60.0, False, 4.0, seed=105),
    Scenario("right90_strong", (1.2, 0.0), -90.0, True, seed=106),
    Scenario("left90_strong", (1.2, 0.0), 90.0, True, seed=107),
    Scenario("behind_strong", (-0.8, 0.3), 180.0, True, seed=108),
    Scenario("far_front_strong", (2.5, 0.5), 0.0, True, seed=109),
]


def random_scenarios(n: int, seed: int) -> list[Scenario]:
    """ボール 0.5〜2 m・全方向、キック方向 全方向、強キック / 距離指定 (1〜8 m の対数一様) 半々。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        dist, ang = rng.uniform(0.5, 2.0), rng.uniform(-math.pi, math.pi)
        kick = rng.uniform(-180.0, 180.0)
        strong = bool(rng.random() < 0.5)
        rng_m = float(np.exp(rng.uniform(math.log(1.0), math.log(8.0))))
        out.append(Scenario(f"rand{i:02d}_{'S' if strong else 'N'}{kick:+04.0f}",
                            (dist * math.cos(ang), dist * math.sin(ang)), kick, strong, rng_m, seed=seed * 1000 + i))
    return out


# --------------------------------------------------------------------------- #
# 世界 (床・ボール 2 個・目標の印)
# --------------------------------------------------------------------------- #
def add_world(spec, robot_geom_names: list[str], friction: float, ball_friction: float, ball_foot_friction: float | None,
              ball_r: float, ball_m: float, n_balls: int = 2):
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=0, conaffinity=1,
                   friction=[friction, 0.005, 0.0001], rgba=[0.35, 0.55, 0.35, 1])
    world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    names = set(robot_geom_names)
    robot_bodies = {g.parent.name for g in spec.geoms if g.name in names}
    colors = ([1.0, 0.45, 0.0, 1], [0.2, 0.4, 1.0, 1])
    for i in range(n_balls):
        ball = world.add_body(name=f"ball{i}", pos=[PARK_XY[i][0], PARK_XY[i][1], ball_r])
        ball.add_freejoint()
        ball.add_geom(name=f"ball{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_r, 0, 0], mass=ball_m, contype=1,
                      conaffinity=1, condim=3, friction=[ball_friction, 0.005, 0.0001], rgba=colors[i % 2],
                      priority=1)  # ボールの接触はボールの摩擦を使う (MuJoCo 既定は大きい方)
        if ball_foot_friction is not None:
            # 学習 (PhysX) では ボール-床 = 1.0、ボール-足 = (ボール 1.0 + 足 0.3〜1.0) / 2 で 0.65〜1.0。
            # ボールとロボットの接触だけ明示の pair にして摩擦を別に決める。
            for b in robot_bodies:
                spec.add_exclude(bodyname1=f"ball{i}", bodyname2=b)
            for gname in robot_geom_names:
                spec.add_pair(geomname1=f"ball{i}", geomname2=gname, condim=3,
                              friction=[ball_foot_friction, ball_foot_friction, 0.005, 0.0001, 0.0001])
    target = world.add_body(name="target_marker", mocap=True, pos=[0, 0, 0.002])
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.002, 0], contype=0, conaffinity=0,
                    rgba=[1.0, 0.0, 0.0, 0.8])


class World:
    """コンパイル後のモデルから、ボール・足・床などの番号を引いておく。"""

    def __init__(self, m: mujoco.MjModel, n_balls: int = 2):
        self.m = m
        self.trunk = m.body("Trunk").id
        self.feet = [m.body("left_foot_link").id, m.body("right_foot_link").id]
        self.balls = [m.body(f"ball{i}").id for i in range(n_balls)]
        self.ball_geoms = [m.geom(f"ball{i}").id for i in range(n_balls)]
        self.ball_q = [m.joint(m.body_jntadr[b]).qposadr[0] for b in self.balls]
        self.ball_v = [m.joint(m.body_jntadr[b]).dofadr[0] for b in self.balls]
        self.floor = m.geom("floor").id
        self.target_mocap = m.body("target_marker").mocapid[0]
        self.trunk_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.trunk}
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in [0] + self.balls
                            and m.body_mocapid[m.geom_bodyid[g]] < 0}
        self.foot_geoms = [{g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for b in self.feet]
        self.ball_inertia = [m.body_inertia[b].copy() for b in self.balls]

    def place_ball(self, d, i, xy, r):
        q, v = self.ball_q[i], self.ball_v[i]
        d.qpos[q:q + 3] = [xy[0], xy[1], r]
        d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        d.qvel[v:v + 6] = 0.0

    def ball_damping(self, d):
        for i, b in enumerate(self.balls):
            v = d.qvel[self.ball_v[i]:self.ball_v[i] + 6]
            m = self.m.body_mass[b]
            f = np.zeros(6)
            f[:3] = -m * BALL_LIN_DAMP * v[:3]
            R = d.xmat[b].reshape(3, 3)
            w_local = R.T @ v[3:6]  # free joint の角速度はボール座標系
            f[3:] = R @ (-self.ball_inertia[i] * BALL_ANG_DAMP * w_local)
            d.xfrc_applied[b] = f

    def contact_flags(self, d, ball_idx: int):
        """(ロボット-ボール ball_idx の接触, 左足-床, 右足-床, 胴体-床)。"""
        bg = self.ball_geoms[ball_idx]
        touch = lf = rf = trunk = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == bg and g2 in self.robot_geoms) or (g2 == bg and g1 in self.robot_geoms):
                touch = True
            if g1 == self.floor or g2 == self.floor:
                o = g2 if g1 == self.floor else g1
                if o in self.foot_geoms[0]:
                    lf = True
                elif o in self.foot_geoms[1]:
                    rf = True
                elif o in self.trunk_geoms:
                    trunk = True
        return touch, lf, rf, trunk

    def foot_edges(self, d, k):
        R = d.xmat[self.feet[k]].reshape(3, 3)
        return d.xpos[self.feet[k]] + FEET_EDGES @ R.T


# --------------------------------------------------------------------------- #
# 記録
# --------------------------------------------------------------------------- #
class Recorder:
    """物理ステップごとにボール 0 との接触の始まりと足の接地を、制御周期ごとに胴体の速度と接地を記録する。"""

    def __init__(self, w: World, d):
        self.w, self.d = w, d
        self.ts, self.speed, self.base_xy, self.contact = [], [], [], []
        self.onsets = []
        self._touch = False
        self._step_contact = [False, False]
        self.trunk_floor = False
        self.ball_idx = 0  # 接触を見るボール (--view では蹴るたびに入れ替わる)

    def substep(self, t: float, kick_dir: float):
        touch, lf, rf, trunk = self.w.contact_flags(self.d, self.ball_idx)
        self._step_contact[0] |= lf
        self._step_contact[1] |= rf
        self.trunk_floor |= trunk
        if touch and not self._touch:
            self.onsets.append(self._snapshot(t, kick_dir, (lf, rf)))
        self._touch = touch

    def control(self, t: float):
        d = self.d
        self.ts.append(t)
        self.speed.append(float(np.linalg.norm(d.qvel[0:2])))
        self.base_xy.append(d.qpos[0:2].copy())
        self.contact.append(tuple(self._step_contact))
        self._step_contact = [False, False]

    def _snapshot(self, t, kick_dir, foot_floor):
        w, d = self.w, self.d
        ball = d.xpos[w.balls[self.ball_idx]].copy()
        vel = np.zeros(6)
        feet = []
        for k, b in enumerate(w.feet):
            e = w.foot_edges(d, k)
            mujoco.mj_objectVelocity(w.m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            feet.append(dict(edge_dist=float(np.min(np.linalg.norm(e[:, :2] - ball[:2], axis=1))),
                             yaw=body_yaw(d, b), pos=d.xpos[b].copy(), hvel=float(np.linalg.norm(vel[3:5])),
                             low_z=float(e[:, 2].min()), floor=bool(foot_floor[k])))
        bodies = set()
        bg = w.ball_geoms[self.ball_idx]
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == bg and c.geom2 in w.robot_geoms:
                bodies.add(w.m.body(w.m.geom_bodyid[c.geom2]).name)
            elif c.geom2 == bg and c.geom1 in w.robot_geoms:
                bodies.add(w.m.body(w.m.geom_bodyid[c.geom1]).name)
        m = w.m
        hip = [float(d.qpos[m.joint(n).qposadr[0]]) for n in ("Left_Hip_Yaw", "Right_Hip_Yaw")]
        return dict(t=t, kick_dir=kick_dir, ball=ball, feet=feet, bodies=sorted(bodies), trunk_yaw=body_yaw(d, w.trunk),
                    hip_yaw=hip)


def contact_metrics(rec: Recorder, kick_t: float, ctrl_dt: float) -> dict:
    """キックした場面の、当たった瞬間とその前後の値 (モジュールの説明の「測る値」)。"""
    S = SPEC
    ons = [o for o in rec.onsets if o["t"] <= kick_t + 1e-9]
    if not ons:
        return {}
    o = ons[-1]
    tc = o["t"]
    k = 0 if o["feet"][0]["edge_dist"] <= o["feet"][1]["edge_dist"] else 1
    f, sup = o["feet"][k], o["feet"][1 - k]
    rel = wrap(f["yaw"] - o["kick_dir"])
    yaw_abs = abs(math.degrees(rel))
    yaw_inside = math.degrees(rel) * (1.0 if k == 0 else -1.0)
    b = rot2(o["ball"][:2] - f["pos"][:2], -f["yaw"])
    x_f, y_in = float(b[0]), float(-b[1] if k == 0 else b[1])
    ts = np.array(rec.ts)
    sp = np.array(rec.speed)
    # 止まる (減速)
    m = (ts >= max(tc - S["stop_window_s"], S["stop_ignore_first_s"])) & (ts <= tc + 1e-9)
    run = best = 0
    for v in sp[m]:
        run = run + 1 if v <= S["slow_run_speed"] else 0
        best = max(best, run)
    stop_s = best * ctrl_dt
    min_speed_win = float(sp[m].min()) if m.any() else None
    swing_ok = (not f["floor"]) and f["hvel"] >= S["swing_foot_speed"]
    # 較正用の分布 (当たった瞬間を含む制御周期から遡る)
    upto = [i for i in range(len(ts)) if ts[i] - ctrl_dt < tc]
    i_tc = upto[-1] if upto else 0

    def streak(fn):
        n = 0
        for i in range(i_tc, -1, -1):
            if not fn(rec.contact[i]):
                break
            n += 1
        return n * ctrl_dt

    m15 = (ts >= tc - S["stop_window_s"]) & (ts <= tc + 1e-9)
    run05 = best05 = 0
    for v in sp[m15]:
        run05 = run05 + 1 if v <= 0.5 else 0
        best05 = max(best05, run05)
    return dict(
        touch_t=tc, kick_foot="L" if k == 0 else "R", touch_bodies=o["bodies"],
        foot_yaw_deg=yaw_abs, foot_yaw_inside_deg=yaw_inside,
        yaw_ok=abs(yaw_abs - S["yaw_center_deg"]) <= S["yaw_tol_deg"],
        yaw_inside_ok=abs(yaw_inside - S["yaw_center_deg"]) <= S["yaw_tol_deg"],
        ball_x_in_foot=x_f, ball_y_inside_foot=y_in, inside_ok=y_in > abs(x_f),
        stop_s=stop_s, min_speed_win=min_speed_win,
        stop_ok=min_speed_win is not None and min_speed_win <= S["stop_speed"],
        kick_foot_on_floor=f["floor"], kick_foot_hvel=f["hvel"], kick_foot_low_z=f["low_z"],
        swing_ok=swing_ok,
        # 胴体の向き - キック方向 (蹴り足が左なら +、右なら - を掛ける。足と同じ向きに回っていれば +)、
        # 蹴り足の股関節 yaw (外向き = 足先を外へ開く向きが +)
        trunk_yaw_inside_deg=math.degrees(wrap(o["trunk_yaw"] - o["kick_dir"])) * (1.0 if k == 0 else -1.0),
        hip_yaw_out_deg=math.degrees(o["hip_yaw"][k]) * (1.0 if k == 0 else -1.0),
        base_speed_touch=float(sp[i_tc]) if len(sp) else None,
        min_speed_pre=float(sp[m15].min()) if m15.any() else None,
        slow05_s=best05 * ctrl_dt,
        support_down_s=(sd := streak(lambda c: c[1 - k]) if sup["floor"] else 0.0),
        plant_ok=sd >= S["plant_min_s"] - 1e-9,
        kick_air_s=streak(lambda c: not c[k]) if not f["floor"] else 0.0,
    )


def walkoff_metrics(rec: Recorder, kick_t: float, place_t: float | None, ball1_xy, ctrl_dt: float, fell: bool) -> dict:
    S = SPEC
    if place_t is None or ball1_xy is None:
        return dict(walkoff_ok=False, walkoff_speed=None, walkoff_closer_m=None)
    ts = np.array(rec.ts)
    sp = np.array(rec.speed)
    n = max(1, int(round(S["walkoff_avg_s"] / ctrl_dt)))
    avg = np.convolve(sp, np.ones(n) / n, mode="full")[: len(sp)]
    m = (ts > place_t) & (ts <= kick_t + S["walkoff_window_s"] + 1e-9)
    if not m.any():
        return dict(walkoff_ok=False, walkoff_speed=None, walkoff_closer_m=None)
    idx = np.nonzero(m)[0]
    vmax = float(avg[idx].max())
    d0 = float(np.linalg.norm(rec.base_xy[idx[0]] - ball1_xy))
    d1 = float(np.linalg.norm(rec.base_xy[idx[-1]] - ball1_xy))
    reached_end = ts[idx[-1]] >= kick_t + S["walkoff_window_s"] - ctrl_dt - 1e-9
    ok = (not fell) and reached_end and vmax >= S["walkoff_speed"] and (d0 - d1) >= S["walkoff_closer_m"]
    return dict(walkoff_ok=bool(ok), walkoff_speed=vmax, walkoff_closer_m=d0 - d1)


# --------------------------------------------------------------------------- #
# 1 場面
# --------------------------------------------------------------------------- #
def run_scene(sim, sc: Scenario, duration: float = 15.0, next_ball_delay: float = 0.4, post_kick: float = 3.0,
              video=None, fps: int = 25) -> dict:
    """sim は reset(sc) / set_task(ball_idx, gdir, strong, range_m) / step(hook) / kick_direction() /
    expected_vel0 / ball_r / ctrl_dt / w (World) / d を持つ。"""
    w, d = sim.w, sim.d
    sim.reset(sc)
    rec = Recorder(w, d)
    rng = np.random.default_rng(sc.seed)
    renderer = None
    frames = []
    if video is not None:
        renderer = mujoco.Renderer(w.m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.0, -25, 135
    ball0 = d.xpos[w.balls[0]].copy()
    gdir0 = sim.kick_direction()
    out = dict(fell=False, fell_t=None, reason="", kick_t=None, kick_speed=0.0, kick_dir_err_deg=None,
               expected_kick_speed=sim.expected_vel0, strong=sc.strong, min_base_z=1.0)
    ctrl_dt = sim.ctrl_dt
    n_steps = int(round(duration / ctrl_dt))
    end_step = n_steps
    ball_at_kick = None
    place_t = None
    ball1_xy = None
    kick_window = []
    t = 0.0
    step = 0
    for step in range(n_steps):
        t0 = step * ctrl_dt

        def hook(sub_t, _t0=t0):
            rec.substep(_t0 + sub_t, sim.kick_direction() if sim.cur == 0 else gdir0)

        sim.step(hook)
        t = (step + 1) * ctrl_dt
        rec.control(t)
        bv = d.qvel[w.ball_v[0]:w.ball_v[0] + 3]
        speed = float(np.linalg.norm(bv[:2]))
        out["min_base_z"] = min(out["min_base_z"], float(d.qpos[2]))
        if out["kick_t"] is None and speed > 0.3 and np.linalg.norm(d.xpos[w.balls[0]][:2] - ball0[:2]) > 0.02:
            out["kick_t"] = t
            ball_at_kick = d.xpos[w.balls[0]][:2].copy()
            end_step = min(n_steps + int(round(post_kick / ctrl_dt)), step + int(round(post_kick / ctrl_dt)))
        kt = out["kick_t"]
        if kt is not None:
            if t - kt <= 0.4 + 1e-9:
                kick_window.append((speed, math.atan2(bv[1], bv[0])))
            if "speed_0p3" not in out and t - kt >= 0.3 - 1e-6:
                out["speed_0p3"] = speed
            if "disp_1s" not in out and t - kt >= 1.0 - 1e-6:
                disp = d.xpos[w.balls[0]][:2] - ball_at_kick
                out["disp_1s"] = float(np.linalg.norm(disp))
                out["disp_dir_err_deg"] = math.degrees(wrap(math.atan2(disp[1], disp[0]) - gdir0)) \
                    if np.linalg.norm(disp) > 0.05 else None
            if place_t is None and t - kt >= next_ball_delay - 1e-6:
                yaw = body_yaw(d, w.trunk)
                dist = rng.uniform(1.0, 1.5)
                ang = yaw + rng.uniform(-math.radians(60), math.radians(60))
                ball1_xy = d.qpos[0:2] + dist * np.array([math.cos(ang), math.sin(ang)])
                w.place_ball(d, 1, ball1_xy, sim.ball_r)
                mujoco.mj_forward(w.m, d)
                ndir = rng.uniform(-math.pi, math.pi)
                nstrong = bool(rng.random() < 0.5)
                sim.set_task(1, ndir, nstrong, 20.0 if nstrong else float(np.exp(rng.uniform(0.0, math.log(8.0)))))
                place_t = t
        if renderer is not None and step % max(1, int(round(1.0 / fps / ctrl_dt))) == 0:
            cb = d.xpos[w.balls[sim.cur]]
            cam.lookat[:] = [(d.qpos[0] + cb[0]) / 2, (d.qpos[1] + cb[1]) / 2, 0.3]
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
        if d.qpos[2] < 0.30:
            out.update(fell=True, fell_t=t, reason=f"base height {d.qpos[2]:.2f} < 0.30")
            break
        if rec.trunk_floor:
            out.update(fell=True, fell_t=t, reason="trunk touched ground")
            break
        if step + 1 >= end_step:
            break
    if kick_window:
        sp, ang = max(kick_window, key=lambda x: x[0])
        out["kick_speed"] = sp
        out["kick_dir_err_deg"] = math.degrees(wrap(ang - gdir0))
    if out["kick_t"] is not None and "disp_1s" not in out:  # 1 秒経たずに終わった (転倒)
        disp = d.xpos[w.balls[0]][:2] - ball_at_kick
        out["disp_1s"] = float(np.linalg.norm(disp))
        out["disp_dir_err_deg"] = math.degrees(wrap(math.atan2(disp[1], disp[0]) - gdir0)) \
            if np.linalg.norm(disp) > 0.05 else None
    if out["kick_t"] is not None:
        out.update(contact_metrics(rec, out["kick_t"], ctrl_dt))
        out.update(walkoff_metrics(rec, out["kick_t"], place_t, ball1_xy, ctrl_dt, out["fell"]))
    out["ball_disp_m"] = float(np.linalg.norm(d.xpos[w.balls[0]][:2] - ball0[:2]))
    out["robot_disp_m"] = float(np.linalg.norm(d.qpos[0:2]))
    out["sim_time_s"] = t
    if frames:
        import imageio

        imageio.mimsave(str(video), frames, fps=fps, macro_block_size=1)
    return out


def fmt(sc: Scenario, r: dict) -> str:
    fell = f"FELL {r['fell_t']:.2f}s" if r["fell"] else "ok"
    if r["kick_t"] is None:
        return f"{sc.name:18s} | {fell:10s} | no kick"
    de = r.get("disp_dir_err_deg")
    de = "   -" if de is None else f"{round(de):>4}"
    s = (f"{sc.name:18s} | {fell:10s} | kick {r['kick_t']:5.2f}s v {r['kick_speed']:.2f} (tgt {r['expected_kick_speed']:.2f})"
         f" dir {de}")
    if "foot_yaw_deg" in r:
        s += (f" | {r['kick_foot']} yaw {r['foot_yaw_inside_deg']:+4.0f} ball x {r['ball_x_in_foot']:+.2f} in {r['ball_y_inside_foot']:+.2f}"
              f" | stop {r['stop_s']:.2f}s plant {'Y' if r['plant_ok'] else 'n'} swing {'Y' if r['swing_ok'] else 'n'}"
              f" ({'air' if not r['kick_foot_on_floor'] else 'floor'} {r['kick_foot_hvel']:.1f}m/s)"
              f" | walkoff {'Y' if r.get('walkoff_ok') else 'n'} ({(r.get('walkoff_speed') or 0):.2f}m/s,"
              f" {(r.get('walkoff_closer_m') or 0):+.2f}m) | {','.join(r['touch_bodies'])}")
    else:
        s += " | (no contact onset)"
    return s


def summarize(scenarios, results: dict) -> dict:
    """仕様の表の各項目。割合は全場面を分母にする (蹴らなかった場面は不合格に数える)。"""
    S = SPEC
    n = len(results)
    rs = list(results.values())
    kicked = [r for r in rs if r["kick_t"] is not None]
    strong = [r for r in kicked if r["strong"]]
    normal = [r for r in kicked if not r["strong"]]
    errs = [abs(r["disp_dir_err_deg"]) for r in kicked if r.get("disp_dir_err_deg") is not None]

    def frac(key):
        return sum(1 for r in rs if r.get(key)) / n if n else 0.0

    def mean(key, pool=kicked):
        v = [r[key] for r in pool if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    s = dict(
        n=n, kicked=len(kicked), falls=sum(r["fell"] for r in rs),
        yaw_frac=frac("yaw_ok"), yaw_inside_frac=frac("yaw_inside_ok"), foot_yaw_mean_deg=mean("foot_yaw_deg"),
        foot_yaw_inside_mean_deg=mean("foot_yaw_inside_deg"),
        inside_frac=frac("inside_ok"), ball_x_in_foot_mean=mean("ball_x_in_foot"), ball_y_inside_mean=mean("ball_y_inside_foot"),
        stop_frac=frac("stop_ok"), stop_s_mean=mean("stop_s"), min_speed_win_mean=mean("min_speed_win"),
        base_speed_touch_mean=mean("base_speed_touch"),
        plant_frac=frac("plant_ok"), support_down_mean=mean("support_down_s"),
        trunk_yaw_inside_mean_deg=mean("trunk_yaw_inside_deg"), hip_yaw_out_mean_deg=mean("hip_yaw_out_deg"),
        swing_frac=frac("swing_ok"), kick_foot_air_frac=sum(1 for r in kicked if r.get("kick_foot_on_floor") is False) / n if n else 0.0,
        kick_foot_hvel_mean=mean("kick_foot_hvel"),
        dir_err_mean_deg=float(np.mean(errs)) if errs else None, dir_err_max_deg=float(np.max(errs)) if errs else None,
        dir_err_big_frac=sum(1 for e in errs if e > S["dir_big_deg"]) / n if n else 0.0,
        strong_speed_mean=float(np.mean([r["kick_speed"] for r in strong])) if strong else None,
        strong_speed_0p3_mean=float(np.mean([r.get("speed_0p3", 0.0) for r in strong])) if strong else None,
        normal_speed_abs_err_mean=float(np.mean([abs(r["kick_speed"] - r["expected_kick_speed"]) for r in normal])) if normal else None,
        walkoff_frac=frac("walkoff_ok"),
    )
    f = S["frac"]
    s["pass"] = {
        "蹴り方": s["yaw_frac"] >= f,
        "当て方": s["inside_frac"] >= f,
        "止まる": s["stop_frac"] >= f,
        "軸足": s["plant_frac"] >= f,
        "振り": s["swing_frac"] >= f,
        "向きの精度": s["dir_err_mean_deg"] is not None and s["dir_err_mean_deg"] <= S["dir_mean_deg"]
        and s["dir_err_big_frac"] <= S["dir_big_frac"] + 1e-9,
        "強キックの強さ": s["strong_speed_mean"] is not None and s["strong_speed_mean"] >= S["strong_speed"],
        "弱キックの強さ": s["normal_speed_abs_err_mean"] is not None and s["normal_speed_abs_err_mean"] <= S["normal_err"],
        "転倒": s["falls"] <= S["max_falls"],
        "つながり": s["walkoff_frac"] >= S["walkoff_frac"],
    }
    return s


def spec_table(s: dict) -> str:
    def p(x, fmt_="{:.0%}"):
        return "-" if x is None else fmt_.format(x)

    rows = [
        ("蹴り方", f"足の向き 90±25° {p(s['yaw_frac'])} (内側向きの符号まで合う {p(s['yaw_inside_frac'])}, "
                f"符号付き平均 {p(s['foot_yaw_inside_mean_deg'], '{:+.0f}°')} = 胴体 {p(s['trunk_yaw_inside_mean_deg'], '{:+.0f}°')}"
                f" + 股 yaw {p(s['hip_yaw_out_mean_deg'], '{:+.0f}°')} ほか)", "≥80%"),
        ("当て方", f"ボールが足の内側 {p(s['inside_frac'])} (足首から前 {p(s['ball_x_in_foot_mean'], '{:+.3f}')} m, "
                f"内側 {p(s['ball_y_inside_mean'], '{:+.3f}')} m)", "≥80%"),
        ("止まる", f"当たる前 1.5 秒に 0.6 m/s 以下まで減速 {p(s['stop_frac'])} (最低速度の平均 {p(s['min_speed_win_mean'], '{:.2f}')} m/s、"
                f"当たった瞬間 {p(s['base_speed_touch_mean'], '{:.2f}')} m/s)", "≥80%"),
        ("軸足", f"0.1 秒以上前から接地 {p(s['plant_frac'])} (接地の長さの平均 {p(s['support_down_mean'], '{:.2f}')} s)", "≥80%"),
        ("振り", f"遊脚かつ 1.0 m/s 以上 {p(s['swing_frac'])} (遊脚 {p(s['kick_foot_air_frac'])}, "
               f"足の水平速度 平均 {p(s['kick_foot_hvel_mean'], '{:.2f}')} m/s)", "≥80%"),
        ("向きの精度", f"平均 {p(s['dir_err_mean_deg'], '{:.1f}°')}、30° 超 {p(s['dir_err_big_frac'])} (最大 {p(s['dir_err_max_deg'], '{:.0f}°')})",
         "平均 ≤10° / 30° 超 ≤3%"),
        ("強キックの強さ", f"平均 {p(s['strong_speed_mean'], '{:.2f}')} m/s (0.3 秒後 {p(s['strong_speed_0p3_mean'], '{:.2f}')})", "≥3.0 m/s"),
        ("弱キックの強さ", f"必要な初速との差 平均 {p(s['normal_speed_abs_err_mean'], '{:.2f}')} m/s", "≤0.8 m/s"),
        ("転倒", f"{s['falls']} / {s['n']}", "≤2"),
        ("つながり", f"歩き出した {p(s['walkoff_frac'])}", "≥90%"),
    ]
    lines = [f"場面 {s['n']}、蹴った {s['kicked']}", "| 項目 | 値 | 条件 | 判定 |", "|---|---|---|---|"]
    for name, val, cond in rows:
        lines.append(f"| {name} | {val} | {cond} | {'PASS' if s['pass'][name] else '未達'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 場面の指定 (コマンドライン) と、画面で実時間に動かすモード (--view)
# --------------------------------------------------------------------------- #
def add_scene_args(ap):
    ap.add_argument("--scene", type=str, default=None,
                    help="--view の場面の名前 (既定の 10 場面の名前、または rand03_S+120 のような --random の場面名)")
    ap.add_argument("--ball_x", type=float, default=None, help="--view: ボールの位置 x [m] (ロボットの初期向き基準)")
    ap.add_argument("--ball_y", type=float, default=0.0, help="--view: ボールの位置 y [m]")
    ap.add_argument("--kick_deg", type=float, default=0.0, help="--view: キック方向 [deg] (ロボットの初期向き基準、左が +)")
    ap.add_argument("--strong", action="store_true", help="--view: 強キック (20 m)。付けなければ距離指定キック")
    ap.add_argument("--range_m", type=float, default=3.0, help="--view: 距離指定キックの距離 [m]")
    ap.add_argument("--view", action="store_true", help="MuJoCo の画面を開いて実時間で動かす (蹴った後も次のボールが出て続く)")
    ap.add_argument("--view_seconds", type=float, default=0.0,
                    help="--view を画面なしで指定秒数だけ回す (動作確認用)。0 なら画面を開く")
    ap.add_argument("--speed", type=float, default=1.0, help="--view の再生速度 (1.0 = 実時間)")


def scene_from_args(a) -> Scenario:
    if a.ball_x is not None:
        return Scenario("custom", (a.ball_x, a.ball_y), a.kick_deg, a.strong, a.range_m, seed=a.seed)
    name = a.scene or DEFAULT_SCENARIOS[0].name
    pool = {s.name: s for s in DEFAULT_SCENARIOS}
    if name.startswith("rand"):
        pool.update({s.name: s for s in random_scenarios(max(a.random, 100), a.seed)})
    if name not in pool:
        raise SystemExit(f"場面 {name} がない。既定の場面: {', '.join(s.name for s in DEFAULT_SCENARIOS)}")
    return pool[name]


def run_viewer(sim, sc: Scenario, next_ball_delay: float = 0.4, seconds: float = 0.0, speed: float = 1.0):
    """画面を開いて実時間で動かす。蹴ったら next_ball_delay 秒後に、もう 1 つのボールをロボットの前方
    (胴体の向き ±60°、1.0〜1.5 m) に置いて新しいキック方向を指令する (学習環境でボールを置き直すのと同じ)。
    蹴るたびに、当たった瞬間の蹴り足の向きなどを 1 行表示する。転倒したら 2 秒後に場面の最初からやり直す。"""
    import time

    w, d = sim.w, sim.d
    rng = np.random.default_rng(sc.seed)

    def start():
        sim.reset(sc)
        r = Recorder(w, d)
        r.ball_idx = sim.cur
        return r

    rec = start()
    t = 0.0
    kick_t = None
    ball_start = d.xpos[w.balls[sim.cur]][:2].copy()
    gdir_at_kick = sim.kick_direction()
    peak = 0.0
    n_kick = 0
    fell_t = None
    strong = sc.strong
    expected = sim.expected_vel0

    def one_step():
        nonlocal t, kick_t, ball_start, gdir_at_kick, peak, n_kick, fell_t, rec, strong, expected
        if fell_t is not None:
            if t - fell_t > 2.0:
                print("[view] 場面の最初からやり直す", flush=True)
                rec = start()
                t, kick_t, fell_t = 0.0, None, None
                ball_start = d.xpos[w.balls[sim.cur]][:2].copy()
                strong, expected = sc.strong, sim.expected_vel0
            else:
                mujoco.mj_step(w.m, d)
                t += w.m.opt.timestep
                return
        t0 = t
        sim.step(lambda sub_t: rec.substep(t0 + sub_t, sim.kick_direction()))
        t += sim.ctrl_dt
        rec.control(t)
        cur = sim.cur
        bv = d.qvel[w.ball_v[cur]:w.ball_v[cur] + 3]
        sp = float(np.linalg.norm(bv[:2]))
        if kick_t is None and sp > 0.3 and np.linalg.norm(d.xpos[w.balls[cur]][:2] - ball_start) > 0.02:
            kick_t, peak, gdir_at_kick = t, sp, sim.kick_direction()
        elif kick_t is not None and t - kick_t <= 0.4 + 1e-9:
            peak = max(peak, sp)
        if kick_t is not None and t - kick_t >= next_ball_delay - 1e-6:
            n_kick += 1
            disp = d.xpos[w.balls[cur]][:2] - ball_start
            err = math.degrees(wrap(math.atan2(disp[1], disp[0]) - gdir_at_kick))
            cm = contact_metrics(rec, kick_t, sim.ctrl_dt)
            kind = "強キック" if strong else f"距離指定 (必要な初速 {expected:.2f} m/s)"
            line = f"[kick {n_kick}] t={kick_t:5.2f}s {kind} ボール最高 {peak:.2f} m/s 向きのずれ {err:+.0f}°"
            if cm:
                line += (f" | 蹴り足 {cm['kick_foot']} 向き {cm['foot_yaw_inside_deg']:+.0f}° (90° でインサイド)"
                         f" ボール 前 {cm['ball_x_in_foot']:+.2f} m / 内側 {cm['ball_y_inside_foot']:+.2f} m"
                         f" | 止まった {cm['stop_s']:.2f}s 軸足 {'接地' if cm['plant_ok'] else '×'}"
                         f" 蹴り足 {'浮き' if not cm['kick_foot_on_floor'] else '接地'} {cm['kick_foot_hvel']:.1f} m/s")
            print(line, flush=True)
            nxt = 1 - cur
            yaw = body_yaw(d, w.trunk)
            dist = rng.uniform(1.0, 1.5)
            ang = yaw + rng.uniform(-math.radians(60), math.radians(60))
            xy = d.qpos[0:2] + dist * np.array([math.cos(ang), math.sin(ang)])
            w.place_ball(d, nxt, xy, sim.ball_r)
            mujoco.mj_forward(w.m, d)
            strong = bool(rng.random() < 0.5)
            rng_m = 20.0 if strong else float(np.exp(rng.uniform(0.0, math.log(8.0))))
            sim.set_task(nxt, rng.uniform(-math.pi, math.pi), strong, rng_m)
            expected = math.sqrt(2.0 * 0.3 * min(rng_m, 20.0))
            rec.ball_idx = nxt
            rec._touch = False
            ball_start = xy.copy()
            kick_t = None
        if d.qpos[2] < 0.30 or rec.trunk_floor:
            print(f"[view] 転倒 t={t:.2f}s", flush=True)
            fell_t = t

    if seconds > 0:  # 画面なしの動作確認
        while t < seconds:
            one_step()
        print(f"[view] {seconds:.0f} 秒回した: キック {n_kick} 回", flush=True)
        return n_kick
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(w.m, d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = w.trunk
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.0, -25.0, 135.0
        while v.is_running():
            wall = time.perf_counter()
            one_step()
            v.sync()
            dt = sim.ctrl_dt / max(speed, 1e-3) - (time.perf_counter() - wall)
            if dt > 0:
                time.sleep(dt)
    return n_kick
