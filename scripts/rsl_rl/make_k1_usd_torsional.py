# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 の URDF を USD へ変換し、足裏に PhysX のトーショナル摩擦 (スピン抵抗) を書き込む。

**なぜ必要か**

実機・MuJoCo では足裏が地面をスピンする際にトーショナル摩擦が抵抗トルクを生むが、
IsaacLab の通常の設定経路ではこれを K1 に適用できない:

  * 摩擦係数 (``RigidBodyMaterialCfg``) は **並進の滑りにしか効かない**
  * スピン抵抗を決めるのは ``CollisionPropertiesCfg.torsional_patch_radius`` だが、
    URDF→USD 変換が collision prim をインスタンス化するため後から編集できない
    (startup イベントでも spawn の collision_props でも "instanced prim" で失敗)
  * ``AssetConverterBaseCfg.make_instanceable=False`` は **URDF 変換器が参照しない**
    (urdf_converter.py は set_make_instanceable を一度も呼ばず instanceable 固定)

その結果 Isaac 内では足裏が実質無抵抗でスピンでき、その場回転タスクは「接地したまま
捻る」戦略に居座った。摩擦 DR を平均 0.82→1.65 に上げても、踏み替え報酬を 5 倍にしても
片足支持時間は 0.0040→0.0037 と不変 (捻りのコストが 0 なので当然)。
sim2sim では Isaac 成功率 97.7% の 130-180° が MuJoCo でほとんど失敗していた。

**何をするか**

変換後の ``configuration/K1_locomotion_physics.usd`` を単体で開くと、足の collision prim
(``/colliders/*_foot_link/*_Foot/node_STL_BINARY_``) は **インスタンス化されていない**
ので直接編集できる。ここに ``PhysxCollisionAPI`` を適用して
``physxCollision:torsionalPatchRadius`` を書き込むと、上位 USD を合成する全 env の
インスタンスにその値が効く。

**使い方**

    bash /home/satoshi/workspace/IsaacLab-2.3.2/isaaclab.sh -p make_k1_usd_torsional.py

出力先 (既定): ``assets_soccer/booster_robotics_robots/K1/usd_torsional/K1_locomotion.usd``

**URDF を変更したら再実行すること。** 生成物は URDF のスナップショットなので、
URDF 側の更新は自動では反映されない。
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert K1 URDF to USD with foot torsional friction.")
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Path to the K1 URDF. Defaults to the one used by K1_LOCOMOTION_CFG.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Output directory for the converted USD. Defaults to assets_soccer/.../K1/usd_torsional.",
)
parser.add_argument(
    "--torsional_patch_radius",
    type=float,
    default=0.051,
    help="Contact patch radius for torsional friction [m]. 既定 0.051 は足裏 0.18 x 0.07 m に一様圧力を仮定したときの重心からの平均距離 (数値積分値)。τ≈μ·N·r の実効レバー長に対応する物理的な推定値。",
)
parser.add_argument(
    "--min_torsional_patch_radius", type=float, default=0.0, help="Minimum torsional patch radius [m]."
)
parser.add_argument(
    "--foot_regex", type=str, default="foot", help="Substring (lowercased) identifying foot collision prims."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_URDF = os.path.join(
    _REPO_ROOT, "assets_soccer", "booster_robotics_robots", "K1", "K1_locomotion.urdf"
)
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "assets_soccer", "booster_robotics_robots", "K1", "usd_torsional")

# 変換後の物理レイヤー。足の collision prim はここで定義されており、単体で開けば
# インスタンス化されていないため編集できる。
_PHYSICS_LAYER = os.path.join("configuration", "K1_locomotion_physics.usd")


def convert(urdf_path: str, out_dir: str) -> str:
    """K1_LOCOMOTION_CFG.spawn と同じ設定で URDF → USD 変換する。

    変換設定が本家 (rough_env_cfg.py の K1_LOCOMOTION_CFG) とずれると、関節の並びや
    コライダー形状が変わって観測レイアウトが壊れるので、値は必ず同期させること。
    """
    cfg = sim_utils.UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=out_dir,
        usd_file_name="K1_locomotion.usd",
        force_usd_conversion=True,
        # --- 以下 K1_LOCOMOTION_CFG.spawn と同値 ---
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    )
    converter = sim_utils.UrdfConverter(cfg)
    print(f"[INFO] converted: {converter.usd_path}")
    return converter.usd_path


def patch_torsional(usd_path: str, radius: float, min_radius: float, foot_key: str) -> int:
    """物理レイヤーの足 collision prim に torsional patch radius を書き込む。

    Returns:
        書き込んだ prim の数。
    """
    layer_path = os.path.join(os.path.dirname(usd_path), _PHYSICS_LAYER)
    if not os.path.isfile(layer_path):
        raise FileNotFoundError(
            f"物理レイヤーが見つかりません: {layer_path}\n"
            "URDF 変換器の出力構造が変わった可能性があります。"
            " 変換後ディレクトリを開いて collision prim を持つレイヤーを特定し、"
            " _PHYSICS_LAYER を更新してください。"
        )

    stage = Usd.Stage.Open(layer_path)
    n = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if foot_key not in prim.GetPath().pathString.lower():
            continue
        if prim.IsInstance() or prim.IsInstanceable():
            # ここに来たら変換器の出力構造が変わっている。黙って飛ばすと
            # 「設定したつもり」で学習が進むので明示的に落とす。
            raise RuntimeError(
                f"collision prim がインスタンス化されています: {prim.GetPath()}。"
                " この経路では編集できません。"
            )
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateTorsionalPatchRadiusAttr().Set(float(radius))
        api.CreateMinTorsionalPatchRadiusAttr().Set(float(min_radius))
        print(f"  [SET] {prim.GetPath()}  torsionalPatchRadius={radius}  min={min_radius}")
        n += 1

    if n == 0:
        raise RuntimeError(
            f"足の collision prim が 1 つも見つかりませんでした (key='{foot_key}')。"
            " --foot_regex を確認してください。"
        )
    stage.GetRootLayer().Save()
    return n


def verify(usd_path: str, foot_key: str) -> None:
    """トップレベル USD から合成した状態で、値が読めることを確認する。"""
    stage = Usd.Stage.Open(usd_path)
    found = 0
    for prim in stage.Traverse():
        if foot_key not in prim.GetPath().pathString.lower():
            continue
        api = PhysxSchema.PhysxCollisionAPI(prim)
        if not api:
            continue
        attr = api.GetTorsionalPatchRadiusAttr()
        if attr and attr.HasAuthoredValue():
            print(f"  [VERIFY] {prim.GetPath()} -> torsionalPatchRadius={attr.Get()}")
            found += 1
    if found == 0:
        print(
            "[WARNING] トップレベル USD からは torsionalPatchRadius が読めませんでした。"
            " インスタンス化により合成結果に現れないだけの可能性がありますが、"
            " 実際にシミュレーションで効いているかは学習ログで確認してください。"
        )
    else:
        print(f"[INFO] verify: {found} prims に値が composed されています。")


def main():
    urdf_path = args_cli.urdf or _DEFAULT_URDF
    out_dir = args_cli.out_dir or _DEFAULT_OUT
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF が見つかりません: {urdf_path}")

    print(f"[INFO] URDF : {urdf_path}")
    print(f"[INFO] OUT  : {out_dir}")
    usd_path = convert(urdf_path, out_dir)

    print("[INFO] patching torsional friction ...")
    n = patch_torsional(
        usd_path, args_cli.torsional_patch_radius, args_cli.min_torsional_patch_radius, args_cli.foot_regex
    )
    print(f"[INFO] patched {n} foot collision prims")

    print("[INFO] verifying ...")
    verify(usd_path, args_cli.foot_regex)
    print(f"\n[DONE] {usd_path}")
    print("  K1FlatTurnCfg はこの USD を UsdFileCfg で読み込む。")
    print("  URDF を変更したら本スクリプトを再実行すること。")


if __name__ == "__main__":
    main()
    simulation_app.close()
