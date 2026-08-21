# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``configs/gk_prediction.yaml`` を読む共有定数ローダ。

★ この YAML は **sim と実機/MuJoCo デプロイノードの唯一の正**。片側だけ値を変えると
  方策が学習時と違う入力を受け取る (過去に goal half-width で同じズレを起こしている)。

読み込みは 1 回だけ行いキャッシュする。ファイルが無い/壊れている場合は
``_FALLBACK`` を使い、**警告を出す** (黙って旧値に戻らないようにする)。
"""

from __future__ import annotations

import os

# YAML が読めなかったときの値。**YAML と同じ値を書いておくこと**。
_FALLBACK = {
    "prediction": {"closing_min": 0.3, "t_max": 3.0, "idle_center_wait": False},
    "drive": {
        "horizon_fast": 0.15, "horizon_idle": 1.0,
        "deadband_y": 0.0, "vy_scale": 1.3, "vx_scale": 1.0,
    },
    # 横追従 (y_track) のゲート。詳細は YAML 側のコメント参照。
    "tracking": {
        "receding_vx_max": 0.3, "min_x": 0.2, "max_x": 4.0, "y_max": 1.3, "y_exit": 1.5,
        "gated_target": "center",
    },
}

_CACHE: dict | None = None


def _repo_root() -> str:
    # .../source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/goalkeeper/mdp
    here = os.path.dirname(os.path.abspath(__file__))
    # mdp/ から 7 段上がるとリポジトリ直下
    #   goalkeeper/mdp -> goalkeeper -> manager_based -> tasks -> <pkg> -> <pkg> -> source -> リポジトリ直下
    root = os.path.abspath(os.path.join(here, *([os.pardir] * 7)))
    # 環境変数で明示指定できる (別の場所にチェックアウトした場合の保険)
    return os.environ.get("GK_REPO_ROOT", root)


def gk_shared() -> dict:
    """共有定数を返す (初回だけ読み込む)。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = os.path.join(_repo_root(), "configs", "gk_prediction.yaml")
    data = None
    err = None
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        err = exc
    if err is not None or not isinstance(data, dict):
        # ★ 既定では **落とす**。フォールバックが黙って効くと、YAML を書き換えた後に
        #   「片方だけ古い値で走る」事故になる。値が同じうちは無害だが、気づくのは
        #   学習が終わってからになる (params/env.yaml を見ないと分からない)。
        #   読めない環境 (yaml 未導入の推論コンテナ等) では環境変数で明示的に許可する。
        msg = (
            f"[gk_shared] {path} を読めませんでした ({err or 'YAML の中身が dict ではない'})。\n"
            "  この値は sim と実機デプロイの唯一の正なので、黙って既定値へ倒しません。\n"
            "  意図的にフォールバックするなら GK_SHARED_ALLOW_FALLBACK=1 を設定してください。"
        )
        if os.environ.get("GK_SHARED_ALLOW_FALLBACK", "") not in ("1", "true", "True"):
            raise RuntimeError(msg)
        print("★★ " + msg + "\n  -> GK_SHARED_ALLOW_FALLBACK により既定値で続行します。")
        data = {}
    merged = {k: dict(v) for k, v in _FALLBACK.items()}
    for sec, vals in data.items():
        if sec in merged and isinstance(vals, dict):
            merged[sec].update(vals)
    _CACHE = merged
    src = "既定値 (フォールバック)" if not data else path
    print(f"[gk_shared] 読込元: {src}")
    print(f"[gk_shared]   prediction: {merged['prediction']}")
    print(f"[gk_shared]   drive     : {merged['drive']}")
    print(f"[gk_shared]   tracking  : {merged['tracking']}")
    return merged


def pred(key: str):
    return gk_shared()["prediction"][key]


def drive(key: str):
    return gk_shared()["drive"][key]


def tracking(key: str):
    return gk_shared()["tracking"][key]
