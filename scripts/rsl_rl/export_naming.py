# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""エクスポート成果物 (policy の TorchScript / ONNX) のファイル名を組み立てる。

なぜ固定名 ``policy.onnx`` をやめるのか
---------------------------------------
``exported/policy.onnx`` は **どの run のどの checkpoint から出たのかがファイル名から
分からない**。実機へ持っていく成果物なので、これは実害になる:

* ``fetch_ckpt.sh --onnx`` は「一番新しい run の一番大きい step」を自動で選ぶ。
  取ってきた ``policy.onnx`` が本当に狙った checkpoint のものか、名前では確認できない
  (実際に「これ model_3600.pt のやつ?」と疑う場面が出た。重みを直接照合しないと
  確かめられなかった)。
* 同じ ``exported/`` へ別の checkpoint を書き出すと **黙って上書き**される。
* 実機側 (``fetch_onnx_ml3.sh`` → ``booster_k1_locomotion/assets``) に複数世代を
  並べたとき、``policy.onnx`` が何個あっても見分けられない。

そこで **エクスポート時刻と タスク名 (experiment_name) と checkpoint の step** を
名前に焼き込む::

    k1_walk_inside_kick_model_3600_20260822-215713.onnx
    ^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^ ^^^^^^^^^^^^^^^
    experiment_name     checkpoint  エクスポート時刻

``model_3600`` の部分は元の checkpoint ファイル名 (``model_3600.pt``) をそのまま
写したもの。これで「どの .pt から出たか」がファイル名だけで確定する。

時刻について
------------
**エクスポートを実行したマシンのローカル時刻**を使う。run ディレクトリ名
(``2026-08-22_11-56-42``) と同じ流儀で、``fetch_ckpt.sh --onnx`` のように学習した
サーバ上でそのままエクスポートする場合は両者の時計が一致する。

.. note::
    学習サーバ (vast.ai など) と手元とで TZ が違うと、ファイル名の時刻は
    **サーバ側の時計**になる。手元の時計と数時間ずれて見えるのは仕様。
    区別したいのは「いつ落としたか」ではなく「どの重みか」で、そちらは
    ``model_<step>`` の側が担っている。
"""

from __future__ import annotations

import os
import re
import time

# 成果物の拡張子 (このモジュールが面倒を見る対象)。
ARTIFACT_SUFFIXES = (".onnx", ".pt")

# rsl_rl が書く checkpoint のファイル名。
_CKPT_RE = re.compile(r"^(model_\d+)\.pt$")


def checkpoint_tag(checkpoint_path: str) -> str:
    """checkpoint のパスから名前に埋め込む識別子を作る。

    ``.../model_3600.pt`` -> ``model_3600``。rsl_rl の命名でない場合は拡張子を
    落とした basename をそのまま使う (``best.pt`` -> ``best``)。
    """
    base = os.path.basename(checkpoint_path)
    m = _CKPT_RE.match(base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0] or "ckpt"


def exported_basename(experiment_name: str, checkpoint_path: str, when: float | None = None) -> str:
    """エクスポート成果物の拡張子なしファイル名を返す。

    Args:
        experiment_name: ``agent_cfg.experiment_name`` (= ``logs/rsl_rl/`` 配下の
            ディレクトリ名。例 ``k1_walk_inside_kick``)。
        checkpoint_path: 元にした checkpoint の絶対/相対パス。
        when: エクスポート時刻 (``time.time()`` 互換の epoch 秒)。既定は現在時刻。
            **同じ呼び出しで .pt と .onnx を書くときは同じ値を渡すこと**
            (秒をまたぐと 2 つの成果物で時刻がずれる)。

    Returns:
        ``k1_walk_inside_kick_model_3600_20260822-215713`` のような文字列。
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when if when is not None else time.time()))
    return f"{experiment_name}_{checkpoint_tag(checkpoint_path)}_{stamp}"


def latest_artifact(export_dir: str, suffix: str = ".onnx") -> str | None:
    """``export_dir`` にある最新 (mtime 最大) の成果物のパスを返す。

    固定名が無くなったので、**名前を知らない側**はこれで拾う。
    見つからなければ ``None``。
    """
    if not os.path.isdir(export_dir):
        return None
    cands = [
        os.path.join(export_dir, f)
        for f in os.listdir(export_dir)
        if f.endswith(suffix) and os.path.isfile(os.path.join(export_dir, f))
    ]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)
