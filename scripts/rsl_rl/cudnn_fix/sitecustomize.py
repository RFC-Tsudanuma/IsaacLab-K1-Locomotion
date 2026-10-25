# コンテナの kit python に nvidia-cudnn-cu13 (9.20) が入っていて torch 2.7+cu128 の cuDNN が
# 初期化できない (CUDNN_STATUS_NOT_INITIALIZED)。ml_archive 同梱の cu12 版 (9.7.1) を先に読み込む。
import ctypes, glob, os
_D = "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib"
for _name in ["libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
              "libcudnn_heuristic.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_adv.so.9", "libcudnn.so.9"]:
    try:
        ctypes.CDLL(os.path.join(_D, _name), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"[cudnn_fix] {_name}: {e}")
