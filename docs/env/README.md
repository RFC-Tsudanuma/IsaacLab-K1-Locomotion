# 学習用 Python 環境 (kit python) の変更記録

## 2026-10-18: rsl-rl-lib 3.x に入れ替え

コンテナを作り直した直後の kit python (`/isaac-sim/python.sh`、Isaac Sim 5.1 / IsaacLab 2.3.2) には
rsl-rl-lib 5.0.1 が入っていたが、このリポジトリのコード (HistoryActorCritic など) は rsl-rl 3.x の API
(`rsl_rl.modules.ActorCritic` / `rsl_rl.networks.MLP`) を使うので動かない。また tensordict 0.15.1 は
torch 2.7 で import エラーになる。そこで次を実行した:

```bash
/isaac-sim/python.sh -m pip freeze > docs/env/pip_freeze_before_2026-10-18.txt
/isaac-sim/python.sh -m pip install --no-deps -r requirements-k1.txt          # rsl-rl-lib 3.0.1, tensordict 0.9.1
/isaac-sim/python.sh -m pip install --no-deps -e source/isaaclab_k1_locomotion
/isaac-sim/python.sh -m pip freeze > docs/env/pip_freeze_after_2026-10-18.txt
```

変更前後の差分 (`diff pip_freeze_before_2026-10-18.txt pip_freeze_after_2026-10-18.txt`) は次の 3 つだけ:

| パッケージ | 変更前 | 変更後 |
|---|---|---|
| rsl-rl-lib | 5.0.1 | 3.0.1 |
| tensordict | 0.15.1 | 0.9.1 |
| isaaclab_k1_locomotion | (未インストール) | editable (このリポジトリ) |

## 元に戻す手順

```bash
/isaac-sim/python.sh -m pip uninstall -y isaaclab_k1_locomotion
/isaac-sim/python.sh -m pip install --no-deps rsl-rl-lib==5.0.1 tensordict==0.15.1
# 確認: 差分が出なければ元どおり
/isaac-sim/python.sh -m pip freeze | diff docs/env/pip_freeze_before_2026-10-18.txt -
```

`--no-deps` を付けるのは、kit 同梱の torch などを pip に入れ替えさせないため。
