# booster-k1ロボットの歩行の強化学習
使用しているフレームワーク
- rsl-rl
- IsaacLab

ロボット
- Booster-K1ロボット 22自由度、腰の関節は無し

## 各種実行方法

- python
  - pythonは、IsaacLab2.3.2のkitを使っている。
- rsl_rl
  - 基本的な学習は、train.shで実行されるK1FlatEnvCfgタスクによって行われている。