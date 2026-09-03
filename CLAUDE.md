# booster-k1ロボットの歩行の強化学習
使用しているフレームワーク
- rsl-rl
- IsaacLab

ロボット
- Booster-K1ロボット 22自由度、腰の関節は無し

## 各種実行方法

- python
  - pythonは、IsaacLab2.3.2のkitを使っている。これは~/.bash_functionsに定義されている、_labpython2からアクセスが可能。
  - pipなどを参照する際も全てここから参照してください。
- rsl_rl
  - 上記の_labpython2のpipにIsaaclab版のrsl_rlがインストールされています。
  - 基本的な学習は、train.shで実行されるK1FlatEnvCfgタスクによって行われている。