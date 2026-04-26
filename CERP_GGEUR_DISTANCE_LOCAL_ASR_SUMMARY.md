# CerP + GGEUR 距离统计与 Local 无训练攻击 ASR 总结

更新时间：2026-04-20

## 实验日志

本次总结以 2026-04-20 重新运行后的日志为准：

| 名称 | 模型 | 日志 | 距离统计 |
| --- | --- | --- | --- |
| RNN | `ggeur_rnn` | `exp/ggeur_rnn_sentiment/ggeur_mdsent_rating4_rnn_cerp_paper/sub_exp_20260420010745/exp_print.log` | 有 |
| LSTM | `ggeur_lstm` | `exp/ggeur_lstm_sentiment/ggeur_mdsent_rating4_lstm_cerp_paper_lr1e-3/sub_exp_20260420010757/exp_print.log` | 有 |

两组配置的关键攻击参数一致：

| 参数 | 值 |
| --- | --- |
| 恶意客户端 | `[1, 2, 3, 4]` |
| 实际参与样本距离统计的攻击客户端 | `[1, 4]` |
| 实际参与模型距离统计的攻击客户端 | `[1, 2, 3, 4]` |
| 目标标签 | `attack.target_label_ind = 3` |
| 触发文本 | `cerp.trigger_text = 'cf mn bb tq'` |
| BERT 特征模型 | `pretrained_models/nlptown_bert_base_multilingual_uncased_senti` |
| 训练轮数 | `federate.total_round_num = 200` |

## 统计口径

距离统计从 CerP 攻击起始轮 Round 10 开始出现，覆盖 Round 10 到 Round 199，共 190 条记录。

| 日志字段 | 含义 |
| --- | --- |
| `attack-generated sample distance` | clean 原始样本与对应 poison 生成样本的一一匹配距离 |
| `generated-generated sample distance` | 同一批次内 poison 生成样本之间的两两距离 |
| `original-generated sample distance` | 同一批次内 clean 原始样本与 poison 生成样本之间的交叉两两距离 |
| `original-generated model distance` | 同一恶意客户端 benign reference 模型与攻击后模型的一一距离 |
| `generated-generated model distance` | 恶意客户端攻击后模型之间的两两距离 |
| `attacker-client model distance` | 恶意客户端攻击后模型与其他被选中客户端模型的距离，其中 `benign_*` 只统计非恶意客户端 |

## RNN 结果

日志：`exp/ggeur_rnn_sentiment/ggeur_mdsent_rating4_rnn_cerp_paper/sub_exp_20260420010745/exp_print.log`

### 样本距离

| 指标 | 首条 Round 10 L2/Cos | 末条 Round 199 L2/Cos | 平均 L2 | 平均 Cos | 计数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始-生成，一一匹配 | 19.343096 / 0.881450 | 19.326788 / 0.864041 | 19.335705 | 0.855143 | 100 samples/round |
| 生成-生成，两两 | 6.161675 / 0.078332 | 6.178565 / 0.072516 | 6.274623 | 0.074952 | 50 pairs/round |
| 原始-生成，交叉两两 | 19.681857 / 0.910963 | 19.652123 / 0.891136 | 19.691799 | 0.884930 | 200 pairs/round |

样本距离极值：

| 指标 | 最小 L2 | 最大 L2 | 最小 Cos | 最大 Cos |
| --- | --- | --- | --- | --- |
| 原始-生成，一一匹配 | 19.126947, Round 136 | 19.754816, Round 12 | 0.835566, Round 123 | 0.891617, Round 12 |
| 生成-生成，两两 | 5.957253, Round 136 | 6.764153, Round 12 | 0.067445, Round 189 | 0.089108, Round 12 |
| 原始-生成，交叉两两 | 19.440877, Round 136 | 20.203018, Round 12 | 0.867399, Round 140 | 0.930575, Round 12 |

### 模型距离

| 指标 | 首条 Round 10 L2/Cos | 末条 Round 199 L2/Cos | 平均 L2 | 平均 Cos | 计数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始模型-生成模型 | 1.732370 / 0.003669 | 2.146936 / 0.001347 | 1.970206 | 0.001950 | 4 models/round |
| 生成模型-生成模型 | 2.365635 / 0.006834 | 2.556481 / 0.001911 | 2.504697 | 0.003226 | 6 pairs/round |
| 恶意模型-其他客户端模型 | 2.397527 / 0.007034 | 2.631013 / 0.002028 | 2.564942 | 0.003381 | 76 pairs/round |
| 恶意模型-benign 客户端模型 | 2.403507 / 0.007071 | 2.644988 / 0.002050 | 2.576238 | 0.003410 | 64 pairs/round |

### Poison ASR 与准确率

| Round | books | dvd | electronics | kitchen | average |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0764 | 0.0891 | 0.1199 | 0.1201 | 0.1014 |
| 10 | 0.1389 | 0.1783 | 0.1760 | 0.1579 | 0.1628 |
| 20 | 0.9931 | 0.9971 | 0.9888 | 0.9960 | 0.9937 |
| 50 | 0.9931 | 0.9964 | 0.9913 | 0.9960 | 0.9942 |
| 100 | 0.9931 | 0.9964 | 0.9925 | 0.9946 | 0.9941 |
| 150 | 0.9931 | 0.9964 | 0.9863 | 0.9919 | 0.9919 |
| 199 | 0.9931 | 0.9949 | 0.9850 | 0.9919 | 0.9912 |
| 峰值 | 1.0000 | 0.9964 | 0.9900 | 0.9946 | 0.9952, Round 117 |

| 指标 | 数值 |
| --- | ---: |
| Classifier average final accuracy | 0.6345 |
| Classifier best average accuracy | 0.6718 |

## LSTM 结果

日志：`exp/ggeur_lstm_sentiment/ggeur_mdsent_rating4_lstm_cerp_paper_lr1e-3/sub_exp_20260420010757/exp_print.log`

### 样本距离

| 指标 | 首条 Round 10 L2/Cos | 末条 Round 199 L2/Cos | 平均 L2 | 平均 Cos | 计数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始-生成，一一匹配 | 18.674628 / 0.867826 | 18.032650 / 0.799525 | 17.975660 | 0.809698 | 100 samples/round |
| 生成-生成，两两 | 7.639578 / 0.131099 | 6.167183 / 0.084728 | 6.131372 | 0.083830 | 50 pairs/round |
| 原始-生成，交叉两两 | 19.243144 / 0.919749 | 18.456791 / 0.836125 | 18.393823 | 0.845959 | 200 pairs/round |

样本距离极值：

| 指标 | 最小 L2 | 最大 L2 | 最小 Cos | 最大 Cos |
| --- | --- | --- | --- | --- |
| 原始-生成，一一匹配 | 17.707484, Round 35 | 18.674628, Round 10 | 0.793357, Round 28 | 0.869662, Round 12 |
| 生成-生成，两两 | 5.621115, Round 150 | 7.639578, Round 10 | 0.070747, Round 150 | 0.131099, Round 10 |
| 原始-生成，交叉两两 | 18.089913, Round 35 | 19.243144, Round 10 | 0.831198, Round 28 | 0.919749, Round 10 |

### 模型距离

| 指标 | 首条 Round 10 L2/Cos | 末条 Round 199 L2/Cos | 平均 L2 | 平均 Cos | 计数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始模型-生成模型 | 3.240581 / 0.003067 | 3.848894 / 0.001359 | 3.655006 | 0.001922 | 4 models/round |
| 生成模型-生成模型 | 4.443978 / 0.005756 | 4.727798 / 0.002050 | 4.668248 | 0.003179 | 6 pairs/round |
| 恶意模型-其他客户端模型 | 4.507755 / 0.005935 | 4.866114 / 0.002176 | 4.784953 | 0.003330 | 76 pairs/round |
| 恶意模型-benign 客户端模型 | 4.519713 / 0.005969 | 4.892048 / 0.002199 | 4.806835 | 0.003358 | 64 pairs/round |

### Poison ASR 与准确率

| Round | books | dvd | electronics | kitchen | average |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0625 | 0.0739 | 0.1024 | 0.0972 | 0.0840 |
| 10 | 0.1667 | 0.1993 | 0.1998 | 0.1822 | 0.1870 |
| 20 | 0.9792 | 0.9601 | 0.9600 | 0.9568 | 0.9640 |
| 50 | 0.9861 | 0.9667 | 0.9675 | 0.9649 | 0.9713 |
| 100 | 0.9861 | 0.9674 | 0.9675 | 0.9649 | 0.9715 |
| 150 | 0.9861 | 0.9638 | 0.9638 | 0.9636 | 0.9693 |
| 199 | 0.9861 | 0.9638 | 0.9625 | 0.9636 | 0.9690 |
| 峰值 | 0.9861 | 0.9674 | 0.9688 | 0.9676 | 0.9725, Round 46 |

| 指标 | 数值 |
| --- | ---: |
| Classifier average final accuracy | 0.6420 |
| Classifier best average accuracy | 0.6669 |

## RNN 与 LSTM 对比

| 指标 | RNN | LSTM |
| --- | ---: | ---: |
| 原始-生成样本平均 L2，一一匹配 | 19.335705 | 17.975660 |
| 原始-生成样本平均 Cos，一一匹配 | 0.855143 | 0.809698 |
| 生成-生成样本平均 L2 | 6.274623 | 6.131372 |
| 生成-生成样本平均 Cos | 0.074952 | 0.083830 |
| 原始-生成样本平均 L2，交叉两两 | 19.691799 | 18.393823 |
| 原始-生成样本平均 Cos，交叉两两 | 0.884930 | 0.845959 |
| 原始模型-生成模型平均 L2 | 1.970206 | 3.655006 |
| 原始模型-生成模型平均 Cos | 0.001950 | 0.001922 |
| 生成模型-生成模型平均 L2 | 2.504697 | 4.668248 |
| 生成模型-生成模型平均 Cos | 0.003226 | 0.003179 |
| 恶意模型-benign 客户端模型平均 L2 | 2.576238 | 4.806835 |
| 恶意模型-benign 客户端模型平均 Cos | 0.003410 | 0.003358 |
| Poison ASR 峰值平均 | 0.9952 | 0.9725 |
| Poison ASR 最后一轮平均 | 0.9912 | 0.9690 |
| Final average accuracy | 0.6345 | 0.6420 |
| Best average accuracy | 0.6718 | 0.6669 |

结论：

- 样本层面，原始-生成距离明显大于生成-生成距离。RNN 中交叉原始-生成 L2 为 19.691799，而生成-生成 L2 为 6.274623；LSTM 中分别为 18.393823 和 6.131372。
- RNN 的原始-生成样本距离更大，说明 RNN 配置下 CerP 触发后的 BERT 表征偏移更强。
- 模型层面，LSTM 的 L2 距离明显大于 RNN。LSTM 原始模型-生成模型平均 L2 为 3.655006，RNN 为 1.970206。
- 模型 cosine distance 都很小，约 0.002 到 0.003，说明参数向量方向接近，L2 差异主要体现为参数空间幅度差异。
- RNN 的攻击效果更强：Poison ASR 峰值平均 0.9952，最后一轮平均 0.9912；LSTM 峰值平均 0.9725，最后一轮平均 0.9690。
- Local 无训练攻击 ASR 远低于完整训练后的 ASR，说明攻击效果主要来自 CerP + GGEUR 训练过程，而不是触发词本身的零训练迁移。

## Local 无训练攻击脚本

新增脚本：

```text
scripts/ggeur_local_cerp_asr.py
```

脚本用途：

- 不启动 FederatedScope 训练。
- 不训练 RNN/LSTM 分类器。
- 不训练 CerP soft trigger。
- 复用 YAML 中的数据划分、恶意客户端、目标标签和触发文本。
- 加载本地 `pretrained_models/nlptown_bert_base_multilingual_uncased_senti` 的 BERT sequence-classification head。
- 将 `cf mn bb tq` 作为离散 token trigger 插入文本后推理。
- 统计恶意客户端上的 clean accuracy、clean target rate 和 poison ASR。

语法检查：

```powershell
conda run -n fs python -m py_compile scripts\ggeur_local_cerp_asr.py
```

### RNN 配置，train split

运行命令：

```powershell
conda run -n fs python scripts\ggeur_local_cerp_asr.py --cfg scripts\example_configs\ggeur_baseline_rnn\ggeur_rnn_mdsent_rating4_cerp_paper.yaml --device cpu
```

结果：

| Client | Domain | clean_total | clean_acc | clean_target_rate | poison_total | ASR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | books | 5 | 0.8000 | 0.2000 | 4 | 0.0000 (0/4) |
| 2 | books | 5 | 1.0000 | 1.0000 | 0 | 0.0000 (0/0) |
| 3 | books | 5 | 0.8000 | 0.8000 | 0 | 0.0000 (0/0) |
| 4 | books | 5 | 0.8000 | 0.0000 | 5 | 0.0000 (0/5) |
| 加权平均 | - | 20 | 0.8500 | 0.5000 | 9 | 0.0000 (0/9) |

### RNN 配置，test split

运行命令：

```powershell
conda run -n fs python scripts\ggeur_local_cerp_asr.py --cfg scripts\example_configs\ggeur_baseline_rnn\ggeur_rnn_mdsent_rating4_cerp_paper.yaml --device cpu --split test
```

结果：

| Client | Domain | clean_total | clean_acc | clean_target_rate | poison_total | ASR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 2 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 3 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 4 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 加权平均 | - | 768 | 0.5417 | 0.1875 | 576 | 0.0486 (28/576) |

### LSTM 配置，test split

运行命令：

```powershell
conda run -n fs python scripts\ggeur_local_cerp_asr.py --cfg scripts\example_configs\ggeur_baseline_lstm\ggeur_lstm_mdsent_rating4_cerp_paper_lr1e-3.yaml --device cpu --split test
```

结果：

| Client | Domain | clean_total | clean_acc | clean_target_rate | poison_total | ASR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 2 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 3 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 4 | books | 192 | 0.5417 | 0.1875 | 144 | 0.0486 (7/144) |
| 加权平均 | - | 768 | 0.5417 | 0.1875 | 576 | 0.0486 (28/576) |

## Local 与完整训练对比

| 设置 | 模型来源 | 是否训练 | Split / Eval | ASR |
| --- | --- | --- | --- | ---: |
| Local probe, RNN cfg | pretrained BERT head | 否 | train | 0.0000 |
| Local probe, RNN cfg | pretrained BERT head | 否 | test | 0.0486 |
| Local probe, LSTM cfg | pretrained BERT head | 否 | test | 0.0486 |
| FS CerP + GGEUR, RNN | 联邦训练 RNN | 是 | poison eval, peak | 0.9952 |
| FS CerP + GGEUR, RNN | 联邦训练 RNN | 是 | poison eval, Round 199 | 0.9912 |
| FS CerP + GGEUR, LSTM | 联邦训练 LSTM | 是 | poison eval, peak | 0.9725 |
| FS CerP + GGEUR, LSTM | 联邦训练 LSTM | 是 | poison eval, Round 199 | 0.9690 |
