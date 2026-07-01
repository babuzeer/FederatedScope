# HeadOnly 真实分布式场景落地说明

本文记录本轮针对 HeadOnly/GGEUR 真实分布式生命周期场景的代码落地情况。所有验证入口仍走 FederatedScope `distributed` server/client 进程通信、OfficeHome 真实数据集、ViT-B/16 特征、round0 统计上传/协方差下发/样本生成/MLP 训练链路，不引入虚拟数据或模拟特征。

## 已落地能力

1. 分布式加入超时

   `cfg.distribute.join_timeout_seconds` 控制 server 从第一个 join 请求开始等待的最大秒数。客户端数量不足时，server 会记录已加入客户端并失败退出，避免无限等待。

2. 迟到/额外客户端拒绝

   server 在训练启动后拒绝新的 `join_in` / `join_in_info`；达到 `federate.client_num` 后继续收到额外客户端 join 时也会拒绝并写入 warning 日志。

3. 重复 join 防护

   已注册 client 再次发送 join 会被忽略，避免重复计数导致提前启动或邻居表污染。

4. GGEUR 阶段 quorum

   新增以下配置，默认值 `0` 表示严格等待全部配置客户端，不改变正式实验语义：

   - `cfg.ggeur.min_statistics_clients`
   - `cfg.ggeur.min_augmentation_clients`
   - `cfg.ggeur.min_train_updates`

   在故障场景验证中，可以把这些值设置为小于 `client_num`，验证部分客户端退出后系统是否能在指定 quorum 下继续。

5. 活跃客户端集合

   GGEUR server 在统计阶段 quorum 达成后固定 `active_client_ids`，后续协方差下发、augmentation ready、模型下发、finish 通知都只面向活跃客户端集合。

6. 训练客户端集合

   augmentation quorum 达成后固定 `training_client_ids`，后续 MLP 参数下发和模型更新聚合只统计这批训练客户端。

7. 过期、未来、重复模型更新防护

   GGEUR server 会拒绝非活跃客户端更新、非当前 round 更新、同一 round 同一 client 重复更新，并在日志中记录原因。

8. 客户端故障注入

   新增 `cfg.ggeur.fail_after_stage` / `cfg.ggeur.fail_on_round`，用于真实分布式场景验证脚本主动让某个 client 在指定阶段退出。支持阶段：

   - `after_statistics_upload`
   - `after_augmentation_ready`
   - `before_train_round`

## 验证脚本

主脚本：

```bash
scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
```

新增场景批跑脚本：

```bash
scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_scenarios.sh
```

批跑脚本覆盖：

| 场景 | 预期 | 验证点 |
| --- | --- | --- |
| `normal_all_clients` | 成功 | 正常 server/client distributed 链路 |
| `delayed_client_join` | 成功 | 客户端延迟加入但未超过 join timeout |
| `missing_client_join_timeout` | 失败 | 配置客户端数不足时 server 超时退出 |
| `extra_client_late_rejected` | 成功 | 额外客户端迟到加入被拒绝 |
| `client_exit_before_train_quorum` | 成功 | 某客户端训练前退出，`MIN_TRAIN_UPDATES=1` 下系统继续 |

输出目录按实验时间和场景名组织：

```text
exp/headonly_system/loopback_multiip_scenario_runs/<SCENARIO_RUN_ID>/
  scenario_summary.tsv
  normal_all_clients/
  delayed_client_join/
  missing_client_join_timeout/
  extra_client_late_rejected/
  client_exit_before_train_quorum/
```

每个场景内部仍包含完整配置、server/client 日志和解析后的训练指标。

## 运行示例

```bash
cd /root/autodl-tmp/FederatedScope

SCENARIO_RUN_ID=headonly_dist_scenarios_$(date +%Y%m%d_%H%M%S) \
GEN_NUM=1 \
TOTAL_ROUNDS=2 \
CASE_TIMEOUT=900 \
bash scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_scenarios.sh
```

`GEN_NUM=1` 只用于快速验证分布式生命周期，不用于准确率结论。正式准确率实验仍应使用 `GEN_NUM=20`、完整 OfficeHome/ViT 缓存和目标轮数。

## 新增真实 FL 边界入口

1. client-side evaluation

   设置：

   ```yaml
   ggeur:
     headonly_eval_mode: 'client'
   ```

   server 在每轮 FedAvg/FedOpt 聚合后，把聚合后的 MLP 下发给参与训练的客户端；客户端只在自己的本地 `test` split 上评估并上传 `client_eval_metrics`。server 只做按样本数加权汇总，并输出：

   ```text
   Server: Round <r> MLP Test Accuracy - client_weighted_average: ...
   ```

   该模式下 GGEUR distributed server 不构建客户端训练/测试数据。

2. OfficeHome per-client manifest

   预切分脚本：

   ```bash
   python scripts/prepare_officehome_client_manifests.py \
     --source-root /root/autodl-tmp/datasets/OfficeHomeDataset_10072016 \
     --output-root /root/autodl-tmp/datasets/officehome_clients_60_lds \
     --client-num 60 \
     --splits 0.7,0.0,0.3 \
     --use-lds \
     --lds-alpha 0.1 \
     --lds-seed 42 \
     --mode symlink
   ```

   每个 client 目录包含：

   ```text
   client_000001/
     client_manifest.json
     Art/...
   ```

   client 配置中指定：

   ```yaml
   data:
     type: 'office-home'
     root: '/root/autodl-tmp/datasets/officehome_clients_60_lds/client_000001'

   distribute:
     data_idx: 1

   ggeur:
     officehome_manifest_path: '/root/autodl-tmp/datasets/officehome_clients_60_lds/client_000001/client_manifest.json'
   ```

   manifest 会固定 train/val/test 图片清单，loader 不再运行时二次随机切分。

3. 单机模拟 IP 脚本支持这两个入口

   ```bash
   HEADONLY_EVAL_MODE=client \
   OFFICEHOME_MANIFEST_BASE=/root/autodl-tmp/datasets/officehome_clients_60_lds \
   bash scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
   ```

## 仍未完全落地的需求

以下内容来自原“真实分布式联邦学习实施清单”，本轮没有伪造通过，仍需后续单独改造：

1. 真实物理多机网络

   loopback 多 IP 验证可以覆盖多进程、不同监听地址、消息生命周期和日志链路，但不能代表真实公网/内网带宽、丢包、跨机延迟和安全组行为。两台可互通服务器可用后，需要复用同一配置模板跑物理多机版本。

2. 多方法/多模型/多数据集矩阵

   本轮聚焦 HeadOnly + OfficeHome + ViT。FedProx、FedProto、FedOpt、MOON、PromptFL、PACS、DomainNet 等仍需要按同样场景矩阵逐项验证。
