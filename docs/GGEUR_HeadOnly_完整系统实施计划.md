# GGEUR HeadOnly 完整系统实施计划

## 1. 目标重新定义

HeadOnly 方案的最终目标不是单独证明某个脚本可以跑出较高训练吞吐，而是在现有 GGEUR / FederatedScope 基础上实现一套完整、可复现、可扩展的联邦学习系统。

最终系统必须同时满足：

1. 使用真实数据集，不使用任何模拟数据、虚拟特征或随机伪样本替代真实数据流程。
2. 保留 GGEUR 第 0 轮完整逻辑：client 提特征、上传统计量，server 聚合 covariance / prototype，client 接收全局统计并生成增强样本。
3. 第 1 轮以后进入 HeadOnly：client 只训练 MLP head，只上传/下载 head 参数，不再长期持有 ViT / CNN / Mixer backbone。
4. 同时输出准确率、loss、训练 QPS、通信 QPS、生成耗时、特征提取耗时、payload 大小、分布式阶段耗时和异常状态。
5. 支持单机多进程真实 gRPC 验证，并最终推进到多机真实网络部署。
6. server 不持有 client 训练数据；正式多机阶段 evaluation 应改为 client-side evaluation，server 只做指标汇总。

当前已经跑过的独立 HeadOnly QPS 脚本只能视为压测原型，不能作为最终完整系统验收依据。它证明了“缓存特征 + MLP head”具备高吞吐潜力，但没有完整覆盖 GGEUR round 0、测试准确率和真实分布式边界。

## 2. 参考基线

第一阶段必须对齐已有 baseline：

- 配置：`scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml`
- 数据集：OfficeHome
- 模型/特征提取器：CLIP ViT-B-16
- 联邦方法：GGEUR + FedAvg
- client 数：60
- 训练轮数：100
- 数据划分：`data.splits: [0.7, 0.0, 0.3]`
- LDS：`use_lds: True`
- LDS alpha：`0.1`
- seed：`42`

HeadOnly 的实验必须使用同样的数据划分、同样的 OfficeHome loader、同样的类别数和同样的 CLIP 权重路径。否则准确率不能和 baseline 对比。

## 3. 系统流程

### 3.1 第 0 轮：完整 GGEUR 统计与增强

client 执行：

1. 只加载本 client 的真实本地训练数据。
2. 加载 frozen backbone，例如 CLIP ViT-B-16。
3. 对真实图像提取 feature。
4. 按类别计算：
   - `count_c`
   - `mean_c`
   - `covariance_c`
   - `prototype_c`
5. 向 server 上传 local statistics。

server 执行：

1. 接收所有参与 client 的 local statistics。
2. 校验 embedding dim、类别数、payload 格式和 client 版本。
3. 聚合 global covariance。
4. 聚合 global prototype。
5. 为每个 client 准备 cross-client prototypes。
6. 向 client 下发 global covariance / prototypes。

client 再执行：

1. 接收 global covariance / prototypes。
2. 在 feature 空间执行 GGEUR augmentation。
3. 生成增强后的 `(feature, label)`。
4. 写入本地 feature cache。
5. 释放 backbone，只保留 MLP head 和增强特征缓存。

第 0 轮必须记录：

- 每个 client 原始样本数
- 每个 client 提取特征耗时和 img/s
- 每个 client 上传 statistics payload 大小
- server 聚合 covariance/prototype 耗时
- server 广播 payload 大小
- 每个 client 生成增强样本数
- 每个 client augmentation 耗时和 samples/s
- feature cache 路径、版本、checksum

### 3.2 第 1 轮以后：HeadOnly 联邦训练

client 执行：

1. 从 parameter service 或 server 获取 `global_mlp@vN`。
2. 校验：
   - `feature_cache_version`
   - `embedding_dim`
   - `num_classes`
   - `head architecture`
   - `checksum`
3. 从本地 cache 读取增强后的真实数据特征。
4. 只训练 MLP head。
5. 上传 MLP update、样本数、训练 loss/acc 和耗时。

sub-server 执行：

1. 接收 client MLP update。
2. 校验 round、base version、cache version。
3. 按 `sample_size` 做局部 FedAvg。
4. 记录 late / dropped / failed client。
5. 上传局部聚合结果给 root server。

root server 执行：

1. 接收各 sub-server 的局部聚合结果。
2. 按总样本数做全局聚合。
3. 发布新的 `global_mlp@vN+1`。
4. 汇总训练和系统指标。

## 4. 准确率评估方案

不能再只报告训练集上的 `avg_train_acc`。必须输出测试准确率。

单机对齐 baseline 阶段：

- 暂时允许使用和 baseline 相同的 OfficeHome test split，用于快速对齐准确率。
- 必须明确标记为 `centralized_test_feature_eval` 或 `baseline-compatible eval`。
- 输出每轮或指定频率的 `test_acc`、`test_loss`、`best_test_acc`、`final_test_acc`。

真实多机 FL 阶段：

- server 不直接读取 client 测试数据。
- 每个 client 在本地 test split 上评估当前 global MLP。
- client 上传：
  - `test_correct`
  - `test_total`
  - `test_loss_sum`
  - domain / client id
- server 只做加权汇总：
  - global test acc
  - per-domain test acc
  - per-client test acc 分布

验收时必须同时报告：

- final train acc
- final test acc
- best test acc
- per-domain test acc
- 100 轮准确率曲线

## 5. QPS 指标定义

HeadOnly 系统中 QPS 必须拆开报告，不能只报一个数字。

必须输出：

1. `train_sample_qps`
   - 定义：每轮训练样本数 / 该轮训练总耗时
   - 这是 10000+ QPS 的主指标。

2. `stable_train_sample_qps`
   - 从第 2 或第 3 轮以后统计，排除第 0 轮特征提取和首轮预热。

3. `end_to_end_qps`
   - 总训练样本处理量 / 包含第 0 轮生成在内的总耗时。
   - 只能作为补充指标，不能替代稳定训练 QPS。

4. `feature_extract_qps`
   - 第 0 轮真实图像特征提取 img/s。

5. `augmentation_qps`
   - 第 0 轮真实统计驱动的增强特征生成 samples/s。

6. `download_qps`
   - client 拉取 MLP 参数请求数 / 下载阶段耗时。

7. `upload_qps`
   - client 上传 MLP update 请求数 / 上传阶段耗时。

8. `aggregation_qps`
   - sub-server/root server 聚合 update 数 / 聚合耗时。

同时必须记录：

- 每轮有效 client 数
- late client 数
- dropped client 数
- failed client 数
- 平均/最大 payload bytes
- sub-server 聚合耗时
- root 聚合耗时
- 参数发布耗时
- GPU 显存峰值
- CPU worker 数

## 6. 数据与缓存要求

禁止项：

- 不允许 synthetic dataset。
- 不允许随机生成原始 feature 代替真实图像提取。
- 不允许旧版无 metadata 的 cache 参与正式实验。
- 不允许 server 在正式多机阶段直接读取 client 训练数据。

cache 必须包含：

- `source: real_dataset`
- dataset 名称
- data root 或 client data shard id
- split 配置
- seed
- client id
- domain
- raw sample count
- augmented sample count
- feature extractor 类型和权重路径
- embedding dim
- num classes
- GGEUR 生成参数
- cache version
- checksum

cache 复用规则：

- config 完全一致才允许复用。
- `feature_cache_version` 不一致必须重建。
- split、seed、client 数、生成量、backbone 权重任一变化必须重建。
- 日志中必须明确写出是 `cache_miss -> regenerate` 还是 `cache_hit -> reuse`。

## 7. 实施阶段

### 阶段 A：纠正当前 HeadOnly 原型口径

目标：

- 停止把独立 QPS 脚本作为最终系统结果。
- 将其定位为性能压测原型。
- 后续实现回到 GGEUR server/client message flow。

工作：

1. 保留本地代码，不删除已有实验脚本。
2. 新增正式 HeadOnly 配置开关。
3. 标记原型脚本输出为 `prototype_qps_only`。
4. 后续报告中不再用训练集 `avg_train_acc` 代替测试准确率。

### 阶段 B：在现有 GGEUR client/server 内实现 HeadOnly

目标：

- 复用 `GGEURClient` 和 `GGEURServer` 的 round 0 statistics / augmentation 路径。
- 在 augmentation 完成后切换到 MLP-only 参数通信。

工作：

1. 在 config 中增加：
   - `ggeur.head_only_mode`
   - `ggeur.head_only_after_round0`
   - `ggeur.headonly_cache_dir`
   - `ggeur.headonly_cache_version`
   - `ggeur.headonly_eval_mode`
2. 修改 client：
   - round 0 保持完整 GGEUR。
   - augmentation 后保存 feature cache。
   - 释放 backbone。
   - 后续训练只使用 MLP head。
3. 修改 server：
   - 保持 statistics 聚合和 broadcast。
   - 后续只聚合 MLP state dict。
   - 增加 QPS / payload / stage timing 统计。
4. 确认 gRPC 大消息可传输 covariance/prototype。

### 阶段 C：补齐测试准确率

目标：

- 每轮训练后或按 `eval.freq` 输出测试准确率。

工作：

1. 单机 baseline-compatible eval：
   - 使用同一 OfficeHome split。
   - 先在 server 或独立 evaluator 提取 test feature。
   - 输出可与 baseline 对齐的 test acc。
2. 多机真实 FL eval：
   - 改为 client-side evaluation。
   - server 只汇总 metrics。
3. 输出 per-domain、per-client、global weighted test acc。

### 阶段 D：单机多进程完整验证

目标：

- 在服务器单机上跑完整系统，不再跑简化原型。

配置：

- OfficeHome
- CLIP ViT-B-16
- 60 clients
- 100 rounds
- `data.splits: [0.7, 0.0, 0.3]`
- `use_lds: True`
- `lds_alpha: 0.1`
- seed 42
- `num_generated_per_sample: 20`
- `num_generated_per_prototype: 20`
- `target_size_per_class: 20`

必须产出：

- `round0_statistics.log`
- `augmentation.log`
- `training.log`
- `eval.log`
- `metrics.json`
- `resolved_config.yaml`
- `cache_manifest.json`
- `system_status.log`

通过标准：

- 60 clients 正常完成 round 0 statistics。
- server 聚合并广播 covariance/prototype。
- 60 clients 完成 augmentation。
- 100 轮训练完成。
- 每轮有 train loss/acc、test loss/acc、QPS。
- 最终报告包含 best/final test acc。

### 阶段 E：QPS 压测与瓶颈分析

目标：

- 在完整系统前提下验证 10000+ QPS，而不是绕开系统流程。

压测矩阵：

1. `60 clients / 100 rounds / gen20`
2. `60 clients / 100 rounds / gen50`
3. `120 clients / short rounds`
4. `300+ clients / synthetic-free real shard cache`

注意：第 4 项不能使用伪数据，必须来自真实数据的预切分或真实 cache 扩展策略。

报告：

- 稳定训练 QPS
- 端到端 QPS
- round 0 特征提取耗时
- round 0 增强耗时
- 通信 QPS
- payload 大小
- CPU/GPU 使用率
- I/O 等待
- 瓶颈归因

### 阶段 F：多机真实部署

目标：

- 从单机多进程升级到多服务器真实网络部署。

工作：

1. 固定机器清单：
   - server 机器
   - client 机器
   - GPU 分配
   - SSH 别名
   - IP / port
2. 预切分数据：
   - 每个 client 只拥有自己的本地数据目录。
   - server 不放 client 训练数据。
3. 启动脚本：
   - start server
   - start clients
   - status
   - stop
   - collect logs
4. 验证：
   - join
   - round 0 statistics
   - augmentation
   - training
   - client-side eval
   - finish
   - cleanup

通过标准：

- 跨主机 gRPC 可稳定传输 covariance/prototype/model update。
- server/client 正常退出，无残留进程。
- 日志按实验时间目录归档。
- 结果可复现。

### 阶段 G：实验矩阵化

在 HeadOnly + OfficeHome + ViT 完整跑通后，再扩展：

1. 模型：
   - ViT / CLIP
   - CNN / ConvNeXt
   - timm / Mixer
2. 数据集：
   - OfficeHome
   - PACS
   - DomainNet
3. 联邦方法：
   - FedAvg
   - FedProx
   - FedOpt
   - FedProto
   - MOON
   - PromptFL

每次只扩一个维度，避免系统问题和方法问题混在一起。

## 8. 实验目录规范

所有实验输出必须按时间和实验名归档：

```text
exp/headonly_system/
  officehome_vit_fedavg_60c_100r_gen20_YYYYMMDD_HHMMSS/
    configs/
    logs/
    metrics/
    caches/
    system/
    artifacts/
```

每个实验目录必须包含：

- 原始配置
- resolved 配置
- server log
- client logs
- round0 log
- train/eval metrics
- QPS summary
- cache manifest
- git commit / diff 摘要
- 环境信息
- GPU 信息

## 9. 验收标准

HeadOnly 方案只有同时满足以下条件，才算完整验证：

1. 使用真实 OfficeHome 数据，没有任何模拟数据或伪特征。
2. 数据划分与 baseline 保持一致。
3. 第 0 轮完整执行 GGEUR statistics / covariance / prototype / augmentation。
4. 第 1 轮以后只训练和通信 MLP head。
5. 输出完整测试准确率，不再只报告训练准确率。
6. 输出稳定训练 QPS 和端到端 QPS。
7. 输出通信、payload、聚合、生成、特征提取等系统指标。
8. 单机多进程完整 60 clients / 100 rounds 跑通。
9. 多机真实网络部署至少完成 1 个完整样板。
10. 所有日志、配置、缓存、指标都可追溯、可复现。

## 10. 下一步执行顺序

1. 先停止继续扩展独立压测脚本作为主线。
2. 回到 `GGEURClient` / `GGEURServer`，实现正式 HeadOnly mode。
3. 补齐 test accuracy 评估。
4. 使用 OfficeHome baseline 相同划分跑 60 clients / 100 rounds / gen20。
5. 生成完整实验报告。
6. 再做 gen50 对齐 baseline。
7. 通过后推进多机真实部署。

