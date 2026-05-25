# 10000+ QPS 指标解释、现状与可行性分析

## 1. 背景

任务书要求系统实现 `10000+ QPS`，但未明确 QPS 的业务口径。由于本项目同时包含联邦学习训练、特征生成、数据增强、分布式通信以及后续前后端系统建设，QPS 可以有多种解释方式。

本文基于当前代码和服务器实验日志，对几种可能的 `10000+ QPS` 口径进行区分，并说明当前是否已经实现、是否具备实现可行性。

服务器项目路径：

```text
/root/autodl-tmp/FederatedScope
```

关键实验日志：

```text
exp/domainnet_4domains/cnn/fedavg/domainnet_4domains_cnn_fedavg/exp_print.log
```

## 2. 推荐主口径：训练阶段样本处理吞吐

### 2.1 定义

推荐将 `10000+ QPS` 定义为：

```text
训练阶段样本级吞吐能力，即系统在训练轮次中每秒完成的训练样本处理数量。
```

计算方式：

```text
Training QPS = 每轮参与训练的总样本数 / 该轮训练耗时
```

该口径更适合当前项目，因为 FederatedScope/GGEUR 的核心任务是训练和联邦学习实验，而不是传统 Web API 服务。

### 2.2 当前实现情况

服务器日志显示，`DomainNet 4 domains + CNN + FedAvg` 单机 standalone 实验已经达到 `10000+ QPS`。

关键日志：

```text
2026-05-08 16:40:56,089 Server: Round 2 aggregation complete, total samples: 85881, round time: 5.0s
2026-05-08 16:41:01,176 Server: Round 3 aggregation complete, total samples: 85881, round time: 5.1s
2026-05-08 16:49:37,841 Server: Round 99 aggregation complete, total samples: 85881, round time: 5.7s
```

对应 QPS：

```text
Round 2: 85881 / 5.0 = 17176 samples/s
Round 3: 85881 / 5.1 = 16839 samples/s
Round 99: 85881 / 5.7 = 15067 samples/s
```

结论：

```text
在稳定训练轮次中，当前系统已经实现 10000+ QPS。
```

### 2.3 表述建议

建议在汇报或任务书验收中使用如下表述：

```text
本项目中的 10000+ QPS 指训练阶段样本级吞吐能力，即系统在单机 standalone 训练模式下，每秒完成的训练样本处理数量。根据 DomainNet 4 domains + CNN + FedAvg 实验日志，稳定训练轮次每轮处理 85881 个样本，单轮耗时约 5.0-5.7 秒，对应吞吐约 1.5w-1.8w samples/s，满足 10000+ QPS 指标。
```

## 3. 其他 QPS 口径及当前情况

### 3.1 完整训练流程吞吐

定义：

```text
完整训练流程 QPS = 所有训练轮次处理样本总数 / 从训练开始到训练结束的总耗时
```

该口径会把预热、特征提取、统计收集、缓存读写、评估等开销算入总耗时。

现状：

- 比稳定轮次训练 QPS 更低。
- 对首轮预热、数据缓存、I/O 状态非常敏感。
- 不建议作为主验收口径。

可行性：

```text
具备实现 10000+ QPS 的可能，但需要排除或优化 Round 0/1 的预处理开销。
```

建议：

```text
验收时从 Round 2 或 Round 3 开始统计稳定训练吞吐，Round 0/1 作为预热阶段单独说明。
```

### 3.2 数据生成 / 数据增强吞吐

定义：

```text
数据生成 QPS = 每秒生成的增强样本数量
```

该口径对应 GGEUR 中基于统计量生成增强特征样本的阶段。

现状：

- 该阶段仍是主要瓶颈之一。
- 旧 OfficeHome ViT/CLIP 分析中，增强阶段约 `50 samples/s`。
- CNN 特征版本明显更快，但仍不建议直接作为 `10000+ QPS` 主证明。

可行性：

```text
当前未稳定证明 10000+ QPS，但通过并行化和 GPU batch 采样具备提升空间。
```

提升方案：

- 将多个 client 的增强从串行改为并行。
- 将 Gaussian 采样从 CPU/numpy 改为 GPU/torch batch。
- 缓存增强结果，后续实验直接复用。
- 多 GPU 或多进程分摊 client。

### 3.3 特征提取吞吐

定义：

```text
特征提取 QPS = 每秒完成 CNN/CLIP/timm 特征编码的图片数量
```

代码中已有对应日志：

```text
Feature extraction QPS=xxxx img/s
```

服务器 `DomainNet + CNN + FedAvg` 日志显示，单 client 特征提取 QPS 多数在 `2000-5300 img/s` 范围内。

示例：

```text
Client 50: Feature extraction QPS=1963.3 img/s
Client 25: Feature extraction QPS=4707.1 img/s
Client 27: Feature extraction QPS=5356.4 img/s
Client 28: Feature extraction QPS=5224.6 img/s
```

现状：

```text
单 client 特征提取尚未达到 10000+ QPS，但多 client 汇总或多 GPU 并行具备达到 10000+ 的可行性。
```

注意：

- 该 QPS 只统计模型前向特征提取时间。
- 不包括图片读取、缓存写入、统计计算和训练。
- 不应与完整训练 QPS 混用。

### 3.4 联邦通信请求吞吐

定义：

```text
通信 QPS = server 每秒处理的 client 上传/下载/聚合请求数量
```

现状：

- 该项目每轮通信次数有限，通信消息体较大。
- 通信 QPS 通常远低于 `10000 requests/s`。
- 该口径不适合作为本项目 `10000+ QPS` 的主解释。

可行性：

```text
不建议承诺通信请求 QPS 达到 10000+。
```

原因：

- 联邦学习通信是低频大包，不是高频小请求。
- 强行按 Web 请求 QPS 解释会偏离系统实际。

### 3.5 前后端在线服务 QPS

定义：

```text
前后端系统上线后，后端 API 每秒处理的查询请求数。
```

适用接口：

- 模型推理接口
- 任务状态查询接口
- 实验结果查询接口
- 特征检索接口

现状：

```text
当前项目尚未完成前后端系统部署，因此没有实际在线服务 QPS 压测数据。
```

可行性：

```text
具备实现可能，但需要单独设计服务架构和压测方案。
```

实现方案：

- FastAPI/Flask 提供后端 API。
- Uvicorn/Gunicorn 多 worker 部署。
- Redis 缓存任务状态和实验结果。
- 模型服务预加载权重。
- 推理接口支持 batch。
- Nginx 做反向代理和负载均衡。
- 多实例横向扩展，总 QPS 按实例累加。

该口径适合系统建设阶段，但不适合证明当前训练实验已经达标。

## 4. 当前达标情况汇总

| QPS 口径 | 当前证据 | 当前数值 | 是否达标 | 建议用途 |
| --- | --- | ---: | --- | --- |
| 稳定训练轮次样本吞吐 | DomainNet CNN FedAvg 日志 | 约 `1.5w-1.8w samples/s` | 是 | 推荐主口径 |
| 完整训练流程吞吐 | 受预热和统计阶段影响 | 低于稳定训练吞吐 | 有条件 | 可作为补充 |
| 数据生成/增强吞吐 | 旧 OfficeHome 分析约 `50 samples/s` | 未达 1w | 否 | 优化目标 |
| 单 client 特征提取吞吐 | DomainNet CNN 日志 | 约 `2k-5.3k img/s` | 否 | 补充指标 |
| 联邦通信请求吞吐 | 通信低频大包 | 远低于 1w req/s | 否 | 不推荐 |
| 前后端 API QPS | 尚未部署压测 | 暂无 | 未验证 | 后续系统指标 |

## 5. 实现 10000+ QPS 的可行性判断

### 5.1 已经实现的部分

```text
单机 standalone 稳定训练阶段样本吞吐已经实现 10000+ QPS。
```

依据：

- 每轮样本数：`85881`
- 稳定轮次耗时：约 `5.0-5.7s`
- 稳定训练 QPS：约 `15067-17176 samples/s`

### 5.2 尚未实现或不建议承诺的部分

以下口径目前不建议作为已达标结论：

- 完整端到端训练流程 QPS。
- 数据生成/增强 QPS。
- 单 client 特征提取 QPS。
- 联邦通信请求 QPS。
- 前后端 API 在线服务 QPS。

### 5.3 最稳妥结论

```text
当前项目已经在单机 standalone 的稳定训练轮次中实现 10000+ samples/s 的训练阶段吞吐；但该指标不应泛化为所有阶段、所有模型、所有部署形态均达到 10000+ QPS。数据生成、特征提取、分布式通信和前后端在线服务仍需分别优化和验证。
```

## 6. 后续实现方案

### 6.1 固化训练 QPS 统计

在 server 每轮聚合完成后，输出统一 QPS 日志：

```text
train_qps = total_samples / round_time
```

建议输出：

- 当前轮 QPS
- 平均 QPS
- 最小 QPS
- P50 QPS
- P95 QPS
- 是否超过 10000

### 6.2 明确验收规则

推荐验收规则：

```text
从 Round 2 或 Round 3 开始统计稳定训练轮次，连续 N 轮平均训练 QPS >= 10000，且最低轮次 QPS 不低于 10000，则认为训练阶段 10000+ QPS 指标达成。
```

不建议把 Round 0 统计阶段和首轮预热纳入主指标。

### 6.3 优化数据生成阶段

数据生成阶段是后续主要优化对象：

- 多 client 并行生成。
- GPU batch Gaussian 采样。
- 增强结果落盘缓存。
- 预计算增强数据。
- 减少重复统计和重复 I/O。

目标：

```text
将数据生成阶段从瓶颈阶段压缩为可忽略的预处理阶段。
```

### 6.4 优化特征提取阶段

可选方案：

- 增大 `extract_batch_size`。
- 使用 FP16。
- 预加载模型，避免重复初始化。
- 合并小文件缓存。
- 多 GPU 分摊 client。
- 缓存已提取特征。

目标：

```text
单 GPU 提升特征提取吞吐，多 GPU 汇总达到 10000+ img/s。
```

### 6.5 前后端系统阶段

前后端系统的 `10000+ QPS` 应单独作为在线服务指标实现：

- 查询型接口通过 Redis 缓存和数据库索引优化。
- 推理型接口通过 batch 推理和模型预加载优化。
- 任务型接口通过异步队列解耦。
- 多实例部署，通过 Nginx 负载均衡横向扩展。
- 使用 wrk/Locust/JMeter 压测。

该阶段不能直接复用训练 QPS 结论，需要独立压测报告。

## 7. 最终建议

建议将任务书中的 `10000+ QPS` 解释为：

```text
训练阶段样本级吞吐能力。系统在稳定训练轮次中，每秒可完成 10000+ 个训练样本的处理。
```

当前可给出的达标结论：

```text
基于服务器 DomainNet 4 domains + CNN + FedAvg 单机 standalone 实验，稳定训练轮次每轮处理 85881 个样本，耗时约 5.0-5.7 秒，训练阶段吞吐约 1.5w-1.8w samples/s，已经满足 10000+ QPS 指标。
```

需要同时保留边界说明：

```text
该指标表示稳定训练阶段的样本处理吞吐，不等同于完整端到端流程吞吐、数据生成吞吐、联邦通信请求 QPS 或前后端在线 API QPS。
```
