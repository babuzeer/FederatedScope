# HeadOnly 分层架构 10000+ QPS 验收口径与实施计划

本文是 HeadOnly 分层参数服务方案的最终确认版口径文档。后续实现、实验、报告和验收均以本文为准。

当前第一阶段只实现和验收一个基础能力：

```text
10000+ 客户端并发下发响应能力
```

对应主指标：

```text
Dispatch-Start QPS
模型下发启动 QPS
```

其他能力，例如客户端并发上传、训练完成吞吐、完整模型调用 QPS，先作为后续阶段规划，不进入第一阶段验收范围。

## 1. 核心目标

HeadOnly 分层参数服务架构要证明的不是单机网络吞吐，也不是单条公网链路带宽，而是：

```text
通过 coordinator -> subservers -> clients 的分层设计，
把万级客户端的模型下发请求分散到多个 subserver，
从而提升同一时间窗口内的客户端并发响应能力。
```

最终验收要突出：

```text
10000+ QPS 来自分层架构的横向扩展能力，
而不是单纯来自更高配置设备。
```

## 2. 第一阶段主指标

第一阶段唯一主指标定义为：

```text
Dispatch-Start QPS
```

一次成功的 dispatch-start 定义为：

```text
1. client 向 assigned subserver 请求本轮模型参数；
2. subserver 返回 model_version、payload_size 等元信息；
3. subserver 发送首个模型参数分片；
4. client 收到首个参数分片后立即 ACK。
```

该指标表示：

```text
系统每秒可以让多少个客户端被成功调度到对应 subserver，
并进入模型参数下发状态。
```

它不表示：

```text
客户端已经完整收到全部模型参数。
```

## 3. QPS 计算方式

统计窗口：

```text
release_time -> last_first_chunk_ack_time
```

其中：

```text
release_time:
  coordinator 发布本轮模型版本后，
  subserver 开始响应客户端下发请求的时间。

last_first_chunk_ack_time:
  最后一个客户端收到首个参数分片并 ACK 的时间。
```

公式：

```text
Dispatch-Start QPS =
  first_chunk_ack_clients
  /
  (last_first_chunk_ack_time - release_time)
```

第一阶段核心输出字段：

```text
expected_clients
first_chunk_ack_clients
success_ratio
dispatch_start_window_sec
dispatch_start_qps
latency_p50_sec
latency_p95_sec
latency_p99_sec
subserver_qps_min
subserver_qps_avg
subserver_qps_max
```

## 4. 10000+ QPS 的固定含义

第一阶段文档、实验和验收中，`10000+ QPS` 固定解释为：

```text
在一次全局模型发布后，系统能够在约 1 秒窗口内，
让 10000+ 个客户端收到模型版本元信息和首个参数分片，
并返回 first-chunk ACK。
```

成功标准：

```text
expected_clients >= 10000
first_chunk_ack_clients >= 10000
success_ratio = 1.0
dispatch_start_qps >= 10000
P95 dispatch_start_latency <= 1.0s
```

如果 `dispatch_start_qps >= 10000`，但 `P95 dispatch_start_latency > 1.0s`，不能直接作为理想验收通过，需要单独说明尾延迟风险。

## 5. 为什么不以完整下载 QPS 为主指标

当前 HeadOnly MLP 参数 payload 约为：

```text
133,380 bytes/client
```

如果要求 10000 个客户端在 1 秒内完整收到该 payload，则纯 payload 带宽需求为：

```text
133,380 * 10,000 = 1,333,800,000 B/s
约 1.24 GiB/s
约 10.67 Gbps
```

在公网场景下，完整 payload 下载主要受公网出口带宽、链路质量和接收侧带宽限制。它应作为辅助带宽指标报告，但不适合作为第一阶段证明分层架构并发响应能力的主指标。

## 6. 辅助指标

第一阶段仍可以辅助报告以下指标，但它们不能替代 Dispatch-Start QPS。

### 6.1 Header Response QPS

定义：

```text
client 发起模型参数请求；
subserver 返回 model_version、payload_size 等 header；
client 收到 header 后 ACK。
```

含义：

```text
请求接入和控制面响应能力。
```

限制：

```text
只返回 header 太接近空请求，不能单独证明模型下发已经开始。
```

### 6.2 Full Payload Download QPS

定义：

```text
client 完整收到本轮 MLP 参数 payload 后 ACK。
```

含义：

```text
完整模型参数下载吞吐能力。
```

限制：

```text
该指标受公网带宽强约束，必须作为带宽相关指标单独报告。
```

## 7. 用扩展曲线证明架构贡献

第一阶段验收不能只展示一次 10000+ 结果，还必须展示扩展曲线，证明提升来自分层架构。

对比对象：

```text
Flat baseline:
  1 个 dispatch process 直接服务所有 clients。

Hierarchical:
  coordinator 发布模型版本；
  N 个 subserver process 分摊 client groups；
  每个 subserver 独立处理自己的 client group。
```

推荐实验矩阵：

```text
Architecture   Subservers   Clients   Ack Mode      Dispatch-Start QPS   P95 Latency   Success
Flat           1            10000     first-chunk   ...                  ...           ...
Hierarchical   2            10000     first-chunk   ...                  ...           ...
Hierarchical   4            10000     first-chunk   ...                  ...           ...
Hierarchical   8            10000     first-chunk   ...                  ...           ...
```

期望趋势：

```text
Hierarchical 2 subservers > Flat 1 process
Hierarchical 4 subservers > Hierarchical 2 subservers
Hierarchical 8 subservers > Hierarchical 4 subservers
```

更理想的趋势：

```text
总 Dispatch-Start QPS 随 subserver 数量近似线性增长。
```

这才是证明架构有效的关键证据。

## 8. 第一阶段实施计划

### 8.1 代码实现范围

扩展现有 QPS benchmark，使其支持：

```text
ack_mode = header | first-chunk | full
```

第一阶段默认模式：

```text
ack_mode = first-chunk
```

行为：

```text
server/subserver:
  发送 model_version / payload_size / chunk_index=0
  发送首个参数分片

client:
  收到首个参数分片后立即 ACK
  不等待完整 payload 下载
```

同时保留 `full` 模式，用于辅助报告完整 payload 下载能力。

### 8.2 进程模型

第一阶段必须使用多进程 subserver 模式：

```text
1 subserver = 1 independent Python process
```

避免多个 subserver 放在同一个 Python event loop 中，导致无法体现横向扩展效果。

推荐测试形态：

```text
server-mp:
  subserver_1 -> port 39301
  subserver_2 -> port 39302
  subserver_3 -> port 39303
  subserver_4 -> port 39304
  ...

client-mp:
  client_group_1 -> subserver_1
  client_group_2 -> subserver_2
  client_group_3 -> subserver_3
  client_group_4 -> subserver_4
  ...
```

### 8.3 实验步骤

步骤 1：功能 smoke test

```text
1 subserver x 10 clients
ack_mode = first-chunk
success_ratio = 1.0
```

步骤 2：固定每个 subserver 压力，验证横向扩展

```text
1 subserver x 500 clients
2 subservers x 500 clients
4 subservers x 500 clients
8 subservers x 500 clients
```

步骤 3：冲击 10000+ 客户端

推荐组合：

```text
4 subservers x 2500 clients = 10000 clients
8 subservers x 1250 clients = 10000 clients
10 subservers x 1000 clients = 10000 clients
```

步骤 4：记录最佳成功配置

成功条件：

```text
expected_clients >= 10000
first_chunk_ack_clients >= 10000
success_ratio = 1.0
dispatch_start_qps >= 10000
P95 dispatch_start_latency <= 1s
```

### 8.4 第一阶段交付物

第一阶段完成时，需要交付：

```text
1. 支持 ack_mode=first-chunk 的 benchmark 脚本。
2. 支持 server-mp/client-mp 的多进程测试路径。
3. flat baseline vs hierarchical 的对比实验结果。
4. 至少一组 10000+ clients、success_ratio=1.0 的成功结果。
5. 汇总表，包含 QPS、P50/P95/P99 latency、per-subserver QPS。
```

## 9. 后续阶段规划

以下指标暂不进入第一阶段实现范围。等 Dispatch-Start QPS 说法和实验真正落地后，再逐步扩展。

### 9.1 Upload-Start QPS

目标：

```text
10000+ 客户端并发上传模型能力。
```

成功定义：

```text
client 向 assigned subserver 上传本地模型更新请求；
subserver 成功接收 update header 和首个 update chunk；
subserver 返回 upload-start ACK。
```

说明：

```text
Upload-Start QPS 证明上传入口并发接入能力；
完整模型更新上传吞吐需要单独报告。
```

### 9.2 Train-Start / Train-Complete Throughput

目标：

```text
训练任务调度能力和训练完成回收能力。
```

说明：

```text
Train-Complete Throughput 受客户端算力、数据加载、local_update_steps、
模型大小和特征缓存影响，不能简单归因于分层参数服务架构。
```

### 9.3 Model-Dispatch Invocation QPS

目标：

```text
10000+ 客户端模型调用响应能力。
```

第一阶段以后优先实现：

```text
model_dispatch_invocation_qps
```

它与 Dispatch-Start QPS 对齐：

```text
请求模型参数下发，并收到首个参数分片。
```

如果后续定义为真实模型推理调用，则它主要受 GPU/CPU 推理性能影响，需要单独作为推理服务指标。

## 10. 禁止混淆的说法

不要把第一阶段结果写成：

```text
10000 个客户端在 1 秒内完整下载了模型参数。
```

除非 Full Payload Download QPS 实验也证明了该结论。

不要把只收到 header 写成：

```text
模型下发已完成。
```

第一阶段主口径必须始终是：

```text
收到首个参数分片并 ACK。
```

## 11. 推荐报告表述

建议报告中使用以下表述：

```text
我们将模型下发能力拆分为“下发启动能力”和“完整载荷传输能力”。
其中 Dispatch-Start QPS 衡量系统在每轮全局模型发布后，
能够以多高的速率将客户端调度到分层 subserver 并启动参数传输。
一次 dispatch-start 成功定义为客户端收到模型版本元信息和首个参数分片。
该指标刻画的是分层参数服务架构的并发响应能力，而非公网带宽上限。
完整模型载荷下载吞吐作为带宽相关指标单独报告。
```

第一阶段验收结论建议写成：

```text
在相同公网环境和相同硬件条件下，分层 subserver 架构通过横向扩展下发入口，
将模型下发启动能力提升到 10000+ Dispatch-Start QPS。
```

## 12. 最终确认

第一阶段最终确认如下：

```text
目标：
  实现 10000+ 客户端并发下发响应能力。

主指标：
  Dispatch-Start QPS。

成功事件：
  client 收到模型版本元信息和首个参数分片，并 ACK。

不包含：
  完整模型参数下载完成。
  完整模型上传完成。
  本地训练完成。
  真实模型推理完成。

证明方式：
  flat baseline vs hierarchical subserver 扩展曲线。
```
