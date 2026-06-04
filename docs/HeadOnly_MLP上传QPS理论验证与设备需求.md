# HeadOnly MLP 上传 QPS 理论验证与设备需求

## 1. 指标定义

本文档讨论的 QPS 不是完整训练轮 QPS，也不是训练样本吞吐，而是 HeadOnly 方案中 MLP 参数上传阶段的请求 QPS：

```text
MLP upload QPS = 同步上传窗口内收到的 MLP 参数上传请求数 / 上传窗口耗时
```

其中同步上传窗口定义为：

```text
子服务器发出同步上传信号后，
从第一个或同步信号时刻开始计时，
直到最后一个 MLP 参数上传请求完成。
```

建议最终实验采用更严格口径：

```text
upload_window_sec = last_upload_complete_time - go_signal_time
upload_qps = received_uploads / upload_window_sec
```

本地训练时间、等待其他客户端训练完成的 barrier 时间、测试准确率评估时间不计入该上传 QPS 分母。

## 2. 当前 HeadOnly MLP 参数大小

当前 OfficeHome + ViT-B-16 HeadOnly 配置中，MLP head 是线性分类头：

```text
input_dim = 512
num_classes = 65
hidden_dim = 0
```

参数量：

```text
weight = 65 * 512 = 33,280
bias = 65
total_params = 33,280 + 65 = 33,345
```

float32 参数大小：

```text
payload_raw_bytes = 33,345 * 4
                  = 133,380 bytes
                  ≈ 130.25 KiB
```

正式 100 轮 cache-hot 训练日志也验证了这个量级：

```text
rounds = 100
clients = 60
upload_requests = 100 * 60 = 6000
total_upload_volume ≈ 763.21 MB
single_upload ≈ 763.21 MB / 6000
             ≈ 127 KiB
```

因此后续理论估计采用：

```text
raw payload/request ≈ 130 KiB
conservative payload/request ≈ 160-180 KiB
```

保守值包含序列化、协议头、TCP/gRPC 开销和系统抖动。

## 3. 1 万 upload QPS 的网络需求

目标：

```text
Q = 10,000 upload requests/s
```

按原始 payload 计算：

```text
bandwidth_raw = 10,000 * 133,380 bytes/s
              = 1,333,800,000 bytes/s
              ≈ 1.33 GB/s
```

换算为 bit/s：

```text
1.33 GB/s * 8 ≈ 10.67 Gbps
```

按保守 payload 170 KiB/request 计算：

```text
bandwidth_safe = 10,000 * 170 KiB/s
               ≈ 1.70 GB/s
               ≈ 13.6 Gbps
```

因此理论结论是：

```text
10GbE：理论带宽不足或余量极小，难以稳定达到 1 万 upload QPS
25GbE：理论可行，有较合理余量
100GbE：稳定性最好，但预算成本更高
```

当前服务器通过 `/sys/class/net/eth0/speed` 查询到：

```text
eth0_speed_Mbps = 10000
```

即当前服务器网卡显示为 10GbE。单台 10GbE 作为所有客户端上传入口时，不适合严格证明 1 万 upload QPS。

## 4. 分层结构下的计算公式

设：

```text
M = 子服务器数量
C = 每台子服务器管理的客户端数量
T = 同步上传窗口秒数
S = 单次 MLP 上传 payload，约 130 KiB
```

全局上传 QPS：

```text
Q_global = M * C / T
```

每台子服务器上传 QPS：

```text
Q_sub = C / T
```

每台子服务器入口带宽：

```text
B_sub = Q_sub * S
```

中心服务器压力较小，因为子服务器先聚合客户端 MLP 参数，中心服务器只接收每个子服务器上传的聚合后 MLP 参数：

```text
center_upload_requests_per_round = M
```

因此 1 万 QPS 的压力主要发生在：

```text
client -> subserver
```

而不是：

```text
subserver -> center
```

## 5. 典型配置计算

### 5.1 十台子服务器，每台 500 clients

```text
M = 10
C = 500
T = 0.5s
```

全局 QPS：

```text
Q_global = 10 * 500 / 0.5
         = 10,000 upload req/s
```

每台子服务器 QPS：

```text
Q_sub = 500 / 0.5
      = 1,000 upload req/s
```

每台子服务器原始入口带宽：

```text
B_sub = 1,000 * 130 KiB
      ≈ 130 MiB/s
      ≈ 1.1 Gbps
```

该方案对单台子服务器压力低，但设备数量多。

### 5.2 两台子服务器，每台 2500 clients

```text
M = 2
C = 2500
T = 0.5s
```

全局 QPS：

```text
Q_global = 2 * 2500 / 0.5
         = 10,000 upload req/s
```

每台子服务器 QPS：

```text
Q_sub = 2500 / 0.5
      = 5,000 upload req/s
```

每台子服务器原始入口带宽：

```text
B_sub = 5,000 * 130 KiB
      ≈ 650 MiB/s
      ≈ 5.3 Gbps
```

按保守 payload 170 KiB/request：

```text
B_sub_safe = 5,000 * 170 KiB
           ≈ 850 MiB/s
           ≈ 7.0 Gbps
```

该方案在 10GbE 上理论可行但余量不大，推荐 25GbE。

### 5.3 一台子服务器硬扛 5000 clients

```text
M = 1
C = 5000
T = 0.5s
```

全局 QPS：

```text
Q_global = 1 * 5000 / 0.5
         = 10,000 upload req/s
```

入口带宽：

```text
B = 10,000 * 130 KiB
  ≈ 1.3 GB/s
  ≈ 10.7 Gbps
```

保守估计：

```text
B_safe ≈ 13.6 Gbps 或更高
```

该方案需要 25GbE 起步，且 CPU、连接调度和序列化压力集中，不建议作为资金有限时的第一选择。

## 6. 服务器数量需求

### 6.1 如果都是 10GbE 云服务器

单台 10GbE 理论上限：

```text
10 Gbps / 8 = 1.25 GB/s
```

而 1 万 upload QPS 原始需求约：

```text
1.33 GB/s
```

加开销后约：

```text
1.6-1.8 GB/s
```

因此单台 10GbE 不适合证明 1 万 upload QPS。更稳妥的最低模拟配置是：

```text
2 台 10GbE 子服务器
2-3 台 10GbE 客户端压测机
1 台中心服务器可先与其中一台子服务器复用
```

也就是：

```text
最低：4 台 10GbE 云服务器
更稳：5 台 10GbE 云服务器
```

每台子服务器承担约：

```text
5000 upload req/s
5.3-7.0 Gbps 入口流量
```

### 6.2 如果有 25GbE 云服务器

25GbE 理论带宽：

```text
25 Gbps / 8 = 3.125 GB/s
```

单台 25GbE 接收 1 万 upload QPS 的带宽余量较合理：

```text
需求：1.3-1.8 GB/s
能力：理论 3.125 GB/s
```

推荐配置：

```text
1 台 25GbE 子服务器
2 台 10GbE 客户端压测机
中心服务器可与子服务器复用
```

或者：

```text
2 台 25GbE 云服务器
一台做子服务器，一台做客户端压测机
```

### 6.3 如果只有两台 10GbE 云服务器

两台 10GbE 可以做可行性下界测试，但不能严格证明 1 万 upload QPS：

```text
A = 子服务器
B = 客户端压测机
```

由于 B 出网和 A 入网都只有 10GbE，而 1 万 upload QPS 原始需求已经约 10.7Gbps，因此两台 10GbE 大概率只能测到：

```text
5000-9000 upload QPS
```

该结果可以用于外推：

```text
如果单台 10GbE 子服务器实测 5000 QPS，
则 2 台同规格子服务器理论可达到约 10000 QPS。
```

但最终严格证明仍需要总入口带宽超过目标需求。

## 7. 设备建议

### 7.1 资金有限但希望较可信验证

```text
2 台 10GbE 子服务器
2 台 10GbE 客户端压测机
中心服务器与其中一台子服务器复用
```

单台子服务器建议：

```text
CPU：16-32 核
内存：64-128GB
网络：10GbE
磁盘：NVMe
GPU：不需要
```

单台客户端压测机建议：

```text
CPU：16 核左右
内存：32-64GB
网络：10GbE
```

### 7.2 更干净的一万 QPS 验证

```text
1 台 25GbE 子服务器
2 台 10GbE 客户端压测机
```

或：

```text
2 台 25GbE 服务器
一台做子服务器
一台做客户端压测机
```

单台 25GbE 子服务器建议：

```text
CPU：32 核以上
内存：128GB
网络：25GbE
磁盘：NVMe
GPU：不需要
```

## 8. 实验设计

### 8.1 阶段一：上传窗口压测

目的：只验证 MLP 参数上传 QPS。

步骤：

```text
1. 使用真实 HeadOnly 训练得到的 MLP state_dict 或同大小 payload。
2. 子服务器等待所有逻辑客户端连接。
3. 子服务器发出 GO 同步信号。
4. 所有客户端集中上传 MLP 参数。
5. 子服务器记录上传窗口和吞吐。
```

记录指标：

```text
expected_uploads
received_uploads
go_signal_time
first_upload_complete_time
last_upload_complete_time
upload_window_sec = last_upload_complete_time - go_signal_time
upload_qps = received_uploads / upload_window_sec
payload_MBps = total_payload_bytes / upload_window_sec
success_ratio = received_uploads / expected_uploads
```

建议通过标准：

```text
upload_qps >= 10000
success_ratio >= 0.99
payload_MBps >= 1300 MB/s 原始吞吐
```

### 8.2 阶段二：分层上传与聚合

目的：验证子服务器分流后，中心服务器压力可控。

结构：

```text
client -> subserver -> center server
```

每个子服务器记录：

```text
client_to_subserver_upload_qps
client_to_subserver_payload_MBps
subserver_aggregation_time
```

中心服务器记录：

```text
subserver_to_center_upload_count
center_aggregation_time
```

### 8.3 阶段三：cache-hot 真实训练接入

目的：验证真实训练逻辑下，训练完成后的集中上传 QPS。

步骤：

```text
1. 第一次完整跑真实 OfficeHome + ViT + GGEUR round0，生成增强样本缓存。
2. 后续开启 SKIP_ROUND0_IF_AUG_CACHE=1，直接读取增强样本缓存。
3. 客户端本地训练 MLP。
4. 训练完成后等待 barrier。
5. 子服务器发同步上传信号。
6. 客户端集中上传 MLP 参数。
7. 子服务器聚合，再上传中心服务器。
```

需要拆分记录：

```text
local_train_time
barrier_wait_time
upload_window_sec
upload_qps
subserver_aggregation_time
center_aggregation_time
download_window_sec
round_total_time
test_accuracy
```

## 9. 当前初步压测结果

已实现轻量 TCP 上传窗口压测脚本：

```text
scripts/benchmark_headonly_mlp_upload_window.py
```

服务器本机 loopback 初步结果：

```text
60 clients:
go_to_last_upload_complete_sec ≈ 0.01936s
upload_qps_go_to_last ≈ 3099 req/s

500 clients:
go_to_last_upload_complete_sec ≈ 0.17281s
upload_qps_go_to_last ≈ 2893 req/s
```

该结果说明：

```text
1. T 可以通过实验测量，不必只靠估计。
2. 单进程/单机 loopback 可以作为保守基线。
3. 要证明 1 万 QPS，需要多接收 worker、多子服务器或更高带宽。
```

注意该压测与最终真实系统仍有差异：

```text
未走 FederatedScope/gRPC 消息栈
未做真实 state_dict 序列化
未经过真实多机网络
未做子服务器 FedAvg 聚合
未接入训练与准确率评估
```

因此它只能作为第一阶段理论验证和设备估计依据，最终仍需进行多机真实协议压测。

## 10. 结论

理论上，HeadOnly MLP 上传 1 万 QPS 是可行的，但关键约束是网络和接收端并发能力：

```text
单次 MLP 上传 ≈ 130 KiB
1 万 upload QPS 原始带宽需求 ≈ 10.7 Gbps
考虑开销后建议总入口带宽 >= 14-16 Gbps
```

因此：

```text
两台 10GbE：可做下界测试，不能严格证明 1 万+
四到五台 10GbE：可较可信模拟 1 万+
两台 25GbE：可更干净验证 1 万+
```

最终推荐资金有限方案：

```text
2 台 10GbE 子服务器
2-3 台 10GbE 客户端压测机
中心服务器与子服务器复用
```

更推荐的干净验证方案：

```text
1 台 25GbE 子服务器
2 台 10GbE 客户端压测机
```

GPU 不属于 MLP 上传 QPS 的主要瓶颈；GPU 主要用于首次真实数据特征提取和增强样本缓存生成。
