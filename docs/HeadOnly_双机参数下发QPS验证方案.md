# HeadOnly 双机参数下发 QPS 验证方案

## 1. 目标调整

本方案将 HeadOnly 系统中的参数服务改为分层下发：

```text
主服务器 center server
  -> 子服务器 subservers
      -> 各自管理的 clients
```

新的 QPS 口径确定为：

```text
各个子服务器向客户端下发全局 MLP 参数的并发请求 QPS
```

只统计通信步骤，不统计：

```text
客户端本地训练时间
客户端等待 barrier 时间
主服务器聚合时间
测试准确率评估时间
日志落盘时间
```

## 2. 修改后的系统流程

原有 HeadOnly 训练流程中，主服务器聚合后直接向所有客户端下发全局 MLP 参数。修改后流程为：

```text
1. clients 本地训练 MLP head
2. clients 上传本地 MLP 参数到各自 subserver
3. subserver 聚合自己管理的 clients 的 MLP 参数
4. subserver 将聚合后的局部参数上传 center server
5. center server 聚合所有 subservers 的局部参数，得到全局 MLP 参数
6. center server 将全局 MLP 参数下发给各个 subserver
7. 每个 subserver 并发下发全局 MLP 参数给自己管理的 clients
8. clients 接收全局参数，进入下一轮训练
```

本次重点验证步骤 7：

```text
subserver -> clients 的全局 MLP 参数下发 QPS
```

## 3. QPS 统计口径

### 3.1 单个子服务器 QPS

对每个子服务器 `s`，记录：

```text
go_time_s
first_send_complete_time_s
last_send_complete_time_s
expected_clients_s
completed_clients_s
total_payload_bytes_s
```

建议采用严格口径：

```text
download_window_sec_s = last_send_complete_time_s - go_time_s
download_qps_s = completed_clients_s / download_window_sec_s
payload_MBps_s = total_payload_bytes_s / download_window_sec_s
success_ratio_s = completed_clients_s / expected_clients_s
```

其中：

```text
go_time_s = 子服务器收到 center 全局参数并开始向客户端下发的时刻
last_send_complete_time_s = 该子服务器最后一个客户端下发完成的时刻
```

如果使用 gRPC 或 TCP：

```text
send complete = server 侧完成参数写入/收到客户端 ACK
```

建议优先使用“收到客户端 ACK”的时间作为完成时刻，更接近真实通信完成。

### 3.2 多子服务器全局 QPS

多子服务器下发时，不建议简单平均各个子服务器 QPS，而应采用全局窗口：

```text
global_go_time = min(go_time_s)
global_last_complete_time = max(last_send_complete_time_s)
global_download_window_sec = global_last_complete_time - global_go_time
global_completed_clients = sum(completed_clients_s)
global_download_qps = global_completed_clients / global_download_window_sec
global_payload_MBps = sum(total_payload_bytes_s) / global_download_window_sec
```

同时保留每个子服务器的局部 QPS：

```text
subserver_download_qps_s
subserver_payload_MBps_s
subserver_success_ratio_s
```

最终报告建议同时写：

```text
global_download_qps
min_subserver_qps
avg_subserver_qps
max_subserver_qps
global_success_ratio
p50/p95/p99 client_receive_latency
```

其中：

```text
global_success_ratio = global_completed_clients / global_expected_clients
```

### 3.3 为什么不用训练轮时间做分母

本指标只验证参数服务能力，因此分母只包含参数下发通信窗口：

```text
下发 QPS = 下发请求数 / 参数下发窗口时间
```

完整训练轮耗时应单独记录：

```text
round_total_time
local_train_time
barrier_wait_time
upload_time
aggregation_time
download_window_sec
eval_time
```

## 4. 两台服务器的角色设计

假设只有两台可内网通信的云服务器：

```text
机器 A
机器 B
```

### 4.1 推荐双机部署

为了让“subserver -> client”下发流量真实经过网络，推荐：

```text
机器 A：
  center server
  subserver_1
  subserver_2
  ...
  subserver_M

机器 B：
  client group 1
  client group 2
  ...
  client group M
```

通信路径：

```text
center server -> subservers：机器 A 本机通信
subservers -> clients：机器 A -> 机器 B，真实跨机器网络
clients -> subservers：机器 B -> 机器 A，真实跨机器网络
```

本次 QPS 重点是：

```text
subservers on A -> clients on B
```

这个部署能保证参数下发经过真实网卡和云内网。

### 4.2 为什么不把 subserver 和 client 放同一台机器

如果部署为：

```text
机器 A：center server
机器 B：subservers + clients
```

则：

```text
subserver -> client
```

大概率走本机 loopback，不经过真实网卡，不能代表真实网络下发 QPS。

因此，若验证下发 QPS，应保证：

```text
subserver 和 client 在不同物理/云主机上
```

## 5. 多账号与多 IP 的模拟方式

### 5.1 多账号能模拟什么

在同一台云服务器上创建多个 Linux 用户账号，可以模拟：

```text
不同客户端身份
不同进程权限
不同日志目录
不同实验隔离
```

但不能模拟：

```text
不同物理网络
不同网卡带宽
不同真实机器
```

因此，多账号对 QPS 的真实性帮助有限。QPS 主要由：

```text
进程数量
连接数量
网卡带宽
网络栈
服务端 worker 数
```

决定。

### 5.2 多 IP 能模拟什么

如果云平台支持给同一台服务器绑定多个内网 IP，可以在机器 B 上配置多个源 IP，让不同 client group 使用不同 source IP。

这可以模拟：

```text
多个客户端来源 IP
更多连接分布
更接近真实客户端网络身份
```

但仍不能突破该机器的总网卡带宽：

```text
多个 IP 共享同一张虚拟网卡/同一条网络带宽
```

因此多 IP 适合做连接身份模拟，不适合当作多台机器的网络带宽替代品。

### 5.3 最低可行模拟

如果云平台无法提供多个内网 IP，可以直接使用：

```text
同一个 client 机器内网 IP
大量 client 进程/协程
不同 source port
不同 client_id
```

这已经足够验证：

```text
subserver 参数服务并发下发能力
单条 A -> B 网络链路的吞吐上限
```

## 6. 两台服务器上的具体模拟拓扑

假设：

```text
机器 A 内网 IP = A_IP
机器 B 内网 IP = B_IP
```

机器 A 启动：

```text
center server: 端口 39000
subserver_1: 端口 39101
subserver_2: 端口 39102
subserver_3: 端口 39103
subserver_4: 端口 39104
```

机器 B 启动：

```text
client group 1 -> A_IP:39101
client group 2 -> A_IP:39102
client group 3 -> A_IP:39103
client group 4 -> A_IP:39104
```

例如总客户端数 5000：

```text
subserver_1 管理 1250 clients
subserver_2 管理 1250 clients
subserver_3 管理 1250 clients
subserver_4 管理 1250 clients
```

下发 QPS 目标：

```text
如果 5000 个 client 在 0.5s 内完成下发：
global_download_qps = 5000 / 0.5 = 10000 req/s
```

## 7. 网络与设备需求估算

当前 HeadOnly MLP 参数大小：

```text
input_dim = 512
num_classes = 65
params = 512 * 65 + 65 = 33,345
payload_raw = 33,345 * 4 = 133,380 bytes ≈ 130 KiB
```

如果下发目标是：

```text
10,000 download req/s
```

原始下发吞吐：

```text
10,000 * 133,380 bytes/s
= 1.33 GB/s
≈ 10.67 Gbps
```

考虑协议、序列化和系统开销：

```text
建议按 14-16 Gbps 设计
```

因此：

```text
单台 10GbE 下发服务器：理论不足或余量极小
单台 25GbE 下发服务器：理论可行
两台 10GbE 子服务器并行：理论可行
```

在只有两台服务器时，如果 A 和 B 都是 25Gbps：

```text
A -> B 总链路理论 25Gbps
目标需求约 14-16Gbps
理论可验证 1 万下发 QPS
```

如果 A 和 B 都是 10Gbps：

```text
A -> B 总链路理论 10Gbps
目标原始需求已约 10.67Gbps
大概率不能严格验证 1 万下发 QPS
```

## 8. 实验阶段设计

### 8.1 阶段一：纯参数下发压测

目的：

```text
只验证 subserver -> clients 参数下发 QPS
```

不跑训练，不做聚合，只使用真实 MLP 参数大小 payload。

步骤：

```text
1. 机器 A 启动 M 个 subserver 下发服务。
2. 机器 B 启动 N 个逻辑 clients。
3. clients 连接到各自 subserver，等待参数。
4. 所有 subservers 同时开始下发全局 MLP 参数。
5. clients 收到参数后返回 ACK。
6. subservers 记录下发窗口、完成数量、payload 吞吐。
```

指标：

```text
global_download_qps
subserver_download_qps_s
global_payload_MBps
success_ratio
p50/p95/p99 receive latency
```

### 8.2 阶段二：加入 center -> subserver

目的：

```text
验证 center 将聚合后全局参数交给 subserver 后，
subserver 再向 clients 下发的完整参数服务链路
```

步骤：

```text
1. center 生成或接收全局 MLP 参数。
2. center 将全局参数发送给所有 subservers。
3. subservers 收到全局参数后，开始并发下发给 clients。
4. 分别统计 center->subserver 时间和 subserver->client 下发 QPS。
```

QPS 指标仍只统计：

```text
subserver -> client
```

center -> subserver 时间单独记录。

### 8.3 阶段三：接入 cache-hot HeadOnly 训练

目的：

```text
在真实 HeadOnly 训练逻辑中验证参数下发 QPS
```

步骤：

```text
1. 使用已有 OfficeHome + ViT + GGEUR 增强样本缓存。
2. 开启 SKIP_ROUND0_IF_AUG_CACHE=1。
3. clients 本地训练 MLP。
4. clients 上传本地参数到 subserver。
5. subserver 聚合后上传 center。
6. center 聚合全局参数并发给 subservers。
7. subservers 并发下发给 clients。
8. 统计下发通信窗口 QPS。
```

需要拆分指标：

```text
local_train_time
client_to_subserver_upload_time
subserver_aggregation_time
subserver_to_center_upload_time
center_aggregation_time
center_to_subserver_download_time
subserver_to_client_download_window_sec
subserver_to_client_download_qps
round_total_time
test_accuracy
```

## 9. 两台服务器验证的局限

两台服务器可以验证：

```text
单条 A -> B 网络链路上的参数下发 QPS
多个 subserver 进程的并发下发能力
多个 client group 的连接和 ACK 行为
QPS 统计口径是否正确
```

不能完全验证：

```text
多个物理 subserver 的总入口/出口带宽叠加
真实多机多网络路径
跨机器 subserver 到 center 的复杂拓扑
云平台多租户网络抖动下的大规模稳定性
```

因此两台服务器实验应定位为：

```text
双机可行性验证
```

最终若要证明多子服务器系统级 1 万 QPS，应扩展到：

```text
多台 subserver
多台 client 压测机
center 独立部署
```

## 10. 推荐执行顺序

1. 查询两台服务器内网带宽：

```bash
cat /sys/class/net/eth0/speed
```

2. 使用 `iperf3` 测 A -> B 实际内网吞吐：

```bash
# 机器 B
iperf3 -s

# 机器 A
iperf3 -c B_IP -P 8 -t 30
```

3. 跑纯参数下发压测：

```text
M = 1, 2, 4 个 subserver 进程
N = 500, 1000, 3000, 5000 个 clients
```

4. 记录全局下发 QPS：

```text
global_download_qps = global_completed_clients / global_download_window_sec
```

5. 如果纯下发压测达到目标，再接入真实 HeadOnly cache-hot 训练。

## 11. 结论

新的参数下发架构是合理的：

```text
center server 只负责全局聚合和向 subservers 下发一次参数
subservers 负责面向 clients 的高并发参数下发服务
```

在两台服务器上，推荐部署为：

```text
机器 A：center + 多个 subserver 进程
机器 B：大量 client 进程/协程
```

这样可以保证核心 QPS 指标：

```text
subserver -> client 参数下发 QPS
```

真实经过 A -> B 网络链路。

如果两台服务器都是 25Gbps，理论上可以验证 1 万下发 QPS；如果都是 10Gbps，则更适合作为下界测试，难以严格证明 1 万 QPS。
