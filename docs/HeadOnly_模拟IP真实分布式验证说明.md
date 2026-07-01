# HeadOnly 模拟 IP 真实分布式验证说明

本文档记录在实验室经费有限、云平台不开放业务端口的条件下，使用单台
bjb1 服务器的 loopback 多 IP 方式验证 HeadOnly 真实分布式联邦学习系统。

## 验证目标

该方案验证以下内容：

- FederatedScope `distributed` 模式的真实 server/client 进程通信；
- OfficeHome 真实数据集；
- ViT-B/16 特征提取/缓存；
- HeadOnly round0 统计上传、协方差/原型下发、增强样本生成；
- MLP head-only 联邦训练；
- accuracy/round time/train QPS 日志提取；
- subserver -> clients 参数下发 QPS 独立通信窗口验证；
- 多个模拟 client IP 的进程级通信拓扑。

该方案不声称验证真实物理多机网络带宽。它是“单机多 IP 分布式系统验证”，
后续真实多机仍需要业务端口公网映射、互通内网 IP，或支持安全组的云主机。

## 运行脚本

主脚本：

```bash
scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
```

默认拓扑：

```text
server: 127.77.0.11
client_1: 127.77.0.101
client_2: 127.77.0.102
...
```

说明：避免使用 `127.77.0.10` 作为 server IP，因为当前 FederatedScope/YACS
日志中曾把该地址显示成 `127.77.0.1`，影响日志审计。

## 小规模 Smoke

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=loopback_multiip_smoke_2c_2r_gen1 \
CLIENT_NUM=2 \
SAMPLE_CLIENT_NUM=2 \
TOTAL_ROUNDS=2 \
GEN_NUM=1 \
SERVER_PORT=51551 \
CLIENT_PORT_BASE=52551 \
CASE_TIMEOUT=1800 \
RUN_QPS=1 \
QPS_SUBSERVERS=2 \
QPS_CLIENT_GROUPS=2 \
QPS_CLIENTS_PER_GROUP=20 \
QPS_BASE_PORT=39551 \
bash scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
```

已完成 smoke 结果：

```text
run_dir:
/root/autodl-tmp/FederatedScope/exp/headonly_system/loopback_multiip_runs/loopback_multiip_smoke_2c_2r_gen1

distributed training:
round_count = 1
total_train_samples = 96
total_round_time_sec = 0.4
overall_train_qps = 240 samples/s
final average accuracy = 0.0176
round0 statistics payload bytes = 31,399,128
round0 augmentation samples = 96

download QPS:
completed = 40/40
success_ratio = 1.0
global_download_qps_go_to_last = 2834.65
download_window_sec = 0.014111
payload bandwidth = 360.57 MiB/s
```

该 smoke 使用 `GEN_NUM=1`，只用于验证链路完整性，不用于准确率结论。

## 正式小规模验证

建议先跑 4 客户端、5 轮、gen20：

```bash
RUN_ID=loopback_multiip_4c_5r_gen20 \
CLIENT_NUM=4 \
SAMPLE_CLIENT_NUM=4 \
TOTAL_ROUNDS=5 \
GEN_NUM=20 \
SERVER_PORT=51651 \
CLIENT_PORT_BASE=52651 \
CASE_TIMEOUT=7200 \
RUN_QPS=1 \
QPS_SUBSERVERS=4 \
QPS_CLIENT_GROUPS=4 \
QPS_CLIENTS_PER_GROUP=250 \
QPS_BASE_PORT=39651 \
bash scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
```

## 扩大到目标规模

60 客户端、100 轮、gen20：

```bash
RUN_ID=loopback_multiip_60c_100r_gen20 \
CLIENT_NUM=60 \
SAMPLE_CLIENT_NUM=60 \
TOTAL_ROUNDS=100 \
GEN_NUM=20 \
SERVER_PORT=51751 \
CLIENT_PORT_BASE=52751 \
CASE_TIMEOUT=28800 \
RUN_QPS=1 \
QPS_SUBSERVERS=8 \
QPS_CLIENT_GROUPS=8 \
QPS_CLIENTS_PER_GROUP=1250 \
QPS_BASE_PORT=39751 \
bash scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
```

输出目录按 `RUN_ID` 保存：

```text
exp/headonly_system/loopback_multiip_runs/<RUN_ID>/
  configs/
  logs/
  metrics/training_metrics_summary.json
  qps/download_qps_loopback_multiip/download_qps_summary.json
  system/run_info.log
```

## 当前平台限制

bjb1 容器缺少 `CAP_NET_ADMIN`，因此不能使用标准
`network namespace + veth + bridge + tc` 方案，也不能做受控 10Gbps/25Gbps
限速实验。当前可用的是 loopback 多 IP fallback。

公网业务端口也无法直接连接；此前 bjb1 监听 `0.0.0.0:39301` 后，从另一台
服务器连接 `connect.bjb1.seetacloud.com:39301` 返回 `ConnectionRefused`。

因此当前结论只能写为：

```text
已完成单机多 IP 真实 distributed FL 系统验证；
已完成参数下发 QPS 统计链路验证；
尚未完成真实物理多机网络 QPS 验证。
```
