# HeadOnly/GGEUR 分布式训练功能现状与缺口清单

本文整理当前 HeadOnly/GGEUR 真实分布式训练相关功能的实现现状、已覆盖脚本、尚未验证内容和后续缺口。目标是为“先把基本分布式训练跑通，再扩展 QPS 和大规模能力”提供实施依据。

本文只讨论 FederatedScope/GGEUR 的真实分布式训练链路。QPS benchmark 的 Dispatch-Start QPS 验收口径见：

```text
docs/HeadOnly_分层下发QPS验收口径与实施目标.md
```

## 1. 当前实现范围总览

当前代码已经覆盖以下核心链路：

```text
server/client gRPC join
join 超时与 late join 拒绝
GGEUR round0 统计上传
server 聚合协方差与原型
server 下发全局协方差
client 本地生成增强样本
client 训练 MLP head
server FedAvg/FedOpt 聚合 MLP
client-side evaluation
OfficeHome per-client manifest
故障注入和 quorum 场景验证
多机配置生成与启动脚本
```

但当前仍处于“功能已落地，真实双机/多机场景尚需系统验证”的状态。

## 2. 已实现的代码能力

### 2.1 基础 gRPC 分布式生命周期

涉及文件：

```text
federatedscope/core/workers/server.py
federatedscope/core/workers/client.py
federatedscope/core/communication.py
```

已具备能力：

```text
client 启动 gRPC listener
client 向 server 发送 join_in
server 为 client 分配 ID
server 记录 client 上报的 client_host/client_port
server 后续通过 gRPC 回连 client
```

注意：

```text
FederatedScope distributed 需要双向可达。
client -> server 用于 join 和上传；
server -> client 用于分配 ID、下发参数、下发协方差、finish 等。
```

### 2.2 join 超时与 late join 防护

涉及文件：

```text
federatedscope/core/workers/server.py
federatedscope/core/configs/cfg_fl_setting.py
```

已实现配置：

```yaml
distribute:
  join_timeout_seconds: 60
```

已具备能力：

```text
server 从第一个 join 请求开始计时。
如果 client_num 个客户端未按时加入，server 失败退出。
训练开始后拒绝 late join。
达到 client_num 后拒绝额外 client join。
重复 join 会被忽略。
```

仍需验证：

```text
真实双机下 join timeout 日志和退出行为是否稳定。
late/extra client 在公网环境下是否能收到 finish/reject。
```

### 2.3 GGEUR 分布式阶段 timeout

涉及文件：

```text
federatedscope/contrib/worker/ggeur_server.py
federatedscope/core/configs/cfg_ggeur.py
```

已实现配置：

```yaml
ggeur:
  distributed_stage_timeout: 1800
```

已具备能力：

```text
statistics 阶段等待超时
augmentation_ready 阶段等待超时
train round 等待超时
client_eval 阶段等待超时
超时后通知已加入 client finish，并关闭 server
```

仍需验证：

```text
真实网络抖动下 timeout 是否设置合理。
超时后 client 进程是否都能退出。
```

### 2.4 GGEUR quorum 与故障容忍验证入口

涉及文件：

```text
federatedscope/contrib/worker/ggeur_server.py
federatedscope/core/configs/cfg_ggeur.py
```

已实现配置：

```yaml
ggeur:
  min_statistics_clients: 0
  min_augmentation_clients: 0
  min_train_updates: 0
```

语义：

```text
0 表示严格等待全部配置 client，保持正式 FL 语义。
大于 0 表示指定阶段只等待最小 quorum，主要用于故障场景验证。
```

已具备能力：

```text
statistics quorum 达成后固定 active_client_ids
augmentation quorum 达成后固定 training_client_ids
后续协方差下发、训练下发、finish 只面向活跃训练客户端
忽略 inactive client 的 late statistics / augmentation_ready / model_update
忽略 stale/future round update
忽略重复 model update
```

注意：

```text
quorum 不改变 join 阶段语义。
server 仍要求 federate.client_num 个 client 完成 join 后才进入 GGEUR 阶段。
少启动 client 会触发 join_timeout，而不是进入 quorum 训练。
```

仍需验证：

```text
真实双机/多机下 client 中途退出后的 quorum 训练是否稳定。
quorum 场景下最终 finish 是否覆盖所有仍活跃进程。
```

### 2.5 客户端故障注入

涉及文件：

```text
federatedscope/contrib/worker/ggeur_client.py
federatedscope/core/configs/cfg_ggeur.py
```

已实现配置：

```yaml
ggeur:
  fail_after_stage: ''
  fail_on_round: -1
```

支持阶段：

```text
after_statistics_upload
after_augmentation_ready
before_train_round
```

用途：

```text
主动让指定 client 在指定阶段退出，用于验证真实分布式故障容忍。
```

仍需验证：

```text
公网双机下故障注入退出是否会造成残留进程。
server 日志是否能清楚体现 quorum 行为。
```

### 2.6 client-side evaluation

涉及文件：

```text
federatedscope/contrib/worker/ggeur_server.py
federatedscope/contrib/worker/ggeur_client.py
federatedscope/core/configs/cfg_ggeur.py
```

已实现配置：

```yaml
ggeur:
  headonly_eval_mode: 'server'  # server | client
```

client 模式语义：

```text
server 聚合 MLP 后，将聚合后的 MLP 下发给训练客户端；
client 在本地 test split 上评估；
client 上传 client_eval_metrics；
server 按样本数加权汇总 accuracy/loss。
```

意义：

```text
更贴近真实联邦学习。
server 不需要持有所有客户端测试数据。
适合双机/多机物理分布式验证。
```

仍需验证：

```text
client-side eval 和 HeadOnly feature cache 的组合是否稳定。
client_eval 阶段 timeout 是否需要单独配置更长。
多 client eval metrics 是否按预期聚合。
```

### 2.7 OfficeHome per-client manifest

涉及文件：

```text
federatedscope/contrib/data/ggeur_data.py
scripts/prepare_officehome_client_manifests.py
```

已实现配置：

```yaml
ggeur:
  officehome_manifest_path: ''
  officehome_domains: []
```

已具备能力：

```text
每个 client 可通过 manifest 加载自己的 train/val/test 图片列表。
避免运行时重复随机切分。
支持 client 数据目录独立部署。
支持 OfficeHome domain filter。
```

manifest 准备脚本支持：

```text
symlink
copy
hardlink
manifest-only
LDS 切分
指定 domains
```

仍需验证：

```text
真实双机下 manifest 路径和图片相对路径是否稳定。
Windows/WSL/Linux 混合路径不应进入训练配置。
OfficeHome Real World 目录名含空格时 manifest 读取是否稳定。
```

## 3. 已有脚本能力

### 3.1 loopback 多 IP 全流程验证

脚本：

```text
scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_full_validation.sh
scripts/distributed_scripts/ggeur_headonly_system_officehome_vit/run_loopback_multiip_scenarios.sh
```

覆盖能力：

```text
单机多进程
loopback 多 IP
正常全部 client
延迟 join
少 client join timeout
额外 late client 拒绝
client 训练前退出 + quorum
可选 QPS benchmark
```

限制：

```text
不是物理双机。
不覆盖公网 IPv6 和真实跨主机网络。
```

### 3.2 双机/多机配置生成和启动

脚本目录：

```text
scripts/distributed_scripts/ggeur_multimachine/
```

主要脚本：

```text
generate_configs.py
launch_server.sh
launch_clients.sh
run_remote_case.sh
stop_case.sh
wait_case.py
summarize_case.py
```

已支持：

```text
officehome/domainnet
vit/cnn/mixer
fedavg/fedprox/fedopt/moon/fedproto/promptfl
head-only 模式
server/client 配置生成
多 client 配置生成
headonly_eval_mode client/server
--no-use-gpu
```

限制：

```text
要求 server/client 机器代码目录一致或手动同步 case。
当前仍是“一台 server 机器 + 一台 client generator 机器模拟多个 client”的部署面。
没有真正拆出长期运行的 coordinator/subserver 参数服务。
```

### 3.3 QPS benchmark

脚本：

```text
scripts/benchmark_headonly_mlp_download_window.py
scripts/distributed_scripts/ggeur_headonly_download_qps/
```

当前用途：

```text
验证参数下发 QPS 口径和公网双机通信能力。
```

注意：

```text
QPS benchmark 是 TCP/asyncio，不是 FederatedScope gRPC 分布式训练。
它不依赖数据集、模型权重和 torch。
```

第一阶段后续要按新口径扩展为：

```text
ack_mode = header | first-chunk | full
server-mp/client-mp 多进程
Dispatch-Start QPS 主指标
```

## 4. 已经做过的本地验证

当前已完成：

```text
核心 Python 文件 py_compile 通过
QPS 本地 local 小样例通过
QPS summary 脚本可读结果并选 best_successful_run
公网 IPv6 双机 QPS smoke 已成功
公网 IPv6 双机 QPS 1x50 / 2x100 已成功
固定 subserver 压力下，单进程 QPS 未体现横向扩展
```

已观察到：

```text
公网 IPv6 full-payload download QPS 大约在几十到 80+ QPS 量级。
增加 subserver 数但仍使用单进程 benchmark 时，总 QPS 没有明显增长。
```

当前判断：

```text
需要使用多进程 subserver benchmark 和 Dispatch-Start QPS 口径继续验证架构扩展性。
```

## 5. 分布式训练尚未充分验证的内容

以下属于“代码已有入口，但真实双机/多机尚未系统验证”：

```text
1. 真实双机 gRPC HeadOnly 2 clients / 1 round smoke test。
2. gRPC IPv6 地址配置和双向回连稳定性。
3. client-side evaluation 的真实双机闭环。
4. OfficeHome manifest 在双机路径下的加载。
5. round0 feature extraction/cache/generation 在小显存本地 GPU 上的稳定性。
6. server 聚合 MLP 后再次下发训练轮次的稳定性。
7. finish 消息是否能让所有 client 进程退出。
8. stop_case 对异常残留进程的清理能力。
9. fault injection + quorum 在真实双机/多机下的表现。
10. 多方法矩阵在真实分布式下的兼容性。
```

## 6. 当前没有涉及或不完整的需求

### 6.1 真正长期运行的 coordinator/subserver 服务

当前 GGEUR 分布式训练仍是 FederatedScope 原生：

```text
server <-> clients
```

尚未真正拆成：

```text
coordinator -> persistent subservers -> clients
```

QPS benchmark 里有 subserver 概念，但它不是 FederatedScope 训练服务的一部分。

缺口：

```text
真实训练中的 subserver 参数服务层尚未落地。
coordinator 和 subserver 之间的协议尚未定义。
subserver 聚合/缓存/转发模型参数的生命周期尚未实现。
```

### 6.2 客户端上传聚合的分层化

当前训练上传仍是 client 直接向 server 上传模型更新。

尚未实现：

```text
client -> subserver 上传
subserver 局部聚合
subserver -> coordinator 上传局部结果
coordinator 全局聚合
```

这会影响后续 Upload-Start QPS 和分层上传能力验收。

### 6.3 大规模客户端仿真

当前脚本可模拟多个 client 进程，但还没有系统支持：

```text
万级 client 进程/连接调度
多 client generator 分布式压测
跨多台机器分配 client groups
统一收集大规模 client 日志和指标
```

### 6.4 安全与认证

当前 gRPC 通信使用 insecure channel。

尚未涉及：

```text
TLS
client 身份认证
token/证书
公网部署安全策略
消息签名或重放防护
```

### 6.5 断点恢复与任务恢复

当前已有 timeout 和 finish 清理，但没有完整恢复机制。

尚未实现：

```text
server 重启恢复
client 断线重连
round 状态持久化
cache/version 对齐检查
失败 client 重新加入
```

### 6.6 观测与运维

当前主要靠日志和 summary JSON。

尚未实现：

```text
统一 metrics exporter
Prometheus/Grafana
进程健康检查
网络连接统计
每阶段耗时仪表盘
自动诊断报告
```

## 7. 建议的后续实施优先级

### P0：先跑通真实双机最小训练闭环

目标：

```text
OfficeHome + ViT HeadOnly + FedAvg
2 clients
1 round
headonly_eval_mode=client
batch_size/extract_batch_size 调小
```

验收：

```text
2 clients join 成功
round0 statistics 上传成功
global covariance 下发成功
augmentation_ready 成功
round1 MLP train/update 成功
client_eval_metrics 成功
Training finished
所有进程退出
```

### P1：manifest 化真实数据切分

目标：

```text
为每个 client 准备独立 OfficeHome manifest。
双机 client 只加载自己的本地 manifest。
```

验收：

```text
client 日志显示 manifest loaded
train/test 样本数符合 manifest
无运行时随机重切分
```

### P2：真实双机故障场景

目标：

```text
验证 join timeout
验证 late/extra client reject
验证 client 中途退出 + quorum
```

验收：

```text
server 日志明确记录阶段、活跃 client、quorum、忽略 late update。
失败场景按预期退出。
成功场景按 quorum 完成训练。
```

### P3：多方法/多模型/多数据集矩阵

目标：

```text
OfficeHome/DomainNet
ViT/CNN/Mixer
FedAvg/FedProx/FedOpt/FedProto/MOON/PromptFL
```

验收：

```text
每个 case 有独立 config、日志、summary。
失败 case 可重跑。
```

### P4：分层 subserver 训练架构

目标：

```text
把 QPS benchmark 中的 subserver 概念迁移到真实训练链路。
```

需要设计：

```text
coordinator-subserver 协议
subserver-client 协议
局部聚合
模型版本管理
上传/下发缓存
故障处理
指标采集
```

## 8. 建议的下一步执行命令

真实双机 smoke test 生成配置建议：

```bash
cd /root/autodl-tmp/FederatedScope

python3 scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id two_machine_grpc_smoke \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --head-only \
  --headonly-eval-mode client \
  --server-host '[服务器IPv6]' \
  --server-bind-host '[::]' \
  --client-host '[本地机器IPv6]' \
  --server-port 51251 \
  --client-port-base 52250 \
  --clients 2 \
  --sample-clients 2 \
  --rounds 1 \
  --batch-size 8 \
  --extract-batch-size 8 \
  --num-generated-per-sample 1 \
  --num-generated-per-prototype 1 \
  --target-size-per-class 1 \
  --feature-cache-version two_machine_grpc_smoke_v1
```

server 启动：

```bash
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly
```

client 启动：

```bash
PYTHON_BIN=python3 \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly
```

成功日志关键字：

```bash
grep -E "Client #[0-9]+ has joined|Received statistics quorum|Broadcasting global covariances|augmentation ready|Received model|client-side eval|Training finished" \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

## 9. 当前结论

当前项目已经具备真实分布式训练的基本代码入口和验证脚本基础：

```text
join 生命周期
stage timeout
HeadOnly/GGEUR round0 -> train -> eval 链路
client-side evaluation
OfficeHome manifest
故障注入与 quorum
双机配置生成和启动脚本
```

但还需要按 P0-P4 顺序完成真实双机/多机验证和后续分层 subserver 训练架构落地。

第一优先级不是继续扩展方法矩阵，而是先让：

```text
真实双机 2 clients / 1 round HeadOnly smoke test
```

稳定成功，并形成可重复执行的日志和 summary。
