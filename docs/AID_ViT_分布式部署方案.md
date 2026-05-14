# AID ViT 分布式部署方案

本文档用于重新整理 `feature/lzy-ViT-AID` 分支的分布式部署设计，并将其落成一套适合 AID + ViT 任务的可执行方案。

## 1. 方案基线

本方案沿用 `feature/lzy-ViT-AID` 的分布式实现思路，而不是单纯使用 FederatedScope 原始的示例脚本。核心基线如下：

- 通信模式使用 FederatedScope 的 `distributed` 模式，底层为 gRPC。
- 启动顺序采用“先服务端，后客户端，客户端顺序拉起”的方式。
- 服务端负责客户端接入、分配 client id，并在所有客户端接入后再触发训练。
- 客户端在收到 `assign_client_id` 前，先缓存非关键消息，避免乱序消息导致状态异常。
- 运维层面使用 PID 文件管理启动的所有进程，支持统一停止和异常清理。
- 日志目录创建兼容冻结配置，避免配置 freeze 状态下创建输出目录失败。

这套设计比仓库自带的 `run_distributed_lr.sh` 更适合真实训练场景，重点解决了三类问题：

- 进程不好停，容易残留。
- 客户端启动有先后差异时，容易出现接入和消息处理不稳定。
- 配置、日志、超时控制在长任务场景下不够稳。

## 2. 分支中已经落地的关键设计

`feature/lzy-ViT-AID` 分支对分布式相关逻辑做了三处关键增强。

### 2.1 服务端接入控制

服务端在 `federatedscope/core/workers/server.py` 中增加了以下行为：

- 记录首个客户端接入时间 `first_join_timestamp`。
- 设置统一接入超时 `join_timeout_seconds = 300`。
- 对匿名接入客户端动态分配 client id。
- 在所有客户端完成 join 之前，不启动正式训练。

这意味着部署时必须保证：

- 服务端先启动。
- 客户端数量与 `federate.client_num` 一致。
- 客户端应在 5 分钟内完成接入，否则服务端会直接报超时错误。

### 2.2 客户端消息缓冲

客户端在 `federatedscope/core/workers/client.py` 中增加了“ID 分配前消息缓存”逻辑：

- 初始 `ID = -1`。
- 若先收到 `assign_client_id`，立即完成身份绑定。
- 在 ID 未分配前收到的其他消息先进入 `pending_messages`。
- ID 分配完成后再按状态顺序处理缓存消息。

这个改动解决的是多机环境中常见的消息先后不稳定问题，尤其适合：

- 服务器已启动且开始收发消息。
- 客户端存在启动延迟。
- 网络存在轻微抖动。

### 2.3 受控启动与停止

分支新增：

- `scripts/distributed_scripts/run_distributed_lr_managed.sh`
- `scripts/distributed_scripts/stop_distributed.sh`

这两个脚本体现的不是 LR 任务本身，而是推荐的运维模式：

- 启动时记录全部 PID。
- 退出时根据 PID 文件逐个清理。
- 再执行一次基于进程特征的兜底清理。
- 支持 `Ctrl+C` 直接停止整套分布式任务。

对于 AID + ViT，建议沿用这套“受控启动”方式，只替换配置文件和启动命令。

## 3. 推荐部署拓扑

推荐采用 1 个 server + N 个 client 的标准拓扑。

### 3.1 角色划分

- Server 节点：负责聚合、调度、全局评估。
- Client 节点：各自持有本地数据，执行本地训练。

### 3.2 数据组织

AID + ViT 不建议继续使用 `distribute.data_idx` 的 toy 示例方式做真实实验。更稳妥的是：

- 每个 client 节点本地放置自己的 AID 子集。
- 服务端仅保留全局验证/测试集，或者不持有训练数据。
- 每个节点使用自己的 yaml，显式指定本地数据路径或本地数据划分配置。

如果只是单机联调，可先保留统一数据根目录，再通过不同配置控制不同 client 的数据划分。

### 3.3 网络要求

- 服务端端口对所有客户端可达。
- 每个客户端监听端口唯一。
- 关闭 HTTP 代理或显式配置 `distribute.grpc_enable_http_proxy: False`。
- ViT 模型参数较大时，显式放宽 gRPC 消息大小限制。

## 4. AID_ViT 配置改造原则

`feature/lzy-ViT-AID` 中的 `scripts/example_configs/AID_ViT.yaml` 和 `AID_ViT_fedprox.yaml` 当前是 `standalone` 模式。迁移到分布式时，建议拆成：

- 1 份公共基础配置：保存模型、训练轮次、优化器、ViT 参数。
- 1 份服务端配置：覆盖 `distribute.role=server`。
- N 份客户端配置：分别覆盖 `distribute.role=client`、本地地址、数据配置。

### 4.1 公共基础配置示例

```yaml
use_gpu: True
device: 0
seed: 12345

early_stop:
  patience: 5

federate:
  client_num: 20
  mode: distributed
  total_round_num: 300
  sample_client_rate: 0.2
  make_global_eval: False

distribute:
  use: True
  grpc_max_send_message_length: 314572800
  grpc_max_receive_message_length: 314572800
  grpc_enable_http_proxy: False

data:
  type: AID
  splitter: iid

dataloader:
  batch_size: 64

model:
  type: ViT
  out_channels: 30

train:
  local_update_steps: 1
  batch_or_epoch: epoch
  optimizer:
    type: Adam
    lr: 2e-4
    weight_decay: 0.01

grad:
  grad_clip: 5.0

criterion:
  type: CrossEntropyLoss

trainer:
  type: cvtrainer

eval:
  freq: 10
  metrics: ['acc', 'correct']
  count_flops: False
```

### 4.2 服务端配置示例

```yaml
federate:
  make_global_eval: True

distribute:
  role: server
  server_host: '10.0.0.1'
  server_port: 50051
```

如果服务端不保留测试集，则改为：

```yaml
federate:
  make_global_eval: False
```

### 4.3 客户端配置示例

```yaml
distribute:
  role: client
  server_host: '10.0.0.1'
  server_port: 50051
  client_host: '10.0.0.2'
  client_port: 50052
```

不同客户端只需要改三项：

- `client_host`
- `client_port`
- 本地数据配置

### 4.4 FedProx 配置

如果沿用 `AID_ViT_fedprox.yaml`，在公共配置上保留：

```yaml
fedprox:
  use: True
  mu: 0.5
```

## 5. 建议的目录与配置组织

建议新增一组 AID 专用分布式配置，例如：

```text
scripts/distributed_scripts/distributed_configs/
  aid_vit_base.yaml
  aid_vit_server.yaml
  aid_vit_client_01.yaml
  aid_vit_client_02.yaml
  ...
  aid_vit_client_20.yaml
```

这样有三个好处：

- 与原始 `AID_ViT*.yaml` 的 standalone 配置解耦。
- 方便不同机器只同步自身需要的配置。
- 更适合后续写批量启动脚本。

## 6. 推荐启动流程

### 6.1 单机联调

先用单机多进程验证流程，再切多机。

启动顺序：

1. 启动 server。
2. 等待 3 到 5 秒。
3. 按顺序启动 client。
4. 每个 client 间隔 1 到 2 秒。

这是 `feature/lzy-ViT-AID` 管理脚本采用的顺序，建议保留。

### 6.2 多机正式部署

#### 服务端

```bash
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/aid_vit_server.yaml
```

#### 客户端

```bash
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/aid_vit_client_01.yaml
```

其他客户端同理，各自使用自己的 yaml。

## 7. 推荐运维脚本设计

建议不要直接复用 `run_distributed_lr.sh`，而是按 `run_distributed_lr_managed.sh` 的方式改一版 AID 专用脚本。

推荐保留以下能力：

- PID 文件记录。
- `trap` 捕获退出信号。
- 服务端先起，客户端延迟启动。
- 停止脚本单独提供。
- 统一清理残留进程。

推荐脚本命名：

- `scripts/distributed_scripts/run_distributed_aid_vit_managed.sh`
- `scripts/distributed_scripts/stop_distributed_aid_vit.sh`

## 8. 关键参数建议

### 8.1 gRPC 消息大小

ViT 参数量明显大于 toy LR，建议显式配置：

```yaml
distribute:
  grpc_max_send_message_length: 314572800
  grpc_max_receive_message_length: 314572800
```

即 300MB。如果后续启用更大的 backbone 或附加原型、统计量，再继续上调。

### 8.2 接入超时

当前分支将客户端 join 超时写死为 300 秒。部署上要么满足这个约束，要么后续单独改代码把它配置化。

在不改代码的前提下，建议：

- 提前准备好所有客户端环境。
- 确保客户端在 5 分钟内全部启动。
- 不要在服务端启动后长时间等待再逐个手动起客户端。

### 8.3 全局评估

若服务端有测试集：

- `federate.make_global_eval: True`

若评估放在客户端：

- `federate.make_global_eval: False`

对于多机 AID 实验，若测试口径要求统一，优先建议服务端持有固定测试集。

## 9. 部署风险与对应处理

### 9.1 服务端一直等待客户端

检查：

- `federate.client_num` 是否与实际客户端数量一致。
- 客户端是否都能连到 `server_host:server_port`。
- 客户端配置中的 `role` 是否正确。

### 9.2 客户端端口冲突

每个客户端必须使用不同 `client_port`。单机联调时尤其容易冲突。

### 9.3 大模型通信失败

优先检查：

- `grpc_max_send_message_length`
- `grpc_max_receive_message_length`
- 网络是否存在代理或中间层限制

### 9.4 进程残留

如果直接后台启动多个 `python federatedscope/main.py`，很容易残留。应优先采用带 PID 文件的 managed 脚本。

## 10. 最终建议

如果你的目标是“沿用 `feature/lzy-ViT-AID` 的分布式设计”继续推进 AID 实验，建议按下面的顺序执行：

1. 保留当前分支的 server/client 容错逻辑，不回退到原始示例脚本。
2. 将 `AID_ViT.yaml` 拆成 base/server/client 多份分布式配置。
3. 新增 AID 专用 managed 启停脚本，复用 PID 管理模式。
4. 先单机多进程联调，再切多机部署。
5. 对 ViT 显式放宽 gRPC 消息大小，并提前规划服务端测试集口径。

如果后续要继续落地，下一步最直接的工作不是再改框架，而是补齐：

- AID 分布式 server/client 配置文件
- AID 专用 managed 启动脚本
- 一份面向实际服务器 IP 的部署清单
