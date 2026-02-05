# GGEUR分布式Worker架构与启动流程周报

## 一、工作概述

本周完成了GGEUR方法的分布式Worker架构集成与特殊启动流程实现。主要工作包括：将GGEUR的自定义Client/Server注册到FederatedScope框架、实现Round 0统计收集的特殊启动流程、修复Message系统兼容性问题，以及在配置文件中完成GGEUR聚合器的注册。该工作为GGEUR方法在分布式环境下的正确运行奠定了基础。

## 二、功能模块

### 2.1 GGEUR分布式Worker架构

**核心组件**：

| 组件 | 文件位置 | 功能 |
|------|----------|------|
| GGEURServer | `federatedscope/contrib/worker/ggeur_server.py` | 统计聚合、协方差矩阵计算、MLP聚合 |
| GGEURClient | `federatedscope/contrib/worker/ggeur_client.py` | 特征提取、本地统计计算、增强训练 |

**GGEURServer核心功能**：
1. 收集各客户端的本地统计量（means, covariances, counts）
2. 使用平行轴定理聚合协方差矩阵
3. 广播全局协方差矩阵给所有客户端
4. 执行标准FedAvg聚合（MLP参数）
5. 在所有domain测试集上评估模型

**GGEURClient核心功能**：
1. CLIP特征提取
2. 本地统计量计算（每类均值、协方差）
3. 接收全局协方差后进行特征增强
4. 在增强特征上训练MLP分类器

### 2.2 特殊启动流程

GGEUR与标准FedAvg的关键区别在于启动流程：

| 阶段 | 标准FedAvg | GGEUR |
|------|------------|-------|
| Round 0 | 广播初始模型参数 | 发送信号，客户端收集统计量 |
| Round 0结束 | 客户端返回更新后参数 | 客户端返回本地统计量 |
| Round 1 | 聚合参数并广播 | 聚合统计量，创建MLP，开始训练 |

### 2.3 配置文件结构

**服务端配置关键项**：
```yaml
federate:
  method: 'ggeur'        # 启用GGEUR自定义Worker
  client_num: 4
  mode: 'distributed'
  total_round_num: 100

distribute:
  use: True
  server_host: '127.0.0.1'
  server_port: 50051
  role: 'server'
  heartbeat_timeout: 600  # 10分钟（适应特征提取时间）
```

---

## 三、技术实现

### 3.1 GGEUR聚合器注册

**文件位置**：`federatedscope/core/configs/constants.py`

```python
AGGREGATOR_TYPE = {
    "fedavg": "clients_avg",
    "local": "no_communication",
    "global": "server_clients_interpolation",
    ...
    "ggeur": "clients_avg"  # GGEUR_Clip (自定义聚合逻辑在GGEURServer中)
}
```

**设计说明**：
- 虽然注册为`clients_avg`，但实际聚合逻辑在`GGEURServer._perform_fedavg()`中自定义实现
- 注册的目的是避免框架抛出"aggregator not implemented"警告

### 3.2 启动流程实现

**文件位置**：`federatedscope/contrib/worker/ggeur_server.py`（第139-168行）

```python
def broadcast_model_para(self,
                         msg_type='model_para',
                         sample_client_num=-1,
                         filter_unseen_clients=True):
    """
    Override broadcast_model_para to handle GGEUR's special startup flow.

    In GGEUR, the server doesn't broadcast initial model parameters at
    Round 0. Instead, clients first collect local statistics, and the
    server creates the global MLP only after receiving all statistics.
    """
    if self.state == 0:
        # Round 0: Send a signal to clients to start collecting statistics
        # Send a non-empty dict with a signal (Message system doesn't
        # support None or empty dict)
        logger.info(
            "Server: Round 0 - sending signal to collect statistics")
        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content={'signal': 'collect_statistics'}))
    else:
        # For subsequent rounds, use the custom training round method
        self._start_training_round()
```

**关键设计点**：

1. **Round 0特殊处理**：不广播模型参数，而是发送收集统计量的信号
2. **Message内容**：使用`{'signal': 'collect_statistics'}`而非空字典或None
3. **后续轮次**：调用自定义的`_start_training_round()`方法

### 3.3 Message系统兼容性

**问题背景**：
FederatedScope的Message系统对content参数有限制：
- `None`类型会导致`ValueError`
- 空字典`{}`会导致`IndexError: list index out of range`

**解决方案**：
```python
# 错误写法
content=None      # ValueError
content={}        # IndexError

# 正确写法
content={'signal': 'collect_statistics'}  # 非空字典，通过验证
```

### 3.4 消息处理器注册

**文件位置**：`federatedscope/contrib/worker/ggeur_server.py`

```python
def _register_default_handlers(self):
    """Register message handlers"""
    super()._register_default_handlers()

    # Register handler for local statistics
    self.register_handlers('local_statistics',
                           self.callback_for_local_statistics)

    # Register handler for augmentation ready signal
    self.register_handlers('augmentation_ready',
                           self.callback_for_augmentation_ready)
```

**消息类型**：
| 消息类型 | 方向 | 用途 |
|----------|------|------|
| `model_para` | Server → Client | Round 0发送收集信号，后续轮次发送MLP参数 |
| `local_statistics` | Client → Server | 客户端返回本地统计量（特征均值、协方差） |
| `augmentation_ready` | Client → Server | 客户端完成增强训练后的信号 |

### 3.5 GGEURServer状态管理

```python
class GGEURServer(Server):
    def __init__(self, ...):
        super().__init__(...)
        
        # Statistics collection buffers
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # Aggregated covariance matrices
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}

        # All client prototypes for cross-client augmentation
        self.all_prototypes = {}  # {client_id: {class_idx: prototype}}

        # Global prototypes (aggregated means) for feature alignment
        self.global_prototypes = {}  # {class_idx: mean_vector}

        # MLP model for aggregation
        self.global_mlp = None
```

---

## 四、测试与验证

### 4.1 验证配置

**测试环境**：
- 4客户端分布式训练（对应Office-Home 4个domain）
- 本地CLIP模型：`/root/open_clip_vitb16.bin`（ViT-B/16）
- 心跳超时：600秒（10分钟）

**启动命令**：
```bash
# 启动Server
python -m federatedscope.main --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_server.yaml

# 启动Clients（4个终端）
python -m federatedscope.main --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_1.yaml
python -m federatedscope.main --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_2.yaml
python -m federatedscope.main --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_3.yaml
python -m federatedscope.main --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_4.yaml
```

### 4.2 验证结果

**Round 0启动流程**：
```
[Server] Round 0 - sending signal to collect statistics
[Client 1] Received signal, starting feature extraction...
[Client 2] Received signal, starting feature extraction...
[Client 3] Received signal, starting feature extraction...
[Client 4] Received signal, starting feature extraction...
```

**统计量收集**：
```
[Client 1] Extracted 2072 features, computed statistics for 65 classes
[Client 2] Extracted 4355 features, computed statistics for 65 classes
[Client 3] Extracted 4146 features, computed statistics for 65 classes
[Client 4] Extracted 4218 features, computed statistics for 65 classes
[Server] Received statistics from 4/4 clients
[Server] Aggregating covariance matrices using parallel axis theorem...
[Server] Built global MLP classifier with 65 classes
```

**问题修复验证**：
| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| Message NoneType | ValueError | ✓ 通过 |
| Empty dict content | IndexError | ✓ 通过 |
| Aggregator warning | "not implemented" | ✓ 无警告 |
| Worker类型 | 使用标准Client/Server | ✓ 使用GGEURClient/Server |

---

## 五、后续工作

基于已完成的Worker架构和启动流程，下一步将进行：

### 5.1 gRPC序列化稳定性
- [ ] 客户端numpy array → list显式转换
- [ ] 服务端list → numpy array转换与形状验证
- [ ] 解决协方差聚合时的形状不匹配问题

### 5.2 配置属性兼容性
- [ ] 统一使用`out_channels`替代`num_classes`
- [ ] 对齐FederatedScope标准模型配置规范

### 5.3 完整训练验证
- [ ] 100轮分布式训练测试
- [ ] 与standalone模式结果对比
- [ ] 性能和通信开销分析

---

## 六、总结

**已完成**：
1. GGEUR自定义Worker（GGEURServer/GGEURClient）架构集成
2. Round 0特殊启动流程实现
3. Message系统兼容性修复
4. GGEUR聚合器注册
5. 5个分布式配置文件完善

**关键成果**：
- GGEUR分布式训练可以正确启动并完成Round 0统计收集阶段
- 解决了FederatedScope框架与GGEUR特殊流程的兼容性问题
- 为后续gRPC通信和完整训练验证奠定基础
