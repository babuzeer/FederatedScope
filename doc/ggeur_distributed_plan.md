# 规划：GGEUR 在 Office-Home 上的分布式训练

## 目标
在不改变算法行为的前提下，使 GGEUR（CLIP 与 CNN 版本）能够在 FederatedScope 的分布式模式下运行。本文件仅给出实现规划，不包含代码修改。

## 当前分布式实现（本分支基线）
- 入口脚本：`scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh`
  - 启动 1 个 server + 3 个 client，记录 PID，支持 Ctrl+C 清理，可配合 stop 脚本。
- 配置模式：`scripts/distributed_scripts/distributed_configs/*_server.yaml` 与 `*_client_*.yaml`
  - `federate.mode: distributed`，`distribute.role` 按进程设置。
  - `distributed_data.enabled` 驱动 FEMNIST 的分片加载。
- 分布式数据加载器：`federatedscope/core/data/distributed_loader.py`
  - 使用分片目录 + `manifest.json`，由 `scripts/tools/prepare_femnist_shards.py` 生成。
- 可靠性特性（近期提交）：
  - 客户端心跳发送 + 服务端心跳检测。
  - 支持断线重连（指数退避）。
  - 支持训练暂停/恢复（`federate.min_clients_threshold`）及超时后正常退出。

## GGEUR 实现（来自 feature/GGEUR）
- 数据加载器：`federatedscope/contrib/data/ggeur_data.py`
  - Office-Home 域（Art、Clipart、Product、Real_World）。
  - 可选 LDS（Dirichlet 分布，`ggeur.use_lds`、`ggeur.lds_alpha`、`ggeur.lds_seed`）。
  - 在 standalone 模式下返回多客户端 data dict。
- Workers：
  - `GGEURClient`：特征抽取（CLIP/CNN）、本地统计量计算、特征增强、训练 MLP（可选 CNN 对齐/蒸馏）。
  - `GGEURServer`：聚合统计量、广播全局协方差/原型，然后对 MLP（及可选 CNN）做 FedAvg。
- 配置（Office-Home）：
  - CLIP 基线 vs GGEUR：`scripts/example_configs/fedavg_officehome_lds_baseline.yaml`，`scripts/example_configs/ggeur_officehome_lds.yaml`。
  - CNN 基线 vs GGEUR：`scripts/example_configs/fedavg_cnn_convnext.yaml`，`scripts/example_configs/ggeur_cnn_convnext.yaml`。
- 数据集位置：`/root/OfficeHomeDataset_10072016`。

## 分布式设计方案
### 1）Office-Home 分片与 manifest（新增）
目标：避免每个分布式客户端都加载全量数据再本地切分。
- 新增分片准备工具，参考 FEMNIST，例如 `scripts/tools/prepare_officehome_shards.py`。
  - 输入：数据根目录、seed、是否 LDS、alpha、每域客户端数、splits。
  - 输出：每客户端分片目录（或索引文件）+ `manifest.json`。
  - 每个 client 记录：train/val/test 图像路径与标签。
- 在 `federatedscope/core/data/distributed_loader.py`（或新模块）中增加 Office-Home 分布式加载器，使用 `distributed_data.dataset: office-home` 触发。
  - 若采用索引文件，提供轻量 Dataset，按需加载图像。
  - 与 FEMNIST 类似加入严格检查（manifest 路径、分片目录）。

### 2）GGEUR 数据加载器支持分布式
目标：保持 standalone 行为不变，同时允许分布式客户端只加载自己的分片。
- 修改 `federatedscope/contrib/data/ggeur_data.py`，检测分布式模式与 `distributed_data.enabled`。
  - server 角色：返回 `None` 或最小数据以保证 `get_model` 可用。
  - client 角色：依据 `distribute.client_id`（或 `distribute.data_idx`）加载分片。
- 若启用分片加载，跳过当前的多客户端内存切分逻辑。

### 3）特征缓存隔离
目标：避免多进程写缓存冲突。
- 确保缓存文件名包含 `client_id` 与特征提取器类型。
- 或使用按客户端区分的缓存目录（如 `feature_cache_dir/client_{id}/`）。

### 4）分布式配置
目标：使用现有分布式配置风格（server + per-client YAML）。
- 在 `scripts/distributed_scripts/distributed_configs/` 下新增：
  - `ggeur_officehome_server.yaml`
  - `ggeur_officehome_client_*.yaml`
  - `ggeur_cnn_officehome_server.yaml`
  - `ggeur_cnn_officehome_client_*.yaml`
- 关键字段：
  - `federate.mode: distributed`，`distribute.role` 按进程设置。
  - `distribute.client_id` 与 `distribute.data_idx` 用于客户端选择。
  - `distributed_data.enabled: True`，`distributed_data.dataset: office-home`，`distributed_data.manifest` 与 `distributed_data.shard_path`。
  - `ggeur.*` 从特征分支的配置复制（CLIP 或 CNN）。
- 本地运行客户端数建议：
  - 60 客户端（与原配置一致）对单机压力较大。
  - 提供小规模 dev 配置（如 4 或 8 客户端）用于验证分布式流程。

### 5）分布式启动脚本
目标：使用 FEMNIST/LR 示例的进程管理流程。
- 新增 `scripts/distributed_scripts/run_distributed_ggeur_officehome_managed.sh`。
  - 启动 server + N 个客户端。
  - 使用 PID 记录与清理逻辑。
  - 启动前检查分片/manifest。
- 如需，新增 stop 脚本（或使用现有 stop 脚本的匹配规则）。

### 6）服务端评估注意事项
- `GGEURServer` 在服务端进行测试特征抽取，需要 `data.root` 与 CLIP/CNN 权重路径有效。
- 如果服务端也写测试特征缓存，需与客户端缓存隔离命名。

## 工作拆分
1) 增加 Office-Home 分片准备工具 + manifest 格式。
2) 实现 Office-Home 分布式加载器，并集成到 `distributed_data` 配置。
3) 在 `ggeur_data.py` 中添加分布式路径（server/client 角色）。
4) 增加 CLIP/CNN 的分布式配置文件。
5) 增加 managed 启动脚本与检查逻辑。
6) 小规模基础测试，验证统计量聚合、增强完成同步与训练轮次推进。

## 风险与待确认项
- 客户端数量：确认分布式运行是否要 60 客户端，还是先做小规模验证。
- CLIP 权重：确认分布式客户端/服务端的本地权重路径（`ggeur.clip_model_path`）。
- 缓存冲突：确认特征缓存文件命名是否包含 client 信息，必要时调整。
- LDS 复现：确保 manifest 记录 seed/alpha/splits 以保证可复现。

## 验证计划
- 最小可运行：1 server + 2-4 client，分别测试 LDS 开/关。
- 核心日志检查：
  - 客户端上报统计量，服务端广播全局协方差。
  - 客户端发送 augmentation ready，服务端开始第 1 轮。
  - 模型更新到达并触发 FedAvg。
- 异常处理：停止一个 client，验证心跳/离线检测与训练暂停/恢复逻辑。
