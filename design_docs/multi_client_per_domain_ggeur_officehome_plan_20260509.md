# Office-Home 多域多客户端训练四周规划

## 1. 背景与当前实现确认

### 1.1 当前结论
- 你当前的理解是正确的：**现有实现确实是 Office-Home 的 4 个域各分配 1 个客户端**。
- 当前 4 个客户端与 4 个域是一一对应关系：
  - Client 1 -> Art
  - Client 2 -> Clipart
  - Client 3 -> Product
  - Client 4 -> Real World

### 1.2 证据
- `scripts/tools/prepare_officehome_shards.py` 中固定定义了 4 个域，并在 `clients % 4 == 0` 的前提下按 `clients_per_domain = clients // 4` 切分。
- 当 `clients=4` 时，每个域只会生成 1 个 client shard。
- 当前运行脚本 `scripts/distributed_scripts/run_distributed_ggeur_officehome.sh` 只启动 4 个 client 配置文件，因此运行时拓扑也固定为 1 域 1 client。

## 2. 目标定义

### 2.1 总体目标
将当前“每域 1 client”的 Office-Home 分布式 GGEUR 训练，扩展为“**每域多个 client**”的训练模式，并同时满足两层异构：

- **跨域异构**：每个 client 只持有其所属域的数据，不同域之间天然异构。
- **域内异构**：同一个域下的多个 client，不是均匀切分数据，而是通过 **Dirichlet 分布** 制造标签分布偏斜。

### 2.2 目标示例
以 `client_num=12` 为例：
- Art 域下 3 个 client
- Clipart 域下 3 个 client
- Product 域下 3 个 client
- Real World 域下 3 个 client

每个域内部 3 个 client 的训练集由 Dirichlet 切分，因此：
- 同域 client 共享同一视觉域分布
- 但类分布不同，形成域内非 IID

## 3. 设计原则

- **KISS**：优先复用现有 shard 生成、distributed loader、GGEUR 统计上传与训练逻辑，不重写主流程。
- **YAGNI**：第一阶段只支持“按域平均分配 client 数”，不引入任意域-客户端映射 DSL。
- **DRY**：统一由 shard manifest 驱动 client 数据定位、domain 元信息和测试集重建，避免脚本、配置、训练代码重复维护域映射。
- **SOLID**：
  - shard 生成负责“数据分配”
  - YAML/脚本负责“进程拓扑”
  - distributed loader 负责“本地 shard 读取”
  - GGEUR worker 负责“统计、增强、训练、评估”

## 4. 需要达成的最终能力

### 4.1 数据准备能力
- 支持 `clients=4k`，其中 `k >= 1`
- 每个域生成 `k` 个 client shard
- 每个 shard 明确记录：
  - `client_id`
  - `domain`
  - `shard_path`
  - `num_samples`
  - `label_hist`

### 4.2 训练启动能力
- 自动生成或维护 `N` 个 client YAML，而不是手工写死 4 份
- 启动脚本支持按 `client_num` 动态拉起所有 client
- server 仍只启动 1 个

### 4.3 训练逻辑兼容能力
- GGEUR 的统计收集、全局协方差聚合、跨 client prototype 广播保持可用
- server 从所有 client shard 的 `test.json` 汇总测试集时，能够正确按 domain 去重与分组

### 4.4 可观测性
- 训练日志中能区分：
  - client id
  - client domain
  - 同域内 label skew 情况
- manifest 可作为实验记录的一部分

## 5. 改造范围

### 5.1 需要修改
- `scripts/tools/prepare_officehome_shards.py`
- `data/officehome/shards/manifest.json` 的结构定义
- `scripts/distributed_scripts/run_distributed_ggeur_officehome.sh`
- `scripts/distributed_scripts/distributed_configs/` 下 Office-Home 相关配置生成方式
- `federatedscope/core/data/distributed_loader.py`
- `federatedscope/contrib/worker/ggeur_server.py`

### 5.2 尽量不修改或少修改
- `federatedscope/main.py`
- `DistributedRunner`
- `gRPCCommManager`
- GGEUR 的核心训练公式与 FedAvg 主体逻辑

## 6. 关键设计决策

### 6.1 client 分配策略
第一阶段采用**等量按域分配**：
- `total_clients % 4 == 0`
- `clients_per_domain = total_clients // 4`

理由：
- 与当前脚本结构兼容
- 简化实现与验证
- 足以支撑“多域异构 + 域内 Dirichlet 非 IID”的实验目标

### 6.2 域内 Dirichlet 切分策略
- 保持现有 `apply_lds(...)` 主逻辑
- 但要明确其语义是：**在单个 domain 的 train split 内，对该 domain 下多个 client 做标签分布偏斜切分**
- 增加最小样本保护，避免极端 alpha 下某些 client 训练样本过少

建议新增保护规则：
- 每个 client 最少保留 `min_samples_per_client`
- 可选地为每个类设置最小分配阈值，避免完全空类过多导致统计不稳定

### 6.3 manifest 元信息增强
manifest 需要补充 `domain` 字段，例如：

```json
{
  "client_id": 5,
  "domain": "Clipart",
  "shard_path": ".../client_5",
  "num_samples": {...},
  "label_hist": {...}
}
```

这样可以避免 server 端再依赖“client 编号区间隐式代表域”的脆弱假设。

### 6.4 配置与启动方式
当前 4 份 client YAML 是硬编码，不适合扩展。

建议改造为二选一：

#### 方案 A：新增配置生成脚本
- 输入 `client_num`
- 自动生成：
  - 1 份 server YAML
  - N 份 client YAML
  - 对应端口
  - 对应 shard_path

#### 方案 B：单 client 模板 + 启动时覆写 opts
- 保留 1 份 client 模板 YAML
- 启动时通过 `--opts distribute.client_id ... distributed_data.shard_path ... distribute.client_port ...`
  动态注入

初版建议优先 **方案 B**：
- 改动更小
- 不会生成一堆重复 YAML
- 更符合 DRY

经过评审后，**最终采用方案 A**：
- 增加配置生成逻辑，把 client 配置展开到代码中处理
- 避免启动脚本拼接过长 `--opts` 命令
- 让配置生成、端口分配、shard_path 绑定都成为可审计的代码逻辑
- 代价是会引入额外的配置生成步骤，但这在工程可维护性上是可接受的

### 6.5 server 评估逻辑
当前 server 从 shard `test.json` 汇总测试集并按路径反推 domain。

这条链路原则上能兼容多 client/域，因为：
- 同一域下多个 client 的 `test.json` 当前是共享同域测试集副本
- server 已经做了路径去重

但仍建议增强：
- 优先从 manifest 读取 `domain`
- 加日志校验同域多个 client 的 `test.json` 是否一致

## 7. 四周实施规划

## 第 1 周：数据分片与元信息改造

### 目标
把“每域 1 client”的静态切分，升级为“每域多个 client + manifest 显式 domain 元信息”的可靠数据准备能力。

### 工作项
- 梳理 `prepare_officehome_shards.py` 的输入/输出契约
- 明确 `clients=4k` 的输入约束与报错信息
- 为 manifest 增加 `domain` 字段
- 在 shard `meta.json` 中增加 `domain`
- 为 Dirichlet 切分增加基本健壮性保护：
  - `min_samples_per_client`
  - 极端空 shard 告警
- 输出多个目标实验规模的数据样本：
  - 4 clients
  - 8 clients
  - 12 clients

### 第 1 周具体实现步骤

1. 扩展 `prepare_officehome_shards.py` 的参数与校验逻辑
- 新增最小样本约束参数
- 把 `clients % 4 != 0` 从“自动降级修正”改为“显式失败”
- 明确输出目录覆盖与数据集域目录缺失时的报错

2. 改造 shard 元信息结构
- 在 `meta.json` 中写入 `domain`
- 在 `manifest.json` 的每个 client 条目中写入 `domain`
- 在 manifest 顶层补充 `domains` 和 `clients_per_domain`，减少后续代码推断

3. 重构域内切分逻辑
- 保留 Dirichlet 切分主逻辑
- 为 Dirichlet/均匀切分统一增加最小样本保护
- 对无法满足最小样本约束的场景直接报错，而不是静默生成劣质 shard

4. 增加生成阶段可观测性
- 输出每个 domain 下各 client 的样本数摘要
- 输出空 shard / 超小 shard 告警
- 输出最终 manifest 路径与 client 统计信息

5. 验证
- 用 `8 client` 和 `12 client` 真实生成样例验证
- 检查 manifest 与各 shard `meta.json` 中是否带有 `domain`
- 检查同域多个 client 的 `label_hist` 是否存在明显差异

### 交付物
- 更新后的 shard 生成脚本
- 新版 manifest 结构说明
- 一组真实生成样例

### 验收标准
- `clients=8/12` 时能成功生成 shard
- 每个 client 能从 manifest 中明确获得所属 domain
- 同一域下多个 client 的 train label histogram 明显不同
- 当 `min_samples_per_client` 无法满足时，脚本会显式失败并给出可读错误

### 第 1 周实际完成情况

已完成以下实现：

- `prepare_officehome_shards.py` 新增 `--min-samples-per-client`
- `clients % 4 != 0` 时改为显式失败，不再自动降级为 4 clients
- `meta.json` 增加 `domain`
- `manifest.json` 增加：
  - `clients_per_domain`
  - `domains`
  - `min_samples_per_client`
  - 每个 client 条目的 `domain`
- Dirichlet/均匀切分后统一执行最小样本保护
- 当最小样本约束无法满足时，脚本会以可读错误退出

已完成验证：

- `8 clients` 样例生成成功
- `12 clients` 样例生成成功
- `12 clients, min_samples_per_client=1000` 时按预期失败
- `clients=10` 时按预期失败
- 抽查同域 `Art` 的 3 个 client，`label_hist` 已明显不同，满足域内非 IID 预期

## 第 2 周：训练配置与启动链路改造

### 目标
去掉“固定 4 个 client YAML + 固定 4 个 client 进程”的限制。

### 工作项
- 设计统一 client 启动模板
- 改造 `run_distributed_ggeur_officehome.sh`：
  - 根据 manifest 或输入参数动态计算 client 数
  - 循环拉起所有 client
  - 自动分配 `client_port`
  - 自动注入 `client_id` 与 `shard_path`
- 保留 server 单点配置
- 增加启动前校验：
  - manifest 中 client 数与 `federate.client_num` 一致
  - 所有 shard 路径存在
  - 所有 `train.json` 完整

### 交付物
- 动态 client 启动脚本
- 精简后的 Office-Home distributed 配置组织方式

### 验收标准
- 无需手工复制 client YAML 即可启动 8/12 client 实验
- 所有 client 能成功 join 到 server
- 日志中可确认各 client 使用了正确的 shard 和端口

## 第 3 周：训练与评估链路兼容性改造

### 目标
让 GGEUR 统计、增强、聚合、评估全流程稳定适配“同域多个 client”。

### 工作项
- 检查 `distributed_loader.py` 是否需要把 `domain` 注入 `metadata`
- 在 `GGEURClient` 中增加 domain 级日志输出
- 在 `GGEURServer` 中增强以下逻辑：
  - 本地统计量收集日志按 domain 汇总
  - `all_prototypes` / `global_prototypes` 在多 client 同域情况下的可解释性日志
  - 测试集重建优先使用 manifest 的 `domain` 字段
- 校验以下关键路径：
  - Round 0 统计收集
  - global covariance 广播
  - augmentation ready 同步
  - round 1+ MLP FedAvg
  - server test set 去重与 domain 评估

### 交付物
- 兼容多 client/域的训练链路修改
- 训练日志增强

### 验收标准
- 8/12 client 实验可跑完整个初始化阶段
- server 能正确收到全部 client 的统计量
- 评估阶段按 4 个 domain 输出准确率，不出现 domain 混淆

## 第 4 周：验证、压测与实验基线沉淀

### 目标
补齐可重复实验与稳定性验证，形成后续研究可复用基线。

### 工作项
- 端到端实验验证：
  - 4 clients（回归基线）
  - 8 clients（每域 2 client）
  - 12 clients（每域 3 client）
- 对不同 `alpha` 做对比验证：
  - `alpha=1.0`
  - `alpha=0.5`
  - `alpha=0.1`
- 检查稳定性问题：
  - gRPC 大消息传输压力
  - 特征缓存目录冲突
  - 心跳超时配置是否仍合理
  - client 极小样本导致统计矩阵退化的问题
- 产出基线报告：
  - 数据规模
  - 每域 client 数
  - label skew 概览
  - 训练是否稳定
  - 初步精度结果

### 交付物
- 一组可复现实验命令
- 一份实验基线记录
- 遗留问题清单与后续优化建议

### 验收标准
- 至少 1 组 8 client 和 1 组 12 client 能完整跑通
- 没有因多 client/域导致的新协议级错误
- 输出结果可支撑后续正式实验

## 8. 任务拆分建议

### 8.1 数据侧
- shard 生成脚本增强
- manifest/schema 设计
- label skew 统计可视化或摘要输出

### 8.2 工程侧
- 启动脚本动态化
- 配置模板化
- 日志与校验增强

### 8.3 训练侧
- GGEUR server/client 对 metadata/domain 的兼容
- 评估链路健壮性
- 多 client 同域场景回归测试

## 9. 主要风险与缓解策略

### 9.1 风险：Dirichlet 过于极端导致空 shard 或超小 shard
- 影响：统计量不稳定，协方差退化，训练样本数过低
- 缓解：
  - 引入 `min_samples_per_client`
  - 对过小 shard 自动重分配或直接报错

### 9.2 风险：client 数变多后 gRPC 序列化和启动时序更脆弱
- 影响：join timeout、统计量广播变慢
- 缓解：
  - 保留较宽的 `heartbeat_timeout`
  - 为 join 阶段和统计阶段增加更明确的日志
  - 必要时增大 join timeout

### 9.3 风险：特征缓存冲突
- 影响：不同 client 误用缓存
- 缓解：
  - 明确以 `client_id` 或 `shard_path hash` 作为 cache 子目录
  - 让 server cache 与 client cache 强隔离

### 9.4 风险：server 测试集聚合依赖路径反推 domain，不够稳健
- 影响：后续数据组织变更时容易失效
- 缓解：
  - manifest 显式记录 domain
  - server 优先使用 manifest 信息

## 10. 验收口径

满足以下条件即可认为本阶段目标完成：

- Office-Home 支持 `4k` 个 client，且每域 `k` 个 client
- 同域多个 client 的 train split 由 Dirichlet 分布生成
- 分布式训练脚本可自动拉起全部 client
- GGEUR 统计、增强、训练、评估链路可正常运行
- server 仍能按 4 个域输出测试结果
- 至少一组 `k > 1` 的实验端到端跑通

## 11. 建议的实现顺序

为降低返工，建议严格按下面顺序推进：

1. 先改 shard 生成与 manifest
2. 再改启动脚本与配置注入
3. 再补训练与评估链路兼容
4. 最后做大规模实验和稳定性优化

这样可以保证每一步都有独立可验证产物，避免同时修改数据、训练、通信三条链路带来的排障复杂度。

## 11.1 第 1 周完成定义

第 1 周完成后，应当达到下面的状态：

- 使用单个脚本即可生成 `4/8/12` client 的 Office-Home shard
- manifest 与每个 shard 的 `meta.json` 都带有 `domain`
- 同域内多 client 已体现 Dirichlet label skew
- 数据准备阶段已经具备为第 2 周“配置生成脚本”提供稳定输入的能力

## 12. 补充说明

本规划默认以下前提成立：
- 第一阶段仍只考虑 Office-Home
- 每个域分配相同数量 client
- server 仍然只负责聚合与评估，不持有训练 shard
- GGEUR 当前“Round 0 先收统计，再进入训练”的主流程不改

如果后续要扩展到“不同域分配不同 client 数”或“一个 client 持有多个域的数据”，建议另起一个设计文档，因为那会引入新的数据映射抽象，不适合混在本次四周计划里。
