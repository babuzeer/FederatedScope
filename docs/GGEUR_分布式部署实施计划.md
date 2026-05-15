# GGEUR 分布式部署实施计划

本文档用于统一整理当前仓库中已经完成的工作、分布式基础设施的现状，以及后续把现有数据集与 GGEUR 各方法完整部署到分布式服务器上的实施计划。

本文档的目标不是讨论单次实验，而是给后续整个分布式推进过程提供统一参照。

## 1. 最终目标

最终目标可以明确拆成两部分：

1. 将当前已经支持的数据集完整迁移到分布式训练流程中。
2. 将 GGEUR 相关的方法矩阵在分布式服务器上形成可复现、可管理、可批量运行的部署方案。

这里的“完整部署”至少应包含：

- 服务端与客户端配置可区分
- 多机通信链路稳定
- 数据、模型、日志目录统一
- 各方法有对应的分布式配置文件
- 训练、停止、异常恢复有统一脚本
- 至少一套最小验证流程可证明方案可用

## 2. 当前已经完成的工作

当前工作已经不是从零开始，主要已经完成了四部分内容。

### 2.1 数据集与数据加载支持

当前 `GGEUR` 数据加载逻辑已经支持：

- `PACS`
- `OfficeHome`
- `DomainNet`

对应入口在：

- [federatedscope/contrib/data/ggeur_data.py](/D:/Projects/FederatedScope/federatedscope/contrib/data/ggeur_data.py)

其中已经实现了：

- 不同数据集的分流加载逻辑
- 多 domain 数据组织
- 按 domain 划分 client 的基本方式
- `LDS` / `Dirichlet` 非 IID 划分支持
- 按配置动态调整 `client_num`

`DomainNet` 的数据集定义和域发现逻辑已经单独落地在：

- [federatedscope/cv/dataset/domainnet.py](/D:/Projects/FederatedScope/federatedscope/cv/dataset/domainnet.py)

### 2.2 GGEUR 配置能力

`GGEUR` 的配置项已经扩展得比较完整，核心配置在：

- [federatedscope/core/configs/cfg_ggeur.py](/D:/Projects/FederatedScope/federatedscope/core/configs/cfg_ggeur.py)

当前已经覆盖的能力包括：

- `clip / cnn / timm` 三类特征提取模式
- 特征缓存与 extractor 卸载
- Gaussian 特征增强相关参数
- MLP 分类器参数
- `FedProto` 集成参数
- `LDS` 非 IID 划分参数
- `CNN distillation / feature alignment / separated training`
- `MOON`
- `PromptFL`
- `DomainNet` 专用域配置：
  - `domainnet_domains`
  - `domainnet_shared_classes_only`

这说明“方法层配置能力”已经比较完整，后面分布式部署时不需要重做方法定义，重点是做部署映射。

### 2.3 实验配置矩阵与脚本

当前已经积累了较完整的实验配置目录，主要包括：

- `scripts/example_configs/ggeur_baseline_vit`
- `scripts/example_configs/ggeur_baseline_cnn`
- `scripts/example_configs/ggeur_baseline_mixer`
- `scripts/example_configs/ggeur_promptfl`
- `scripts/example_configs/ggeur_baseline_vit_pacs`
- `scripts/example_configs/ggeur_baseline_cnn_pacs`
- `scripts/example_configs/ggeur_baseline_mixer_pacs`
- `scripts/example_configs/ggeur_baseline_vit_domainnet`
- `scripts/example_configs/ggeur_baseline_cnn_domainnet`
- `scripts/example_configs/ggeur_baseline_mlp_domainnet`

其中，`DomainNet` 方向已经补齐了多分支 baseline：

- `FedAvg`
- `FedProx`
- `FedProto`
- `FedOpt`
- `MOON`
- `GGEUR + FedAvg`

此外还已经有批量运行脚本：

- [scripts/run_domainnet_all.sh](/D:/Projects/FederatedScope/scripts/run_domainnet_all.sh)

该脚本已经具备：

- 按 `vit / cnn / mlp / all` 切组执行
- 统一日志输出
- 汇总运行计划和运行结果
- `DRY_RUN` 和 `ON_ERROR` 控制

这部分成果说明：单机/本地实验矩阵已经有一定基础，后续重点是把这些现有配置体系映射到分布式部署体系。

### 2.4 文档与分析资料

当前已经有的相关文档包括：

- [docs/GGEUR_技术文档.md](/D:/Projects/FederatedScope/docs/GGEUR_技术文档.md)
- [docs/GGEUR_使用说明.md](/D:/Projects/FederatedScope/docs/GGEUR_使用说明.md)
- [docs/GGEUR_PromptFL_技术文档.md](/D:/Projects/FederatedScope/docs/GGEUR_PromptFL_技术文档.md)
- [docs/DomainNet_完整说明.md](/D:/Projects/FederatedScope/docs/DomainNet_完整说明.md)
- [docs/AID_ViT_分布式部署方案.md](/D:/Projects/FederatedScope/docs/AID_ViT_分布式部署方案.md)
- [docs/分布式训练实施方案.md](/D:/Projects/FederatedScope/docs/分布式训练实施方案.md)

其中：

- `DomainNet_完整说明.md` 已经可以作为当前 DomainNet 数据处理逻辑的说明依据
- `分布式训练实施方案.md` 已经总结了这轮分布式基础设施修改的设计意图
- `AID_ViT_分布式部署方案.md` 虽然任务对象不再继续推进，但里面关于“先 server、后 client、PID 管理、分配置拆分”的思路仍然可复用

## 3. 当前分布式实现的现状

当前分布式工作还处在“基础设施补稳”的阶段，还没有进入“任务级大规模部署”。

### 3.1 已经在工作区中落地的分布式改动

当前工作区里，分布式相关的主要修改集中在：

- [federatedscope/core/workers/client.py](/D:/Projects/FederatedScope/federatedscope/core/workers/client.py)
- [federatedscope/core/workers/server.py](/D:/Projects/FederatedScope/federatedscope/core/workers/server.py)
- [federatedscope/core/auxiliaries/logging.py](/D:/Projects/FederatedScope/federatedscope/core/auxiliaries/logging.py)
- [scripts/distributed_scripts/run_distributed_lr_managed.sh](/D:/Projects/FederatedScope/scripts/distributed_scripts/run_distributed_lr_managed.sh)
- [scripts/distributed_scripts/stop_distributed.sh](/D:/Projects/FederatedScope/scripts/distributed_scripts/stop_distributed.sh)

这些改动的目的很明确：

- `client.py`
  - 在客户端收到 `assign_client_id` 之前，先缓存其他消息
  - 避免 client 启动顺序不同导致消息乱序异常

- `server.py`
  - 增加 join 超时控制
  - 记录首个 client 接入时间
  - 超时后输出当前 join 状态
  - 增加更明确的 ID 分配日志

- `logging.py`
  - 兼容 frozen config 状态下的输出目录创建

- `managed shell scripts`
  - 给单机多进程 distributed 模式提供 PID 管理
  - 支持统一停止和残留清理

### 3.2 这部分分布式改动已经解决的问题

这套分布式基础设施改动主要解决了三个常见问题：

1. 客户端启动顺序不一致时，可能先收到非 ID 消息而触发异常。
2. 服务端在缺 client 的情况下容易一直卡死等待，没有清晰错误边界。
3. 后台拉起多个进程后，训练中断时很容易留残进程。

换句话说，这一轮改动解决的是“分布式模式本身先稳定下来”，不是“某个具体数据集立刻分布式跑完”。

### 3.3 这部分还没有完成的地方

当前还没完成的是：

- 这套分布式基础设施还没有做一次完整的最小验证闭环
- 还没有形成一次独立提交，作为后续工作的基线
- 还没有把 `GGEUR` 的具体任务配置拆成：
  - `server.yaml`
  - `client_1.yaml ... client_n.yaml`
- 还没有把远程服务器目录、模型缓存、数据路径、端口分配规则统一化
- 还没有真正把 `OfficeHome / PACS / DomainNet` 的方法矩阵接入分布式运行

## 4. 当前实现能支撑什么，不能支撑什么

### 当前已经能支撑的事

- 可以继续完善并验证 FederatedScope 的 distributed 基础逻辑
- 可以把现有单机配置体系逐步拆成分布式配置体系
- 可以在 toy / 最小样例层面验证 gRPC 链路是否稳定
- 可以以 `PACS / OfficeHome / DomainNet` 为目标，逐步扩展到真实多机训练

### 当前还不能直接保证的事

- 不能保证“所有现有 GGEUR 方法”现在立刻在多机上直接跑通
- 不能保证“所有现有数据集”都已经有现成的分布式 server/client 配置
- 不能保证“远程服务器环境”已经统一到适合批量部署的状态
- 不能保证“现有所有方法”在分布式部署后都没有额外通信/缓存/路径问题

所以当前阶段最重要的不是盲目扩方法，而是先建立一套稳定的部署路径。

## 5. 总体实施策略

后续实施不应按“数据集全部一起上、方法全部一起上”的方式推进，而应该采用“先底座、后样板、再扩展”的顺序。

建议采用以下原则：

- 先验证分布式底座本身可行
- 再选一个数据集和一个方法做第一个样板
- 样板跑通后再横向复制到其他方法
- 每推进一步，都要形成固定的配置、脚本和文档

## 6. 分阶段实施计划

### 阶段一：固化分布式基础设施

目标：

- 将当前分布式相关改动固化为一套明确的基线

本阶段工作：

- 只处理 distributed 相关改动
- 不混入新的任务逻辑和新的实验配置
- 将以下内容整理并确认：
  - `client.py`
  - `server.py`
  - `logging.py`
  - `run_distributed_lr_managed.sh`
  - `stop_distributed.sh`
  - `分布式训练实施方案.md`

验收标准：

- 分布式基础设施逻辑清晰
- 文件范围稳定
- 具备作为后续任务部署基线的条件

### 阶段二：做最小可行验证

目标：

- 证明“当前 distributed 方案本身可用”

本阶段不使用真实大任务，只做最小验证。

建议验证内容：

1. 单机 toy LR 冒烟
2. client 延迟启动场景
3. 缺少一个 client 的 join 超时场景
4. 手动中断后的统一停止与残留清理

建议使用：

- [scripts/distributed_scripts/run_distributed_lr_managed.sh](/D:/Projects/FederatedScope/scripts/distributed_scripts/run_distributed_lr_managed.sh)
- [scripts/distributed_scripts/stop_distributed.sh](/D:/Projects/FederatedScope/scripts/distributed_scripts/stop_distributed.sh)

验收标准：

- server 能正常监听并等待 client
- client 能被分配 ID 并进入训练
- 超时场景能明确失败而不是静默卡死
- 强制中断后无残留 distributed 进程

### 阶段三：统一远程部署规范

目标：

- 在服务器上形成统一目录规范和运行规则

建议明确以下内容：

- 代码目录
- 数据目录
- 预训练模型目录
- 日志输出目录
- 端口使用规则
- client 命名规则
- GPU 编号使用规则

建议产出：

- 一份服务器目录规范文档
- 一份端口和角色分配表
- 一份部署前检查清单

验收标准：

- 所有后续任务的 server/client 配置都能按同一套路径规范生成

### 阶段四：先落一个样板任务

目标：

- 先把一个真实任务完整接入 distributed，而不是一下子扩所有方法

建议优先顺序：

1. `OfficeHome`
2. `PACS`
3. `DomainNet`

建议优先方法：

1. `GGEUR + FedAvg`
2. `FedProx`

原因：

- `OfficeHome/PACS` 相对更轻，问题更少
- `DomainNet` 规模更大，更适合在样板跑通后再推进

本阶段应补齐：

- `server.yaml`
- `client_1.yaml ... client_n.yaml`
- `start_server.sh`
- `start_client.sh`
- `stop.sh`

验收标准：

- 至少一个真实数据集、一个真实方法可在分布式服务器上完整跑通

### 阶段五：扩展到完整方法矩阵

目标：

- 将现有方法矩阵逐步扩展到分布式版本

建议扩展顺序：

- `FedAvg`
- `FedProx`
- `FedProto`
- `FedOpt`
- `MOON`
- `GGEUR + FedAvg`
- `PromptFL`

每个方法推进时要补齐：

- 对应分布式配置
- 对应分布式启动脚本
- 对应验证记录

验收标准：

- 目标数据集上的方法矩阵具备完整分布式部署能力

## 7. 推荐的优先级顺序

如果只考虑下一步最应该做什么，推荐顺序如下：

1. 固化并验证分布式基础设施
2. 明确远端服务器目录和端口规范
3. 选择一个样板任务先落地
4. 再逐个把其他方法和数据集扩上去

当前不建议直接做的事情是：

- 一次性给所有数据集都补 distributed 配置
- 一次性把所有 GGEUR 方法都扔到远端跑
- 在没有最小验证结论前直接投入真实大规模训练

这样做的问题不是工作量大，而是失败时无法定位到底是：

- distributed 底座问题
- 数据路径问题
- 任务配置问题
- 特定方法的问题

## 8. 下一步建议

基于当前状态，最合理的下一步是：

1. 先把当前分布式基础设施改动单独确认下来
2. 做一次最小 distributed 验证
3. 验证通过后，选定第一个真实样板任务

建议第一个样板任务优先选：

- 数据集：`OfficeHome` 或 `PACS`
- 方法：`GGEUR + FedAvg`

原因：

- 这是最能代表现有 GGEUR 主路径的组合
- 配置基础现成
- 规模比 `DomainNet` 更适合做第一轮分布式落地

## 9. 本文档的使用方式

后续推进时，建议始终按本文档分层执行：

- 先看“当前已完成工作”，避免重复建设
- 再看“当前分布式现状”，明确边界
- 再按“分阶段实施计划”推进

如果后续某一步实施完成，应直接回写到本文档，更新：

- 已完成项
- 当前阶段
- 下一步计划

这样这份文档就会从“初始实施计划”逐步变成“持续更新的部署主文档”。
