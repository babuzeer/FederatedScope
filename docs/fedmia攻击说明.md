好的，我先执行任务二：攻击流程审计。需要查阅关键代码文件来梳理完整流程。
现在我已经收集了所有必要信息，输出FedMIA攻击流程审计文档：

---

## FedMIA 攻击流程审计文档

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FedMIA 攻击完整链路                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【阶段一：离线特征提取】                                                      │
│  CIFAR-100 原始图像 ──→ ConvNeXt-Base ──→ 特征向量 (1024-dim) ──→ .pt 文件    │
│       (32x32)          (frozen)          (N x 1024 tensor)                  │
│                                                                             │
│                          ↓                                                  │
│                                                                             │
│  【阶段二：联邦训练】                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Server (FedMIAServer)                                              │    │
│  │    ├─ 持有: server_test.pt (10000 samples)                          │    │
│  │    ├─ 聚合: FedAvg                                                   │    │
│  │    ├─ 每轮: 接收 client 模型 → 保存 PKL → 聚合                        │    │
│  │    └─ 最后一轮: 执行 8 种攻击                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          ↑↓ 通信                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐                   │
│  │ Client 1 │ │ Client 2 │ │ Client 3 │ ... │ Client 10│                   │
│  │ (target) │ │  (val)   │ │ (shadow) │     │ (shadow) │                   │
│  │ 3682样本 │ │ 3419样本  │ │ 4416样本 │     │ 4502样本  │                   │
│  └──────────┘ └──────────┘ └──────────┘     └──────────┘                   │
│       ↓              ↓            ↓               ↓                        │
│  client_1_train.pt ... client_10_train.pt                                   │
│                                                                             │
│                          ↓                                                  │
│                                                                             │
│  【阶段三：攻击执行】                                                         │
│  PKL 文件 ──→ 提取 loss/logit/gradient 特征 ──→ 8 种攻击方法 ──→ AUC/TPR    │
│                                                                             │
│                          ↓                                                  │
│                                                                             │
│  【阶段四：评估】                                                            │
│  Member (target client 训练样本) vs Non-member (test/mix 样本)              │
│  → AUC, TPR@FPR=0.1/0.01/0.001                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. 关键代码文件和函数调用关系

#### 2.1 文件清单

| 文件路径 | 功能 | 角色 |
|---------|------|------|
| `extract_features.py` | 离线特征提取入口 | 数据准备 |
| `federatedscope/contrib/data/cifar100_features.py` | 加载 .pt 特征文件 | 数据加载器 |
| `federatedscope/contrib/model/mlp_head.py` | 纯 MLP 分类头 | 模型定义 |
| `scripts/fedmia_offline_features.yaml` | 实验配置 | 配置文件 |
| `federatedscope/attack/worker_as_attacker/fedmia_server.py` | FedMIA 攻击服务器 | 攻击核心 |

#### 2.2 函数调用关系

```
main.py
  └── FedRunner.run()
        ├── 初始化阶段
        │     ├── create_data() → cifar100_features.py::load_cifar100_features()
        │     │                        └── 加载 client_{id}_train.pt, meta.pt, server_test.pt
        │     │                        └── 构建 StandaloneDataDict
        │     ├── create_model() → mlp_head.py::build_mlp_head()
        │     │                        └── MLPHead(feature_dim=1024, hidden_dim=256, num_classes=100)
        │     └── worker_builder.py → FedMIAServer()
        │                              └── _init_data_references()
        │                                    ├── 加载 all_train.pt → self.full_train_dataset
        │                                    ├── 加载 meta.pt → self.client_data_indices
        │                                    └── 加载 server_test.pt → self.full_test_dataset
        │
        ├── 训练循环 (每轮)
        │     ├── Server._broadcast_model_para()
        │     ├── Client.train() → GeneralTorchTrainer.train()
        │     │                        └── forward/backward on local data (1 epoch)
        │     ├── Client._report_to_server()
        │     └── FedMIAServer.callback_funcs_model_para()
        │           ├── _initialize_pkl_dir()
        │           ├── 存储 self.global_model_state
        │           ├── _save_client_pkl()  ← 每个参与的 client
        │           │     ├── 计算 test_acc, test_loss
        │           │     ├── _get_client_indices() → train/val/mix indices
        │           │     ├── _get_all_losses_from_indexes() → train_res/val_res/mix_res
        │           │     ├── _compute_gradient_features() → cos/diff/norm (仅 target client)
        │           │     └── torch.save() → client_{id}_losses_epoch{r}.pkl
        │           └── Server.callback_funcs_model_para() → FedAvg 聚合
        │
        └── 最后一轮 (round == total_round_num - 1)
              └── FedMIAServer.run_final_attack()
                    ├── _load_all_pkl_files()
                    ├── _attack_blackbox_loss()
                    ├── _attack_grad_cosine()
                    ├── _attack_grad_diff()
                    ├── _attack_grad_norm()
                    ├── _attack_loss_series()
                    ├── _attack_avg_cosine()
                    ├── _attack_fedmia_i()
                    ├── _attack_fedmia_ii()
                    └── _print_and_save_results()
```

---

### 3. 离线特征提取实现

#### 3.1 输入

| 项目 | 说明 |
|------|------|
| 数据集 | CIFAR-100 (torchvision) |
| 训练集 | 50,000 张 32x32 彩色图像 |
| 测试集 | 10,000 张 32x32 彩色图像 |
| 预处理 | Resize(224) → ToTensor → Normalize(ImageNet均值/标准差) |

#### 3.2 Backbone

| 项目 | 说明 |
|------|------|
| 模型 | ConvNeXt-Base (torchvision 预训练) |
| 权重 | ImageNet-1K 预训练 |
| 输出维度 | 1024 (feature_dim) |
| 参数量 | ~88M (全部冻结) |
| 特征提取方式 | `features` + `avgpool` + `flatten` |

#### 3.3 数据划分

| 项目 | 说明 |
|------|------|
| 划分方式 | LDA (Dirichlet, α=0.5) Non-IID |
| 客户端数量 | 10 |
| Per-client 划分 | train:val:test = 0.8:0.1:0.1 |

#### 3.4 输出文件结构

```
data/cifar100_features/
├── meta.pt                     # 元信息 + client 索引映射
│   └── {client_num, alpha, seed, feature_dim, split,
│        client_train_indices: {cid -> [idx in 0..49999]},
│        client_val_indices: {cid -> [idx]},
│        n_train: 50000, n_test: 10000}
│
├── all_train.pt                # 全量训练特征 (FedMIA 必需)
│   └── {'x': Tensor[50000,1024], 'y': Tensor[50000]}
│
├── server_test.pt              # 全局测试集
│   └── {'x': Tensor[10000,1024], 'y': Tensor[10000]}
│
├── client_1_train.pt           # Client 1 训练数据
│   └── {'x': Tensor[3682,1024], 'y': Tensor[3682], 'indices': [idx...]}
├── client_1_val.pt
├── client_1_test.pt
├── ...
└── client_10_test.pt
```

---

### 4. 联邦训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **参与方** | | |
| `federate.client_num` | 10 | 总客户端数 |
| `federate.sample_client_num` | 3 | 每轮采样的客户端数 |
| **通信** | | |
| `federate.total_round_num` | 300 | 总通信轮数 |
| `federate.mode` | standalone | 单机模拟模式 |
| **本地训练** | | |
| `train.local_update_steps` | 1 | 本地 epoch 数 |
| `train.batch_or_epoch` | epoch | 按 epoch 训练 |
| `train.optimizer.lr` | 0.01 | 学习率 |
| `train.optimizer.weight_decay` | 0.0 | 权重衰减 |
| **模型** | | |
| `model.type` | mlp_head | 纯 MLP 分类头 |
| `model.feature_dim` | 1024 | 输入特征维度 |
| `model.hidden_dim` | 256 | 隐藏层维度 |
| `model.num_classes` | 100 | 分类数 |
| `model.dropout` | 0.1 | Dropout 率 |
| **数据加载** | | |
| `dataloader.batch_size` | 256 | 批大小 |
| `dataloader.shuffle` | True | 是否打乱 |
| **聚合策略** | | |
| 聚合算法 | FedAvg | 平均聚合 |
| **评估** | | |
| `eval.freq` | 5 | 每 5 轮评估一次 |
| `eval.metrics` | [acc, correct] | 评估指标 |

#### 模型架构

```
MLPHead:
  LayerNorm(1024)
  → Linear(1024, 256)
  → ReLU
  → Dropout(0.1)
  → Linear(256, 100)

参数量: 290,148
```

---

### 5. 攻击方法

#### 5.1 攻击类型总览

| 攻击方法 | 特征类型 | 单轮/多轮 | 判定逻辑 |
|---------|---------|----------|---------|
| **Blackbox-Loss** | Loss | 单轮 | 最后一轮 loss，越大越可能是 non-member |
| **Grad-Cosine** | 梯度余弦相似度 | 单轮 | 最后一轮 cos，越大越可能是 member |
| **Grad-Diff** | 梯度差异 | 单轮 | 最后一轮 diff，越大越可能是 member |
| **Grad-Norm** | 梯度范数 | 单轮 | 最后一轮 norm，越小越可能是 member |
| **Loss-Series** | Loss | 多轮 | 所有轮 loss 平均 |
| **Avg-Cosine** | 梯度余弦相似度 | 多轮 | 所有轮 cos 平均 |
| **FedMIA-I** | Loss + CDF | 多轮 | 每轮 loss → CDF 归一化 → 平均 |
| **FedMIA-II** | Cos + CDF | 多轮 | 每轮 cos → CDF 归一化 → 平均 |

#### 5.2 攻击者能力假设

| 假设 | 说明 |
|------|------|
| **被动攻击者** | Server 只观察不篡改 |
| **可访问数据** | Server 持有 `all_train.pt` (50000 样本) + `server_test.pt` (10000 样本) |
| **可访问模型** | 每轮聚合后的全局模型 + 各 client 上传的本地模型 |
| **知道索引** | 知道每个 client 的训练数据索引 (从 meta.pt 获取) |
| **可计算梯度** | 可对任意样本计算 per-sample gradient |

#### 5.3 Shadow Model 设置

| 项目 | 说明 |
|------|------|
| Shadow clients | Client 2-10 (共 9 个，排除 target client 1) |
| Shadow 数据 | 从各 shadow client 随机采样 ≈1000 样本/轮 |
| 用途 | 为 FedMIA-I/II 计算 CDF 归一化参数 (μ, σ) |

#### 5.4 成员推断判定逻辑

**Member 定义**: Target client (client 1) 的训练样本 (indices in `client_train_indices[1]`)

**Non-member 定义**: 
- `test` 模式: server_test.pt 中的样本
- `mix` 模式 (默认): server_test.pt 部分 + shadow clients 采样混合

**判定流程**:
```
对于样本 x:
  1. 提取特征 score(x) (loss / cos / diff / norm)
  2. 若是多轮攻击: score(x) = mean(score_r(x) for r in rounds)
  3. 若是 CDF 攻击: score(x) = norm.cdf(score(x), μ_shadow, σ_shadow)
  4. 比较 score(x) 与阈值 → 判定 member / non-member
```

---

### 6. 评估指标

| 指标 | 说明 |
|------|------|
| **AUC** | ROC 曲线下面积，衡量整体区分能力 |
| **TPR@FPR=0.1** | FPR=10% 时的真阳性率 |
| **TPR@FPR=0.01** | FPR=1% 时的真阳性率 |
| **TPR@FPR=0.001** | FPR=0.1% 时的真阳性率 |

#### 实验结果

| 攻击方法 | AUC | TPR@0.1 | TPR@0.01 | TPR@0.001 |
|---------|-----|---------|----------|-----------|
| Blackbox-Loss | nan | nan | nan | nan |
| Grad-Cosine | nan | nan | nan | nan |
| Grad-Diff | nan | nan | nan | nan |
| Grad-Norm | nan | nan | nan | nan |
| **Loss-Series** | **0.7512** | **0.6871** | **0.6162** | **0.5576** |
| **Avg-Cosine** | **0.8295** | **0.6184** | **0.4068** | **0.3569** |
| FedMIA-I | 0.762 | 0.3745 | 0.0 | 0.0 |
| FedMIA-II | 0.7629 | 0.0 | 0.0 | 0.0 |

**FL 最终测试精度**: 76.05%

---

### 7. 可能影响攻击可靠性的问题

#### 7.1 ⚠️ 梯度特征缺失问题 (已确认)

**现象**: Grad-Cosine / Grad-Diff / Grad-Norm / Blackbox-Loss 四个攻击输出 nan

**原因**: `_compute_gradient_features()` 仅在 `client_id == target_client_id` 时执行，且仅在 target client 被采样参与当轮训练时才有数据

**代码位置**: [fedmia_server.py#L334](federatedscope/attack/worker_as_attacker/fedmia_server.py#L334)
```python
if client_id == self.target_client_id:
    grad_features = self._compute_gradient_features(...)
else:
    # For non-target clients, store empty lists
    save_dict['tarin_cos'] = []
```

**影响**: 若 target client 某轮未被采样，该轮 PKL 文件中梯度字段为空列表，导致相应攻击失效

#### 7.2 ⚠️ 数据泄露风险

| 潜在泄露点 | 说明 |
|-----------|------|
| Server 持有全量数据 | `all_train.pt` 包含所有 50000 训练样本，Server 可直接访问 |
| 索引映射已知 | `meta.pt` 明确记录每个 client 的训练索引 |
| 无差分隐私 | 未使用 DP-SGD 或其他隐私保护机制 |

#### 7.3 ⚠️ 评估公平性问题

| 问题 | 说明 |
|------|------|
| Member/Non-member 分布不均衡 | Member 数量约 3682，Non-member (mix) 约 1000+ |
| 测试集可能包含分布外样本 | CIFAR-100 test set 与 train set 分布可能有偏移 |
| 未使用独立的 held-out set | Non-member 直接使用 test set，可能高估攻击效果 |

#### 7.4 ⚠️ 离线特征模式的特殊性

| 特点 | 影响 |
|------|------|
| 无 backbone 更新 | 梯度仅来自 MLP head (290K 参数)，信息量较少 |
| 特征冻结 | ConvNeXt 提取的特征固定，攻击者可预先计算所有样本特征 |
| 训练速度快 | 每轮仅 0.12s，300 轮共 13 分钟，但可能导致欠拟合 |

#### 7.5 ⚠️ 实验配置建议

| 当前配置 | 建议 | 原因 |
|---------|------|------|
| `sample_client_num=3` | 增大至 5+ | 确保 target client 更频繁被采样 |
| `local_update_steps=1` | 可增大 | 更多本地迭代可能增加隐私泄露 |
| 无 learning rate decay | 添加 | 模型收敛后梯度特征更稳定 |

---

### 总结

当前 FedMIA 攻击实现在离线特征+MLP架构下，**Avg-Cosine (AUC=0.83)** 是最有效的攻击方法。但存在以下问题需要关注：

1. **梯度特征攻击失效**: 需要修改代码确保 target client 每轮都被采样，或收集所有轮的梯度特征
2. **评估可能高估**: 需要使用独立的 non-member 集合进行验证
3. **隐私保护缺失**: 建议添加 DP-SGD 等防御机制