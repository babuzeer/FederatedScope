# FedMIA 攻击使用指南

## 1. 项目简介
**FedMIA** (Federated Membership Inference Attack) 是一种针对联邦学习的成员推断攻击。其核心原理是通过分析服务器收集到的客户端梯度更新（gradients），提取余弦相似度（Cosine Similarity）和梯度范数（Gradient Norm）等特征，并利用非目标客户端的梯度分布构建高斯模型进行假设检验（z-test），从而推断特定的数据样本是否参与了目标客户端的本地训练过程。

## 2. 环境配置
本项目运行在专用的虚拟环境中，确保所有依赖项已正确安装。

- **虚拟环境路径**: `/root/.local/share/mamba/envs/fs`
- **激活环境/运行命令示例**:
  ```bash
  # 使用完整路径执行
  /root/.local/share/mamba/envs/fs/bin/python -m federatedscope.main --cfg ...
  ```

## 3. 文件结构
以下是 FedMIA 攻击新增的核心文件及其作用：

| 文件路径 | 作用描述 |
| :--- | :--- |
| `federatedscope/attack/worker_as_attacker/fedmia_server.py` | 实现 `FedMIAServer` 类，负责收集梯度历史、触发攻击逻辑并计算成员得分。 |
| `federatedscope/attack/privacy_attacks/fedmia_metrics.py` | 提供底层的度量函数，包括损失计算、梯度余弦相似度计算及梯度范数计算。 |
| `scripts/fedmia_attack_example.yaml` | 攻击实验的配置文件，定义了数据集（CIFAR-100）、模型及攻击参数。 |
| `tests/test_fedmia.py` | 针对 FedMIA 实现的单元测试，包含 Mock 数据模拟及核心算法验证。 |

## 4. 启动攻击
### 启动命令
在 `FederatedScope` 根目录下执行以下命令启动攻击实验：
```bash
/root/.local/share/mamba/envs/fs/bin/python -m federatedscope.main --cfg scripts/fedmia_attack_example.yaml
```

### 关键配置项说明
| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `attack.attack_method` | `fedmia` | 指定攻击类型为 FedMIA。 |
| `attack.fedmia_target_client_id` | `1` | 攻击的目标客户端 ID。 |
| `attack.fedmia_variant` | `"FedMIA-I"` | 攻击变体类型，目前主要使用 MDM (Multi-Dimensional Metrics) 模式。 |

## 5. 输出结果说明
攻击程序会在最后一轮训练结束后自动运行，并输出每个查询样本的 `attack score`。

- **Attack Score 含义**: 该分数表示样本属于“成员（Member）”的置信度，取值范围在 `[0, 1]` 之间。
- **判断标准**:
    - **Score > 0.5**: 样本极大概率为“成员”（即该样本参与了目标客户端的训练）。
    - **Score <= 0.5**: 样本可能为“非成员”。
    - 分数越接近 1，推断的确定性越高。

## 6. 单元测试
为确保攻击逻辑的正确性，建议在修改代码后运行单元测试。

### 运行测试命令
```bash
/root/.local/share/mamba/envs/fs/bin/python tests/test_fedmia.py
```

### 测试覆盖场景
- **Mock 联邦环境**: 模拟多轮次、多客户端的梯度上传流程。
- **MDM 特征计算**: 验证梯度提取方式（`loss.backward()`）及特征计算的准确性。
- **假设检验逻辑**: 验证 z-score 的计算过程及 `norm.cdf` 的转换结果是否符合统计学预期。
