# 周报（修复 gRPC 序列化与配置属性错误）

## 1. 目标&场景
- 目标：解决 GGEUR 分布式训练中客户端与服务器间 gRPC 通信数据序列化不一致的问题，以及配置属性不兼容导致的模型初始化失败
- 场景：GGEUR 分布式训练流程，客户端提取 CLIP 特征后向服务器上传本地统计量（均值、协方差矩阵），服务器聚合后分发全局统计量
- 问题表现：训练启动后在统计量上传/聚合阶段报错 `ValueError: operands could not be broadcast together with shapes (512,) (26112,)`，以及模型初始化阶段报错 `AttributeError: 'model' has no attribute 'num_classes'`

## 2. 问题分析

### 2.1 问题一：gRPC 序列化导致数组形状异常

#### 现象描述
客户端计算完本地统计量后通过 gRPC 发送给服务器，服务器在聚合协方差矩阵时报错：
```
ValueError: operands could not be broadcast together with shapes (512,) (26112,)
```
其中 26112 = 512 × 51，表明二维数组在传输过程中被展平为一维。

#### 根因分析
客户端直接将 NumPy 数组（`np.ndarray`）作为消息内容通过 gRPC 发送。gRPC 底层使用 Protocol Buffers 进行序列化，对于 NumPy 数组的处理依赖 pickle 序列化。在跨进程传输过程中，NumPy 数组的序列化/反序列化行为不确定：
- 一维数组 `mean (512,)` 通常能正确还原
- 二维数组 `cov (512, 512)` 在某些情况下被展平为一维 `(262144,)` 或发生维度丢失 `(26112,)`
- 不同客户端的序列化行为可能不一致，导致部分客户端正常而部分客户端异常

问题代码位于 `ggeur_client.py` 的 `_upload_statistics()` 方法：
```python
# 修复前：直接传递 numpy 对象
content = {
    'means': self.local_means,       # dict of np.ndarray
    'covs': self.local_covs,         # dict of np.ndarray
    'counts': self.local_counts,
    'prototypes': prototypes         # dict of np.ndarray
}
```

### 2.2 问题二：配置属性名不兼容

#### 现象描述
模型初始化（MLP、CNN）阶段报错：
```
AttributeError: 'CN' object has no attribute 'num_classes'
```

#### 根因分析
GGEUR 代码中使用了 `self._cfg.model.num_classes` 读取类别数，但 FederatedScope 框架标准配置使用 `self._cfg.model.out_channels` 作为模型输出维度（分类任务即类别数）。两者不兼容导致属性访问失败。

涉及 `ggeur_client.py` 中 6 处、`ggeur_server.py` 中 3 处使用了非标准的 `num_classes` 属性。

## 3. 解决方案

### 3.1 gRPC 序列化修复

**策略**：在发送端将 NumPy 数组显式转换为 Python 原生 list，在接收端将 list 转换回 NumPy 数组并进行形状校验。

#### 客户端（`ggeur_client.py`）修改
在 `_upload_statistics()` 中，发送前进行显式序列化：
```python
# 修复后：显式转换为 Python list
for class_idx in self.local_means.keys():
    class_idx = int(class_idx)
    
    mean = self.local_means[class_idx]
    assert mean.shape == (self.ggeur_cfg.embedding_dim,)
    means_serialized[class_idx] = mean.tolist()       # (512,) -> list[512]
    
    cov = self.local_covs[class_idx]
    assert cov.shape == (self.ggeur_cfg.embedding_dim, self.ggeur_cfg.embedding_dim)
    covs_serialized[class_idx] = cov.tolist()         # (512,512) -> list[512][512]
    
    prototypes_serialized[class_idx] = mean.tolist()

content = {
    'means': means_serialized,
    'covs': covs_serialized,
    'counts': self.local_counts,
    'prototypes': prototypes_serialized
}
```

#### 服务器（`ggeur_server.py`）修改
在接收统计量的回调中，将 list 转换回 NumPy 数组并校验形状：
```python
# 修复后：接收端显式反序列化并校验
for class_idx in means.keys():
    class_idx = int(class_idx)
    
    mean = np.array(means[class_idx], dtype=np.float32)
    cov = np.array(covs[class_idx], dtype=np.float32)
    
    if mean.shape != (expected_dim,):
        raise ValueError(f"Client {client_id}, Class {class_idx}: "
                         f"mean shape mismatch - expected ({expected_dim},), got {mean.shape}")
    
    if cov.shape != (expected_dim, expected_dim):
        raise ValueError(f"Client {client_id}, Class {class_idx}: "
                         f"cov shape mismatch - expected ({expected_dim}, {expected_dim}), got {cov.shape}")
    
    validated_means[class_idx] = mean
    validated_covs[class_idx] = cov
```

### 3.2 配置属性修复

将所有 `self._cfg.model.num_classes` 替换为 `self._cfg.model.out_channels`，与 FederatedScope 标准配置对齐。

### 3.3 特征维度校验增强

在特征提取阶段（`_extract_features()`）和统计量计算阶段（`_compute_statistics()`）增加维度校验，确保：
- 特征矩阵为二维 `(n_samples, embedding_dim)`
- 均值向量为 `(embedding_dim,)`
- 协方差矩阵为 `(embedding_dim, embedding_dim)`

提前发现维度问题，避免错误传播到聚合阶段。

## 4. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `contrib/worker/ggeur_client.py` | 1) `_upload_statistics()` 增加 numpy→list 显式转换和形状断言；2) `_extract_features()` 增加特征维度校验；3) `_compute_statistics()` 增加均值、协方差形状校验；4) 6 处 `num_classes` → `out_channels` |
| `contrib/worker/ggeur_server.py` | 1) 统计量接收回调中增加 list→numpy 转换和形状校验；2) 3 处 `num_classes` → `out_channels` |

## 5. 测试与验证

### 5.1 已完成验证
- 静态检查：所有修改文件通过 Pylance 静态分析，无语法或类型错误
- 导入测试：
  ```bash
  python3 -c "from federatedscope.contrib.worker.ggeur_client import GGEURClient; print('OK')"
  python3 -c "from federatedscope.contrib.worker.ggeur_server import GGEURServer; print('OK')"
  ```

### 5.2 未完成验证
由于开发机器在测试过程中出现死机，以下端到端测试未能完成：
- **分布式训练全流程测试**：启动 GGEUR 分布式训练（1 server + 多 client），验证统计量上传→聚合→分发完整链路是否正常
- **多客户端场景测试**：验证不同客户端具有不同类别子集时，序列化/反序列化是否一致
- **协方差聚合正确性测试**：对比分布式聚合结果与单机计算结果，确认数值一致性
- **模型初始化测试**：验证 `out_channels` 替换后 MLP/CNN 模型能否正确初始化并完成训练

> **注**：机器死机原因初步判断为 GPU 显存溢出或系统 OOM（训练过程涉及大量 512×512 协方差矩阵的内存分配），后续恢复后需优先补全上述测试。

## 6. 实现总结
- **问题根因**：gRPC 传输 NumPy 数组时序列化行为不确定，导致多维数组形状丢失；配置属性名 `num_classes` 与框架标准 `out_channels` 不一致
- **解决方案**：发送端显式 `tolist()` + 接收端显式 `np.array()` + 形状校验；统一使用 `out_channels`
- **修改范围**：`ggeur_client.py`（+93 行）、`ggeur_server.py`（+58 行）
- **向后兼容**：无配置变更，无 API 变更，仅影响 GGEUR 分布式训练路径

## 7. 下周计划
- 恢复机器后补全端到端分布式测试
- 关注协方差矩阵内存占用，考虑是否需要引入对角近似或低秩近似以降低内存压力
- 继续排查之前遗留的问题 1（重连竞态条件死锁）和问题 4（多客户端同时重连的 gRPC 消息队列竞态问题）
