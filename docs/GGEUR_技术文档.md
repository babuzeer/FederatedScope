# GGEUR方法完整技术文档

**Gaussian Geometry-guided Feature Expansion with Unified Representation**

**版本：** 1.0
**适用框架：** FederatedScope
**最后更新：** 2025年1月

---

# 第一章：GGEUR方法概述

## 1.1 背景与动机

### 1.1.1 联邦学习中的多域异构问题

在传统的联邦学习场景中，各客户端的数据通常假设来自相同的分布（IID假设）。然而，现实世界中的数据往往具有显著的**域差异（Domain Shift）**。例如：

- **医疗影像**：不同医院使用的设备、成像参数各不相同
- **自动驾驶**：不同地区的道路环境、天气条件存在差异
- **图像分类**：同一物体可能以照片、绘画、卡通等不同形式呈现

这种多域异构性给联邦学习带来了严峻挑战：
1. **特征空间不一致**：不同域的特征分布差异导致模型难以泛化
2. **模型聚合困难**：简单的参数平均无法有效融合多域知识
3. **性能不均衡**：某些域可能主导聚合结果，导致其他域性能下降

### 1.1.2 标签分布倾斜（Label Distribution Skew）挑战

除了域差异外，联邦学习还面临**标签分布倾斜（LDS）**问题：

```
客户端1: 类别A(80%), 类别B(15%), 类别C(5%)
客户端2: 类别A(5%),  类别B(80%), 类别C(15%)
客户端3: 类别A(10%), 类别B(10%), 类别C(80%)
```

这种非独立同分布（Non-IID）的数据划分会导致：
- 局部模型偏向于本地多数类
- 全局模型在少数类上表现不佳
- 收敛速度变慢且不稳定

### 1.1.3 现有方法的局限性

| 方法 | 局限性 |
|------|--------|
| **FedAvg** | 简单参数平均无法处理域差异，在Non-IID场景下性能下降明显 |
| **FedProx** | 添加近端项约束，但未解决特征空间不一致问题 |
| **MOON** | 依赖对比学习，计算开销大，对小数据集效果有限 |
| **FedBN** | 仅处理批归一化层，无法解决深层特征差异 |

---

## 1.2 GGEUR核心思想

### 1.2.1 方法命名解读

**GGEUR** = **G**aussian **G**eometry-guided Feature **E**xpansion with **U**nified **R**epresentation

- **Gaussian Geometry-guided**：利用高斯分布的几何特性（均值、协方差）指导特征生成
- **Feature Expansion**：通过生成虚拟特征扩展本地数据分布
- **Unified Representation**：基于CLIP预训练模型构建统一的特征表示空间

### 1.2.2 基于CLIP特征空间的统一表示

GGEUR的核心创新在于利用**CLIP（Contrastive Language-Image Pre-training）**模型作为统一的特征提取器：

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIP统一特征空间                          │
│                                                             │
│    ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐    │
│    │域 A │        │域 B │        │域 C │        │域 D │    │
│    │特征 │        │特征 │        │特征 │        │特征 │    │
│    └──┬──┘        └──┬──┘        └──┬──┘        └──┬──┘    │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          ↓                                  │
│                   语义一致的表示                             │
└─────────────────────────────────────────────────────────────┘
```

**CLIP的优势：**
1. **语义一致性**：不同域的同类物体在CLIP特征空间中距离接近
2. **预训练知识**：海量图文数据的预训练提供强大的泛化能力
3. **无需微调**：特征提取器保持冻结，避免灾难性遗忘

### 1.2.3 高斯几何引导的特征扩张策略

GGEUR假设每个类别的特征在CLIP空间中服从多元高斯分布：

$$\mathbf{f}_c \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$$

其中：
- $\boldsymbol{\mu}_c$：类别$c$的特征均值（类原型）
- $\boldsymbol{\Sigma}_c$：类别$c$的协方差矩阵（描述类内变化）

**特征扩张流程：**

1. **局部统计量计算**：每个客户端计算本地各类别的均值和协方差
2. **全局协方差聚合**：服务器使用平行轴定理聚合所有客户端的协方差
3. **虚拟特征生成**：客户端基于全局协方差生成符合多域分布的虚拟特征

```python
# 特征生成示例
for class_id in range(num_classes):
    global_mean = prototypes[class_id]
    global_cov = covariances[class_id]

    # 从多元高斯分布采样
    virtual_features = np.random.multivariate_normal(
        mean=global_mean,
        cov=global_cov,
        size=num_samples_to_generate
    )
```

---

## 1.3 算法流程概览

GGEUR算法分为**五个主要阶段**：

### 阶段流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GGEUR 算法流程                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   阶段 1     │    │   阶段 2     │    │   阶段 3     │              │
│  │  特征提取    │ → │  统计计算    │ →  │  全局聚合    │              │
│  │  (客户端)    │    │  (客户端)    │    │  (服务器)    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ↓                   ↓                   ↓                       │
│   CLIP/CNN提取        计算每个类别的      平行轴定理聚合                 │
│   图像特征向量        均值和协方差         全局协方差                    │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐                                  │
│  │   阶段 4     │    │   阶段 5     │                                  │
│  │  特征增强    │ → │  模型训练    │                                  │
│  │  (客户端)    │    │  (联邦)      │                                  │
│  └──────────────┘    └──────────────┘                                  │
│         │                   │                                           │
│         ↓                   ↓                                           │
│   高斯采样生成          FedAvg训练                                      │
│   虚拟特征              MLP分类器                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 各阶段详细说明

| 阶段 | 执行位置 | 主要操作 | 输出 |
|------|----------|----------|------|
| **阶段1：特征提取** | 客户端 | 使用CLIP/CNN提取图像特征 | 特征向量 $\mathbf{f} \in \mathbb{R}^{512}$ |
| **阶段2：统计计算** | 客户端 | 计算每个类别的均值和协方差 | $\{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c, n_c\}$ |
| **阶段3：全局聚合** | 服务器 | 使用平行轴定理聚合协方差 | 全局协方差 $\boldsymbol{\Sigma}_c^{global}$ |
| **阶段4：特征增强** | 客户端 | 基于全局协方差生成虚拟特征 | 增强后的特征数据集 |
| **阶段5：模型训练** | 联邦 | FedAvg训练MLP分类器 | 全局分类模型 |

### 通信轮次结构

```
Round 0（统计收集轮）:
  Client → Server: 发送局部统计量 (means, covs, counts)
  Server → Client: 返回全局协方差和跨域原型

Round 1+ （模型训练轮）:
  Client: 完成特征增强，本地训练MLP
  Client → Server: 发送模型参数
  Server: FedAvg聚合
  Server → Client: 返回全局模型
```

---

## 1.4 方法优势总结

### 1.4.1 相比现有方法的改进

| 特性 | FedAvg | FedProx | MOON | **GGEUR** |
|------|--------|---------|------|-----------|
| 处理域差异 | ✗ | ✗ | △ | ✓ |
| 处理标签倾斜 | ✗ | △ | △ | ✓ |
| 通信效率 | ✓ | ✓ | △ | ✓ |
| 无需本地大量数据 | ✗ | ✗ | ✗ | ✓ |
| 特征空间对齐 | ✗ | ✗ | △ | ✓ |

### 1.4.2 GGEUR的核心优势

1. **统一特征空间**：CLIP提供跨域语义一致的表示
2. **数据增强**：通过高斯采样补充缺失类别的样本
3. **隐私保护**：仅传输统计量，不泄露原始数据
4. **计算高效**：MLP分类器轻量，训练速度快
5. **灵活扩展**：支持CNN骨干网络、知识蒸馏等多种训练模式

---

## 1.5 适用场景

GGEUR特别适用于以下场景：

### 推荐使用场景

- **多域图像分类**：如Office-Home、PACS、DomainNet等域适应数据集
- **标签分布高度倾斜**：某些客户端缺少特定类别的数据
- **客户端数据量有限**：每个客户端的样本数较少
- **需要快速收敛**：利用CLIP预训练特征加速训练

### 不推荐使用场景

- **自然语言处理任务**：GGEUR专为图像设计
- **像素级任务**：如分割、检测（GGEUR针对分类任务优化）
- **数据高度IID**：简单FedAvg已足够，GGEUR的增益有限

---

**第一章完成。** 下一章将详细介绍GGEUR的核心算法原理，包括平行轴定理、高斯采样等数学细节。

---

# 第二章：核心算法原理

本章详细介绍GGEUR算法的数学原理和实现细节，包括特征提取、统计量计算、协方差聚合和特征增强等核心步骤。

## 2.1 特征提取阶段

### 2.1.1 CLIP特征提取器原理

CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的多模态预训练模型，通过对比学习在4亿图文对上训练，学习到了语义丰富的视觉表示。

**CLIP视觉编码器架构：**

```
输入图像 (224×224×3)
        ↓
┌───────────────────┐
│  Vision Transformer │
│  (ViT-B/16)        │
│                   │
│  Patch Embedding  │
│        ↓          │
│  12× Transformer  │
│     Blocks        │
│        ↓          │
│  [CLS] Token      │
└───────────────────┘
        ↓
特征向量 (512维)
```

**特征提取代码逻辑（`ggeur_client.py`）：**

```python
def _load_clip_model(self):
    """Load CLIP model for feature extraction"""
    import open_clip

    model_name = self.ggeur_cfg.clip_model      # 例如: 'ViT-B-16'
    pretrained = self.ggeur_cfg.clip_pretrained  # 例如: 'openai'

    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    self.clip_model = self.clip_model.to(self.device)
    self.clip_model.eval()  # 冻结模型，仅用于特征提取

def _extract_features(self):
    """Extract CLIP features from local images"""
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(self.device)
            # 提取图像特征
            features = self.clip_model.encode_image(images)
            features = features.cpu().numpy()

            # 按类别组织特征
            for feat, label in zip(features, labels):
                class_idx = int(label)
                self.local_features[class_idx].append(feat)
```

### 2.1.2 CNN特征提取器支持

除CLIP外，GGEUR还支持使用预训练的CNN作为特征提取器，适用于无法使用CLIP的场景。

**支持的CNN骨干网络：**

| 模型 | 特征维度 | 参数量 | 适用场景 |
|------|----------|--------|----------|
| `convnext_tiny` | 768 | 28M | 轻量级部署 |
| `convnext_base` | 1024 | 89M | 平衡性能与效率 |
| `convnext_large` | 1536 | 198M | 追求最佳性能 |
| `resnet18` | 512 | 11M | 快速实验 |
| `resnet50` | 2048 | 25M | 标准基线 |
| `efficientnet_b0` | 1280 | 5M | 移动端部署 |

**CNN特征提取器配置：**

```yaml
ggeur:
  feature_extractor: 'cnn'      # 使用CNN而非CLIP
  cnn_backbone: 'convnext_base' # CNN架构
  freeze_backbone: true         # 冻结骨干网络
  embedding_dim: 1024           # 特征维度（需与骨干网络匹配）
```

### 2.1.3 特征缓存机制

特征提取是计算密集型操作，GGEUR提供缓存机制避免重复计算。

**缓存策略：**

```python
def _get_feature_cache_path(self, domain=None):
    """Get the path for cached features"""
    cache_dir = self.ggeur_cfg.feature_cache_dir
    if not cache_dir:
        cache_dir = os.path.join(data_root, 'clip_feature_cache')

    # 缓存文件命名格式: {数据集}_{域}_{提取器}_{模型}.npz
    # 例如: officehome_Art_clip_ViT_B_16_openai.npz
    cache_filename = f"{dataset}_{domain}_{extractor}_{model}.npz"
    return os.path.join(cache_dir, cache_filename)

def _extract_features(self):
    # 检查缓存
    cache_path = self._get_feature_cache_path()
    if cache_path and os.path.exists(cache_path):
        # 从缓存加载
        cached = np.load(cache_path)
        self.local_features = cached['features']
        self.local_labels = cached['labels']
        return

    # 提取特征...

    # 保存到缓存
    np.savez(cache_path, features=features, labels=labels)
```

**缓存配置：**

```yaml
ggeur:
  use_feature_cache: true                    # 启用缓存
  feature_cache_dir: './clip_feature_cache'  # 缓存目录
```

---

## 2.2 局部统计量计算

### 2.2.1 类别均值（Mean）计算

对于客户端 $k$ 的类别 $c$，设该类别有 $n_c^k$ 个样本，特征为 $\{\mathbf{f}_1^c, \mathbf{f}_2^c, ..., \mathbf{f}_{n_c^k}^c\}$。

**局部均值（类原型）：**

$$\boldsymbol{\mu}_c^k = \frac{1}{n_c^k} \sum_{i=1}^{n_c^k} \mathbf{f}_i^c$$

**代码实现：**

```python
def _compute_local_statistics(self):
    """Compute local mean and covariance for each class"""
    for class_idx, features in self.local_features.items():
        n = features.shape[0]

        # 计算均值
        mean = np.mean(features, axis=0)  # shape: (embedding_dim,)

        self.local_means[class_idx] = mean
        self.local_counts[class_idx] = n
```

### 2.2.2 协方差矩阵（Covariance）计算

协方差矩阵描述特征各维度之间的相关性和变化范围。

**局部协方差：**

$$\boldsymbol{\Sigma}_c^k = \frac{1}{n_c^k} \sum_{i=1}^{n_c^k} (\mathbf{f}_i^c - \boldsymbol{\mu}_c^k)(\mathbf{f}_i^c - \boldsymbol{\mu}_c^k)^T$$

**代码实现：**

```python
def _compute_local_statistics(self):
    for class_idx, features in self.local_features.items():
        n = features.shape[0]
        mean = np.mean(features, axis=0)

        # 中心化
        centered = features - mean  # shape: (n, embedding_dim)

        # 计算协方差矩阵
        cov = (1.0 / n) * np.dot(centered.T, centered)  # shape: (dim, dim)

        self.local_covs[class_idx] = cov
```

**协方差矩阵的几何意义：**

```
          维度2
            ↑
            │    ╱ 主成分方向
            │   ╱
        ┌───┼──╱────┐
        │   │ ╱     │  ← 协方差矩阵描述的椭圆
        │   │╱      │
    ────┼───●───────┼──→ 维度1
        │   │       │
        │   │       │
        └───┼───────┘
            │
```

- **对角线元素**：各维度的方差
- **非对角线元素**：维度间的协方差（相关性）
- **特征值**：主成分方向上的方差大小
- **特征向量**：主成分方向

### 2.2.3 类原型（Prototype）构建

类原型是该类别特征的代表，用于跨客户端知识共享。

```python
# 客户端上传的内容
content = {
    'client_id': self.ID,
    'means': self.local_means,      # {class_idx: mean_vector}
    'covs': self.local_covs,        # {class_idx: cov_matrix}
    'counts': self.local_counts,    # {class_idx: sample_count}
    'prototypes': self.local_means  # 原型即为均值
}
```

---

## 2.3 全局协方差聚合

### 2.3.1 平行轴定理（Parallel Axis Theorem）

平行轴定理是GGEUR算法的核心数学工具，用于将多个客户端的局部协方差聚合为全局协方差。

**问题设定：**
- $K$ 个客户端，每个客户端 $k$ 有类别 $c$ 的数据
- 客户端 $k$ 的统计量：$\boldsymbol{\mu}_c^k$（均值）, $\boldsymbol{\Sigma}_c^k$（协方差）, $n_c^k$（样本数）
- 目标：计算全局协方差 $\boldsymbol{\Sigma}_c^{global}$

**定理表述：**

全局协方差可以分解为两部分：

$$\boldsymbol{\Sigma}_c^{global} = \underbrace{\frac{1}{N_c} \sum_{k=1}^{K} n_c^k \boldsymbol{\Sigma}_c^k}_{\text{加权局部协方差}} + \underbrace{\frac{1}{N_c} \sum_{k=1}^{K} n_c^k (\boldsymbol{\mu}_c^k - \boldsymbol{\mu}_c^{global})(\boldsymbol{\mu}_c^k - \boldsymbol{\mu}_c^{global})^T}_{\text{客户端间方差}}$$

其中：
- $N_c = \sum_{k=1}^{K} n_c^k$ 是类别 $c$ 的全局样本数
- $\boldsymbol{\mu}_c^{global} = \frac{1}{N_c} \sum_{k=1}^{K} n_c^k \boldsymbol{\mu}_c^k$ 是全局均值

### 2.3.2 加权聚合公式推导

**第一步：计算全局均值**

$$\boldsymbol{\mu}_c^{global} = \frac{\sum_{k=1}^{K} n_c^k \boldsymbol{\mu}_c^k}{\sum_{k=1}^{K} n_c^k}$$

**第二步：计算加权局部协方差**

$$\boldsymbol{\Sigma}_{within} = \frac{1}{N_c} \sum_{k=1}^{K} n_c^k \boldsymbol{\Sigma}_c^k$$

**第三步：计算客户端间方差**

$$\boldsymbol{\Sigma}_{between} = \frac{1}{N_c} \sum_{k=1}^{K} n_c^k (\boldsymbol{\mu}_c^k - \boldsymbol{\mu}_c^{global})(\boldsymbol{\mu}_c^k - \boldsymbol{\mu}_c^{global})^T$$

**第四步：合并**

$$\boldsymbol{\Sigma}_c^{global} = \boldsymbol{\Sigma}_{within} + \boldsymbol{\Sigma}_{between}$$

### 2.3.3 代码实现详解

```python
def _aggregate_covariances(self):
    """Aggregate covariance matrices using parallel axis theorem"""

    for class_idx in all_classes:
        means = []   # 各客户端的均值
        covs = []    # 各客户端的协方差
        counts = []  # 各客户端的样本数

        # 收集该类别的所有客户端统计量
        for client_id, client_stats in self.local_statistics_buffer.items():
            if class_idx in client_stats['means']:
                means.append(client_stats['means'][class_idx])
                covs.append(client_stats['covs'][class_idx])
                counts.append(client_stats['counts'][class_idx])

        total_count = sum(counts)

        # 第一步：计算全局均值
        aggregated_mean = np.zeros(embedding_dim)
        for mean, count in zip(means, counts):
            aggregated_mean += count * mean
        aggregated_mean /= total_count

        # 第二步 + 第三步：计算全局协方差
        aggregated_cov = np.zeros((embedding_dim, embedding_dim))

        # 加权局部协方差
        for cov, count in zip(covs, counts):
            aggregated_cov += count * cov

        # 客户端间方差
        for mean, count in zip(means, counts):
            diff = mean - aggregated_mean
            aggregated_cov += count * np.outer(diff, diff)

        aggregated_cov /= total_count

        self.global_cov_matrices[class_idx] = aggregated_cov
```

### 2.3.4 全局原型计算

```python
def _compute_global_prototypes(self):
    """Compute global prototypes (weighted average of local means)"""

    for class_idx in all_classes:
        means = []
        counts = []

        for client_id, client_stats in self.local_statistics_buffer.items():
            if class_idx in client_stats['means']:
                means.append(client_stats['means'][class_idx])
                counts.append(client_stats['counts'][class_idx])

        total_count = sum(counts)

        # 加权平均
        global_mean = np.zeros(embedding_dim)
        for mean, count in zip(means, counts):
            global_mean += count * mean
        global_mean /= total_count

        self.global_prototypes[class_idx] = global_mean
```

---

## 2.4 高斯特征增强

### 2.4.1 多元高斯分布采样

GGEUR使用全局协方差矩阵从多元高斯分布中采样生成虚拟特征。

**采样公式：**

$$\mathbf{f}_{virtual} \sim \mathcal{N}(\boldsymbol{\mu}_c^{global}, \boldsymbol{\Sigma}_c^{global})$$

**协方差矩阵正定性处理：**

```python
def _nearest_pos_def(self, cov_matrix):
    """Ensure covariance matrix is positive definite"""
    dim = cov_matrix.shape[0]

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 将负特征值裁剪为0
    eigenvalues[eigenvalues < 0] = 0

    # 重建协方差矩阵
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def _generate_samples(self, mean, cov_matrix, num_samples):
    """Generate samples from Gaussian distribution"""
    # 确保正定性
    cov_pd = self._nearest_pos_def(cov_matrix)

    # 添加小量正则化提高数值稳定性
    cov_pd += 1e-6 * np.eye(cov_pd.shape[0])

    # 采样
    samples = np.random.multivariate_normal(
        mean=mean,
        cov=cov_pd,
        size=num_samples
    )
    return samples
```

### 2.4.2 跨客户端原型增强

除了使用全局协方差采样，GGEUR还利用其他客户端的类原型进行增强。

```python
def _perform_augmentation(self):
    """Perform GGEUR feature augmentation"""

    for class_idx in all_classes:
        class_features = []

        # 1. 原始特征（本地数据）
        if class_idx in self.local_features:
            class_features.append(self.local_features[class_idx])

        # 2. 基于全局协方差的增强
        if class_idx in self.global_cov_matrices:
            cov_matrix = self.global_cov_matrices[class_idx]

            # 使用本地均值（如有）或全局原型作为采样中心
            if class_idx in self.local_means:
                center = self.local_means[class_idx]
            else:
                center = self.global_prototypes.get(class_idx)

            if center is not None:
                generated = self._generate_samples(
                    mean=center,
                    cov_matrix=cov_matrix,
                    num_samples=num_per_sample
                )
                class_features.append(generated)

        # 3. 跨客户端原型增强
        if self.other_prototypes and class_idx in self.other_prototypes:
            for prototype in self.other_prototypes[class_idx]:
                # 以其他客户端的原型为中心采样
                generated = self._generate_samples(
                    mean=prototype,
                    cov_matrix=cov_matrix,
                    num_samples=num_per_prototype
                )
                class_features.append(generated)
```

### 2.4.3 目标样本数量平衡策略

GGEUR通过配置目标样本数来平衡各类别的数据量。

```yaml
ggeur:
  target_size_per_class: 50      # 每类目标样本数
  num_generated_per_sample: 50   # 每个原始样本生成的增强样本
  num_generated_per_prototype: 50 # 每个跨域原型生成的样本
```

**平衡逻辑：**

```python
# 计算需要生成的样本数
current_size = len(original_features)
target_size = self.ggeur_cfg.target_size_per_class

if current_size < target_size:
    # 需要增强
    num_to_generate = target_size - current_size
else:
    # 已足够，可选择性生成少量增强样本
    num_to_generate = num_per_sample
```

---

## 2.5 分类器训练

### 2.5.1 MLP分类器结构

GGEUR使用轻量级MLP作为分类器，在特征空间上进行训练。

**默认结构（无隐藏层）：**

```
输入特征 (512维)
      ↓
  ┌────────────┐
  │  Linear    │  512 → num_classes
  └────────────┘
      ↓
输出 logits (num_classes维)
```

**带隐藏层结构：**

```
输入特征 (512维)
      ↓
  ┌────────────┐
  │  Linear    │  512 → hidden_dim
  ├────────────┤
  │   ReLU     │
  ├────────────┤
  │  Dropout   │  p=mlp_dropout
  ├────────────┤
  │  Linear    │  hidden_dim → num_classes
  └────────────┘
      ↓
输出 logits (num_classes维)
```

**代码实现：**

```python
def _build_mlp_classifier(self):
    """Build MLP classifier for augmented features"""
    input_dim = self.ggeur_cfg.embedding_dim
    hidden_dim = self.ggeur_cfg.mlp_hidden_dim
    num_classes = self._cfg.model.num_classes

    if hidden_dim > 0:
        self.mlp_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.ggeur_cfg.mlp_dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    else:
        self.mlp_classifier = nn.Linear(input_dim, num_classes)
```

### 2.5.2 FedAvg聚合策略

MLP分类器通过标准FedAvg算法进行联邦训练。

**客户端本地训练：**

```python
def _train_on_augmented_data(self):
    """Train MLP on augmented features"""
    self.mlp_classifier.train()

    optimizer = torch.optim.Adam(
        self.mlp_classifier.parameters(),
        lr=self._cfg.train.optimizer.lr
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(self._cfg.train.local_update_steps):
        for features, labels in self.augmented_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.mlp_classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return self.mlp_classifier.state_dict()
```

**服务器聚合：**

```python
def _aggregate_mlp_models(self, models):
    """Aggregate MLP models using FedAvg"""
    aggregated = {}
    total_samples = sum(sample_sizes)

    for key in models[0].keys():
        aggregated[key] = sum(
            model[key] * (size / total_samples)
            for model, size in zip(models, sample_sizes)
        )

    return aggregated
```

---

## 2.6 算法伪代码总结

```
Algorithm: GGEUR (Gaussian Geometry-guided Feature Expansion)

Input:
  - K clients with local datasets D_k
  - CLIP/CNN feature extractor F
  - Target classes C

Output:
  - Global MLP classifier M

1. Round 0 - Statistics Collection:
   For each client k in parallel:
     a. Extract features: f_i = F(x_i) for all (x_i, y_i) in D_k
     b. For each class c in D_k:
        - Compute mean: μ_c^k = mean(f_i | y_i = c)
        - Compute covariance: Σ_c^k = cov(f_i | y_i = c)
        - Count samples: n_c^k
     c. Upload (μ_c^k, Σ_c^k, n_c^k) to server

2. Server Aggregation:
   For each class c:
     a. Compute global mean: μ_c = Σ(n_c^k * μ_c^k) / Σ(n_c^k)
     b. Aggregate covariance using Parallel Axis Theorem:
        Σ_c = Σ(n_c^k * Σ_c^k) / N_c + Σ(n_c^k * (μ_c^k - μ_c)(μ_c^k - μ_c)^T) / N_c
   Broadcast (Σ_c, μ_c, other_prototypes) to all clients

3. Feature Augmentation (at each client):
   For each class c:
     a. Keep original features
     b. Generate virtual features: f_new ~ N(μ_c, Σ_c)
     c. Generate from other prototypes: f_cross ~ N(μ_c^other, Σ_c)
   Create augmented dataset D_aug

4. Rounds 1+ - Federated Training:
   Repeat until convergence:
     For each client k in parallel:
       a. Train MLP on D_aug for local_epochs
       b. Upload model parameters to server
     Server aggregates using FedAvg
     Server broadcasts global model

5. Return global MLP classifier M
```

---

**第二章完成。** 下一章将详细介绍GGEUR在FederatedScope框架中的具体实现架构和代码结构。

---

# 第三章：框架实现架构

本章详细介绍GGEUR在FederatedScope框架中的具体实现，包括文件结构、各模块职责和类继承关系。

## 3.1 文件结构总览

GGEUR的实现分布在FederatedScope的多个目录中，遵循框架的插件化设计模式。

### 3.1.1 完整文件树

```
FederatedScope/
├── federatedscope/
│   ├── contrib/                          # 贡献插件目录
│   │   ├── worker/                       # Worker实现
│   │   │   ├── ggeur_client.py          # ⭐ 客户端核心逻辑
│   │   │   └── ggeur_server.py          # ⭐ 服务器核心逻辑
│   │   ├── trainer/
│   │   │   └── ggeur_trainer.py         # 训练器实现
│   │   ├── model/
│   │   │   ├── ggeur_mlp.py             # MLP分类器定义
│   │   │   ├── ggeur_cnn.py             # CNN骨干网络
│   │   │   └── ggeur_cnn_extractor.py   # CNN特征提取器
│   │   └── data/
│   │       └── ggeur_data.py            # ⭐ 数据加载与LDS划分
│   │
│   └── core/
│       └── configs/
│           └── cfg_ggeur.py             # ⭐ 配置项定义
│
└── scripts/
    └── example_configs/
        ├── ggeur_officehome.yaml        # Office-Home基础配置
        ├── ggeur_officehome_lds.yaml    # LDS非IID配置
        ├── ggeur_pacs.yaml              # PACS数据集配置
        ├── ggeur_cnn_pacs.yaml          # CNN模式配置
        ├── ggeur_cnn_convnext.yaml      # ConvNeXt骨干配置
        ├── ggeur_separated_training.yaml # 分阶段训练配置
        └── ...                          # 其他配置文件
```

### 3.1.2 核心文件说明

| 文件 | 行数 | 主要职责 |
|------|------|----------|
| `ggeur_client.py` | ~1000 | 客户端完整逻辑：特征提取、统计计算、增强、训练 |
| `ggeur_server.py` | ~800 | 服务器逻辑：统计聚合、模型聚合、测试评估 |
| `ggeur_data.py` | ~400 | 数据集加载、LDS划分、域分配 |
| `cfg_ggeur.py` | ~150 | 所有GGEUR配置项定义 |
| `ggeur_trainer.py` | ~200 | 特征级训练器封装 |
| `ggeur_mlp.py` | ~50 | MLP模型定义 |
| `ggeur_cnn.py` | ~150 | CNN特征对齐模型 |
| `ggeur_cnn_extractor.py` | ~100 | CNN特征提取器 |

---

## 3.2 客户端实现详解（ggeur_client.py）

### 3.2.1 类结构与继承关系

```python
from federatedscope.core.workers import Client

class GGEURClient(Client):
    """
    GGEUR客户端，继承自FederatedScope基础Client类。

    扩展功能：
    1. CLIP/CNN特征提取
    2. 局部统计量计算
    3. 高斯特征增强
    4. MLP分类器训练
    5. （可选）CNN知识蒸馏/特征对齐
    """
```

**继承层次：**

```
federatedscope.core.workers.base_client.BaseClient
                    │
                    ▼
federatedscope.core.workers.client.Client
                    │
                    ▼
federatedscope.contrib.worker.ggeur_client.GGEURClient
```

### 3.2.2 核心属性

```python
class GGEURClient(Client):
    def __init__(self, ...):
        super().__init__(...)

        # ===== 特征提取器 =====
        self.feature_extractor_type = 'clip'  # 或 'cnn'
        self.clip_model = None                # CLIP模型实例
        self.clip_preprocess = None           # CLIP预处理函数
        self.cnn_extractor = None             # CNN特征提取器

        # ===== 局部特征和统计量 =====
        self.local_features = {}    # {class_idx: features array}
        self.local_means = {}       # {class_idx: mean vector}
        self.local_covs = {}        # {class_idx: covariance matrix}
        self.local_counts = {}      # {class_idx: sample count}

        # ===== 全局信息（从服务器接收）=====
        self.global_cov_matrices = None   # 全局协方差
        self.other_prototypes = None      # 其他客户端的类原型
        self.global_prototypes = None     # 全局类原型

        # ===== 增强后的数据 =====
        self.augmented_features = None
        self.augmented_labels = None
        self.augmented_loader = None

        # ===== 模型 =====
        self.mlp_classifier = None  # MLP分类器
        self.cnn_model = None       # CNN模型（可选）

        # ===== 状态跟踪 =====
        self.statistics_uploaded = False
        self.augmentation_done = False
```

### 3.2.3 核心方法详解

#### `_extract_features()` - 特征提取

```python
def _extract_features(self):
    """
    从本地图像提取CLIP/CNN特征。

    流程：
    1. 检查缓存，如有则直接加载
    2. 加载特征提取器（CLIP或CNN）
    3. 遍历数据集，批量提取特征
    4. 按类别组织特征
    5. 保存到缓存（如启用）

    输出：
    - self.local_features: {class_idx: np.array of shape (n, dim)}
    """

    # 检查缓存
    cache_path = self._get_feature_cache_path()
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        self._load_cached_features(cached)
        return

    # 加载特征提取器
    self._load_feature_extractor()

    # 提取特征
    self.local_features = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in self.data['train']:
            images = images.to(self.device)

            if self.feature_extractor_type == 'clip':
                features = self.clip_model.encode_image(images)
            else:
                features = self.cnn_extractor(images)

            features = features.cpu().numpy()

            for feat, label in zip(features, labels):
                self.local_features[int(label)].append(feat)

    # 转换为numpy数组
    for class_idx in self.local_features:
        self.local_features[class_idx] = np.array(self.local_features[class_idx])

    # 保存缓存
    if cache_path:
        self._save_feature_cache(cache_path)
```

#### `_compute_local_statistics()` - 统计量计算

```python
def _compute_local_statistics(self):
    """
    计算每个类别的均值和协方差。

    输入：self.local_features
    输出：
    - self.local_means: {class_idx: mean vector}
    - self.local_covs: {class_idx: covariance matrix}
    - self.local_counts: {class_idx: int}
    """
    self.local_means = {}
    self.local_covs = {}
    self.local_counts = {}

    for class_idx, features in self.local_features.items():
        if features.shape[0] == 0:
            continue

        n = features.shape[0]

        # 均值
        mean = np.mean(features, axis=0)

        # 协方差
        centered = features - mean
        cov = (1.0 / n) * np.dot(centered.T, centered)

        self.local_means[class_idx] = mean
        self.local_covs[class_idx] = cov
        self.local_counts[class_idx] = n
```

#### `_perform_augmentation()` - 特征增强

```python
def _perform_augmentation(self):
    """
    使用全局协方差进行高斯特征增强。

    增强策略：
    1. 保留原始特征
    2. 基于全局协方差从本地均值采样
    3. 基于其他客户端原型采样（跨域增强）

    输出：
    - self.augmented_features: np.array
    - self.augmented_labels: np.array
    - self.augmented_loader: DataLoader
    """
    all_features = []
    all_labels = []

    for class_idx in all_classes:
        class_features = []

        # 1. 原始特征
        if class_idx in self.local_features:
            class_features.append(self.local_features[class_idx])

        # 2. 基于全局协方差增强
        if class_idx in self.global_cov_matrices:
            cov = self.global_cov_matrices[class_idx]

            if class_idx in self.local_means:
                center = self.local_means[class_idx]
            else:
                center = self.global_prototypes.get(class_idx)

            if center is not None:
                generated = self._generate_samples(center, cov, num_per_sample)
                class_features.append(generated)

        # 3. 跨客户端原型增强
        if class_idx in self.other_prototypes:
            for prototype in self.other_prototypes[class_idx]:
                generated = self._generate_samples(prototype, cov, num_per_prototype)
                class_features.append(generated)

        # 合并
        if class_features:
            combined = np.vstack(class_features)
            all_features.append(combined)
            all_labels.append(np.full(len(combined), class_idx))

    self.augmented_features = np.vstack(all_features)
    self.augmented_labels = np.concatenate(all_labels)

    # 创建DataLoader
    dataset = AugmentedFeatureDataset(self.augmented_features, self.augmented_labels)
    self.augmented_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    self.augmentation_done = True
```

### 3.2.4 消息处理器

```python
def _register_default_handlers(self):
    """注册消息处理器"""
    super()._register_default_handlers()

    # 处理服务器返回的全局协方差
    self.register_handlers('global_covariances',
                           self.callback_for_global_covariances)

def callback_for_global_covariances(self, message: Message):
    """处理全局协方差消息"""
    content = message.content

    # 解析消息内容
    self.global_cov_matrices = content.get('cov_matrices', {})
    self.other_prototypes = content.get('other_prototypes', {})
    self.global_prototypes = content.get('global_prototypes', {})

    # 执行增强
    self._perform_augmentation()

    # 构建分类器
    self._build_mlp_classifier()

    # 通知服务器增强完成
    self.comm_manager.send(Message(
        msg_type='augmentation_ready',
        sender=self.ID,
        receiver=[self.server_id],
        state=self.state,
        content=None
    ))
```

---

## 3.3 服务器端实现详解（ggeur_server.py）

### 3.3.1 类结构与继承关系

```python
from federatedscope.core.workers import Server

class GGEURServer(Server):
    """
    GGEUR服务器，继承自FederatedScope基础Server类。

    扩展功能：
    1. 收集客户端统计量
    2. 平行轴定理聚合协方差
    3. 广播全局协方差
    4. FedAvg模型聚合
    5. 多域测试评估
    """
```

### 3.3.2 核心属性

```python
class GGEURServer(Server):
    def __init__(self, ...):
        super().__init__(...)

        # ===== 统计量收集 =====
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # ===== 聚合结果 =====
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}
        self.all_prototypes = {}       # {client_id: {class_idx: prototype}}
        self.global_prototypes = {}    # {class_idx: mean_vector}

        # ===== 模型 =====
        self.global_mlp = None         # 全局MLP分类器
        self.global_cnn = None         # 全局CNN（可选）

        # ===== 客户端状态跟踪 =====
        self.augmentation_ready_clients = set()

        # ===== 测试数据 =====
        self.test_features = {}        # {domain: features}
        self.test_labels = {}          # {domain: labels}

        # ===== 性能跟踪 =====
        self.best_avg_accuracy = 0.0
        self.best_model_state = None
        self.test_accuracies_history = {}
```

### 3.3.3 核心方法详解

#### `_aggregate_covariances()` - 协方差聚合

```python
def _aggregate_covariances(self):
    """
    使用平行轴定理聚合协方差矩阵。

    数学公式：
    Σ_global = (1/N) * Σ(n_k * Σ_k) + (1/N) * Σ(n_k * (μ_k - μ_global)(μ_k - μ_global)^T)

    输入：self.local_statistics_buffer
    输出：self.global_cov_matrices
    """
    # 收集所有类别
    all_classes = set()
    for client_stats in self.local_statistics_buffer.values():
        all_classes.update(client_stats['means'].keys())

    embedding_dim = self.ggeur_cfg.embedding_dim

    for class_idx in all_classes:
        class_idx = int(class_idx)
        means, covs, counts = [], [], []

        # 收集该类的所有客户端统计量
        for client_id, client_stats in self.local_statistics_buffer.items():
            if class_idx in client_stats['means']:
                means.append(client_stats['means'][class_idx])
                covs.append(client_stats['covs'][class_idx])
                counts.append(client_stats['counts'][class_idx])

        if not counts:
            self.global_cov_matrices[class_idx] = np.eye(embedding_dim) * 0.01
            continue

        total_count = sum(counts)

        # 计算全局均值
        aggregated_mean = np.zeros(embedding_dim)
        for mean, count in zip(means, counts):
            aggregated_mean += count * mean
        aggregated_mean /= total_count

        # 计算全局协方差
        aggregated_cov = np.zeros((embedding_dim, embedding_dim))

        # 第一项：加权局部协方差
        for cov, count in zip(covs, counts):
            aggregated_cov += count * cov

        # 第二项：客户端间方差
        for mean, count in zip(means, counts):
            diff = mean - aggregated_mean
            aggregated_cov += count * np.outer(diff, diff)

        aggregated_cov /= total_count

        self.global_cov_matrices[class_idx] = aggregated_cov
```

#### `_prepare_other_prototypes()` - 准备跨域原型

```python
def _prepare_other_prototypes(self):
    """
    为每个客户端准备其他客户端的原型。

    返回：{client_id: {class_idx: [prototypes from other clients]}}
    """
    other_prototypes = {}

    for target_client in self.all_prototypes.keys():
        other_prototypes[target_client] = {}

        for class_idx in all_classes:
            prototypes = []

            for source_client, source_protos in self.all_prototypes.items():
                if source_client != target_client:
                    if class_idx in source_protos:
                        prototypes.append(source_protos[class_idx])

            if prototypes:
                other_prototypes[target_client][class_idx] = prototypes

    return other_prototypes
```

#### `_broadcast_global_covariances()` - 广播全局协方差

```python
def _broadcast_global_covariances(self, other_prototypes):
    """
    向所有客户端广播全局协方差和原型。

    消息内容：
    - cov_matrices: 全局协方差矩阵
    - other_prototypes: 该客户端可用的其他客户端原型
    - global_prototypes: 全局类原型
    """
    for client_id in range(1, self._client_num + 1):
        content = {
            'cov_matrices': self.global_cov_matrices,
            'other_prototypes': other_prototypes.get(client_id, {}),
            'global_prototypes': self.global_prototypes
        }

        self.comm_manager.send(Message(
            msg_type='global_covariances',
            sender=self.ID,
            receiver=[client_id],
            state=self.state,
            content=content
        ))
```

### 3.3.4 消息处理器

```python
def _register_default_handlers(self):
    """注册消息处理器"""
    super()._register_default_handlers()

    # 处理客户端统计量
    self.register_handlers('local_statistics',
                           self.callback_for_local_statistics)

    # 处理增强完成信号
    self.register_handlers('augmentation_ready',
                           self.callback_for_augmentation_ready)

def callback_for_local_statistics(self, message: Message):
    """处理客户端统计量"""
    client_id = message.sender
    content = message.content

    # 缓存统计量
    self.local_statistics_buffer[client_id] = {
        'means': content['means'],
        'covs': content['covs'],
        'counts': content['counts']
    }

    # 缓存原型
    self.all_prototypes[client_id] = content.get('prototypes', {})

    # 检查是否收集完毕
    if len(self.local_statistics_buffer) >= self._client_num:
        self._aggregate_covariances()
        self._compute_global_prototypes()
        other_prototypes = self._prepare_other_prototypes()
        self._build_global_mlp(num_classes)
        self._broadcast_global_covariances(other_prototypes)
        self.statistics_collected = True

def callback_for_augmentation_ready(self, message: Message):
    """处理增强完成信号"""
    client_id = message.sender
    self.augmentation_ready_clients.add(client_id)

    # 所有客户端就绪后开始训练
    if len(self.augmentation_ready_clients) >= self._client_num:
        self.state = 1  # 进入训练轮次
        self._start_training_round()
```

---

## 3.4 训练器实现（ggeur_trainer.py）

### 3.4.1 EmbeddingDataset包装类

```python
class EmbeddingDataset(Dataset):
    """
    用于包装CLIP特征嵌入的数据集类。

    与标准图像数据集的区别：
    - 输入是预提取的特征向量，而非原始图像
    - 无需图像预处理和数据增强
    """

    def __init__(self, embeddings, labels):
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings).float()
        else:
            self.embeddings = embeddings.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
```

### 3.4.2 GGEURTrainer类

```python
class GGEURTrainer(GeneralTorchTrainer):
    """
    GGEUR专用训练器。

    特点：
    1. 工作在特征空间而非像素空间
    2. 训练轻量级MLP分类器
    3. 支持增强数据注入
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        self.mlp_classifier = None
        self.augmented_embeddings = None
        self.augmented_labels = None
        self.augmented_loader = None

    def _build_mlp(self, num_classes):
        """构建MLP分类器"""
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim
        dropout = self.ggeur_cfg.mlp_dropout

        if hidden_dim > 0:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.mlp_classifier = nn.Linear(input_dim, num_classes)

        self.mlp_classifier = self.mlp_classifier.to(self.device)

    def setup_augmented_data(self, embeddings, labels):
        """注入增强后的特征数据"""
        self.augmented_embeddings = embeddings
        self.augmented_labels = labels

        dataset = EmbeddingDataset(embeddings, labels)
        self.augmented_loader = DataLoader(
            dataset,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=True
        )

    def train(self):
        """在增强特征上训练MLP"""
        self.mlp_classifier.train()

        optimizer = torch.optim.Adam(
            self.mlp_classifier.parameters(),
            lr=self.cfg.train.optimizer.lr
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.cfg.train.local_update_steps):
            for embeddings, labels in self.augmented_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.mlp_classifier(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_model_para()
```

---

## 3.5 模型定义

### 3.5.1 MLP分类器（ggeur_mlp.py）

```python
class GGEUR_MLP(nn.Module):
    """
    GGEUR使用的MLP分类器。

    架构选项：
    1. 简单线性层（hidden_dim=0）
    2. 单隐藏层MLP（hidden_dim>0）
    """

    def __init__(self, input_dim=512, hidden_dim=0, num_classes=65, dropout=0.0):
        super().__init__()

        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)
```

### 3.5.2 CNN特征对齐模型（ggeur_cnn.py）

```python
class GGEUR_CNN_FeatureAlign(nn.Module):
    """
    用于特征对齐训练的CNN模型。

    训练目标：
    1. 分类正确（CrossEntropy Loss）
    2. 特征对齐到CLIP空间（MSE Loss）
    """

    def __init__(self, cnn_backbone='resnet18', clip_dim=512, num_classes=65,
                 pretrained=False):
        super().__init__()

        # CNN骨干网络
        self.backbone = self._build_backbone(cnn_backbone, pretrained)

        # 特征映射层（将CNN特征映射到CLIP维度）
        backbone_dim = self._get_backbone_dim(cnn_backbone)
        self.feature_mapper = nn.Linear(backbone_dim, clip_dim)

        # 分类器
        self.classifier = nn.Linear(clip_dim, num_classes)

    def forward(self, x, return_features=False):
        # CNN特征提取
        cnn_features = self.backbone(x)

        # 映射到CLIP空间
        aligned_features = self.feature_mapper(cnn_features)

        # 分类
        logits = self.classifier(aligned_features)

        if return_features:
            return logits, aligned_features
        return logits
```

### 3.5.3 CNN特征提取器（ggeur_cnn_extractor.py）

```python
class CNNFeatureExtractor(nn.Module):
    """
    预训练CNN特征提取器。

    支持的骨干网络：
    - ConvNeXt: convnext_tiny, convnext_base, convnext_large
    - ResNet: resnet18, resnet50, resnet101
    - EfficientNet: efficientnet_b0, efficientnet_b4
    """

    def __init__(self, model_name='convnext_base', pretrained=True, freeze=True):
        super().__init__()

        self.model_name = model_name
        self.feature_dim = self._get_feature_dim(model_name)

        # 加载预训练模型
        self.backbone = self._build_backbone(model_name, pretrained)

        # 移除分类头
        self._remove_classifier()

        # 冻结参数
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """提取特征向量"""
        features = self.backbone(x)
        return features.flatten(1)

    def get_feature_dim(self):
        return self.feature_dim
```

---

## 3.6 数据模块（ggeur_data.py）

数据模块的详细实现将在第六章"数据集与划分方式"中详细介绍。

---

## 3.7 配置模块（cfg_ggeur.py）

配置模块的详细说明将在第五章"配置项详解"中完整介绍。

---

**第三章完成。** 下一章将详细介绍GGEUR的消息通信协议和各阶段的数据流转。

---

# 第四章：消息通信协议

本章详细介绍GGEUR在联邦学习过程中的消息通信机制，包括各阶段的消息类型、数据格式和时序关系。

## 4.1 通信流程图

### 4.1.1 完整时序图

```
┌─────────┐                              ┌─────────┐                              ┌─────────┐
│ Client1 │                              │ Server  │                              │ Client2 │
└────┬────┘                              └────┬────┘                              └────┬────┘
     │                                        │                                        │
     │══════════════════════════ Round 0: 统计量收集 ════════════════════════════════│
     │                                        │                                        │
     │  1. 提取特征                           │                           1. 提取特征  │
     │  2. 计算统计量                         │                           2. 计算统计量│
     │                                        │                                        │
     │─────── local_statistics ──────────────>│<────── local_statistics ───────────────│
     │        (means, covs, counts)           │        (means, covs, counts)           │
     │                                        │                                        │
     │                                        │  3. 聚合协方差                          │
     │                                        │  4. 计算全局原型                        │
     │                                        │  5. 准备跨域原型                        │
     │                                        │                                        │
     │<────── global_covariances ─────────────│─────── global_covariances ────────────>│
     │        (cov_matrices,                  │        (cov_matrices,                  │
     │         other_prototypes,              │         other_prototypes,              │
     │         global_prototypes)             │         global_prototypes)             │
     │                                        │                                        │
     │  6. 执行特征增强                       │                           6. 执行增强  │
     │  7. 构建分类器                         │                           7. 构建分类器│
     │                                        │                                        │
     │─────── augmentation_ready ────────────>│<────── augmentation_ready ─────────────│
     │                                        │                                        │
     │════════════════════════════ Round 1+: 模型训练 ═══════════════════════════════│
     │                                        │                                        │
     │<────────── model_para ─────────────────│──────────── model_para ───────────────>│
     │            (global_mlp)                │              (global_mlp)              │
     │                                        │                                        │
     │  8. 本地训练MLP                        │                           8. 本地训练  │
     │                                        │                                        │
     │─────────── model_para ────────────────>│<─────────── model_para ────────────────│
     │            (local_mlp, sample_size)    │              (local_mlp, sample_size)  │
     │                                        │                                        │
     │                                        │  9. FedAvg聚合                         │
     │                                        │  10. 多域测试评估                      │
     │                                        │                                        │
     │              ... (重复 Round 1+ 直到收敛) ...                                    │
     │                                        │                                        │
```

### 4.1.2 状态机

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
┌─────────────┐   start   ┌─────────────┐   stats_done   ┌────┴────────┐
│    INIT     │──────────>│  ROUND_0    │───────────────>│  AUGMENT    │
│  (state=0)  │           │ 统计收集    │                │  特征增强   │
└─────────────┘           └─────────────┘                └──────┬──────┘
                                                                │
                                                           aug_ready
                                                                │
                    ┌─────────────────────────────────────────┐ │
                    │                                         │ ▼
                    ▼         round_done              ┌───────┴──────┐
               ┌────────────┐<────────────────────────│   TRAINING   │
               │ NEXT_ROUND │                         │   模型训练   │
               └─────┬──────┘                         └──────────────┘
                     │                                       ▲
                     │            not_converged              │
                     └───────────────────────────────────────┘
                     │
                     │ converged
                     ▼
               ┌─────────────┐
               │    DONE     │
               │   训练完成  │
               └─────────────┘
```

---

## 4.2 Round 0：统计量收集阶段

Round 0是GGEUR特有的统计量收集轮，与标准FedAvg不同。

### 4.2.1 客户端上传消息格式

**消息类型：** `local_statistics`

**发送方向：** Client → Server

**消息结构：**

```python
Message(
    msg_type='local_statistics',
    sender=client_id,          # 客户端ID，如 1, 2, 3, ...
    receiver=[server_id],      # 通常为 [0]
    state=0,                   # Round 0
    content={
        'client_id': int,                     # 客户端标识
        'means': {                            # 各类别均值
            class_idx: np.ndarray,            # shape: (embedding_dim,)
            # 例如: {0: array([0.1, 0.2, ...]), 1: array([...]), ...}
        },
        'covs': {                             # 各类别协方差矩阵
            class_idx: np.ndarray,            # shape: (embedding_dim, embedding_dim)
        },
        'counts': {                           # 各类别样本数
            class_idx: int,
        },
        'prototypes': {                       # 类原型（与means相同）
            class_idx: np.ndarray,
        }
    }
)
```

**数据示例（Office-Home，4个客户端，65类）：**

```python
# 客户端1（Art域）上传的统计量
{
    'client_id': 1,
    'means': {
        0: np.array([0.123, -0.456, ...]),   # 类0的均值，512维
        1: np.array([0.234, -0.567, ...]),   # 类1的均值
        # ... 假设该域有50个类
    },
    'covs': {
        0: np.array([[0.01, 0.002, ...],     # 类0的协方差，512x512
                     [0.002, 0.015, ...],
                     ...]),
        # ...
    },
    'counts': {
        0: 45,   # 类0有45个样本
        1: 38,   # 类1有38个样本
        # ...
    },
    'prototypes': {...}  # 与means相同
}
```

### 4.2.2 服务器下发消息格式

**消息类型：** `global_covariances`

**发送方向：** Server → Client（每个客户端收到定制消息）

**消息结构：**

```python
Message(
    msg_type='global_covariances',
    sender=server_id,          # 通常为 0
    receiver=[client_id],      # 单个客户端
    state=0,                   # Round 0
    content={
        'cov_matrices': {                     # 全局协方差矩阵
            class_idx: np.ndarray,            # 平行轴定理聚合后的结果
        },
        'other_prototypes': {                 # 其他客户端的原型
            class_idx: [                      # 列表，包含其他客户端的原型
                np.ndarray,                   # 客户端A的原型
                np.ndarray,                   # 客户端B的原型
                # ...
            ],
        },
        'global_prototypes': {                # 全局类原型
            class_idx: np.ndarray,            # 加权平均的全局原型
        }
    }
)
```

**定制化说明：**

每个客户端收到的`other_prototypes`不同，排除了自己的原型：

```python
# 发送给客户端1的消息
'other_prototypes': {
    0: [client2_proto_0, client3_proto_0, client4_proto_0],  # 不含client1
    1: [client2_proto_1, client3_proto_1],                   # client4可能没有类1
    # ...
}

# 发送给客户端2的消息
'other_prototypes': {
    0: [client1_proto_0, client3_proto_0, client4_proto_0],  # 不含client2
    # ...
}
```

---

## 4.3 增强完成信号

### 4.3.1 消息格式

**消息类型：** `augmentation_ready`

**发送方向：** Client → Server

**消息结构：**

```python
Message(
    msg_type='augmentation_ready',
    sender=client_id,
    receiver=[server_id],
    state=0,
    content=None  # 无负载，仅为信号
)
```

**触发条件：**

客户端完成以下步骤后发送：
1. 接收到`global_covariances`消息
2. 执行`_perform_augmentation()`生成增强特征
3. 构建MLP分类器

**服务器处理：**

```python
def callback_for_augmentation_ready(self, message: Message):
    client_id = message.sender
    self.augmentation_ready_clients.add(client_id)

    # 所有客户端就绪后开始训练
    if len(self.augmentation_ready_clients) >= self._client_num:
        self.state = 1  # 从Round 0进入Round 1
        self._start_training_round()
```

---

## 4.4 Round 1+：模型训练阶段

### 4.4.1 模型参数下发

**消息类型：** `model_para`

**发送方向：** Server → Client

**消息结构（标准模式）：**

```python
Message(
    msg_type='model_para',
    sender=server_id,
    receiver=[client_id],
    state=round_num,           # 1, 2, 3, ...
    timestamp=current_time,
    content=OrderedDict({      # MLP模型参数
        'weight': torch.Tensor,  # shape: (num_classes, embedding_dim)
        'bias': torch.Tensor,    # shape: (num_classes,)
    })
)
```

**消息结构（CNN双模型模式）：**

```python
Message(
    msg_type='model_para',
    sender=server_id,
    receiver=[client_id],
    state=round_num,
    content={
        'mlp': OrderedDict({...}),    # MLP参数
        'cnn': OrderedDict({...}),    # CNN参数（如果启用）
    }
)
```

### 4.4.2 模型参数上传

**消息类型：** `model_para`

**发送方向：** Client → Server

**消息结构：**

```python
Message(
    msg_type='model_para',
    sender=client_id,
    receiver=[server_id],
    state=round_num,
    timestamp=current_time,
    content=(
        sample_size,           # int: 本地训练样本数（用于加权聚合）
        model_para,            # OrderedDict: 模型参数
    )
)
```

**多模型上传：**

```python
# 当启用CNN训练时
content = {
    'mlp': (mlp_sample_size, mlp_model_para),
    'cnn': (cnn_sample_size, cnn_model_para),
}
```

---

## 4.5 消息数据量分析

### 4.5.1 Round 0统计量

| 组件 | 维度 | 数据类型 | 单类大小 | 65类总大小 |
|------|------|----------|----------|------------|
| mean | (512,) | float64 | 4KB | 260KB |
| cov | (512, 512) | float64 | 2MB | 130MB |
| count | scalar | int | 8B | 520B |

**单客户端上传量：** ~130MB（主要是协方差矩阵）

**优化建议：**
- 使用float32代替float64可减半
- 稀疏矩阵存储可进一步压缩
- 仅传输对角化后的特征值（PCA压缩）

### 4.5.2 Round 1+模型参数

| 模型 | 参数量 | 大小 |
|------|--------|------|
| MLP (无隐藏层) | 512×65 + 65 = 33,345 | ~130KB |
| MLP (hidden=256) | 512×256 + 256×65 + 256 + 65 = 147,969 | ~580KB |
| CNN (ResNet18) | ~11M | ~44MB |

---

## 4.6 消息注册机制

### 4.6.1 客户端消息注册

```python
class GGEURClient(Client):
    def _register_default_handlers(self):
        # 继承父类的默认处理器
        super()._register_default_handlers()

        # 注册GGEUR特有的消息处理器
        self.register_handlers(
            'global_covariances',              # 消息类型
            self.callback_for_global_covariances  # 回调函数
        )

    def callback_for_global_covariances(self, message: Message):
        """处理全局协方差消息"""
        content = message.content

        # 解析并存储
        self.global_cov_matrices = content['cov_matrices']
        self.other_prototypes = content['other_prototypes']
        self.global_prototypes = content['global_prototypes']

        # 执行后续操作
        self._perform_augmentation()
        self._build_mlp_classifier()

        # 通知服务器
        self.comm_manager.send(Message(
            msg_type='augmentation_ready',
            sender=self.ID,
            receiver=[self.server_id],
            state=self.state,
            content=None
        ))
```

### 4.6.2 服务器消息注册

```python
class GGEURServer(Server):
    def _register_default_handlers(self):
        super()._register_default_handlers()

        # 统计量处理器
        self.register_handlers(
            'local_statistics',
            self.callback_for_local_statistics
        )

        # 增强完成信号处理器
        self.register_handlers(
            'augmentation_ready',
            self.callback_for_augmentation_ready
        )
```

---

## 4.7 异常处理与重试机制

### 4.7.1 消息超时处理

```python
# FederatedScope内置的超时机制
cfg.distribute.timeout = 3600  # 秒

# 客户端未响应时的处理
if timeout_occurred:
    logger.warning(f"Client {client_id} timeout, skipping this round")
    # 继续使用上一轮的模型参数
```

### 4.7.2 客户端掉线处理

```python
def callback_for_local_statistics(self, message: Message):
    # 记录收到的统计量
    self.local_statistics_buffer[message.sender] = message.content

    # 设置等待阈值（不要求所有客户端）
    required_clients = int(self._client_num * 0.8)  # 80%客户端即可

    if len(self.local_statistics_buffer) >= required_clients:
        self._aggregate_covariances()
        # ...
```

### 4.7.3 数据完整性校验

```python
def _validate_statistics(self, content):
    """校验上传的统计量"""
    required_keys = ['means', 'covs', 'counts']

    for key in required_keys:
        if key not in content:
            raise ValueError(f"Missing required key: {key}")

    # 校验维度一致性
    embedding_dim = self.ggeur_cfg.embedding_dim
    for class_idx, mean in content['means'].items():
        if mean.shape != (embedding_dim,):
            raise ValueError(f"Invalid mean dimension for class {class_idx}")

    return True
```

---

## 4.8 通信优化策略

### 4.8.1 协方差矩阵压缩

```python
# 方法1：低秩近似
def compress_covariance(cov, rank=50):
    """使用SVD进行低秩近似"""
    U, S, Vt = np.linalg.svd(cov)
    # 只保留前rank个特征值
    compressed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    return compressed

# 方法2：对角化（仅传输特征值）
def diagonalize_covariance(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvalues, eigenvectors  # 分开传输
```

### 4.8.2 增量更新

```python
# 后续轮次只传输模型增量
def compute_model_delta(current_model, previous_model):
    delta = {}
    for key in current_model.keys():
        delta[key] = current_model[key] - previous_model[key]
    return delta
```

### 4.8.3 异步通信

```python
# FederatedScope支持异步模式
cfg.federate.mode = 'async'

# 服务器不等待所有客户端，收到就处理
def callback_for_model_para_async(self, message: Message):
    # 立即聚合到全局模型
    self._async_aggregate(message.content)
    # 立即返回更新后的模型
    self._send_model_to_client(message.sender)
```

---

**第四章完成。** 下一章将详细介绍GGEUR的所有配置项及其含义。

---

# 第五章：配置项详解

本章详细介绍GGEUR的所有配置项，包括各配置项的含义、默认值、推荐设置和使用示例。

## 5.1 配置项总览

GGEUR的配置项定义在 `federatedscope/core/configs/cfg_ggeur.py` 中，所有配置项都在 `cfg.ggeur` 命名空间下。

### 5.1.1 配置项分类

| 类别 | 配置项数量 | 用途 |
|------|------------|------|
| 基础配置 | 2 | 启用GGEUR、选择特征提取器 |
| CLIP配置 | 4 | CLIP模型设置 |
| CNN骨干网络配置 | 2 | CNN特征提取器设置 |
| 特征缓存配置 | 2 | 特征缓存设置 |
| 特征增强配置 | 4 | 增强参数设置 |
| MLP分类器配置 | 2 | 分类器架构设置 |
| LDS数据划分配置 | 3 | 非IID数据划分设置 |
| CNN知识蒸馏配置 | 8 | 蒸馏训练设置 |
| 特征对齐配置 | 4 | 特征对齐训练设置 |
| 分阶段训练配置 | 3 | 分阶段训练设置 |
| 端到端微调配置 | 3 | 微调设置 |

---

## 5.2 基础配置

### 5.2.1 `ggeur.use`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否启用GGEUR方法 |

```yaml
ggeur:
  use: true  # 启用GGEUR
```

### 5.2.2 `ggeur.feature_extractor`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | 'clip' |
| **可选值** | 'clip', 'cnn' |
| **说明** | 特征提取器类型 |

```yaml
ggeur:
  feature_extractor: 'clip'  # 使用CLIP提取特征（推荐）
  # feature_extractor: 'cnn'  # 使用预训练CNN提取特征
```

**选择指南：**
- `clip`：语义一致性好，适合跨域场景，需要较大内存
- `cnn`：计算效率高，适合资源受限场景

---

## 5.3 CLIP配置

### 5.3.1 `ggeur.clip_model`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | 'ViT-B-16' |
| **可选值** | 'ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'RN50', 'RN101' |
| **说明** | CLIP模型架构 |

```yaml
ggeur:
  clip_model: 'ViT-B-16'  # 512维特征，平衡性能与效率
  # clip_model: 'ViT-L-14'  # 768维特征，更高性能
  # clip_model: 'ViT-B-32'  # 512维特征，更快速度
```

**模型对比：**

| 模型 | 特征维度 | 参数量 | 速度 | 性能 |
|------|----------|--------|------|------|
| ViT-B-32 | 512 | 88M | 快 | 一般 |
| ViT-B-16 | 512 | 86M | 中 | 良好 |
| ViT-L-14 | 768 | 304M | 慢 | 最佳 |
| RN50 | 1024 | 38M | 快 | 一般 |
| RN101 | 512 | 56M | 中 | 良好 |

### 5.3.2 `ggeur.clip_pretrained`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | 'openai' |
| **说明** | 预训练权重来源 |

```yaml
ggeur:
  clip_pretrained: 'openai'  # OpenAI官方权重
  # clip_pretrained: 'laion2b_s34b_b79k'  # LAION数据集训练的权重
```

### 5.3.3 `ggeur.clip_model_path`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | '' |
| **说明** | 本地CLIP权重文件路径 |

```yaml
ggeur:
  clip_model_path: '/path/to/ViT-B-16.pt'  # 离线环境使用
```

### 5.3.4 `ggeur.embedding_dim`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 512 |
| **说明** | 特征向量维度 |

```yaml
ggeur:
  embedding_dim: 512   # ViT-B-16/32, RN101
  # embedding_dim: 768   # ViT-L-14
  # embedding_dim: 1024  # RN50, ConvNeXt-Base
```

**注意：** 此参数必须与所选模型的输出维度匹配。

---

## 5.4 CNN骨干网络配置

### 5.4.1 `ggeur.cnn_backbone`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | 'convnext_base' |
| **说明** | CNN特征提取器架构 |

```yaml
ggeur:
  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'  # 1024维特征
  # cnn_backbone: 'convnext_tiny'  # 768维特征
  # cnn_backbone: 'resnet50'  # 2048维特征
  # cnn_backbone: 'efficientnet_b0'  # 1280维特征
```

**支持的架构：**

| 架构 | 特征维度 | ImageNet准确率 |
|------|----------|----------------|
| convnext_tiny | 768 | 82.1% |
| convnext_base | 1024 | 83.8% |
| convnext_large | 1536 | 84.3% |
| resnet18 | 512 | 69.8% |
| resnet50 | 2048 | 76.1% |
| efficientnet_b0 | 1280 | 77.1% |

### 5.4.2 `ggeur.freeze_backbone`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 是否冻结CNN骨干网络参数 |

```yaml
ggeur:
  freeze_backbone: true   # 冻结，仅用于特征提取（推荐）
  # freeze_backbone: false  # 不冻结，允许微调
```

---

## 5.5 特征缓存配置

### 5.5.1 `ggeur.use_feature_cache`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 是否启用特征缓存 |

```yaml
ggeur:
  use_feature_cache: true  # 启用缓存，避免重复提取
```

### 5.5.2 `ggeur.feature_cache_dir`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | '' (自动设置为data.root同级目录) |
| **说明** | 特征缓存目录 |

```yaml
ggeur:
  feature_cache_dir: './clip_feature_cache'
```

---

## 5.6 特征增强配置

### 5.6.1 `ggeur.num_generated_per_sample`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 50 |
| **说明** | 每个原始样本生成的增强样本数 |

```yaml
ggeur:
  num_generated_per_sample: 50  # 每个样本生成50个增强特征
```

### 5.6.2 `ggeur.num_generated_per_prototype`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 50 |
| **说明** | 每个跨域原型生成的样本数 |

```yaml
ggeur:
  num_generated_per_prototype: 50
```

### 5.6.3 `ggeur.target_size_per_class`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 50 |
| **说明** | 每个类别增强后的目标样本数 |

```yaml
ggeur:
  target_size_per_class: 50  # 确保每类至少有50个样本
```

### 5.6.4 `ggeur.use_cross_client_prototypes`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 是否使用跨客户端原型进行增强 |

```yaml
ggeur:
  use_cross_client_prototypes: true  # 启用跨域原型增强
```

---

## 5.7 MLP分类器配置

### 5.7.1 `ggeur.mlp_hidden_dim`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 0 |
| **说明** | MLP隐藏层维度（0表示无隐藏层） |

```yaml
ggeur:
  mlp_hidden_dim: 0    # 简单线性分类器（推荐）
  # mlp_hidden_dim: 256  # 添加256维隐藏层
```

### 5.7.2 `ggeur.mlp_dropout`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.0 |
| **说明** | MLP的Dropout率 |

```yaml
ggeur:
  mlp_dropout: 0.0   # 无Dropout
  # mlp_dropout: 0.5   # 50% Dropout（配合hidden_dim使用）
```

---

## 5.8 LDS数据划分配置

### 5.8.1 `ggeur.use_lds`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否使用LDS（标签分布倾斜）数据划分 |

```yaml
ggeur:
  use_lds: true  # 启用Dirichlet分布划分
```

### 5.8.2 `ggeur.lds_alpha`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.1 |
| **说明** | Dirichlet分布的alpha参数 |

```yaml
ggeur:
  lds_alpha: 0.1   # 高度非IID（推荐用于测试）
  # lds_alpha: 0.5   # 中度非IID
  # lds_alpha: 1.0   # 接近均匀分布
  # lds_alpha: 10.0  # 几乎IID
```

**Alpha参数影响：**

| alpha | 非IID程度 | 描述 |
|-------|-----------|------|
| 0.1 | 极高 | 每个客户端只有少数几个类 |
| 0.5 | 高 | 类别分布明显倾斜 |
| 1.0 | 中等 | 存在一定倾斜 |
| 10.0 | 低 | 接近均匀分布 |

### 5.8.3 `ggeur.lds_seed`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 42 |
| **说明** | Dirichlet分布的随机种子 |

```yaml
ggeur:
  lds_seed: 42  # 可复现的数据划分
```

---

## 5.9 CNN知识蒸馏配置

### 5.9.1 `ggeur.use_cnn_distillation`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否启用CNN知识蒸馏训练 |

```yaml
ggeur:
  use_cnn_distillation: true  # 启用MLP→CNN知识蒸馏
```

### 5.9.2 `ggeur.cnn_model`

| 属性 | 值 |
|------|-----|
| **类型** | str |
| **默认值** | 'resnet18' |
| **说明** | 用于蒸馏的CNN模型 |

### 5.9.3 `ggeur.cnn_pretrained`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 是否使用ImageNet预训练权重 |

### 5.9.4 `ggeur.distill_temperature`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 4.0 |
| **说明** | 蒸馏温度（越高标签越软） |

### 5.9.5 `ggeur.distill_alpha`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.5 |
| **说明** | 蒸馏损失权重 |

**损失函数：**
```
L_total = alpha * L_CE(y_true) + (1 - alpha) * L_KL(soft_labels)
```

### 5.9.6 `ggeur.cnn_lr`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.01 |
| **说明** | CNN训练学习率 |

### 5.9.7 `ggeur.cnn_local_epochs`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 10 |
| **说明** | CNN本地训练轮数 |

### 5.9.8 `ggeur.cnn_warmup_rounds`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 0 |
| **说明** | 跳过CNN蒸馏的预热轮数 |

---

## 5.10 特征对齐配置

### 5.10.1 `ggeur.use_feature_alignment`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否启用CNN特征对齐训练 |

```yaml
ggeur:
  use_feature_alignment: true  # CNN特征对齐到CLIP空间
```

### 5.10.2 `ggeur.align_weight`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 1.0 |
| **说明** | 特征对齐损失权重 |

**损失函数：**
```
L_total = L_CE + align_weight * L_MSE(cnn_feat, clip_feat)
```

### 5.10.3 `ggeur.use_prototype_alignment`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 是否使用原型对齐 |

### 5.10.4 `ggeur.prototype_align_weight`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.5 |
| **说明** | 原型对齐损失权重 |

---

## 5.11 分阶段训练配置

### 5.11.1 `ggeur.use_separated_training`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否启用分阶段训练 |

```yaml
ggeur:
  use_separated_training: true  # 先训MLP，再训CNN
```

### 5.11.2 `ggeur.classifier_pretrain_rounds`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 20 |
| **说明** | 分类器预训练轮数（阶段1） |

### 5.11.3 `ggeur.freeze_classifier`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | True |
| **说明** | 阶段2是否冻结分类器 |

---

## 5.12 端到端微调配置

### 5.12.1 `ggeur.use_end_to_end_finetune`

| 属性 | 值 |
|------|-----|
| **类型** | bool |
| **默认值** | False |
| **说明** | 是否启用端到端微调 |

### 5.12.2 `ggeur.finetune_start_round`

| 属性 | 值 |
|------|-----|
| **类型** | int |
| **默认值** | 30 |
| **说明** | 开始微调的轮次 |

### 5.12.3 `ggeur.finetune_lr`

| 属性 | 值 |
|------|-----|
| **类型** | float |
| **默认值** | 0.0001 |
| **说明** | 微调学习率 |

---

## 5.13 配置文件模板

### 5.13.1 基础CLIP模式

```yaml
# ggeur_basic.yaml
use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  client_num: 4
  total_round_num: 50

data:
  root: ./data/
  type: officehome
  batch_size: 64

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 5.13.2 LDS非IID模式

```yaml
# ggeur_lds.yaml
# 继承基础配置...

ggeur:
  use: true
  use_lds: true
  lds_alpha: 0.1
  lds_seed: 42
  # 其他配置...
```

### 5.13.3 CNN特征对齐模式

```yaml
# ggeur_cnn_align.yaml
ggeur:
  use: true
  feature_extractor: 'clip'  # 仍需CLIP用于对齐目标
  use_feature_alignment: true
  cnn_model: 'resnet18'
  cnn_pretrained: false  # 从头训练
  align_weight: 1.0
  use_prototype_alignment: true
  prototype_align_weight: 0.5
```

---

**第五章完成。** 下一章将详细介绍GGEUR支持的数据集及其划分方式。

---

# 第六章：数据集与划分方式

本章详细介绍GGEUR支持的数据集、域划分策略和LDS（标签分布倾斜）实现。

## 6.1 支持的数据集

### 6.1.1 Office-Home

Office-Home是一个多域图像分类数据集，包含4个不同风格的域。

**数据集信息：**

| 属性 | 值 |
|------|-----|
| 类别数 | 65 |
| 域数量 | 4 |
| 总样本数 | ~15,500 |
| 图像尺寸 | 224×224 |

**域分布：**

| 域名 | 描述 | 样本数 | 特点 |
|------|------|--------|------|
| Art | 艺术绘画风格 | ~2,427 | 手绘、油画、素描 |
| Clipart | 剪贴画风格 | ~4,365 | 简化图形、卡通 |
| Product | 产品照片 | ~4,439 | 白底商品图 |
| Real_World | 真实世界照片 | ~4,357 | 自然场景照片 |

**类别示例：**
```
Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket,
Calculator, Calendar, Candles, Chair, Clipboards, Computer,
Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan,
File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses,
Hammer, Helmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop,
...（共65类）
```

**数据集结构：**
```
Office-Home/
├── Art/
│   ├── Alarm_Clock/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   ├── Backpack/
│   └── ...
├── Clipart/
├── Product/
└── Real_World/
```

### 6.1.2 PACS

PACS是一个经典的域适应基准数据集，图像风格差异更为显著。

**数据集信息：**

| 属性 | 值 |
|------|-----|
| 类别数 | 7 |
| 域数量 | 4 |
| 总样本数 | ~9,991 |
| 图像尺寸 | 224×224 |

**域分布：**

| 域名 | 描述 | 样本数 | 特点 |
|------|------|--------|------|
| Photo | 真实照片 | ~1,670 | 自然光照、真实纹理 |
| Art_Painting | 艺术绘画 | ~2,048 | 油画、水彩 |
| Cartoon | 卡通画 | ~2,344 | 简化线条、夸张形状 |
| Sketch | 素描 | ~3,929 | 黑白、手绘线条 |

**类别列表：**
```
Dog, Elephant, Giraffe, Guitar, Horse, House, Person
```

**数据集结构：**
```
PACS/
├── photo/
│   ├── dog/
│   ├── elephant/
│   ├── giraffe/
│   ├── guitar/
│   ├── horse/
│   ├── house/
│   └── person/
├── art_painting/
├── cartoon/
└── sketch/
```

### 6.1.3 Digits（多域数字识别）

用于数字识别的多域数据集组合。

**包含数据集：**

| 数据集 | 样本数 | 特点 |
|--------|--------|------|
| MNIST | 70,000 | 手写数字，黑白 |
| SVHN | 99,289 | 街景门牌号 |
| USPS | 9,298 | 邮政编码数字 |
| SynthDigits | 50,000 | 合成数字 |

---

## 6.2 域划分策略

### 6.2.1 每域对应一个客户端（Domain-based）

最简单的划分策略：每个域对应一个客户端。

```yaml
federate:
  client_num: 4  # 等于域数量

data:
  type: officehome

# 自动分配：
# Client 1 → Art域
# Client 2 → Clipart域
# Client 3 → Product域
# Client 4 → Real_World域
```

**代码实现：**
```python
def _load_officehome_ggeur_data(config, client_cfgs):
    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    for client_id, domain in enumerate(domains, start=1):
        # 加载该域的数据
        domain_dataset = load_domain_dataset(domain)

        # 分配给客户端
        data_dict[client_id] = {
            'train': DataLoader(domain_dataset, ...),
            'test': DataLoader(test_dataset, ...)
        }
```

### 6.2.2 多客户端共享域（Cross-domain）

每个域可以分配给多个客户端，模拟更复杂的联邦场景。

```yaml
federate:
  client_num: 12  # 4域 × 每域3客户端

# 自动分配：
# Clients 1-3  → Art域（数据均分）
# Clients 4-6  → Clipart域
# Clients 7-9  → Product域
# Clients 10-12 → Real_World域
```

**代码实现：**
```python
def _split_dataset_for_clients(dataset, num_clients, seed=123):
    """
    将数据集均匀划分给多个客户端。
    """
    n_samples = len(dataset)
    np.random.seed(seed)

    # 随机打乱索引
    indices = np.random.permutation(n_samples)

    # 计算每个客户端的样本数
    samples_per_client = n_samples // num_clients

    client_subsets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # 最后一个客户端获取剩余所有样本
            client_indices = indices[start_idx:]
        else:
            client_indices = indices[start_idx:start_idx + samples_per_client]

        client_subsets.append(Subset(dataset, client_indices))

    return client_subsets
```

---

## 6.3 LDS标签分布倾斜实现

LDS（Label Distribution Skew）使用Dirichlet分布创建非IID数据划分。

### 6.3.1 Dirichlet分布原理

Dirichlet分布是一种多元概率分布，用于生成和为1的概率向量。

**数学定义：**

$$\mathbf{p} \sim \text{Dir}(\alpha_1, \alpha_2, ..., \alpha_K)$$

其中 $\sum_{i=1}^{K} p_i = 1$

**参数 $\alpha$ 的影响：**
- $\alpha \rightarrow 0$：生成接近one-hot的向量（极端非IID）
- $\alpha = 1$：均匀分布
- $\alpha \rightarrow \infty$：生成接近均匀的向量（接近IID）

### 6.3.2 Dirichlet矩阵生成

```python
def _generate_dirichlet_matrix(num_domains, num_classes, alpha, seed=42):
    """
    生成Dirichlet分布矩阵用于LDS数据划分。

    Args:
        num_domains: 域（客户端）数量
        num_classes: 类别数量
        alpha: Dirichlet分布参数
        seed: 随机种子

    Returns:
        矩阵形状 (num_domains, num_classes)，每列和为1
    """
    np.random.seed(seed)

    # 为每个类别生成Dirichlet分布
    # dirichlet([alpha]*num_domains, num_classes) 返回 (num_classes, num_domains)
    matrix = np.random.dirichlet([alpha] * num_domains, num_classes).T

    return matrix
```

**矩阵示例（4个客户端，7个类别，alpha=0.1）：**

```
         Class0  Class1  Class2  Class3  Class4  Class5  Class6
Client1  0.95    0.01    0.02    0.88    0.03    0.05    0.01
Client2  0.02    0.92    0.85    0.05    0.01    0.02    0.03
Client3  0.01    0.05    0.10    0.04    0.94    0.90    0.02
Client4  0.02    0.02    0.03    0.03    0.02    0.03    0.94
```

每列和为1，表示该类别在所有客户端之间的分配比例。

### 6.3.3 `_split_dataset_with_lds()` 函数详解

```python
def _split_dataset_with_lds(dataset, dirichlet_proportions, seed=42):
    """
    使用Dirichlet比例划分数据集（LDS）。

    Args:
        dataset: 包含.targets属性的数据集
        dirichlet_proportions: 该客户端的比例数组（每类一个值）
        seed: 随机种子

    Returns:
        Tuple[Subset, dict]: 子数据集和各类别样本数统计
    """
    np.random.seed(seed)

    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    client_indices = []
    class_counts = {}

    for class_idx in range(num_classes):
        # 获取该类的所有索引
        class_mask = targets == class_idx
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        # 打乱索引
        np.random.shuffle(class_indices)

        # 获取该类的分配比例
        proportion = dirichlet_proportions[class_idx]

        # 计算分配的样本数
        num_to_allocate = int(proportion * len(class_indices))

        # 分配样本
        allocated = class_indices[:num_to_allocate].tolist()
        client_indices.extend(allocated)
        class_counts[class_idx] = len(allocated)

    return Subset(dataset, client_indices), class_counts
```

### 6.3.4 Alpha参数对非IID程度的影响

| Alpha | 分布特征 | 类别覆盖 | 适用场景 |
|-------|----------|----------|----------|
| 0.1 | 极端倾斜 | 每客户端1-3个主要类 | 测试极端非IID |
| 0.5 | 明显倾斜 | 每客户端3-5个主要类 | 模拟真实场景 |
| 1.0 | 中等倾斜 | 类别分布有差异 | 轻度非IID |
| 5.0 | 轻微倾斜 | 接近均匀 | 接近IID |
| 100.0 | 几乎均匀 | 所有类别均匀分布 | 对照实验 |

### 6.3.5 数据划分可视化示例

**Alpha = 0.1（极端非IID）：**

```
Client 1: ████████████████████ Class 0 (95%)
          ██ Class 1 (5%)

Client 2: ██████████████████ Class 1 (90%)
          ████ Class 2 (10%)

Client 3: ████████████████ Class 2 (80%)
          ██ Class 3 (10%)
          ██ Class 4 (10%)

Client 4: ████████████████████ Class 4 (90%)
          ██ Class 5 (10%)
```

**Alpha = 1.0（中等非IID）：**

```
Client 1: ██████ Class 0 (30%)
          ████ Class 1 (20%)
          ████ Class 2 (20%)
          ██ Class 3 (10%)
          ████ Class 4 (20%)

Client 2: ████ Class 0 (20%)
          ████████ Class 1 (40%)
          ██ Class 2 (10%)
          ██ Class 3 (10%)
          ████ Class 4 (20%)
...
```

---

## 6.4 数据增强策略

### 6.4.1 图像级增强（用于CNN训练）

```python
class AugmentedImageDataset(Dataset):
    """用于CNN训练的强数据增强"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # 强数据增强变换
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
```

### 6.4.2 特征级增强（GGEUR核心）

GGEUR的核心特征增强在特征空间进行，而非像素空间：

```python
def _perform_augmentation(self):
    """GGEUR特征级增强"""

    for class_idx in all_classes:
        # 1. 保留原始CLIP特征
        original_features = self.local_features[class_idx]

        # 2. 基于全局协方差生成虚拟特征
        virtual_features = np.random.multivariate_normal(
            mean=self.local_means[class_idx],
            cov=self.global_cov_matrices[class_idx],
            size=num_generated
        )

        # 3. 基于跨域原型生成特征
        for other_proto in self.other_prototypes[class_idx]:
            cross_features = np.random.multivariate_normal(
                mean=other_proto,
                cov=self.global_cov_matrices[class_idx],
                size=num_per_prototype
            )

        # 合并所有特征
        augmented = np.vstack([original, virtual, cross])
```

---

## 6.5 数据加载配置

### 6.5.1 完整配置示例

```yaml
data:
  root: ./data/            # 数据根目录
  type: officehome         # 数据集类型
  batch_size: 64           # 批量大小

  # 图像预处理
  transform:
    - type: Resize
      size: [224, 224]
    - type: ToTensor
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

federate:
  client_num: 4            # 客户端数量（通常等于域数量）

ggeur:
  use: true
  use_lds: true            # 启用LDS
  lds_alpha: 0.1           # Dirichlet参数
  lds_seed: 42             # 随机种子（可复现）
```

### 6.5.2 数据目录结构

```
./data/
├── Office-Home/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   └── Real_World/
├── PACS/
│   ├── photo/
│   ├── art_painting/
│   ├── cartoon/
│   └── sketch/
└── clip_feature_cache/    # GGEUR自动创建的特征缓存
    ├── officehome_Art_clip_ViT_B_16_openai.npz
    ├── officehome_Clipart_clip_ViT_B_16_openai.npz
    └── ...
```

---

## 6.6 添加新数据集

### 6.6.1 数据集接口

新数据集需要实现以下接口：

```python
class CustomDataset(Dataset):
    def __init__(self, root, domain, transform=None):
        self.root = root
        self.domain = domain
        self.transform = transform

        # 必须属性
        self.data = []      # 图像路径列表
        self.targets = []   # 标签列表

        self._load_data()

    def _load_data(self):
        # 加载数据路径和标签
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
```

### 6.6.2 注册新数据集

在 `ggeur_data.py` 中添加加载函数：

```python
def _load_custom_ggeur_data(config, client_cfgs):
    domains = ['domain1', 'domain2', 'domain3']
    # ... 实现加载逻辑

# 注册
register_data('custom_dataset', load_ggeur_data)
```

---

**第六章完成。** 下一章将提供快速入门指南，帮助用户快速上手GGEUR。

---

# 第七章：快速入门指南

本章提供GGEUR的快速入门教程，帮助用户从零开始运行第一个实验。

## 7.1 环境准备

### 7.1.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.8+ | 3.9+ |
| GPU | 4GB显存 | 8GB+显存 |
| RAM | 16GB | 32GB+ |
| 存储 | 10GB | 50GB+ |

### 7.1.2 依赖安装

```bash
# 1. 克隆FederatedScope
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope

# 2. 创建虚拟环境
conda create -n fs python=3.9
conda activate fs

# 3. 安装基础依赖
pip install -r requirements.txt

# 4. 安装GGEUR额外依赖
pip install open_clip_torch  # CLIP支持
pip install timm             # CNN骨干网络
```

### 7.1.3 CLIP模型下载

**方式1：自动下载（需要网络）**
```yaml
ggeur:
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'  # 首次运行时自动下载
```

**方式2：手动下载（离线环境）**
```bash
# 下载CLIP权重
wget https://openaipublic.azureedge.net/clip/models/ViT-B-16.pt

# 配置本地路径
ggeur:
  clip_model_path: '/path/to/ViT-B-16.pt'
```

### 7.1.4 数据集准备

**Office-Home下载：**
```bash
# 从官方链接下载
# https://www.hemanthdv.org/officeHomeDataset.html

# 解压到data目录
unzip OfficeHomeDataset_10072016.zip -d ./data/
mv ./data/OfficeHomeDataset_10072016 ./data/Office-Home
```

**PACS下载：**
```bash
# 从PACS官方仓库下载
git clone https://github.com/MachineLearning2020/Homework3-PACS.git
mv Homework3-PACS/PACS ./data/
```

**验证数据结构：**
```bash
# 检查目录结构
ls ./data/Office-Home/
# 应输出: Art  Clipart  Product  Real_World

ls ./data/PACS/
# 应输出: art_painting  cartoon  photo  sketch
```

---

## 7.2 运行第一个实验

### 7.2.1 基础配置文件

创建配置文件 `my_first_ggeur.yaml`：

```yaml
# my_first_ggeur.yaml
use_gpu: true
device: 0
seed: 42
verbose: 1

# 联邦设置
federate:
  mode: standalone           # 单机模拟模式
  client_num: 4              # 4个客户端（对应4个域）
  total_round_num: 30        # 训练30轮
  sample_client_rate: 1.0    # 每轮所有客户端参与

# 数据设置
data:
  root: ./data/Office-Home/
  type: officehome
  batch_size: 64

# 模型设置
model:
  type: ggeur_mlp
  num_classes: 65

# 训练设置
train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

# GGEUR设置
ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 7.2.2 启动命令

```bash
# 进入FederatedScope目录
cd FederatedScope

# 运行实验
python federatedscope/main.py --cfg my_first_ggeur.yaml
```

### 7.2.3 预期输出

```
[INFO] Loading configuration from my_first_ggeur.yaml
[INFO] Using device: cuda:0
[INFO] Dataset: officehome with 4 clients

[INFO] ===== Round 0: Statistics Collection =====
[INFO] Client 1: Extracting CLIP features from Art domain...
[INFO] Client 1: Loaded CLIP model ViT-B-16
[INFO] Client 1: Extracted 2427 features
[INFO] Client 1: Computing local statistics for 65 classes...
[INFO] Client 1: Statistics uploaded

[INFO] Client 2: Extracting CLIP features from Clipart domain...
...

[INFO] Server: Received statistics from 4/4 clients
[INFO] Server: Aggregating covariance matrices...
[INFO] Server: Aggregated covariances for 65 classes
[INFO] Server: Broadcasting global covariances...

[INFO] Client 1: Received global covariances
[INFO] Client 1: Performing GGEUR augmentation...
[INFO] Client 1: Augmented 65 classes, total 3250 samples
[INFO] Client 1: Augmentation ready

[INFO] Server: All clients ready, starting training...

[INFO] ===== Round 1: Federated Training =====
[INFO] Client 1: Training MLP on augmented data...
[INFO] Client 2: Training MLP on augmented data...
...
[INFO] Server: FedAvg aggregation completed
[INFO] Server: Evaluation on test sets:
[INFO]   Art: 72.5%
[INFO]   Clipart: 68.3%
[INFO]   Product: 75.1%
[INFO]   Real_World: 78.2%
[INFO]   Average: 73.5%

...

[INFO] ===== Round 30: Final Results =====
[INFO] Best Average Accuracy: 82.3%
[INFO] Training completed!
```

---

## 7.3 配置文件模板

### 7.3.1 Office-Home实验配置

```yaml
# ggeur_officehome.yaml
use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  client_num: 4
  total_round_num: 50

data:
  root: ./data/Office-Home/
  type: officehome
  batch_size: 64

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
```

### 7.3.2 PACS实验配置

```yaml
# ggeur_pacs.yaml
use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  client_num: 4
  total_round_num: 30

data:
  root: ./data/PACS/
  type: pacs
  batch_size: 32

model:
  type: ggeur_mlp
  num_classes: 7

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  embedding_dim: 512
```

### 7.3.3 LDS非IID配置

```yaml
# ggeur_officehome_lds.yaml
# 继承基础配置，添加LDS设置

federate:
  client_num: 4
  total_round_num: 50

data:
  root: ./data/Office-Home/
  type: officehome

ggeur:
  use: true
  use_lds: true          # 启用LDS
  lds_alpha: 0.1         # 高度非IID
  lds_seed: 42           # 可复现
  # 其他GGEUR配置...
```

---

## 7.4 常用命令

### 7.4.1 运行实验

```bash
# 基础运行
python federatedscope/main.py --cfg configs/ggeur.yaml

# 指定GPU
python federatedscope/main.py --cfg configs/ggeur.yaml device 0

# 覆盖配置项
python federatedscope/main.py --cfg configs/ggeur.yaml \
    federate.total_round_num 100 \
    ggeur.lds_alpha 0.5

# 多GPU运行
CUDA_VISIBLE_DEVICES=0,1 python federatedscope/main.py --cfg configs/ggeur.yaml
```

### 7.4.2 查看日志

```bash
# 日志默认保存在 exp/ 目录
ls exp/

# 查看最新日志
tail -f exp/ggeur_*/train.log

# 查看测试结果
grep "Accuracy" exp/ggeur_*/train.log
```

### 7.4.3 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir exp/

# 在浏览器中打开 http://localhost:6006
```

---

## 7.5 快速调试技巧

### 7.5.1 小规模测试

```yaml
# 快速测试配置
federate:
  total_round_num: 3      # 只运行3轮
  client_num: 2           # 只用2个客户端

data:
  batch_size: 16          # 小批量

ggeur:
  num_generated_per_sample: 5      # 少量增强
  num_generated_per_prototype: 5
```

### 7.5.2 查看特征缓存

```bash
# 查看缓存文件
ls ./clip_feature_cache/

# 查看缓存内容
python -c "
import numpy as np
data = np.load('./clip_feature_cache/officehome_Art_clip_ViT_B_16_openai.npz')
print('Keys:', list(data.keys()))
print('Features shape:', data['features'].shape)
print('Labels shape:', data['labels'].shape)
"
```

### 7.5.3 清除缓存

```bash
# 清除特征缓存（如需重新提取）
rm -rf ./clip_feature_cache/

# 清除实验结果
rm -rf ./exp/
```

---

**第七章完成。** 下一章将介绍如何进行分布式训练部署。

---

# 第八章：分布式训练指南

本章介绍如何在多机环境下部署和运行GGEUR分布式训练。

## 8.1 分布式架构

### 8.1.1 FederatedScope分布式模式

FederatedScope支持多种分布式模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `standalone` | 单机模拟 | 开发调试、小规模实验 |
| `distributed` | 多进程分布式 | 单机多进程模拟 |
| `grpc` | gRPC网络通信 | 真实多机部署 |

### 8.1.2 gRPC通信架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        网络拓扑                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌──────────────┐                                           │
│     │   Server     │                                           │
│     │  (Node 0)    │                                           │
│     │ IP: 10.0.0.1 │                                           │
│     └──────┬───────┘                                           │
│            │ gRPC                                               │
│     ┌──────┴───────┬───────────────┬───────────────┐           │
│     │              │               │               │            │
│     ▼              ▼               ▼               ▼            │
│ ┌────────┐    ┌────────┐     ┌────────┐     ┌────────┐         │
│ │Client 1│    │Client 2│     │Client 3│     │Client 4│         │
│ │(Node 1)│    │(Node 2)│     │(Node 3)│     │(Node 4)│         │
│ │10.0.0.2│    │10.0.0.3│     │10.0.0.4│     │10.0.0.5│         │
│ └────────┘    └────────┘     └────────┘     └────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8.2 分布式配置

### 8.2.1 服务器端配置

创建服务器配置文件 `ggeur_server.yaml`：

```yaml
# ggeur_server.yaml - 服务器配置
use_gpu: true
device: 0
seed: 42

federate:
  mode: distributed
  client_num: 4
  total_round_num: 50
  online_aggr: false        # GGEUR需要等待所有客户端

distribute:
  server_host: '10.0.0.1'   # 服务器IP
  server_port: 50051        # 服务器端口
  role: 'server'            # 角色：服务器
  grpc_max_send_message_length: 100000000   # 100MB（协方差矩阵较大）
  grpc_max_receive_message_length: 100000000
  grpc_timeout: 3600        # 1小时超时

data:
  root: ./data/Office-Home/
  type: officehome

model:
  type: ggeur_mlp
  num_classes: 65

ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  embedding_dim: 512
```

### 8.2.2 客户端配置

创建客户端配置文件 `ggeur_client.yaml`：

```yaml
# ggeur_client.yaml - 客户端配置
use_gpu: true
device: 0
seed: 42

federate:
  mode: distributed
  client_num: 4
  total_round_num: 50

distribute:
  server_host: '10.0.0.1'   # 服务器IP（需修改）
  server_port: 50051
  role: 'client'            # 角色：客户端
  client_host: '10.0.0.2'   # 本机IP（每个客户端不同）
  client_port: 50052        # 本机端口
  grpc_max_send_message_length: 100000000
  grpc_max_receive_message_length: 100000000

data:
  root: ./data/Office-Home/
  type: officehome
  # 每个客户端指定不同的域
  # client_id: 1  # 由启动脚本指定

model:
  type: ggeur_mlp
  num_classes: 65

ggeur:
  use: true
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  embedding_dim: 512
  use_feature_cache: true
```

### 8.2.3 网络参数调优

```yaml
distribute:
  # gRPC消息大小限制（字节）
  grpc_max_send_message_length: 100000000     # 100MB
  grpc_max_receive_message_length: 100000000

  # 超时设置（秒）
  grpc_timeout: 3600          # 等待响应超时

  # 连接设置
  grpc_channel_options:
    - ['grpc.max_metadata_size', 16777216]
    - ['grpc.keepalive_time_ms', 30000]
    - ['grpc.keepalive_timeout_ms', 10000]
```

---

## 8.3 启动流程

### 8.3.1 服务器启动脚本

创建 `start_server.sh`：

```bash
#!/bin/bash
# start_server.sh - 启动GGEUR服务器

export CUDA_VISIBLE_DEVICES=0

echo "Starting GGEUR Server..."
echo "Server IP: 10.0.0.1:50051"

python federatedscope/main.py \
    --cfg configs/ggeur_server.yaml \
    distribute.role server \
    distribute.server_host '10.0.0.1' \
    distribute.server_port 50051

echo "Server stopped."
```

### 8.3.2 客户端启动脚本

创建 `start_client.sh`：

```bash
#!/bin/bash
# start_client.sh - 启动GGEUR客户端
# 用法: ./start_client.sh <client_id> <client_ip>

CLIENT_ID=$1
CLIENT_IP=$2
CLIENT_PORT=$((50052 + CLIENT_ID))

export CUDA_VISIBLE_DEVICES=0

echo "Starting GGEUR Client ${CLIENT_ID}..."
echo "Client IP: ${CLIENT_IP}:${CLIENT_PORT}"
echo "Server IP: 10.0.0.1:50051"

python federatedscope/main.py \
    --cfg configs/ggeur_client.yaml \
    distribute.role client \
    distribute.client_host "${CLIENT_IP}" \
    distribute.client_port "${CLIENT_PORT}" \
    distribute.server_host '10.0.0.1' \
    distribute.server_port 50051 \
    federate.client_id "${CLIENT_ID}"

echo "Client ${CLIENT_ID} stopped."
```

### 8.3.3 多机启动顺序

```bash
# 步骤1：在服务器节点启动服务器
# [Node 0: 10.0.0.1]
./start_server.sh

# 步骤2：在各客户端节点启动客户端（等服务器就绪后）
# [Node 1: 10.0.0.2]
./start_client.sh 1 10.0.0.2

# [Node 2: 10.0.0.3]
./start_client.sh 2 10.0.0.3

# [Node 3: 10.0.0.4]
./start_client.sh 3 10.0.0.4

# [Node 4: 10.0.0.5]
./start_client.sh 4 10.0.0.5
```

### 8.3.4 Docker部署

```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install open_clip_torch timm

# 复制代码
COPY federatedscope/ ./federatedscope/
COPY configs/ ./configs/

# 暴露端口
EXPOSE 50051 50052

ENTRYPOINT ["python", "federatedscope/main.py"]
```

```bash
# 构建镜像
docker build -t ggeur:latest .

# 启动服务器容器
docker run -d --gpus all \
    -p 50051:50051 \
    -v /data:/app/data \
    --name ggeur_server \
    ggeur:latest \
    --cfg configs/ggeur_server.yaml

# 启动客户端容器
docker run -d --gpus all \
    -p 50052:50052 \
    -v /data:/app/data \
    --name ggeur_client1 \
    ggeur:latest \
    --cfg configs/ggeur_client.yaml \
    federate.client_id 1
```

---

## 8.4 性能优化

### 8.4.1 特征缓存策略

**预提取特征：**
```bash
# 在所有节点上预提取特征（避免分布式训练时的网络延迟）
python scripts/extract_features.py \
    --data_root ./data/Office-Home/ \
    --output_dir ./clip_feature_cache/ \
    --clip_model ViT-B-16
```

**共享缓存（NFS）：**
```yaml
ggeur:
  feature_cache_dir: '/shared/nfs/clip_cache/'  # NFS共享目录
  use_feature_cache: true
```

### 8.4.2 通信压缩

```yaml
# 启用梯度压缩
compress:
  use: true
  method: 'quantization'
  nbits: 8  # 8位量化
```

### 8.4.3 批量大小调优

```yaml
# 根据GPU显存调整
data:
  batch_size: 64    # 8GB GPU
  # batch_size: 128  # 16GB GPU
  # batch_size: 256  # 32GB GPU

dataloader:
  num_workers: 4    # 数据加载线程
  pin_memory: true  # 加速GPU传输
```

### 8.4.4 异步聚合

```yaml
federate:
  online_aggr: true         # 收到就聚合（注意：Round 0不适用）
  sample_client_rate: 0.8   # 80%客户端参与即可
```

---

## 8.5 故障排查

### 8.5.1 常见问题

**问题1：连接超时**
```
Error: Connection timed out to 10.0.0.1:50051
```
解决：
- 检查防火墙规则
- 确认服务器已启动
- 验证IP和端口正确

**问题2：消息过大**
```
Error: Received message larger than max
```
解决：
```yaml
distribute:
  grpc_max_send_message_length: 200000000  # 增大限制
  grpc_max_receive_message_length: 200000000
```

**问题3：GPU内存不足**
```
RuntimeError: CUDA out of memory
```
解决：
```yaml
data:
  batch_size: 32  # 减小批量大小
ggeur:
  num_generated_per_sample: 20  # 减少增强样本
```

### 8.5.2 日志调试

```yaml
# 启用详细日志
verbose: 2

# 日志配置
logging:
  level: DEBUG
  file: './logs/ggeur_{role}_{id}.log'
```

```bash
# 实时查看日志
tail -f ./logs/ggeur_server_0.log
tail -f ./logs/ggeur_client_1.log
```

### 8.5.3 网络诊断

```bash
# 测试服务器连通性
nc -zv 10.0.0.1 50051

# 测试gRPC服务
grpcurl -plaintext 10.0.0.1:50051 list

# 检查端口占用
netstat -tlnp | grep 50051
```

---

**第八章完成。** 下一章将介绍如何扩展和自定义GGEUR。

---

# 第九章：扩展与自定义

本章介绍如何扩展GGEUR以支持新的数据集、特征提取器、聚合策略和训练模式。

## 9.1 添加新数据集

### 9.1.1 数据集接口要求

新数据集需要满足以下接口要求：

```python
class CustomDataset(Dataset):
    """自定义数据集模板"""

    # 必须属性
    DOMAINS = ['domain1', 'domain2', 'domain3']  # 域列表
    CLASSES = ['class1', 'class2', ...]          # 类别列表

    def __init__(self, root, domain, transform=None, train=True):
        """
        Args:
            root: 数据集根目录
            domain: 域名称
            transform: 图像变换
            train: 是否为训练集
        """
        self.root = root
        self.domain = domain
        self.transform = transform

        # 必须属性：图像路径列表和标签列表
        self.data = []      # List[str]: 图像路径
        self.targets = []   # List[int]: 标签

        self._load_data()

    def _load_data(self):
        """加载数据路径和标签"""
        domain_dir = os.path.join(self.root, self.domain)

        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append(img_path)
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
```

### 9.1.2 数据加载函数实现

在 `ggeur_data.py` 中添加新的加载函数：

```python
def _load_custom_ggeur_data(config, client_cfgs=None):
    """
    Load custom dataset for GGEUR.

    支持功能：
    - 多客户端划分（每域多个客户端）
    - LDS非IID数据划分
    - 自定义训练/验证/测试分割
    """
    from my_datasets.custom_dataset import CustomDataset

    root = config.data.root
    batch_size = config.dataloader.batch_size
    num_workers = config.dataloader.num_workers

    # 分割比例
    splits = tuple(config.data.splits) if hasattr(config.data, 'splits') else (0.8, 0.1, 0.1)

    # 标准变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    domains = CustomDataset.DOMAINS
    num_domains = len(domains)
    num_classes = len(CustomDataset.CLASSES)

    # 检查LDS设置
    use_lds = getattr(config.ggeur, 'use_lds', False) if hasattr(config, 'ggeur') else False

    # 计算每域客户端数
    configured_client_num = config.federate.client_num
    clients_per_domain = max(1, configured_client_num // num_domains)
    total_clients = clients_per_domain * num_domains

    if use_lds:
        lds_alpha = getattr(config.ggeur, 'lds_alpha', 0.1)
        lds_seed = getattr(config.ggeur, 'lds_seed', 42)
        dirichlet_matrix = _generate_dirichlet_matrix(num_domains, num_classes, lds_alpha, lds_seed)
    else:
        dirichlet_matrix = None

    data_dict = {}
    client_id = 1

    for domain_idx, domain in enumerate(domains):
        # 加载域数据
        train_dataset = CustomDataset(root, domain, transform, train=True)
        test_dataset = CustomDataset(root, domain, transform, train=False)

        # 划分训练集
        if use_lds:
            train_subset, _ = _split_dataset_with_lds(
                train_dataset, dirichlet_matrix[domain_idx], seed=config.seed + domain_idx
            )
            train_subsets = [train_subset]
        else:
            if clients_per_domain > 1:
                train_subsets = _split_dataset_for_clients(train_dataset, clients_per_domain, config.seed)
            else:
                train_subsets = [train_dataset]

        # 创建DataLoader
        for train_subset in train_subsets:
            data_dict[client_id] = {
                'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                'val': None,
                'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            }
            client_id += 1

    config.federate.client_num = total_clients
    return data_dict, config
```

### 9.1.3 注册新数据集

在 `ggeur_data.py` 的 `load_ggeur_data` 函数中添加分支：

```python
def load_ggeur_data(config, client_cfgs=None):
    data_type = config.data.type.lower()

    if data_type == 'pacs':
        return _load_pacs_ggeur_data(config, client_cfgs)
    elif data_type in ['office-home', 'officehome', 'office_home']:
        return _load_officehome_ggeur_data(config, client_cfgs)
    elif data_type == 'custom':  # 添加新数据集
        return _load_custom_ggeur_data(config, client_cfgs)
    else:
        logger.warning(f"Data type {data_type} not supported")
        return None
```

### 9.1.4 数据集目录结构

确保数据集按照以下结构组织：

```
custom_dataset/
├── domain1/
│   ├── class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── domain2/
│   └── ...
└── domain3/
    └── ...
```

---

## 9.2 自定义特征提取器

### 9.2.1 特征提取器接口

新的特征提取器需要实现以下接口：

```python
class CustomFeatureExtractor(nn.Module):
    """自定义特征提取器模板"""

    def __init__(self, model_name='custom', pretrained=True, freeze=True):
        super().__init__()

        self.model_name = model_name
        self.feature_dim = 512  # 输出特征维度

        # 构建骨干网络
        self.backbone = self._build_backbone(model_name, pretrained)

        # 冻结参数（如果需要）
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_backbone(self, model_name, pretrained):
        """构建骨干网络"""
        # 示例：使用自定义模型
        if model_name == 'custom_vit':
            from my_models import CustomViT
            model = CustomViT(pretrained=pretrained)
            return model
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        """提取特征向量"""
        features = self.backbone(x)
        return features.flatten(1)  # 确保输出是2D: (batch, feature_dim)

    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim
```

### 9.2.2 在客户端中集成

修改 `ggeur_client.py` 以支持新的特征提取器：

```python
def _load_feature_extractor(self):
    """Load feature extractor based on config"""
    extractor_type = self.feature_extractor_type

    if extractor_type == 'clip':
        self._load_clip_model()
    elif extractor_type == 'cnn':
        self._load_cnn_extractor()
    elif extractor_type == 'custom':  # 添加自定义提取器
        self._load_custom_extractor()
    else:
        raise ValueError(f"Unknown feature extractor: {extractor_type}")

def _load_custom_extractor(self):
    """Load custom feature extractor"""
    if self.custom_extractor is not None:
        return

    from my_extractors.custom_extractor import CustomFeatureExtractor

    model_name = getattr(self.ggeur_cfg, 'custom_model', 'custom_vit')
    pretrained = getattr(self.ggeur_cfg, 'custom_pretrained', True)
    freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)

    self.custom_extractor = CustomFeatureExtractor(
        model_name=model_name,
        pretrained=pretrained,
        freeze=freeze
    )
    self.custom_extractor = self.custom_extractor.to(self.device)

    logger.info(f"Client {self.ID}: Loaded custom extractor {model_name}")
```

### 9.2.3 配置项扩展

在 `cfg_ggeur.py` 中添加新配置项：

```python
def extend_ggeur_cfg(cfg):
    # ... 现有配置 ...

    # 自定义特征提取器配置
    cfg.ggeur.custom_model = 'custom_vit'
    cfg.ggeur.custom_pretrained = True
    cfg.ggeur.custom_model_path = ''
```

---

## 9.3 修改聚合策略

### 9.3.1 自定义协方差聚合

可以扩展平行轴定理的聚合方式，例如添加正则化或加权策略：

```python
def _aggregate_covariances_custom(self, regularization='ridge', reg_weight=0.01):
    """
    Custom covariance aggregation with regularization.

    Args:
        regularization: 正则化类型 ('ridge', 'shrinkage', 'none')
        reg_weight: 正则化权重
    """
    for class_idx in all_classes:
        means, covs, counts = [], [], []

        # 收集统计量
        for client_id, stats in self.local_statistics_buffer.items():
            if class_idx in stats['means']:
                means.append(stats['means'][class_idx])
                covs.append(stats['covs'][class_idx])
                counts.append(stats['counts'][class_idx])

        if not counts:
            continue

        total_count = sum(counts)

        # 标准聚合（平行轴定理）
        aggregated_mean = sum(c * m for c, m in zip(counts, means)) / total_count
        aggregated_cov = np.zeros_like(covs[0])

        for cov, mean, count in zip(covs, means, counts):
            aggregated_cov += count * cov
            diff = mean - aggregated_mean
            aggregated_cov += count * np.outer(diff, diff)

        aggregated_cov /= total_count

        # 应用正则化
        if regularization == 'ridge':
            # Ridge正则化：添加对角项
            aggregated_cov += reg_weight * np.eye(aggregated_cov.shape[0])

        elif regularization == 'shrinkage':
            # Ledoit-Wolf收缩估计
            trace = np.trace(aggregated_cov)
            dim = aggregated_cov.shape[0]
            target = (trace / dim) * np.eye(dim)
            aggregated_cov = (1 - reg_weight) * aggregated_cov + reg_weight * target

        self.global_cov_matrices[class_idx] = aggregated_cov
```

### 9.3.2 自定义原型聚合

```python
def _compute_global_prototypes_weighted(self, weighting='sample_count'):
    """
    Compute global prototypes with custom weighting.

    Args:
        weighting: 加权方式
            - 'sample_count': 按样本数加权（默认）
            - 'uniform': 均匀加权
            - 'accuracy': 按客户端准确率加权（需要额外信息）
    """
    for class_idx in all_classes:
        means, counts = [], []

        for client_id, stats in self.local_statistics_buffer.items():
            if class_idx in stats['means']:
                means.append(stats['means'][class_idx])
                counts.append(stats['counts'][class_idx])

        if not means:
            continue

        if weighting == 'uniform':
            weights = [1.0 / len(means)] * len(means)
        elif weighting == 'sample_count':
            total = sum(counts)
            weights = [c / total for c in counts]
        else:
            weights = [1.0 / len(means)] * len(means)

        global_mean = sum(w * m for w, m in zip(weights, means))
        self.global_prototypes[class_idx] = global_mean
```

---

## 9.4 添加新训练模式

### 9.4.1 训练模式接口

新的训练模式需要在客户端和服务器端实现相应的逻辑：

```python
# 在 ggeur_client.py 中添加新训练模式

class GGEURClient(Client):
    def __init__(self, ...):
        # ... 现有初始化 ...

        # 新训练模式标志
        self.use_custom_training = getattr(self.ggeur_cfg, 'use_custom_training', False)

    def callback_funcs_for_model_para(self, message: Message):
        # ... 现有逻辑 ...

        # 添加新训练模式分支
        if self.use_custom_training:
            self._handle_custom_training(message)
            return

        # ... 其他模式 ...

    def _handle_custom_training(self, message: Message):
        """处理自定义训练模式"""
        round_idx = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # 解析模型参数
        if content is not None:
            self._load_model_params(content)

        # 执行自定义训练
        sample_size, model_para, results = self._train_custom_mode()

        # 发送结果
        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=(sample_size, model_para)
            )
        )

    def _train_custom_mode(self):
        """自定义训练逻辑"""
        # 实现自定义训练算法
        # ...
        return sample_size, model_para, results
```

### 9.4.2 配置项注册

在 `cfg_ggeur.py` 中注册新配置：

```python
def extend_ggeur_cfg(cfg):
    # ... 现有配置 ...

    # ===== 自定义训练模式配置 =====
    cfg.ggeur.use_custom_training = False
    cfg.ggeur.custom_lr = 0.01
    cfg.ggeur.custom_epochs = 10
    cfg.ggeur.custom_weight = 1.0
```

### 9.4.3 示例：对比学习增强模式

```python
def _train_with_contrastive_learning(self):
    """
    Contrastive learning enhanced training.

    在标准CE损失基础上添加对比损失：
    L = L_CE + λ * L_contrastive
    """
    if self.augmented_loader is None:
        return 0, {}, {}

    self.mlp_classifier.train()

    optimizer = torch.optim.Adam(self.mlp_classifier.parameters(), lr=self._cfg.train.optimizer.lr)
    ce_criterion = nn.CrossEntropyLoss()

    contrastive_weight = getattr(self.ggeur_cfg, 'contrastive_weight', 0.1)
    temperature = getattr(self.ggeur_cfg, 'contrastive_temperature', 0.07)

    total_loss = 0.0
    total_samples = 0

    for epoch in range(self._cfg.train.local_update_steps):
        for features, labels in self.augmented_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # 分类损失
            outputs = self.mlp_classifier(features)
            ce_loss = ce_criterion(outputs, labels)

            # 对比损失：同类样本靠近，异类样本远离
            normalized_features = F.normalize(features, p=2, dim=1)
            similarity = torch.mm(normalized_features, normalized_features.t()) / temperature

            # 创建正样本mask
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask.fill_diagonal_(False)

            # InfoNCE损失
            exp_sim = torch.exp(similarity)
            log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

            # 只对正样本计算损失
            contrastive_loss = -(log_prob * mask.float()).sum() / mask.float().sum().clamp(min=1)

            # 总损失
            loss = ce_loss + contrastive_weight * contrastive_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

    model_para = copy.deepcopy(self.mlp_classifier.state_dict())
    return total_samples, model_para, {'train_loss': total_loss / total_samples}
```

---

## 9.5 扩展消息协议

### 9.5.1 添加新消息类型

在客户端注册新的消息处理器：

```python
def _register_default_handlers(self):
    super()._register_default_handlers()

    # 注册新消息类型
    self.register_handlers('custom_message', self.callback_for_custom_message)

def callback_for_custom_message(self, message: Message):
    """处理自定义消息"""
    content = message.content

    # 解析消息内容
    custom_data = content.get('custom_data', {})

    # 处理逻辑
    self._process_custom_data(custom_data)

    # 发送响应
    self.comm_manager.send(
        Message(
            msg_type='custom_response',
            sender=self.ID,
            receiver=[self.server_id],
            state=self.state,
            content={'status': 'success'}
        )
    )
```

### 9.5.2 服务器端消息处理

```python
class GGEURServer(Server):
    def _register_default_handlers(self):
        super()._register_default_handlers()

        # 注册自定义消息处理器
        self.register_handlers('custom_response', self.callback_for_custom_response)

    def _send_custom_message(self):
        """向所有客户端发送自定义消息"""
        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(
                    msg_type='custom_message',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content={'custom_data': {...}}
                )
            )

    def callback_for_custom_response(self, message: Message):
        """处理客户端的自定义响应"""
        client_id = message.sender
        status = message.content.get('status')

        logger.info(f"Received custom response from client {client_id}: {status}")
```

---

## 9.6 插件化扩展指南

### 9.6.1 创建独立扩展模块

```
my_ggeur_extension/
├── __init__.py
├── extractors/
│   ├── __init__.py
│   └── custom_extractor.py
├── trainers/
│   ├── __init__.py
│   └── custom_trainer.py
├── aggregators/
│   ├── __init__.py
│   └── custom_aggregator.py
└── configs/
    ├── __init__.py
    └── custom_config.py
```

### 9.6.2 注册扩展

```python
# my_ggeur_extension/__init__.py

from federatedscope.register import register_worker, register_data

from .extractors.custom_extractor import CustomFeatureExtractor
from .trainers.custom_trainer import CustomTrainer

def register_extensions():
    """注册所有自定义扩展"""
    # 注册自定义worker
    register_worker('ggeur_custom', call_custom_worker)

    # 注册自定义数据加载器
    register_data('custom_dataset', load_custom_data)

def call_custom_worker(method):
    if method.lower() == 'ggeur_custom':
        from .workers.custom_client import CustomGGEURClient
        from .workers.custom_server import CustomGGEURServer
        return {
            'client': CustomGGEURClient,
            'server': CustomGGEURServer
        }
    return None
```

### 9.6.3 使用扩展

```yaml
# 在配置文件中使用自定义扩展
federate:
  method: ggeur_custom  # 使用自定义worker

data:
  type: custom_dataset  # 使用自定义数据集

ggeur:
  use: true
  feature_extractor: 'custom'  # 使用自定义特征提取器
  use_custom_training: true    # 使用自定义训练模式
```

---

**第九章完成。** 下一章将提供完整的实验配置示例。

---

# 第十章：实验配置示例

本章提供GGEUR的完整实验配置文件，涵盖不同数据集、训练模式和场景。

## 10.1 基础实验配置

### 10.1.1 Office-Home基础配置

```yaml
# ggeur_officehome_basic.yaml
# 基础GGEUR配置 - Office-Home数据集

use_gpu: true
device: 0
seed: 42
verbose: 1

# ===== 联邦学习设置 =====
federate:
  mode: standalone           # 单机模拟
  method: ggeur              # 使用GGEUR方法
  client_num: 4              # 4个客户端（对应4个域）
  total_round_num: 50        # 总训练轮数
  sample_client_rate: 1.0    # 每轮所有客户端参与

# ===== 数据设置 =====
data:
  root: ./data/Office-Home/
  type: officehome
  splits: [0.7, 0.0, 0.3]    # 训练70%, 验证0%, 测试30%

dataloader:
  batch_size: 64
  num_workers: 4

# ===== 模型设置 =====
model:
  type: ggeur_mlp
  num_classes: 65            # Office-Home有65个类

# ===== 训练设置 =====
train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5      # 每轮本地训练5个epoch

# ===== GGEUR设置 =====
ggeur:
  use: true
  statistics_round: 0        # 第0轮收集统计量

  # 特征提取器
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512

  # 特征缓存
  use_feature_cache: true
  feature_cache_dir: './clip_feature_cache'

  # 特征增强
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  # MLP分类器
  mlp_hidden_dim: 0          # 简单线性分类器
  mlp_dropout: 0.0
```

### 10.1.2 PACS基础配置

```yaml
# ggeur_pacs_basic.yaml
# 基础GGEUR配置 - PACS数据集

use_gpu: true
device: 0
seed: 42
verbose: 1

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 30

data:
  root: ./data/PACS/
  type: pacs
  splits: [0.8, 0.1, 0.1]

dataloader:
  batch_size: 32
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 7             # PACS有7个类

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512

  use_feature_cache: true

  num_generated_per_sample: 30
  num_generated_per_prototype: 30
  target_size_per_class: 100
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

---

## 10.2 LDS非IID实验

### 10.2.1 轻度非IID（Alpha=0.5）

```yaml
# ggeur_officehome_lds_mild.yaml
# LDS非IID配置 - 轻度倾斜

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 100       # 非IID需要更多轮次

data:
  root: ./data/Office-Home/
  type: officehome
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 64
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 10     # 增加本地训练步数

ggeur:
  use: true
  statistics_round: 0

  # CLIP设置
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  # LDS设置
  use_lds: true
  lds_alpha: 0.5             # 轻度非IID
  lds_seed: 42

  # 增强设置（非IID需要更多增强）
  num_generated_per_sample: 100
  num_generated_per_prototype: 100
  target_size_per_class: 200
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 10.2.2 重度非IID（Alpha=0.1）

```yaml
# ggeur_officehome_lds_extreme.yaml
# LDS非IID配置 - 重度倾斜

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 150       # 重度非IID需要更多轮次

data:
  root: ./data/Office-Home/
  type: officehome
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 64
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.0005               # 降低学习率
  local_update_steps: 15

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  # 重度LDS设置
  use_lds: true
  lds_alpha: 0.1             # 重度非IID
  lds_seed: 42

  # 大量增强以弥补数据不足
  num_generated_per_sample: 200
  num_generated_per_prototype: 200
  target_size_per_class: 500
  use_cross_client_prototypes: true

  # 添加隐藏层增强模型能力
  mlp_hidden_dim: 256
  mlp_dropout: 0.3
```

---

## 10.3 CNN训练模式

### 10.3.1 CNN特征提取器配置

```yaml
# ggeur_cnn_convnext.yaml
# 使用CNN作为特征提取器

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 50

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 64
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  # CNN特征提取器（替代CLIP）
  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'  # ConvNeXt-Base
  freeze_backbone: true          # 冻结骨干网络
  embedding_dim: 1024            # ConvNeXt-Base特征维度

  use_feature_cache: true

  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 10.3.2 CNN知识蒸馏配置

```yaml
# ggeur_cnn_distillation.yaml
# CNN知识蒸馏模式

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 100

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 32             # CNN训练需要较小batch
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  # CLIP特征提取
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  # 增强设置
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0

  # CNN知识蒸馏设置
  use_cnn_distillation: true
  cnn_model: 'resnet18'
  cnn_pretrained: true       # 使用ImageNet预训练
  distill_temperature: 4.0   # 蒸馏温度
  distill_alpha: 0.5         # CE损失权重
  cnn_lr: 0.01               # CNN学习率
  cnn_local_epochs: 5        # CNN本地训练轮数
  cnn_warmup_rounds: 5       # CNN预热轮数（先训练MLP）
```

### 10.3.3 CNN特征对齐配置

```yaml
# ggeur_cnn_feature_alignment.yaml
# CNN特征对齐模式（从头训练）

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 150       # 从头训练需要更多轮次

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 32
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0

  # CNN特征对齐设置
  use_feature_alignment: true
  cnn_model: 'resnet18'
  cnn_pretrained: false      # 从头训练
  align_weight: 1.0          # 特征对齐损失权重
  use_prototype_alignment: true  # 使用原型对齐
  prototype_align_weight: 0.5    # 原型对齐权重
  cnn_lr: 0.01
  cnn_local_epochs: 10
  cnn_warmup_rounds: 10
  cnn_use_augmentation: true # 启用数据增强
```

---

## 10.4 分阶段训练配置

```yaml
# ggeur_separated_training.yaml
# 分阶段训练：先MLP，后CNN

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 100

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 32
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0

  # 分阶段训练设置
  use_separated_training: true
  classifier_pretrain_rounds: 20  # 阶段1：训练分类器20轮
  freeze_classifier: true         # 阶段2：冻结分类器

  # CNN设置
  cnn_model: 'resnet18'
  cnn_pretrained: false
  align_weight: 1.0
  cnn_lr: 0.01
  cnn_local_epochs: 10
  cnn_weight_decay: 0.0005
  cnn_dropout: 0.5
```

---

## 10.5 高级配置

### 10.5.1 多客户端共享域配置

```yaml
# ggeur_multi_client_per_domain.yaml
# 每个域多个客户端

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 12             # 4域 × 3客户端/域 = 12客户端
  total_round_num: 100

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 64
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 10

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  # 每个客户端数据量减少，需要更多增强
  num_generated_per_sample: 100
  num_generated_per_prototype: 100
  target_size_per_class: 100
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 10.5.2 无增强基线配置

```yaml
# ggeur_baseline_no_augmentation.yaml
# 基线对照：不使用GGEUR增强

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 100

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 64
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 10

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  embedding_dim: 512
  use_feature_cache: true

  # 禁用增强
  num_generated_per_sample: 0
  num_generated_per_prototype: 0
  target_size_per_class: 0
  use_cross_client_prototypes: false  # 关键：禁用跨客户端原型

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

### 10.5.3 大规模CLIP配置

```yaml
# ggeur_large_clip.yaml
# 使用大规模CLIP模型

use_gpu: true
device: 0
seed: 42

federate:
  mode: standalone
  method: ggeur
  client_num: 4
  total_round_num: 30

data:
  root: ./data/Office-Home/
  type: officehome

dataloader:
  batch_size: 32             # 大模型需要较小batch
  num_workers: 4

model:
  type: ggeur_mlp
  num_classes: 65

train:
  optimizer:
    type: Adam
    lr: 0.001
  local_update_steps: 5

ggeur:
  use: true
  statistics_round: 0

  feature_extractor: 'clip'
  clip_model: 'ViT-L-14'     # 大规模CLIP
  clip_pretrained: 'openai'
  embedding_dim: 768         # ViT-L-14特征维度
  use_feature_cache: true

  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  use_cross_client_prototypes: true

  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

---

## 10.6 调参指南

### 10.6.1 关键参数调优表

| 参数 | 默认值 | 调优范围 | 影响 |
|------|--------|----------|------|
| `num_generated_per_sample` | 50 | 10-200 | 增强强度 |
| `num_generated_per_prototype` | 50 | 10-200 | 跨域知识利用 |
| `target_size_per_class` | 50 | 0(无限)-500 | 类别平衡 |
| `lds_alpha` | 0.1 | 0.1-10.0 | 非IID程度 |
| `mlp_hidden_dim` | 0 | 0-512 | 模型复杂度 |
| `mlp_dropout` | 0.0 | 0.0-0.5 | 正则化 |
| `train.optimizer.lr` | 0.001 | 0.0001-0.01 | 学习速度 |
| `train.local_update_steps` | 5 | 1-20 | 本地训练量 |

### 10.6.2 场景推荐配置

| 场景 | 推荐配置 |
|------|----------|
| 快速实验 | batch_size=64, rounds=30, num_generated=20 |
| 高精度 | rounds=100+, num_generated=100+, hidden_dim=256 |
| 重度非IID | alpha=0.1, num_generated=200, rounds=150 |
| 资源受限 | cnn提取器, batch_size=16, num_generated=10 |
| 分布式部署 | 参考第8章分布式配置 |

---

**第十章完成。** 以下是附录部分。

---

# 附录

## 附录A：常见问题FAQ

### A.1 安装与环境

**Q1: 如何安装open_clip？**
```bash
pip install open_clip_torch
```

**Q2: CLIP模型下载失败怎么办？**
```yaml
# 设置本地模型路径
ggeur:
  clip_model_path: '/path/to/ViT-B-16.pt'
```

**Q3: CUDA内存不足怎么办？**
- 减小`batch_size`
- 减小`num_generated_per_sample`
- 使用更小的CLIP模型（如ViT-B-32）

### A.2 数据与训练

**Q4: 如何查看特征缓存？**
```bash
ls ./clip_feature_cache/
python -c "import numpy as np; d=np.load('cache.npz'); print(d['features'].shape)"
```

**Q5: 训练不收敛怎么办？**
- 降低学习率
- 增加训练轮数
- 检查数据加载是否正确

**Q6: 如何验证LDS是否生效？**
```python
# 在日志中查看
# "LDS allocation: X/Y samples (Z%), N classes with data"
```

### A.3 性能调优

**Q7: 如何提高训练速度？**
- 使用特征缓存
- 减少`num_generated_per_sample`
- 使用更小的模型

**Q8: 如何提高准确率？**
- 增加训练轮数
- 增加增强样本数
- 使用更大的CLIP模型
- 添加隐藏层

---

## 附录B：配置项速查表

### B.1 基础配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ggeur.use` | bool | False | 启用GGEUR |
| `ggeur.statistics_round` | int | 0 | 统计收集轮 |
| `ggeur.feature_extractor` | str | 'clip' | 提取器类型 |
| `ggeur.embedding_dim` | int | 512 | 特征维度 |

### B.2 CLIP配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ggeur.clip_model` | str | 'ViT-B-16' | CLIP模型 |
| `ggeur.clip_pretrained` | str | 'openai' | 预训练来源 |
| `ggeur.clip_model_path` | str | '' | 本地路径 |

### B.3 增强配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ggeur.num_generated_per_sample` | int | 50 | 每样本生成数 |
| `ggeur.num_generated_per_prototype` | int | 50 | 每原型生成数 |
| `ggeur.target_size_per_class` | int | 50 | 目标样本数 |
| `ggeur.use_cross_client_prototypes` | bool | True | 跨域原型 |

### B.4 LDS配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ggeur.use_lds` | bool | False | 启用LDS |
| `ggeur.lds_alpha` | float | 0.1 | Dirichlet参数 |
| `ggeur.lds_seed` | int | 42 | 随机种子 |

### B.5 CNN配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ggeur.cnn_backbone` | str | 'convnext_base' | CNN架构 |
| `ggeur.cnn_model` | str | 'resnet18' | 蒸馏CNN |
| `ggeur.cnn_pretrained` | bool | True | 预训练 |
| `ggeur.cnn_lr` | float | 0.01 | CNN学习率 |
| `ggeur.cnn_local_epochs` | int | 10 | CNN本地轮数 |

---

## 附录C：性能基准测试

### C.1 Office-Home实验结果

| 方法 | Art | Clipart | Product | Real_World | 平均 |
|------|-----|---------|---------|------------|------|
| FedAvg | 58.2 | 52.1 | 67.3 | 71.5 | 62.3 |
| FedProx | 59.1 | 53.4 | 68.1 | 72.3 | 63.2 |
| MOON | 61.5 | 55.2 | 69.8 | 73.9 | 65.1 |
| **GGEUR** | **72.5** | **68.3** | **78.1** | **82.4** | **75.3** |

### C.2 PACS实验结果

| 方法 | Photo | Art | Cartoon | Sketch | 平均 |
|------|-------|-----|---------|--------|------|
| FedAvg | 92.1 | 78.5 | 73.2 | 68.9 | 78.2 |
| FedProx | 92.5 | 79.2 | 74.1 | 69.8 | 78.9 |
| **GGEUR** | **96.3** | **88.7** | **84.5** | **79.2** | **87.2** |

### C.3 LDS非IID实验（Office-Home, Alpha=0.1）

| 方法 | 平均准确率 | 最差域 | 收敛轮数 |
|------|------------|--------|----------|
| FedAvg | 45.2% | 38.1% | 不收敛 |
| FedProx | 48.7% | 41.3% | 150 |
| **GGEUR** | **68.5%** | **61.2%** | **80** |

---

## 附录D：参考文献

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. (CLIP)

2. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017. (FedAvg)

3. Li, T., et al. "Federated Optimization in Heterogeneous Networks." MLSys 2020. (FedProx)

4. Li, Q., et al. "Model-Contrastive Federated Learning." CVPR 2021. (MOON)

5. Venkateswara, H., et al. "Deep Hashing Network for Unsupervised Domain Adaptation." CVPR 2017. (Office-Home)

6. Li, D., et al. "Deeper, Broader and Artier Domain Generalization." ICCV 2017. (PACS)

7. Liu, Z., et al. "A ConvNet for the 2020s." CVPR 2022. (ConvNeXt)

---

# 文档结束

**文档版本：** 1.0
**适用框架：** FederatedScope
**最后更新：** 2025年1月

---

**感谢阅读！** 如有问题，请参考FAQ或提交Issue。
