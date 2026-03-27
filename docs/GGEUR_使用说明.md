# GGEUR 方法使用说明

GGEUR（Gaussian Geometry-guided Feature Expansion with Unified Representation）是一种联邦学习特征增强方法，支持 CNN（ConvNeXt/ResNet）和 ViT（CLIP）两种特征提取器。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [模型权重准备](#3-模型权重准备)
4. [启动方式](#4-启动方式)
5. [配置文件详解](#5-配置文件详解)
6. [CNN 模式（ConvNeXt）](#6-cnn-模式convnext)
7. [ViT 模式（CLIP）](#7-vit-模式clip)
8. [ggeur_mlp 模型介绍](#8-ggeur_mlp-模型介绍)
9. [常用配置文件索引](#9-常用配置文件索引)
10. [常见问题](#10-常见问题)

---

## 1. 环境准备

```bash
# 安装依赖
pip install -e .

# ViT/CLIP 模式额外需要
pip install open_clip_torch

# CNN 模式需要（torchvision 通常已包含）
pip install torchvision
```

---

## 2. 数据集准备

### Office-Home 数据集

**下载地址**：https://hemanthdv.github.io/officehome-dataset/

下载后解压，目录结构如下：

```
OfficeHomeDataset_10072016/
├── Art/
│   ├── Alarm_Clock/
│   ├── Backpack/
│   └── ...（65 个类别）
├── Clipart/
│   └── ...
├── Product/
│   └── ...
└── Real_World/
    └── ...
```

**数据集放置位置**：将 `OfficeHomeDataset_10072016/` 放在项目根目录下，或在配置文件中通过 `data.root` 指定绝对路径。

```yaml
data:
  type: 'office-home'
  root: 'OfficeHomeDataset_10072016'   # 相对路径（相对于运行目录）
  # 或使用绝对路径：
  # root: '/data/OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]             # 训练/验证/测试比例
```

### PACS 数据集（可选）

```
PACS/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

```yaml
data:
  type: 'pacs'
  root: 'PACS'
```

---

## 3. 模型权重准备

### CNN 模式（ConvNeXt / ResNet）

CNN 模式使用 torchvision 的预训练权重，**首次运行会自动从网络下载**。

如需离线使用，可手动下载后放置到 torch 缓存目录（通常为 `~/.cache/torch/hub/checkpoints/`）。

| 骨干网络 | 特征维度 | 说明 |
|---|---|---|
| `convnext_base` | 1024 | 推荐，精度与速度平衡最佳 |
| `convnext_tiny` | 768 | 轻量版 |
| `convnext_small` | 768 | 轻量版 |
| `convnext_large` | 1536 | 高精度，显存需求更大 |
| `resnet18` | 512 | 轻量，速度快 |
| `resnet50` | 2048 | 标准 ResNet |

### ViT 模式（CLIP）

CLIP 模式需要 `open_clip` 预训练权重。

**方式一：自动下载（需要网络）**

```yaml
ggeur:
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: ''   # 留空则自动下载
```

**方式二：本地权重（推荐离线环境）**

从 OpenAI 或 HuggingFace 下载 CLIP ViT-B/16 权重，保存为 `.bin` 文件：

```yaml
ggeur:
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/model/open_clip_vitb16.bin'  # 本地权重路径
  embedding_dim: 512
```

| CLIP 模型 | 特征维度 | 说明 |
|---|---|---|
| `ViT-B-16` | 512 | 推荐，精度高 |
| `ViT-B-32` | 512 | 速度更快 |
| `ViT-L-14` | 768 | 更高精度，显存需求大 |

---

## 4. 启动方式

### 使用 run.py（推荐）

```bash
# 在项目根目录下运行
python run.py --cfg scripts/example_configs/ggeur_cnn_convnext.yaml
```

### 使用 federatedscope.main

```bash
python -m federatedscope.main --cfg scripts/example_configs/ggeur_cnn_convnext.yaml
```

### 覆盖配置参数

可以在命令行直接覆盖配置文件中的参数：

```bash
# 修改设备
python run.py --cfg scripts/example_configs/ggeur_cnn_convnext.yaml device 1

# 修改轮次和客户端数
python run.py --cfg scripts/example_configs/ggeur_cnn_convnext.yaml \
  federate.total_round_num 50 \
  federate.client_num 60

# 修改数据集路径
python run.py --cfg scripts/example_configs/ggeur_cnn_convnext.yaml \
  data.root /data/OfficeHomeDataset_10072016
```

---

## 5. 配置文件详解

### 基础结构

```yaml
# ========== 基础设置 ==========
use_gpu: True
device: 0          # GPU 编号
seed: 42
verbose: 1

# ========== 联邦学习设置 ==========
federate:
  method: 'ggeur'          # 必须为 'ggeur'
  mode: 'standalone'       # 单机模拟
  client_num: 60           # 客户端总数（LDS 模式下为虚拟客户端数）
  total_round_num: 100     # 总训练轮次
  sample_client_num: 0     # 每轮参与客户端数（0 = 全部参与）

# ========== 数据设置 ==========
data:
  type: 'office-home'
  root: 'OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 32
  num_workers: 0

# ========== 模型设置 ==========
model:
  type: 'ggeur_mlp'        # 固定为 ggeur_mlp
  num_classes: 65          # Office-Home 为 65 类

# ========== 训练设置 ==========
train:
  local_update_steps: 10   # 每轮本地训练步数
  optimizer:
    type: 'Adam'
    lr: 0.001

# ========== GGEUR 专属设置 ==========
ggeur:
  use: True
  # ... 见下方各模式说明

# ========== 输出设置 ==========
outdir: 'exp/my_experiment'
expname: 'run_name'
```

### ggeur 配置项完整说明

**基础开关**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use` | `False` | 是否启用 GGEUR，必须设为 `True` |

**特征提取器选择**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `feature_extractor` | `'clip'` | 特征提取器类型：`'clip'`（CLIP ViT）、`'cnn'`（ConvNeXt/ResNet 等）、`'timm'`（任意 timm 模型） |
| `embedding_dim` | `512` | 特征向量维度，必须与所选提取器的输出维度一致 |

**CLIP 相关（`feature_extractor: 'clip'` 时生效）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `clip_model` | `'ViT-B-16'` | CLIP 骨干网络，可选 `ViT-B-16`、`ViT-B-32`、`ViT-L-14` |
| `clip_pretrained` | `'openai'` | 预训练权重来源，通常填 `'openai'` |
| `clip_model_path` | `''` | 本地 CLIP 权重文件路径（`.bin`），留空则自动从网络下载 |

**CNN 相关（`feature_extractor: 'cnn'` 时生效）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `cnn_backbone` | `'convnext_base'` | CNN 骨干网络，支持 `convnext_tiny/small/base/large`、`resnet18/34/50/101`、`efficientnet_b0/b4` 等 |
| `cnn_pretrained` | `True` | 是否加载 ImageNet 预训练权重 |
| `freeze_backbone` | `True` | 是否冻结骨干网络参数（推荐开启，仅用骨干提取特征） |

**timm 相关（`feature_extractor: 'timm'` 时生效）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `timm_model` | `'mixer_b16_224'` | timm 模型 ID，如 `gfnet_tiny_patch4_224`、`mixer_b16_224` |
| `timm_pretrained` | `True` | 是否加载 timm 预训练权重 |
| `timm_checkpoint_path` | `''` | 本地权重文件路径，设置后忽略 `timm_pretrained` |

**特征缓存**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_feature_cache` | `True` | 是否将提取的特征缓存到磁盘，重复实验时直接加载，避免重复提取 |
| `feature_cache_dir` | `''` | 缓存目录路径，留空则自动放在 `data.root` 同级目录 |

**特征增强**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_generated_per_sample` | `50` | 对每个原始样本，以其特征为均值、全局协方差为方差，生成多少个高斯增强样本 |
| `num_generated_per_prototype` | `50` | 对其他客户端上传的每个类原型，生成多少个高斯增强样本 |
| `target_size_per_class` | `50` | 增强后每个类的最终样本数上限，`0` 表示不限制 |
| `use_cross_client_prototypes` | `True` | 是否利用其他客户端的类原型参与增强，关闭后仅用本地样本生成 |

**MLP 分类器**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mlp_hidden_dim` | `0` | MLP 隐藏层维度，`0` 表示线性分类器（论文默认），`>0` 则加一层 `Linear→ReLU→Dropout` |
| `mlp_dropout` | `0.0` | Dropout 比率，仅在 `mlp_hidden_dim > 0` 时生效 |

**训练控制**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `statistics_round` | `0` | 收集本地统计量（均值、协方差）的轮次编号，通常为 `0`（第一轮） |

**LDS（标签分布偏斜）数据划分**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_lds` | `False` | 是否用 Dirichlet 分布将数据集划分为非IID的多客户端分区 |
| `lds_alpha` | `0.1` | Dirichlet 分布的 α 参数，越小数据分布越不均匀（`0.1` 为论文默认，高度非IID） |
| `lds_seed` | `42` | 数据划分的随机种子，保证实验可复现 |

**FedProto 集成（可选）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_fedproto` | `False` | 是否在 MLP 训练时加入原型正则化损失 |
| `proto_weight` | `1.0` | 原型损失权重，总损失 = CE + `proto_weight` × 原型损失 |
| `proto_distance` | `'cosine'` | 原型距离度量方式：`'euclidean'` 或 `'cosine'` |
| `proto_temperature` | `0.1` | cosine 距离的温度系数 |

**PromptFL（可选，仅 `feature_extractor: 'clip'` 时有效）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_promptfl` | `False` | 是否启用联邦软提示训练（CoOp 风格），开启后联邦聚合 prompt 向量而非 MLP |
| `prompt_length` | `16` | 可学习的 context token 数量 |
| `prompt_lr` | `0.002` | prompt 优化器学习率 |
| `prompt_local_epochs` | `10` | 每轮 prompt 本地训练轮数 |
| `prompt_template` | `'a photo of a {}'` | 文本提示模板，`{}` 替换为类别名 |
| `hf_clip_model_id` | `'openai/clip-vit-base-patch16'` | HuggingFace CLIP 模型 ID 或本地路径 |

**MOON（可选）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_moon` | `False` | 是否启用 MOON 对比学习正则化 |
| `moon_mu` | `5.0` | 对比损失权重 |
| `moon_temperature` | `0.5` | 对比损失温度系数 |

---

## 6. CNN 模式（ConvNeXt）

### 最小配置示例

```yaml
use_gpu: True
device: 0
seed: 42

federate:
  method: 'ggeur'
  mode: 'standalone'
  client_num: 60
  total_round_num: 100
  sample_client_num: 0

data:
  type: 'office-home'
  root: 'OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 16
  num_workers: 0

model:
  type: 'ggeur_mlp'
  num_classes: 65

train:
  local_update_steps: 10
  optimizer:
    type: 'Adam'
    lr: 0.001

ggeur:
  use: True
  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'   # 骨干网络
  cnn_pretrained: True             # 使用 ImageNet 预训练权重
  freeze_backbone: True            # 冻结骨干（推荐）
  embedding_dim: 1024              # 必须与 convnext_base 输出维度匹配
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  mlp_hidden_dim: 0
  use_cross_client_prototypes: True
  use_lds: True
  lds_alpha: 0.1
  use_feature_cache: False

outdir: 'exp/ggeur_cnn'
expname: 'cnn_convnext_base'
```

### 启动命令

```bash
python run.py --cfg scripts/example_configs/ggeur_cnn_convnext.yaml
```

### 不同骨干网络的 embedding_dim 对应关系

```yaml
# convnext_tiny 或 convnext_small
cnn_backbone: 'convnext_tiny'
embedding_dim: 768

# convnext_base（推荐）
cnn_backbone: 'convnext_base'
embedding_dim: 1024

# convnext_large
cnn_backbone: 'convnext_large'
embedding_dim: 1536

# resnet18 或 resnet34
cnn_backbone: 'resnet18'
embedding_dim: 512

# resnet50
cnn_backbone: 'resnet50'
embedding_dim: 2048
```

---

## 7. ViT 模式（CLIP）

### 最小配置示例

```yaml
use_gpu: True
device: 0
seed: 42

federate:
  method: 'ggeur'
  mode: 'standalone'
  client_num: 60
  total_round_num: 100
  sample_client_num: 0

data:
  type: 'office-home'
  root: 'OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 32
  num_workers: 0

model:
  type: 'ggeur_mlp'
  num_classes: 65

train:
  local_update_steps: 10
  optimizer:
    type: 'Adam'
    lr: 0.0001

ggeur:
  use: True
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: ''              # 留空自动下载，或填本地路径
  embedding_dim: 512
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
  mlp_hidden_dim: 0
  use_cross_client_prototypes: True
  use_lds: True
  lds_alpha: 0.1
  use_feature_cache: True          # CLIP 提取较慢，建议开启缓存

outdir: 'exp/ggeur_vit'
expname: 'vit_clip_fedavg'
```

### 启动命令

```bash
# 使用 vit_comparison 目录下的配置
python run.py --cfg scripts/example_configs/vit_comparison/1_ggeur_clip_fedavg.yaml

# 使用 ggeur_baseline_vit 目录下的配置
python run.py --cfg scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml
```

### 本地 CLIP 权重配置

```yaml
ggeur:
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/model/open_clip_vitb16.bin'
  embedding_dim: 512
```

---

## 8. ggeur_mlp 模型介绍

`ggeur_mlp` 是 GGEUR 方法的分类器模型，定义在 `federatedscope/contrib/model/ggeur_mlp.py`。

### 模型结构

```
输入特征 (embedding_dim)
       ↓
[可选] Linear(embedding_dim → mlp_hidden_dim)
       ↓
[可选] ReLU + Dropout(mlp_dropout)
       ↓
Linear(embedding_dim 或 mlp_hidden_dim → num_classes)
       ↓
输出 logits (num_classes)
```

模型有两种形态，由 `mlp_hidden_dim` 控制：

**线性分类器（`mlp_hidden_dim: 0`，论文默认）**

```
Linear(512 → 65)
```

直接将特征映射到类别，参数量最少，泛化性好，是论文中的默认设置。

**带隐藏层的 MLP（`mlp_hidden_dim > 0`）**

```
Linear(512 → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 65)
```

增加非线性表达能力，适合特征维度较高或类别较多的场景。

### 配置参数

在配置文件中通过 `ggeur` 节控制 MLP 结构：

```yaml
model:
  type: 'ggeur_mlp'
  num_classes: 65        # 分类数，Office-Home 为 65

ggeur:
  embedding_dim: 512     # 输入维度，需与特征提取器输出匹配
  mlp_hidden_dim: 0      # 0 = 线性分类器；>0 = 加一层隐藏层
  mlp_dropout: 0.0       # Dropout 比率（仅在 mlp_hidden_dim > 0 时生效）
```

### 联邦聚合方式

`ggeur_mlp` 的参数通过 FedAvg 在服务端聚合。每轮训练流程：

1. 客户端在增强后的特征上本地训练 MLP（`train.local_update_steps` 步）
2. 上传 MLP 参数到服务端
3. 服务端 FedAvg 聚合，得到全局 MLP
4. 下发全局 MLP 到各客户端，开始下一轮

---

## 9. 常用配置文件索引

### CNN 系列

| 文件路径 | 说明 |
|---|---|
| `scripts/example_configs/ggeur_cnn_convnext.yaml` | CNN + ConvNeXt-Base + GGEUR（主配置） |
| `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_ggeur_fedavg.yaml` | CNN + GGEUR + FedAvg |
| `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedavg.yaml` | CNN + FedAvg（无 GGEUR，基线） |
| `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedprox.yaml` | CNN + FedProx（基线） |
| `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedproto.yaml` | CNN + FedProto（基线） |
| `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_moon.yaml` | CNN + MOON（基线） |
| `scripts/example_configs/cnn_comparison/1_ggeur_convnext_fedavg.yaml` | ConvNeXt + GGEUR + FedAvg |
| `scripts/example_configs/cnn_comparison/2_ggeur_convnext_fedprox.yaml` | ConvNeXt + GGEUR + FedProx |
| `scripts/example_configs/cnn_comparison/5_ggeur_convnext_fedproto.yaml` | ConvNeXt + GGEUR + FedProto |

### ViT 系列

| 文件路径 | 说明 |
|---|---|
| `scripts/example_configs/ggeur_officehome_lds.yaml` | CLIP + GGEUR + LDS（主配置） |
| `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml` | CLIP + GGEUR + FedAvg |
| `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedavg.yaml` | CLIP + FedAvg（无 GGEUR，基线） |
| `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedprox.yaml` | CLIP + FedProx（基线） |
| `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedproto.yaml` | CLIP + FedProto（基线） |
| `scripts/example_configs/vit_comparison/1_ggeur_clip_fedavg.yaml` | CLIP + GGEUR + FedAvg |
| `scripts/example_configs/vit_comparison/2_ggeur_clip_fedprox.yaml` | CLIP + GGEUR + FedProx |
| `scripts/example_configs/vit_comparison/3_ggeur_clip_fedproto.yaml` | CLIP + GGEUR + FedProto |
| `scripts/example_configs/vit_comparison/5_ggeur_clip_fedopt.yaml` | CLIP + GGEUR + FedOpt |

---

## 10. 常见问题

**Q: 运行时提示找不到数据集？**

检查 `data.root` 路径是否正确。路径相对于运行命令时的工作目录（通常是项目根目录）。建议使用绝对路径：

```yaml
data:
  root: 'D:/data/OfficeHomeDataset_10072016'
```

**Q: CNN 模式下 embedding_dim 报错？**

`embedding_dim` 必须与 `cnn_backbone` 的输出维度严格匹配，参考第 6 节的对应关系表。

**Q: CLIP 权重下载失败？**

设置 `clip_model_path` 指向本地已下载的权重文件，或配置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 显存不足？**

- 减小 `dataloader.batch_size`（如从 32 改为 16）
- 使用更小的骨干网络（`convnext_tiny` 代替 `convnext_base`）
- 确保 `freeze_backbone: True`（CNN 模式）

**Q: 如何加速重复实验？**

开启特征缓存，第一次运行后特征会保存到磁盘，后续实验直接加载：

```yaml
ggeur:
  use_feature_cache: True
  feature_cache_dir: 'feature_cache'  # 留空则自动放在 data.root 旁边
```

**Q: 如何复现论文结果？**

使用以下设置：

```yaml
ggeur:
  use_lds: True
  lds_alpha: 0.1
  num_generated_per_sample: 100
  num_generated_per_prototype: 100
  target_size_per_class: 100
  mlp_hidden_dim: 0       # 线性分类器
  use_cross_client_prototypes: True
```
