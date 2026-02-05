# Office-Home分片工具开发周报

## 一、工作概述

开发了Office-Home数据集的分布式训练分片工具，支持多客户端数据划分、LDS(Label Distribution Skew)非独立同分布数据生成，以及标准化的元数据管理。该工具为后续GGEUR方法的分布式训练提供数据准备支持。

## 二、工具功能

### 2.1 核心功能

**文件位置**：`scripts/tools/prepare_officehome_shards.py`

1. **多客户端数据分片**
   - 支持任意客户端数量（必须被 4 整除，对应 4 个domain）
   - 每个domain平均分配客户端
   - 自动处理类别平衡的训练/验证/测试集划分

2. **LDS(Label Distribution Skew) 支持**
   - 基于Dirichlet分布实现非独立同分布数据划分
   - 可配置alpha参数控制数据偏斜程度（alpha越小，偏斜越严重）
   - 可选关闭LDS，使用均匀分布

3. **数据集划分**
   - 支持自定义train/val/test比例
   - 类别平衡的分层划分（stratified split）
   - 可配置随机种子保证可复现性

4. **元数据管理**
   - 生成全局 `manifest.json` 记录所有客户端信息
   - 每个客户端生成独立的 `meta.json` 记录本地统计信息
   - 记录标签分布直方图便于数据分析

### 2.2 命令行参数

```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \  # 数据集根路径
    --output data/officehome/shards \           # 分片输出目录
    --clients 4 \                               # 客户端数量（必须被4整除）
    --lds-alpha 0.1 \                           # Dirichlet alpha（0表示关闭LDS）
    --lds-seed 42 \                             # 随机种子
    --splits 0.7 0.0 0.3 \                     # train/val/test比例
    --manifest data/officehome/shards/manifest.json \  # manifest输出路径（可选）
    --overwrite                                 # 覆盖已存在的分片
```

### 2.3 输出结构

```
data/officehome/shards/
├── manifest.json                    # 全局元数据
├── client_1/                        # Art domain
│   ├── train.json                   # 训练集 [{path, label}, ...]
│   ├── val.json                     # 验证集（可选，取决于 splits）
│   ├── test.json                    # 测试集
│   └── meta.json                    # 客户端元数据
├── client_2/                        # Clipart domain
│   └── ...
├── client_3/                        # Product domain
│   └── ...
└── client_4/                        # Real World domain
    └── ...
```

### 2.4 数据格式

**train/val/test.json格式**：
```json
[
  {
    "path": "/root/OfficeHomeDataset_10072016/Art/Alarm_Clock/00001.jpg",
    "label": 0
  },
  {
    "path": "/root/OfficeHomeDataset_10072016/Art/Alarm_Clock/00002.jpg",
    "label": 0
  }
]
```

**meta.json格式**：
```json
{
  "num_samples": {
    "train": 1450,
    "val": 0,
    "test": 622
  },
  "label_hist": {
    "0": 45,
    "1": 38,
    "2": 52,
    ...
  }
}
```

**manifest.json格式**：
```json
{
  "dataset": "office-home",
  "root": "/root/OfficeHomeDataset_10072016",
  "seed": 42,
  "lds_alpha": 0.1,
  "splits": [0.7, 0.0, 0.3],
  "total_clients": 4,
  "clients": [
    {
      "client_id": 1,
      "shard_path": "/root/FederatedScope/data/officehome/shards/client_1",
      "num_samples": {
        "train": 1450,
        "val": 0,
        "test": 622
      },
      "label_hist": {
        "0": 45,
        "1": 38,
        ...
      }
    },
    ...
  ]
}
```

---

## 三、技术实现

### 3.1 核心算法

#### 1. 数据加载（第 62-92 行）
```python
def load_office_home_data(root: str, domain: str) -> Tuple[List[str], List[int]]:
    """加载指定domain的所有图像路径和标签"""
    - 扫描 domain 目录下的所有类别子目录
    - 自动构建 class_to_idx 映射
    - 返回图像路径列表和对应标签列表
```

#### 2. 分层划分（第 95-144 行）
```python
def split_data(image_paths, labels, splits, seed):
    """按类别进行分层划分，保证每个类别的比例一致"""
    - 按类别分组样本
    - 对每个类别按比例划分 train/val/test
    - 保证类别平衡性
```

#### 3.LDS生成（第 147-218 行）
```python
def apply_lds(split_data, num_clients_per_domain, alpha, seed):
    """使用 Dirichlet 分布生成非独立同分布数据"""
    - 为每个类别生成 Dirichlet 分布矩阵
    - 矩阵形状：(num_clients, num_classes)
    - 根据分布比例分配样本到各客户端
    - 处理舍入误差（分配剩余样本到比例最高的客户端）
```

#### 4. 分片保存（第 227-282 行）
```python
def save_shard(output_dir, split_data, client_train_data):
    """保存客户端分片数据"""
    - 保存train.json（客户端独有）
    - 保存val.json（domain内共享）
    - 保存test.json（domain内共享）
    - 保存meta.json（统计信息）
```

### 3.2 设计亮点

1. **参考FEMNIST工具设计**
   - 保持与 `prepare_femnist_shards.py` 一致的接口风格
   - 兼容FederatedScope的分布式数据加载器
   - 统一的manifest格式

2. **灵活的配置选项**
   - LDS可开可关（alpha=0 关闭）
   - 支持任意train/val/test比例
   - 客户端数量可配置（4/8/60）

3. **错误处理与验证**
   - 检查客户端数量必须被 4 整除
   - 检查domain目录存在性
   - 防止覆盖已存在的分片（除非 --overwrite）

4. **可复现性保障**
   - manifest记录完整的划分参数
   - 随机种子固定
   - 输出绝对路径便于追溯

---

## 四、测试与验证

### 4.1 测试配置

**测试命令**：
```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \
    --output data/officehome/shards \
    --clients 4 \
    --lds-alpha 0.1 \
    --lds-seed 42 \
    --splits 0.7 0.0 0.3
```

### 4.2 验证结果

**数据完整性**：
```bash
$ ls -la data/officehome/shards/
drwxr-xr-x 6 root root 4096 Jan 22 11:56 .
drwxr-xr-x 3 root root 4096 Jan 22 11:56 ..
drwxr-xr-x 2 root root 4096 Jan 22 11:56 client_1  # Art domain
drwxr-xr-x 2 root root 4096 Jan 22 11:56 client_2  # Clipart domain
drwxr-xr-x 2 root root 4096 Jan 22 11:56 client_3  # Product domain
drwxr-xr-x 2 root root 4096 Jan 22 11:56 client_4  # Real World domain
-rw-r--r-- 1 root root 5771 Jan 22 11:56 manifest.json
```

**各客户端数据统计**（从 manifest.json）：
| Client ID | Domain | Train | Val | Test | Total |
|-----------|--------|-------|-----|------|-------|
| 1 | Art | 1,450 | 0 | 622 | 2,072 |
| 2 | Clipart | 3,048 | 0 | 1,307 | 4,355 |
| 3 | Product | 2,902 | 0 | 1,244 | 4,146 |
| 4 | Real World | 2,952 | 0 | 1,266 | 4,218 |
| Total | - | 10,352 | 0 | 4,439 | 14,791 |

**LDS验证**：
- 标签分布直方图显示各客户端类别分布不均衡
- alpha=0.1 产生明显的数据偏斜
- 符合联邦学习非独立同分布场景

**文件格式验证**：
- 所有JSON文件格式正确
- 图像路径均为绝对路径
- 标签范围：0-64（65 个类别）

---

## 五、辅助功能开发

### 5.1Manifest工具模块

**文件位置**：[`federatedscope/core/auxiliaries/manifest_utils.py`](../federatedscope/core/auxiliaries/manifest_utils.py:1)

**功能**：
1. `load_manifest(path, strict=True)`
   - 加载并解析manifest.json
   - 可选的严格模式校验

2. `find_client_in_manifest(manifest, client_id, shard_path)`
   - 根据客户端ID或分片路径查找客户端元数据
   - 返回匹配的客户端信息字典

3. `summarize_manifest(manifest)`
   - 生成manifest摘要信息
   - 包括数据集名称、客户端数量、总样本数等

**设计价值**：
- 统一manifest处理逻辑
- 减少重复代码
- 便于后续分布式加载器集成

---

## 六、使用示例

### 6.1 生成 4 客户端分片（默认配置）
```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \
    --output data/officehome/shards \
    --clients 4
```

### 6.2 生成 8 客户端分片（LDS alpha=0.5）
```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \
    --output data/officehome/shards_8clients \
    --clients 8 \
    --lds-alpha 0.5 \
    --overwrite
```

### 6.3 生成 60 客户端分片（无 LDS）
```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \
    --output data/officehome/shards_60clients \
    --clients 60 \
    --lds-alpha 0 \
    --splits 0.8 0.0 0.2
```

### 6.4 自定义 manifest 路径
```bash
python scripts/tools/prepare_officehome_shards.py \
    --root /root/OfficeHomeDataset_10072016 \
    --output data/officehome/shards \
    --clients 4 \
    --manifest /tmp/my_manifest.json
```

---

## 七、后续工作

基于已完成的分片工具，下一步将进行GGEUR方法的完整分布式迁移实现：

### 7.1 分布式数据加载器开发

- [ ] 在 `distributed_loader.py` 中增加Office-Home支持
- [ ] 实现 `OfficeHomeShardDataset` 类，按需加载图像
- [ ] 集成分片工具生成的manifest和元数据
- [ ] 验证数据加载正确性

### 7.2GGEUR数据加载分布式分支

- [ ] 修改 `ggeur_data.py` 添加分布式模式检测
- [ ] 分布式模式调用 `load_distributed_data()`
- [ ] 保持standalone模式逻辑不变
- [ ] 测试两种模式切换

### 7.3Worker缓存隔离

- [ ] Client缓存路径添加 `client_id` 隔离
- [ ] Server缓存路径独立管理
- [ ] 防止多进程缓存冲突
- [ ] 验证特征缓存正确性

### 7.4 分布式配置与脚本

- [ ] 编写server和client配置文件
- [ ] 统一CLIP模型路径配置
- [ ] 开发启动和停止脚本
- [ ] 进程管理和错误处理

### 7.5 完整验证

- [ ] 4 客户端分布式训练验证
- [ ] 特征提取和统计聚合测试
- [ ] 数据增强和模型训练测试
- [ ] 与standalone模式结果对比

---

## 八、总结

**已完成**：
1. 完整实现Office-Home分片工具
2. 支持LDS非独立同分布数据生成
3. 生成标准化的manifest和元数据
4. 验证 4 客户端配置正常工作
5. 开发配套的manifest工具模块


