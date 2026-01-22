# GGEUR 分布式实施细化规划

## 目标
将 GGEUR (CLIP/CNN) 从 standalone 模式迁移到分布式，保持算法行为不变。CLIP 权重统一使用 `/root/open_clip_vitb16.bin`。

## 当前问题
1. **数据加载**：[`ggeur_data.py`](federatedscope/contrib/data/ggeur_data.py:20) 仅支持 standalone，在内存中切分多客户端
2. **分布式加载器**：[`distributed_loader.py`](federatedscope/core/data/distributed_loader.py:23) 仅支持 FEMNIST
3. **缓存隔离**：Worker 未考虑多进程缓存冲突

## 实施计划

### 1. Office-Home 分片工具
**新增**：`scripts/tools/prepare_officehome_shards.py`

**参数**：
- `--root`：数据集路径（默认 `/root/OfficeHomeDataset_10072016`）
- `--output`：分片输出目录（默认 `data/officehome/shards`）
- `--clients`：客户端数量（60/8/4）
- `--lds-alpha`：Dirichlet 参数（默认 0.1）
- `--lds-seed`：随机种子（默认 42）
- `--splits`：train/val/test 划分（默认 0.7/0.0/0.3）

**输出结构**：
```
output/
├── manifest.json         # 全局元数据
└── client_{id}/
    ├── train.json        # {path, label} 列表
    ├── val.json
    ├── test.json
    └── meta.json         # 样本数、标签分布
```

**manifest.json 内容**：
```json
{
  "dataset": "office-home",
  "seed": 42,
  "lds_alpha": 0.1,
  "splits": [0.7, 0.0, 0.3],
  "total_clients": 60,
  "clients": [
    {
      "client_id": 1,
      "shard_path": "...",
      "num_samples": {...},
      "label_hist": {...}
    }
  ]
}
```

### 2. 分布式 Office-Home 加载器
**修改**：[`distributed_loader.py`](federatedscope/core/data/distributed_loader.py:23)

**实现**：
- 在 `_SUPPORTED_DATASETS` 添加 `'office-home'`
- 新增 `OfficeHomeShardDataset`：读取 JSON 中的路径+标签，按需加载图像
- 保留 FEMNIST 逻辑不变

**关键接口**：
```python
class OfficeHomeShardDataset(Dataset):
    def __init__(self, shard_json, transform):
        # 读取 {path, label} 列表
        # 支持按需加载图像
    
def load_distributed_data(config):
    if config.distributed_data.dataset == 'office-home':
        # 使用 OfficeHomeShardDataset
    elif config.distributed_data.dataset == 'femnist':
        # 现有 FEMNIST 逻辑
```

### 3. ggeur_data 分布式分支
**修改**：[`ggeur_data.py`](federatedscope/contrib/data/ggeur_data.py:20)

**逻辑**：
```python
if config.federate.mode == 'distributed' and config.distributed_data.enabled:
    if config.distribute.role == 'server':
        return None, config  # 占位
    else:
        # client 直接使用分布式 loader
        return load_distributed_data(config)
else:
    # 现有 standalone 逻辑不变
```

### 4. 缓存隔离
**修改**：
- [`ggeur_client.py:267`](federatedscope/contrib/worker/ggeur_client.py:267) `_get_feature_cache_path`
  - 添加 `client_{self.ID}` 子目录
  - 文件名包含模型类型（clip/cnn）
  
- [`ggeur_server.py:230`](federatedscope/contrib/worker/ggeur_server.py:230) `_get_test_cache_path`
  - 添加 `server` 前缀

**缓存路径示例**：
```
cache_dir/
├── client_1/
│   └── officehome_Art_clip_vitb16_openai.npz
├── client_2/
│   └── officehome_Clipart_clip_vitb16_openai.npz
└── server/
    └── officehome_test_clip_vitb16_openai.npz
```

### 5. CLIP 权重配置
**路径**：`/root/open_clip_vitb16.bin`

**配置**：
- 所有 YAML 中设置 `ggeur.clip_model_path: /root/open_clip_vitb16.bin`
- Server 需配置 `data.root: /root/OfficeHomeDataset_10072016`
- 在 `_load_clip_model` 中检查文件存在性，不存在则报错

### 6. 分布式配置文件
**新增**：`scripts/distributed_scripts/distributed_configs/`
- `ggeur_officehome_server.yaml`
- `ggeur_officehome_client_{1,2,3,4}.yaml`（4 客户端 dev 版）
- `ggeur_cnn_officehome_server.yaml`
- `ggeur_cnn_officehome_client_{1,2,3,4}.yaml`

**核心配置**：
```yaml
federate:
  mode: distributed
  client_num: 4

distribute:
  role: server|client
  client_id: <id>
  
distributed_data:
  enabled: True
  dataset: office-home
  shard_path: data/officehome/shards/client_<id>
  manifest: data/officehome/shards/manifest.json
  strict_check: True

ggeur:
  clip_model_path: /root/open_clip_vitb16.bin
  
data:
  type: office-home
  root: /root/OfficeHomeDataset_10072016
```

### 7. 启动脚本
**新增**：`scripts/distributed_scripts/run_distributed_ggeur_officehome_managed.sh`

**功能**：
- 启动 1 server + N clients
- PID 管理与清理（参考 FEMNIST 脚本）
- 启动前检查：
  - `manifest.json` 存在
  - 各 client shard 目录存在
  - `/root/open_clip_vitb16.bin` 存在

### 8. 验证方案
**最小测试**：1 server + 2/4 clients

**检查项**：
1. Client 上报统计量
2. Server 聚合协方差
3. Server 广播全局协方差/原型
4. Client 完成 augmentation
5. Round 1 FedAvg 正常运行
6. Server 评测加载 CLIP 权重

**单测**（可选）：
- 参考 [`test_distributed_femnist_loader.py`](tests/test_distributed_femnist_loader.py:12)
- 构造临时 manifest + 假图像路径
- 验证样本数、metadata 正确性

## 注意事项
1. **资源**：60 客户端压力大，优先 4/8 客户端验证
2. **可复现性**：manifest 必须记录 seed/alpha/splits
3. **并发安全**：缓存严格隔离 client/server
4. **依赖检查**：启动前确认 CLIP 权重文件存在
