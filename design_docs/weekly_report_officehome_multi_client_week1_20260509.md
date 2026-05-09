# 周报（Office-Home多域多客户端——数据分片与元信息改造）

## 1. 背景
- 将"每域1 client"的静态切分，升级为"每域多个client + manifest显式domain元信息"的数据准备能力

## 2. 改造内容

### 2.1 改造前现状
- `prepare_officehome_shards.py`固定4个域各1个client
- 当`clients`不能被4整除时，脚本静默降级为4 clients，无任何警告
- `manifest.json`和`meta.json`中不包含`domain`字段
- 没有最小样本保护机制，极端Dirichlet参数下可能产生空shard

### 2.2 改造后能力
- 支持`clients=4k`（k≥1），每个域生成k个client shard
- 当`clients % 4 != 0`时，脚本以可读错误信息退出，不再静默降级
- `manifest.json`顶层新增`clients_per_domain`、`domains`、`min_samples_per_client`字段
- `manifest.json`每个client条目和每个shard的`meta.json`均包含`domain`字段
- 域内多client通过Dirichlet分布实现label skew
- 新增最小样本保护机制（`--min-samples-per-client`），无法满足时显式报错

## 3. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `scripts/tools/prepare_officehome_shards.py` | 新增`validate_args()`严格校验；新增`--min-samples-per-client`参数；提取`split_uniformly()`函数；新增`enforce_min_samples_per_client()`最小样本保护；manifest/meta.json增加`domain`等字段；移除静默降级逻辑 |

## 4. 关键设计决策

### 4.1 校验策略
将`clients % 4 != 0`从"静默降级"改为"显式失败"。

理由：
- 静默降级会导致用户对实际实验规模产生误解
- 在后续自动化流程中，这种静默行为更加危险
- 显式失败配合清晰错误信息，可避免后续排障成本

### 4.2 最小样本保护修复策略
采用简单的"从最大shard搬运到最小shard"策略：

```python
def enforce_min_samples_per_client(client_train_data, min_samples_per_client, domain):
    # 1. 先检查总样本数是否能满足所有client的最小要求
    # 2. 循环：从最大shard搬运样本到最小shard
    # 3. 若无法满足约束，直接报错退出
```

理由：
- 保持切分后的分布特征尽可能接近原始Dirichlet结果
- 只在极端情况下做最小修复，不引入复杂重分配逻辑
- 无法满足时直接报错，避免生成质量过差的shard

### 4.3 manifest元信息增强
新增字段使下游代码无需依赖"client编号区间隐式代表域"的脆弱假设：

```json
{
  "clients_per_domain": 3,
  "domains": ["Art", "Clipart", "Product", "Real World"],
  "min_samples_per_client": 32,
  "clients": [
    {"client_id": 1, "domain": "Art", ...},
    ...
  ]
}
```

## 5. 验证结果

### 5.1 参数校验测试
| 测试用例 | 预期行为 | 实际结果 |
|----------|----------|----------|
| `--clients 10` | 报错退出 | ✓ 报错："must be divisible by 4" |
| `--clients 0` | 报错退出 | ✓ 报错："must be positive" |
| `--min-samples-per-client -1` | 报错退出 | ✓ 报错："must be positive" |
| `--splits 0.5 0.2 0.4` | 报错退出 | ✓ 报错："must sum to 1.0" |

### 5.2 正常生成测试
| clients数 | 每域clients | 生成结果 |
|-----------|-------------|----------|
| 4 | 1 | ✓ 成功，与原有行为兼容 |
| 8 | 2 | ✓ 成功，每域2个shard |
| 12 | 3 | ✓ 成功，每域3个shard |

### 5.3 最小样本保护测试
| 测试用例 | 预期行为 | 实际结果 |
|----------|----------|----------|
| `--clients 12 --min-samples-per-client 1000` | Art域无法满足，报错退出 | ✓ 报错："Domain 'Art' has only 1677 train samples..." |

### 5.4 域内label skew验证（12 clients, alpha=0.1, Art域）
| 指标 | Client 1 | Client 2 | Client 3 |
|------|----------|----------|----------|
| 训练样本数 | 554 | 610 | 513 |
| 非零类数 | 32/65 | 37/65 | 25/65 |
| Top-1类占比 | 11.2%(类22) | 8.4%(类0) | 13.5%(类5) |

两两L1距离（归一化分布）：
- Client 1 vs Client 2: 1.83
- Client 1 vs Client 3: 1.88
- Client 2 vs Client 3: 1.85

L1最大值为2.0，表明Dirichlet(alpha=0.1)下同域各client的标签分布已高度异构。

## 6. 遗留问题
- 当前均匀切分模式下各shard大小几乎相同（如839/838），后续可考虑是否也需要样本量层面的异构
- 第2周需在配置生成脚本中消费manifest的`domain`字段，完成启动链路动态化

## 7. 下周计划
- 设计并实现配置生成脚本，根据manifest自动生成N份client YAML
- 改造`run_distributed_ggeur_officehome.sh`支持动态client数
- 增加启动前校验：manifest中client数与`federate.client_num`一致、shard路径存在性等
