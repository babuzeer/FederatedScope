# GGEUR / OfficeHome / DomainNet 协作者配置与运行说明

本文档用于协作者快速定位上次实验使用的两类数据集、三类模型分支、GGEUR 与五个对比方法的配置文件、服务器路径和运行方式。

> 说明：本文档基于本仓库当前配置文件、运行脚本、`.vscode/sftp.json` 和部署脚本整理。2026-07-14 通过当前环境尝试 SSH 在线核实服务器路径时，`bjb1.seetacloud` 连接被远端关闭，`10.112.81.135` 直连被拒绝，因此服务器路径按仓库记录给出，首次运行前建议在服务器上执行本文的检查命令再确认一次。

## 1. 实验矩阵口径

本轮说明覆盖 2 个数据集：

- `officehome`：OfficeHome-LDS，配置中 `data.type: office-home`，默认 65 类。
- `domainnet`：DomainNet 4 domains，配置中 `data.type: domainnet`，默认使用 `clipart / painting / real / sketch`，当前配置为 345 类。

三类模型分支实际含义如下：

| 模型分支 | 实际特征提取器 | OfficeHome 配置目录 | DomainNet 配置目录 |
|---|---|---|---|
| `vit` | CLIP `ViT-B-16`，输出 512 维特征 | `scripts/example_configs/ggeur_baseline_vit/` | `scripts/example_configs/ggeur_baseline_vit_domainnet/` |
| `cnn` | `torchvision` ConvNeXt-Base，输出 1024 维特征 | `scripts/example_configs/ggeur_baseline_cnn/` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/` |
| `mixer` / `mlp` | `timm` 的 MLP-Mixer `mixer_b16_224`，输出 768 维特征 | `scripts/example_configs/ggeur_baseline_mixer/` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/` |

注意：DomainNet 目录名使用 `mlp`，但配置中实际是 `feature_extractor: timm` + `timm_model: mixer_b16_224`，与 OfficeHome 的 `mixer` 分支是同类模型。

方法口径：

- `ggeur`：GGEUR + FedAvg。通过 `num_generated_per_sample / num_generated_per_prototype / target_size_per_class` 大于 0 开启特征增强。
- 五个对比方法：`fedavg`、`fedprox`、`fedproto`、`fedopt`、`moon`。这些配置仍复用 GGEUR 管线做特征抽取和 MLP 训练，但把 GGEUR 生成数量设为 0；方法差异通过 FedProx/FedProto/FedOpt/MOON 对应参数打开。

## 2. 服务器路径

仓库记录中的服务器与路径如下：

| 内容 | 服务器路径 / 地址 | 来源 |
|---|---|---|
| 4090 服务器 | `10.112.81.135`，用户 `root` | `.vscode/sftp.json`、部署文档 |
| SSH 别名 | `bjb1.seetacloud` -> `connect.bjb1.seetacloud.com:51467` | 本机 SSH 配置 |
| 项目目录 | `/root/autodl-tmp/FederatedScope` | `.vscode/sftp.json` |
| Python 解释器 | `/root/miniconda3/envs/fs/bin/python` | `.vscode/settings.json`、运行脚本 |
| OfficeHome 数据集 | `/root/autodl-tmp/datasets/OfficeHomeDataset_10072016` | YAML 配置、部署脚本 |
| DomainNet 数据集 | `/root/autodl-tmp/datasets/DomainNet` | YAML 配置、DomainNet 分布式脚本 |
| CLIP ViT-B/16 权重 | `/root/autodl-tmp/models/open_clip_vitb16.bin` | YAML 配置、部署脚本 |
| MLP-Mixer 权重 | `/root/autodl-tmp/models/mixer_b16_224_complete.pth` | Mixer/DomainNet MLP YAML 配置 |
| CNN 权重 | 通常走 `torchvision` 预训练权重缓存，如 `~/.cache/torch/hub/checkpoints/` | CNN YAML 使用 `cnn_pretrained: True`，未指定项目内 checkpoint |

服务器上建议先执行：

```bash
cd /root/autodl-tmp/FederatedScope

ls -ld \
  /root/autodl-tmp/FederatedScope \
  /root/autodl-tmp/datasets/OfficeHomeDataset_10072016 \
  /root/autodl-tmp/datasets/DomainNet \
  /root/autodl-tmp/models/open_clip_vitb16.bin \
  /root/autodl-tmp/models/mixer_b16_224_complete.pth \
  /root/miniconda3/envs/fs/bin/python
```

## 3. 运行环境

优先使用服务器已有环境：

```bash
cd /root/autodl-tmp/FederatedScope
export PYTHON_BIN=/root/miniconda3/envs/fs/bin/python
$PYTHON_BIN -V
$PYTHON_BIN -c "import torch, torchvision, open_clip, timm; print(torch.__version__); print(torchvision.__version__)"
```

仓库基础要求见 `setup.py`：

- Python：`>=3.9`
- FederatedScope 基础依赖：`numpy<1.23.0`、`scikit-learn==1.0.2`、`scipy==1.7.3`、`grpcio`、`pyyaml`、`matplotlib` 等。

GGEUR 分支额外依赖：

- `torch` / `torchvision`
- `open_clip_torch`，用于 `vit` 分支。
- `timm`，用于 `mixer` / `mlp` 分支。
- `pandas`、`matplotlib`，用于结果汇总和画图脚本。

安装或补依赖的常用命令：

```bash
cd /root/autodl-tmp/FederatedScope
/root/miniconda3/envs/fs/bin/python -m pip install -e .
/root/miniconda3/envs/fs/bin/python -m pip install open_clip_torch timm
```

如果需要重建 GGEUR 原始环境，可参考 `ProjectsFederatedScopeGGEUR_source/environment.yml`。该文件记录过的关键包版本包括 `torch==2.1.0+cu121`、`torchvision==0.16.0+cu121`、`open-clip-torch==2.24.0`、`timm==0.9.16`；但当前 FederatedScope 仓库的 `setup.py` 与旧环境存在版本差异，协作者复现实验时建议优先复用服务器已有 `fs` 环境。

## 4. 配置文件位置

完整矩阵也记录在 `scripts/thirdparty_accuracy_cases.yaml` 和 `scripts/run_ggeur_accuracy_matrix_queue.py`。

### OfficeHome-LDS

| 模型 | GGEUR | FedAvg | FedProx | FedProto | FedOpt | MOON |
|---|---|---|---|---|---|---|
| `cnn` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_ggeur_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_moon.yaml` |
| `mixer` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_ggeur_fedavg_local_weights.yaml` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedavg_baseline_local_weights.yaml` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_moon.yaml` |
| `vit` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_moon.yaml` |

补充：`scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg_small.yaml` 是小规模/调试变体，不属于主矩阵。

### DomainNet 4 Domains

| 模型 | GGEUR | FedAvg | FedProx | FedProto | FedOpt | MOON |
|---|---|---|---|---|---|---|
| `cnn` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_ggeur_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_moon.yaml` |
| `mlp` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_ggeur_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_moon.yaml` |
| `vit` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_ggeur_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedavg.yaml` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedprox.yaml` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedproto.yaml` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedopt.yaml` | `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_moon.yaml` |

## 5. 使用方法

### 5.1 单个配置运行

在服务器项目根目录执行：

```bash
cd /root/autodl-tmp/FederatedScope

/root/miniconda3/envs/fs/bin/python federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml
```

常用命令行覆盖参数：

```bash
# 指定 GPU
/root/miniconda3/envs/fs/bin/python federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml \
  device 0

# 指定输出目录和实验名
/root/miniconda3/envs/fs/bin/python federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml \
  outdir exp/manual_runs \
  expname officehome_vit_ggeur_debug

# 临时改数据或模型路径
/root/miniconda3/envs/fs/bin/python federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml \
  data.root /root/autodl-tmp/datasets/OfficeHomeDataset_10072016 \
  ggeur.clip_model_path /root/autodl-tmp/models/open_clip_vitb16.bin
```

### 5.2 单个 case 后台运行

`scripts/start_thirdparty_accuracy_case.sh` 从 `scripts/thirdparty_accuracy_cases.yaml` 选择 case，适合在服务器后台启动一个实验：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=thirdparty_accuracy_manual \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_thirdparty_accuracy_case.sh officehome__vit__ggeur
```

case 命名规则：`<dataset>__<model>__<method>`，例如：

- `officehome__cnn__fedprox`
- `officehome__mixer__moon`
- `domainnet__vit__fedopt`
- `domainnet__mlp__ggeur`

日志和结果默认写到：

```text
exp/ggeur_accuracy_reruns/<RUN_ID>/
```

### 5.3 运行完整 OfficeHome + DomainNet 矩阵

推荐使用：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=thirdparty_accuracy_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0 \
THIRDPARTY_DATASETS="officehome domainnet" \
bash scripts/run_thirdparty_accuracy_matrix.sh
```

该脚本会：

1. 调用 `scripts/run_ggeur_accuracy_matrix_queue.py` 顺序执行配置。
2. 将每个 case 写入独立目录。
3. 汇总 `manifest.json / state.json / summary.json / summary.csv / experiment_log.md`。
4. 调用 `scripts/plot_ggeur_accuracy_curves.py` 生成每组 dataset/model 的对比图。
5. 打包轻量结果到 `thirdparty_accuracy_results_<RUN_ID>.tar.gz`。

只跑某个数据集：

```bash
THIRDPARTY_DATASETS="domainnet" bash scripts/run_thirdparty_accuracy_matrix.sh
```

只跑某些方法/模型可直接调用队列脚本：

```bash
/root/miniconda3/envs/fs/bin/python scripts/run_ggeur_accuracy_matrix_queue.py \
  --repo-dir /root/autodl-tmp/FederatedScope \
  --python-bin /root/miniconda3/envs/fs/bin/python \
  --output-root exp/ggeur_accuracy_reruns \
  --run-id manual_domainnet_vit \
  --cuda-visible-devices 0 \
  --dataset domainnet \
  --model vit \
  --method ggeur \
  --method fedavg \
  --opt device=0 \
  --opt ggeur.use_feature_cache=True \
  --opt ggeur.reuse_augmented_feature_cache=True \
  --opt ggeur.save_augmented_feature_cache=True
```

### 5.4 DomainNet 专用批量脚本

DomainNet 还有一个较简单的批量脚本：

```bash
cd /root/autodl-tmp/FederatedScope

# all / vit / cnn / mlp
bash scripts/run_domainnet_all.sh all
```

可用环境变量：

```bash
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
LOG_ROOT=exp/domainnet_4domains/_batch_runner/manual \
ON_ERROR=continue \
DRY_RUN=0 \
bash scripts/run_domainnet_all.sh vit
```

## 6. 数据集处理口径

### OfficeHome-LDS

- 数据根目录：`/root/autodl-tmp/datasets/OfficeHomeDataset_10072016`
- 配置类型：`data.type: office-home`
- 默认 domains：OfficeHome 原始 4 个域，通常为 `Art / Clipart / Product / Real_World`
- 类别数：`model.num_classes: 65`
- 数据划分：`data.splits: [0.7, 0.0, 0.3]`
- 非 IID 划分：`ggeur.use_lds: True`，`ggeur.lds_alpha: 0.1`

### DomainNet

- 数据根目录：`/root/autodl-tmp/datasets/DomainNet`
- 配置类型：`data.type: domainnet`
- 当前使用 domains：`['clipart', 'painting', 'real', 'sketch']`
- 类别数：`model.num_classes: 345`
- 数据划分：`data.splits: [0.7, 0.0, 0.3]`
- 非 IID 划分：`ggeur.use_lds: True`，`ggeur.lds_alpha: 0.01`
- 类别口径：`domainnet_shared_classes_only` 默认是 `False`，即使用所选 domains 类别并集。
- 当前测试口径见 `docs/DomainNet_完整说明.md`：每个 domain 内部按比例切分 train/test，属于同域测试，不是跨域 leave-one-domain-out 测试。

## 7. 方法差异参数

五个 baseline 都禁用 GGEUR 生成：

```yaml
ggeur:
  num_generated_per_sample: 0
  num_generated_per_prototype: 0
  target_size_per_class: 0
```

`fedavg`：

- 不额外打开 `fedprox / fedopt / use_fedproto / use_moon`。

`fedprox`：

```yaml
fedprox:
  use: True
  mu: 0.1
```

`fedproto`：

```yaml
ggeur:
  use_fedproto: True
  proto_weight: 1.0
  proto_distance: 'cosine'
  proto_temperature: 0.1
```

`fedopt`：

```yaml
fedopt:
  use: True
  optimizer:
    type: 'Adam'
    lr: 0.01
  annealing: False
```

`moon`：

```yaml
ggeur:
  use_moon: True
  moon_mu: 5.0
  moon_temperature: 0.5
```

GGEUR 主配置通常是：

```yaml
ggeur:
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
```

Mixer 分支有些配置使用 `100`，以具体 YAML 为准。

## 8. 示例配置逐项解释

以下以 `scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml` 为例。

### 基础设置

```yaml
use_gpu: True
device: 0
seed: 42
verbose: 1
```

| 参数 | 含义 |
|---|---|
| `use_gpu` | 是否使用 GPU。服务器训练应设为 `True`。 |
| `device` | 使用的 GPU 编号。若设置了 `CUDA_VISIBLE_DEVICES=0`，这里通常仍写 `0`。 |
| `seed` | 随机种子，控制数据划分和训练初始化等随机性。 |
| `verbose` | 日志详细程度。`1` 表示输出常规训练日志。 |

### 联邦训练设置

```yaml
federate:
  method: 'ggeur'
  mode: 'standalone'
  client_num: 60
  total_round_num: 100
  sample_client_num: 0
```

| 参数 | 含义 |
|---|---|
| `federate.method` | 这里固定为 `ggeur`，表示使用本仓库自定义的 GGEUR worker/runner 管线。即使跑 FedAvg/FedProx 等 baseline，也仍通过这条管线执行。 |
| `federate.mode` | `standalone` 表示单机模拟多客户端；不是多进程/多机真实分布式。 |
| `federate.client_num` | 模拟客户端数量。当前 OfficeHome/DomainNet 主矩阵一般是 60。 |
| `federate.total_round_num` | 联邦训练总轮数。 |
| `federate.sample_client_num` | 每轮采样客户端数量。`0` 在当前配置中表示全客户端参与。 |

### 数据设置

```yaml
data:
  type: 'office-home'
  root: '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]
```

| 参数 | 含义 |
|---|---|
| `data.type` | 数据集加载器类型。OfficeHome 使用 `office-home`，DomainNet 使用 `domainnet`。 |
| `data.root` | 数据集根目录。协作者换服务器或本地运行时优先覆盖这个路径。 |
| `data.splits` | 训练/验证/测试划分比例。当前为 70% train、0% val、30% test。 |

### DataLoader 设置

```yaml
dataloader:
  batch_size: 32
  num_workers: 0
```

| 参数 | 含义 |
|---|---|
| `dataloader.batch_size` | 特征抽取和本地训练时使用的 batch size。显存不足时先调小。 |
| `dataloader.num_workers` | PyTorch DataLoader worker 数。服务器上如 IO 足够可调大；Windows/调试场景设 `0` 更稳。 |

### 模型设置

```yaml
model:
  type: 'ggeur_mlp'
  num_classes: 65
```

| 参数 | 含义 |
|---|---|
| `model.type` | 下游分类器类型。GGEUR 管线中固定使用 `ggeur_mlp`，即在抽取后的 embedding 上训练 MLP/线性分类头。 |
| `model.num_classes` | 分类类别数。OfficeHome 是 65，DomainNet 当前配置是 345。 |

### 本地训练设置

```yaml
train:
  local_update_steps: 1
  optimizer:
    type: 'Adam'
    lr: 0.0001
```

| 参数 | 含义 |
|---|---|
| `train.local_update_steps` | 每轮每个客户端本地更新步数。GGEUR 配置通常比 baseline 小，用于控制增强数据后的训练强度。 |
| `train.optimizer.type` | 本地分类头优化器类型。当前多为 `Adam`。 |
| `train.optimizer.lr` | 本地优化器学习率。不同方法/模型分支可能不同，以 YAML 为准。 |

### GGEUR 开关

```yaml
ggeur:
  use: True
```

| 参数 | 含义 |
|---|---|
| `ggeur.use` | 启用 GGEUR 自定义逻辑。当前所有矩阵配置都应为 `True`；baseline 是禁用“生成”，不是禁用整条 GGEUR 管线。 |

### ViT / CLIP 特征提取设置

```yaml
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/autodl-tmp/models/open_clip_vitb16.bin'
  embedding_dim: 512
```

| 参数 | 含义 |
|---|---|
| `feature_extractor` | 特征提取器类型。`clip` 表示 CLIP/ViT；CNN 分支为 `cnn`；Mixer 分支为 `timm`。 |
| `clip_model` | open_clip 模型名，当前为 `ViT-B-16`。 |
| `clip_pretrained` | open_clip 预训练权重来源。配置保留 `openai`，但若 `clip_model_path` 存在，会优先使用本地权重。 |
| `clip_model_path` | 本地 CLIP 权重文件，避免运行时联网下载。 |
| `embedding_dim` | 特征维度。CLIP ViT-B/16 为 512；CNN ConvNeXt-Base 为 1024；MLP-Mixer 为 768。 |

### 特征增强设置

```yaml
  num_generated_per_sample: 50
  num_generated_per_prototype: 50
  target_size_per_class: 50
```

| 参数 | 含义 |
|---|---|
| `num_generated_per_sample` | 围绕本地原始样本 embedding 生成的增强特征数量。大于 0 是 GGEUR 与 FedAvg baseline 的关键差异之一。 |
| `num_generated_per_prototype` | 围绕跨客户端/跨域 prototype 生成的增强特征数量。多域场景用于模拟其他客户端/域的类别分布。 |
| `target_size_per_class` | 每个类别增强后的目标样本规模上限/目标值。baseline 配置设为 0。 |

### MLP 分类头设置

```yaml
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
```

| 参数 | 含义 |
|---|---|
| `mlp_hidden_dim` | MLP 隐藏层维度。`0` 表示不使用隐藏层，等价于线性分类头。CNN 分支有配置为 `256` 的情况。 |
| `mlp_dropout` | MLP dropout 概率。`0.0` 表示不使用 dropout。 |

### 多域与统计轮设置

```yaml
  use_cross_client_prototypes: True
  statistics_round: 0
```

| 参数 | 含义 |
|---|---|
| `use_cross_client_prototypes` | 是否使用其他客户端上传的类别 prototype 做增强。多域/非 IID 场景建议保持 `True`。 |
| `statistics_round` | 收集均值、协方差、prototype 等统计量的轮次。当前通常在第 0 轮先收集统计量再训练。 |

### FedProto 开关

```yaml
  use_fedproto: False
```

| 参数 | 含义 |
|---|---|
| `use_fedproto` | 是否在 GGEUR trainer 内加入 FedProto 式 prototype 正则。FedProto baseline 中设为 `True`；GGEUR+FedAvg 与 FedAvg baseline 中设为 `False`。 |

### LDS 非 IID 划分

```yaml
  use_lds: True
  lds_alpha: 0.1
  lds_seed: 42
```

| 参数 | 含义 |
|---|---|
| `use_lds` | 是否使用 Dirichlet label distribution skew 划分客户端数据。 |
| `lds_alpha` | Dirichlet 分布参数。值越小，客户端标签分布越不均衡。OfficeHome 常用 `0.1`，DomainNet 当前配置常用 `0.01`。 |
| `lds_seed` | LDS 划分随机种子。 |

### 其他模式开关

```yaml
  use_cnn_distillation: False
  use_feature_alignment: False
  use_separated_training: False
```

| 参数 | 含义 |
|---|---|
| `use_cnn_distillation` | 是否启用 CNN 蒸馏训练模式。当前矩阵关闭。 |
| `use_feature_alignment` | 是否启用 CNN 从头训练并对齐 CLIP 特征的模式。当前矩阵关闭。 |
| `use_separated_training` | 是否启用“先训练分类器、再训练 backbone”的分阶段训练模式。当前矩阵关闭。 |

### 输出设置

```yaml
outdir: 'exp/ggeur_baselines'
expname: 'officehome_lds/vit/ggeur/fedavg'
```

| 参数 | 含义 |
|---|---|
| `outdir` | 实验输出根目录。批量脚本会通过命令行覆盖为每个 case 的独立目录。 |
| `expname` | 实验名。可包含路径分隔符，用于组织输出。 |

## 9. 换模型分支时重点改哪些参数

ViT/CLIP：

```yaml
ggeur:
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_model_path: '/root/autodl-tmp/models/open_clip_vitb16.bin'
  embedding_dim: 512
```

CNN/ConvNeXt：

```yaml
ggeur:
  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'
  cnn_pretrained: True
  freeze_backbone: True
  embedding_dim: 1024
```

MLP-Mixer：

```yaml
ggeur:
  feature_extractor: 'timm'
  timm_model: 'mixer_b16_224'
  timm_pretrained: False
  timm_checkpoint_path: '/root/autodl-tmp/models/mixer_b16_224_complete.pth'
  timm_in_chans: 3
  timm_global_pool: 'avg'
  freeze_backbone: True
  embedding_dim: 768
```

## 10. 常见检查项

运行前检查：

```bash
cd /root/autodl-tmp/FederatedScope

# 配置文件是否存在
test -f scripts/thirdparty_accuracy_cases.yaml
test -f scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml

# 数据和权重是否存在
test -d /root/autodl-tmp/datasets/OfficeHomeDataset_10072016
test -d /root/autodl-tmp/datasets/DomainNet
test -f /root/autodl-tmp/models/open_clip_vitb16.bin
test -f /root/autodl-tmp/models/mixer_b16_224_complete.pth

# 关键依赖是否可导入
/root/miniconda3/envs/fs/bin/python -c "import torch, torchvision, open_clip, timm, yaml; print('ok')"
```

如果显存不足：

- 先降低 `dataloader.batch_size`。
- 再降低 `ggeur.extract_batch_size`，该项默认在 `federatedscope/core/configs/cfg_ggeur.py` 中为 64，可通过命令行覆盖。
- 对大矩阵实验使用 `CUDA_VISIBLE_DEVICES` 限制 GPU，并保证 YAML 中 `device` 与可见 GPU 编号一致。

如果需要强制重建缓存：

```bash
/root/miniconda3/envs/fs/bin/python federatedscope/main.py \
  --cfg <config.yaml> \
  ggeur.reuse_augmented_feature_cache False \
  ggeur.save_augmented_feature_cache True
```

如果只想快速检查命令，不真正运行：

```bash
DRY_RUN=1 bash scripts/run_domainnet_all.sh vit
```
