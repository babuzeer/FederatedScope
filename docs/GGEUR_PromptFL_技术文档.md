# GGEUR + PromptFL 技术实现文档

## 概述

本方案将 **GGEUR**（高斯几何引导特征扩展）与 **PromptFL**（联邦软提示学习）结合，在联邦学习框架下实现两项技术的协同增益：

- **GGEUR** 解决 few-shot 场景下数据不足的问题，通过跨客户端协方差聚合生成增强特征
- **PromptFL** 用极少的可训练参数（软提示向量）替代传统分类头，充分利用 CLIP 预训练知识

两者结合的核心思路：**用 GGEUR 增强后的特征来训练 PromptFL 的软提示**，弥补原始 PromptFL 在 few-shot 场景下数据不足的缺陷。

---

## 背景

### GGEUR

GGEUR 的核心算法：

1. 每个客户端提取本地图像的 CLIP 特征，计算每个类的均值和协方差矩阵
2. 服务器用**平行轴定理**聚合全局协方差矩阵，并计算全局类原型
3. 客户端用全局协方差对本地特征做高斯采样扩增，同时利用其他客户端的类原型生成跨域样本
4. 在增强后的特征上训练 MLP 分类器，通过 FedAvg 聚合

### PromptFL

PromptFL（Guo et al., 2022）的核心思路：

- 将联邦学习的训练目标从**模型参数**改为**提示向量**
- 每个客户端持有冻结的 CLIP 模型，只训练少量可学习的文本提示向量（soft prompts）
- 服务器只聚合提示向量，通信开销极小

---

## 方法设计

### 整体流程

```
Round 0（统计收集轮）
  客户端：提取 CLIP 图像特征 → 计算本地均值/协方差 → 上传至服务器
  服务器：聚合协方差（平行轴定理）→ 计算全局原型 → 广播给所有客户端

客户端收到全局协方差后：
  → 高斯特征增强（GGEUR）
  → 构建增强特征数据集
  → 初始化 MLP 分类器
  → 初始化 CustomCLIP（PromptLearner + TextEncoder）

Round 1 ~ N（训练轮）
  服务器广播：全局 MLP 参数 + 全局 ctx 向量
  客户端：
    ① 在增强特征上训练 MLP（原有逻辑，保持不变）
    ② 在增强特征上训练软提示 ctx（新增）
  客户端上传：MLP 参数 + ctx 向量
  服务器：
    ① FedAvg 聚合 MLP
    ② FedAvg 聚合 ctx
    ③ 用 MLP 评估测试集
    ④ 用 prompt 评估测试集
```

### 软提示的分类机制

对于每个类 $i$，使用模板（默认 `"a photo of a {}"`）构造文本序列：

$$[\text{BOS}]\ [\mathbf{v}_1]\ \cdots\ [\mathbf{v}_p]\ [\text{class\_name}_i\ \text{EOS}\ \text{PAD}\ldots]$$

其中 $\mathbf{v}_1, \ldots, \mathbf{v}_p \in \mathbb{R}^d$ 是可学习的上下文向量（ctx），$p$ 为提示长度（默认 16），$d=512$。

通过 CLIP 文本编码器得到每个类的文本特征 $\mathbf{w}_i$，分类 logits 为：

$$\text{logits}_{i} = \tau \cdot \cos\left[g(\mathbf{x}),\ \mathbf{w}_i\right]$$

其中 $\tau = \exp(\text{logit\_scale})$ 为 CLIP 自带的可学习温度参数，$g(\mathbf{x})$ 为预提取的 CLIP 图像特征。

训练时只对 ctx 向量 $\mathbf{P}$ 反向传播，CLIP 图像编码器和文本编码器均冻结。

### 与原版 PromptFL 的区别

| 维度 | 原版 PromptFL | GGEUR + PromptFL |
|------|--------------|-----------------|
| 训练数据 | 原始图像（实时过图像编码器） | GGEUR 增强后的特征向量 |
| 数据量 | 客户端本地原始样本 | 增强后可达 `target_size_per_class × num_classes` |
| 跨域信息 | 无 | 利用其他客户端原型生成跨域样本 |
| 温度系数 | 固定值 | CLIP 自带可学习 `logit_scale` |
| 适用场景 | 一般 few-shot | 极端 few-shot + 非 IID |

---

## 代码实现

### 新增文件

#### `federatedscope/contrib/model/ggeur_prompt.py`

基于 HuggingFace `transformers` 库实现，包含三个类：

**`PromptLearner`**
- 持有可学习的 ctx 向量，形状 `[n_ctx, d]`，正态初始化（std=0.02）
- 初始化时用模板文本预计算固定 token embeddings：
  - `token_prefix`：BOS token，形状 `[K, 1, d]`
  - `token_suffix`：类名 tokens + EOS + padding，形状 `[K, L-1, d]`
  - `attention_mask`：对应的注意力掩码
- `forward()` 拼接三部分，同时构造完整 attention mask，返回 `(prompt_embeds, attn_mask)`

**`TextEncoder`**
- 封装 CLIP 文本 transformer 的编码逻辑
- 正确处理 causal attention mask 和 4D attention mask
- EOS 位置通过 `attention_mask.sum(dim=-1) - 1` 精确定位
- 所有参数冻结，不参与梯度计算

**`CustomCLIP`**
- 整合 PromptLearner + TextEncoder + 冻结的 CLIP 模型
- `forward(pixel_values)` 支持端到端图像分类（原始图像输入）
- 使用 CLIP 自带的可学习 `logit_scale` 作为温度参数
- 实现 `clone_prompt_only()` 和 `__deepcopy__`：深拷贝时只复制 prompt 参数，不复制庞大的 CLIP 权重，节省内存

### 修改文件

#### `federatedscope/core/configs/cfg_ggeur.py`

新增 8 个配置项：

```python
cfg.ggeur.use_promptfl = False                      # 是否启用 PromptFL（开关）
cfg.ggeur.prompt_length = 16                        # 软提示 token 数量
cfg.ggeur.prompt_lr = 0.002                         # 提示优化器学习率（Adam）
cfg.ggeur.prompt_local_epochs = 10                  # 每轮本地训练 epoch 数
cfg.ggeur.prompt_temperature = 0.07                 # 兼容旧版，新版用 logit_scale
cfg.ggeur.prompt_class_names = []                   # 可选：手动指定类名
cfg.ggeur.prompt_template = 'a photo of a {}'       # 文本提示模板
cfg.ggeur.hf_clip_model_id = 'openai/clip-vit-base-patch16'  # HF 模型路径或 Hub ID
```

#### `federatedscope/contrib/worker/ggeur_client.py`

以 `if use_promptfl` 分支插入，不改动现有逻辑：

1. `__init__` 追加属性：`use_promptfl`、`custom_clip`、`hf_clip_model`、`prompt_learner`、`text_encoder`
2. `callback_for_global_covariances` 中，增强完成后调用 `_build_prompt_model()`
3. `callback_funcs_for_model_para` 中，加载全局 ctx → 训练 prompt → 附加到上传参数
4. 新增方法：
   - `_get_class_names()`：从数据集类（OfficeHome/PACS 等）或配置获取类名
   - `_build_prompt_model()`：加载 HuggingFace CLIPModel，构建 `CustomCLIP`
   - `_train_prompt_on_augmented_data()`：用预提取的增强特征训练 ctx，使用 `logit_scale` 缩放

#### `federatedscope/contrib/worker/ggeur_server.py`

同样以 `if use_promptfl` 分支插入：

1. `__init__` 追加属性：`use_promptfl`、`global_prompt_ctx`、`prompt_test_accuracies_history`、`_eval_hf_clip` 等
2. `callback_for_local_statistics` 中，统计聚合后调用 `_init_global_prompt()`
3. `_start_training_round` 中，广播时附加 `{'prompt': {'ctx': global_prompt_ctx}}`
4. `_perform_fedavg` 中，聚合 MLP 后调用 `_aggregate_prompt()` 和 `_evaluate_prompt_on_test_sets()`
5. `_finish` 中打印 PromptFL 最终结果
6. 新增方法：
   - `_init_global_prompt()`：初始化全局 ctx 为零向量
   - `_aggregate_prompt()`：FedAvg 聚合各客户端 ctx
   - `_evaluate_prompt_on_test_sets()`：构建 CustomCLIP，用全局 ctx 评估测试集

---

## 依赖库

| 库 | 用途 |
|----|------|
| `open_clip` | CLIP 图像特征提取（GGEUR 原有） |
| `transformers` | HuggingFace CLIPModel，用于 PromptFL 文本编码 |

两套 CLIP 并行存在，互不干扰：
- `open_clip` 负责图像特征提取（`feature_extractor='clip'` 时）
- HuggingFace `CLIPModel` 负责 PromptFL 的文本编码器

---

## 参数量对比

| 方法 | 可训练参数 | 说明 |
|------|-----------|------|
| MLP（线性） | 512 × 65 = **33,280** | 输入维度 × 类别数 |
| PromptFL ctx | 16 × 512 = **8,192** | 提示长度 × 词嵌入维度 |
| 通信节省 | **75.4%** | 每轮聚合传输量减少 |

---

## 配置说明

配置文件位于 `scripts/example_configs/ggeur_promptfl/ggeur_promptfl_clip.yaml`。

关键配置项：

```yaml
ggeur:
  # 图像特征提取（open_clip）
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/model/open_clip_vitb16.bin'

  # PromptFL 开关及超参数
  use_promptfl: True
  prompt_length: 16
  prompt_lr: 0.002
  prompt_local_epochs: 10
  prompt_template: 'a photo of a {}'

  # HuggingFace CLIP（文本编码器，需为目录格式）
  hf_clip_model_id: 'openai/clip-vit-base-patch16'
```

### 约束条件

- `use_promptfl: True` 时，`feature_extractor` 必须为 `'clip'`
- `hf_clip_model_id` 需指向 HuggingFace 格式目录（含 `config.json`），不支持单 `.bin` 文件
- CNN/timm 特征提取器不支持文本软提示（特征空间与 CLIP 文本编码器不对齐）

---

## 运行

```bash
python federatedscope/main.py --cfg scripts/example_configs/ggeur_promptfl/ggeur_promptfl_clip.yaml
```

### 日志输出示例

```
Client 1: Loading HuggingFace CLIP from openai/clip-vit-base-patch16
Client 1: Built CustomCLIP with 16 ctx tokens, 65 classes
Client 1: Prompt training - loss=2.1234, acc=0.4521, samples=3250
Server: Round 5 MLP Test Accuracy    - Art: 0.6123, Clipart: 0.5234, average: 0.5891
Server: Round 5 Prompt Test Accuracy - Art: 0.6341, Clipart: 0.5512, average: 0.6102
...
PromptFL Final Test Results:
  Art: final=0.7123, best=0.7234
  Clipart: final=0.6512, best=0.6634
  Product: final=0.7891, best=0.7923
  Real_World: final=0.7634, best=0.7712
Prompt Best Average Accuracy: 0.7376
```

---

## 参考文献

- **PromptFL**: Guo T, Guo S, Wang J, Xu W. *PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models*. 2022.
- **CoOp**: Zhou K, Yang J, Loy C C, et al. *Learning to Prompt for Vision-Language Models*. IJCV, 2022.
- **CLIP**: Radford A, et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.
- **GGEUR**: 本项目实现，基于高斯几何引导特征扩展的联邦学习方法。
