# GGEUR方法完整技术文档 - 提纲

---

## 第一章：GGEUR方法概述

### 1.1 背景与动机
- 联邦学习中的多域异构问题
- 标签分布倾斜（Label Distribution Skew）挑战
- 现有方法的局限性

### 1.2 GGEUR核心思想
- Gaussian Geometry-guided Feature Expansion with Unified Representation
- 基于CLIP特征空间的统一表示
- 高斯几何引导的特征扩张策略

### 1.3 算法流程概览
- 五阶段流程：特征提取 → 统计计算 → 全局聚合 → 特征增强 → 模型训练

---

## 第二章：核心算法原理

### 2.1 特征提取阶段
- CLIP特征提取器原理
- CNN特征提取器（ConvNeXt等）支持
- 特征缓存机制

### 2.2 局部统计量计算
- 类别均值（Mean）计算
- 协方差矩阵（Covariance）计算
- 类原型（Prototype）构建

### 2.3 全局协方差聚合
- 平行轴定理（Parallel Axis Theorem）
- 加权聚合公式推导
- 全局原型计算

### 2.4 高斯特征增强
- 多元高斯分布采样
- 跨客户端原型增强
- 目标样本数量平衡策略

### 2.5 分类器训练
- MLP分类器结构
- FedAvg聚合策略

---

## 第三章：框架实现架构

### 3.1 文件结构总览
```
federatedscope/
├── contrib/
│   ├── worker/
│   │   ├── ggeur_client.py      # 客户端实现
│   │   └── ggeur_server.py      # 服务器实现
│   ├── trainer/
│   │   └── ggeur_trainer.py     # 训练器实现
│   ├── model/
│   │   ├── ggeur_mlp.py         # MLP分类器
│   │   └── ggeur_cnn.py         # CNN骨干网络
│   └── data/
│       └── ggeur_data.py        # 数据加载与划分
├── core/configs/
│   └── cfg_ggeur.py             # 配置项定义
└── scripts/example_configs/
    └── ggeur_*.yaml             # 配置文件示例
```

### 3.2 客户端实现详解（ggeur_client.py）
- 类结构与继承关系
- `_extract_features()`：特征提取流程
- `_compute_local_statistics()`：统计量计算
- `_perform_augmentation()`：特征增强实现
- 回调函数与状态管理

### 3.3 服务器端实现详解（ggeur_server.py）
- 类结构与继承关系
- `_aggregate_covariances()`：协方差聚合算法
- `_compute_global_prototypes()`：全局原型计算
- 消息调度与状态机

### 3.4 训练器实现（ggeur_trainer.py）
- EmbeddingDataset包装类
- 特征级别训练流程
- 损失函数与优化器配置

---

## 第四章：消息通信协议

### 4.1 通信流程图
- 完整时序图
- 各阶段消息类型

### 4.2 Round 0：统计量收集阶段
- 客户端上传消息格式
  - `msg_type='local_statistics'`
  - 包含字段：`means`, `covs`, `counts`, `prototypes`
- 服务器下发消息格式
  - `msg_type='global_covariances'`
  - 包含字段：`cov_matrices`, `other_prototypes`, `global_prototypes`

### 4.3 Round 1+：模型训练阶段
- `msg_type='augmentation_ready'`
- `msg_type='model_para'`：模型参数交换
- MLP与CNN双模型参数传输

### 4.4 异常处理与重试机制

---

## 第五章：配置项详解

### 5.1 基础配置
- `ggeur.use`：启用GGEUR
- `ggeur.feature_extractor`：特征提取器选择（clip/cnn）
- `ggeur.embedding_dim`：特征维度

### 5.2 CLIP配置
- `ggeur.clip_model`：CLIP模型类型
- `ggeur.clip_pretrained`：预训练权重来源
- `ggeur.clip_model_path`：本地权重路径

### 5.3 CNN骨干网络配置
- `ggeur.cnn_backbone`：CNN架构选择
- `ggeur.freeze_backbone`：是否冻结骨干网络

### 5.4 特征增强配置
- `ggeur.num_generated_per_sample`：每样本生成数量
- `ggeur.num_generated_per_prototype`：每原型生成数量
- `ggeur.target_size_per_class`：目标类别样本数

### 5.5 MLP分类器配置
- `ggeur.mlp_hidden_dim`：隐藏层维度
- `ggeur.mlp_dropout`：Dropout率

### 5.6 LDS数据划分配置
- `ggeur.use_lds`：启用标签分布倾斜
- `ggeur.lds_alpha`：Dirichlet分布参数
- `ggeur.lds_seed`：随机种子

### 5.7 高级训练模式配置
- 知识蒸馏模式（`use_cnn_distillation`）
- 特征对齐模式（`use_feature_alignment`）
- 分阶段训练模式（`use_separated_training`）
- 端到端微调模式（`use_end_to_end_finetune`）

---

## 第六章：数据集与划分方式

### 6.1 支持的数据集
- Office-Home（4个域：Art, Clipart, Product, Real_World）
- PACS（4个域：Photo, Art_Painting, Cartoon, Sketch）
- Digits（多域数字识别）

### 6.2 域划分策略
- 每域对应一个客户端（Domain-based）
- 多客户端共享域（Cross-domain）

### 6.3 LDS标签分布倾斜实现
- Dirichlet分布原理
- `_split_dataset_with_lds()`函数详解
- alpha参数对非IID程度的影响
- 数据划分可视化示例

### 6.4 数据增强策略
- 图像级增强（用于CNN训练）
- 特征级增强（GGEUR核心）

---

## 第七章：快速入门指南

### 7.1 环境准备
- 依赖安装
- CLIP模型下载
- 数据集准备

### 7.2 运行第一个实验
- 基础配置文件解读
- 命令行启动方式
- 日志与输出解读

### 7.3 配置文件模板
- Office-Home实验配置
- PACS实验配置
- LDS非IID配置

---

## 第八章：分布式训练指南

### 8.1 分布式架构
- gRPC通信模式
- 多机多卡部署

### 8.2 分布式配置
- 服务器端配置
- 客户端配置
- 网络参数调优

### 8.3 启动流程
- 服务器启动脚本
- 客户端启动脚本
- 同步与等待机制

### 8.4 性能优化
- 特征缓存策略
- 通信压缩
- 批量大小调优

---

## 第九章：扩展与自定义

### 9.1 添加新数据集
- 数据格式要求
- 注册新数据集
- 域划分实现

### 9.2 自定义特征提取器
- 继承与实现接口
- 特征维度适配

### 9.3 修改聚合策略
- 协方差聚合算法扩展
- 自定义原型计算

### 9.4 添加新训练模式
- 训练器扩展
- 配置项注册

---

## 第十章：实验配置示例

### 10.1 基础实验配置
- `ggeur_officehome.yaml`
- `ggeur_pacs.yaml`

### 10.2 LDS非IID实验
- `ggeur_officehome_lds.yaml`
- alpha参数调优指南

### 10.3 CNN训练模式
- `ggeur_cnn_pacs.yaml`
- `ggeur_cnn_feature_alignment.yaml`
- `ggeur_cnn_convnext.yaml`

### 10.4 分阶段训练
- `ggeur_separated_training.yaml`

---

## 附录

### A. 常见问题FAQ
### B. 配置项速查表
### C. 性能基准测试结果
### D. 参考文献

---

**预计文档总页数：40-50页**

**建议补充内容：**
1. 实验结果对比表格（与FedAvg、FedProx等基线对比）
2. 可视化示例（t-SNE特征可视化、训练曲线等）
3. 调参建议（不同场景下的推荐配置）
