# DomainNet 在本项目中的处理说明

本文档只说明 **本项目里** `DomainNet` 是怎么读取、处理、划分和测试的。

## 1. 本项目用到的 DomainNet

本项目的 `DomainNet` 数据按下面的目录结构读取：

```text
root/
  clipart/
    class_name/
      xxx.png
  painting/
    class_name/
      yyy.jpg
  real/
  sketch/
```

代码位置：

- `federatedscope/cv/dataset/domainnet.py:1`

当前示例配置主要使用 4 个域：

- `clipart`
- `painting`
- `real`
- `sketch`

例如：

- `scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedavg.yaml:34`

## 2. 类别怎么处理

本项目不是把类别名写死在代码里，而是从目录里自动发现类别。

代码入口：

- `federatedscope/cv/dataset/domainnet.py:28`

当前默认配置是：

- `domainnet_shared_classes_only = False`

位置：

- `federatedscope/core/configs/cfg_ggeur.py:92`

这表示：

- 当前用的是 **所选域类别的并集**
- 不是“所有域共同拥有的交集类别”

标签映射方式：

- 先得到一个统一的 `classes` 列表
- 再把 `class_name -> label_id`

代码位置：

- `federatedscope/cv/dataset/domainnet.py:96`

## 3. 图像怎么预处理

在 `GGEUR` 的 `DomainNet` 加载逻辑里，当前使用的是：

- `Resize((224, 224))`
- `ToTensor()`
- `CLIP` 风格归一化

代码位置：

- `federatedscope/contrib/data/ggeur_data.py:491`

当前默认 **没有额外数据增强**，比如：

- 没有随机裁剪
- 没有随机翻转
- 没有颜色扰动

## 4. train / val / test 怎么划分

单个 domain 内部的基础划分逻辑在：

- `federatedscope/cv/dataset/domainnet.py:78`

当前默认比例是：

- `train = 0.7`
- `val = 0.0`
- `test = 0.3`

代码位置：

- `federatedscope/contrib/data/ggeur_data.py:489`

也就是说：

- 每个 domain 内部先随机打乱
- 再按 `7:3` 切成训练和测试

所以当前测试方式是：

- `clipart` 的训练样本和测试样本都来自 `clipart`
- `painting` 的训练样本和测试样本都来自 `painting`

这属于 **同域测试**，不是跨域测试。

## 5. 联邦训练里怎么切 client

`DomainNet` 的联邦数据加载逻辑在：

- `federatedscope/contrib/data/ggeur_data.py:483`

当前流程是：

1. 先确定使用哪些域、哪些类别
2. 对每个 domain 单独加载 `train / val / test`
3. 如果开启 `LDS`，先对该 domain 的训练集做一次 `Dirichlet` 划分
4. 再把这个 domain 的训练子集平均切给该 domain 下的多个 client

关键代码：

- 读取域和类别：`federatedscope/contrib/data/ggeur_data.py:507`
- 生成 `Dirichlet` 矩阵：`federatedscope/contrib/data/ggeur_data.py:530`
- 对 domain 训练集做 `LDS`：`federatedscope/contrib/data/ggeur_data.py:558`
- 把子集均分给多个 client：`federatedscope/contrib/data/ggeur_data.py:564`

这说明当前不是：

- 每个 client 独立做一次 Dirichlet

而是：

- **先按 domain 做一次 LDS**
- **再在该 domain 内部切多个 client**

## 6. 当前测试是怎么做的

服务端测试逻辑在：

- `federatedscope/contrib/worker/ggeur_server.py:425`
- `federatedscope/contrib/worker/ggeur_server.py:546`

当前流程是：

1. 服务端按和训练一致的 domain / classes 重新构造测试集
2. 提取测试图像特征
3. 用全局分类器做预测
4. 分别计算每个 domain 的测试准确率
5. 再计算平均准确率

服务端测试缓存现在也会区分：

- `domainnet_shared_classes_only`
- `domainnet_domains`

避免切换配置后误读旧缓存。

## 7. 当前实现的重点总结

如果只看本项目，当前 `DomainNet` 的处理方式可以直接理解成：

- 使用 `4` 个 domain
- 类别默认取并集
- 图像缩放到 `224x224`
- 使用 `CLIP` 风格归一化
- 不做额外增强
- 每个 domain 内部按 `7:3` 切训练/测试
- 若开启 `LDS`，先做 domain 级 `Dirichlet`，再切多个 client
- 测试是同域测试，不是跨域测试

## 8. 相关代码文件

- 数据集定义：`federatedscope/cv/dataset/domainnet.py`
- 数据加载：`federatedscope/contrib/data/ggeur_data.py`
- 服务端测试：`federatedscope/contrib/worker/ggeur_server.py`
- 默认配置：`federatedscope/core/configs/cfg_ggeur.py`
