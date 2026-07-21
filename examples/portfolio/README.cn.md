# 组合优化策略

## 简介

在 `qlib/examples/benchmarks` 中，我们提供了多种预测股票收益的 **alpha** 模型。我们还使用了一个简单的基于规则的 `TopkDropoutStrategy` 来评估这些模型的投资表现。然而，这样的策略过于简单，无法控制诸如相关性和波动率之类的组合风险。

为此，应当采用基于优化的策略来权衡收益与风险。在本文档中，我们将展示如何使用 `EnhancedIndexingStrategy` 在最大化组合收益的同时，最小化相对于基准的跟踪误差。


## 准备工作

我们以中国股票市场数据作为示例。

1. 准备 CSI300 权重：

   ```bash
   wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
   unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
   rm -f csi300_weight.zip
   ```
   注意：我们没有找到任何公开免费的资源来获取该基准中的权重。为了运行本示例，我们手动创建了这份权重数据。

2. 准备风险模型数据：

   ```bash
   python prepare_riskdata.py
   ```

这里我们使用了在 `qlib.model.riskmodel` 中实现的**统计风险模型（Statistical Risk Model）**。
不过，强烈建议用户使用其他风险模型以获得更高的质量：
* **基本面风险模型（Fundamental Risk Model）**，例如 MSCI BARRA
* [深度风险模型（Deep Risk Model）](https://arxiv.org/abs/2107.05201)


## 端到端工作流

你可以通过运行 `qrun config_enhanced_indexing.yaml` 来使用 `EnhancedIndexingStrategy` 完成整个工作流。

在这份配置中，相较于 `qlib/examples/benchmarks/workflow_config_lightgbm_Alpha158.yaml`，我们主要改动了策略（strategy）部分。
