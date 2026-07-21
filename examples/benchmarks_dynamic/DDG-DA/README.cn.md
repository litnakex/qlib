# 简介
这是基于 `Qlib` 提供的 `Meta Controller` 组件实现的 `DDG-DA`。

更多细节请参阅论文：*DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation* [[arXiv](https://arxiv.org/abs/2201.04038)]


# 背景
在许多真实场景中，我们常常需要处理随时间顺序采集的流式数据。由于环境的非平稳特性，流式数据的分布可能以不可预测的方式发生变化，这被称为概念漂移（concept drift）。为了应对概念漂移，以往的方法首先检测概念漂移发生的时间/位置，然后调整模型以拟合最新数据的分布。然而，仍有许多情形下，环境演化的某些底层因素是可预测的，这使得对流式数据未来的概念漂移趋势进行建模成为可能，而这类情形在以往的工作中尚未得到充分探索。

因此，我们提出了一种新方法 `DDG-DA`，它能够有效地预测数据分布的演化并提升模型的性能。具体而言，我们首先训练一个预测器来估计未来的数据分布，然后利用它来生成训练样本，最后在生成的数据上训练模型。

# 数据集
论文中使用的数据是私有的。因此我们在 Qlib 的公开数据集上开展实验。
尽管数据集不同，但结论保持一致。通过应用 `DDG-DA`，用户可以在测试阶段看到代理模型的 IC 以及预测模型的性能均呈现上升趋势。

# 运行代码
用户可以通过运行以下命令来尝试 `DDG-DA`：
```bash
    python workflow.py run
```

默认的预测模型是 `Linear`。用户可以在 `DDG-DA` 初始化时通过修改 `forecast_model` 参数来选择其他预测模型。例如，用户可以通过运行以下命令来尝试 `LightGBM` 预测模型：
```bash
    python workflow.py --conf_path=../workflow_config_lightgbm_Alpha158.yaml run
```

# 结果
相关方法在 Qlib 公开数据集上的结果可以在[这里](../)找到

# 环境要求
以下是运行 DDG-DA 的 ``workflow.py`` 所需的最低硬件要求。
* 内存：45G
* 磁盘：4G

使用 CPU 与内存的 Pytorch 即可满足本示例的需求。
