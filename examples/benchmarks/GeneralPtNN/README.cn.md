

# 简介

什么是 GeneralPtNN
- 修复了此前无法同时支持时间序列数据与表格数据的设计
- 现在你只需替换 Pytorch 的模型结构即可运行一个 NN 模型。

我们提供了一个示例来展示当前设计的有效性。
- `workflow_config_gru.yaml` 与此前的结果 [GRU(Kyunghyun Cho, et al.)](../README.md#Alpha158-dataset) 保持一致
  - `workflow_config_gru2mlp.yaml` 用于展示我们能够以最小的改动将配置从时间序列数据转换为表格数据
    - 你只需更改 net 与 dataset 类即可完成转换。
- `workflow_config_mlp.yaml` 实现了与 [MLP](../README.md#Alpha158-dataset) 类似的功能

# TODO

- 我们将把现有模型对齐到当前设计。

- `workflow_config_mlp.yaml` 的结果与 [MLP](../README.md#Alpha158-dataset) 的结果不同,因为 GeneralPtNN 采用了与此前实现不同的停止方法。具体来说,GeneralPtNN 依据 epoches 控制训练,而此前的方法则依据 max_steps 进行控制。
