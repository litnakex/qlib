# Rolling Process Data

本工作流是 `Rolling Process Data` 的一个示例。

## 背景

在滚动训练模型时，数据也需要在不同的滚动窗口中生成。当滚动窗口移动时，训练数据会发生变化，处理器（Processor）的可学习状态（例如标准差、均值等）也会随之改变。

为了避免重新生成数据，本示例使用 `DataHandler-based DataLoader` 来加载与滚动窗口无关的原始特征，然后使用处理器（Processors）生成与滚动窗口相关的已处理特征。


## 运行代码

通过运行以下命令来运行本示例：
```bash
    python workflow.py rolling_process
```
