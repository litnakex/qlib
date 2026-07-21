# 简介
本文件夹包含 2 个示例
- 一个高频数据集示例
- 一个预测高频数据价格趋势的示例

## 高频数据集

该数据集是用于强化学习（RL）高频交易的示例。

### 获取高频数据

通过运行以下命令获取高频数据：
```bash
    python workflow.py get_data
```

### 转储、重载与重新初始化数据集


高频数据集在 `workflow.py` 中实现为 `qlib.data.dataset.DatasetH`。`DatatsetH` 是 [`qlib.utils.serial.Serializable`](https://qlib.readthedocs.io/en/latest/advanced/serial.html) 的子类，其状态可以以 `pickle` 格式转储到磁盘或从磁盘加载。

### 关于重新初始化

从磁盘重载 `Dataset` 后，`Qlib` 还支持重新初始化数据集。这意味着用户可以重置 `Dataset` 或 `DataHandler` 的某些状态，例如 `instruments`、`start_time`、`end_time` 和 `segments` 等，并根据这些状态生成新的数据。

示例在 `workflow.py` 中给出，用户可以按如下方式运行代码。

### 运行代码

通过运行以下命令运行该示例：
```bash
    python workflow.py dump_and_load_dataset
```

## 基准表现（预测高频数据的价格趋势）

以下是用于预测高频数据价格趋势的模型结果。我们将在未来持续更新基准模型。

| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Long precision| Short Precision | Long-Short Average Return | Long-Short Average Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Alpha158 | 0.0349±0.00 | 0.3805±0.00| 0.0435±0.00 | 0.4724±0.00 | 0.5111±0.00 | 0.5428±0.00 | 0.000074±0.00 | 0.2677±0.00 |
