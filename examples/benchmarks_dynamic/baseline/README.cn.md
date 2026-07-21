# 简介

这是周期性滚动再训练（Rolling Retrain, RR）预测模型的框架。RR 通过周期性地利用最新数据来适应市场动态。

## 运行代码
用户可以通过运行以下命令来尝试 RR：
```bash
    python rolling_benchmark.py run
```

默认的预测模型是 `Linear`。用户可以通过修改 `model_type` 参数来选择其他预测模型。
例如，用户可以通过运行以下命令来尝试 `LightGBM` 预测模型：
```bash
    python rolling_benchmark.py --conf_path=workflow_config_lightgbm_Alpha158.yaml run

```
