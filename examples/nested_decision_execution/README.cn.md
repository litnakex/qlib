# 嵌套决策执行

本工作流是回测中嵌套决策执行的一个示例。Qlib 支持在回测中进行嵌套决策执行，这意味着用户可以使用不同的策略在不同频率下做出交易决策。

## 周度组合生成与日度订单执行

本工作流提供了一个示例，它在周度频率上使用 DropoutTopkStrategy（一个基于日度频率 Lightgbm 模型的策略）来生成组合，并使用 SBBStrategyEMA（一个使用 EMA 进行决策的基于规则的策略）在日度频率上执行订单。

### 用法

运行以下命令开始回测：
```bash
    python workflow.py backtest
```

运行以下命令开始收集数据：
```bash
    python workflow.py collect_data
```

## 日度组合生成与分钟级订单执行

本工作流还提供了一个高频示例，它在日度频率上使用 DropoutTopkStrategy 来生成组合，并使用 SBBStrategyEMA 在分钟级频率上执行订单。

### 用法

运行以下命令开始回测：
```bash
    python workflow.py backtest_highfreq
```
