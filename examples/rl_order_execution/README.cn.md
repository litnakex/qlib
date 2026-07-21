# 订单执行的强化学习示例

本文件夹包含一个用于订单执行场景的强化学习(RL)工作流示例，包括训练工作流和回测工作流。

## 数据处理

### 获取数据

```
python -m qlib.cli.data qlib_data --target_dir ./data/bin --region hs300 --interval 5min
```

### 生成 Pickle 格式数据

要运行本示例中的代码，我们需要 pickle 格式的数据。为此，请运行以下命令(可能需要几分钟才能完成):

[//]: # (TODO: Instead of dumping dataframe with different format &#40;like `_gen_dataset` and `_gen_day_dataset` in `qlib/contrib/data/highfreq_provider.py`&#41;, we encourage to implement different subclass of `Dataset` and `DataHandler`. This will keep the workflow cleaner and interfaces more consistent, and move all the complexity to the subclass.)

```
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py
```

完成后，`data/` 下的目录结构应为:

```
data
├── bin
├── orders
└── pickle
```

## 训练

每个训练任务由一个配置文件指定。任务 `TASKNAME` 的配置文件为 `exp_configs/train_TASKNAME.yml`。本示例提供了两个训练任务:

- **PPO**: IJCAL 2020 论文 "[An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization](https://www.ijcai.org/proceedings/2020/0627.pdf)" 提出的方法。
- **OPDS**: AAAI 2021 论文 "[Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860)" 提出的方法。

这两种方法的主要区别在于它们的奖励函数。详情请参见各自的配置文件。

以 OPDS 为例，要运行训练工作流，请运行:

```
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
```

指标、日志和检查点将存储在 `outputs/opds` 下(由 `exp_configs/train_opds.yml` 配置)。

## 回测

训练工作流完成后，训练好的模型即可用于回测工作流。仍以 OPDS 为例，训练完成后，模型的最新检查点可在 `outputs/opds/checkpoints/latest.pth` 找到。要运行回测工作流:

1. 取消 `exp_configs/train_opds.yml` 中 `weight_file` 参数的注释(默认是被注释掉的)。虽然可以在不设置检查点的情况下运行回测工作流，但这会导致模型结果随机初始化，从而使结果毫无意义。
2. 运行 `python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml`。

回测结果存储在 `outputs/checkpoints/backtest_result.csv` 中。

除了 OPDS 和 PPO 之外，我们还提供了 TWAP ([Time-weighted average price](https://en.wikipedia.org/wiki/Time-weighted_average_price)) 作为一个弱基线。TWAP 的配置文件为 `exp_configs/backtest_twap.yml`。

### 回测与训练流水线测试之间的差距

值得注意的是，回测过程的结果可能与训练期间所用的测试过程的结果不同。
这是因为在训练和回测期间使用了不同的模拟器来模拟市场状况。
在训练流水线中，出于效率原因使用了名为 `SingleAssetOrderExecutionSimple` 的简化模拟器。
`SingleAssetOrderExecutionSimple` 对交易量不作任何限制。
无论订单量是多少，都可以被完全执行。
然而，在回测期间，使用了更真实的模拟器 `SingleAssetOrderExecution`。
它考虑了更接近真实世界场景中的实际约束(例如，交易量必须是最小交易单位的整数倍)。
因此，回测期间实际执行的订单量可能与预期执行的订单量不同。

如果你希望获得与训练流水线中测试时完全相同的结果，可以只运行训练流水线的回测阶段。
为此:
- 修改训练配置。添加你想使用的检查点的路径(参见下面的示例)。
- 运行 `python -m qlib.rl.contrib.train_onpolicy --config_path PATH/TO/CONFIG --run_backtest --no_training`

```yaml
...
policy:
  class: PPO  # PPO, DQN
  kwargs:
    lr: 0.0001
    weight_file: PATH/TO/CHECKPOINT
  module_path: qlib.rl.order_execution.policy
...
```

## 基准测试 (TBD)

为了准确评估使用强化学习算法的模型性能，最好多次运行实验并计算所有试验的平均性能。然而，鉴于模型训练的耗时性，这并不总是可行的。一种替代方法是每个训练任务只运行一次，选取验证性能最高的 10 个检查点来模拟多次试验。在本示例中，我们使用 "Price Advantage (PA)" 作为选取这些检查点的指标。这 10 个检查点在测试集上的平均性能如下:

| **Model**                   | **PA mean with std.** |
|-----------------------------|-----------------------|
| OPDS (with PPO policy)      |  0.4785 ± 0.7815      |
| OPDS (with DQN policy)      | -0.0114 ± 0.5780      |
| PPO                         | -1.0935 ± 0.0922      |
| TWAP                        |   ≈ 0.0 ± 0.0         |

上表还包括 TWAP 作为一个基于规则的基线。TWAP 的理想 PA 应为 0.0，然而，在本示例中，订单执行被分为两个步骤:首先，订单在每半小时内平均拆分，然后在每半小时内的每五分钟平均拆分。由于当天最后五分钟禁止交易，这种做法在一整天的过程中可能与传统 TWAP 略有不同(因为最后一个"半小时"缺少了 5 分钟)。因此，TWAP 的 PA 可以视为一个接近 0.0 的数字。为验证这一点，你可以运行一次 TWAP 回测并查看结果。
