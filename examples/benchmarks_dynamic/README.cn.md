# 简介
由于金融市场环境的非平稳性,数据分布在不同时期可能发生变化,这使得基于训练数据构建的模型在未来的测试数据上表现下降。
因此,使预测模型/策略适应市场动态对于模型/策略的表现非常重要。

下表展示了不同解决方案在不同预测模型上的表现。

## Alpha158 Dataset
这是 [qlib 数据的众包版本](data_collector/crowd_source/README.md):https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
rm -f qlib_bin.tar.gz
```

| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|------|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.0945|0.5989|0.1069   |0.6495     |0.0857             |1.3682             |-0.0986       |
| DDG-DA[Linear]   |Alpha158 |0.0983|0.6157|0.1108   |0.6646     |0.0764             |1.1904             |-0.0769       |
| RR[LightGBM]     |Alpha158 |0.0816|0.5887|0.0912   |0.6263     |0.0771             |1.3196             |-0.0909       |
| DDG-DA[LightGBM] |Alpha158 |0.0878|0.6185|0.0975   |0.6524     |0.1261             |2.0096             |-0.0744       |

- `Alpha158` 数据集的标签预测跨度(label horizon)设置为 20。
- 滚动时间间隔设置为 20 个交易日。
- 测试滚动周期为 2017 年 1 月至 2020 年 8 月。
- 结果基于众包版本。Yahoo 版本的 qlib 数据不包含 `VWAP`,因此所有相关因子均缺失并以 0 填充,这导致矩阵秩亏(matrix does not have full rank),使得 DDG-DA 的下层优化无法求解。
