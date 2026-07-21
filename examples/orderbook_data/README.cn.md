# 简介

本示例尝试演示 Qlib 如何支持没有固定共享频率的数据。

例如，
- 每日价格与成交量数据是固定频率数据。这类数据以固定频率（即每日）到来
- 订单不是固定数据，它们可能在任意时间点到来

为了支持这种非固定频率，Qlib 实现了一个基于 Arctic 的后端。
下面是一个基于该后端导入和查询数据的示例。

# 安装

请参考 mongodb 的[安装文档](https://docs.mongodb.com/manual/installation/)。
当前脚本的默认值会尝试**通过默认端口以无认证方式**连接 localhost。

运行以下命令以安装必要的库
```
pip install pytest coverage gdown
pip install arctic  # NOTE: pip may fail to resolve the right package dependency !!! Please make sure the dependency are satisfied.
```

# 导入示例数据


1. （可选）请按照[本节](https://github.com/microsoft/qlib#data-preparation)的第一部分来**获取 Qlib 的 1min 数据**。
2. 请按照以下步骤下载示例数据
```bash
cd examples/orderbook_data/
gdown https://drive.google.com/uc?id=15FuUqWn2rkCi8uhJYGEQWKakcEqLJNDG  # Proxies may be necessary here.
python ../../scripts/get_data.py _unzip --file_path highfreq_orderbook_example_data.zip --target_dir .
```

3. 请将示例数据导入到你的 mongo db
```bash
python create_dataset.py initialize_library  # Initialization Libraries
python create_dataset.py import_data  # Initialization Libraries
```

# 查询示例

导入这些数据后，你可以运行 `example.py` 来创建一些高频特征。
```bash
pytest -s --disable-warnings example.py   # If you want run all examples
pytest -s --disable-warnings example.py::TestClass::test_exp_10  # If you want to run specific example
```


# 已知限制
尚不支持不同频率之间的表达式计算
