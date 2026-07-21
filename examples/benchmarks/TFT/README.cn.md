# Temporal Fusion Transformers 基准测试
## 来源
**参考文献**: Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." arXiv preprint arXiv:1912.09363 (2019).

**GitHub**: https://github.com/google-research/google-research/tree/master/tft

## 运行工作流
用户可以参照 ``workflow_by_code_tft.py`` 来运行该基准测试。

### 注意事项
1. 请**注意**，此脚本仅支持 `Python 3.6 - 3.7`。
2. 如果你机器上的 CUDA 版本不是 10.0，请记得在机器上运行以下命令 `conda install anaconda cudatoolkit=10.0` 和 `conda install cudnn`。
3. 该模型必须在 GPU 上运行，否则会抛出错误。
4. 新的数据集应当在 ``data_formatters`` 中注册，详情请参阅源代码。
