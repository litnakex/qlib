# KRNN
* 代码: [https://github.com/microsoft/FOST/blob/main/fostool/model/krnn.py](https://github.com/microsoft/FOST/blob/main/fostool/model/krnn.py)


# 关于设置/配置的说明
* FOST 中的原始模型使用了 Torch_geometric，但我们没有使用它。
* 请确保你的 CUDA 版本与 torch 版本匹配，以便能够使用 GPU，我们使用 CUDA==10.2 以及 torch.__version__==1.12.1
