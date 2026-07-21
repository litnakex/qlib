# DoubleEnsemble
* DoubleEnsemble 是一个集成框架，它利用基于学习轨迹的样本重加权和基于打乱的特征选择，来同时解决低信噪比和特征数量不断增加这两个问题。它根据每个样本的训练动态来识别关键样本，并通过打乱来根据每个特征的消融影响提取关键特征。该模型适用于广泛的基础模型，能够提取复杂的模式，同时缓解金融市场预测中的过拟合和不稳定问题。
* Qlib 中使用的这段代码由我们自己实现。
* 论文：DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis [https://arxiv.org/pdf/2010.01265.pdf](https://arxiv.org/pdf/2010.01265.pdf)。
