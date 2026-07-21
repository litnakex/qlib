# 用于序列学习的时序相关任务调度 (Temporally Correlated Task Scheduling for Sequence Learning)
### 背景
近年来,序列学习吸引了机器学习领域大量的研究关注。在许多应用中,一个序列学习任务通常与多个时序相关的辅助任务相关联,这些任务在使用多少输入信息或预测哪个未来步骤方面有所不同。在股票趋势预测中,如图1所示,人们可以预测股票在不同未来日期(例如明天、后天)的价格。在本文中,我们提出了一个框架,以利用这些时序相关的任务来相互帮助。

### 方法
鉴于通常存在多个时序相关的任务,关键挑战在于在训练过程中使用哪些任务以及何时使用它们。这项工作为序列学习引入了一个可学习的任务调度器,它在训练过程中自适应地选择时序相关的任务。调度器访问模型状态和当前训练数据(例如,当前 minibatch 中的数据),并选择最佳的辅助任务来帮助主任务的训练。调度器和主任务的模型通过双层优化(bi-level optimization)进行联合训练:调度器被训练以最大化模型的验证性能,而模型被训练以最小化由调度器引导的训练损失。该过程如图2所示。

<p align="center"> 
<img src="workflow.png"/>
</p>

在步骤 <img src="https://latex.codecogs.com/png.latex?s" title="s" /> 处,利用训练数据 <img src="https://latex.codecogs.com/png.latex?x_s,y_s" title="x_s,y_s" />,调度器 <img src="https://latex.codecogs.com/png.latex?\varphi" title="\varphi" /> 选择一个合适的任务 <img src="https://latex.codecogs.com/png.latex?T_{i_s}" title="T_{i_s}" />(绿色实线)来更新模型 <img src="https://latex.codecogs.com/png.latex?f" title="f" />(蓝色实线)。在经过 <img src="https://latex.codecogs.com/png.latex?S" title="S" /> 个步骤后,我们在验证集上评估模型 <img src="https://latex.codecogs.com/png.latex?f" title="f" /> 并更新调度器 <img src="https://latex.codecogs.com/png.latex?\varphi" title="\varphi" />(绿色虚线)。

### 实验
由于数据版本不同以及 Qlib 版本不同,本文实验设置中的原始数据和数据预处理方法与现有 Qlib 版本中的实验设置不同。因此,我们根据两种设置提供了两个版本的代码:1) 可用于复现论文实验结果的[代码](https://github.com/lwwang1995/tcts);2) 当前 Qlib 基线中的[代码](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcts.py)。

#### 设置1 (Setting1)
* 数据集:我们使用 [CSI300](http://www.csindex.com.cn/en/indices/index-detail/000300) 上 300 只股票从 01/01/2008 到 08/01/2020 的历史交易数据。我们根据交易时间将数据划分为训练集 (01/01/2008-12/31/2013)、验证集 (01/01/2014-12/31/2015) 和测试集 (01/01/2016-08/01/2020)。

* 主任务 <img src="https://latex.codecogs.com/png.latex?T_k" title="T_k" /> 指的是如下预测股票 <img src="https://latex.codecogs.com/png.latex?i" title="i" /> 的收益率,
<div align=center>
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;k}}{price_i^{t&plus;k-1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+k}}{price_i^{t+k-1}}-1" />
</div>

* 时序相关任务集 <img src="https://latex.codecogs.com/png.latex?\mathcal{T}_k&space;=&space;\{T_1,&space;T_2,&space;...&space;,&space;T_k\}" title="\mathcal{T}_k = \{T_1, T_2, ... , T_k\}" />,在本文中,<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" />、<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_5" title="\mathcal{T}_5" /> 和 <img src="https://latex.codecogs.com/png.latex?\mathcal{T}_{10}" title="\mathcal{T}_{10}" /> 分别用于 <img src="https://latex.codecogs.com/png.latex?T_1" title="T_1" />、<img src="https://latex.codecogs.com/png.latex?T_2" title="T_2" /> 和 <img src="https://latex.codecogs.com/png.latex?T_3" title="T_3" />。

#### 设置2 (Setting2)
* 数据集:我们使用 [CSI300](http://www.csindex.com.cn/en/indices/index-detail/000300) 上 300 只股票从 01/01/2008 到 08/01/2020 的历史交易数据。我们根据交易时间将数据划分为训练集 (01/01/2008-12/31/2014)、验证集 (01/01/2015-12/31/2016) 和测试集 (01/01/2017-08/01/2020)。

* 主任务 <img src="https://latex.codecogs.com/png.latex?T_k" title="T_k" /> 指的是如下预测股票 <img src="https://latex.codecogs.com/png.latex?i" title="i" /> 的收益率,
<div align=center>
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;1&plus;k}}{price_i^{t&plus;1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+1+k}}{price_i^{t+1}}-1" />
</div>

* 在 Qlib 基线中,<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" /> 用于 <img src="https://latex.codecogs.com/png.latex?T_1" title="T_1" />。

### 实验结果
您可以在[论文](http://proceedings.mlr.press/v139/wu21e/wu21e.pdf)中找到设置1的实验结果,并在此[页面](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)中找到设置2的实验结果。
