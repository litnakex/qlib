# GATs
* 图注意力网络(Graph Attention Networks, GATs)在图结构数据上运用带掩码的自注意力层。堆叠层中的节点具有不同的权重，它们能够关注其
邻居的特征，而无需任何代价高昂的矩阵运算(例如求逆)，也无需预先知道图的结构。
* Qlib 中使用的这段代码由我们自己用 PyTorch 实现。
* 论文：Graph Attention Networks https://arxiv.org/pdf/1710.10903.pdf
