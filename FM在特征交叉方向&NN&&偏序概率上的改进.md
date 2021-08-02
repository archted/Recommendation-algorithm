LR，它最大的优势是简单，于是可以以很高的效率去学习和预测，因而在很多领域都被广泛应用。由于 LR 只能捕捉关于特征的线性信息，而无法捕捉非线性信息——特别是交叉特征信息，人们对 LR 进行了各种升级改造。

FM 模型不仅在模型本身能够满足下列两个特性，还保证了训练和预测的效率为 O(kn)，因而是非常优秀的模型、被广泛运用：

- 能够处理大规模的稀疏数据，并保有足够好的泛化性能（generalization performance）；
- 同时，能够自动地学习到特征交叉带来的信息。

FM 模型特点总结成一句话就是，FM 解决了稀疏数据场景下的自动特征组合问题。

## 在特征交叉方向上的改进

对 FM 的第一个改进方向是在特征交叉方向上去做改进，以捕捉更多信息；

### FFM (Field-aware FM)

相比 FM，FFM 为每个特征构造的不再是隐向量，而是隐矩阵

缺点是它训练时的复杂度是非常高的，使得其应用受限。

## 引入深度神经网络

对 FM 进行改进的第二个方向是引入深度神经网络，利用 NN 捕捉高阶交叉特征。

### DeepFM

在 Wide & Deep 框架上，将 LR 部分替换成了 FM，以增强 wide 部分对二阶交叉特征的捕捉能力。目前来说，它已被业界快速跟进和应用到推荐、广告、搜索等场景。

### NFM (Neural FM)

DeepFM 是用 Wide & Deep 框架，在 FM 旁边加了一个 NN，最后一并 sigmoid 输出。NFM 的做法则是利用隐向量逐项相乘得到的向量作为 MLP 的输入，构建的 FM + NN 模型。

## 利用偏序概率，迁移到 rank 任务上

以前提到过 [RankNet 的创新](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/#RankNet-的创新)。通过引入偏序概率，我们可以把任何一个模型转变成 pairwise 的排序模型。

具体来说，对于待排序的 item xi 和 xj，有模型的打分函数 f，从而求得得分 si=f(xi), sj=f(xj)，而后得到偏序概率

Pij=P(xi⊳xj)=sigmoid(si,sj).

这样，我们就将排序问题中的偏序，转换成了二分类问题（xi 是否应该排在 xj 之前）。之后只需要套用二分类问题的解法即可。

### Pairwise FM

Pairwise FM 的做法就是如此。只需要将上述打分函数 f 设为 FM 即可。

### LambdaFM

和 [LambdaMART](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/) 中的做法一样，若在 FM 的基础上，将梯度上辅以 ΔZ 作为 pair 的权重，则变成了 listwise 的排序算法。





在排序问题中使用的机器学习算法，被称为 Learning to Rank (LTR) 算法，或者 Machine-Learning Rank (MLR) 算法。

LTR 算法通常有三种手段，分别是：Pointwise、Pairwise 和 Listwise。Pointwise 和 Pairwise 类型的 LTR 算法，将排序问题转化为**回归**、**分类**或者**有序分类**问题。Listwise 类型的 LTR 算法则另辟蹊径，将用户查询（Query）所得的结果作为整体，作为训练用的实例（Instance）。