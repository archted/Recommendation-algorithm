##  CTR预估算法

CTR预估，简而言之，就是给某个用户推送一个广告，预测该广告被点击的概率。

### GBDT+LR模型

#### 问题背景

针对CTR预估问题，算法提出的背景：

点击率预估模型中的训练样本量大，可大上亿级别

常采用速度较快的LR。但LR是线性模型，学习能力有限

特征工程非常重要，主要通过人工方式找到有区分度的特征、特征组合，对人的要求高，时间成本高

=> 如何自动发现有效的特征及特征组合，弥补人工经验不足，缩短LR实验周期

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064526469.png" alt="image-20210724064526469" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064605220.png" alt="image-20210724064605220" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064532962.png" alt="image-20210724064532962" style="zoom:100%;" />

#### GBDT原理



论文 Greedy function Approximation – A Gradient Boosting Machine

https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451

1. 模型原理

DBDT（Gradient Boosting Decision Tree），从命名看由两部分组成。

1）回归树（•Regression Decision Tree）

GBDT由多棵CART回归树组成，将累加所有树的结果作为最终结果，拟合的目标是一个梯度值（连续值，实数），GBDT用来做回归预测，调整后也可以用于分类

2）•梯度迭代 Gradient Boosting

Boosting，迭代，即通过迭代多棵树来共同决策

Gradient Boosting，每一次建立模型是在之前模型损失函数的梯度下降方向

因此，GBDT基于集成学习中的boosting思想，将所有树的结果累加起来，最为最终的结果 => 每一棵树学的是之前所有树结果和的残差。

2. 模型公式

第t次迭代后的函数 = 第t-1次迭代后的函数 + 第t次迭代函数的增量

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724070020422.png" alt="image-20210724070020422" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724070025617.png" alt="image-20210724070025617" style="zoom:100%;" />

拟合负梯度，最终结果为每次迭代增量的累加，f0(x)为初始值

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724070032650.png" alt="image-20210724070032650" style="zoom:100%;" />

3. GBDT做分类任务

GBDT如何做分类：CART输出的是实数，将实数映射到0-1之间就可以作为概率，

类似于将sigmoid函数作用于线性回归就可以等价于逻辑回归的模型。

4. GBDT用于iris数据集的分类任务

```python
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
scores = cross_val_score(clf, iris.data,iris.target, cv=3)
print('GBDT准确率：%0.4lf' %scores.mean()) #0.9604
```

5. GBDT用于波士顿房价的回归任务

```python
clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')
clf.fit(X_train, y_train)
print('GBDT回归MSE：',mean_squared_error(y_test, clf.predict(X_test)))
print('各特征的重要程度：',clf.feature_importances_)
# train_score_:表示在样本集上每次迭代以后的对应的损失函数值。
plt.plot(np.arange(500), clf.train_score_, 'b-')  
plt.show()

/*
GBDT回归MSE： 25.005547777904933
各特征的重要程度： [2.93069780e-02 1.31862591e-04 3.51183430e-03 2.03479008e-04
 3.20052149e-02 4.87968151e-01 9.45362802e-03 3.92311619e-02
 2.56743188e-03 1.76580519e-02 2.87743458e-02 1.06581698e-02
 3.38529691e-01]
*/

```

<img src="assets\image-20210724082459722.png" alt="image-20210724082459722" style="zoom: 67%;" />

6. 集成学习方法

Boosting，通过将弱学习器提升为强学习器的集成方法来提高预测精度（比如AdaBoost，GBDT）

Bagging，通过自助采样的方法生成众多并行式的分类器，通过“少数服从多数”的原则来确定最终的结果（比如Random Forest）

#### GBDT+LR算法原理

0. 总览

Practical Lessons from Predicting Clicks on Ads at Facebook,2014 （Facebook经典CTR预估论文）

http://quinonero.net/Publications/predicting-clicks-facebook.pdf

具有stacking思想的二分类器模型(GBDT+LR)，用来解决二分类问题

通过GBDT将特征进行组合，然后传入给线性分类器

LR对GBDT产生的输入数据进行分类（使用L1正则化防止过拟合）

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064838708.png" alt="image-20210724064838708" style="zoom:100%;" />

1. 使用GBDT进行新特征构造：

当GBDT训练好做预测的时候，输出的并不是最终的二分类概率值，**而是要把模型中的每棵树计算得到的预测概率值所属的叶子结点位置记为1(即特征激活)**=> 构造新的训练数据

右图有2棵决策树，一共有5个叶子节点

如果一个实例，选择了第一棵决策树的第2个叶子节点。同时，选择第2棵子树的第1个叶子节点。那么前3个叶子节点中，第2位设置为1，后2个叶子节点中，第1位设置为1。concatenate所有特征向量，得到[0,1,0,1,0]，即Transformed Features。



<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075037124.png" alt="image-20210724075037124" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075056072.png" alt="image-20210724075056072" style="zoom:100%;" />向量由0，1构成，输出给LR，最后由LR模型输出二分类概率值。

2. GBDT+LR与LR的对比试验

使用GBDT+LR，相比单纯的LR和GBDT，在Loss上减少了3%，提升作用明显

前500棵决策树，会让NE下降，而后1000棵决策树对NE下降不明显，不超过0.1%（GBDT决策树规模超过500时，效果不明显）

限制每棵树不超过12个叶子节点

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075216728.png" alt="image-20210724075216728" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075222042.png" alt="image-20210724075222042" style="zoom:100%;" />

3. 特征重要度试验

所有特征的重要度总和是1

TOP10个特征，贡献了50%的重要程度，而后面300个特征，贡献了1%的重要度。

大量弱特征的累积也很重要，不能都去掉。如果去掉部分不重要的特征，对模型的影响比较小

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075429767.png" alt="image-20210724075429767" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075439502.png" alt="image-20210724075439502" style="zoom:100%;" />

4. 评价指标

a)NE

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724075542488.png" alt="image-20210724075542488" style="zoom:100%;" />

NE，Normalized Cross-Entropy，NE = 每次展现时预测得到的log loss的平均值，除以对整个数据集的平均log loss值

p代表训练数据的平均经验CTR，即background CTR，NE对background CTR不敏感，NE数值越小预测效果越好

b)Calibration

预估CTR除以真实CTR，即预测的点击数与真实点击数的比值。数值越接近1，预测效果越好。

c)AUC

**衡量排序质量的良好指标**，但是无法进行校准，也就是如果我们希望得到预估准确的比值，而不是最优的排序结果，那么需要使用NE或者Calibration

e.g 模型预估值都是实际值的两倍，对于NE来说会提升，此时乘以0.5进行校准。但是对于AUC来说会保持不变

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724080102386.png" alt="image-20210724080102386" style="zoom:100%;" />

所有真实类别为1的样本中，预测类别为1的比例

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724080137687.png" alt="image-20210724080137687" style="zoom:100%;" />

所有真实类别为0的样本中，预测类别为1的比例

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724080225437.png" alt="image-20210724080225437" style="zoom:100%;" />

ROC曲线，横轴是FPRate，纵轴是TPRate

AUC=ROC曲线下的面积

如果TPRate=FPRate，也就是y=x，即真实类别不论是1还是0的样本，分类器预测为1的概率是相等的

分类器对于正样本和负样本没有区分能力，此时AUC=0.5

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724080355779.png" alt="image-20210724080355779" style="zoom:100%;" />

如果TPRate>FPRate，也就是y>x，也就是AUC>0.5，有分类能力，极端情况下AUC=0，此时TPRate=1, FPRate=0，也就是正样本预测正确，负样本错分为0

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724080400931.png" alt="image-20210724080400931" style="zoom:100%;" />

#### 代码(GBDT+LR对8万个样本进行分类预测)



`from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier`

数据集
随机生成**二分类样本**8万个，每个样本20个特征
采用RF，RF+LR，GBDT，GBDT+LR进行二分类预测
使用AUC指标进行评估
方法:
Step1，将数据分成训练集和测试集，其中训练集分成GBDT训练集和LR训练集

总样本数8万个，训练集4万 测试集4万,训练集分为Train1 2万个 Train2 2万个

Step2，使用train1训练得到GBDT模型参数，

Step3，在训练好的GBDT模型上使用train2训练LR。
Step4，对比不同算法，绘制ROC曲线

一、基于随机森林的特征变换

```python
# 基于随机森林的特征变换
n_estimator = 10
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf.fit(X_train, y_train)
# 得到OneHot编码
rf_enc = OneHotEncoder(categories='auto')
rf_enc.fit(rf.apply(X_train))
# 使用OneHot编码作为特征，训练LR
rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
# 使用LR进行预测
y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)


```

二、基于GBDT的特征变换

```python
# 基于GBDT的特征变换
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd.fit(X_train, y_train)
# 得到OneHot编码
grd_enc = OneHotEncoder(categories='auto')
grd_enc.fit(grd.apply(X_train)[:, :, 0])
# 使用OneHot编码作为特征，训练LR
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 使用LR进行预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
```

三、分别直接使用GBDT、RF、LR进行预测

```python
# 直接使用GBDT进行预测
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
# 直接使用RF进行预测
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_skl = roc_curve(y_test, y_pred_rf)
# 直接使用LR进行预测
LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
LR.fit(X_train, y_train)
y_pred = LR.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred)
# 绘制ROC曲线
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724081638250.png" alt="image-20210724081638250" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724081647619.png" alt="image-20210724081647619" style="zoom:100%;" />

**对预测结果影响：**

**有了正确的特征（基于决策树）和模型（基于LR），对预测结果的影响最大，其他的影响因素都不大**

**数据实时性，学习率，数据采样对结果的影响较小，而采用一个正确的模型至关重要**

### Wide & Deep模型

#### 总览

Wide & Deep Learning for Recommender Systems，2016
https://arxiv.org/abs/1606.07792

Google在2016年基于TensorFlow发布的用于分类和回归的模型，并应用到了 Google Play 的应用推荐中 

推荐系统的挑战是memorization与generalization（这不是所有机器学习都要面对的问题吗）
结合线性模型的记忆能力和DNN模型的泛化能力（即LR+DNN），在训练过程中同时优化两个模型的参数

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724083656356.png" alt="image-20210724083656356" style="zoom:100%;" />



#### wide推荐

Linear Regression，特征组合需要人来设计

一个有d个特征的样本$X=[x_1,x_2,x_3,⋯,x_d ]$,模型的参数$W=[w_1,w_2,w_3,⋯,w_d ]$

实际中往往需要交叉特征<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724084032252.png" alt="image-20210724084032252" style="zoom:100%;" />(第k个交叉特征)

最终Wide模型<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724084010797.png" alt="image-20210724084010797" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724084310926.png" alt="image-20210724084310926" style="zoom:100%;" />

wide模型是线性模型，输入特征可以是连续特征，也可以是稀疏的离散特征。**离散特征通过交叉可以组成更高维的离散特征。**

#### Deep推荐

1. Wide & deep 模型特征

a)Deep模型使用的特征：连续特征，Embedding后的离散特征，

使用前馈网络模型，特征首先转换为低维稠密向量，作为第一个隐藏层的输入，解决维度爆炸问题

b)Wide模型使用的特征：Cross Product Transformation生成的组合特征，但无法学习到训练集中没有出现的组合特征

2. 模型融合

Wide join Deep ：<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724084754069.png" alt="image-20210724084754069" style="zoom:100%;" />

​       $φ(X)$代表X的Cross Product Transformation

​       $W_{wide}$代表Wide模型的权重向量

​       $W_{deep}$代表Deep模型的权重向量

​       $α^{l_f}$代表最终的神经网络数结果，b为偏差

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724084954867.png" alt="image-20210724084954867" style="zoom:100%;" />

两个模型融合的方法：

ensemble：两个模型分别对全量数据进行预测，然后根据权重组合最终的预测结果

joint training：wide和deep的特征合一，构成一个模型进行预测

#### 对照试验

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724085202179.png" alt="image-20210724085202179" style="zoom:100%;" />

Wide & Deep模型不论在线下还是线上相比于单独的wide模型和单独的Deep模型，效果都有明显提升。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724085219112.png" alt="image-20210724085219112" style="zoom:100%;" />



### NFM模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091527890.png" alt="image-20210724091527890" style="zoom:100%;" />

FNN, Wide & Deep, DeepFM都是在DNN部分，对embedding之后的特征进行concatenate，没有充分进行特征交叉计算。

NFM算法是对embedding直接采用对位相乘（element-wise）后相加起来作为交叉特征，然后通过DNN直接将特征压缩，最后concatenate linear部分和deep部分的特征。

两种FM和DNN的结合方式：

DeepFM, 并行结构，FM和DNN分开计算

NFM,串行架构，将FM的结果作为DNN的输入



<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091657862.png" alt="image-20210724091657862" style="zoom:100%;" />

对于输入X的预测公式：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091555254.png" alt="image-20210724091555254" style="zoom:100%;" />

Embedding层：全连接层，将稀疏输入映射到一个密集向量，得到$V_x={x_1 v_1,⋯x_n v_n }$

BI层: 池化操作，将一系列的Embedding向量转换为一个向量<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091637177.png" alt="image-20210724091637177" style="zoom:100%;" />

隐藏层：神经网络的全连接层

预测层：将隐藏层输出到n*1的全连接层，得到预测结果<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091643792.png" alt="image-20210724091643792" style="zoom:100%;" />



FM 通过隐向量完成特征组合工作，同时解决了稀疏的问题。但是 FM 对于 non-linear 和 higher-order 特征交叉能力不足，因此可以使用FM和DNN来弥补这个不足 => NFM

BI层，将每个特征embedding进行两两做元素积， BI层的输出是一个 k维向量（隐向量的大小），BI层负责了二阶特征组合

可以将FM看成是NFM模型 Hidden Layer层数为0的一种特殊情况

### CTR模型总结

#### Poly2模型

LR模型可解释性强，但是需要人工交叉特征，Poly2在LR的基础上考虑特征自动交叉

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090650914.png" alt="image-20210724090650914" style="zoom:100%;" />

Ploy2(Degree-2 Polynomial)只有在这两个特征同时出现时才能使用，如果某个特征缺失的情况下，就可以使用FM

#### FM模型

1. FM与MF比较

MF只考虑了UserID, ItemID的特征，而实际我们需要考虑更多的特征，甚至是多个特征之间的组合

MF只解决评分预测问题，而实际问题可能是回归和分类问题，需要更通用的解决方式

FM考虑了更多的特征，以及二阶特征组合，可以作为通用的回归和分类算法

2.FM模型讲解

Poly2可以自动交叉特征的，但是当特征维度高且稀疏时，权重W很难收敛

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064323237.png" alt="image-20210724064323237" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724064328659.png" alt="image-20210724064328659" style="zoom:100%;" />

通过MF完成二阶特征交叉的**矩阵补全**

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090839353.png" alt="image-20210724090839353" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090843714.png" alt="image-20210724090843714" style="zoom:100%;" />

#### FFM模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090923218.png" alt="image-20210724090923218" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090928628.png" alt="image-20210724090928628" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724090932103.png" alt="image-20210724090932103" style="zoom:100%;" />

#### FNN模型

CTR中大部分特征是离散、高维且稀疏的，需要embedding后才能进行深度学习

使用FM对embedding层进行初始化，即每个特征对应一个偏置项$w_i$  和一个k维向量$v_i$

$z_i=w_0^i⋅x[start_i:end_i ]=(w_i,v_i^1,v_i^2,⋯v_i^k )$

$l_1=tanh⁡(W_1 z+b_1 )$

$l_2=tanh⁡(W_2 l_1+b_2 )$

$y=sigmoid(W_3 l_2+b_3 )$

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091234940.png" alt="image-20210724091234940" style="zoom:100%;" />

**FNN实际上是wide** **&** **deep模型中的deep模型**，同时FNN使用FM进行参数初始化

#### Wide & Deep模型

Wide模型，采用Linear Regression，解决模型的记忆能力（特征组合需要人来设计）

Deep模型，即FNN，解决模型的泛化能力

Joint Training，同时训练Wide模型和Deep模型，并将两个模型的结果的加权作为最终的预测结果

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091330902.png" alt="image-20210724091330902" style="zoom:100%;" />

#### DeepFM模型

DeepFM是将Wide & Deep模型中的Wide替换成了FM模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091500191.png" alt="image-20210724091500191" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091411440.png" alt="image-20210724091411440" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091425957.png" alt="image-20210724091425957" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091437082.png" alt="image-20210724091437082" style="zoom:100%;" />

#### NFM模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091527890.png" alt="image-20210724091527890" style="zoom:100%;" />

FNN, Wide & Deep, DeepFM都是在DNN部分，对embedding之后的特征进行concatenate，没有充分进行特征交叉计算。

可以将FM看成是NFM模型 Hidden Layer层数为0的一种特殊情况





### 工具：DeepCTR

https://github.com/shenweichen/DeepCTR

实现了多种CTR深度模型

与Tensorflow 1.4和2.0兼容

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724085326737.png" alt="image-20210724085326737" style="zoom:100%;" />

#### 代码(使用WDL/NFM对Movielens进行推荐)

数据集：MovieLens_Sample

包括了多个特征：user_id, movie_id, rating, timestamp, title, genres, gender, age, occupation, zip

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724085411481.png" alt="image-20210724085411481" style="zoom:100%;" />

##### Wide & Deep模型代码

使用DeepCTR中的WDL，计算RMSE值

```python
data = pd.read_csv("movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['rating']
……
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}
# 使用WDL进行训练
model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=10, verbose=True, validation_split=0.2, )
# 使用WDL进行预测
pred_ans = model.predict(test_model_input, batch_size=256)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724085507865.png" alt="image-20210724085507865" style="zoom:100%;" />

##### NFM模型代码

使用DeepCTR中的NFM，计算RMSE值

```python
data = pd.read_csv("movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['rating']
……
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}
# 使用NFM进行训练
model = NFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=10, verbose=True, validation_split=0.2, )
# 使用NFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724091932630.png" alt="image-20210724091932630" style="zoom:100%;" />

### 总结(CTR预估)

•低阶和高阶特征在CTR预估中都很重要

•低阶特征组合采用FM效果好，高阶特征组合采用DNN效果好

•采用FM和DNN的两种结合方式，串行 VS 并行

•一般使用FM做二阶特征组合，在CTR预估中应用范围很广（衍生出来各种模型）