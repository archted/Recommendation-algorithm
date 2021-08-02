# 矩阵分解与surprise工具使用

## 初识矩阵分解

### 什么是矩阵分解

矩阵分解(MF,Matrix Factorization),属于隐语义模型的一种。可解释性差，隐含特征计算机能理解就好，相比之下ItemCF可解释性强。

Matrix factorization techniques for recommender systems, [J]. Computer 2009

http://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Recommender-Systems.pdf

用户与物品之间存在着隐含的联系

通过隐含特征（Latent Factor）联系用户兴趣和物品，基于用户行为的自动聚类

我们可以指定隐特征的个数，粒度可粗（隐特征少），可细（隐特征多）

计算物品属于每个隐特征的权重，物品有多个隐特征的权重





现在想要为用户找到其感兴趣的item推荐给他

用矩阵表示收集到的用户行为数据，12个用户，9部电影

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724114123522-1627796985680.png" alt="image-20210724114123522" style="zoom:100%;" />

**矩阵分解要做的是预测出矩阵中缺失的评分**，使得预测评分能反映用户的喜欢程度

可以把预测评分最高的前K个电影推荐给用户了。



如何从评分矩阵中分解出User矩阵和Item矩阵？

**目标：学习出User矩阵和Item矩阵，使得User矩阵*Item矩阵与评分矩阵中已知的评分差异最小 => 最优化问题**

隐含特征个数k，k越大，隐类别分得越细，计算量越大。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724114216412-1627796985681.png" alt="image-20210724114216412" style="zoom:100%;" />

假如对上面电影评分矩阵经过矩阵分解得到User矩阵和Item矩阵，进而得到补全后的评分矩阵：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724114624734-1627796985681.png" alt="image-20210724114624734" style="zoom:100%;" /> = 

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724114635528-1627796985681.png" alt="image-20210724114635528" style="zoom:100%;" />  X  <img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724114649335-1627796985681.png" alt="image-20210724114649335" style="zoom:100%;" />



$r_{ui}$表示用户u 对item i 的评分

当 >0时，表示有评分，当=0时，表示没有评分，

$x_u$表示用户u 的向量，k维列向量

$y_i$表示item i 的向量，k维列向量

用户矩阵X，用户数为N

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724115136844-1627796985681.png" alt="image-20210724115136844" style="zoom:100%;" />

商品矩阵Y，商品数为M

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724115143401-1627796985681.png" alt="image-20210724115143401" style="zoom:100%;" />

矩阵的目标函数：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724115151238-1627796985682.png" alt="image-20210724115151238" style="zoom:100%;" />

用户向量与物品向量的内积$x_u^Ty_i$，表示用户u 对物品i 的预测评分

实际评分$r_{ui}$

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724135824953-1627796985682.png" alt="image-20210724135824953" style="zoom:100%;" />表示L2正则项，保证数值计算稳定性，防止过拟合





### 矩阵分解方法

**矩阵分解(matrix factorization,MF)：将矩阵拆解为多个矩阵的乘积**

矩阵分解方法：

EVD（特征值分解）

SVD（奇异值分解）

求解近似矩阵分解的最优化问题

ALS（交替最小二乘法）：ALS-WR

SGD（随机梯度下降）：FunkSVD, BiasSVD, SVD++

### 目标函数最优化问题

ALS，Alternating Least Squares，交替最小二乘法

SGD，Stochastic Gradient Descent，随机梯度下降

详解见【矩阵分解 目标函数最优化问题详解.md】

### EVD（特征值分解）

#### 普通矩阵的矩阵分解

矩阵的特征分解：

特征分解，是将矩阵分解为特征值和特征向量表示的矩阵之积的方法，也称为谱分解

N 维非零向量 v 是 **N×N** 的矩阵 A 的特征向量，当且仅当下式成立

<img src="L5/assets/image-20210725142628457.png" alt="image-20210725142628457" style="zoom:50%;" />

λ为特征值（标量），v为特征值λ对应的特征向量。特征向量被施以线性变换 A 只会使向量伸长或缩短，而方向保持不变

求解$|A-λI|=0$，也称为特征方程

令$p(λ):=|A-λI|=0$称为矩阵的特征多项式

特征多项式是关于λ的N次多项式，特征方程有N个解

对多项式p(λ)进行因式分解，可得

$p(λ)=(λ-λ_1 )^{n_1} (λ-λ_2 )^{n_2}…(λ-λ_k )^{n_k}$

其中$∑_{i=1}^kn_i =N$

而对于每一个特征值$λ_i$，都可以使得

$(A-λ_i I)v=0$

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725143505387-1627797399215.png" alt="image-20210725143505387" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144001926-1627797399216.png" alt="image-20210725144001926" style="zoom:100%;" />



```python
A = np.array([[5,3],
	        [1,1]])
lamda, U = np.linalg.eig(A)
print('特征值: ',lamda)
print('特征向量: ',U)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725143809836-1627797399216.png" alt="image-20210725143809836" style="zoom:100%;" />

#### 对称矩阵的矩阵分解

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725143939212-1627797399216.png" alt="image-20210725143939212" style="zoom:100%;" />

此时，可以发现特征向量不仅线性无关，而且还正交，即

0.97324899*-0.22975292+0.22975292*0.97324899=0

于是，可以将<img src="L5/assets/image-20210725144106300.png" alt="image-20210725144106300" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144114828-1627797399216.png" alt="image-20210725144114828" style="zoom:100%;" />



### 奇异值分解SVD

#### SVD公式推导

矩阵分解中的问题:

1. 很多矩阵都是非对称的

2. 矩阵A不是方阵，即维度为m*n

因此，我们可以将它转化为对称的方阵，因为：$AA^T $与$A^T A$是对称的方阵

因为<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144551835-1627797399216.png" alt="image-20210725144551835" style="zoom:100%;" />

我们可以将A和A的转置矩阵进行相乘，得到对称方阵：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144557400-1627797399216.png" alt="image-20210725144557400" style="zoom:100%;" />



此时<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144606192-1627797399216.png" alt="image-20210725144606192" style="zoom:100%;" />均为对角矩阵，具有相同的非零特征值。

假设这些特征值为<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144630821-1627797399216.png" alt="image-20210725144630821" style="zoom:100%;" />，k不超过m和n，也就是k<=min(m,n)

此时矩阵A的特征值<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144645119-1627797399216.png" alt="image-20210725144645119" style="zoom:100%;" />

我们可以得到为奇异值分解

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144703206-1627797399216.png" alt="image-20210725144703206" style="zoom:100%;" />

P为左奇异矩阵，m*m维

Q为右奇异矩阵，n*n维

Λ对角线上的非零元素为特征值λ1, λ2, ... , λk

在推荐系统中

左奇异矩阵：User矩阵

右奇异矩阵：Item矩阵



<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144757035-1627797399217.png" alt="image-20210725144757035" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144907829-1627797399217.png" alt="image-20210725144907829" style="zoom:100%;" />

奇异值分解：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725144957824-1627797399217.png" alt="image-20210725144957824" style="zoom:100%;" />

λ1为特征值，p1为左奇异矩阵的特征向量

q1为右奇异矩阵的特征向量

```python
from scipy.linalg import svd
import numpy as np
from scipy.linalg import svd
A = np.array([[1,2],
	    [1,1],
	    [0,0]])
p,s,q = svd(A,full_matrices=False)
print('P=', p)
print('S=', s)
print('Q=', q)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725145112868-1627797399217.png" alt="image-20210725145112868" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725145055707-1627797399217.png" alt="image-20210725145055707" style="zoom:100%;" />



#### 使用SVD进行降维

矩阵A：大小为1440*1080的图片

Step1，将图片转换为矩阵

Step2，对矩阵进行奇异值分解，得到p,s,q

Step3，包括特征值矩阵中的K个最大特征值，其余特征值设置为0

Step4，通过p,s',q得到新的矩阵A'，对比A'与A的差别

```python
from scipy.linalg import svd
import matplotlib.pyplot as plt
# 取前k个特征，对图像进行还原
def get_image_feature(s, k):
	# 对于S，只保留前K个特征值
	s_temp = np.zeros(s.shape[0])
	s_temp[0:k] = s[0:k]
	s = s_temp * np.identity(s.shape[0])
	# 用新的s_temp，以及p,q重构A
	temp = np.dot(p,s)
	temp = np.dot(temp,q)
	plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
	plt.show()
    
# 加载256色图片
image = Image.open('./256.bmp') 
A = np.array(image)
# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(A, full_matrices=False)
# 取前k个特征，对图像进行还原
get_image_feature(s, 5)
get_image_feature(s, 50)
get_image_feature(s, 500)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725145818067-1627797399217.png" alt="image-20210725145818067" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725145830423-1627797399217.png" alt="image-20210725145830423" style="zoom:100%;" />

少量的信息（比如10%），可以还原大部分图像信息（比如99%）

当K=50时，我们只需要保存$ (1440+1+1080)*50=126050$个元素，占比$126050/(1440*1080)=$8%

#### 传统SVD在推荐系统中的应用

将user-item评分问题，转化为SVD矩阵分解:



<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150038330-1627797399217.png" alt="image-20210725150038330" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150048825-1627797399217.png" alt="image-20210725150048825" style="zoom:100%;" />

如果我们想要看user2对item3评分，可以得到

$4=-0.46*16.47*(-0.29)+(-0.30)*6.21*(-0.38)+(-0.65)*4.40*(-0.13)+0.28*2.90*0.87+0.02*1.58*(-0.03)$

A中各元素=user行向量，item列向量，奇异值的加权内积

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150415238-1627797399217.png" alt="image-20210725150415238" style="zoom:100%;" />

实际上，我们发现user矩阵的最后一列是没有用到的，而且我们还可以使用更少的特征，比如特征个数=2

得到近似解A':

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150431472-1627797399217.png" alt="image-20210725150431472" style="zoom:100%;" />

综上，传统SVD在推荐系统中的应用可总结为下面几步：

我们可以通过k来对矩阵降维<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150552151-1627797399217.png" alt="image-20210725150552151" style="zoom:100%;" />

第i个用户对第j个物品的评分<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725150556513-1627797399217.png" alt="image-20210725150556513" style="zoom:100%;" />

完整的SVD，可以将M无损的分解成为三个矩阵.

为了简化矩阵分解，我们可以使用k，远小于min(m,n)，对矩阵M近似还原。

但是，传统SVD在使用上存在一些局限性：

**SVD分解要求矩阵是稠密的 => 矩阵中的元素不能有缺失**

所以，类似于数据清洗，我们需要先对矩阵中的缺失元素进行补全

先有鸡，还是先有蛋。实际上**传统SVD更适合做降维**

而且在对矩阵缺失值进行补全时也存在一些问题：

•矩阵往往是稀疏的，大量缺失值 => 计算量大

•填充方式简单粗暴 => 噪音大

### SVD算法家族：FunkSVD, BiasSVD, SVD++

#### FunkSVD算法

1.FunkSVD算法思想：

我们需要设置k，来对矩阵近似求解

矩阵补全以后，再预测，实际上噪音大。矩阵分解之后的还原，只需要关注与原来矩阵中有值的位置进行对比即可，不需要对所有元素进行对比



2.解决思路：

•避开稀疏问题，而且只用两个矩阵进行相乘<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151120502-1627797399217.png" alt="image-20210725151120502" style="zoom:100%;" />

•损失函数=P和Q矩阵乘积得到的评分，与实际用户评分之差

•让损失函数最小化 => 最优化问题(FunkSVD使用SVD进行损失函数的优化)



3.目标函数优化：

最小化损失函数<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151242826-1627797399218.png" alt="image-20210725151242826" style="zoom:100%;" />

为了防止过拟合，增加正则项

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151253318-1627797399218.png" alt="image-20210725151253318" style="zoom:100%;" />

Step1，通过梯度下降法(SGD)，求解P和Q使得损失函数最小化

Step2，通过P和Q将矩阵补全

Step3，针对某个用户i，查找之前值缺失的位置，按照补全值从大到小进行推荐

#### BiasSVD算法

相当于是FunkSVD+Baseline

用户有自己的偏好(Bias)，比如乐观的用户打分偏高

商品也有自己的偏好(Bias)，比如质量好的商品，打分偏高

将与个性化无关的部分，设置为偏好(Bias)部分

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151417300-1627797399218.png" alt="image-20210725151417300" style="zoom:100%;" />

其中，μ：所有记录的整体平均数

bi：用户偏好（自身属性，与商品无关）

bj：商品偏好（自身属性，与用户无关）

优化目标函数

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151457478-1627797399218.png" alt="image-20210725151457478" style="zoom:100%;" />

在迭代过程中，bi,bj的初始值可以设置为0向量，然后进行迭代：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151539918-1627797399218.png" alt="image-20210725151539918" style="zoom:100%;" />

最终得到P和Q

#### SVD++算法

在BiasSVD算法基础上进行了改进，考虑用户的隐式反馈

隐式反馈：没有具体的评分，但可能有点击，浏览等行为

对于某一个用户i，假设他的隐式反馈item集合为I(i)

用户i对商品j对应的隐式反馈修正值为<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151726566-1627797399218.png" alt="image-20210725151726566" style="zoom:100%;" />

用户i所有的隐式反馈修正值之和为<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151731054-1627797399218.png" alt="image-20210725151731054" style="zoom:100%;" />

优化目标函数：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151734481-1627797399218.png" alt="image-20210725151734481" style="zoom:100%;" />

其中，上式除以|I(i)|的平方根，可以消除不同个数引起的差异

在考虑用户隐式反馈的情况下，最终得到P和Q

### Baseline算法

ALS和SGD作为优化方法，应用于很多优化问题

Factor in the Neighbors: Scalable and Accurate Collaborative Filtering，ACM Transactions on Knowledge Discovery from Data, 2010

Baseline算法：基于统计的基准预测线打分

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724144917854-1627796985685.png" alt="image-20210724144917854" style="zoom:100%;" />

μ为所有用户对电影评分的均值

bui：待求的基线模型中用户u给物品i打分的预估值

bu：user偏差（如果用户比较苛刻，打分都相对偏低， 则bu<0；反之，bu>0）；

bi：item偏差，反映商品受欢迎程度

使用ALS进行优化

Step1，固定bu，优化bi

Step2，固定bi，优化bu

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724145105459-1627796985685.png" alt="image-20210724145105459" style="zoom:100%;" />

### 总结

MF是一种隐语义模型，它通过隐类别匹配用户和item来做推荐。

MF对原有的评分矩阵R进行了降维，分成了两个小矩阵：User矩阵和Item矩阵，User矩阵每一行代表一个用户的向量，Item矩阵的每一列代表一个item的向量。将User矩阵和Item矩阵的维度降低到隐类别个数的维度。

根据用户行为，矩阵分解分为显式矩阵分解和隐式矩阵

在显式MF中，用户向量和物品向量的内积拟合的是用户对物品的实际评分

在隐式MF中，用户向量和物品向量的内积拟合的是用户对物品的偏好(0或1)，拟合的强度由置信度控制，置信度又由行为的强度决定。

ALS和SGD都是数学上的优化方法，可以解决最优化问题（损失函数最小化）

ALS-WR算法，可以解决过拟合问题，当隐特征个数很多的时候也不会造成过拟合

ALS，SGD都可以进行并行化处理

SGD方法可以不需要遍历所有的样本即可完成特征向量的求解(随机梯度下降)

Facebook把SGD和ALS两个算法进行了揉合，提出了旋转混合式求解方法，可以处理1000亿数据，效率比普通的Spark MLlib快了10倍

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724164033069-1627796985685.png" alt="image-20210724164033069" style="zoom:100%;" />

### 总结

奇异值分解可以对矩阵进行无损分解

在实际中，我们可以抽取前K个特征，对矩阵进行降维

SVD在降维中有效，抽取不同的K值（10%的特征包含99%的信息）

在评分预测中使用funkSVD，只根据实际评分误差进行目标最优化

在funkSVD的基础上，加入用户/商品偏好 => BiasSVD

在BiasSVD的基础上，考虑用户的隐式反馈 => SVD++

MF存在不足，只考虑user和item特征，对于其他特征的利用我们需要使用新的工具(FM)

## 工具：Surprise与LightFM

### 工具:Surprise

Surprise是scikit系列中的一个推荐系统库

文档：https://surprise.readthedocs.io/en/stable/

#### Surprise中的常用算法

Baseline算法

基于邻域的协同过滤

矩阵分解：SVD，SVD++，PMF，NMF

SlopeOne 协同过滤算法

| **算法**                  | **描述**                                                     |
| ------------------------- | ------------------------------------------------------------ |
| NormalPredictor()         | 基于统计的推荐系统预测打分，假定用户打分的分布是基于正态分布的 |
| BaselineOnly              | 基于统计的基准预测线打分                                     |
| knns.KNNBasic             | 基本的协同过滤算法                                           |
| knns.KNNWithMeans         | 协同过滤算法的变种，考虑每个用户的平均评分                   |
| knns.KNNWithZScore        | 协同过滤算法的变种，考虑每个用户评分的归一化操作             |
| knns.KNNBaseline          | 协同过滤算法的变种，考虑每个用户评分的基线                   |
| matrix_factorzation.SVD   | SVD 矩阵分解算法                                             |
| matrix_factorzation.SVDpp | SVD++ 矩阵分解算法                                           |
| matrix_factorzation.NMF   | 一种非负矩阵分解的协同过滤算法                               |
| SlopeOne                  | SlopeOne 协同过滤算法                                        |

基准算法包含两个主要的算法NormalPredictor和BaselineOnly

Normal Perdictor 认为用户对物品的评分是服从正态分布的，从而可以根据已有的评分的均值和方差 预测当前用户对其他物品评分的分数。

Baseline算法的思想就是设立基线，并引入用户的偏差以及item的偏差

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724145601587-1627796985685.png" alt="image-20210724145601587" style="zoom:100%;" />

μ为所有用户对电影评分的均值

bui：待求的基线模型中用户u给物品i打分的预估值

bu：user偏差（如果用户比较苛刻，打分都相对偏低， 则bu<0；反之，bu>0）；

bi为item偏差，反映商品受欢迎程度

ALS 求得Bi和Bu，也就是评分矩阵

#### Surprise中的评价指标

RMSE(Root Mean Square Error,均方根误差)，根指根号

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150814744-1627796985685.png" alt="image-20210724150814744" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150827052-1627796985685.png" alt="image-20210724150827052" style="zoom:100%;" />

MAE(Mean Absolute Error ，平均绝对误差)

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150841550-1627796985685.png" alt="image-20210724150841550" style="zoom:100%;" />

FCP(Fraction of Concordant Pairs,一致对的分数)

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724151225037-1627796985685.png" alt="image-20210724151225037" style="zoom:100%;" />

#### [code]对Movelens进行推荐

##### 使用ALS的Python代码实现

数据集：MovieLens，下载地址：https://www.kaggle.com/jneupane12/movielens/download，主要使用的文件：ratings.csv，格式：userId, movieId, rating, timestamp，记录了用户在某个时间对某个movieId的打分情况

**我们需要补全评分矩阵，然后对指定用户，比如userID为1-5进行预测**。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724143419533-1627796985686.png" alt="image-20210724143419533" style="zoom:100%;" />

ALS 算法Python代码的实现：

https://github.com/tushushu/imylu/blob/master/imylu/recommend/als.py

直接使用该ALS的实现方法：

```python
class Matrix(object):
    ...
class ALS(object):
    ...
X = load_movie_ratings()
model = ALS()
model.fit(X, k=3, max_iter=3)
print("对用户进行推荐")
user_ids = range(1, 5)
predictions = model.predict(user_ids, n_items=2)
for user_id, prediction in zip(user_ids, predictions):
    _prediction = [format_prediction(item_id, score)
                   for item_id, score in prediction]
    print("User id:%d recommedation: %s" % (user_id, _prediction))

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724143601459-1627796985685.png" alt="image-20210724143601459" style="zoom:100%;" />

##### 使用Surprise对Movelens进行推荐

```python
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic
from surprise import accuracy
from surprise.model_selection import KFold
# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()
```

1.Baseline算法，使用ALS进行优化

```python
# Baseline算法，使用ALS进行优化
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
algo = BaselineOnly(bsl_options=bsl_options)
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

```

ALS参数:

reg_i：物品的正则化参数，默认为10。

reg_u：用户的正则化参数，默认为15 。

n_epochs：迭代次数，默认为10

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150407407-1627796985686.png" alt="image-20210724150407407" style="zoom:100%;" />



2.Baseline算法，使用SGD进行优化

```python
bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

```

SGD参数:

reg：代价函数的正则化项，默认为0.02。

learning_rate：学习率，默认为0.005。

n_epochs：迭代次数，默认为20。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150513106-1627796985686.png" alt="image-20210724150513106" style="zoom:100%;" />

3.NormalPredictor进行求解

```python
algo = NormalPredictor()
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724150617026-1627796985686.png" alt="image-20210724150617026" style="zoom:100%;" />

#### Surprise工具中的SVD家族

1.biasSVD算法

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151854435-1627797532311.png" alt="image-20210725151854435" style="zoom:100%;" />

使用Surprise工具中的SVD

参数:

n_factors: k值，默认为100

n_epochs：迭代次数，默认为20

biased：是否使用biasSVD，默认为True

verbose:输出当前epoch，默认为False

reg_all:所有正则化项的统一参数，默认为0.02

reg_bu：bu的正则化参数，reg_bi：bi的正则化参数

reg_pu：pu的正则化参数，reg_qi：qi的正则化参数

2.funkSVD算法

使用Surprise工具中的SVD

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725151957743-1627797532312.png" alt="image-20210725151957743" style="zoom:100%;" />

参数:

n_factors: k值，默认为100

n_epochs：迭代次数，默认为20

**biased：是否使用biasSVD，设置为False**

verbose:输出当前epoch，默认为False

reg_all:所有正则化项的统一参数，默认为0.02

reg_bu：bu的正则化参数，reg_bi：bi的正则化参数

reg_pu：pu的正则化参数，reg_qi：qi的正则化参数

3.SVD++算法

使用Surprise工具中的SVDpp

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725152037031-1627797532312.png" alt="image-20210725152037031" style="zoom:100%;" />

参数:

n_factors: k值，**默认为20**

n_epochs：迭代次数，默认为20

verbose:输出当前epoch，默认为False

reg_all:所有正则化项的统一参数，默认为0.02

reg_bu：bu的正则化参数，reg_bi：bi的正则化参数

reg_pu：pu的正则化参数，reg_qi：qi的正则化参数

reg_yj：yj的正则化参数

#### [code]对MovieLens进行推荐

数据集：MovieLens

下载地址：https://www.kaggle.com/jneupane12/movielens/download

主要使用的文件：ratings.csv

格式：userId, movieId, rating, timestamp

记录了用户在某个时间对某个movieId的打分情况

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725152306083-1627797532313.png" alt="image-20210725152306083" style="zoom:100%;" />

我们需要补全评分矩阵，然后对指定用户，比如userID为1-5进行预测

```python
# 使用funkSVD
algo = SVD(biased=False)
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
pred = algo.predict(uid, iid, r_ui=4, verbose=True)


```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725152436266-1627797532313.png" alt="image-20210725152436266" style="zoom:100%;" />

K折交叉验证只是为了取到多个RMSE，然后取平均值，每次训练都是在一个新模型上训练，并不会改善训练效果？

### 工具:LightFM

LightFM是Python推荐算法库，具有隐式和显式反馈的多种推荐算法实现。

易用、快速（通过多线程模型估计），能够产生高质量的结果。

### 总结

python开源推荐系统，包含了多种经典的推荐算法

官方文档：https://surprise.readthedocs.io/en/stable/

数据集：可以使用内置数据集（Movielens等），也可以自定义数据集

优化算法：支持多种优化算法，ALS，SGD

预测算法：包括基线算法，邻域方法，矩阵分解，SlopeOne等

相似性度量：内置cosine，MSD，pearson等

scikit家族，可以使用GridSearchCV自动调参，方便比较各种算法结果

