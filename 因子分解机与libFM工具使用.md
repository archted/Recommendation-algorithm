# 因子分解机与libFM工具使用

## 因子分解机

### MF的局限性

矩阵分解：

将矩阵拆解为多个矩阵的乘积

矩阵分解方法：

EVD（特征值分解）

SVD（奇异值分解）

求解近似矩阵分解的最优化问题

ALS（交替最小二乘法）：ALS-WR

SGD（随机梯度下降）：FunkSVD, BiasSVD, SVD++

以上的MF，我们都只考虑user和item特征，但实际上一个预测问题包含的特征维度可能很多：比如时间维度



### FM算法

**因子分解机（Factorization Machine，简称FM），又称分解机器，旨在解决大规模稀疏数据下的特征组合问题。**

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726062329272.png" alt="image-20210726062329272" style="zoom:100%;" />

<center>User-Item矩阵只有两个维度</center>

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726062527320.png" alt="image-20210726062527320" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726062513964.png" alt="image-20210726062513964" style="zoom:100%;" />

<center>多维度矩阵</center>

蓝色部分：当前评分用户信息

红色部分：当前被评分电影

黄色：当前评分用户，评分过的其他电影

绿色：评分时间（按照2009年1月开始的间隔月数）

棕色：用户评分过的上一部电影



线性回归目标预估函数

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063122677.png" alt="image-20210726063122677" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063126115.png" alt="image-20210726063126115" style="zoom:100%;" />

认为变量之间是相互独立的，没有考虑变量之间的相互关系，比如性别+年龄的组合

对于二阶表达式：<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063143632.png" alt="image-20210726063143632" style="zoom:100%;" />

二阶特征组合：考虑了两个特征变量之间的相互影响。但是使用Wij进行二阶特征组合的参数估计存在问题，即如果观察样本中没有出现过该交互的特征分量，那么直接估计将为0

**已经将泛化的问题转换为使用MF预测w系数的问题。对于Wij的估计，转换成了矩阵分解问题**

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063319663.png" alt="image-20210726063319663" style="zoom:100%;" />

<img src="https://github.com/archted/markdown-img/blob/main/img/image-20210726063324878.png?raw=true" alt="image-20210726063324878" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063358395.png" alt="image-20210726063358395" style="zoom:100%;" />

​                                       <img src="https://github.com/archted/markdown-img/blob/main/img/image-20210726063324878.png?raw=true" alt="image-20210726063324878" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063423560.png" alt="image-20210726063423560" style="zoom:100%;" />

直接计算，复杂度为$O(k∗n^2)$

其中，n是特征个数，k是特征的embedding size（即隐分类的个数）

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726063512080.png" alt="image-20210726063512080" style="zoom:100%;" />

[1w * 3] * [3 * 2w] = [1w * 2w] 其中k=3表示隐分类个数，相当于将 1w个用户聚类成3类，2w个item也聚类为3类。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726064135562.png" alt="image-20210726064135562" style="zoom:100%;" />

<img src="https://github.com/archted/markdown-img/blob/main/img/image-20210726063324878.png?raw=true" alt="image-20210726063324878" style="zoom:50%;" />

​               下矩阵=全矩阵-对角矩阵

其中，i=1,...n;j=i+1,...n 因为特征是两两之间的组合，而不是排列

通过公式变换，复杂度由$O(k∗n^2)=>O(k∗n)$

综上所述，FM的目标函数为：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726064907687.png" alt="image-20210726064907687" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726064911832.png" alt="image-20210726064911832" style="zoom:100%;" />

其中，<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065317507.png" alt="image-20210726065317507" style="zoom:100%;" />是FM的核心思想，使得稀疏数据下学习不充分的问题也能得到充分解决

优化方法：

1.ALS，交替最小二乘法

2.SGD，随机梯度下降法

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065037816.png" alt="image-20210726065037816" style="zoom:100%;" />

可以提前计算（与i无关），因此梯度的计算复杂度为O(1)，参数更新的计算复杂度为O(k*n)

3.MCMC，马尔科夫链蒙特卡罗法（马尔科夫链：状态推导，蒙特卡罗：随机搜索，类似于SGD）

* D-阶FM算法：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065226042.png" alt="image-20210726065226042" style="zoom:100%;" />

因为计算量大，一般FM采用2阶特征组合的方式

实际上高阶/非线性的特征组合适合采用深度模型

* FM模型的损失函数

回归问题，y'(x)直接作为预测值，损失函数可采用least square error

二分类问题，FM的输出还需要经过 sigmoid 函数变换，也就是将y'(x)转化为二分类标签，即0,1

### FM与MF的区别

FM优点：

1.FM 有多个特征，所以更加泛化。MF只利用了User和Item两维特征，是FM的特例。

2.因为FM特征采用oneHot编码，因此可以解决稀疏的问题

### FFM算法

FM是FFM的特例(fileds=1),MF是FM的特例(只有user和item两个特征)

Field-aware Factorization Machines for CTR Prediction

https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071033053.png" alt="image-20210726071033053" style="zoom:100%;" />

通过引入field的概念，FFM把相同性质的特征归于同一个field，比如“Day=26/11/15”、“Day=1/7/14”、“Day=19/2/15”这三个特征代表日期，放到同一个field中

当“Day=26/11/15”与Country特征，Ad_type特征进行特征组合时，使用不同的隐向量（Field-aware），这是因为Country特征和Ad_type特征，本身的field不同

### FM与FFM的区别

1.计算公式

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071123896.png" alt="image-20210726071123896" style="zoom:100%;" />

对于FM算法：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071114543.png" alt="image-20210726071114543" style="zoom:100%;" />

**每个特征有唯一的一个隐向量表示**，这个隐向量被用来学习与其他任何特征之间的影响。

w(ESPN)用来学习与Nike的隐性影响w(ESPN) * w(Nike)，同时也用来学习与Male的影响w(ESPN)*w(Male)。但是Nike和Male属于不同的领域，它们的隐性影响（latent effects）应该是不一样的。

对于FFM算法：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071244114.png" alt="image-20210726071244114" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071341195.png" alt="image-20210726071341195" style="zoom:100%;" />

**每个特征会有几个不同的隐向量**，fj 是第 j 个特征所属的field。我的理解是评分矩阵有几个特征，一个特征就有几个隐向量，每个隐向量都表示该特征与其他特征向量乘积的系数，j2=j1+1表明特征之间是组合而不是排列。

FM算法：

每个特征只有一个隐向量

FM是FFM的特例

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071457165.png" alt="image-20210726071457165" style="zoom:100%;" />

FFM算法：

每个特征有多个隐向量

使用哪个，取决于和哪个向量进行点乘

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071516498.png" alt="image-20210726071516498" style="zoom:100%;" />

2.时间复杂度

FFM算法：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726071950219.png" alt="image-20210726071950219" style="zoom:100%;" />

隐向量的长度为 k，FFM的二次参数有 nfk 个，多于FM模型的 nk 个

由于隐向量与field相关，FFM二次项并不能够化简，计算复杂度是 $O(k∗n^2)$,FFM的k值一般远小于FM的k值

3.数据格式

特征格式：field_id:feat_id:value

field_id代表field编号，feat_id代表特征编号，value是特征值。

如果特征为数值型，只需分配单独的field编号，比如评分，item的历史CTR/CVR等。

如果特征为分类（categorical）特征，需要经过One-Hot编码成数值型，编码产生的所有特征同属于一个field。特征值是0或1，比如性别、商品的品类id等

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726072112554.png" alt="image-20210726072112554" style="zoom:100%;" />

FM： Yes  P-ESPN:1  A-Nike:1  G-Male:1

FFM：Yes  P:P-ESPN:1  A:A-Nike:1  G:G-Male:1

* 示例

Ge特征名下有两个值：Co，Dr

原本输入数据时只是输入特征的数值/onehot编码，即模型只知道输入了[[0,1],[1,0]]

现在相当于将特征名也输入进去,即输入[[2:0,2:1][2:1,2:0]],即告诉FFM模型这两个向量是属于同一个特征下的不同向量

FFM的输入格式：<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074439095.png" alt="image-20210726074439095" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074443218.png" alt="image-20210726074443218" style="zoom:100%;" />

类别特征编码成onehot形式；数值特征依旧为数值：<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074616659.png" alt="image-20210726074616659" style="zoom:100%;" />

FFM的特征组合($C_5^2 = 10$):

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074653501.png" alt="image-20210726074653501" style="zoom:100%;" />

### DeepFM算法

**1.总览**

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction，2017

https://arxiv.org/abs/1703.04247

FM可以做特征组合，但是计算量大，一般只考虑2阶特征组合

如何既考虑低阶（1阶+2阶），又能考虑到高阶特征 => **DeepFM=FM+DNN**

设计了一种end-to-end的模型结构 => 无须特征工程

在各种benchmark和工程中效果好

Criteo点击率预测, 4500万用户点击记录，90%样本用于训练，10%用于测试

Company*游戏中心，10亿记录，连续7天用户点击记录用于训练，之后1天用于测试

**2.网络结构**

DeepFM = FM + DNN：

提取低阶(low order)特征 => 因子分解机FM

既可以做1阶特征建模，也可以做2阶特征建模

提取高阶(high order)特征 => 神经网络DNN

end-to-end，共享特征输入

对于特征i，wi是1阶特征的权重，

**Vi表示该特征与其他特征的交互影响，输入到FM模型中可以获得特征的2阶特征表示，输入到DNN模型得到高阶特征。**

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075444909.png" alt="image-20210726075444909" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075508050.png" alt="image-20210726075508050" style="zoom:100%;" />

一)、FM模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075619761.png" alt="image-20210726075619761" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075610160.png" alt="image-20210726075610160" style="zoom:100%;" />

二)、Deep模型

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075713145.png" alt="image-20210726075713145" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075717389.png" alt="image-20210726075717389" style="zoom:100%;" />

设计子网络结构（从输入层=>嵌入层），将原始的稀疏表示特征映射为稠密的特征向量。

Input Layer => Embedding Layer

不同field特征长度不同，但是子网络输出的向量具有相同维度k

**利用FM模型的隐特征向量V作为网络权重初始化来获得子网络输出向量**，即<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075953033.png" alt="image-20210726075953033" style="zoom:100%;" />，参数更新：<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080010911.png" alt="image-20210726080010911" style="zoom:100%;" />

**3.什么是Embedding：**

一种降维方式，将不同特征转换为维度相同的向量

在推荐系统中，对于离线变量我们需要转换成one-hot => 维度非常高，可以将其转换为embedding向量

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080245956.png" alt="image-20210726080245956" style="zoom:100%;" />

原来每个Field i维度很高，都统一降成k维embedding向量

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080249888.png" alt="image-20210726080249888" style="zoom:100%;" />

方法：接入全连接层，对于每个Field只有一个位置为1，其余为0，因此得到的embedding就是图中连接的红线，对于Field 1来说就是<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080338920.png" alt="image-20210726080338920" style="zoom:100%;" />

FM模型和Deep模型中的子网络权重共享，也就是对于同一个特征，向量Vi是相同的

**4.embedding在模型中起到的作用**

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080729275.png" alt="image-20210726080729275" style="zoom:100%;" />

DeepFM中的模块：

1)Sparse Features，输入多个稀疏特征

2)Dense Embeddings

**对每个稀疏特征做embedding，学习到他们的embedding向量(维度相等，均为k），因为需要将这些embedding向量送到FM层做内积。同时embedding进行了降维，更好发挥Deep Layer的高阶特征学习能力**.MF和FM做的不就是embedding吗，将特征都映射到k维。

3)FM Layer

一阶特征：原始特征相加

二阶特征：原始特征embedding后的embedding向量两两内积

4)Deep Layer，将每个embedding向量做级联，然后做多层的全连接，学习更深的特征

5)Output Units，将FM层输出与Deep层输出进行级联，接一个dense层，作为最终输出结果.<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726080917260.png" alt="image-20210726080917260" style="zoom:100%;" />

**5.对比实验**

relu对于所有深度模型来说更适合（除了IPNN）

Dropout设置为0.6-0.9之间最适合

隐藏层3层比较适合



## 工具：libFM，libFFM，xlearn，DeepFM工具

### libFM

1.libFM数据格式

每一行都包含一个训练数据（x，y），首先规定y的值，然后是x的非零值。

对于二分类问题，y>0的类型被认为是正分类，y<=0被认为是负分类。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065625140.png" alt="image-20210726065625140" style="zoom:100%;" />

数据格式为 INDEX:VALUE

第一行的含义：$y=4,x_0=1.5,x_3=-7.9$

整个矩阵可以表示为：

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065740822.png" alt="image-20210726065740822" style="zoom:100%;" />

2.使用libFM自带的libsvm格式转换

triple_format_to_libfm.pl （perl文件）

-target 目标变量  target 2表示第3列

-delete_column 不需要的变量  

perl triple_format_to_libfm.pl -in ratings.dat -target 2 -delete_column 3 -separator "::"

自动将.dat文件 => .libfm文件

Userid::itemid::rate::timestamp    y是Rate 

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065829964.png" alt="image-20210726065829964" style="zoom:100%;" />    <img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726065833879.png" alt="image-20210726065833879" style="zoom:100%;" />

ratings.dat格式 => 目标格式

3.使用libFM训练FM模型

-train 指定训练集，libfm格式或者二进制文件

-test 指定测试集，libfm格式或者二进制文件

-task，说明任务类型classification还是regression

dim，指定k0，k1，k2，

-iter，迭代次数，默认100

-method，优化方式，可以使用SGD, SGDA, ALS, MCMC，默认为MCMC

-out，指定输出文件

libFM -task r -train ratings.dat.libfm -test ratings.dat.libfm -dim '1,1,8' -out out.txt

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726070143365.png" alt="image-20210726070143365" style="zoom:100%;" />

​                                        预测结果文件

4.使用libFM进行分类

Titanic数据集，train.csv和test.csv

Step1，对train.csv和test.csv进行处理

去掉列名，针对test.csv增加虚拟target列（值设置为1）

Step2，将train.csv, test.csv转换成libfm格式

perl triple_format_to_libfm.pl -in ./titanic/train.csv -target 1 -delete_column 0 -separator ","

perl triple_format_to_libfm.pl -in ./titanic/test.csv -target 1 -delete_column 0 -separator ","

Step3，使用libfm进行训练，输出结果文件 titanic_out.txt

libFM -task c -train ./titanic/train.csv.libfm -test ./titanic/test.csv.libfm -dim '1,1,8' -out titanic_out.txt

### libFFM

https://github.com/ycjuan/libffm

### xlearn

https://xlearn-doc-cn.readthedocs.io/en/latest/index.html

提供Python接口

支持LR，FM，FFM算法等

运行效率高，比libfm, libffm快

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074902851.png" alt="image-20210726074902851" style="zoom:100%;" />

criteo_ctr数据集

展示广告CTR预估比赛

欧洲大型重定向广告公司Criteo的互联网广告数据集（4000万训练样本，500万测试样本）

原始数据：https://labs.criteo.com/2013/12/download-terabyte-click-logs/

small_train.txt 和 small_test.txt文件

（FFM数据格式，200条记录）

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726074944854.png" alt="image-20210726074944854" style="zoom:100%;" />



```python
import xlearn as xl
# 创建FFM模型
ffm_model = xl.create_ffm()
# 设置训练集和测试集
ffm_model.setTrain("./small_train.txt")
ffm_model.setValidate("./small_test.txt")
# 设置参数，任务为二分类，学习率0.2，正则项lambda: 0.002，评估指标 accuracy
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}
# FFM训练，并输出模型
ffm_model.fit(param, './model.out')
# 设置测试集，将输出结果转换为0-1
ffm_model.setTest("./small_test.txt")
ffm_model.setSigmoid()
ffm_model.predict("./model.out", "./output.txt")

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075038744.png" alt="image-20210726075038744" style="zoom:100%;" />

xlearn工具输入数据格式：

LR ，FM 算法： CSV 或者 libsvm

FFM 算法：libffm 格式

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726075118661.png" alt="image-20210726075118661" style="zoom:100%;" />

### DeepFM工具

DeepCTR工具

https://github.com/shenweichen/DeepCTR



数据集：MovieLens_Sample

包括了多个特征：user_id, movie_id, rating, timestamp, title, genres, gender, age, occupation, zip

使用DeepFM，计算RMSE值

```python
data = pd.read_csv("movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['rating']
……
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}
# 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)

```

## 总结

FM算法的作用：

•泛化能力强，解决大规模稀疏数据下的特征组合问题

•MF是FM的特例，使用了特征embedding（User，Item）。FM使用了更多Side Information作为特征，同时在进行二阶特征组合权重预估的时候，使用到了MF

•计算复杂度，可以在线性时间对样本做出预测，通过公式变换将计算复杂度降到O(k*n)

DeepFM采用了FM+DNN的方式，在低阶和高阶特征组合上更接近真实世界，因此效果也更好