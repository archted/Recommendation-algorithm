## YouTube推荐系统

### 参考资料

[经典！YouTube推荐系统-年度最佳Paper(附实践代码)](https://zhuanlan.zhihu.com/p/138777815)

基本上内容和PPT一致。

### 总结

召回阶段完成快速筛选（几百万=>几百个），排序阶段完成精排（几百个=>十几个）

基于DNN模型完成召回，排序阶段，自动学习item的embedding特征

DNN的任务是基于用户信息和上下文环境，来学习用户的embedding向量，模拟矩阵分解的过程，DNN最后一层的输出近似作为用户的特征

特征embedding：

1. 将用户观看过的视频id列表做embedding，取embedding向量的平均值，作为watch vector

2. 把用户搜索过的视频id列表做embedding，取embedding向量的平均值，作为search vector

3. 用户的人口统计学属性做embedding，作为geographic embedding

4. 一些非多值类的特征如性别，还有数值类特征直接做DNN的输入

5. 一些数值类特征，对其进行变换。如对example age进行平方，平方根操作，作为新的特征。

把推荐问题转换成多分类问题，采用Negative Sampling提升模型效果（随机从全量样本中抽取用户没有点击过的item作为label=0，因为推荐列表页中展示的item是算法模型计算出来的 => 用户最有可能会点击的item）

在召回阶段，采用的近似最近邻查找 => 提升效率

Youtube的用户对新视频有偏好，引入Example Age（视频上传时间特征） => 与经验分布更Match

不对称的共同浏览问题，采用predicting next watch的方式，利用上文信息，预估下一次浏览的视频 => 从用户的历史视频观看记录中随机拿出来一个作为正样本，然后只用这个视频之前的历史观看记录作为输入

对每个用户提取等数量的训练样本 => 防止一部分非常活跃的用户主导损失函数值

针对某些特征，比如#previous impressions，进行平方和平方根处理，引入3个特征对DNN进行输入 => 简单有效的特征工程，引入了特征的非线性

在优化目标上，没有采用经典的CTR，或者Play Rate，而是采用了每次曝光预期播放时间作为优化目标

### 推荐系统的架构（基于DNN）

Deep Neural Networks for YouTube Recommendations

https://dl.acm.org/citation.cfm?doid=2959100.2959190

推荐系统分为召回（候选集生成）和排序两个阶段，召回阶段完成快速筛选（几百万=>几百个），排序阶段完成精排（几百个=>十几个）。

召回阶段通过“粗糙”的方式召回候选item；排序阶段，采用更精细的特征计算user-item之间的排序score，作为最终输出推荐结果的依据。

<img src="assets\image-20210721093655081.png" alt="image-20210721093655081" style="zoom:80%;" />

采用经典的两阶段法：

召回阶段 => deep candidate generation model 

排序阶段 => deep ranking model

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721094015175.png" alt="image-20210721094015175" style="zoom:100%;" />

<center>召回阶段</center>

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721094020827.png" alt="image-20210721094020827" style="zoom:100%;" />

<center>排序阶段</center>

### 召回阶段的DNN模型

1. 召回阶段

把推荐问题看成一个“超大规模多分类”问题

即在时刻t，为用户U（上下文信息C）在视频库V中精准的预测出视频i的类别（每个具体的视频视为一个类别，i即为一个类别）:

<img src="assets\image-20210721094514046.png" alt="image-20210721094514046" style="zoom: 80%;" />

其中，U为<用户，上下文>的高维embedding向量，vj每个候选视频的embedding向量

DNN的任务就是在用户信息，上下文信息为输入条件的情况下，学习用户的embedding向量u，通过一个softmax分类器，u能够有效的从视频语料库中识别视频的类别（也就是推荐的结果）

由于数据稀疏性问题，训练数据的正负样本选取采用用户隐式反馈数据，即完成了观看的事件作为样本。比如训练数据为(u1,c1,5) => 用户u1在上下文c1的情况下观看了类别为5的视频。

2. DNN模型

模型架构中间是三个隐层的DNN结构

输出分线上和离线训练两个部分。

将用户观看历史和搜索历史通过embedding的方式映射成为一个稠密的向量，同时用户场景信息以及用户画像信息（比如年龄，性别等离散特征）也被归一化到[0,1]作为DNN的输入

​										ANN，采用LSH算法

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721094936605.png" alt="image-20210721094936605" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721094904766.png" alt="image-20210721094904766" style="zoom:100%;" />





#### 主要特征处理

1.embedded video watches => watch vector，用户的历史观看是一个稀疏的，变长的**视频id序列**，采用类似于word2vec的做法，每个视频都会被embedding到固定维度的向量中。**最终通过加权平均（可根据重要性和时间进行加权）得到固定维度的watch vector**

2.embedded search tokens => Search vector，和watch vector生成方式类似。把历史搜索的query分词后的token的embedding向量进行加权平均，可以反映用户的整体的搜索历史情况

3.用户画像特征：如地理位置，设备，性别，年龄，登录状态等连续或离散特征都被归一化为[0,1]， 和watch vector以及search vector做拼接（concatenate）。

4.样本年龄(Example Age)

Example age特征表示视频被上传之后的时间。

用户更倾向于推荐尽管相关度不高但是新鲜fresh的视频。

推荐系统往往是利用用户过去的行为来预测未来，那么对于历史行为，推荐系统通常是能够学习到一种隐式的基准的。但是对于视频的流行度分布，往往是高度不稳定的。

将example age作为一个特征拼接到DNN的输入向量。训练时，时间窗口越靠后，该值越接近于0或者为一个小负数。加入了example age特征后，模型效果和观测到的实际数据更加逼近。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721100949728.png" alt="image-20210721100949728" style="zoom:100%;" />



#### 样本和上下文选择中的不对称的共同浏览问题

（asymmetric co-watch）

用户在浏览视频时候，往往都是序列式的，通常会先看一些比较流行的，然后才是观看一些小众的视频。剧集系列通常也是顺序地观看

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721101203194.png" alt="image-20210721101203194" style="zoom:100%;" />

图(a)是held-out方式，利用上下文信息预估中间的一个视频

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721101212944.png" alt="image-20210721101212944" style="zoom:100%;" />

图(b)是predicting next watch的方式，则是利用上文信息，预估下一次浏览的视频。

发现预测用户的下一个观看视频的效果要好得多，而不是预测随机推出的视频。

论文发现图(b)的方式在线上A/B test中表现更佳。而实际上，传统的协同过滤类的算法，都是隐含的采用图(a)的held-out方式，忽略了不对称的浏览模式。

方法：从用户的历史视频观看记录中随机拿出来一个作为正样本，然后只用这个视频之前的历史观看记录作为输入（图b）。

#### 负采样(Negative Sampling)

采用负采样，也就是随机从全量item中抽取用户没有点击过的item作为label=0的item。

在当次展现的情况下，虽然用户只点击了click的item，其他item没有点击，但是很多用户在后续浏览的时候未click的item也在其他非列表页的地方进行click，如果将该item标记为label=0，可能是误标记。论文中提到，推荐列表展示的item极有可能为热门item，虽然该item该用户未点击，但是我们不能降低热门item的权重（通过label=0的方式）。

#### 不同网络深度和特征的实验

1. 不同网络深度的试验

所有的视频和search token都embedded到256维的向量中，开始input层直接全连接到256维的softmax层，依次增加网络深度

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721101636573.png" alt="image-20210721101636573" style="zoom:100%;" />

随着网络深度加大，预测准确率在提升，但增加第4层之后，MAP（Mean Average Precision）已经变化不大了。

2. 不同特征的试验。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721101726906.png" alt="image-20210721101726906" style="zoom:100%;" />

从图中看到，增加了观看历史之外的特征，对预测准确率提升很明显。

### 排序阶段

用于精准的预估用户对视频的喜好程度。针对数百个item，需要更多feature来描述item，以及用户与视频（user-item）的关系。比如用户可能很喜欢某个视频，但如果推荐结果采用的“缩略图”选择不当，用户也许不会点击，等等。

Ranking阶段的模型和召回阶段的基本相似，不同在于Training最后一层是Weighted LR，Serving时激励函数使用的<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721102102757.png" alt="image-20210721102102757" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721102112636.png" alt="image-20210721102112636" style="zoom:100%;" />

相比召回阶段，引入了更多的feature（当前要计算的video的embedding，用户观看过的最后N个视频embedding的average，用户语言的embedding和当前视频语言的embedding，自上次观看同channel视频的时间，该视频已经被曝光给该用户的次数）

#### 对观看时间进行建模

CTR指标对于视频搜索具有一定的欺骗性，所以论文提出采用期望观看时间作为评估指标。

观看时长不是只有0，1两种标签，所以YouTube采用了Weighted Logistic Regression来模拟这个输出，相比于LR的计算公式，Weighted Logistic Regression多了正负样本权重的计算。

划分样本空间时，正样本为点击，输出值即阅读时长值；负样本为无点击视频，输出值则统一采用1，即采用单位权值，不进行加权。

在一般的逻辑回归中，LR模型学到的是几率odds，表示样本为正例概率与负例概率的比例

$odds=e^{Wx+b}=p/(1-p)$

引入观看时间时，LR学到的odds为

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210721102505469.png" alt="image-20210721102505469" style="zoom:100%;" />

，其中$p = \sum T_i/ALL$,$1-p = (N-k)/ALL$。

解释：正样本用观看时间赋予权值，负样本赋予单位权值（即不加权）.其中k为正样本个数，负样本个数就为$N-k$，因为负样本权值为1，因此负样本概率为$1-p = (N-k)/ALL$。

每个展示impression的观看时长的期望为

$E[T]=(∑T_i )/N$

$odds=E[T]×N/(N-k)=E[T]×1/(1-p)≈E[T]×(1+p)≈E[T]$,其中$p=k/N$

N代表样本总数，k代表正样本数量，Ti是第i正样本的观看时长。k比N小，因此公式中的odds可以转换成$E[T](1+P)$，其中P是点击率，点击率一般很小，所以**odds接近于$E[T]$，即期望观看时长**。因此在线上serving阶段，YouTube采用$e^{Wx+b}=e^{odds}$为激励函数 => 近似的估计期望的观看时长。

补充：

逻辑回归的输入是一个线性组合（与线性回归一样)，但输出是概率$ln⁡( odds)=ln⁡(p/(1-p))=ln⁡( p)-ln⁡( 1-p)$，即为logit函数的定义$logit⁡( p)=ln⁡(odds)$

$logit^{-1}⁡( z)=1/(1+e^{-z} )$其中$z=Wx+b$

$ln⁡(p/(1-p))=z=Wx+b$

$p=e^{Wx+b}/(1+e^{Wx+b} )=1/(1+e^{-(Wx+b)} )$

比如成功的概率p=0.8 ,那么失败的概率q = 1-0.8 = 0.2，成功的胜率为：odds(success) = p/q = 0.8/0.2 =4，失败的胜率为：odds(failure) = q/p = 0.2/0.8 =0.25。



#### 特征工程（分类特征、连续特征）

尽管DNN能够减轻人工特征工程的负担，但是依然需要花费精力将用户及视频数据转化为有效的特征（参考Facebook提出的GBDT+LR模型）

难点在于对用户行为序列建模，并关联视频打分机制

用户对于某Channel的历史行为很重要，比如浏览该频道的次数，最近一次浏览该频道距离现在的时间

把召回阶段的信息传播到Ranking阶段同样能提升效果，比如推荐来源和所在来源的分数

1. 排序阶段中的分类特征Embedding（Embedding Categorical Features）：

采用embedding的方式映射稀疏离散特征为密集向量，YouTube为每一个类别特征维度生成一个独立的embedding空间

对于相同域的特征可以共享embedding，好处在于加速迭代，降低内存开销

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092023112.png" alt="image-20210722092023112" style="zoom:100%;" />

2. 排序阶段中的连续特征归一化（Normalizing Continuous Features）：

神经网络对于输入数据的规模和分布非常敏感，而决策树模型（GBDT，RF）对于各个特征的缩放是不受什么影响

连续特征进行归一化对于收敛很重要

设计一种积分函数将特征映射为一个服从[0,1)分布的变量。一个符合f分布的特征x，等价转化成<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092130329.png" alt="image-20210722092130329" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092123064.png" alt="image-20210722092123064" style="zoom:100%;" />

除了输入归一化  ，还输入  <img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092130329.png" alt="image-20210722092130329" style="zoom:100%;" />的平方根和平方，特征的子线性和超线性，会让网络有更强的表达能力

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092148957.png" alt="image-20210722092148957" style="zoom:100%;" />

#### 隐藏层的实验

Hideden Layers实验：

在单个页面上，对展示给用户的正例和负例的这两个impression进行打分，如果对负例打分高于正例打分的话，那么我们认为对于正例预测的观看时间属于错误预测的观看时间

YouTube定义了模型评估指标weighted，per-user loss，即错误预测的观看时间占比总的观看时间的比例。

对每个用户的错误预测loss求和即可获得该用户的loss

实验证明，YouTube采用的Tower塔式模型效果最好，即第一层1024，第二层512，第三层256

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210722092347738.png" alt="image-20210722092347738" style="zoom:100%;" />

### 附：融合推荐模型

使用卷积神经网络（Convolutional Neural Networks）来学习视频名称的表示。下面会依次介绍文本卷积神经网络以及融合推荐模型。

- 文本卷积神经网络（CNN）
  卷积神经网络经常用来处理具有类似网格拓扑结构（grid-like topology）的数据。例如，图像可以视为二维网格的像素点，自然语言可以视为一维的词序列。卷积神经网络可以提取多种局部特征，并对其进行组合抽象得到更高级的特征表示。实验表明，卷积神经网络能高效地对图像及文本问题进行建模处理。
  卷积神经网络主要由卷积（convolution）和池化（pooling）操作构成，其应用及组合方式灵活多变，种类繁多。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/v2-f08f194a352f1fdb2ccd42a0692fefca_720w.jpg" alt="img" style="zoom:100%;" />卷积神经网络文本分类模型

- 融合推荐模型概览

在融合推荐模型的电影个性化推荐系统中：

（1）首先，使用用户特征和电影特征作为神经网络的输入，其中：

- - 用户特征融合了四个属性信息，分别是用户ID、性别、职业和年龄。
  - 电影特征融合了三个属性信息，分别是电影ID、电影类型ID和电影名称。

（2）对用户特征，将用户ID映射为维度大小为256的向量表示，输入全连接层，并对其他三个属性也做类似的处理。然后将四个属性的特征表示分别全连接并相加。

（3）对电影特征，将电影ID以类似用户ID的方式进行处理，电影类型ID以向量的形式直接输入全连接层，电影名称用文本卷积神经网络得到其定长向量表示。然后将三个属性的特征表示分别全连接并相加。

（4）得到用户和电影的向量表示后，计算二者的余弦相似度作为个性化推荐系统的打分。最后，用该相似度打分和用户真实打分的差异的平方作为该回归模型的损失函数。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/v2-f6ad7b24e6e521332a07507f4c34d493_720w.jpg" alt="img" style="zoom:100%;" />融合推荐模型
