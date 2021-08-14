# lesson04

## Thinking&Action

Thinking1：ALS都有哪些应用场景

Thinking2：ALS进行矩阵分解的时候，为什么可以并行化处理

Thinking3：梯度下降法中的批量梯度下降（BGD），随机梯度下降（SGD），和小批量梯度下降有什么区别（MBGD）

Thinking4：对数据进行可视化EDA都有哪些方式，你都是用过哪些工具？

Thinking5：你阅读过和推荐系统/计算广告/预测相关的论文么？有哪些论文是你比较推荐的，可以分享到微信群中



Action1：对MovieLens数据集进行评分预测（2周）

数据集：https://github.com/cypredict/predict/tree/master/L4/MovieLens

工具：可以使用Surprise或者其他

Action2：Paper Reading（1周）

Action3：使用WordCloud对新闻进行词云分析

## 简历Updating

银行产品购买预测：采用Item-based CF方法，对Santandery银行的用户产品购买数据进行分析，并对未来可能购买的产品进行预测

https://github.com/xxx/Santandery



电影推荐算法：基于矩阵分解的协同过滤算法（ALS，SVD，SVD++，FunkSVD） 给Netflix网站进行推荐算法，RMSE降低到0.9111

https://github.com/xxx/netflix



CTR广告点击率预测：采用基于神经网络的DeepFM算法，对DSP公司Avazu的网站的广告转化率进行预测，项目中使用了线性模型及非线性模型，并进行了对比分析

https://github.com/xxx/avazu-ctr-prediction



房屋价格走势预测引擎：通过时间序列算法，分析北京、上海、广州过去4年（2015.8-2019.12）的房屋历史价格，预测未来6个月（2020.1-2020.6）不同区的价格走势 https://github.com/xxx/house-price-prediction

邮件数据分析：通过PageRank算法分析邮件中的人物关系图谱，并针对邮件数量较大的情况筛选出重要的人物，进行绘制：https://github.com/xxx/PageRank



电影数据集关联规则挖掘：采用Apriori算法，分析电影数据集中的导演和演员信息，从而发现导演和演员之间的频繁项集及关联规则：https://github.com/xxx/Apriori



信用卡违约率分析：针对台湾某银行信用卡的数据，构建一个分析信用卡违约率的分类器。采用Random Forest算法，信用卡违约率识别率在80%左右：https://github.com/xxx/credit_default

 

信用卡欺诈分析：针对欧洲某银行信用卡交易数据，构建一个信用卡交易欺诈识别器。采用逻辑回归算法，通过数据可视化方式对混淆矩阵进行展示，统计模型的精确率，召回率和F1值，F1值为0.712，并绘制了精确率和召回率的曲线关系：https://github.com/xxx/credit_fraud

 

比特币走势分析：分析2012年1月1日到2018年10月31日的比特币价格数据，并采用时间序列方法，构建自回归滑动平均模型（ARMA模型），预测未来8个月比特币的价格走势。预测结果表明比特币将在8个月内降低到4000美金左右，与实际比特币价格趋势吻合（实际最低降到4000美金以下）：https://github.com/xxx/bitcoin

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724172258642.png" alt="image-20210724172258642" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724172301990.png" alt="image-20210724172301990" style="zoom:100%;" />



# lesson05

## Thinking&Action

Thinking1：奇异值分解SVD的原理是怎样的，都有哪些应用场景

Thinking2：funkSVD, BiasSVD，SVD++算法之间的区别是怎样的

Thinking3：矩阵分解算法在推荐系统中有哪些应用场景，存在哪些不足

Thinking4：item流行度在推荐系统中有怎样的应用

Thinking5：推荐系统的召回阶段都有哪些策略



Action1：选择任意一张图片，对其进行灰度化，然后使用SVD进行图像的重构，当奇异值数量为原有的1%，10%，50%时，输出重构后的图像

Action2：使用Google Colab编辑器，对MovieLens数据集进行评分预测，计算RMSE

（使用funkSVD, BiasSVD，SVD++）

Action3：使用PowerBI（或其他工具），对nCoV数据进行可视化呈现

## 工作中的应用

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725165128013-1627889824661.png" alt="image-20210725165128013" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725165138553.png" alt="image-20210725165138553" style="zoom:100%;" />

# lesson06

## Thinking&Action

Thinking1：在推荐系统中，FM和MF哪个应用的更多，为什么

A:FM应用更广泛，MF只是FM的一个特例

Thinking2：FFM与FM有哪些区别？

A:FFM加了filed的概念

Thinking3：DeepFM相比于FM解决了哪些问题，原理是怎样的

A:DeepFM=DNN+FM，解决高维非线性的特征组合问题，FM解决一阶和二阶的特征组合问题。

Thinking4：假设一个小说网站，有N部小说，每部小说都有摘要描述。如何针对该网站制定基于内容的推荐系统，即用户看了某部小说后，推荐其他相关的小说。原理和步骤是怎样的

Thinking5：Word2Vec的应用场景有哪些



Action1：使用libfm工具对movielens进行评分预测，采用SGD优化算法

Action2：使用DeepFM对movielens进行评分预测

Tricks：使用DeepCtr工具箱

Action3：使用Gensim中的Word2Vec对三国演义进行Word Embedding，分析和曹操最相近的词有哪些，曹操+刘备-张飞=?

## 工作中的应用

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726111751294.png" alt="image-20210726111751294" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210726111810049.png" alt="image-20210726111810049" style="zoom:100%;" />

# lesson07

## Thinking&Action

Thinking1：在CTR点击率预估中，使用GBDT+LR的原理是什么？

Thinking2：Wide & Deep的模型结构是怎样的，为什么能同时具备记忆和泛化能力（memorization and generalization）

Thinking3：在CTR预估中，使用FM与DNN结合的方式，有哪些结合的方式，代表模型有哪些？

Thinking4：Surprise工具中的baseline算法原理是怎样的？BaselineOnly和KNNBaseline有什么区别？

Thinking5：GBDT和随机森林都是基于树的算法，它们有什么区别？

Thinking6：基于邻域的协同过滤都有哪些算法，请简述原理





Action1：使用Wide&Deep模型对movielens进行评分预测

Action2：使用基于邻域的协同过滤（KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline中的任意一种）对MovieLens数据集进行协同过滤，采用k折交叉验证(k=3)，输出每次计算的RMSE, MAE



使用信用卡欺诈分析数据集 来演示机器学习中的可解释性

## 工作中的应用

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724105936000.png" alt="image-20210724105936000" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724105940078.png" alt="image-20210724105940078" style="zoom:100%;" />

# lesson08

## Thingk & Action

Thinking1：什么是近似最近邻查找，常用的方法有哪些

Thinking2：为什么两个集合的minhash值相同的概率等于这两个集合的Jaccard相似度

Thinking3：SimHash在计算文档相似度的作用是怎样的？

Thinking4：为什么YouTube采用期望观看时间作为评估指标

Thinking5：为什么YouTube在排序阶段没有采用经典的LR（逻辑回归）当作输出层，而是采用了Weighted Logistic Regression？



Action1：使用MinHashLSHForest对微博新闻句子进行检索 weibo.txt

针对某句话进行Query，查找Top-3相似的句子

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210723071554240.png" alt="image-20210723071554240" style="zoom:100%;" />





Action2：请设计一个基于DNN模型的推荐系统

•阐述两阶段的架构（召回、排序）

•以及每个阶段的DNN模型设计：

DNN输入层（如何进行特征选择）

DNN隐藏层结构

DNN输出层