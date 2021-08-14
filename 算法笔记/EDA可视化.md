## EDA可视化

### 可视化视图都有哪些

4种类别：

比较：展示事物的排列顺序，比如条图。

联系：查看两个变量之间关系，比如气泡图。

构成：每个部分所占整体的百分比，如饼图。

分布：关心各数值范围包含多少项目，如柱图。

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724164628611.png" alt="image-20210724164628611" style="zoom:100%;" />

### 工具：Matplotlib、Seaborn

import matplotlib.pyplot as plt

import seaborn as sns

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170133437.png" alt="image-20210724170133437" style="zoom:100%;" />

### 9种可视化视图

1. 散点图

```python
散点图：
# 数据准备
N = 500
x = np.random.randn(N)
y = np.random.randn(N)
# 用Matplotlib画散点图
plt.scatter(x, y,marker='x')
plt.show()
# 用Seaborn画散点图
df = pd.DataFrame({'x': x, 'y': y})
sns.jointplot(x="x", y="y", data=df, kind='scatter');
plt.show()
```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170235289.png" alt="image-20210724170235289" style="zoom:100%;" /><img src="assets\image-20210724170239426.png" alt="image-20210724170239426" style="zoom:80%;" />

2.折线图

```python
# 数据准备
x = [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910]
y = [265, 323, 136, 220, 305, 350, 419, 450, 560, 720, 830]
# 使用Matplotlib画折线图
plt.plot(x, y)
plt.show()
# 使用Seaborn画折线图
df = pd.DataFrame({'x': x, 'y': y})
sns.lineplot(x="x", y="y", data=df)
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170308721.png" alt="image-20210724170308721" style="zoom:100%;" />

3.条形图

```python
# 数据准备
x = ['c1', 'c2', 'c3', 'c4']
y = [15, 18, 5, 26]
# 用Matplotlib画条形图
plt.bar(x, y)
plt.show()
# 用Seaborn画条形图
sns.barplot(x, y)
plt.show()

```

<img src="assets\image-20210724170350128.png" alt="image-20210724170350128" style="zoom:67%;" /><img src="assets\image-20210724170353129.png" alt="image-20210724170353129" style="zoom:67%;" />

4.箱线图

```python
# 数据准备
# 生成0-1之间的20*4维度数据
data=np.random.normal(size=(10,4)) 
lables = ['A','B','C','D']
# 用Matplotlib画箱线图
plt.boxplot(data,labels=lables)
plt.show()
# 用Seaborn画箱线图
df = pd.DataFrame(data, columns=lables)
sns.boxplot(data=df)
plt.show()

```

<img src="assets\image-20210724170431983.png" alt="image-20210724170431983" style="zoom:67%;" /><img src="assets\image-20210724170435604.png" alt="image-20210724170435604" style="zoom:67%;" />

5.饼图

```python
# 数据准备
nums = [25, 33, 37]
# 射手adc：法师apc：坦克tk
labels = ['ADC','APC', 'Tk']
# 用Matplotlib画饼图
plt.pie(x = nums, labels=labels)
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170520858.png" alt="image-20210724170520858" style="zoom:100%;" />

6.热力图

```python
# 数据准备
np.random.seed(33)
data = np.random.rand(3, 3)
heatmap = sns.heatmap(data)
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170548991.png" alt="image-20210724170548991" style="zoom:100%;" />

7.蜘蛛图

```python
# 数据准备
labels=np.array([u"推进","KDA",u"生存",u"团战",u"发育",u"输出"])
stats=[76, 58, 67, 97, 86, 58]
# 画图数据准备，角度、状态值
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))
# 用Matplotlib画蜘蛛图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)   
ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)
# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
ax.set_thetagrids(angles * 180/np.pi, labels, FontProperties=font)
plt.show()
```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170617203.png" alt="image-20210724170617203" style="zoom:100%;" />

8.二元变量分布

```python
# 数据准备
flights = sns.load_dataset("flights")
# 用Seaborn画二元变量分布图（散点图，核密度图，Hexbin图）
sns.jointplot(x="year", y="passengers", data=flights, kind='scatter')
sns.jointplot(x="year", y="passengers", data=flights, kind='kde')
sns.jointplot(x="year", y="passengers", data=flights, kind='hex')
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170644113.png" alt="image-20210724170644113" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170649047.png" alt="image-20210724170649047" style="zoom:100%;" />

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170651754.png" alt="image-20210724170651754" style="zoom:100%;" />

9.成对关系

```python
# 数据准备
flights = sns.load_dataset('flights')
# 用Seaborn画成对关系
sns.pairplot(flights)
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170729918.png" alt="image-20210724170729918" style="zoom:100%;" />

10.显示特征之间的相关系数

```python
# 
plt.figure(figsize=(10, 10))
plt.title('Pearson Correlation between Features',y=1.05,size=15)
train_data_hot_encoded = train_features.drop('Embarked',1).join(train_features.Embarked.str.get_dummies())
train_data_hot_encoded = train_data_hot_encoded.drop('Sex',1).join(train_data_hot_encoded.Sex.str.get_dummies())
# 计算特征之间的Pearson系数，即相似度
sns.heatmap(train_data_hot_encoded.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white',annot=True)
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170817509.png" alt="image-20210724170817509" style="zoom:100%;" />

11.使用饼图来进行Survived取值的可视化

```python
train_data["Survived"].value_counts().plot(kind = "pie", label='Survived')
plt.show()

# 不同的Pclass,幸存人数(条形图)
sns.barplot(x = 'Pclass', y = "Survived", data = train_data);
plt.show()

# 不同的Embarked,幸存人数(条形图)
sns.barplot(x = 'Embarked', y = "Survived", data = train_data);
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170953039.png" alt="image-20210724170953039" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170956387.png" alt="image-20210724170956387" style="zoom:100%;" /><img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724170959528.png" alt="image-20210724170959528" style="zoom:100%;" />

12.树模型的特征重要性可视化

```python
def train(train_features, train_labels):
	# 构造CART决策树
	clf = DecisionTreeClassifier()
	clf.fit(train_features, train_labels)
	# 显示特征向量的重要程度
	coeffs = clf.feature_importances_
	df_co = pd.DataFrame(coeffs, columns=["importance_"])
	# 下标设置为Feature Name
	df_co.index = train_features.columns
	df_co.sort_values("importance_", ascending=True, inplace=True)
	df_co.importance_.plot(kind="barh")
	plt.title("Feature importance")
	plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724171025177.png" alt="image-20210724171025177" style="zoom:100%;" />

### 决策树可视化

通过pydot+GraphViz实现决策树可视化

安装Graphviz库需要下面的几步：

安装graphviz工具：http://www.graphviz.org/download/

将Graphviz添加到环境变量PATH中

需要Graphviz库，使用pip install graphviz进行安装

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210724171315941.png" alt="image-20210724171315941" style="zoom:100%;" />

```python
# 决策树可视化
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
def show_tree(clf):
	dot_data = StringIO()
	export_graphviz(clf, out_file=dot_data)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf("titanic_tree.pdf")
show_tree(clf)
```



### Project：对news进行词云展示

常用的分词工具：

英文：NLTK

中文：jieba

```python
from wordcloud import WordCloud
def create_word_cloud(f):
	f = remove_stop_words(f)
	cut_text = jieba.cut(f)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		font_path="simhei.ttf",
		max_words=100,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	wordcloud.to_file("wordcloud.jpg")

```

<img src="assets\image-20210724171220568.png" alt="image-20210724171220568" style="zoom:50%;" />