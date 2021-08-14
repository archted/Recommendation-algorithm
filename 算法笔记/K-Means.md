## K-Means

### Project Review：给18支亚洲球队进行聚类

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725135536085.png" alt="image-20210725135536085" style="zoom:100%;" />

Thinking：聚类的K取多少适合？

K-Means 手肘法

计算inertia簇内误差平方和`model.inertia_`

```python
# 统计不同K取值的误差平方和
sse = []
for k in range(1, 11):
	# kmeans算法
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	# 计算inertia簇内误差平方和
	sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725135626117.png" alt="image-20210725135626117" style="zoom:100%;" />

### K-Means 在图像分割中的应用

```python
# 用K-Means对图像进行2聚类
kmeans =KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)

```

<img src="assets\image-20210725140213454.png" alt="image-20210725140213454" style="zoom:50%;" />

K-Means图像分割的不足：

按照图像的灰度值定义距离，对图像的文理，内容缺乏理解