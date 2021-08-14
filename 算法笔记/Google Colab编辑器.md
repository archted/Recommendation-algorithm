## Google Colab编辑器

https://colab.research.google.com

Google的云端Jupyter编辑器，计算效率高

存储在Google 云端硬盘，可以共享

支持GPU/TPU加速（Tesla K80 or Tesla T4）

可以使用Keras、Tensorflow和Pytorch

不足：Colab最多12个小时环境就会重置，而且需要将数据放到Google Drive

1.Google Colab使用

上传文件到Google云盘

挂在Google云硬盘

from google.colab import drive

drive.mount('/content/drive')

更改运行目录

import os

os.chdir("/content/drive/My Drive/Colab Notebooks/")

可以选择GPU或TPU进行加速

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725152552612.png" alt="image-20210725152552612" style="zoom:100%;" />

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/")

time1 = time.time()
...
```

2.将数据集加载到Colab

Github数据集

!git clone https://github.com/JameyWoo/myDataSet.git

!pip install git+https://github.com/shenweichen/DeepCTR.git

Kaggle数据集

Step1，生成Kaggle API

Step2，找到自己的username和token

Step3，创建kaggle文件夹，配置kaggle.json

Step4，使用kaggle api下载数据

Step5，解压数据并使用

```python
# 安装kaggle
!pip install -U -q kaggle
# 创建kaggle.json
!mkdir -p ~/.kaggle
!echo '{"username":"cystanford","key":"7b91422ebb9b6eef133f27f95362456b"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
# 设置数据集文件夹
!mkdir -p credit_data
# 下载数据
!kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p credit_data
# 解压数据
!unzip  /content/credit_data/default-of-credit-card-clients-dataset.zip

```

## 