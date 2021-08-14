## GridSearchCV

GridSearchCV()参数：

| estimator  | 想要采用的分类器，比如随机森林、决策树、SVM、KNN等           |
| ---------- | ------------------------------------------------------------ |
| param_grid | 想要优化的参数及取值，输入的是字典或者列表的形式             |
| cv         | 交叉验证的折数，默认为None，代表使用三折交叉验证。  也可以为整数，代表的是交叉验证的折数 |
| scoring    | 准确度的评价标准，默认为None，也就是需要使用score函数。  可以设置具体的评价标准，比如accuray，f1等 |

Pipeline使用：
使用Pipeline管道机制进行流水线作业。因为在做机器学习之前，还需要一些准备过程，比如数据规范化，或者数据降维等。
GridSearchCV + Pipeline：
对于每一个模型，都进行Pipeline流水作业，找到最适合的参数

```python
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')
```

示例：


```python
# 构造各种分类器
classifiers = [
    SVC(random_state = 1, kernel = 'rbf'),    
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
    RandomForestClassifier(random_state = 1, criterion = 'gini'),
    KNeighborsClassifier(metric = 'minkowski'),
]
# 分类器名称
classifier_names = [
            'svc', 
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
]
# 分类器参数
classifier_param_grid = [
            {'svc__C':[1], 'svc__gamma':[0.01]},
            {'decisiontreeclassifier__max_depth':[6,9,11]},
            {'randomforestclassifier__n_estimators':[3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,8]},
]
# 对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" %search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" %accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response
 # 调用GridSearchCV_work，寻找最优分类器及参数
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')



```

<img src="https://cdn.jsdelivr.net/gh/archted/markdown-img@main/img/image-20210725153359949.png" alt="image-20210725153359949" style="zoom:100%;" />
