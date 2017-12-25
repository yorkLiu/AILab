# AI Lab

## 机器学习

**1.决策树学习  主要参数**
- criterion: gini or entropy
- splitter: best or random 前者是在所有的特征中找最好的切分点，后者是在部分特征中（数据量大的时候）
- max_features: None (所有), log2, sqrt, N 特征小于50的时候一般全用所有的
- max_depth: 数据小或者特征少的时候可以不管这个值，如果模型样本量多，特征也多的情况下，可能 深度限制下
- min_samples_split: 如果某节点的样本数少于 min_samples_split,则不会继续再深度选择最优特征来进行划分，如果样本量不大，不需要管这个值，如果样本量数量级非常大，则推荐增大这个值
- min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝，如果样本量不大，不需要管这个值，大些如10W可以们测试下5.
- min_weight_fraction_leaf: 这个值限制了叶子节点所有样本权重各最小值，如果小于这个值，则会和兄弟节点一起被剪枝，默认是0,就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，著分类树样本的分布类别偏差很大，就会引入样本权重，这是我们就要注意这个值了。
- max_leaf_nodes: 通过限制最大叶了节点数，可以防止过拟合，默认是 "None", 即不限制最大的叶子节点数。
                如果加了限制，算法会建立在最大叶子节点数内最优决策树。
                如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到

- class_weight: 指定样本各类别的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重。
              如果全用 "balanced"，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。

- min_impurity_split: 这个值限制了决策树的增长，如果某些节点不纯度（基尼系数，信息增益，均方差，绝对差）小于这个阀值, 则该节点不再生成了节点。即为叶子节点。

**2. 贝叶斯算法**
- 贝叶斯要解决的问题：
    - 正向概率： 假设袋子里面的N个白球，M个黑球，你伸手进去摸一把，摸出黑球的概率是多少？
    - 向概率: 如果我们事先并不知道黑球与白球的比例，而是闭着眼睛摸出一个（或好几个）球，观察这些取出来的球的颜色之后，那我们可以就此对袋子中的黑白球的比例作出什么样的推测？

## scikit-Learn
简单高效的数据挖掘和数据分析工具
所有人都适用，可在不同的上下文中重用
基于NumPy、SciPy和matplotlib构建
开源、商业可用 - BSD许可
- [scikit-learn 中文手册Git Hub](https://github.com/lzjqsdd/scikit-learn-doc-cn)
- [scikit-learn 中文手册(official guide)](http://sklearn.lzjqsdd.com/])

## XGBoost
XGBoost是大规模并行boosted tree的工具，它是目前最快最好的开源boosted tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量kaggle选手选用它进行数据挖掘比赛，其中包括两个以上kaggle比赛的夺冠方案。在工业界规模方面，xgboost的分布式版本有广泛的可移植性，支持在YARN, MPI, Sungrid Engine等各个平台上面运行，并且保留了单机并行版本的各种优化，使得它可以很好地解决于工业界规模的问题。

- [XGBoost Slides](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
- [XGBoost中文版原理介绍](http://www.52cs.org/?p=429)
- [原始论文XGBoost: A Scalable Tree Boosting System](http://arxiv.org/pdf/1603.02754v1.pdf)
- [XGBoost Parameters (official guide)](http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)
- **精彩博文：** 
    - [XGBoost浅入浅出——wepon](http://wepon.me/2016/05/07/XGBoost%E6%B5%85%E5%85%A5%E6%B5%85%E5%87%BA/) 
    - [XGBoost: 速度快效果好的boosting模型](https://cosx.org/2015/03/xgboost) 
    - [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
    - [XGBoost Plotting API以及GBDT组合特征实践](http://blog.csdn.net/sb19931201/article/details/65445514)
    
- [请戳我 看如何安装 XGBoost](https://xgboost.readthedocs.io/en/latest/build.html)
    
- **XGBoost 常用参数详解**
    - [官方参数戳这里](http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)
    
    - **1. General Parameters（常规参数** 
        - **1.booster [default=gbtree]**：选择基分类器，gbtree: tree-based models/gblinear: linear models 
        - **2.silent [default=0]**:设置成1则没有运行信息输出，最好是设置为0. 
        - **3.nthread [default to maximum number of threads available if not set]**：线程数
    - **2. Booster Parameters（模型参数** 
        - 1. **eta [default=0.3]** :shrinkage参数，用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛。把学习率 eta 设置的小一些，小学习率可以使得后面的学习更加仔细。 
        - 2. **min_child_weight [default=1]**:这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        - 3.**max_depth [default=6]**: 每颗树的最大深度，树高越深，越容易过拟合。 
        - 4.**max_leaf_nodes**:最大叶结点数，与max_depth作用有点重合。 
        - 5.**gamma [default=0]**：后剪枝时，用于控制是否后剪枝的参数。 
        - 6.**max_delta_step [default=0]**：这个参数在更新步骤中起作用，如果取0表示没有约束，如果取正值则使得更新步骤更加保守。可以防止做太大的更新步子，使更新更加平缓。 
        - 7.**subsample [default=1]**：样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合。 
        - 8.**colsample_bytree [default=1]**：列采样，对每棵树的生成用的特征进行列采样.一般设置为： 0.5-1 
        - 9.**lambda [default=1]**：控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。 
        - 10.**alpha [default=0]**:控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。 
        - 11.**scale_pos_weight [default=1]**：如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
                      
    - **3. Learning Task Parameters（学习任务参数）**
        - 1.**objective [default=reg:linear]**：定义最小化损失函数类型，常用参数： 
            - **binary:logistic** –logistic regression for binary classification, returns predicted probability (not class) 
            - **ulti:softmax** –multiclass classification using the softmax objective, returns predicted class (not probabilities) 
        you also need to set an additional num_class (number of classes) parameter defining the number of unique classes 
            - **multi:softprob** –same as softmax, but returns predicted probability of each data point belonging to each class. 
        - 2.**eval_metric [ default according to objective ]**： 
        The metric to be used for validation data. 
        The default values are rmse for regression and error for classification. 
        Typical values are: 
            - **rmse** – root mean square error 
            - **mae** – mean absolute error 
            - **logloss** – negative log-likelihood 
            - **error** – Binary classification error rate (0.5 threshold) 
            - **merror** – Multiclass classification error rate 
            - **mlogloss** – Multiclass logloss 
            - **auc**: Area under the curve 
        - 3.**seed [default=0]**： 
        The random number seed. 随机种子，用于产生可复现的结果 
        Can be used for generating reproducible results and also for parameter tuning.
    - **注意**: python sklearn style参数名会有所变化 
        - eta –> learning_rate 
        - lambda –> reg_lambda 
        - alpha –> reg_alpha
                                      
```
    pip install xgboost
    # -*- coding: UTF-8 -*- 
    from numpy import loadtxt
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from xgboost.sklearn import XGBClassifier
    
    # load data
    dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    seed = 7
    
    test_size=0.33
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    model = XGBClassifier()
    eval_set = [(x_test, y_test)]
    # early_stopping_rounds: 如果连续N 次结果没有提升,则停止
    # eval_metric: 损失函数
    # eval_set: A list of (X, y) pairs to use as a validation set for early-stopping
    # verbose: print 学习结果
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    
    # make predictions for test data
    y_pred = model.predict(x_test)
    predictions =[round(value) for value in y_pred]
    
    accuracy = metrics.accuracy_score(y_test, predictions)
    
    print "Accuracy: %.2f %%" % (accuracy*100)
    
    -----------------------------
    [OutPut]
    [0]	validation_0-logloss:0.660186
    Will train until validation_0-logloss hasn't improved in 10 rounds.
    [1]	validation_0-logloss:0.634854
    [2]	validation_0-logloss:0.612239
    [3]	validation_0-logloss:0.593118
    [4]	validation_0-logloss:0.578303
    [5]	validation_0-logloss:0.564942
    [6]	validation_0-logloss:0.555113
    [7]	validation_0-logloss:0.54499
    [8]	validation_0-logloss:0.539151
    [9]	validation_0-logloss:0.531819
    [10]	validation_0-logloss:0.526065
    [11]	validation_0-logloss:0.51977
    [12]	validation_0-logloss:0.514979
    [13]	validation_0-logloss:0.50927
    [14]	validation_0-logloss:0.506086
    [15]	validation_0-logloss:0.503565
    [16]	validation_0-logloss:0.503591
    [17]	validation_0-logloss:0.500805
    [18]	validation_0-logloss:0.497605
    [19]	validation_0-logloss:0.495328
    [20]	validation_0-logloss:0.494777
    [21]	validation_0-logloss:0.494274
    [22]	validation_0-logloss:0.493333
    [23]	validation_0-logloss:0.492211
    [24]	validation_0-logloss:0.491936
    [25]	validation_0-logloss:0.490578
    [26]	validation_0-logloss:0.490895
    [27]	validation_0-logloss:0.490646
    [28]	validation_0-logloss:0.491911
    [29]	validation_0-logloss:0.491407
    [30]	validation_0-logloss:0.488828
    [31]	validation_0-logloss:0.487867
    [32]	validation_0-logloss:0.487297
    [33]	validation_0-logloss:0.487562
    [34]	validation_0-logloss:0.487788
    [35]	validation_0-logloss:0.487962
    [36]	validation_0-logloss:0.488218
    [37]	validation_0-logloss:0.489582
    [38]	validation_0-logloss:0.489334
    [39]	validation_0-logloss:0.490969
    [40]	validation_0-logloss:0.48978
    [41]	validation_0-logloss:0.490704
    [42]	validation_0-logloss:0.492369
    Stopping. Best iteration:
    [32]	validation_0-logloss:0.487297

    Accuracy: 78.35 %
    
```


## TensorFlow

## Numpy
  [Click Here to get more numpy examples](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
  ```
    pip install numpy
    
    import numpy as np
    
    print np.arange(10)
    print np.arange(10).reshape(3,4)
    ...
  ```
  
  ## Pandas
  [Pandas](http://pandas.pydata.org/pandas-docs/stable/) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
  
  [Click Here to get more Pandas Guide](http://pandas.pydata.org/pandas-docs/stable/10min.html#min)
  
  ```
  pip install pandas
  
  import pandas as pd
  
  titanic_survival = pd.read_csv('titanic_train.csv')
  
  titanic_survival.loc([0:5])
  
  titanic_survival.head()
  
  titanic_survival.tail()
  
  pd.pivot_table(titanic_survival, values=['Fare'], index='Pclass')
  
  ...
  
  ```
  
  ## Seaborn 画图
  - [Click Here to get more seaborn](https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles)
  - [点这里查看更多关于 Seaborn 的例子](http://blog.csdn.net/u013082989/article/details/73278458)
  ```
  pip install seaborn
  
  
  import numpy as np
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  sns.set_style("whitegrid")
  data = np.random.random(size=(20, 6)) + np.arange(6)/2
  sns.boxplot(data=data)
  sns.plt.show()
  
  ```