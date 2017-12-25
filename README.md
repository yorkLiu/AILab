# AI Lab

## 机器学习 e

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