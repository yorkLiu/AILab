# AI Lab

## 机器学习 

**决策树学习  主要参数**
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