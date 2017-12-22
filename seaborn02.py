import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# please visit: http://blog.csdn.net/longgb123/article/details/53228256

# x,y= np.random.multivariate_normal([0,0],[[1,-.5],[-.5,1]],size=300).T
#
# # pal = sns.light_palette('green', as_cmap=True)
# pal = sns.dark_palette('green', as_cmap=True)
# sns.kdeplot(x, y, cmap=pal)
# sns.plt.show()



mean, cov = [0,0], [(1,-0.5), (-0.5, 1)]
data = np.random.multivariate_normal(mean, cov, size=100)
df = pd.DataFrame(data, columns=['x', 'y'])

sns.jointplot(x='x', y='y', data=df)


data2 = np.random.multivariate_normal(mean, cov, size=1000)
df2 = pd.DataFrame(data, columns=['x', 'y'])
with sns.axes_style("white"):
    sns.jointplot(x='x', y='y', data=df2, color='r', kind='hex')

with sns.axes_style('darkgrid'):
    sns.jointplot(x='x', y='y', data=df, color='k', kind='reg')

with sns.axes_style('dark'):
    sns.jointplot(x='x', y='y', data=df, color='b', kind='kde')

sns.jointplot(x='x', y='y', data=df, color='r').plot_joint(sns.kdeplot, zorder=0, n_level=6)


iris = sns.load_dataset('iris')
sns.pairplot(iris)




sns.plt.show()