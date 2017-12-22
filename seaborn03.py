import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


titanic = pd.read_csv('titanic.csv')

# with sns.axes_style('white'):
#     sns.barplot(x='sex', y='survived', hue='class', data=titanic)
#
#
# with sns.axes_style('white'):
#     sns.violinplot(x='class', y='survived', hue='embarked', data=titanic)

with sns.axes_style('whitegrid'):
    sns.pointplot(x='class', y='survived', hue='sex', data=titanic, palette={'male': 'r', 'female': 'b'}, markers=["o", 'p'], linestyles=['-', '--'])

sns.plt.show()
