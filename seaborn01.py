import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

unrate = pd.read_csv("unrate.csv")
first_year =  unrate.head(12)
dates = [pd.to_datetime(d) for d in first_year['DATE']]
rates = first_year['VALUE']



data = pd.DataFrame({'DATE': dates, 'VALUE': rates})


sns.set_style('whitegrid')
sns.despine()

#sns.distplot(unrate['DATE'], kde=False, axlabel='Year')


# sns.barplot(x='DATE', y='VALUE',data=first_year)

# g = sns.factorplot(x="DATE", y='VALUE', data=first_year, kind='bar', aspect=1.5, color="b")
# g = sns.factorplot(x="DATE", y='VALUE', data=first_year, kind='bar', aspect=1.5)
# g.set_xticklabels(rotation=30)



# sub plot

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

sns.pointplot(x='DATE', y='VALUE', data = first_year, ax=ax1)

g = sns.factorplot(x="DATE", y='VALUE', data=first_year, kind='bar', aspect=1.5, ax=ax2)
g.set_xticklabels(rotation=30)
plt.close(g.fig)

sns.plt.show()

