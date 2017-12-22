import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

unrate_content = pd.read_csv("unrate.csv")

first_year_content = unrate_content[0:3]

print first_year_content

print unrate_content['DATE'].dt.month
print first_year_content['VALUE']

# plt.plot(first_year_content['VALUE'], first_year_content['VALUE'])

fig, ax = plt.subplots()
ax.plot_date(first_year_content['DATE'], first_year_content['VALUE'], '-')

ax.fmt_xdata = DateFormatter('%Y/%m/%d')

# plt.plot(first_year_content['DATE'], first_year_content['VALUE'] )
# fig.autofmt_xdate()

plt.xticks(rotation=45)

plt.ioff()
plt.show()
