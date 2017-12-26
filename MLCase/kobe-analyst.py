# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv('kobe-data.csv')
kobe = data[pd.notnull(data['shot_made_flag'])]

def kobe_analyst_01():
    print kobe.shape
    print kobe.head()

    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    plt.scatter(kobe.loc_x, kobe.loc_y, color="R", alpha=0.02)
    plt.title("loc_x and loc_Y")

    plt.subplot(122)
    plt.scatter(kobe.lon, kobe.lat, color="B", alpha=0.02)
    plt.title("lat and lon")
    plt.show()


def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = kobe.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


def kobe_anays_02():
    print data['shot_zone_area'].value_counts()

    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    scatter_plot_by_category('shot_zone_area')

    plt.subplot(122)
    scatter_plot_by_category("shot_zone_basic")

    plt.show()


if __name__ == '__main__':
    # kobe_analyst_01()
    # kobe_anays_02()

    # one-hot
    print pd.get_dummies(data['combined_shot_type'], prefix='combined_shot_type')[:2]





