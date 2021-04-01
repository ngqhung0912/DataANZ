import os
import numpy as np
import geopandas
import pandas as pd
import ast
from shapely import geometry
from shapely import wkt
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import DBSCAN
from dateutil.parser import parse
import datetime
from rtree import index

# from mpl_toolkits.basemap import Basemap
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def group_by_data(data, columns):
    """
    This is a wrapper function which wraps around the pandas group by function.
    :param data: Pandas data frame
    :param columns: The columns which are to be used to be group the data frame on
    :return: Pandas data frame
    """
    return data.groupby(columns).size().to_frame('count').reset_index()


def plot_bar_plot(x, y, x_axis_label, y_axis_label, title, topvisible):
    """
    This function plots a bar plot of values of x against the y
    :param x: list of x values
    :param y: list of y values
    :param x_axis_label: list of labels for x
    :param y_axis_label: list of labels for y
    :param title: string value for the title of the plot
    :return: the filename of the figure that has been saved.
    """
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=10, rotation=90)
    ax.set_title(title)
    if topvisible == True: 
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/3.5, p.get_height() * 1.01 ))
   
    file_name = ((((title + '.png').replace('/', ''))).replace(' ', '_')).lower()
    fig.savefig(file_name)
    plt.show()



    