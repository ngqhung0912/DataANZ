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


def build_geometry(data):
    """
    This function builds a Shapely Point geometry based on the longitude and latitude values.
    :param data: Pandas data frame
    :return: Shapely Point geometry
    """
    location=data['merchant_long_lat']
    if type(location) is str:
        longitude=location.split()[0]
        longitude=float(ast.literal_eval(longitude))
        latitude=location.split()[1]
        latitude=float(ast.literal_eval(latitude))
        if longitude==0 and latitude==0:
            return None
        else:    
            return geometry.Point(longitude, latitude)
    else:
        return None
    
def build_longitude(data):
    """
    This function builds a Shapely Point geometry based on the longitude and latitude values.
    :param data: Pandas data frame
    :return: Shapely Point geometry
    """
    location=data['merchant_long_lat']
    if type(location) is str:
        longitude=location.split()[0]
        longitude=float(ast.literal_eval(longitude))
        latitude=location.split()[1]
        latitude=float(ast.literal_eval(latitude))
        if longitude==0 and latitude==0:
            return 0
        else:    
            return longitude
    else:
        return 0

    
def build_latitude(data):
    """
    This function builds a Shapely Point geometry based on the longitude and latitude values.
    :param data: Pandas data frame
    :return: Shapely Point geometry
    """
    location=data['merchant_long_lat']
    if type(location) is str:
        longitude=location.split()[0]
        longitude=float(ast.literal_eval(longitude))
        latitude=location.split()[1]
        latitude=float(ast.literal_eval(latitude))
        if longitude==0 and latitude==0:
            return 0
        else:    
            return latitude
    else:
        return 0


def get_range(numerical_value, bins):
    bins_sets=generate_range_bins(bins)
    range=""
    for bin in bins_sets:
        if numerical_value >= bin[0] and numerical_value<=bin[1]:
            range="{}-{}".format(bin[0],bin[1])
            break

    if range is "":
        return "{}+".format(bins_sets[-1][1])
    else:
        return range 
    
    
def generate_range_bins(bins):
    """
    This function returns generates a range bin from a list of increasing numbers.
    :param bins: list of numbers
    :return: string representation of a range
    """
    bins = [int(x) for x in bins]
    bins_sets=[]
    for i, val in enumerate(bins):
        if i==0:
            bins_sets.append((bins[i],bins[i+1]))
        elif i>0 and i<len(bins)-1:
            bins_sets.append((bins[i]+1,bins[i+1]))
    return bins_sets

def sort_ranges(range_bin, ranges, count_list):
    # The following code gets the order of the range according to the bin associated with that range, and then 
    # in the end sorts the ranges based on the index that has been retrieved bringing the ranges in the correct order.
    range_bins=[]
    for i,r in enumerate(ranges):
        if '+' in r:
            p1=int(r.split('+')[0])            
            if p1>= range_bin[-1][1]:
                range_bins.append([len(range_bin)-1,r, count_list[i]])
        else:            
            p1=int(r.split('-')[0])
            p2=int(r.split('-')[1])
            index=range_bin.index((p1,p2))
            range_bins.append([index,r, count_list[i]])
    range_bins=sorted(range_bins, key=lambda x: x[0])
    range_bin=[x[1] for x in range_bins]
    range_counts=[x[2] for x in range_bins]
    return range_bin, range_counts







        