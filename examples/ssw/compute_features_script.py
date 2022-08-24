# This is a script to read a database of netcdf files, compute the specified nonlinear features on them, and write the resulting reduced dataset to disk as another netcdf.

import sys
sys.path.append("../..")

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import netCDF4 as nc
from importlib import reload
import sys
import os
from os import mkdir, makedirs
from os.path import join,exists
from importlib import reload
import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from datetime import datetime
from calendar import monthrange
import cartopy
from cartopy import crs as ccrs
