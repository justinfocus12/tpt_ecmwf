# Methods to compute relevant observable functions from the Crommelin model. 
import numpy as np
import xarray as xr
import netCDF4 as nc
from numpy import save,load
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
import sys
from abc import ABC,abstractmethod


class CrommelinModelFeatures(TPTFeatures):
    def __init__(self,featspec_file,szn_start,szn_end,Nt_szn,szn_avg_window,delaytime=0):
        self.featspec_file = featspec_file 
        self.delaytime = delaytime # The length of time-delay embedding to use as features in the model. 
        super().__init__(featspec_file,szn_start,szn_end,Nt_szn,szn_avg_window)
        return
    def create_features_from_climatology(self,raw_file_list):
        #TODO
        """
        Computes features from climatological database, e.g., find the EOFs and store them
        Parameters
        ----------
        raw_file_list: list of str's
            A list of netcdf files (each ending with '.nc'), each with 't_szn' as a coordinate. Ths function processes these files to generate climatological statistics.
        Returns 
        ------
        featspec: a dict with information on evaluating features downstream. 
        """
        # For EOFs, we'll want to stack all the geopotential heights into a big xarray, and then pass that to get_seasonal_stats. 
        return
    def evaluate_features_database(self,raw_file):
        #TODO
        """
        """
        return
    def abtest(self,Y,featspec,tpt_bndy):
        """
        Parameters
        ----------
        Y: xarray.DataArray
            Dimensions must be ('feature','snapshot')
        featspec: xarray.Dataset
            A metadata structure that specifies how feaures are computed and evaluated. This will depend on the application.
        tpt_bndy: dict
            Metadata specifying the boundaries of the current TPT problem. 
        Returns 
        -------
        ab_tag: xarray.DataArray
            DataArray with one single dimension, 'snapshot', and an integer data array. Each entry is either 0 (if in A), 1 (if in B), or 2 (if in D).
        """
        x1 = Y.sel(feature='x1') # This feature determines whether we're in A or B
        t_szn = Y.sel(feature='t_szn')
        time_window_flag = 1.0*(t_szn.data > tpt_bndy['tthresh'][0])*(t_szn.data < tpt_bndy['tthresh'][1])
        blocked_flag = 1.0*(x1 <= tpt_bndy['block_thresh'])
        zonal_flag = 1.0*(x1 >= tpt_bndy['zonal_thresh'])
        A_flag = (1-time_window_flag) + time_window_flag*zonal_flag
        B_flag = time_window_flag*blocked_flag
        D_flag = (1-b_flag)*(1-a_flag)
        ab_tag = 0*A_flag + 1*B_flag + 2*D_flag
        return ab_tag

