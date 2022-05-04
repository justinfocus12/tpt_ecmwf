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


class SeasonalCrommelinModelFeatures:
    def __init__(self,featspec_filename,szn_start,szn_length,Nt_szn,szn_avg_window,dt_samp,delaytime=0):
        self.featspec_filename = featspec_filename 
        self.dt_samp = dt_samp # For now assume a constant temporal resolution on all data. Will relax this in the future once we have sampling-interval-independent observables. 
        self.delaytime = delaytime # The length of time-delay embedding to use as features in the model, in terms of time units (not time samples). Hopefully we can make time delay embedding features independent of the sampling rate. 
        self.ndelay = int(round(self.delaytime/self.dt_samp)) + 1
        self.szn_start = szn_start 
        self.szn_length = szn_length 
        self.Nt_szn = Nt_szn # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.dt_szn = self.szn_length/self.Nt_szn
        self.t_szn_edge = self.szn_start + np.linspace(0,self.szn_length,self.Nt_szn+1)
        self.t_szn_cent = 0.5*(self.t_szn_edge[:-1] + self.t_szn_edge[1:])
        self.szn_avg_window = szn_avg_window
        return
    def create_features_from_climatology(self,raw_file_list):
        #TODO
        """
        Computes features from climatological database, e.g., find the EOFs and store them
        Parameters
        ----------
        raw_file_list: list of str's
            A list of netcdf files (each ending with '.nc') in xarray Dataset format, each with 't_sim' as a coordinate. Ths function processes these files to generate climatological statistics.
        Returns 
        ------
        featspec: a dict with information on evaluating features downstream. 

        Side effects
        ------------
        write featspec to self.featspec_filename
        """
        # For EOFs, we'll want to stack all the geopotential heights into a big xarray, and then pass that to get_seasonal_stats. 
        # For Crommelin, simply find the mean and variance for each feature
        return
    def evaluate_features_database(self,raw_filename_list,save_filename,featspec=None):
        """
        Parameters
        ----------
        raw_filename_list: list of str
            Filenames, with .nc extensions, for the raw data to read.
        save_filenamd: str
            Filename, with .nc extension, to write the features. 
        """
        # Concatenate all the DataArrays together
        print(f"file = {raw_filename_list[0]}")
        traj = xr.open_dataset(raw_filename_list[0])
        print(f"traj = \n{traj}")
        X = xr.open_dataset(raw_filename_list[0])['X']
        # Which dimensions to concatenate along?
        for f in raw_filename_list[1:]:
            Xnew = xr.open_dataset(f)['X']
            # TODO: compute observables 
            X = xr.concat([X,Xnew], dim='ensemble')
        X.coords['ensemble'] = np.arange(len(raw_filename_list))
        X.to_netcdf(save_filename)
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
        x1 = Y.sel(feature='x1').data.flatten() # This feature determines whether we're in A or B
        t_szn = Y.sel(feature='t_szn') # Time since beginning of the season
        time_window_flag = 1.0*(t_szn.data > tpt_bndy['tthresh'][0])*(t_szn.data < tpt_bndy['tthresh'][1])
        blocked_flag = 1.0*(x1 <= tpt_bndy['block_thresh'])
        zonal_flag = 1.0*(x1 >= tpt_bndy['zonal_thresh'])
        A_flag = (1-time_window_flag) + time_window_flag*zonal_flag
        B_flag = time_window_flag*blocked_flag
        D_flag = (1-b_flag)*(1-a_flag)
        ab_tag = 0*A_flag + 1*B_flag + 2*D_flag
        return ab_tag

