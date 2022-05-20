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
    def __init__(self,featspec_filename,szn_start,szn_length,year_length,Nt_szn,szn_avg_window,dt_samp,delaytime=0):
        self.featspec_filename = featspec_filename 
        self.dt_samp = dt_samp # For now assume a constant temporal resolution on all data. Will relax this in the future once we have sampling-interval-independent observables. 
        self.delaytime = delaytime # The length of time-delay embedding to use as features in the model, in terms of time units (not time samples). Hopefully we can make time delay embedding features independent of the sampling rate. 
        self.ndelay = int(round(self.delaytime/self.dt_samp)) + 1
        self.szn_start = szn_start 
        self.szn_length = szn_length 
        self.year_length = year_length
        self.Nt_szn = Nt_szn # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.dt_szn = self.szn_length/self.Nt_szn
        self.t_szn_edge =  np.linspace(0,self.szn_length,self.Nt_szn+1) #+ self.szn_start
        self.t_szn_cent = 0.5*(self.t_szn_edge[:-1] + self.t_szn_edge[1:])
        self.szn_avg_window = szn_avg_window
        return
    def time_conversion_from_absolute(self,t_abs):
        year = (t_abs / self.year_length).astype(int)
        t_cal = t_abs - year*self.year_length
        szn_start_year = year*(t_cal >= self.szn_start) + (year-1)*(t_cal < self.szn_start)
        t_szn = t_abs - (szn_start_year*self.year_length + self.szn_start)
        ti_szn = (t_szn / self.dt_szn).astype(int)
        return szn_start_year,t_cal,t_szn,ti_szn
    def illustrate_dataset(self,Xra_filename,Xhc_filename,results_dir,szns2illustrate):
        """
        Plot two different seasons, on top of the background climatology.
        Parameters
        ----------
        Xra_filename: str 
            the .nc file with long reanalysis trajectories (one per season)
        Xhc_filename: str 
            the .nc file with short hindcast trajectories (many per season) 
        results_dir: str
            directory where to save the illustrations
        szns2illustrate: list of str
            Year numbers to plot
        Returns
        --------
        Nothing
        Side effects
        --------
        Plot seasons with hindcasts atop background climatology
        """
        features2plot = ['x1','x4']
        quantile_midranges = [0.2,0.4,0.6,0.8,1.0]
        quantiles = np.sort([0.5 + 0.5*qmr*sgn for qmr in quantile_midranges for sgn in [-1,1]])
        Xra = xr.open_dataset(Xra_filename)['X']
        Xhc = xr.open_dataset(Xhc_filename)['X']
        # Get seasonal average
        t_abs_ra = Xra.sel(feature='t_abs')
        szn_id_ra,t_cal_ra,t_szn_ra,ti_szn_ra = self.time_conversion_from_absolute(t_abs_ra)
        t_abs_hc = Xhc.sel(feature='t_abs')
        szn_id_hc,t_cal_hc,t_szn_hc,ti_szn_hc = self.time_conversion_from_absolute(t_abs_hc)
        print(f"t_szn_hc = {t_szn_hc}")
        #t_szn_ra = np.mod(t_abs.isel(t_sim=0), self.year_length) + Xra.t_sim - self.szn_start
        Xra = Xra.sel(feature=features2plot)
        Xhc = Xhc.sel(feature=features2plot)
        Xclim = xr.DataArray(
                dims=['feature', 't_szn_cent', 'quantile'],
                coords={'feature': features2plot, 't_szn_cent': self.t_szn_cent, 'quantile': quantiles},
                data=np.zeros((len(features2plot), self.Nt_szn, len(quantiles))),
                )
        print(f"t_szn_cent = {Xclim.t_szn_cent}\nt_szn_ra = {t_szn_ra}")
        for i_tszn in range(self.Nt_szn):
            print(f"Starting i_tszn = {i_tszn} out of {self.Nt_szn}")
            selection = Xra.where(ti_szn_ra==i_tszn) #Xra.where((t_szn >= self.t_szn_edge[i_tszn])*(t_szn < self.t_szn_edge[i_tszn+1])).sel(feature=features2plot)
            for i_quant in range(len(quantiles)):
                #print(f"lhs = \n{lhs}\nrhs = \n{rhs}")
                Xclim.isel(t_szn_cent=i_tszn,quantile=i_quant)[:] = selection.quantile(quantiles[i_quant],dim=['ensemble','t_sim'],skipna=True).data.flatten()
        for szn_id in szns2illustrate:
            # Find all years matching a specific szn_id
            if not (szn_id in szn_id_ra):
                raise Exception(f"You asked for a szn_id of {szn_id}, which is not in the existing szn_id_start array of {szn_id_start_ra}")
            for feat in features2plot:
                # Plot climatology
                Xclim_feat = Xclim.sel(feature=feat)
                fig,ax = plt.subplots()
                for i_quant in range(len(quantiles)//2):
                    lower = Xclim_feat.isel(quantile=i_quant)
                    upper = Xclim_feat.isel(quantile=len(quantiles)-1-i_quant)
                    ax.fill_between(Xclim_feat.t_szn_cent,lower,upper,color=plt.cm.binary(0.2 + 0.7*quantiles[i_quant]), zorder=-len(quantiles)//2+i_quant)
                # Plot any trajectories matching the specific year 
                i_ens_ra = np.where(szn_id_ra==szn_id)[0]
                ax.plot(t_szn_ra.isel(ensemble=i_ens_ra[0],member=0), Xra.sel(feature=feat,member=0).isel(ensemble=i_ens_ra[0]), color='black', zorder=1)
                # Plot two hindcast trajectories on top
                i_ens_hc = np.where(szn_id_hc==szn_id)[0]
                print(f"t_szn_hc = {t_szn_hc}")
                print(f"i_ens_hc = {i_ens_hc}")
                i_ens_hc_2plot = []
                for szn_frac in [0.1,0.4,0.7]:
                    #print(f"list to choose from = {t_szn_hc.isel(ensemble=i_ens_hc,t_sim=0,member=0)}")
                    i_ens_hc_2plot += [i_ens_hc[np.argmin(np.abs(t_szn_hc.isel(ensemble=i_ens_hc,t_sim=0,member=0).data.flatten() - szn_frac*self.szn_length))]]
                for i_ens in i_ens_hc_2plot:
                    for i_mem in range(Xhc.member.size):
                        ax.plot(t_szn_hc.isel(ensemble=i_ens,member=i_mem), Xhc.isel(ensemble=i_ens,member=i_mem).sel(feature=feat), color='purple', zorder=2)
                fig.savefig(join(results_dir,f"clim_{feat}_{szn_id}"))
                plt.close(fig)
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
        # For the Crommelin model, feature space is the entire 6-dimensional staate space. Just Concatenate all the DataArrays together
        print(f"file = {raw_filename_list[0]}")
        traj = xr.open_dataset(raw_filename_list[0])['X']
        print(f"traj = \n{traj}")
        X_list = []
        for f in raw_filename_list:
            Xnew = xr.open_dataset(f)['X']
            X_list += [Xnew]
        X = xr.concat(X_list, dim='ensemble')
        X.coords['ensemble'] = np.arange(len(raw_filename_list))
        X.to_netcdf(save_filename)
        print("Finishing featurization")
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

