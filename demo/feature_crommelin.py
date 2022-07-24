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
    def __init__(self):
        return
    def set_event_params(self,epd):
        # epd = event parameter dictionary
        self.szn_start = epd["szn_start"] 
        self.szn_length = epd["szn_length"] 
        self.year_length = epd["year_length"]
        self.Nt_szn = epd["Nt_szn"] # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.szn_avg_window = epd["szn_avg_window"]
        self.dt_szn = self.szn_length/self.Nt_szn
        self.t_szn_edge =  np.linspace(0,self.szn_length,self.Nt_szn+1) #+ self.szn_start
        self.t_szn_cent = 0.5*(self.t_szn_edge[:-1] + self.t_szn_edge[1:])
    def time_conversion_from_absolute(self,t_abs):
        year = (t_abs / self.year_length).astype(int)
        t_cal = t_abs - year*self.year_length
        szn_start_year = year*(t_cal >= self.szn_start) + (year-1)*(t_cal < self.szn_start)
        t_szn = t_abs - (szn_start_year*self.year_length + self.szn_start)
        ti_szn = (t_szn / self.dt_szn).astype(int)
        return szn_start_year,t_cal,t_szn,ti_szn
    def illustrate_dataset(self,Xra_filename,Xhc_filename,results_dir,szns2illustrate,Xclim_filename,plot_climatology=True,suffix=""):
        """
        Parameters
        ----------
        Xra_filename: str 
            the .nc file with long reanalysis trajectories (one per season)
        Xhc_filename: str 
            the .nc file with short hindcast trajectories (many per season) 
        Xclim_filename: str
            the .nc file with the pre-computed climatology
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
        Xra = xr.open_dataset(Xra_filename)['X']
        Xhc = xr.open_dataset(Xhc_filename)['X']
        Xclim = xr.open_dataset(Xclim_filename)
        # Get seasonal average
        t_abs_ra = Xra.sel(feature='t_abs')
        szn_id_ra,t_cal_ra,t_szn_ra,ti_szn_ra = self.time_conversion_from_absolute(t_abs_ra)
        t_abs_hc = Xhc.sel(feature='t_abs')
        szn_id_hc,t_cal_hc,t_szn_hc,ti_szn_hc = self.time_conversion_from_absolute(t_abs_hc)
        Xra = Xra.sel(feature=features2plot)
        Xhc = Xhc.sel(feature=features2plot)
        for szn_id in szns2illustrate:
            # Find all years matching a specific szn_id
            if not (szn_id in szn_id_ra):
                raise Exception(f"You asked for a szn_id of {szn_id}, which is not in the existing szn_id array of {szn_id_ra}")
            for feat in features2plot:
                fig,ax = plt.subplots()
                if plot_climatology:
                    for i_quant in range(Xclim.coords['quantile'].size//2):
                        print(f"About to compute lower and upper for i_quant {i_quant} of feature {feat}")
                        lower = Xclim['Xquant'].sel(feature=feat).isel(quantile=2*i_quant)
                        upper = Xclim['Xquant'].sel(feature=feat).isel(quantile=2*i_quant+1)
                        ax.fill_between(Xclim.t_szn_cent,lower,upper,color=plt.cm.binary(0.2 + 0.7*Xclim.coords['quantile'].data[2*i_quant]), zorder=-Xclim.coords['quantile'].size//2+i_quant)
                    # Plot the mean and standard deviation in orange
                    #print(f"About to compute avg for feature {feat}")
                    avg = Xclim['Xmom'].sel(feature=feat,moment=1)
                    #print(f"In ilustrate, at initial time and feature, mean is {Xclim['Xmom'].isel(t_szn_cent=0,moment=0).sel(feature=feat)}, whereas avg = {avg}.")
                    std = np.sqrt(Xclim['Xmom'].sel(feature=feat,moment=2) - avg**2)
                    ax.plot(Xclim.t_szn_cent, avg, color='orange')
                    for sgn in [-1,1]:
                        ax.plot(Xclim.t_szn_cent, avg+sgn*std, color='orange', linestyle='--')
                # Plot any trajectories matching the specific year 
                i_ens_ra = np.where(szn_id_ra==szn_id)[0]
                ax.plot(t_szn_ra.isel(ensemble=i_ens_ra[0],member=0), Xra.sel(feature=feat,member=0).isel(ensemble=i_ens_ra[0]), color='black', zorder=1)
                # Plot two hindcast trajectories on top
                i_ens_hc = np.where(szn_id_hc==szn_id)[0]
                #print(f"i_ens_hc = {i_ens_hc}")
                i_ens_hc_2plot = []
                for szn_frac in [0.3,0.5,0.95]:
                    #print(f"list to choose from = {t_szn_hc.isel(ensemble=i_ens_hc,t_sim=0,member=0)}")
                    i_ens_hc_2plot += [i_ens_hc[np.argmin(np.abs(t_szn_hc.isel(ensemble=i_ens_hc,t_sim=-1,member=0).data.flatten() - szn_frac*self.szn_length))]]
                for i_ens in i_ens_hc_2plot:
                    for i_mem in range(Xhc.member.size):
                        ax.plot(t_szn_hc.isel(ensemble=i_ens,member=i_mem), Xhc.isel(ensemble=i_ens,member=i_mem).sel(feature=feat), color='purple', zorder=2, alpha=0.3)
                fig.savefig(join(results_dir,f"clim_{feat}_{szn_id}{suffix}"))
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
    def compute_climatology(self, in_file: str, out_file: str, quantile_midranges=None):
        """
        Parameters
        ---------
        in_file: filename containing xarray with coordinates (member, t_sim, feature) 
        out_file: filename for an xarray where to store relevant statistics of the input data to be used for defining the features for clustering.
        """
        print(f"Beginning compute_climatology")
        X = xr.open_dataset(in_file)['X']
        # Group by the season time and average period by period
        if quantile_midranges is None:
            quantile_midranges = [0.2,0.4,0.6,0.8,1.0]
        quantile_midranges = np.sort(quantile_midranges)[::-1] # from high tolow, because that's the order in which we should overlay them  
        quantiles = []
        for qmr in quantile_midranges:
            quantiles += [0.5-0.5*qmr, 0.5+0.5*qmr]
        quantiles += [0.5] # Median
        quantiles = np.array(quantiles)
        moments = np.array([1, 2, 3])
        t_abs = X.sel(feature='t_abs')
        print(f"Starting time conversion")
        szn_id,t_cal,t_szn,ti_szn = self.time_conversion_from_absolute(t_abs)
        print(f"Finished time conversion")
        Xclim = xr.Dataset(
                coords={'feature': X.coords['feature'], 't_szn_cent': self.t_szn_cent, 'quantile': quantiles, 'moment': moments},
                data_vars = {
                    'Xquant': (['feature', 't_szn_cent', 'quantile'], np.zeros((X.coords['feature'].size, self.Nt_szn, len(quantiles)))), 
                    'Xmom': (['feature', 't_szn_cent', 'moment'], np.zeros((X.coords['feature'].size, self.Nt_szn, len(moments)))),
                    }
                )
        print(f"Starting to sweep through i_tszn")
        for i_tszn in range(self.Nt_szn):
            if i_tszn % 20 == 0: print(f"Starting i_tszn = {i_tszn} out of {self.Nt_szn}")
            selection = X.where(ti_szn==i_tszn) 
            for i_quant in range(len(quantiles)):
                Xclim['Xquant'].isel(t_szn_cent=i_tszn,quantile=i_quant)[:] = selection.quantile(quantiles[i_quant],dim=['ensemble','t_sim'],skipna=True).data.flatten()
            for i_mom in range(len(moments)):
                Xclim['Xmom'].isel(t_szn_cent=i_tszn,moment=i_mom)[:] = np.power(selection, moments[i_mom]).mean(dim=['ensemble','t_sim'], skipna=True).data.flatten()
                #print(f"At initial time and feature, mean is {Xclim['Xmom'].isel(t_szn_cent=i_tszn,moment=i_mom,feature=0)}. min X = {selection.isel(feature=0).min()}, max = {selection.isel(feature=0).max()}")
        Xclim.to_netcdf(out_file)
        return 
    def seasonalize(self,X,Xclim,inverse=False):
        """
        Demean and de-seasonalize, or re-mean and re-seasonalize 
        Parameters
        ----------
        X: xr.DataArray
            Contains the data to transform according to Xclim
        Xclim: xr.Dataset
            Contains the climatological statistics
        Returns
        -------
        Xszn: xr.DataArray
            A normalized (or un-normalized) version of X

        Side effects
        ------------
        None
        """
        print(f"X.coords = {X.coords}")
        print(f"Xclim.coords = {Xclim.coords}")
        t_abs = X.sel(feature='t_abs')
        szn_id,t_cal,t_szn,ti_szn = self.time_conversion_from_absolute(t_abs)
        Xszn = xr.DataArray(
                data = np.zeros(X.shape),
                coords = X.coords, 
                dims = X.dims,
                )
        modified_flag = 0*Xszn
        for i_tszn in range(self.Nt_szn):
            Xmom_t = Xclim['Xmom'].isel(t_szn_cent=i_tszn, drop=True)
            Xmom_mean = Xmom_t.sel(moment=1, drop=True)
            Xmom_std = np.sqrt(Xmom_t.sel(moment=2, drop=True) - Xmom_mean**2)
            if inverse:
                rhs = Xmom_std * X + Xmom_mean
            else:
                rhs = (X - Xmom_mean)/Xmom_std
            #print(f"Xmom_t.coords = {Xmom_t.coords}")
            #print(f"rhs.coords = {rhs.coords}")
            #print(f"Xszn.coords = {Xszn.coords}")
            #print(f"modified_flag.coords = {modified_flag.coords}")
            Xszn += (ti_szn == i_tszn) * rhs * (modified_flag == 0)
            modified_flag += (ti_szn == i_tszn)
        # Correct the time coordinate, which should be the same as X
        Xszn.loc[dict(feature='t_abs')] = X.sel(feature='t_abs').data
        return Xszn
    def evaluate_features_for_dga(self,in_file,out_file,clim_file,ndelay,inverse=False):
        """
        Demean and de-seasonalize to feed into clustering algorithm
        Parameters
        ----------
        input_filename: str
            Filename (ending with X.nc, probably) for the data transformed from raw
        output_filename: str
            Filename (ending with Y.nc, probably) for the data to use 
        clim_file: str
            Filename (ending with Xclim.nc, probably) the climatology so that the features can be normalized.
        inverse: bool
            True if we are normalizing, False if we are unnormalizing

        Returns
        -------
        Nothing

        Side effects
        ------------
        Write output (e.g., with time delays) to output_filename. 
        """
        Xclim = xr.open_dataset(clim_file)
        X = xr.open_dataset(in_file)['X']
        print(f"X['feature'] = {list(X['feature'].data)}")
        t_abs = X.sel(feature='t_abs')
        szn_id,t_cal,t_szn,ti_szn = self.time_conversion_from_absolute(t_abs)
        if inverse:
            X = X.sel(feature=Xclim.coords['feature'].data, drop=True)
        Xszn = self.seasonalize(X,Xclim,inverse=inverse)
        if inverse:
            Y = Xszn
        else:
            # Build in time-delays
            Ycoords = dict()
            Ycoords['t_sim'] = X.coords['t_sim'].data[ndelay:]
            Yfeat = []
            for feat in list(X.coords['feature'].data):
                print(f"Beginning to time-delay feature {feat}")
                Yfeat += [f"{str(feat)}"]
                for i_delay in range(1,ndelay+1):
                    Yfeat += [f"{str(feat)}_delay{i_delay}"]
            Ycoords['feature'] = Yfeat
            Ycoords['ensemble'] = Xszn.coords['ensemble']
            Ycoords['member'] = Xszn.coords['member']
            Y = xr.DataArray(
                data=np.zeros((len(Ycoords['ensemble']),len(Ycoords['member']),len(Ycoords['t_sim']),len(Ycoords['feature']))),
                coords=Ycoords,
                dims=["ensemble","member","t_sim","feature"],
                )
            # TODO: make a design decision about how to deal with time-delays. It's inefficient, but maybe general, to make such a huger feature space
            for feat in list(Xszn.coords['feature'].data):
                Y.sel(feature=feat)[:] = Xszn.sel(feature=feat,drop=True).isel(t_sim=slice(ndelay,Xszn.coords['t_sim'].size)).data
                for i_delay in range(1,ndelay+1):
                    Y.sel(feature=f"{str(feat)}_delay{i_delay}")[:] = Xszn.sel(feature=feat,drop=True).isel(t_sim=slice(ndelay-i_delay,Xszn.coords['t_sim'].size-i_delay)).data
        Y_ds = xr.Dataset(data_vars={"X": Y})
        Y_ds.to_netcdf(out_file)
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
            da_new = xr.open_dataset(f)['X']
            # Here we perform the data reduction. For this model, transfer the information to magnitude and phase. 
            Xnew = xr.DataArray(
                    coords = {
                        "member": da_new.coords["member"],
                        "t_sim": da_new.coords['t_sim'], 
                        "feature": ["E01","E02","E11","ph11","E12","ph12","t_abs"],
                        },
                    data = np.zeros((da_new.member.size,da_new.t_sim.size,7)),
                    dims = ["member","t_sim","feature"],
                    )
            Xnew.loc[dict(feature="t_abs")] = da_new.sel(feature="t_abs")
            Xnew.loc[dict(feature="E01")] = 0.5*da_new.sel(feature="x1")**2
            Xnew.loc[dict(feature="E02")] = 2.0*da_new.sel(feature="x4")**2
            Xnew.loc[dict(feature="E11")] = (da_new.sel(feature=["x2","x3"])**2).sum(dim="feature")
            Xnew.loc[dict(feature="E12")] = (da_new.sel(feature=["x5","x6"])**2).sum(dim="feature")
            Xnew.loc[dict(feature="ph11")] = np.arctan2(-da_new.sel(feature="x3"), da_new.sel(feature="x2"))
            Xnew.loc[dict(feature="ph12")] = np.arctan2(-da_new.sel(feature="x6"), da_new.sel(feature="x5"))
            X_list += [Xnew]
        X = xr.concat(X_list, dim='ensemble')
        X.coords['ensemble'] = np.arange(len(raw_filename_list))
        X.to_netcdf(save_filename)
        print("Finishing featurization")
        return
    # ------------------- Below is a collection of observable functions ----------------
    def orography_cycle(self,ds):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        gamma_t: numpy.ndarray
            The gamma parameter corresponding to the given time of year. It varies sinusoidaly. Same is (Nx,m_max) where m_max is the maximum zonal wavenumber.
        gamma_tilde_t: numpy.ndarray
            The gamma_tilde parameter corresponding to the given time of year. Same shape as gamma_t.
        """
        t_abs = ds["X"].sel(feature="t_abs")
        gamma = xr.DataArray(
                coords={
                    "member": ds.coords['member'],
                    "t_sim": ds.coords['t_sim'],
                    "feature": ["g","g_dot"],
                    "tilde": [0,1], # 0 is for gamma; 1 is for gamma_tilde
                    "m": [1,2],
                    },
                data = np.zeros((ds['member'].size, ds['t_sim'].size, 2, 2, 2)),
                dims = ["member","t_sim","feature","tilde","m"],
                )
        cosine = np.cos(2*np.pi*t_abs/ds.attrs["year_length"])
        sine = np.sin(2*np.pi*t_abs/ds.attrs["year_length"])
        for i_m in range(gamma["m"].size):
            for tilde in [0,1]:
                limits = ds.attrs["gamma_tilde_limits"][:,i_m] if tilde else ds.attrs["gamma_limits"]
                amplitude = (limits[1] - limits[0])/2
                offset = (limits[1] + limits[0])/2
                gamma.loc[dict(feature="g",tilde=tilde,m=gamma["m"][i_m])] = cosine * amplitude + offset
                gamma.loc[dict(feature="g_dot",tilde=tilde,m=gamma["m"][i_m])] = -sine * amplitude
        return gamma
    def energy_enstrophy_coeffs(self,ds):
        # Return the coefficients for each energy and enstrophy reservoir given a dataset's attributes
        b = ds.attrs["b"]
        coeffs_energy = dict({
            "E01": 0.5,
            "E02": 2.0,
            "E11": (b**2 + 1)/2,
            "E12": (b**2 + 4)/2,
            })
        coeffs_enstrophy = dict({
            "Om01": 1.0/(2*b**2),
            "Om02": 8.0/b**2,
            "Om11": (b**2 + 1)**2/(2*b**2),
            "Om12": (b**2 + 4)**2/(2*b**2),
            })
        return coeffs_energy,coeffs_enstrophy
    def energy_observable(self,ds):
        # ds is the dataset, including metadata
        # Return a dataarray with all the energy components
        cE,_ = self.energy_enstrophy_coeffs(ds)
        da_E = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "feature": ["E01","E02","E11","E12","Etot",]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 5)),
                dims = ["member","t_sim","feature"],
                )
        da_E.loc[dict(feature="E01")] = cE["E01"]*ds["X"].sel(feature="x1")**2
        da_E.loc[dict(feature="E02")] = cE["E02"]*ds["X"].sel(feature="x4")**2
        da_E.loc[dict(feature="E11")] = cE["E11"]*(ds["X"].sel(feature=["x2","x3"])**2).sum(dim=["feature"])
        da_E.loc[dict(feature="E12")] = cE["E12"]*(ds["X"].sel(feature=["x5","x6"])**2).sum(dim=["feature"])
        da_E.loc[dict(feature="Etot")] = da_E.sel(feature=["E01","E02","E11","E12"]).sum(dim=["feature"])
        return da_E
    def energy_tendency_observable(self,ds,da_E=None):
        # Compute the tendency of the total energy.
        da_Edot = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "feature": ["dissipation","forcing","quadratic","Etot"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 4)),
                dims = ["member","t_sim","feature"],
                )
        if da_E is None:
            da_E = self.energy_observable(ds)
        da_Edot.loc[dict(feature="dissipation")] = -2*ds.attrs["C"]*da_E.sel(feature="Etot")
        da_Edot.loc[dict(feature="forcing")] = ds.attrs["C"]*(
                ds["X"].sel(feature="x1")*ds.attrs["xstar"][0] + 
                4*ds["X"].sel(feature="x4")*ds.attrs["xstar"][3]
                )
        da_Edot.loc[dict(feature="quadratic")] = 0.0
        da_Edot.loc[dict(feature="Etot")] = da_Edot.sel(feature=["dissipation","forcing","quadratic"]).sum(dim=["feature"])
        return da_Edot
    def energy_tendency_observable_findiff(self,ds,da_E=None):
        # Approximate the tendency of energy by taking a finite difference. The dataset must be time-ordered.
        if da_E is None:
            da_E = self.energy_observable(ds)
        da_Edot_findiff = da_E.differentiate('t_sim')
        return da_Edot_findiff
    def energy_exchange_observable(self,ds,da_E=None):
        # Compute the energy transfers 
        if da_E is None:
            da_E = self.energy_observable(ds)
        da_Ex = xr.DataArray(
                coords = {
                    "member": ds.coords["member"], 
                    "t_sim": ds.coords["t_sim"], 
                    "source": ["E01","E02","E11","E12","forcing","dissipation",],
                    "sink": ["E01","E02","E11","E12","forcing","dissipation",],
                    },
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 6, 6)),
                dims = ["member","t_sim","source","sink"],
                )
        gamma = self.orography_cycle(ds)
        # First, the input from forcing
        da_Ex.loc[dict(source="forcing",sink="E01")] = 2*cE["E01"]*ds.attrs["C"]*ds.sel(feature="x1")*ds.attrs["xstar"][0]
        da_Ex.loc[dict(source="forcing",sink="E02")] = 2*cE["E02"]*ds.attrs["C"]*ds.sel(feature="x4")*ds.attrs["xstar"][3]
        # Second, the leakage due to dissipation 
        for key in ["E01","E02","E11","E12"]:
            da_Ex.loc[dict(source=key,sink="dissipation")] = 2*ds.attrs["C"]*da_E.sel(feature=key)
        da_Ex.loc[dict(source="E11",sink="dissipation")] += 2*gamma.sel(tilde=0,m=1,feature="g")*ds.sel(feature="x1")*ds.sel(feature="x3")*cE["E11"]
        da_Ex.loc[dict(source="E12",sink="dissipation")] += 2*gamma.sel(tilde=0,m=2,feature="g")*ds.sel(feature="x4")*ds.sel(feature="x6")*cE["E12"]
        da_Ex.loc[dict(source="E11",sink="E02")] += 2*ds.attrs["epsilon"]*cE["E02"]*ds.sel(feature="x4")*(
                ds.sel(feature="x2")*ds.sel(feature="x6") - 
                ds.sel(feature="x3")*ds.sel(feature="x5")
                )
        da_Ex.loc[dict(source="E11",sink="E12")] += 2*ds.attrs["delta"][1]*cE["E12"]*ds.sel(feature="x4")*(
                ds.sel(feature="x2")*ds.sel(feature="x6") - 
                ds.sel(feature="x3")*ds.sel(feature="x5")
                )
        return da_Ex
    def enstrophy_observable(self,ds):
        # ds is the dataset, including metadata
        # Return a dataarray with all the energy components
        _,cOm = self.energy_enstrophy_coeffs(ds)
        dsOm = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "feature": ["Om01","Om02","Om11","Om12","Omtot"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 5)),
                dims = ["member","t_sim","feature"],
                )
        dsOm.loc[dict(feature="Om01")] = cOm["Om01"]*ds["X"].sel(feature="x1")**2
        dsOm.loc[dict(feature="Om02")] = cOm["Om02"]*ds["X"].sel(feature="x4")**2
        dsOm.loc[dict(feature="Om11")] = cOm["Om11"]*(ds["X"].sel(feature=["x2","x3"])**2).sum(dim=["feature"])
        dsOm.loc[dict(feature="Om12")] = cOm["Om12"]*(ds["X"].sel(feature=["x5","x6"])**2).sum(dim=["feature"])
        dsOm.loc[dict(feature="Omtot")] = dsOm.sel(feature=["Om01","Om02","Om11","Om12"]).sum(dim=["feature"])
        return dsOm
    def phase_observable(self,ds):
        # Return the phases of waves 1 and 2
        dsph = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "feature": ["ph11","ph12"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 2)),
                dims = ["member","t_sim","feature"],
                )
        dsph.loc[dict(feature="ph11")] = np.arctan2(-ds["X"].sel(feature="x3"), ds["X"].sel(feature="x2"))
        dsph.loc[dict(feature="ph12")] = np.arctan2(-ds["X"].sel(feature="x6"), ds["X"].sel(feature="x5"))
        return dsph
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

