# Make a class to handle different, and heterogeneous, types of SSW data. Retain knowledge of multiple file sources at once, and mix and match them (with different durations and everything). Be able to work with any subset of the data for feature creation, training, and testing. Note that the minimal unit for training and testing is a file, i.e., an ensemble. The subset indices could be generated randomly externally. 
# Ultimately, create the information to compute any quantity of interest on new, unseen data: committors and lead times crucially. Also backward quantities and rates, given an initial distribution.
# the DGA_SSW object will have no knowledge of what year anything is; that will be implmented at a higher level. 
# Maybe even have variable lag time?
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas
import datetime
import time as timelib
from calendar import monthrange
import matplotlib
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'serif', 'size': 12}
font = {'family': 'serif', 'size': 18}
bigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
from scipy import interpolate
import sys
import os
from os import mkdir
from os.path import join,exists
from sklearn import linear_model
import helper
import cartopy
from cartopy import crs as ccrs
import pickle
import itertools
from feature_template import SeasonalFeatures
import xr_utils

def pca_several_levels(gh,Nsamp,Nlat,Nlon,Npc,arr_eof,arr_singvals,arr_totvar,lev_subset):
    # Perform PCA on one level of geopotential height and fill in the results 
    # gh is a numpy array ith only the relevant levels, while level_subset tells us where to fill in the entries of the output Arrays
    Nlev = len(lev_subset)
    for i_lev in lev_subset:
        U,S,Vh = np.linalg.svd(ghilev.reshape((Nsamp,Nlat*Nlon)),full_matrices=False)
    arr_eof[idx_eof:idx_eof+Nsamp*Npc] = U[:,:Npc].flatten()
    arr_sv[idx_sv:idx_sv+Npc] = S[:Npc]
    arr_tv[idx_tv] = np.sum(S**2)
    return

class WinterStratosphereFeatures(SeasonalFeatures):
    # Create a set of features, including out-of-sample extension. 
    def __init__(self):
        self.lat_uref = 60 # Degrees North for CP07 definition of SSW
        self.pres_uref = 10 # hPa for CP07 definition of SSW
        self.time_origin = np.datetime64("1900-01-01T00:00:00")
        self.time_unit = np.timedelta64(1, 'D') # The time change unit 
        super().__init__()
        return
    def set_ab_code(self):
        self.ab_code = {"A": 0, "B": 1, "D": 2}
        return
    def set_ab_boundaries(self, t_thresh0, t_thresh1, ubar_10_60_thresh):
        self.tpt_bndy = dict({
            "t_thresh": [t_thresh0,t_thresh1],
            "ubar_10_60_thresh": ubar_10_60_thresh,
        })
        # The time thresholds are in days since the beginning of the season.
        return
    def set_event_seasonal_params(self):
        self.szn_start = {"month": 10, "day": 1}
        self.szn_length = np.sum([monthrange(1901,month) for month in [10, 11, 12, 1 ,2, 3]])
        self.Nt_szn = int(self.szn_length / 2) # Approximately two days per averaging window 
        self.dt_szn = self.szn_length / self.Nt_szn
        self.t_szn_edge = np.linspace(0, self.szn_length, self.Nt_szn+1)
        return 
    # ---------------- Observables -------------------------
    # TODO: for all the following, augment coordinates of observables to include multiple members. In fact, require every dataset input to include (time,ensemble,member) as coordinates. The calling function must expand dimensions accordingly. 
    def prepare_blank_observable(self, ds, feature_coord_names):
        # A general method to create a blank DataArray that replaces the coordinates "longitude" and "latitude" in the input dataset (ds) with a dimension "feature" with coordinates feature_names
        F_coords = {key: ds.coords[key].to_numpy() for key in list(ds.coords.keys())}
        for spatial_coord in ["longitude","latitude","level"]:
            F_coords.pop(spatial_coord)
        F_coords["feature"] = feature_coord_names 
        F = xr.DataArray(
                coords = F_coords, 
                dims = list(F_coords.keys()),
                data = 0.0, # This is necessary to assign later by broadcasting 
                )
        return F
    def pc_observable(self, ds, src, ds_eofs, ds_monclim, subtract_monthly_mean=False):
        # Project the geopotential height fields onto the EOFs 
        ds = xr_utils.preprocess_netcdf(ds, src)
        pc_features = []
        for level in [10, 100, 500, 850]:
            for i_pc in [1, 2, 3, 4]:
                pc_features += [f"pc_{level}_{i_pc}"]
        if subtract_monthly_mean:
            pc_all = (
                    (ds["gh"].groupby("time.month") - ds_monclim["gh"]) * ds_eofs["eofs"] * np.sqrt(np.maximum(0, np.cos(ds_eofs["latitude"] * np.pi/180)))
                    ).sum(dim=["latitude","longitude"]) / np.sqrt(ds_eofs["variance_fraction"])
        else:
            pc_all = (
                    ds["gh"] * ds_eofs["eofs"] * np.sqrt(np.maximum(0, np.cos(ds_eofs["latitude"] * np.pi/180)))
                    ).sum(dim=["latitude","longitude"]) / np.sqrt(ds_eofs["variance_fraction"])
        pc = self.prepare_blank_observable(ds, pc_features)
        for level in [10, 100, 500, 850]:
            for i_pc in [1, 2, 3, 4]:
                pc.loc[dict(feature=f"pc_{level}_{i_pc}")] = (
                        pc.sel(feature=f"pc_{level}_{i_pc}") +
                        pc_all.sel(level=level, mode=i_pc)
                        )
        return pc
    def temperature_observable(self, ds, src):
        #zonal-mean zonal wind at a range of latitudes, and all altitudes
        ds = xr_utils.preprocess_netcdf(ds, src)
        temp_features = ["Tcap_10_60to90", "Tcap_100_60to90", "Tcap_500_60to90", "Tcap_850_60to90"] #ds.coords['level'].to_numpy()
        Tcap = self.prepare_blank_observable(ds, temp_features)
        lat_idx = np.sort(np.where((ds.latitude >= 60)*(ds.latitude < 90))[0])
        cos_weight = np.cos(ds.latitude[lat_idx] * np.pi/180) 
        cos_weight *= 1.0/(np.sum(cos_weight) * ds.longitude.size)
        Tcap_all_levels = (ds['t'].isel(latitude=lat_idx) * cos_weight).sum(dim=["longitude","latitude"])
        for level in [10, 100, 500, 850]:
            Tcap.loc[dict(feature=f"Tcap_{level}_60to90")] = (
                    Tcap.sel(feature=f"Tcap_{level}_60to90") + 
                    Tcap_all_levels.sel(level=level)
                    )
        return Tcap
    def heatflux_observable(self, ds, src):
        # Heat flux at various wavenumbers between 45 and 75 degrees N. 
        ds = xr_utils.preprocess_netcdf(ds, src)
        min_lat = 45 
        max_lat = 75 
        Nlon = ds.longitude.size 
        max_mode = 3
        lat_idx = np.sort(np.where((ds.latitude >= min_lat)*(ds.latitude < max_lat))[0])
        cos_weight = np.cos(ds.latitude[lat_idx] * np.pi/180) 
        cos_weight *= 1.0/cos_weight.sum()
        T = (ds['t'].isel(latitude=lat_idx) * cos_weight).sum(dim="latitude")
        v = (ds['v'].isel(latitude=lat_idx) * cos_weight).sum(dim="latitude")
        ax_lon = T.dims.index("longitude") 
        vhat = xr.zeros_like(v, dtype=complex).rename(longitude="mode").assign_coords(mode=np.arange(Nlon))
        That = xr.zeros_like(T, dtype=complex).rename(longitude="mode").assign_coords(mode=np.arange(Nlon))
        vhat[:] = np.fft.fft(v.to_numpy(), axis=ax_lon)
        That[:] = np.fft.fft(T.to_numpy(), axis=ax_lon) 
        vT_wavenumbers = 1/Nlon**2 * vhat.conj()*That
        #vT_wavenumbers.loc[dict(mode=slice(1,None))] *= 2
        vT_mean_space = (v*T).mean(dim="longitude")
        vT_mean_freq = vT_wavenumbers.sum(dim="mode")
        #print(f"max parseval error = {np.max(np.abs(vT_mean_space - vT_mean_freq))}")
        vT_features = []
        for level in [10, 100, 500, 850]:
            for mode in range(max_mode+1):
                vT_features += [f"vT_{level}_{mode}"]
        vT = self.prepare_blank_observable(ds, vT_features)
        for level in [10, 100, 500, 850]:
            for mode in range(max_mode+1):
                factor = 1.0 if mode == 0 else 2.0 # We only looked at the first half of the spectrum.
                vT.loc[dict(feature=f"vT_{level}_{mode}")] = (
                        vT.sel(feature=f"vT_{level}_{mode}") +
                        vT_wavenumbers.sel(level=level, mode=mode).real
                        ) * factor
        return vT
    def qbo_observable(self, ds_in, src):
        # Zonal-mean zonal wind at 10 hPa in a region near the equator
        ds = xr_utils.preprocess_netcdf(ds_in, src)
        qbo_features = ["ubar_10_0pm5", "ubar_100_0pm5"]
        qbo = self.prepare_blank_observable(ds, qbo_features)
        lat_idx, = np.where((ds['latitude'].to_numpy() >= -5)*(ds['latitude'].to_numpy() <= 5))
        ubar_equatorial = ds['u'].isel(latitude=lat_idx).sel(level=[10,100]).mean(dim=["longitude","latitude"])
        for level in [10, 100]:
            qbo.loc[dict(feature=f"ubar_{level}_0pm5")] = (
                    qbo.sel(feature=f"ubar_{level}_0pm5") + 
                    ubar_equatorial.sel(level=level)
                    )
        return qbo
    def ubar_observable(self, ds_in, src):
        #zonal-mean zonal wind at a range of latitudes, and all altitudes
        ds = xr_utils.preprocess_netcdf(ds_in, src)
        ubar_features = ["ubar_10_60", "ubar_100_60", "ubar_500_60", "ubar_850_60"] #ds.coords['level'].to_numpy()
        ubar = self.prepare_blank_observable(ds, ubar_features)
        ubar_60 = ds['u'].sel(latitude=60).mean(dim='longitude')
        for level in [10, 100, 500, 850]:
            ubar.loc[dict(feature=f"ubar_{level}_60")] = ubar.sel(feature=f"ubar_{level}_60") + ubar_60.sel(level=level)
        return ubar
    def time_observable(self, ds_in, src):
        # Return all the possible kinds of time we might need from this object. The basic unit will be DAYS 
        ds = xr_utils.preprocess_netcdf(ds_in, src) 
        time_types = ["t_dt64", "t_abs", "t_cal", "t_szn", "year_cal", "year_szn_start"]
        tda = self.prepare_blank_observable(ds, time_types)
        Nt = ds["t_sim"].size
        # Regular time 
        treg = ds['t_init'] + self.time_unit * ds["t_sim"]
        tda.loc[dict(feature="t_dt64")] += treg.astype('float64')
        # Absolute time
        tda.loc[dict(feature="t_abs")] += ((treg - self.time_origin) / self.time_unit).astype('float64')
        # Calendar year
        year_cal = treg.dt.year
        tda.loc[dict(feature="year_cal")] += year_cal.astype('float64')
        # Calendar time
        year_start = np.array(
                [np.datetime64(f"{int(yc)}-01-01") for yc in year_cal.to_numpy().flat]
                ).reshape(year_cal.shape)
        
        tda.loc[dict(feature="t_cal")] += (treg - year_start) / self.time_unit  # TODO: make this agnostic to time units 
        # Time since most recent season start
        szn_start_same_year = np.array(
                [np.datetime64(
                    f"{int(yc)}-{self.szn_start['month']:02}-{self.szn_start['day']:02}"
                    ) for yc in year_cal.to_numpy().flat]
                ).reshape(year_cal.shape)
        year_szn_start = year_cal - 1*(treg < szn_start_same_year) #.to_numpy()
        tda.loc[dict(feature="year_szn_start")] += year_szn_start
        szn_start_most_recent = np.array(
                [np.datetime64(
                    f"{int(yss)}-{self.szn_start['month']:02}-{self.szn_start['day']:02}"
                    ) for yss in year_szn_start.to_numpy().flat]
                ).reshape(year_szn_start.shape)
        tda.loc[dict(feature="t_szn")] += (treg - szn_start_most_recent) / self.time_unit
        return tda
    # -----------------------------------------------
    # ------------ feat_all -------------------
    def preprocess_mfdataset(self, ds, src):
        # Only sample the beginning of each day 
        if src == "e5":
            day_start_idx, = np.where(ds["time"].dt.hour == 0)
            ds = ds.isel(time=day_start_idx)
        ds = ds.expand_dims({"t_init": ds['time'][0:1]})
        ds = ds.assign_coords({"time": (ds['time'] - ds['time'][0])/np.timedelta64(1,'D')})
        ds = ds.rename({"time": "t_sim"})
        return ds
    def compute_all_features_in_chunk(self, src, input_dir, input_file_list, featdef, obs2compute, dask_flag=False, print_timing_flag=False):
        obs_dict = dict({obsname: [] for obsname in obs2compute})
        input_path_list = [join(input_dir, f) for f in input_file_list]
        preprocess = lambda ds,src=src: self.preprocess_mfdataset(ds, src)
        if dask_flag:
            t0 = datetime.datetime.now()
            ds = xr.open_mfdataset(input_path_list, preprocess=preprocess, combine="nested", concat_dim="t_init")
            t1 = datetime.datetime.now()
            if "time_observable" in obs2compute:
                obs_dict["time_observable"] += [self.time_observable(ds, src)]
            t2 = datetime.datetime.now()
            if "ubar_observable" in obs2compute:
                obs_dict["ubar_observable"] += [self.ubar_observable(ds, src)]
            t3 = datetime.datetime.now()
            if "heatflux_observable" in obs2compute:
                obs_dict["heatflux_observable"] += [self.heatflux_observable(ds, src)]
            t4 = datetime.datetime.now()
            if "temperature_observable" in obs2compute:
                obs_dict["temperature_observable"] += [self.temperature_observable(ds, src)]
            t5 = datetime.datetime.now()
            if "pc_observable" in obs2compute:
                obs_dict["pc_observable"] += [self.pc_observable(ds, src, featdef["ds_eofs"], featdef["ds_monclim"])]
            t6 = datetime.datetime.now()
            if "qbo_observable" in obs2compute:
                obs_dict["qbo_observable"] += [self.qbo_observable(ds, src)]
            t8 = datetime.datetime.now()
        else: 
            for i_path in range(len(input_path_list)):
                t0 = datetime.datetime.now()
                ds = preprocess(xr.open_dataset(input_path_list[i_path]))
                t1 = datetime.datetime.now()
                if "time_observable" in obs2compute:
                    obs_dict["time_observable"] += [self.time_observable(ds, src)]
                t2 = datetime.datetime.now()
                if "ubar_observable" in obs2compute:
                    obs_dict["ubar_observable"] += [self.ubar_observable(ds, src)]
                t3 = datetime.datetime.now()
                if "heatflux_observable" in obs2compute:
                    obs_dict["heatflux_observable"] += [self.heatflux_observable(ds, src)]
                t4 = datetime.datetime.now()
                if "temperature_observable" in obs2compute:
                    obs_dict["temperature_observable"] += [self.temperature_observable(ds, src)]
                t5 = datetime.datetime.now()
                if "pc_observable" in obs2compute:
                    obs_dict["pc_observable"] += [self.pc_observable(ds, src, featdef["ds_eofs"], featdef["ds_monclim"])]
                t6 = datetime.datetime.now()
                if "qbo_observable" in obs2compute:
                    obs_dict["qbo_observable"] += [self.qbo_observable(ds, src)]
                t7 = datetime.datetime.now()
        ds.close()
        if print_timing_flag:
            print(f"\t----Times--- \n\topening file {t1-t0}\n\ttime obs {t2-t1}\n\tubar obs {t3-t2}\n\tvT obs {t4-t3}\n\ttemp obs {t5-t4}\n\tpc obs {t6-t5}\n\tqbo obs {t7-t6}")
        return obs_dict
    def compute_all_features(self, src, input_dir, input_file_list, output_dir, featdef, obs2compute=None, chunk_size=25, print_every=5):
        if obs2compute is None:
            obs2compute = [f"{obs}_observable" for obs in ["time","ubar", "pc", "temperature", "heatflux", "qbo"]]
        obs_dict = dict({obsname: [] for obsname in obs2compute})
        # Separate the computation into chunks 
        chunk_starts = np.arange(0, len(input_file_list), chunk_size)
        for i_chunk in range(len(chunk_starts)):
            print_flag = (i_chunk % print_every == 0)
            if print_flag:
                print(f"Beginning chunk {i_chunk}/{len(chunk_starts)}")
            chunk_o_files = input_file_list[chunk_starts[i_chunk]:min(chunk_starts[i_chunk]+chunk_size, len(input_file_list))]
            obs_dict_chunk = self.compute_all_features_in_chunk(src, input_dir, chunk_o_files, featdef, obs2compute, dask_flag=(chunk_size>1), print_timing_flag=print_flag)
            for obsname in obs2compute:
                obs_dict[obsname] += obs_dict_chunk[obsname]
        for obsname in obs2compute:
            ds_obs = xr.concat(obs_dict[obsname], dim='t_init').sortby("t_init")
            ds_obs.to_netcdf(join(output_dir, f"{obsname}.nc"))
            ds_obs.close()
        return 
    # --------------------------------------------------
    def assemble_all_features(self, output_dir, obs2assemble=None):
        if obs2assemble is None:
            obs2assemble = [f"{obs}_observable" for obs in ["time","ubar", "pc", "temperature", "heatflux", "qbo"]]
        feat_all = dict({
            obsname: xr.open_dataarray(join(output_dir, f"{obsname}.nc"))
            for obsname in [
                "time_observable", "ubar_observable", "pc_observable",
                "temperature_observable", "heatflux_observable", "qbo_observable"
                ]
            })
        return feat_all
    # ------------ Assemble feat_tpt dictionary ------------
    def assemble_tpt_features(self, feat_all, savedir):
        # List the features to put into feat_tpt
        # First, the features needed to define A and B: the time, the x1 coordinate, and its running mean, min, and max
        # over some time horizon. 
        num_time_delays = 20 # Units are days
        levels = [10, 100, 500, 850]
        pcs = [1, 2, 3, 4]
        modes = np.arange(4)
        feat_tpt_list = (
            ["t_abs","t_szn","year_szn_start","t_cal",] + 
            [f"ubar_{level}_60_delay{delay}" for level in levels for delay in range(num_time_delays+1)] + 
            [f"pc_{level}_{i_pc}" for level in levels for i_pc in pcs] + 
            [f"Tcap_{level}_60to90" for level in levels] + 
            [f"vT_{level}_{mode}_runavg{delay}" for level in levels for mode in modes for delay in range(num_time_delays+1)] + 
            [f"ubar_{level}_0pm5" for level in [10, 100]]
        )
        t_abs = feat_all["time_observable"].sel(feature="t_abs").to_numpy()
        t_sim = t_abs - t_abs[0]
        feat_tpt = xr.DataArray(
            coords = {
                "ensemble": np.arange(1),
                "member": np.arange(1,2),
                "t_sim":  t_sim,
                "feature": feat_tpt_list,
            },
            dims = ["ensemble","member","t_sim","feature"],
        )
        # Time observables
        t_names = ["t_abs","t_szn","year_szn_start","t_cal"]
        feat_tpt.loc[dict(feature=t_names)] = (
            feat_all["time_observable"].sel(feature=t_names)
        )
        # PC observables
        pc_names = [f"pc_{level}_{i_pc}" for level in levels for i_pc in pcs]
        feat_tpt.loc[dict(feature=pc_names)] = (
            feat_all["pc_observable"].sel(feature=pc_names)
        )
        # Time-delayed zonal wind observables
        for i_delay in range(num_time_delays+1):
            tidx_in = np.arange(num_time_delays-i_delay,len(t_abs)-i_delay)
            tidx_out = np.arange(num_time_delays,len(t_abs))
            ubar_names_in = [f"ubar_{level}_60" for level in levels]
            ubar_names_out = [f"ubar_{level}_60_delay{i_delay}" for level in levels]
            feat_tpt.loc[dict(feature=ubar_names_out,t_sim=t_sim[tidx_out])] = (
                feat_all["ubar_observable"].sel(feature=ubar_names_in)
                .isel(time=tidx_in).to_numpy()
            )
        # Temperature observables 
        temp_names = [f"Tcap_{level}_60to90" for level in levels]
        feat_tpt.loc[dict(feature=temp_names)] = (
            feat_all["temperature_observable"].sel(feature=temp_names)
        )
        # Heat flux observables
        for i_delay in range(num_time_delays+1):
            vT_names_in = [f"vT_{level}_{mode}" for level in levels for mode in modes]
            vT_names_out = [f"vT_{level}_{mode}_runavg{i_delay}" for level in levels for mode in modes]
            feat_tpt.loc[dict(feature=vT_names_out)] = (
                feat_all["heatflux_observable"].sel(feature=vT_names_in)
                .rolling(time=i_delay+1, center=False).mean().to_numpy()
            )
        # QBO observable
        qbo_names = [f"ubar_{level}_0pm5" for level in [10, 100]]
        feat_tpt.loc[dict(feature=qbo_names)] = (
            feat_all["qbo_observable"].sel(feature=qbo_names)
        )
        # Save
        feat_tpt.to_netcdf(join(savedir, "features_tpt.nc"))
        return feat_tpt
    def ab_test(self, Xtpt):
        time_window_flag = (
                (Xtpt.sel(feature="t_szn") >= self.tpt_bndy["t_thresh"][0]) &  
                (Xtpt.sel(feature="t_szn") <= self.tpt_bndy["t_thresh"][1])
                )
        weak_vortex_flag = (Xtpt.sel(feature="ubar_10_60_delay0") <= self.tpt_bndy["ubar_10_60_thresh"])
        ab_tag = (
                self.ab_code["A"]*(~time_window_flag) + 
                self.ab_code["B"]*(time_window_flag & weak_vortex_flag) + 
                self.ab_code["D"]*(time_window_flag & ~weak_vortex_flag)
                )
        return ab_tag
    def hours_since_oct1(self,ds):
        # Given the time from a dataset, convert the number to time in days since the most recent November 1
        dstime = ds['time']
        Nt = dstime.size
        date = nc.num2date(dstime[:],dstime.units,dstime.calendar)
        year = np.array([date[i].year for i in range(Nt)])
        month = np.array([date[i].month for i in range(Nt)])
        oct1_year = year*(month >= 10) + (year-1)*(month < 10)
        oct1_date = np.array([datetime.datetime(oct1_year[i], 10, 1) for i in range(Nt)])
        oct1_time = np.array([nc.date2num(oct1_date[i],dstime.units,dstime.calendar) for i in range(Nt)])
        #ensemble_size = ds['number'].size
        #dstime_adj = np.outer(np.ones(ensemble_size), (dstime - oct1_time)/24.0)
        dstime_adj = dstime - oct1_time 
        return dstime_adj # This is just one-dimensional. 
    def get_ilev_ilat(self,ds):
        # Get the latitude and longitude indices
        ds_plev_hPa = ds['plev'][:]
        if ds['plev'].units == 'Pa':
            ds_plev_hPa *= 1.0/100
        i_lev = np.argmin(np.abs(self.pres_uref - ds_plev_hPa))
        i_lat = np.argmin(np.abs(self.lat_uref - ds['lat'][:]))
        return i_lev,i_lat
    def get_ensemble_source_size(self,ds):
        vbls = list(ds.variables.keys())
        if 'var131' in vbls: # This means it's from era20c OR eraint OR era5
            dssource = 'era'
            Nmem = 1
        elif 'gh' in vbls:
            dssource = 's2s'
            Nmem = 0
            for v in vbls:
                if v[:2] == "gh":
                    Nmem += 1
        return dssource,Nmem
    def get_u_gh_from_filename(self,ds_filename):
        ds = nc.Dataset(ds_filename,"r")
        ugh = self.get_u_gh(ds)
        ds.close()
        return ugh
    def get_u_gh(self,ds):
        # All features will follow from this. 
        dssource,Nmem = self.get_ensemble_source_size(ds)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ["time","plev","lat","lon"]]
        grav_accel = 9.80665
        gh = np.zeros((Nmem,Nt,Nlev,Nlat,Nlon))
        u = np.zeros((Nmem,Nt,Nlev,Nlat,Nlon))
        for i_mem in range(Nmem):
            if dssource == 's2s':
                memkey_gh = 'gh' if i_mem==0 else 'gh_%i'%(i_mem+1)
                gh[i_mem] = ds[memkey_gh][:]
                memkey_u = 'u' if i_mem==0 else 'u_%i'%(i_mem+1)
                u[i_mem] = ds[memkey_u][:]
                ghflag = True
            elif dssource == 'era': 
                u[i_mem] = ds['var131'][:]
                if 'var129' in ds.variables.keys():
                    gh[i_mem] = ds['var129'][:]/grav_accel
                    ghflag = True
                else:
                    gh[i_mem] = np.nan*np.ones((Nt,Nlev,Nlat,Nlon)) # This is for ERA5
                    ghflag = False
            else:
                raise Exception("The dssource you gave me, %s, is not recognized"%(dssource))
        time,fall_year = self.time_since_oct1(ds['time'])
        return gh,u,time,ds['plev'][:],ds['lat'][:],ds['lon'][:],fall_year,ghflag
    def compute_geostrophic_wind(self,gh,lat,lon):
        # gh shape should be (Nx,Nlev,Nlat,Nlon). 
        Omega = 2*np.pi/(3600*24*365)
        fcor = np.outer(2*Omega*np.sin(lat*np.pi/180), np.ones(lon.size))
        fcor_pole = np.max(fcor)
        fcor_eps = 2*Omega*np.sin(10.0*np.pi/180) # 10 degrees N should be the highest altitude where geostrophic balance is assumed. We won't even consider lower laitudes.
        earth_radius = 6371e3 
        grav_accel = 9.80665
        dx = np.outer(earth_radius*np.cos(lat*np.pi/180), np.roll(lon,-1) - np.roll(lon,1)) # this counts both sides
        dy = np.outer((np.roll(lat,-1) - np.roll(lat,1))*earth_radius, np.ones(lon.size))
        gh_x = (np.roll(gh,-1,axis=3) - np.roll(gh,1,axis=3))/dx 
        gh_y = (np.roll(gh,-1,axis=2) - np.roll(gh,1,axis=2))/dy
        u = -gh_y*(np.abs(fcor) > fcor_eps)/(fcor + 1.0*(np.abs(fcor) < fcor_eps)) * grav_accel
        v = gh_x*(np.abs(fcor) > fcor_eps)/(fcor + 1.0*(np.abs(fcor) < fcor_eps)) * grav_accel
        return u,v
    def get_temperature(self,gh,plev,lat,lon):
        # Use the hypsometric law: d(gz)/dp = -RT/p
        grav_accel = 9.80665
        ideal_gas_const = 287.0 # J / (kg.K)
        Nx,Nlev,Nlat,Nlon = gh.shape
        dgh_dlnp = np.zeros((Nx,Nlev,Nlat,Nlon))
        dgh_dlnp[:,0] = (gh[:,1] - gh[:,0])/np.log(plev[1]/plev[0])
        dgh_dlnp[:,-1] = (gh[:,-1] - gh[:,-2])/np.log(plev[-1]/plev[-2])
        for i_lev in range(1,Nlev-1):
            dgh_dlnp[:,i_lev] = (gh[:,i_lev+1] - gh[:,i_lev-1])/np.log(plev[i_lev+1]/plev[i_lev-1])
        T = -dgh_dlnp*grav_accel / ideal_gas_const
        print(f"T: min={T.min()}, max={T.max()}")
        if T.min() < 0:
            print(f"max diff gh = {np.max(np.diff(gh,axis=1))}")
            print(f"min log p ratio = {np.min(np.log(plev[1:]/plev[:-1]))}")
            raise Exception(f"ERROR: negative temperature: min(T) = {T.min()}")
        return T
    def get_meridional_heat_flux(self,gh,temperature,plev,lat,lon): # Returns average between 45N and 75N 
        Nx,Nlev,Nlat,Nlon = gh.shape
        cosine = np.outer(np.cos(lat*np.pi/180), np.ones(Nlon))
        imin,imax = np.argmin(np.abs(lat-75)),np.argmin(np.abs(lat-45))
        _,vmer = self.compute_geostrophic_wind(gh,lat,lon)
        T_bandavg = np.sum((temperature*cosine)[:,:,imin:imax,:], axis=2)/np.sum(cosine[imin:imax,:], axis=0)
        vmer_bandavg = np.sum((vmer*cosine)[:,:,imin:imax,:], axis=2)/np.sum(cosine[imin:imax,:], axis=0) 
        That = 1/Nlon*np.abs(np.fft.rfft(T_bandavg, axis=2)[:,:,:self.heatflux_wavenumbers_per_level_max])
        vhat = 1/Nlon*np.abs(np.fft.rfft(vmer_bandavg, axis=2)[:,:,:self.heatflux_wavenumbers_per_level_max])
        # Now extract wavenumbers one at a time
        vT_decomp = np.zeros((Nx,Nlev,self.heatflux_wavenumbers_per_level_max))
        vT_decomp[:,:,0] = That[:,:,0]*vhat[:,:,0]
        for k in range(1,self.heatflux_wavenumbers_per_level_max):
            vT_decomp[:,:,k] = 2*(vhat[:,:,k]*That[:,:,k].conjugate()).real
        return vT_decomp
    def classify_split_displacement(self,gh,lat,lon):
        # Compute the split vs. displacement criterion from cp07
        return
    def spherical_horizontal_laplacian(self,field,lat,lon):
        # Compute the spherical Laplacian of a field on a lat-lon grid. Assume unit sphere.
        Nx,Nlev,Nlat,Nlon = field.shape
        if Nlat != len(lat) or Nlon != len(lon):
            raise Exception(f"ERROR: Shape mismatch in spherical_laplacian. len(lat) = {len(lat)} and len(lon) = {len(lon)}, while field.shape = {field.shape}.")
        dlat = np.pi/180 * (lat[1] - lat[0])
        dlon = np.pi/180 * (lon[1] - lon[0])
        field_lon2 = (np.roll(field,-1,axis=3) - 2*field + np.roll(field,1,axis=3))/dlon**2
        field_lat2 = (np.roll(field,-1,axis=2) - 2*field + np.roll(field,1,axis=2))/dlat**2
        field_lat2[:,:,0,:] = field_lat2[:,:,1,:]
        field_lat2[:,:,-1,:] = field_lat2[:,:,-2,:]
        field_lat = (np.roll(field,-1,axis=2) - np.roll(field,1,axis=2))/(2*dlat)
        field_lat[:,:,0,:] = (-3*field[:,:,0,:] + 4*field[:,:,1,:] - field[:,:,2,:])/(2*dlat)
        field_lat[:,:,-1,:] = (3*field[:,:,-1,:] - 4*field[:,:,-2,:] + field[:,:,-3,:])/(2*dlat)
        cos = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))
        sin = np.outer(np.sin(lat*np.pi/180), np.ones(len(lon)))
        # Make poles nan
        cos[np.abs(cos)<1e-3] = np.nan
        sin[np.abs(sin)<1e-3] = np.nan
        lap_f = 1.0/cos**2*field_lon2 - (sin/cos)*field_lat + field_lat2
        return lap_f,cos,sin,field_lat,field_lat2,field_lon2
    def smooth_spherical_field(self,field,lat,lon,nT=11):
        # Smooth a field using a spherical harmonics truncation
        return
    def compute_qgpv(self,gh,lat,lon):
        # gh shape should be (Nx, Nlev,Nlat,Nlon)
        # Quasigeostrophic potential vorticity: just do horizontal component for now
        # QGPV = (g/f)*(laplacian(gh) - d(gh)/dy * beta/f) + f
        #      = (g/f)*(laplacian(gh) - 1/(earth radius)**2 * cos(lat)/sin(lat) * d(gh)/dlat) + f
        gh_lap,cos,sin,gh_lat,gh_lat2,gh_lon2 = self.spherical_horizontal_laplacian(gh,lat,lon)
        Omega = 2*np.pi/(3600*24*365)
        fcor = np.outer(2*Omega*np.sin(lat*np.pi/180), np.ones(lon.size))
        earth_radius = 6371.0e3 
        grav_accel = 9.80665
        qgpv = fcor + grav_accel/(fcor*earth_radius**2)*(gh_lap - (cos/sin)*gh_lat)
        #Nx,Nlev,Nlat,Nlon = gh.shape
        #dlat = np.pi/180 * (lat[1] - lat[0])
        #dlon = np.pi/180 * (lon[1] - lon[0])
        #gh_lon2 = (np.roll(gh,-1,axis=3) - 2*gh + np.roll(gh,1,axis=3))/dlon**2
        #gh_lat2 = (np.roll(gh,-1,axis=2) - 2*gh + np.roll(gh,1,axis=2))/dlat**2
        #gh_lat2[:,:,0,:] = gh_lat2[:,:,1,:]
        #gh_lat2[:,:,-1,:] = gh_lat2[:,:,-2,:]
        #gh_lat = (np.roll(gh,-1,axis=2) - np.roll(gh,1,axis=2))/(2*dlat)
        #gh_lat[:,:,0,:] = (-3*gh[:,:,0,:] + 4*gh[:,:,1,:] - gh[:,:,2,:])/(2*dlat)
        #gh_lat[:,:,-1,:] = (3*gh[:,:,-1,:] - 4*gh[:,:,-2,:] + gh[:,:,-3,:])/(2*dlat)
        #cos = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))
        #sin = np.outer(np.sin(lat*np.pi/180), np.ones(len(lon)))
        ## Make poles nan
        #cos[np.abs(cos)<1e-3] = np.nan
        #sin[np.abs(sin)<1e-3] = np.nan
        #qgpv = fcor + grav_accel/(fcor*earth_radius**2)*(
        #        1.0/cos**2*gh_lon2 - (sin/cos + cos/sin)*gh_lat + gh_lat2)
        return qgpv
    def compute_vortex_moments_sphere(self,gh,lat,lon,i_lev_subset=None,num_vortex_moments=4):
        # Do the calculation in lat/lon coordinates. Regridding is too expensive
        Nsamp,Nlev_full,Nlat_full,Nlon = gh.shape
        dlat,dlon = np.pi/2*np.array([lat[1]-lat[0],lon[1]-lon[0]])
        if i_lev_subset is None:
            i_lev_subset = np.arange(Nlev_full)
        Nlev = len(i_lev_subset)
        i_lat_max = np.where(lat < 30.0)[0][0]  # All the way to the equator
        print(f"i_lat_max = {i_lat_max}")
        Nlat = i_lat_max # - 2
        stereo_factor = np.cos(lat[:i_lat_max]*np.pi/180)/(1 + np.sin(lat[:i_lat_max]*np.pi/180))
        X = np.outer(stereo_factor, np.cos(lon*np.pi/180)).flatten()
        Y = np.outer(stereo_factor, np.sin(lon*np.pi/180)).flatten()
        qgpv = self.compute_qgpv(gh,lat,lon)[:,i_lev_subset,:i_lat_max,:] #.reshape((Nsamp*Nlev,Nlat*Nlon))
        print(f"qgpv: nanfrac={np.mean(np.isnan(qgpv))}, min={np.nanmin(qgpv)}, max={np.nanmax(qgpv)}")
        # Smooth the qgpv field before finding vortex edge
        qgpv_filter_degrees = 20.0 # degrees
        qgpv_filter_pixels = int(qgpv_filter_degrees / dlon)
        qgpv_smoothed = np.zeros(qgpv.shape)
        for j in range(-qgpv_filter_pixels//2,qgpv_filter_pixels//2):
            qgpv_smoothed += np.roll(qgpv,j,axis=3)/qgpv_filter_pixels
        qgpv = qgpv.reshape((Nsamp*Nlev,Nlat*Nlon))
        qgpv_smoothed = qgpv_smoothed.reshape((Nsamp*Nlev,Nlat*Nlon))
        # Assign an area to each grid cell. 
        area_factor = np.outer(np.cos(lat[:i_lat_max]*np.pi/180), np.ones(Nlon)).flatten()
        # Find vortex edge by ranking grid cells and finding the maximum slope of area fraction with respect to PV
        qgpv_order = np.argsort(qgpv_smoothed,axis=1)
        area_fraction = np.cumsum(np.array([area_factor[qgpv_order[i]] for i in range(Nsamp*Nlev)]),axis=1)
        area_fraction = (area_fraction.T /area_fraction[:,-1]).T
        qgpv_sorted = np.array([qgpv_smoothed[i,qgpv_order[i]] for i in np.arange(Nsamp*Nlev)])
        equiv_lat = np.arcsin(area_fraction)
        # Verify qgpv_sorted is monotonic with equiv_lat
        print(f"min diff = {np.nanmin(np.diff(qgpv_sorted, axis=1))}")
        if np.nanmin(np.diff(qgpv_sorted, axis=1)) < 0:
            raise Exception("qgpv_sorted must be monotonically increasing")
        window = 30
        dq_deqlat = (qgpv_sorted[:,window:] - qgpv_sorted[:,:-window])/(equiv_lat[:,window:] - equiv_lat[:,:-window])
        #idx_crit = np.nanargmax(dq_deqlat, axis=1) + window//2
        idx_crit = np.argmin(np.abs(area_fraction - 0.8), axis=1) #int(dA_dq.shape[1]/2) 
        #qgpv_crit = qgpv_sorted[np.arange(Nsamp*Nlev),idx_crit]
        qgpv_crit = np.zeros(Nsamp*Nlev) # Only count positive QGPV
        print(f"qgpv_crit: min={np.nanmin(qgpv_crit)}, max={np.nanmax(qgpv_crit)}")
        # Threshold and find moments
        q = (np.maximum(0, qgpv.T - qgpv_crit).T)
        print(f"q: frac>0 is {np.mean(q>0)},  min={np.nanmin(q)}, max={np.nanmax(q)}")
        moments = {}
        for i_mom in range(num_vortex_moments+1):
            key = f"m{i_mom}" # Normalized moments
            moments[key] = np.zeros((Nsamp*Nlev,i_mom+1))
            for j in range(i_mom+1):
                moments[key][:,j] = np.nansum(area_factor*q*(X**j)*Y**(i_mom-j), axis=1) 
            if i_mom == 1:
                Ybar = moments['m1'][:,0]/moments['m0'][:,0]
                Xbar = moments['m1'][:,1]/moments['m0'][:,0]
                Xcent = np.add.outer(-Xbar, X)
                Ycent = np.add.outer(-Ybar, Y)
                print(f"Xcent.shape = {Xcent.shape}, q.shape = {q.shape}")
            if i_mom >= 2:
                key = f"J{i_mom}" # Centralized moments
                moments[key] = np.zeros((Nsamp*Nlev,i_mom+1))
                for j in range(i_mom+1):
                    moments[key][:,j] = np.nansum(area_factor*q*Xcent**j*Ycent**(i_mom-j), axis=1)
        # Normalize the moments
        moments['area'] = np.nansum(area_factor*q, axis=1) #moments['m0'][:,0]
        if num_vortex_moments >= 1:
            moments['centerlat'] = np.arcsin((1 - (Xbar**2+Ybar**2))/(1 + (Xbar**2+Ybar**2))) * 180/np.pi
            moments['centerlon'] = np.arctan2(Ybar,Xbar) * 180/np.pi
        if num_vortex_moments >= 2:
            J02,J11,J20 = moments['J2'].T
            term0 = J20 + J02
            term1 = np.sqrt(4*J11**2 + (J20-J02)**2)
            moments['aspect_ratio'] = np.sqrt((term0 + term1)/(term0 - term1))
        if num_vortex_moments >= 4:
            J04,J13,J22,J31,J40 = moments['J4'].T
            r = moments['aspect_ratio']
            moments['excess_kurtosis'] = (J40+J02+2*J22)/(J20+J02)**2 - 2/(3*moments['m0'][0])*(3*r**4+2*r**2+3)/(r**4+2*r**2+1)
        #print(f"m00: min={np.nanmin(m00)}, max={np.nanmax(m00)}")
        #area = m00 #/ qgpv_crit
        ##area = np.sum((q>0)*area_factor, axis=1)
        ## First moment: mean x and mean y
        #m10 = np.nansum(area_factor*q*X, axis=1)/m00
        #m01 = np.nansum(area_factor*q*Y, axis=1)/m00
        #print(f"m10: min={np.nanmin(m10)}, max={np.nanmax(m10)}")
        #print(f"m01: min={np.nanmin(m01)}, max={np.nanmax(m01)}")
        #center = np.array([m10, m01]).T
        ## Determine latitude and longitude of center
        #center_lat = np.arcsin((1 - (center[:,0]**2 + center[:,1]**2))/(1 + (center[:,0]**2 + center[:,1]**2))) * 180/np.pi
        #center_lon = np.arctan2(center[:,1],center[:,0]) * 180/np.pi
        ## Reshape
        #area = area.reshape((Nsamp,Nlev))
        #center_latlon = np.array([center_lat,center_lon]).T
        ##center = center.reshape((Nsamp,Nlev,2))
        #print(f"area: min={np.nanmin(area)}, max={np.nanmax(area)}, mean={np.nanmean(area)}\ncenter(x): min={np.nanmin(center[:,0])}, max={np.nanmax(center[:,0])}, mean={np.nanmean(center[:,0])}")
        return moments
    def get_wavenumbers(self,gh,i_lev,lat_range,lat,lon):
        # Given a band of latitudes (a whole ensemble thereof), get waves 1 and 2
        i_lat_range = np.where((lat >= lat_range[0])*(lat <= lat_range[1]))[0]
        #print("i_lat_range = {}".format(i_lat_range))
        cosine = np.cos(lat[i_lat_range] * np.pi/180)
        cosweight = cosine/np.sum(cosine)
        #print("gh.shape = {}".format(gh.shape))
        fft = np.fft.fft(gh[:,i_lev,i_lat_range,:],axis=2)
        wave = np.zeros((gh.shape[0], 2*self.num_wavenumbers)) # real 1, imag 1, real 2, imag 2, ...
        for i in range(self.num_wavenumbers):
            wave[:,2*i] = fft[:,:,i+1].real.dot(cosweight)
            wave[:,2*i+1] = fft[:,:,i+1].imag.dot(cosweight)
        return wave
    def get_seasonal_mean(self,t_szn,field):
        # Given a field whose first dimension is time, deseasonalize the field
        if len(t_szn) != field.shape[0]:
            raise Exception("You gave me a t_szn of length {}, whereas the field has shape {}. First dimensions must match".format(len(t_szn),field.shape))
        field_szn_mean_shape = np.array(field.shape).copy()
        field_szn_mean_shape[0] = self.Ntwint
        field_szn_mean = np.zeros(field_szn_mean_shape)
        field_szn_std = np.zeros(field_szn_mean_shape)
        for i_time in range(self.Ntwint):
            idx = np.where(np.abs(t_szn - self.wtime[i_time]) < self.szn_hour_window)[0]
            field_szn_mean[i_time] = np.mean(field[idx],axis=0)
            field_szn_std[i_time] = np.std(field[idx],axis=0)
        return field_szn_mean,field_szn_std
    def unseason(self,t_field,field,field_szn_mean,field_szn_std,normalize=True,delayed=False):
        wtime = self.wtime_delayed if delayed else self.wtime
        wti = ((t_field - wtime[0])/self.dtwint).astype(int)
        wti = np.maximum(0, np.minimum(len(wtime)-1, wti))
        field_unseasoned = field - field_szn_mean[wti]
        if normalize:
            field_unseasoned *= 1.0/field_szn_std[wti]
        return field_unseasoned
    def reseason(self,t_field,field_unseasoned,t_szn,field_szn_mean,field_szn_std,delayed=True):
        #print("t_field.shape = {}, field_unseasoned.shape = {}, t_szn.shape = {}, field_szn_mean.shape = {}, field_szn_std.shape = {}".format(t_field.shape, field_unseasoned.shape, t_szn.shape, field_szn_mean.shape, field_szn_std.shape))
        wtime = self.wtime_delayed if delayed else self.wtime
        wti = ((t_field - wtime[0])/self.dtwint).astype(int)
        wti = np.maximum(0, np.minimum(len(wtime)-1, wti))
        #print("field_szn_std[wti].shape = {}, field_unseasoned.shape = {}, field_szn_mean[wti].shape = {}".format(field_szn_std[wti].shape,field_unseasoned.shape,field_szn_mean[wti].shape))
        field = field_szn_std[wti] * field_unseasoned + field_szn_mean[wti]
        return field
    def time_since_oct1(self,dstime):
        Nt = dstime.size
        date = nc.num2date(dstime[0],dstime.units,dstime.calendar)
        #print(f"dstime.units = {dstime.units}, dstime.calendar = {dstime.calendar}")
        year = date.year 
        month = date.month
        oct1_year = year*(month >= 10) + (year-1)*(month < 10)
        #print(f"oct1_year = {oct1_year}")
        oct1_date = datetime.datetime(oct1_year, 10, 1)
        #print(f"oct1_date = {oct1_date}")
        oct1_time = nc.date2num(oct1_date,dstime.units,dstime.calendar)
        #print(f"oct1_time = {oct1_time}")
        dstime_adj = dstime - oct1_time
        #print(f"dstime_adj before adding date = {dstime_adj}")
        #dstime_adj += nc.date2num(date,dstime.units,dstime.calendar)  
        #print(f"dstime_adj after adding date = {dstime_adj}")
        return dstime_adj,oct1_year # This is just one-dimensional. 
    def create_features(self,data_file_list):
        # Use data in data_file_list as training, and dump the results into feature_file. Note this is NOT a DGA basis yet, just a set of features.
        # Time-delay embedding happens not at this stage, but possibly at the next stage when creating DGA features.
        # Mark the boundary of ensembles with a list of indices.
        ds0 = nc.Dataset(data_file_list[0],"r")
        Nlev,Nlat,Nlon = [ds0[v].size for v in ["plev","lat","lon"]]
        plev,lat,lon = [ds0[v][:] for v in ["plev","lat","lon"]]
        i_lev_uref,i_lat_uref = self.get_ilev_ilat(ds0)
        ds0.close()
        # ---------------- Build up big arrays of gh, u, and t_szn ---------------------
        reading_start = timelib.time()
        gh = np.zeros((0,Nlev,Nlat,Nlon)) # First dimension will have both time and ensemble members
        u = np.zeros((0,Nlev,Nlat,Nlon)) 
        t_szn = np.zeros(0)
        grid_shp = np.array([Nlev,Nlat,Nlon])
        for i_file in range(len(data_file_list)):
            print("Creating features: file {} out of {}".format(i_file,len(data_file_list)))
            ds = nc.Dataset(data_file_list[i_file],"r")
            gh_new,u_new,time,_,_,_,_,_ = self.get_u_gh(ds)
            Nmem,Nt = gh_new.shape[:2]
            shp_new = np.array(gh_new.shape)
            if np.any(shp_new[2:5] != grid_shp):
                raise Exception("The file {} has a geopotential height field of shape {}, whereas it was supposed to have a shape {}".format(data_file_list[i_file],shp_new[2:5],grid_shp))
            gh = np.concatenate((gh,gh_new.reshape((Nmem*Nt,Nlev,Nlat,Nlon))),axis=0)
            u = np.concatenate((u,u_new.reshape((Nmem*Nt,Nlev,Nlat,Nlon))),axis=0)
            t_szn = np.concatenate((t_szn,self.hours_since_oct1(ds)))
            ds.close()
        reading_duration = timelib.time() - reading_start
        print(f"Reading duration = {reading_duration}")
        # Vortex moment diagnostics, only at reference level
        vtx_moments = self.compute_vortex_moments_sphere(gh,lat,lon,i_lev_subset=[i_lev_uref])
        vtx_diags = np.array([vtx_moments[v] for v in ['area','centerlat','aspect_ratio','excess_kurtosis']]).T
        vtx_diags_szn_mean,vtx_diags_szn_std = self.get_seasonal_mean(t_szn,vtx_diags)
        # Zonal wind
        #u,v = self.compute_geostrophic_wind(gh,lat,lon)
        uref = np.mean(u[:,i_lev_uref,i_lat_uref,:],axis=1)
        uref_szn_mean,uref_szn_std = self.get_seasonal_mean(t_szn,uref)
        uref_mean = np.mean(uref)
        uref_std = np.std(uref)
        # Waves 1 and 2
        waves = self.get_wavenumbers(gh,i_lev_uref,self.lat_range_uref,lat,lon)
        print("waves.shape = {}".format(waves.shape))
        waves_szn_mean,waves_szn_std = self.get_seasonal_mean(t_szn,waves)
        wave_mag = np.sqrt(waves[:,np.arange(0,2*self.num_wavenumbers-1,2)]**2 + waves[:,np.arange(1,2*self.num_wavenumbers,2)]**2)
        wave_mag_szn_mean,wave_mag_szn_std = self.get_seasonal_mean(t_szn,wave_mag)
        print("wave_mag.shape = {}".format(wave_mag.shape))
        print("waves_szn_mean.shape = {}, waves_szn_std.shape = {}".format(waves_szn_mean.shape,waves_szn_std.shape))
        # EOFs level by level
        Nlat_nh = np.argmin(np.abs(lat - 0.0)) # Equator
        gh_szn_mean,gh_szn_std = self.get_seasonal_mean(t_szn,gh)
        gh_unseasoned = self.unseason(t_szn,gh,gh_szn_mean,gh_szn_std,normalize=False)
        cosine = np.cos(np.pi/180 * lat[:Nlat_nh])
        weight = 1/np.sqrt(len(gh_unseasoned))*np.outer(np.sqrt(cosine),np.ones(Nlon)).flatten()
        eofs = np.zeros((Nlev,Nlat_nh*Nlon,self.Npc_per_level_max))
        singvals = np.zeros((Nlev,self.Npc_per_level_max))
        tot_var = np.zeros(Nlev)
        svd_start = timelib.time()
        for i_lev in range(Nlev):
            print("svd'ing level %i out of %i"%(i_lev,Nlev))
            U,S,Vh = np.linalg.svd((gh_unseasoned[:,i_lev,:Nlat_nh,:].reshape((len(gh),Nlat_nh*Nlon))*weight).T, full_matrices=False)
            eofs[i_lev,:,:] = U[:,:self.Npc_per_level_max]
            singvals[i_lev,:] = S[:self.Npc_per_level_max]
            tot_var[i_lev] = np.sum(S**2)
        svd_duration = timelib.time() - svd_start
        # Temperature: first compute cap average, then deseasonalize
        temperature = self.get_temperature(gh,plev,lat,lon)
        i_lat_cap = np.argmin(np.abs(lat - 60))
        area_factor = np.outer(np.cos(lat*np.pi/180), np.ones(Nlon))
        temp_capavg = np.sum((temperature*area_factor)[:,:,:i_lat_cap,:], axis=(2,3))/np.sum(area_factor[:i_lat_cap,:])
        print(f"temp_capavg: min={np.min(temp_capavg)}, max={np.max(temp_capavg)}")
        temp_capavg_szn_mean,temp_capavg_szn_std = self.get_seasonal_mean(t_szn,temp_capavg)
        vT = self.get_meridional_heat_flux(gh,temperature,plev,lat,lon) 
        vT_szn_mean,vT_szn_std = self.get_seasonal_mean(t_szn,vT)
        print(f"vT_szn_std: shape = {vT_szn_std.shape}, min={vT_szn_std.min()}, max={vT_szn_std.max()}")
        feat_def = {
                "t_szn": t_szn, "plev": plev, "lat": lat, "lon": lon,
                "i_lev_uref": i_lev_uref, "i_lat_uref": i_lat_uref, "Nlat_nh": Nlat_nh,
                "uref_mean": uref_mean, "uref_std": uref_std,
                "uref_szn_mean": uref_szn_mean, "uref_szn_std": uref_szn_std,
                "waves_szn_mean": waves_szn_mean, "waves_szn_std": waves_szn_std,
                "wave_mag_szn_mean": wave_mag_szn_mean, "wave_mag_szn_std": wave_mag_szn_std,
                "gh_szn_mean": gh_szn_mean, "gh_szn_std": gh_szn_std,
                "eofs": eofs, "singvals": singvals, "tot_var": tot_var, "Nlat_nh": Nlat_nh,
                "vtx_diags_szn_mean": vtx_diags_szn_mean, "vtx_diags_szn_std": vtx_diags_szn_std,
                "temp_capavg_szn_mean": temp_capavg_szn_mean, "temp_capavg_szn_std": temp_capavg_szn_std,
                "vT_szn_mean": vT_szn_mean, "vT_szn_std": vT_szn_std,
                }
        pickle.dump(feat_def,open(self.feature_file,"wb"))
        return
    def evaluate_features_database(self,file_list,feat_def,feat_filename,ens_start_filename,fall_year_filename,tmin,tmax):
        # Stack a bunch of forecasts together. They can start at different times, but must all have same length.
        ens_start_idx = np.zeros(len(file_list), dtype=int)
        fall_year_list = np.zeros(len(file_list), dtype=int)
        i_ens = 0
        for i in range(len(file_list)):
            if i % 1 == 0:
                print("file %i out of %i: %s"%(i,len(file_list),file_list[i]))
            ens_start_idx[i] = i_ens
            ds = nc.Dataset(file_list[i],"r")
            Xnew,fall_year = self.evaluate_features(ds,feat_def)
            print(f"Xnew.shape = {Xnew.shape}, fall_year = {fall_year}")
            fall_year_list[i] = fall_year
            # New: we subtract off the first time, assuming all time arrays start at zero
            ti_initial = np.where(ds['time'][:]-ds['time'][0] >= tmin)[0][0]
            ti_final = np.where(ds['time'][:]-ds['time'][0] <= tmax)[0][-1]
            #print(f"ds['time'][:] = {ds['time'][:]}")
            #print(f"Xnew[0,:,0] = {Xnew[0,:,0]}")
            #print(f"ti_initial = {ti_initial}, ti_final = {ti_final}")
            Xnew = Xnew[:,ti_initial:ti_final+1,:]
            if i == 0:
                X = Xnew.copy()
            else:
                X = np.concatenate((X,Xnew),axis=0)
            i_ens += Xnew.shape[0]
            ds.close()
        # Save them in the directory
        np.save(feat_filename,X)
        np.save(ens_start_filename,ens_start_idx)
        np.save(fall_year_filename,fall_year_list)
        return X
    def evaluate_tpt_features(self,feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=False,fy_resamp=None):
        print(f" -------------- Inside evaluate_tpt_features: tpt_feat_filename = {tpt_feat_filename}, resample_flag = {resample_flag} --------------")
        #print(f"fy_resamp = {fy_resamp}")
        # Evaluate a subset of the full features to use for clustering TPT.
        # A normalized version of these will be used for clustering.
        # The data set for clustering will have fewer time steps, due to time-delay embedding.
        X = np.load(feat_filename)
        #print("Before resampling: X.shape = {}".format(X.shape))
        if resample_flag:
            ens_start_idx = np.load(ens_start_filename)
            fall_year_list = np.load(fall_year_filename)
            fall_year_x = np.zeros(len(X), dtype=int)
            for i in range(len(ens_start_idx)):
                if i < len(ens_start_idx)-1:
                    ens_size = ens_start_idx[i+1] - ens_start_idx[i]
                else:
                    ens_size = len(X) - ens_start_idx[i]
                fall_year_x[ens_start_idx[i]:ens_start_idx[i]+ens_size] = fall_year_list[i]
            idx_resamp = np.zeros(0, dtype=int)
            for i in range(len(fy_resamp)):
                matches = np.where(fall_year_x == fy_resamp[i])[0]
                idx_resamp = np.concatenate((idx_resamp,np.sort(matches)))
            X = X[idx_resamp]
        #print("len(idx_resamp) = {}".format(len(idx_resamp)))
        #print("After resampling: X.shape = {}".format(X.shape))
        Nlev = len(feat_def['plev'])
        Nx,Ntx,xdim = X.shape
        #print(f"Nx = {Nx}, Ntx = {Ntx}, xdim = {xdim}")
        # ------------- Define the cluster features Y ------------------
        # Y will have time-delay features built in. 
        Nty = Ntx - self.ndelay + 1
        ydim = len(set(self.fidx_Y.values()))
        Y = np.zeros((Nx,Nty,ydim))
        # Store information to unseason Y, simply as a set of seasonal means, one per column.
        szn_mean_Y = np.zeros((self.Ntwint-self.ndelay+1,ydim-1))
        szn_std_Y = np.zeros((self.Ntwint-self.ndelay+1,ydim-1))
        # ------------- Time ---------------
        Y[:,:,self.fidx_Y["time_h"]] = X[:,self.ndelay-1:,self.fidx_X["time_h"]]
        #print(f"Y[0,0,0] = {Y[0,0,0]}")
        # ------------ Uref ------------------
        # Build time delays of u into Y
        i_feat_x = self.fidx_X["uref"]
        for i_dl in range(self.ndelay):
            i_feat_y = self.fidx_Y["uref_dl%i"%(i_dl)]
            #Y[:,:,i_feat_y] = X[:,i_dl:i_dl+Nty,i_feat_x]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1-i_dl:self.ndelay-1-i_dl+Nty,i_feat_x]
            #print(f"szn_mean_Y.shape = {szn_mean_Y.shape}")
            #print(f"feat_def['uref_szn_mean'].shape = {feat_def['uref_szn_mean'].shape}")
            #szn_mean_Y[:,i_feat_y-1] = feat_def["uref_szn_mean"][self.ndelay-1:] 
            #szn_std_Y[:,i_feat_y-1] = feat_def["uref_szn_std"][self.ndelay-1:]
            szn_mean_Y[:,i_feat_y-1] = feat_def["uref_szn_mean"][self.ndelay-1-i_dl:self.ndelay-1-i_dl+szn_mean_Y.shape[0]]
            szn_std_Y[:,i_feat_y-1] = feat_def["uref_szn_std"][self.ndelay-1-i_dl:self.ndelay-1-i_dl+szn_std_Y.shape[0]]
            #offset_Y[i_feat_y-1] = feat_def["uref_mean"]
            #scale_Y[i_feat_y-1] = feat_def["uref_std"]
        # ----------- Waves -------------------
        for i_wave in np.arange(algo_params["Nwaves"]):
            i_feat_y = self.fidx_Y["real%i"%(i_wave)]
            i_feat_x = self.fidx_X["real%i"%(i_wave)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["waves_szn_mean"][self.ndelay-1:,2*i_wave]
            szn_std_Y[:,i_feat_y-1] = feat_def["waves_szn_std"][self.ndelay-1:,2*i_wave]
            i_feat_y = self.fidx_Y["imag%i"%(i_wave)]
            i_feat_x = self.fidx_X["imag%i"%(i_wave)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["waves_szn_mean"][self.ndelay-1:,2*i_wave+1]
            szn_std_Y[:,i_feat_y-1] = feat_def["waves_szn_std"][:,2*i_wave+1]
            #offset_Y[i_feat_y-1:i_feat_y+1] = np.mean(feat_def["waves_szn_mean"][:,2*i_wave:2*i_wave+2], axis=0)
            #scale_Y[i_feat_y-1:i_feat_y+1] = np.std(feat_def["waves_szn_mean"][:,2*i_wave:2*i_wave+2], axis=0)
        # -------- EOFs ---------------------
        for i_lev in range(Nlev):
            for i_pc in range(algo_params["Npc_per_level"][i_lev]):
                i_feat_y = self.fidx_Y["pc%i_lev%i"%(i_pc,i_lev)]
                i_feat_x = self.fidx_X["pc%i_lev%i"%(i_pc,i_lev)]
                Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
                szn_mean_Y[:,i_feat_y-1] = 0.0
                szn_std_Y[:,i_feat_y-1] = 1.0
        # ------- Vortex moments ------------
        # TODO: make this a time average
        for i_mom in range(algo_params["num_vortex_moments"]):
            i_feat_y = self.fidx_Y["vxmom%i"%(i_mom)]
            i_feat_x = self.fidx_X["vxmom%i"%(i_mom)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            ## ---------- average ----------
            #for i_dl in range(self.ndelay):
            #    Y[:,:,i_feat_y] += X[:,self.ndelay-1-i_dl:Ntx-i_dl,i_feat_x]/self.ndelay
            ## ----------------------------
            szn_mean_Y[:,i_feat_y-1] = feat_def["vtx_diags_szn_mean"][self.ndelay-1:,i_mom]
            szn_std_Y[:,i_feat_y-1] = feat_def["vtx_diags_szn_std"][:,i_mom]
        # ------- Polar cap temperature ------------
        for i_lev in range(Nlev):
            if algo_params["captemp_flag"][i_lev]:
                i_feat_y = self.fidx_Y["captemp_lev%i"%(i_lev)]
                i_feat_x = self.fidx_X["captemp_lev%i"%(i_lev)]
                Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
                szn_mean_Y[:,i_feat_y-1] = feat_def["temp_capavg_szn_mean"][self.ndelay-1:,i_lev]
                szn_std_Y[:,i_feat_y-1] = feat_def["temp_capavg_szn_std"][:,i_lev]
        # ------- Heat flux ------------
        # TODO: make this a time integral
        for i_lev in range(Nlev):
            for i_wn in range(algo_params["heatflux_wavenumbers"][i_lev]):
            #if algo_params["heatflux_flag"][i_lev]:
                i_feat_y = self.fidx_Y["heatflux_lev%i_wn%i"%(i_lev,i_wn)]
                i_feat_x = self.fidx_X["heatflux_lev%i_wn%i"%(i_lev,i_wn)]
                Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
                ## ---------- average ----------
                #for i_dl in range(self.ndelay):
                #    Y[:,:,i_feat_y] += X[:,self.ndelay-1-i_dl:Ntx-i_dl,i_feat_x]/self.ndelay
                ## ----------------------------
                szn_mean_Y[:,i_feat_y-1] = feat_def["vT_szn_mean"][self.ndelay-1:,i_lev,i_wn]
                szn_std_Y[:,i_feat_y-1] = feat_def["vT_szn_std"][:,i_lev,i_wn]
        tpt_feat = {"Y": Y, "szn_mean_Y": szn_mean_Y, "szn_std_Y": szn_std_Y, "idx_resamp": idx_resamp}
        pickle.dump(tpt_feat, open(tpt_feat_filename,"wb"))
        #print(f"Y.shape = {Y.shape}")
        return 
    def set_feature_indices_X(self,feat_def,fidx_X_filename):
        # Build a mapping from feature names to indices in X.
        fidx = dict() # maps feature abbreviation to column of x
        flab = dict() # maps feature abbreviation to name of observable
        Nlev = len(feat_def["plev"])
        i_feat = 0
        # ----------- Time (hours) -------------
        key = "time_h"
        fidx[key] = i_feat
        flab[key] = r"Hours since Oct. 1"
        i_feat += 1
        # ---------- Reference zonal wind --------
        key = "uref"
        fidx[key] = i_feat
        flab[key] = r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]"
        i_feat += 1
        # --------- Zonal wind at other levels -------
        for i_lev in range(Nlev):
            key = "ubar_60N_lev%i"%(i_lev)
            fidx[key] = i_feat
            flab[key] = r"$\overline{u}$ (%i hPa, 60$^\circ$N) [m/s]"%(feat_def["plev"][i_lev]/100)
            i_feat += 1
        # -------------- Waves -------------------
        for i_wave in range(1,self.num_wavenumbers+1):
            key = "real%i"%(i_wave)
            fidx[key] = i_feat
            flab[key] = r"$\mathrm{Re}\{\mathrm{Wave %i}\}$"%(i_wave)
            i_feat += 1
            key = "imag%i"%(i_wave)
            fidx[key] = i_feat
            flab[key] = r"$\mathrm{Im}\{\mathrm{Wave %i}\}$"%(i_wave)
            i_feat += 1
        # -------- Principal components ----------
        for i_lev in range(Nlev):
            for i_pc in range(self.Npc_per_level_max):
                key = "pc%i_lev%i"%(i_pc,i_lev)
                fidx[key] = i_feat
                flab[key] = r"PC %i at %i hPa"%(i_pc+1,feat_def["plev"][i_lev]/100.0)
                i_feat += 1
        # -------- Vortex moments ----------------
        vxmom_names = ["Area", "Center latitude", "Aspect ratio", "Kurtosis"]
        for i_mom in range(self.num_vortex_moments_max):
            key = "vxmom%i"%(i_mom)
            fidx[key] = i_feat
            flab[key] = vxmom_names[i_mom]
            i_feat += 1
        # -------- Polar cap temperature ---------
        for i_lev in range(Nlev):
            key = "captemp_lev%i"%(i_lev)
            fidx[key] = i_feat
            flab[key] = r"Polar cap temp. at %i hPa [K]"%(feat_def["plev"][i_lev]/100.0) 
            i_feat += 1
        # -------- Heat flux -----------
        for i_lev in range(Nlev):
            for i_wn in range(self.heatflux_wavenumbers_per_level_max):
                key = "heatflux_lev%i_wn%i"%(i_lev,i_wn)
                fidx[key] = i_feat
                flab[key] = r"Heat flux wave %i at 45-75$^\circ$N, %i hPa [K$\cdot$m/s]"%(i_wn,feat_def["plev"][i_lev]/100.0)
                i_feat += 1
        # Save the file
        pickle.dump(fidx,open(fidx_X_filename,"wb"))
        self.fidx_X = fidx
        self.flab_X = flab
        return fidx
    def set_feature_indices_Y(self,feat_def,fidx_Y_filename,algo_params):
        # Build a mapping from feature names to indices in Y. (These will approximately match those in X, but will also include time-delay embeddings and perhaps more complicated combinations.) 
        fidx = dict()
        flab = dict()
        Nlev = len(feat_def["plev"])
        i_feat = 0
        # ---------- Time ---------------
        key = "time_h"
        fidx[key] = i_feat
        flab[key] = "Hours since Oct. 1"
        i_feat += 1
        # -------- Time-delayed reference zonal wind ---------------
        for i_dl in range(self.ndelay):
            key = "uref_dl%i"%(i_dl)
            fidx[key] = i_feat
            delaystr = ", $t-%i$ day"%(i_dl*self.dtwint/24.0) if i_dl>0 else ""
            flab[key] = r"$\overline{u}$ (10 hPa, 60$^\circ$N%s) [m/s]"%(delaystr)
            i_feat += 1
        # --------- Waves ----
        for i_wave in range(1,algo_params["Nwaves"]+1):
            key = "real%i"%(i_wave)
            fidx[key] = i_feat
            flab[key] = r"$\mathrm{Re}\{\mathrm{Wave %i}\}$"%(i_wave)
            i_feat += 1
            key = "imag%i"%(i_wave)
            fidx[key] = i_feat
            flab[key] = r"$\mathrm{Im}\{\mathrm{Wave %i}\}$"%(i_wave)
            i_feat += 1
        # ----- Principal components ---------------
        for i_lev in range(Nlev):
            for i_pc in range(algo_params["Npc_per_level"][i_lev]):
                key = "pc%i_lev%i"%(i_pc,i_lev)
                fidx[key] = i_feat
                flab[key] = r"PC %i at %i hPa"%(i_pc+1,feat_def["plev"][i_lev]/100.0)
                i_feat += 1
        # -------- Vortex moments ----------------
        for i_mom in range(algo_params["num_vortex_moments"]):
            key = "vxmom%i"%(i_mom)
            fidx[key] = i_feat
            flab[key] = self.flab_X[key]
            i_feat += 1
        # -------- Polar cap temperature ---------
        for i_lev in range(Nlev):
            if algo_params["captemp_flag"][i_lev]:
                key = "captemp_lev%i"%(i_lev)
                fidx[key] = i_feat
                flab[key] = r"Polar cap temp. at %i hPa [K]"%(feat_def["plev"][i_lev]/100.0) 
                i_feat += 1
        # -------- Heat flux -----------
        for i_lev in range(Nlev):
            for i_wn in range(algo_params["heatflux_wavenumbers"][i_lev]):
                key = "heatflux_lev%i_wn%i"%(i_lev,i_wn)
                fidx[key] = i_feat
                flab[key] = r"Heat flux wave %i at 45-75$^\circ$N, %i hPa [K$\cdot$m/s]"%(i_wn,feat_def["plev"][i_lev]/100.0)
                i_feat += 1
        # Save the file
        pickle.dump(fidx,open(fidx_Y_filename,"wb"))
        self.fidx_Y = fidx
        self.flab_Y = flab
        return fidx
    def evaluate_features_from_filename(self,ds_filename,feat_def):
        ds = nc.Dataset(ds_filename,"r")
        features = self.evaluate_features(ds,feat_def)
        print(f"evaluated ds_filename {ds_filename}")
        ds.close()
        return features
    def evaluate_features(self,ds,feat_def):
        # Given a single ensemble in ds, evaluate the features and return a big matrix
        i_lev_uref,i_lat_uref = self.get_ilev_ilat(ds)
        trun = timelib.time()
        gh,u,time,plev,lat,lon,fall_year,ghflag = self.get_u_gh(ds)
        time_ugh = timelib.time() - trun
        Nmem,Nt,Nlev,Nlat,Nlon = gh.shape
        Nlat_nh = feat_def["Nlat_nh"]
        area_factor = np.outer(np.cos(lat*np.pi/180), np.ones(Nlon))
        gh = gh.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        u = u.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        trun = timelib.time()
        _,vmer = self.compute_geostrophic_wind(gh,lat,lon) # for meridional wind
        time_vmer = timelib.time() - trun
        Nfeat = len(set(self.fidx_X.values()))
        #print(f"Nfeat = {Nfeat}")
        #print(f"self.fidx_X.values() = {self.fidx_X.values()}")
        #Nfeat = 1 # Time
        #Nfeat += 1 # zonal-mean zonal wind
        #Nfeat += 2*self.num_wavenumbers # Real and imaginary part of each 
        #Nfeat += Nlev*self.Npc_per_level_max # PCs at each level
        #Nfeat += self.num_vortex_moments_max # vortex moments
        #Nfeat += 2*Nlev # polar cap-averaged temperature and meridional heat flux
        #Nfeat = 2 + 2*self.num_wavenumbers + Nlev*self.Npc_per_level_max + 4 + 2*Nlev # Last four for area, center latitude, aspect ratio, and excess kurtosis of the vortex
        X = np.zeros((Nmem*Nt,Nfeat))
        # -------------- Time ------------------
        i_feat = self.fidx_X['time_h']
        X[:,i_feat] = np.outer(np.ones(Nmem),time).flatten()
        # ---------- Zonal-mean zonal wind -------
        uref = np.mean(u[:,i_lev_uref,i_lat_uref,:],axis=1)
        i_feat = self.fidx_X['uref']
        X[:,i_feat] = uref
        print(f"uref: min={uref.min()}, max={uref.max()}")
        # ------------ ubar at other levels ------------
        for i_lev in range(Nlev):
            i_feat = self.fidx_X["ubar_60N_lev%i"%(i_lev)]
            X[:,i_feat] = np.mean(u[:,i_lev,i_lat_uref,:],axis=1)
        if ghflag:
            # ---------- Waves ---------------------
            trun = timelib.time()
            waves = self.get_wavenumbers(gh,i_lev_uref,self.lat_range_uref,lat,lon)
            time_waves = timelib.time() - trun
            for i_wave in range(1,self.num_wavenumbers+1):
                i_feat = self.fidx_X['real%i'%(i_wave)]
                X[:,i_feat] = waves[:,2*(i_wave-1)]
                i_feat = self.fidx_X['imag%i'%(i_wave)]
                X[:,i_feat] = waves[:,2*(i_wave-1)+1]
            # -------- EOFs ----------------------
            gh_unseasoned = self.unseason(X[:,0],gh,feat_def["gh_szn_mean"],feat_def["gh_szn_std"],normalize=False)
            for i_lev in range(Nlev):
                for i_pc in range(self.Npc_per_level_max):
                    i_feat = self.fidx_X["pc%i_lev%i"%(i_pc,i_lev)]
                    X[:,i_feat] = (gh_unseasoned[:,i_lev,:Nlat_nh,:].reshape((Nmem*Nt,Nlat_nh*Nlon)) @ (feat_def["eofs"][i_lev,:,i_pc])) / feat_def["singvals"][i_lev,i_pc] 
            # ---------- Vortex moments ------------
            vtx_moments = self.compute_vortex_moments_sphere(gh,lat,lon,i_lev_subset=[i_lev_uref])
            moment_names = ["area","centerlat","aspect_ratio","excess_kurtosis"]
            for i_mom in range(self.num_vortex_moments_max):
                i_feat = self.fidx_X["vxmom%i"%(i_mom)]
                X[:,i_feat] = vtx_moments[moment_names[i_mom]]
            # --------- Temperature ---------------
            trun = timelib.time()
            temperature = self.get_temperature(gh,plev,lat,lon)
            i_lat_cap = np.argmin(np.abs(lat - 60))
            temp_capavg = np.sum((temperature*area_factor)[:,:,:i_lat_cap,:], axis=(2,3))/np.sum(area_factor[:i_lat_cap,:])
            #print(f"temp_capavg.shape = {temp_capavg.shape}")
            time_temperature = timelib.time() - trun
            for i_lev in range(Nlev):
                i_feat = self.fidx_X["captemp_lev%i"%(i_lev)]
                X[:,i_feat] = temp_capavg[:,i_lev]
            # ---------- Heat flux ----------------
            trun = timelib.time()
            vT = self.get_meridional_heat_flux(gh,temperature,plev,lat,lon)
            time_vT = timelib.time() - trun
            #print(f"Nlev = {Nlev}, vT.shape = {vT.shape}, i_feat = {i_feat}, X.shape = {X.shape}")
            for i_lev in range(Nlev):
                for i_wn in range(self.heatflux_wavenumbers_per_level_max):
                    i_feat = self.fidx_X["heatflux_lev%i_wn%i"%(i_lev,i_wn)]
                    X[:,i_feat] = vT[:,i_lev,i_wn]
        # ------------ Unroll X -------------------
        X = X.reshape((Nmem,Nt,Nfeat))
        return X,fall_year
    def plot_vortex_evolution(self,dsfile,savedir,save_suffix,i_mem=0):
        # Plot the holistic information about a single member of a single ensemble. Include some timeseries and some snapshots, perhaps along the region of maximum deceleration in zonal wind. 
        ds = nc.Dataset(dsfile,"r")
        print("self.num_wavenumbers = {}, self.Npc_per_level_max = {}".format(self.num_wavenumbers,self.Npc_per_level_max))
        funlib = self.observable_function_library_X()
        feat_def = pickle.load(open(self.feature_file,"rb"))
        Nlev = len(feat_def["plev"])
        X,fall_year = self.evaluate_features(ds,feat_def)
        X = X[i_mem]
        print("X.shape = {}".format(X.shape))
        # Determine the period of maximum deceleration
        time = X[:,0]
        decel_window = int(24*10.0/(time[1]-time[0]))
        uref = X[:,1]
        decel10 = uref[decel_window:] - uref[:-decel_window]
        print("uref: min={}, max={}. decel10: min={}, max={}".format(uref.min(),uref.max(),decel10.min(),decel10.max()))
        start = np.argmin(decel10)
        print("start = {}".format(start))
        decel_time_range = [max(0,start-decel_window), min(len(time)-1, start+2*decel_window)]
        full_time_range = self.wtime[[0,-1]]
        # Make a list of things to plot
        obs_key_list = ["mag1","mag2","captemp_lev0","heatflux_lev4_wn0","heatflux_lev4_wn1","heatflux_lev4_wn2","uref","pc0_lev0","pc1_lev0","pc2_lev0","pc3_lev0","pc4_lev0","pc5_lev0"]
        for obs_key in obs_key_list:
            fig,ax = plt.subplots()
            ydata = funlib[obs_key]["fun"](X)
            #ydata = X[:,self.fidx_X[obs_key]]
            ylab = funlib[obs_key]["label"]
            #ylab = self.flab_X[obs_key]
            xdata = funlib["time_h"]["fun"](X)
            #xdata = X[:,self.fidx_X["time_h"]]
            xlab = "%s %i"%(funlib["time_h"]["label"], fall_year)
            #xlab = self.flab_X["time_h"]
            if obs_key.startswith("pc0"):
                ydata *= -1
            ax.plot(xdata,ydata,color='black')
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.axvspan(time[decel_time_range[0]],time[decel_time_range[1]],color='orange',zorder=-1)
            fig.savefig(join(savedir,"timeseries_%s_%s"%(save_suffix,obs_key)))
            plt.close(fig)
        # Plot polar cap evolution
        num_snapshots = 30
        i_lev_ref,i_lat_ref = self.get_ilev_ilat(ds)
        tidx = np.round(np.linspace(decel_time_range[0],decel_time_range[1],min(num_snapshots,decel_time_range[1]-decel_time_range[0]+2))).astype(int)
        gh,u,_,plev,lat,lon,fall_year,_ = self.get_u_gh(ds)
        gh = gh[i_mem]
        u = u[i_mem]
        print("gh.shape = {}".format(gh.shape))
        _,v = self.compute_geostrophic_wind(gh,lat,lon)
        qgpv = self.compute_qgpv(gh,lat,lon)
        print("u.shape = {}".format(u.shape))
        u = u[tidx,i_lev_ref,:,:]
        v = v[tidx,i_lev_ref,:,:]
        gh = gh[tidx,i_lev_ref,:,:]
        qgpv = qgpv[tidx,i_lev_ref,:,:]
        ds.close()
        i_lat_max = np.where(lat < 5)[0][0]
        gh[:,i_lat_max:,:] = np.nan
        qgpv[:,i_lat_max:,:] = np.nan
        for k in range(len(tidx)):
            i_time = tidx[k]
            fig,ax = self.show_ugh_onelevel_cartopy(gh[k],u[k],v[k],lat,lon,vmin=None,vmax=None)
            ax.set_title(r"$\Phi$, $u$ at day {}, {}".format(time[tidx[k]]/24.0,fall_year))
            fig.savefig(join(savedir,"vortex_gh_{}_day{}_yr{}".format(save_suffix,int(time[tidx[k]]/24.0),fall_year)))
            plt.close(fig)
            fig,ax = self.show_ugh_onelevel_cartopy(qgpv[k],u[k],v[k],lat,lon,vmin=np.nanmin(qgpv),vmax=np.nanmax(qgpv))
            ax.set_title(r"$\Phi$, $u$ at day {}, {}".format(time[tidx[k]]/24.0,fall_year))
            fig.savefig(join(savedir,"vortex_qgpv_{}_day{}_yr{}".format(save_suffix,int(time[tidx[k]]/24.0),fall_year)))
            plt.close(fig)
        return
    def wave_mph(self,Y,feat_def,wn,widx=None,unseason_mag=False):
        # wn is wavenumber 
        # mph is magnitude and phase
        if wn <= 0:
            raise Exception("Need an integer wavenumber >= 1. You gave wn = {}".format(wn))
        if widx is None:
            widx = 1 + self.ndelay + 2*wn*np.arange(2)
        wave = Y[:,widx]
        #wave = self.reseason(x[:,0],x[:,widx],feat_def['t_szn'],feat_def['waves_szn_mean'][:,2*(wn-1):2*(wn-1)+2],feat_def['waves_szn_std'][:,2*(wn-1):2*(wn-1)+2])
        phase = np.arctan(-wave[:,1]/(wn*wave[:,0]))
        mag = np.sqrt(np.sum(wave**2,axis=1))
        #if unseason_mag:
        #    mag = self.unseason(x[:,0],mag,feat_def["wave_mag_szn_mean"][:,wn-1],feat_def["wave_mag_szn_std"][:,wn-1])
        return np.array([mag,phase]).T
    def uref_history(self,y,feat_def): # Return wind history
        wind_idx = np.arange(1,1+self.ndelay)
        uref = y[:,wind_idx]
        return uref
    def observable_function_library_X(self):
        feat_def = pickle.load(open(self.feature_file,"rb"))
        funlib = dict()
        for key in self.fidx_X.keys():
            funlib[key] = {
                    "fun": lambda X,key=key: X[:,self.fidx_X[key]],
                    "label": self.flab_X[key],
                    }
        # Linear functions
        funlib["time_d"] = {
                "fun": lambda X: X[:,self.fidx_X["time_h"]]/24.0, 
                "label": "Time since Oct. 1 [days]",
                }
        funlib["time_d_nov1"] = {
                "fun": lambda X: X[:,self.fidx_X["time_h"]]/24.0 - 30.0,
                "label": "Time since Nov. 1 [days]",
                }
        for i_lev in range(len(feat_def["plev"])):
            key = "heatflux_lev%i_total"%(i_lev)
            idx_ilev = np.array([self.fidx_X["heatflux_lev%i_wn%i"%(i_lev,i_wn)] for i_wn in range(self.heatflux_wavenumbers_per_level_max)])
            funlib[key] = {
                    "fun": lambda X,i_lev=i_lev,idx_ilev=idx_ilev: np.sum(X[:,idx_ilev],axis=1),
                    "label": "Heat flux at %i hPa"%(feat_def["plev"][i_lev]/100.0)
                    }
        # Nonlinear functions
        for i_wave in range(1,self.num_wavenumbers+1):
            funlib["mag%i"%(i_wave)] = {
                    "fun": lambda X,i_wave=i_wave: np.sqrt(X[:,self.fidx_X["real%i"%(i_wave)]]**2 + X[:,self.fidx_X["imag%i"%(i_wave)]]**2),
                    "label": "Wave %i magnitude"%(i_wave),
                    }
            funlib["ph%i"%(i_wave)] = {
                    "fun": lambda X,i_wave=i_wave: np.arctan2(X[:,self.fidx_X["imag%i"%(i_wave)]]**2, X[:,self.fidx_X["real%i"%(i_wave)]]),
                    "label": "Wave %i phase"%(i_wave),
                    }
        return funlib
    def observable_function_library_Y(self,algo_params):
        # Build the database of observable functions
        feat_def = pickle.load(open(self.feature_file,"rb"))
        Nlev = len(feat_def['plev'])
        funlib = dict()
        for key in self.fidx_Y.keys():
            funlib[key] = {
                    "fun": lambda Y,key=key: Y[:,self.fidx_Y[key]],
                    "label": self.flab_Y[key],
                    }
        # Linear functions
        funlib["time_d"] = {
                "fun": lambda Y: Y[:,self.fidx_Y["time_h"]]/24.0, 
                "label": "Time since Oct. 1 [days]",
                }
        uref_idx = np.array([self.fidx_Y["uref_dl%i"%(i_dl)] for i_dl in range(self.ndelay)])
        if len(uref_idx) > 1:
            funlib["windfall"] = {
                    "fun": lambda Y: self.windfall(Y[:,uref_idx]),
                    "label": r"Max. decel. $\Delta\overline{u}/\Delta t$ [m/s/day]",
                    }
        for i_dl in range(self.ndelay-2):
            funlib["uref_inc_%i"%(i_dl)] = {
                    "fun": lambda Y,i_dl=i_dl: Y[:,self.fidx_Y["uref_dl%i"%(i_dl)]] - Y[:,self.fidx_Y["uref_dl%i"%(i_dl+2)]],
                    "label": r"$\Delta\overline{u}(t-%i,t-%i)$"%(i_dl,i_dl+2),
                    }
        # Nonlinear functions
        for i_wave in range(1,self.num_wavenumbers+1):
            funlib["mag%i"%(i_wave)] = {
                    "fun": lambda Y,i_wave=i_wave: np.sqrt(Y[:,self.fidx_Y["real%i"%(i_wave)]]**2 + Y[:,self.fidx_Y["imag%i"%(i_wave)]]**2),
                    "label": "Wave %i magnitude"%(i_wave),
                    }
            funlib["ph%i"%(i_wave)] = {
                    "fun": lambda Y,i_wave=i_wave: np.arctan2(Y[:,self.fidx_Y["imag%i"%(i_wave)]]**2, Y[:,self.fidx_Y["real%i"%(i_wave)]]),
                    "label": "Wave %i phase"%(i_wave),
                    }
        return funlib
    def windfall(self,U):
        # Return the most negative change in zonal wind U (time is axis 1)
        Nx,Nt = U.shape
        dU = np.zeros((Nx,int(Nt*(Nt-1)/2)))
        k = 0
        for i in range(Nt-1):
            for j in range(i+1,Nt):
                dU[:,k] = (U[:,j] - U[:,i])/((j-i)*self.dtwint/24)
                k += 1
        if k != dU.shape[1]:
            raise Exception(f"ERROR: After the double for loop, k should be {dU.shape[1]}, but actually it's {k}")
        return np.min(dU, axis=1)
    def get_pc(self,Y,i_lev,i_eof,Nwaves=None,Npc_per_level=None):
        if Nwaves is None:
            Nwaves = self.num_wavenumbers
        if Npc_per_level is None:
            Npc_per_level = self.Npc_per_level_max * np.ones(len(feat_def["plev"]), dtype=int)
        idx = 1 + self.ndelay + 2*Nwaves + np.sum(Npc_per_level[:i_lev]) + i_eof
        eof = Y[:,idx]
        return eof
    def get_vortex_area(self,x,i_lev,Nwaves,Npc_per_level):
        idx = 2 + 2*Nwaves + np.sum(Npc_per_level)
        area = self.reseason(x[:,0],x[:,idx],feat_def['t_szn'],feat_def['vtx_area_szn_mean'],feat_def['vtx_area_szn_std'])
        return area
    def get_vortex_displacement(self,x,i_lev,Nwaves,Npc_per_level):
        idx = 2 + 2*Nwaves + np.sum(Npc_per_level) + 1
        centerlat = self.reseason(x[:,0],x[:,idx],feat_def['t_szn'],feat_def['vtx_centerlat_szn_mean'],feat_def['vtx_centerlat_szn_std'])
        return centerlat 
    def show_ugh_onelevel_cartopy(self,gh,u,v,lat,lon,vmin=None,vmax=None): 
        # Display the geopotential height at a single pressure level
        fig,ax,data_crs = self.display_pole_field(gh,lat,lon,vmin=vmin,vmax=vmax)
        lon_subset = np.linspace(0,lon.size-1,20).astype(int)
        lat_subset = np.linspace(0,lat.size-2,60).astype(int)
        #ax.quiver(lon[lon_subset],lat[lat_subset],u[lat_subset,:][:,lon_subset],v[lat_subset,:][:,lon_subset],transform=data_crs,color='black',zorder=5)
        ax.set_title(r"$\Phi$, $u$")
        return fig,ax
    def show_multiple_eofs(self,savedir):
        feat_def = pickle.load(open(self.feature_file,"rb"))
        i_lev_uref = feat_def["i_lev_uref"]
        for i_eof in range(min(6,self.Npc_per_level_max)):
            self.show_eof(feat_def,i_lev_uref,i_eof,savedir)
        return
    def show_eof(self,feat_def,i_lev,i_eof,savedir):
        # Display a panel of principal components
        plev,lat,lon = [feat_def[v] for v in ["plev","lat","lon"]]
        Nlev,Nlat,Nlon = len(plev),len(lat),len(lon)
        Nlat_nh = feat_def["Nlat_nh"]
        eof = feat_def["eofs"][i_lev,:,i_eof].reshape((Nlat_nh,Nlon))
        fig,ax,_ = self.display_pole_field(eof,lat[:Nlat_nh],lon)
        ax.set_title("EOF %i at $p=%i$ hPa"%(i_eof+1,plev[i_lev]/100))
        fig.savefig(join(savedir,"eof_ilev%i_ieof%i"%(i_lev,i_eof)))
        plt.close(fig)
        return
    def display_pole_field(self,field,lat,lon,vmin=None,vmax=None):
        data_crs = ccrs.PlateCarree() 
        ax_crs = ccrs.Orthographic(-10,90)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection=ax_crs)
        im = ax.pcolormesh(lon,lat,field,shading='nearest',cmap='coolwarm',transform=data_crs,vmin=vmin,vmax=vmax)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=3, edgecolor='black')
        fig.colorbar(im,ax=ax)
        return fig,ax,data_crs
    def plot_zonal_wind_every_year(self,
            feat_filename_ra_dict,fall_year_filename_ra_dict,
            feat_def,savedir,colors,labels,
            uthresh_a,uthresh_list,tthresh):
        # Every year in which we have data, plot it. Plot both datasets if they both have data.
        keys_ra = list(feat_filename_ra_dict.keys())
        print(f"keys_ra = {keys_ra}")
        all_years = []
        uref = dict({})
        time_d = dict({})
        years = dict({})
        rates = dict({})
        timespan = np.array([np.inf,-np.inf])
        funlib_X = self.observable_function_library_X()
        for k in keys_ra:
            years[k] = np.load(fall_year_filename_ra_dict[k]).astype(int)
            all_years = np.union1d(all_years, years[k])
            X = np.load(feat_filename_ra_dict[k])
            Nx,Nt,xdim = X.shape
            uref[k] = funlib_X["uref"]["fun"](X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
            time_d[k] = funlib_X["time_h"]["fun"](X.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))/24.0
            timespan[0] = min(timespan[0],np.min(time_d[k]))
            timespan[1] = max(timespan[0],np.max(time_d[k]))
        common_years = all_years.copy()
        for k in keys_ra:
            years[k] = np.load(fall_year_filename_ra_dict[k]).astype(int)
            common_years = np.intersect1d(common_years,years[k])
        for i_yr,yr in enumerate(common_years):
            print(f"yr = {yr}")
            fig,ax = plt.subplots()
            handles = []
            ax.axvspan(timespan[0],tthresh[0]/24.0,color='lightskyblue')
            ax.axvspan(tthresh[1]/24.0,timespan[1],color='lightskyblue')
            for i_uth,uth in enumerate(uthresh_list):
                ax.plot(tthresh/24.0,uth*np.ones(2),color='purple',linestyle='--')
            for k in keys_ra:
                if yr in years[k]:
                    idx_yr = np.where(years[k] == yr)[0][0]
                    print(f"idx_yr = {idx_yr}")
                    h, = ax.plot(time_d[k][idx_yr],uref[k][idx_yr],color=colors[k],label=labels[k])
                    handles += [h]
            ax.legend(handles=handles)
            ax.set_title(f"{int(yr)}-{int(yr)+1}")
            ax.set_xlabel("Time since October 1 [days]")
            ax.set_ylabel(funlib_X["uref"]["label"])
            fig.savefig(join(savedir,f"uref_{int(yr)}-{int(yr)+1}"))
            plt.close(fig)
            # TODO: Finish computing rates independently 
        return
    def illustrate_dataset(self,
            uthresh_a,uthresh_b_list,tthresh,sswbuffer,
            feat_filename_ra,feat_filename_hc,
            label_ra,label_hc,
            tpt_feat_filename_ra,tpt_feat_filename_hc,
            ens_start_filename_ra,ens_start_filename_hc,
            fall_year_filename_ra,fall_year_filename_hc,
            feat_def,feat_display_dir):
        # Plot zonal wind over time with both reanalysis and hindcast datasets. Use multiple thresholds to demonstrate the difference between SSWs of different severity.
        tpt_feat_ra = pickle.load(open(tpt_feat_filename_ra,"rb"))
        Yra = tpt_feat_ra["Y"]
        idx_resamp_ra = tpt_feat_ra["idx_resamp"]
        print(f"idx_resamp_ra = {idx_resamp_ra}")
        tpt_feat_hc = pickle.load(open(tpt_feat_filename_hc,"rb"))
        Yhc = tpt_feat_hc["Y"]
        idx_resamp_hc = tpt_feat_hc["idx_resamp"]
        Xra = np.load(feat_filename_ra)[idx_resamp_ra]#[:,self.ndelay-1:]
        Nxra,Ntra,xdim = Xra.shape
        Xhc = np.load(feat_filename_hc)[idx_resamp_hc]#[:,self.ndelay-1:]
        Nxhc,Nthc,_ = Xhc.shape
        print(f"Nxhc,Nthc,_ = {Xhc.shape}")
        fy_ra = np.load(fall_year_filename_ra)[idx_resamp_ra]
        fy_hc = np.load(fall_year_filename_hc)
        print(f"fy_hc.shape = {fy_hc.shape}")
        enst_hc = np.load(ens_start_filename_hc)
        Nmem_hc = enst_hc[1] - enst_hc[0]
        print(f"fy_hc[:4] = {fy_hc[:4]}")
        print(f"enst_hc[:4] = {enst_hc[:4]}")
        # For each threshold, identify which years achieved the event
        tpt_bndy = dict({"tthresh": tthresh, "uthresh_a": uthresh_a, "sswbuffer": sswbuffer})
        rare_event_idx = []
        for i_uth,uthresh_b in enumerate(uthresh_b_list):
            tpt_bndy["uthresh_b"] = uthresh_b
            src_tag,dest_tag,time2dest = self.compute_src_dest_tags(Yra,feat_def,tpt_bndy)
            ab_idx = np.where(np.any((src_tag==0)*(dest_tag==1), axis=1))[0]
            print(f"At threshold {uthresh_b}, ab_idx = {ab_idx}, corresponding to years {fy_ra[ab_idx]}")
            rare_event_idx.append(ab_idx)
            if i_uth > 0:
                if not all(i_y in rare_event_idx[i_uth-1] for i_y in rare_event_idx[i_uth]):
                    raise Exception(f"ERROR: Some SSWs registered with threshold {uthresh_b_list[i_uth]} but not {uthresh_b_list[i_uth-1]}")
                rare_event_idx[i_uth-1] = np.setdiff1d(rare_event_idx[i_uth-1],rare_event_idx[i_uth])
        # Now plot the years in each subset
        funlib_X = self.observable_function_library_X()
        time_d_ra = funlib_X["time_d"]["fun"](Xra.reshape((Nxra*Ntra,xdim))).reshape((Nxra,Ntra))
        uref_ra = funlib_X["uref"]["fun"](Xra.reshape((Nxra*Ntra,xdim))).reshape((Nxra,Ntra))
        time_d_hc = funlib_X["time_d"]["fun"](Xhc.reshape((Nxhc*Nthc,xdim))).reshape((Nxhc,Nthc))
        uref_hc = funlib_X["uref"]["fun"](Xhc.reshape((Nxhc*Nthc,xdim))).reshape((Nxhc,Nthc))
        # Before randomly sampling a reanalysis year, make sure there are corresponding hindcasts 
        print(f"fy_ra = {fy_ra}\n fy_hc = {fy_hc}")
        common_years_hc_ra = np.intersect1d(fy_ra,fy_hc)
        print(f"common_years_hc_ra = {common_years_hc_ra}")
        # Quantile ranges 
        quantile_ranges = [0.4, 0.8, 1.0]
        lower = np.zeros((len(quantile_ranges),Ntra))
        upper = np.zeros((len(quantile_ranges),Ntra))
        for ti in range(Ntra):
            #idx = np.where(np.abs(time_d_ra - time_d_ra[0,ti]) < 1.5)
            idx = np.where(time_d_ra == time_d_ra[0,ti])
            for qi in range(len(quantile_ranges)):
                lower[qi,ti] = np.quantile(uref_ra[idx[0],idx[1]], 0.5-0.5*quantile_ranges[qi])
                upper[qi,ti] = np.quantile(uref_ra[idx[0],idx[1]], 0.5+0.5*quantile_ranges[qi])
        prng = np.random.RandomState(3)
        for i_uth,uthresh_b in enumerate(uthresh_b_list):
            tpt_bndy["uthresh_b"] = uthresh_b
            src_tag,dest_tag,time2dest = self.compute_src_dest_tags(Yra,feat_def,tpt_bndy)
            fig,ax = plt.subplots()
            # ---------- Plot the climatology in the background ----------
            # TODO: line these up by timing. 
            for qi in range(len(quantile_ranges))[::-1]:
                ax.fill_between(time_d_ra[0],lower[qi],upper[qi],color=plt.cm.binary(0.2 + 0.6*(1-quantile_ranges[qi])),zorder=-1)
            # ------------------------------------------------------------
            handles = []
            ax.plot([np.min(time_d_ra),tthresh[1]/24.0], uthresh_b*np.ones(2), color='red', linewidth=2.5)
            ax.axvline(tthresh[1]/24.0, color='dodgerblue', linewidth=2.5)
            #ax.axhspan(np.min(uref_ra),uthresh_b,color='red',zorder=-3)
            #ax.axvspan(np.min(time_d_ra),tthresh[0]/24.0,color='lightskyblue',zorder=-2)
            #ax.axvspan(tthresh[1]/24.0,np.max(time_d_ra),color='lightskyblue',zorder=-2)
            print(f"For uthresh = {uthresh_b}, rare event idx = {rare_event_idx[i_uth]}, corresponding to years {fy_ra[rare_event_idx[i_uth]]}")
            admissible_idx = np.intersect1d(rare_event_idx[i_uth], np.where(np.in1d(fy_ra, fy_hc))[0])
            if len(admissible_idx) == 0:
                print("WARNING: no common RA and HC years")
            else:
                #y_ss = prng.choice(rare_event_idx[i_uth],size=1,replace=False)
                y_ss = prng.choice(admissible_idx,size=1,replace=False)
                print(f"y_ss.shape = {y_ss.shape}")
                for i_y in y_ss: #rare_event_idx[i_uth]:
                    rxn_idx = np.where((src_tag[i_y]==0)*(dest_tag[i_y]==1))[0]
                    i0,i1 = rxn_idx[0],rxn_idx[-1]+1
                    i0 += self.ndelay-1
                    i1 += self.ndelay-1
                    h, = ax.plot(time_d_ra[i_y,i0:i1+1],uref_ra[i_y,i0:i1+1],color='black',label=r"%s %s-%s"%(label_ra,fy_ra[i_y],fy_ra[i_y]+1),zorder=2,linewidth=2)
                    handles += [h]
                    ax.plot(time_d_ra[i_y,:i0+1],uref_ra[i_y,:i0+1],color='black',linewidth=1.0,zorder=2,linestyle='--')
                    ax.plot(time_d_ra[i_y,i1:],uref_ra[i_y,i1:],color='black',linewidth=1.0,zorder=2,linestyle='--')
                    # ----------- Now identify three hindcast ensembles corresponding to this year ---------------
                    print(f"fy_ra[i_y] = {fy_ra[i_y]}")
                    idx_hc = np.where(fy_hc == fy_ra[i_y])[0]
                    print(f"idx_hc.shape = {idx_hc.shape}")
                    days_idx_hc = time_d_hc[enst_hc[idx_hc],0]
                    idx_hc_ss = idx_hc[np.array([np.argmin(np.abs(days_idx_hc - d)) for d in [30,110]])]
                    #idx_hc_ss = idx_hc[np.linspace(0,len(idx_hc)-1,5).astype(int)[1:-1]]
                    #idx_hc_ss = prng.choice(idx_hc, size=3, replace=False)
                    colorlist = ['darkviolet']*3
                    i_col = 0
                    for i_ens in idx_hc_ss:
                        for i_mem in range(Nmem_hc):
                            h, = ax.plot(time_d_hc[enst_hc[i_ens]+i_mem],uref_hc[enst_hc[i_ens]+i_mem],color=colorlist[i_col],zorder=1,linewidth=0.75,label=label_hc)
                        i_col += 1
                    handles += [h]
                ax.set_xlabel(funlib_X["time_d"]["label"])
                ax.set_ylabel(funlib_X["uref"]["label"])
                leg = ax.legend(handles=handles,loc='upper left')
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.5)
                ax.set_ylim([np.min(uref_ra),np.max(uref_ra)])
                ax.set_xlim([tthresh[0]/24.0,np.max(time_d_ra)])
                ax.set_xticks(np.cumsum([31,30,31,31,28,31]))
                ax.set_xticklabels(['Nov. 1', 'Dec. 1', 'Jan. 1', 'Feb. 1', 'Mar. 1','Apr. 1'])
                fig.savefig(join(feat_display_dir,"illustration_uth%i"%(uthresh_b)))
                plt.close(fig)
                print(f"Saved an illustration in directory {feat_display_dir}")
        return
