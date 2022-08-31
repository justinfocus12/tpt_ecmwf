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
    def generate_seasonal_xticks(self, t_szn):
        # Given some t_szn array, convert it to months and days 
        idx = np.linspace(0,len(t_szn)-1,4).astype(int)
        print(idx)
        t_dt64_ticks = np.datetime64(f"1901-{self.szn_start['month']:02}-{self.szn_start['day']:02}") + self.time_unit * t_szn[idx]
        xticks = t_szn[idx]
        xticklabels = [t64.astype(datetime.datetime).strftime("%b %d") for t64 in t_dt64_ticks]
        return xticks, xticklabels
    def set_event_seasonal_params(self):
        self.szn_start = {"month": 9, "day": 1} # Note this is earlier than the earliest possible SSW
        self.szn_length = np.sum([monthrange(1901,month) for month in [10, 11, 12, 1 ,2, 3]])
        self.t_szn_edge = np.arange(0, self.szn_length+0.5, 1).astype('float64')
        self.Nt_szn = len(self.t_szn_edge) - 1
        self.dt_szn = self.t_szn_edge[1] - self.t_szn_edge[0]
        #self.Nt_szn = int(self.szn_length / 2) # Approximately two days per averaging window 
        #self.dt_szn = self.szn_length / self.Nt_szn
        #self.t_szn_edge = np.linspace(0, self.szn_length, self.Nt_szn+1)
        return 
    # ---------------- Observables -------------------------
    # TODO: for all the following, augment coordinates of observables to include multiple members. In fact, require every dataset input to include (time,ensemble,member) as coordinates. The calling function must expand dimensions accordingly. 
    def prepare_blank_observable(self, ds, feature_coord_names):
        # A general method to create a blank DataArray that replaces the coordinates "longitude" and "latitude" in the input dataset (ds) with a dimension "feature" with coordinates feature_names
        F_coords = {key: ds.coords[key].to_numpy() for key in list(ds.coords.keys())}
        for spatial_coord in ["longitude","latitude","level"]:
            if spatial_coord in list(F_coords.keys()):
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
    def qbo_observable(self, ds, src):
        # Zonal-mean zonal wind at 10 hPa in a region near the equator
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
    def ubar_observable(self, ds, src):
        #zonal-mean zonal wind at a range of latitudes, and all altitudes
        ubar_features = ["ubar_10_60", "ubar_100_60", "ubar_500_60", "ubar_850_60"] #ds.coords['level'].to_numpy()
        ubar = self.prepare_blank_observable(ds, ubar_features)
        ubar_60 = ds['u'].sel(latitude=60).mean(dim='longitude')
        for level in [10, 100, 500, 850]:
            ubar.loc[dict(feature=f"ubar_{level}_60")] = ubar.sel(feature=f"ubar_{level}_60") + ubar_60.sel(level=level)
        return ubar
    def time_observable(self, ds, src):
        # TODO: get rid of stuff in here that uses all the externally imposed conventions from the event definition. That should come downstream. 
        # Return all the possible kinds of time we might need from this object. The basic unit will be DAYS 
        time_types = ["t_dt64", "t_abs", "t_cal", "year_cal"]
        # TODO: get rid of t_szn and year_szn_start. Compute those after the fact. 
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
        
        tda.loc[dict(feature="t_cal")] += (treg - year_start) / self.time_unit  
        return tda
    def augment_time_observable(self, tda):
        time_types_aug = list(tda["feature"].data) + ["year_szn_start","t_szn"]
        tda_aug = self.prepare_blank_observable(tda, time_types_aug) 
        # Recover the regular time 
        treg = tda['t_init'] + self.time_unit * tda['t_sim']
        treg = treg.expand_dims({"member": tda.member})
        print(f"treg first few members = {treg.isel(member=slice(0,2),t_sim=0,t_init=0).data}")
        print(f"treg dims = {treg.dims}")
        print(f"treg shape = {treg.shape}")
        #treg = tda.sel(feature="t_abs").astype(tda['t_init'].dtype)
        print(f"treg dtype = {treg.dtype}")
        year_cal = treg.dt.year #tda.sel(feature="year_cal")
        print(f"year_cal first few members = {year_cal.isel(member=slice(0,2),t_sim=0,t_init=0).data}")
        print(f"year_cal.dims = {year_cal.dims}")
        ycflat = year_cal.to_numpy().flatten()
        sssy = np.array([np.datetime64(f"{int(ycflat[i])}-{self.szn_start['month']:02}-{self.szn_start['day']:02}") for i in range(len(ycflat))])
        szn_start_same_year = xr.DataArray(coords=year_cal.coords, dims=year_cal.dims, data=sssy.reshape(year_cal.shape))
        print(f"sssy first few members = {szn_start_same_year.isel(member=slice(0,2),t_sim=0,t_init=0).data}")
        print(f"szn_start_same_year.dims = {szn_start_same_year.dims}")
        year_szn_start = year_cal - 1*(treg < szn_start_same_year) #.to_numpy()
        yssflat = year_szn_start.to_numpy().flatten()
        ssmr = np.array([np.datetime64(f"{int(yssflat[i])}-{self.szn_start['month']:02}-{self.szn_start['day']:02}") for i in range(len(yssflat))])
        szn_start_most_recent = xr.DataArray(coords=year_cal.coords, dims=year_cal.dims, data=ssmr.reshape(year_cal.shape))
        print(f"ssmr first few members = {szn_start_most_recent.isel(member=slice(0,2),t_sim=0,t_init=0).data}")
        print(f"szn_start_most_recent.dims = {szn_start_most_recent.dims}")

        #print(f"ssmr first few members = {szn_start_most_recent.isel(member=slice(0,2),t_sim=0,t_init=0).data}")
        tda_aug.loc[dict(feature=tda["feature"].data)] += tda
        tda_aug.loc[dict(feature="year_szn_start")] += year_szn_start
        rhs = (treg - szn_start_most_recent) / self.time_unit
        tda_aug.loc[dict(feature="t_szn")] += rhs
        return tda_aug
    # -----------------------------------------------
    # ------------ feat_all -------------------
    def preprocess_netcdf(self, ds, src):
        # Only sample the beginning of each day 
        if src == "e5":
            day_start_idx, = np.where(ds["time"].dt.hour == 0)
            ds = ds.isel(time=day_start_idx)
            ds["z"] *= 1.0/9.806
            ds = ds.rename({"z": "gh"})
        elif src == "s2":
            ds = ds.rename({"number": "member"})
        ds = ds.expand_dims({"t_init": ds['time'][0:1]})
        ds = ds.assign_coords({"time": (ds['time'] - ds['time'][0])/np.timedelta64(1,'D')})
        ds = ds.rename({"time": "t_sim"})
        return ds
    def compute_all_features(self, src, input_dir, input_file_list, output_dir, featdef, obs2compute=None, print_every=5):
        if obs2compute is None:
            obs2compute = [f"{obs}_observable" for obs in ["time","ubar", "pc", "temperature", "heatflux", "qbo"]]
        obs_dict = dict({obsname: [] for obsname in obs2compute})
        concat_dim = "t_init"
        for i_file in range(len(input_file_list)):
            print_flag = (i_file % print_every == 0)
            if print_flag:
                print(f"Beginning file {i_file}/{len(input_file_list)}")
            t0 = datetime.datetime.now()
            ds = xr.open_dataset(join(input_dir, input_file_list[i_file]))
            ds = self.preprocess_netcdf(ds, src)
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
            if print_flag:
                print(f"\t----Times--- \n\topening/processing file {t1-t0}\n\ttime obs {t2-t1}\n\tubar obs {t3-t2}\n\tvT obs {t4-t3}\n\ttemp obs {t5-t4}\n\tpc obs {t6-t5}\n\tqbo obs {t7-t6}")
        for obsname in obs2compute:
            ds_obs = xr.concat(obs_dict[obsname], dim=concat_dim).sortby("t_init")
            # If ERA5, stack back into places
            # PROBLEM: some ERA5 units have the same time dimension
            if src == "e5":
                t_sim_offsets = (ds_obs["t_init"] - ds_obs["t_init"][0])/self.time_unit
                print(f"t_sim_offsets[:24] = {t_sim_offsets[:24]}")
                t_sim_new = (t_sim_offsets + ds_obs["t_sim"]).transpose('t_init','t_sim').to_numpy().flatten()
                ds_obs = ds_obs.stack(t_new=('t_init','t_sim')).transpose('t_new','feature').assign_coords({"t_new": t_sim_new}).rename({"t_new": "t_sim"})
                nonnan_idx, = np.where(np.isnan(ds_obs).sum(dim="feature") == 0)
                ds_obs = ds_obs.isel(t_sim=nonnan_idx)
            ds_obs.to_netcdf(join(output_dir, f"{obsname}.nc"))
            ds_obs.close()
        return 
    # --------------------------------------------------
    def assemble_all_features(self, src, output_dir):
        obs2assemble = [f"{obs}_observable" for obs in ["ubar", "pc", "temperature", "heatflux", "qbo"]]
        feat_all = dict({
            obsname: xr.open_dataarray(join(output_dir, f"{obsname}.nc"))
            for obsname in obs2assemble
            })
        # Special case: time
        t_obs = xr.open_dataarray(join(output_dir,"time_observable.nc"))
        if src == "e5":
            t_init = (
                    t_obs
                    .sel(feature="t_dt64",drop=True)
                    .isel(t_sim=0,drop=True)
                    .to_numpy()
                    .astype("datetime64[ns]")
                    )
            for obsname in obs2assemble:
                feat_all[obsname] = feat_all[obsname].expand_dims(dim={"t_init": [t_init], "member": [1]})
            t_obs = t_obs.expand_dims(dim={"t_init": [t_init], "member": [1]})
        feat_all["time_observable"] = self.augment_time_observable(t_obs)
        return feat_all
    # ------------ Assemble feat_tpt dictionary ------------
    def assemble_tpt_features(self, feat_all, savedir):
        # List the features to put into feat_tpt
        # First, the features needed to define A and B: the time, the x1 coordinate, and its running mean, min, and max
        # over some time horizon. 
        num_time_delays = 30 # Units are days
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
        t_sim = feat_all["time_observable"]["t_sim"].to_numpy() 
        print(f"t_sim: min diff = {np.min(np.diff(t_sim))}, max diff = {np.max(np.diff(t_sim))}")
        # Prepare the blank observable to hold all the features. It may contain multiple t_init's and members, but it doesn't have to 
        feat_tpt = self.prepare_blank_observable(feat_all["time_observable"], feat_tpt_list)

        # --------------------
        # Check coordinates work out 
        print(f"feat_tpt.shape = {feat_tpt.shape}")
        print(f"feat_all['time_observable'].shape = {feat_all['time_observable'].shape}")
        print(f"feat_all['pc_observable'].shape = {feat_all['pc_observable'].shape}")

        # --------------------
        # Time observables
        t_names = ["t_abs","t_szn","year_szn_start","t_cal"]
        LHS = feat_tpt.loc[dict(feature=t_names)]
        RHS = feat_all["time_observable"].sel(feature=t_names)
        print(f"LHS.shape = {LHS.shape}, RHS.shape = {RHS.shape}")
        feat_tpt.loc[dict(feature=t_names)] += (
            feat_all["time_observable"].sel(feature=t_names)
        )
        print(f"Finished time observable")
        # PC observables
        pc_names = [f"pc_{level}_{i_pc}" for level in levels for i_pc in pcs]
        feat_tpt.loc[dict(feature=pc_names)] += (
            feat_all["pc_observable"].sel(feature=pc_names)
        )
        print(f"Finished PC observable")
        # Time-delayed zonal wind observables
        # TODO: fix
        for i_delay in range(num_time_delays+1):
            tidx_in = np.arange(0,len(t_sim)-i_delay)
            tidx_out = np.arange(i_delay,len(t_sim))
            ubar_names_in = [f"ubar_{level}_60" for level in levels]
            ubar_names_out = [f"ubar_{level}_60_delay{i_delay}" for level in levels]
            print(f"t_sim[tidx_out] = {t_sim[tidx_out]}")
            print(f"min diff t_sim[tidx_out] = {np.min(np.diff(t_sim))}")
            feat_tpt.loc[dict(feature=ubar_names_out,t_sim=t_sim[tidx_out])] = (
                    feat_tpt.sel(feature=ubar_names_out,t_sim=t_sim[tidx_out]) + 
                    feat_all["ubar_observable"].sel(feature=ubar_names_in)
                    .isel(t_sim=tidx_in).
                    assign_coords(t_sim=t_sim[tidx_out], feature=ubar_names_out)
                    )
            print(f"Assigned with i_delay = {i_delay}")
            if i_delay > 0:
                feat_tpt.loc[dict(feature=ubar_names_out,t_sim=t_sim[:tidx_out[0]])] = np.nan
        print(f"Finished zmzw observable")
        # Temperature observables 
        temp_names = [f"Tcap_{level}_60to90" for level in levels]
        feat_tpt.loc[dict(feature=temp_names)] += (
            feat_all["temperature_observable"].sel(feature=temp_names)
        )
        print(f"Finished temperature observable")
        # Heat flux observables
        for i_delay in range(num_time_delays+1):
            vT_names_in = [f"vT_{level}_{mode}" for level in levels for mode in modes]
            vT_names_out = [f"vT_{level}_{mode}_runavg{i_delay}" for level in levels for mode in modes]
            feat_tpt.loc[dict(feature=vT_names_out)] = (
                    feat_tpt.sel(feature=vT_names_out) + 
                    feat_all["heatflux_observable"].sel(feature=vT_names_in)
                    .rolling(t_sim=i_delay+1, center=False).mean().assign_coords({"feature": vT_names_out})
                    )
        print(f"Finished heat flux observable")
        # QBO observable
        qbo_names = [f"ubar_{level}_0pm5" for level in [10, 100]]
        feat_tpt.loc[dict(feature=qbo_names)] += (
            feat_all["qbo_observable"].sel(feature=qbo_names)
        )
        print(f"Finished QBO observable")
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
