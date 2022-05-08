# This file does TPT analysis on ERA-20C. More generally, it does TPT with a dataset of full trajectories. We will compute the committor, density, current, and lead times using only real data samples. Then visualize them on any subspace of interest. Furthermore, we will do machine learning on these point clouds to find most dangerous directions, i.e., Lyapunov vectors with respect to the event of hitting B within a given time, etc. Purely data analysis, no approximation error---only estimation error. 
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('AGG')
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'monospace', 'size': 12}
font = {'family': 'monospace', 'size': 18}
bigfont = {'family': 'monospace', 'size': 40}
giantfont = {'family': 'monospace', 'size': 80}
ggiantfont = {'family': 'monospace', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import sys
import os
from os import mkdir
from os.path import join,exists
from sklearn import linear_model
import helper
import cartopy
from cartopy import crs as ccrs
import pickle

def get_wavenumbers(ds,dssource,i_lev,lat_range,i_mem=0,num_waves=2):
    # Given a band of latitudes (a whole ensemble thereof), get waves 1 and 2
    i_lat_range = np.where((ds['lat'][:] >= lat_range[0])*(ds['lat'][:] <= lat_range[1]))[0]
    #print("i_lat_range = {}".format(i_lat_range))
    cosine = np.cos(ds['lat'][:][i_lat_range] * np.pi/180)
    cosweight = cosine/np.sum(cosine)
    gh = get_gh(ds,dssource,i_mem)
    #print("gh.shape = {}".format(gh.shape))
    fft = np.fft.fft(gh[:,i_lev,i_lat_range,:],axis=2)
    wave = np.zeros((ds['time'].size, 2*num_waves)) # real 1, imag 1, real 2, imag 2, ...
    for i in range(num_waves):
        wave[:,2*i] = fft[:,:,i+1].real.dot(cosweight)
        wave[:,2*i+1] = fft[:,:,i+1].imag.dot(cosweight)
    #mag = np.abs(fft)
    #arg = np.arctan(fft.imag/fft.real) # Gives us something between -pi and pi
    #Nlon = gh_onelev.shape[2]
    #wave1_new_mag = 2/Nlon*np.sum(mag[:,:,1]*cosweight)
    #wave1_new_phase = np.sum(-arg[:,:,1]*cosweight)
    #wave2_new_mag = 2/Nlon*np.sum(mag[:,:,2]*cosweight)
    #wave2_new_phase = np.sum(-arg[:,:,2]/2*cosweight)
    #wave1_new = np.array([wave1_new_mag,wave1_new_phase]).T
    #wave2_new = np.array([wave2_new_mag,wave2_new_phase]).T
    return wave #1_new,wave2_new

def get_gh(ds,dssource,i_mem=0):
    # Determine the prefix for geopotential height. Will depend on whether the dataset comes from s2s or era20c
    grav_accel = 9.80665
    if dssource == 's2s':
        memkey = 'gh' if i_mem==0 else 'gh_%i'%(i_mem+1)
        gh = ds[memkey][:]
    elif dssource == 'era20c':
        gh = ds['var129'][:]/grav_accel
    else:
        raise Exception("The dssource you gave me, %s, is not recognized"%(dssource))
    return gh

def get_ensemble_size(ds,dssource):
    # Count the number of members based on the number of fields that start with 'gh'
    vbls = list(ds.variables.keys())
    if dssource == 'era20c':
        return 1
    Nmem = 0
    for v in vbls:
        if v[:2] == "gh":
            Nmem += 1
    return Nmem

def get_i_lev_hPa(ds,dssource,pres):
    # Get the level with the nearest hPa
    plev = ds['plev'][:]
    if dssource in ['s2s','era20c']:
        plev *= 1.0/100
    i_lev = np.argmin(np.abs(plev - pres))
    return i_lev

def show_gh_onelevel_cartopy(ds,dssource,i_time,i_lev,i_mem=0):
    # Display the geopotential height at a single pressure level
    gh = get_gh(ds,dssource,i_mem)[i_time,i_lev,:,:]
    fig,ax,data_crs = display_pole_field(gh,ds['lat'][:],ds['lon'][:])
    ax.set_title("Geo. Hgt. at %i hPa, member %i, day %i"%(ds['plev'][:][i_lev]/100,i_mem, ds['time'][:][i_time]/24.0))
    return fig,ax

def show_ugh_onelevel_cartopy(gh,u,v,lat,lon): 
    # Display the geopotential height at a single pressure level
    fig,ax,data_crs = display_pole_field(gh,lat,lon)
    lon_subset = np.linspace(0,lon.size-1,20).astype(int)
    lat_subset = np.linspace(0,lat.size-2,60).astype(int)
    ax.quiver(lon[lon_subset],lat[lat_subset],u[lat_subset,:][:,lon_subset],v[lat_subset,:][:,lon_subset],transform=data_crs,color='black',zorder=5)
    ax.set_title(r"$\Phi$, $u$")
    return fig,ax

def display_pole_field(field,lat,lon):
    data_crs = ccrs.PlateCarree() 
    ax_crs = ccrs.Orthographic(-10,90)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection=ax_crs)
    im = ax.pcolormesh(lon,lat,field,shading='nearest',cmap='coolwarm',transform=data_crs)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=3, edgecolor='black')
    fig.colorbar(im,ax=ax)
    return fig,ax,data_crs

def compute_geostrophic_wind(ds,dssource,i_mem=0):
    gh = get_gh(ds,dssource,i_mem=i_mem)
    Omega = 2*np.pi/(3600*24*365)
    fcor = np.outer(2*Omega*np.sin(ds['lat'][:]*np.pi/180), np.ones(ds['lon'].size))
    earth_radius = 6371e3 
    dx = np.outer(earth_radius*np.cos(ds['lat'][:]*np.pi/180), np.roll(ds['lon'][:],-1) - np.roll(ds['lon'][:],1)) # this counts both sides
    dy = np.outer((np.roll(ds['lat'][:],-1) - np.roll(ds['lat'][:],1))*earth_radius, np.ones(ds['lon'].size))
    gh_x = (np.roll(gh,-1,axis=3) - np.roll(gh,1,axis=3))/dx 
    gh_y = (np.roll(gh,-1,axis=2) - np.roll(gh,1,axis=2))/dy
    u = -gh_y/fcor
    v = gh_x/fcor
    return u,v

class TPTRA:
    # A class to compute TPT quantities with reanalysis
    def __init__(self,datadir,ftdatadir,fall_year_list,wstart_min,wend_max,Npc=10):
        self.ref_pres = 10.0 # hPa
        self.ref_lat = 60.0 # degrees N
        self.lat_range = np.array([self.ref_lat-5, self.ref_lat+5]) # range of latitudes for doing FFT to get waves 1 and 2
        self.num_waves = 2 
        self.seasonality_window = 7.0 # 1 week is the averaging window for seasonality
        self.wstart_min = wstart_min
        self.wend_max = wend_max
        self.wtime = np.arange(self.wstart_min,self.wend_max)
        self.dtwint = self.wtime[1] - self.wtime[0]
        self.Npc = Npc # Determine from SVD if not specified
        self.datadir = datadir
        self.fall_year_list = fall_year_list
        self.data_file_list = []
        for fall_year in self.fall_year_list:
            self.data_file_list += [join(self.datadir,"%i-11-01_to_%i-04-30.nc"%(fall_year,fall_year+1))]
        self.ftdatadir = ftdatadir
        self.feature_filename = join(self.ftdatadir,"era20c_features")
        self.dssource = 'era20c'
        self.meta = dict()
        self.meta['wintertime'] = {'units': 'days', 'label': 'Time since Nov. 1', 'abbrv': 'wtime'}
        self.meta['u_zm_10hPa_60dN'] = {'units': 'm/s', 'label': r"$\overline{u}$ (10 hPa, 60$^\circ$N)", 'abbrv': 'u10hpa60N'}
        return
    def wave_mph(self,x,feat_def,wn):
        # wn is wavenumber 
        # mph is magnitude and phase
        if wn <= 0:
            raise Exception("Need an integer wavenumber >= 1. You gave wn = {}".format(wn))
        wti = np.round((x[:,0] - self.wstart_min)/self.dtwint).astype(int)
        wti = np.maximum(0, np.minimum(len(self.wtime)-1, wti))
        wave = x[:,(2 + 2*(wn-1)):(2 + 2*(wn-1) + 1)] * feat_def["seasonal_std_wave"][wti] + feat_def["seasonal_mean_wave"][wti]
        mag = np.sum(wave**2, axis=1)
        phase = np.arctan(-wave[:,1]/(wn*wave[:,0]))
        return np.array([mag,phase]).T
    def observable_function_library(self):
        # Build the database of observable functions
        feat_def = pickle.load(open(join(self.ftdatadir,"struct_feat_def"),"rb"))
        # TODO: build in PC projections and other stuff as observable functions
        funlib = {
                "time": {"fun": lambda x: x[:,0],
                    "label": r"Days since Nov. 1",},
                "uref": {"fun": lambda x: x[:,1]*self.uref_std + self.uref_mean,
                    "label": r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]",},
                "mag1": {"fun": lambda x: self.wave_mph(x,feat_def,1)[:,0],
                    "label": "Wave 1 magnitude",},
                "mag2": {"fun": lambda x: self.wave_mph(x,feat_def,2)[:,0],
                    "label": "Wave 2 magnitude",},
                "ph1": {"fun": lambda x: self.wave_mph(x,feat_def,1)[:,1],
                    "label": "Wave 1 phase",},
                "ph2": {"fun": lambda x: self.wave_mph(x,feat_def,2)[:,1],
                    "label": "Wave 2 phase",},
                }
        return funlib
    def vectorize_ensemble(self,ds):
        # Take a netcdf file with records in order and put it into a big matrix. This is where the ordering is defined. 
        # Output shape: (Nmem,Nt,xdim) (the first entry is for time (days since Nov. 1))
        # This can then be put into a bigger array with Nens as the first dimension
        Nmem = get_ensemble_size(ds,self.dssource)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
        xdim = Nlev*Nlat*Nlon + 1
        x = np.zeros((Nmem,Nt,xdim))
        for i_mem in range(Nmem):
            gh = get_gh(ds,self.dssource,i_mem=i_mem)
            x[i_mem,:,0] = ds['time'][:] # Stick to hourly
            x[i_mem,:,:] = gh.reshape((Nt,xdim))
        return x
    def create_structured_features(self,flist=None):
        # This is where we define the EOF basis functions and store the info as a file. 
        # Before doing anything, deseasonalize
        # 1st function: zonal-mean zonal wind at 60N, 10 hPa
        # 2nd function: wavenumber 1 at 10 hPa
        # 3rd function: wavenumber 2 at 10 hPa
        # After that: first 3 EOFs at every single level
        if flist is None:
            flist = self.data_file_list
        ds0 = nc.Dataset(flist[0],"r")
        lat,lon,plev = [ds0[v][:] for v in ['lat','lon','plev']]
        i_lev = get_i_lev_hPa(ds0,self.dssource,self.ref_pres)
        i_lat = np.argmin(np.abs(lat - self.ref_lat))
        cosine = np.cos(np.pi/180 * lat)
        #cosine = np.zeros((len(plev),len(lat),len(lon)))
        #for i in range(len(plev)):
        #    cosine[i,:,:] = np.cos(np.pi/180 * np.outer(lat,np.ones(len(lon))))
        Nmem = get_ensemble_size(ds0,self.dssource)
        Nlev,Nlat,Nlon = [ds0[v].size for v in ['plev','lat','lon']]
        Ngrid = Nlev*Nlat*Nlon
        ds0.close()
        # All the level-by-level EOF stuff will happen using the retained 3D arrays
        gh = np.zeros((0,Nlev,Nlat*Nlon))
        T = np.zeros(0)
        uref = np.zeros(0)
        wave = np.zeros((0,2*self.num_waves))
        for i in range(len(flist)):
            if i % 10 == 0:
                print("Reading file {} out of {}".format(i,len(flist)))
                #print("gh current shape = {}".format(gh.shape))
            ds = nc.Dataset(flist[i],"r")
            ghi = get_gh(ds,self.dssource,i_mem=0)
            T = np.concatenate((T,ds['time'][:]))
            ghnew = ghi.reshape((ds['time'].size,Nlev,Nlat*Nlon))
            gh = np.concatenate((gh,ghnew),axis=0)
            u,v = compute_geostrophic_wind(ds,self.dssource,i_mem=0)
            uref_new = np.mean(u[:,i_lev,i_lat,:],axis=1)
            uref = np.concatenate((uref,uref_new))
            # Now get the wave-1 and wave-2 components
            wave_new = get_wavenumbers(ds,self.dssource,i_lev,self.lat_range,i_mem=0,num_waves=self.num_waves)
            wave = np.concatenate((wave,wave_new),axis=0)
            ds.close()
        # Now EOF analysis
        seasonal_time = np.arange(T.min()/24.0, T.max()/24.0+self.dtwint, self.dtwint)
        seasonal_mean_gh = np.zeros((len(seasonal_time),Nlev,Nlat*Nlon))
        seasonal_mean_wave = np.zeros((len(seasonal_time),2*self.num_waves))
        seasonal_std_wave = np.zeros((len(seasonal_time),2*self.num_waves))
        for ti in range(len(seasonal_time)):
            idx = np.where(np.abs(T/24.0 - seasonal_time[ti]) < self.seasonality_window)[0]
            seasonal_mean_gh[ti] = np.mean(gh[idx],axis=0)
            seasonal_mean_wave[ti] = np.mean(wave[idx],axis=0)
            seasonal_std_wave[ti] = np.std(wave[idx],axis=0)
        gh_unseasoned = gh.copy()
        wave_unseasoned = wave.copy()
        wti = np.round((T/24.0 - seasonal_time[0])/self.dtwint).astype(int)
        print("wti = {}".format(wti))
        gh_unseasoned = gh - seasonal_mean_gh[wti]
        print("gh.shape = {}, gh_unseasoned.shape = {}".format(gh.shape,gh_unseasoned.shape))
        #for i in range(len(gh)):
        #    ti = np.argmin(self.wtime - np.abs(T[i]/24.0))
        #    gh_unseasoned[i] = gh[i] - seasonal_mean_gh[ti]
        # Perform SVD level by level
        weight = 1/np.sqrt(len(gh_unseasoned))*np.outer(np.sqrt(cosine),np.ones(Nlon)).flatten()
        eofs = np.zeros((Nlev,Nlat*Nlon,self.Npc))
        singvals = np.zeros((Nlev,self.Npc))
        print("Starting SVD level by level")
        for j_lev in range(Nlev):
            print("\tj_lev = {}".format(j_lev))
            U,S,Vh = np.linalg.svd((gh_unseasoned[:,j_lev,:]*weight).T, full_matrices=False) # columns of U are EOFs 
            eofs[j_lev,:,:] = U[:,:self.Npc]
            singvals[j_lev,:] = S[:self.Npc]
        self.Nfeat = self.Npc*Nlev + 2 + 2*self.num_waves # Time, zonal-mean zonal wind, wave 1 (real and imaginary), wave 2 (real and imaginary)
        print("self.Npc = {} out of {}".format(self.Npc,len(S)))
        # Save mean and variance of zonal wind
        self.uref_mean = np.mean(uref)
        self.uref_std = np.std(uref)
        # In general, for a linear or nonlinear dimensionality reduction, we need a function to project the data. In the EOF case, that means U, S, (not V), and Xmean.
        # Record mean and variance of zonal mean
        # Save the EOFs as their own netcdf
        feat_def = {"Nfeat": self.Nfeat, "eofs": eofs, "singvals": singvals, "seasonal_time": seasonal_time, "seasonal_mean_gh": seasonal_mean_gh, "seasonal_mean_wave": seasonal_mean_wave, "seasonal_std_wave": seasonal_std_wave, "lat": lat, "lon": lon, "plev": plev, "dssource": self.dssource, "uref_mean": self.uref_mean, "uref_std": self.uref_std, "seasonality_window": self.seasonality_window}
        pickle.dump(feat_def,open(join(self.ftdatadir,"struct_feat_def"),"wb"))
        return 
    def create_features(self,Npc=None,flist=None):
        # This is where we define the EOF basis functions and store the info as a file. 
        # Before doing anything, deseasonalize
        # 1st function: zonal-mean zonal wind at 60N, 10 hPa
        # 2nd function: wavenumber 1 at 10 hPa
        # 3rd function: wavenumber 2 at 10 hPa
        # After that: first 3 EOFs at every single level
        if flist is None:
            flist = self.data_file_list
        ds0 = nc.Dataset(flist[0],"r")
        lat,lon,plev = [ds0[v][:] for v in ['lat','lon','plev']]
        i_lev = get_i_lev_hPa(ds0,self.dssource,self.ref_pres)
        i_lat = np.argmin(np.abs(lat - self.ref_lat))
        cosine = np.zeros((len(plev),len(lat),len(lon)))
        for i in range(len(plev)):
            cosine[i,:,:] = np.cos(np.pi/180 * np.outer(lat,np.ones(len(lon))))
        cosine = cosine.flatten()
        Nmem = get_ensemble_size(ds0,self.dssource)
        Nlev,Nlat,Nlon = [ds0[v].size for v in ['plev','lat','lon']]
        Ngrid = Nlev*Nlat*Nlon
        ds0.close()
        X = np.zeros((0,Ngrid+1))
        uref = np.zeros(0)
        for i in range(len(flist)):
            if i % 10 == 0:
                print("Reading file {} out of {}".format(i,len(flist)))
                print("X current shape = {}".format(X.shape))
            ds = nc.Dataset(flist[i],"r")
            Xnew = self.vectorize_ensemble(ds).reshape((Nmem*ds['time'].size,Ngrid))
            X = np.concatenate((X,Xnew),axis=0)
            u,v = compute_geostrophic_wind(ds,self.dssource)
            uref_new = np.mean(u[:,i_lev,i_lat,:],axis=1)
            uref = np.concatenate((uref,uref_new))
            ds.close()
        # Before demeaning, deseasonalize
        X_demeaned = X.copy()
        seasonal_mean = np.zeros((self.wend_max-self.wstart_min,Ngrid+1))
        for ti in range(self.wstart_min,self.wend_max):
            idx = np.where(np.abs(X[:,0]/24.0 - ti) < 7)[0]
            seasonal_mean[ti] = np.mean(X[idx],axis=0)
            X_demeaned[idx] -= seasonal_mean[ti]
        # Perform SVD level by level

        print("About to perform SVD")
        # Now do SVD 
        U,S,Vh = np.linalg.svd((X_demeaned*np.sqrt(cosine)).T/np.sqrt(len(X)), full_matrices=False) # Columns of U are spatial patterns; rows of Vh are coefficients
        print("S.shape = {}, S[:20] = {}".format(S.shape,S[:20]))
        # Truncate at 90% of variance
        running_var = np.cumsum(S**2)
        if self.Npc is None:
            self.Npc = np.where(running_var/running_var[-1] >= 0.9)[0][0]
        self.Nfeat = self.Npc + 2 # One for time, and one for ZMZW
        print("self.Npc = {} out of {}".format(self.Npc,len(S)))
        # Save mean and variance of zonal wind
        self.uref_mean = np.mean(uref)
        self.uref_std = np.std(uref)
        # In general, for a linear or nonlinear dimensionality reduction, we need a function to project the data. In the EOF case, that means U, S, (not V), and Xmean.
        # Record mean and variance of zonal mean
        # Save the EOFs as their own netcdf
        feat_def = {"Nfeat": self.Nfeat, "left_singvecs": U[:,:self.Npc].reshape((len(plev),len(lat),len(lon),self.Npc)), "singvals": S, "mean": Xmean.reshape((len(plev),len(lat),len(lon))), "lat": lat, "lon": lon, "plev": plev, "dssource": self.dssource, "uref_mean": self.uref_mean, "uref_std": self.uref_std}
        pickle.dump(feat_def,open(join(self.ftdatadir,"feat_def"),"wb"))
        return 
    def structured_featurize_ensemble(self,ds):
        # Note: at some point, this may advance to include time-delay information. 
        feat_def = pickle.load(open(join(self.ftdatadir,"struct_feat_def"),"rb"))
        Nmem = get_ensemble_size(ds,self.dssource)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
        i_lev = get_i_lev_hPa(ds,self.dssource,self.ref_pres)
        i_lat = np.argmin(np.abs(ds['lat'][:] - self.ref_lat))
        # Set up the feature vector
        x = np.zeros((Nmem,Nt,self.Nfeat))
        # Put in zonal wind
        for i_mem in range(Nmem):
            u,v = compute_geostrophic_wind(ds,self.dssource,i_mem=i_mem)
            x[i_mem,:,0] = ds['time'][:]/24.0
            x[i_mem,:,1] = (np.mean(u[:,i_lev,i_lat,:],axis=1) - feat_def["uref_mean"])/feat_def["uref_std"]
            wave = get_wavenumbers(ds,self.dssource,i_lev,self.lat_range,i_mem=i_mem,num_waves=self.num_waves)
            # Unseason wave
            wti = np.round((ds['time'][:]/24.0 - self.wstart_min)/self.dtwint).astype(int)
            x[i_mem,:,2:2*self.num_waves+2] = (wave - feat_def["seasonal_mean_wave"][wti])/feat_def["seasonal_std_wave"][wti]
        # Project PCs 
        pc = self.structured_project_onto_eofs(ds,feat_def=feat_def,Npc_per_level=self.Npc*np.ones(Nlev,dtype=int))
        #print("pc.shape = {}".format(pc.shape))
        x[:,:,2*self.num_waves+2:self.Npc*Nlev+2*self.num_waves+2] = pc
        return x
    def featurize_ensemble(self,ds):
        # Note: at some point, this may advance to include time-delay information. 
        feat_def = pickle.load(open(join(self.ftdatadir,"feat_def"),"rb"))
        Nmem = get_ensemble_size(ds,self.dssource)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
        i_lev = get_i_lev_hPa(ds,self.dssource,self.ref_pres)
        i_lat = np.argmin(np.abs(ds['lat'][:] - self.ref_lat))
        # Set up the feature vector
        x = np.zeros((Nmem,Nt,self.Nfeat))
        # Put in zonal wind
        for i_mem in range(Nmem):
            u,v = compute_geostrophic_wind(ds,self.dssource)
            x[i_mem,:,1] = np.mean(u[:,i_lev,i_lat,:],axis=1)
            x[i_mem,:,0] = ds['time'][:]/24.0
        x[:,:,1] = (x[:,:,1] - feat_def["uref_mean"])/feat_def["uref_std"]
        # Project PCs 
        x[:,:,2:self.Npc+2] = self.project_onto_eofs(ds)
        return x
    def plot_vortex_evolution(self,i_year,num_snapshots=300):
        # Plot a series of observables over the whole year.
        # First, determine the range of maximum deceleration.
        funlib = self.observable_function_library()
        x = np.load(join(self.ftdatadir,"features.npy"))[i_year,:,:]
        time = funlib["time"]["fun"](x)
        decel_window = int(10/(time[1]-time[0]))
        # Determine the time range as the 30 days bracketing the 10-day period of maximum deceleration of zonal wind
        uref = funlib["uref"]["fun"](x)
        decel10 = uref[decel_window:] - uref[:-decel_window]
        start = np.argmin(decel10)
        decel_time_range = [time[max(0,start-decel_window)],time[min(len(time)-1, start+2*decel_window)]]
        full_time_range = np.array([self.wstart_min,self.wend_max])
        obs_key_list = ["uref","mag1","mag2"]
        for oki in range(len(obs_key_list)):
            obs_key = obs_key_list[oki]
            fig,ax = self.plot_observable_timeseries(x,full_time_range,obs_key)
            ax.axvspan(decel_time_range[0],decel_time_range[1],color='orange',zorder=-1)
            fig.savefig(join(self.ftdatadir,"time_%s_%s"%(obs_key,self.fall_year_list[i_year])))
            plt.close(fig)
        # Given an netcdf filename, plot the vortex evolution in geopotential height
        fname = self.data_file_list[i_year]
        ds = nc.Dataset(fname,"r")
        i_lev = get_i_lev_hPa(ds,self.dssource,self.ref_pres)
        tidx = np.linspace(decel_time_range[0],decel_time_range[1],min(num_snapshots,decel_time_range[1]-decel_time_range[0]+1)).astype(int)
        gh = get_gh(ds,self.dssource,i_mem=0)[tidx,i_lev,:,:]
        u,v = compute_geostrophic_wind(ds,self.dssource,i_mem=0)
        u = u[tidx,i_lev,:,:]
        v = v[tidx,i_lev,:,:]
        for k in range(len(tidx)):
            i_time = tidx[k]
            fig,ax = show_ugh_onelevel_cartopy(gh[k],u[k],v[k],ds['lat'][:],ds['lon'][:])
            ax.set_title(r"$\Phi$, $u$ at day {}, {}".format(ds['time'][tidx[k]]/24.0,self.fall_year_list[i_year]))
            fig.savefig(join(self.ftdatadir,"vortex_fy{}_day{}".format(self.fall_year_list[i_year],int(ds['time'][tidx[k]]/24.0))))
            plt.close(fig)
        return
    def featurize_database(self,flist=None,fname=None):
        # Now stack together a bunch of featurizations. Result will be (Nx,Nfeat) -- no distinguishing different ensembles and times, because the different netcdf's in the database might have different timespans.
        Nt = len(self.wtime) # This will be a fixed time horizon for everyone. 
        if flist is None:
            flist = self.data_file_list
        ds0 = nc.Dataset(flist[0],"r")
        ds0.close()
        X = np.zeros((0,Nt,self.Nfeat))
        if fname is None:
            fname = "features"
        print("Featurizing database")
        for i in range(len(flist)):
            if i % 10 == 0:
                print("Reading file {} out of {}".format(i,len(flist)))
            ds = nc.Dataset(flist[i],"r")
            Xnew = self.structured_featurize_ensemble(ds)
            tidx = np.where((ds['time'][:]/24 >= self.wstart_min)*(ds['time'][:]/24 < self.wend_max))[0]
            if len(tidx) != Nt:
                raise Exception("Time dimension of Xnew and X don't match")
            X = np.concatenate((X,Xnew[:,tidx,:]),axis=0)
            ds.close()
        # Now save it to the ftdatadir
        self.ftdata_filename = join(self.ftdatadir,fname)
        np.save(join(self.ftdatadir,self.ftdata_filename),X)
        return #X
    def read_feature_database(self):
        X = np.load(self.ftdata_filename+".npy")
        return X
    def structured_project_onto_eofs(self,ds,feat_def=None,Npc_per_level=None):
        # Take a dataset and project it onto the EOFs to determine the reduced coordinates.
        Nmem = get_ensemble_size(ds,self.dssource)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
        if feat_def is None:
            feat_def = pickle.load(open(join(self.ftdatadir,"struct_feat_def"),"rb"))
        if Npc_per_level is None:
            Npc_per_level = self.Npc*np.ones(Nlev,dtype=int)
        # Unseason
        seasonal_mean_ds = np.nan*np.ones((Nt,Nlev,Nlat*Nlon))
        wti = np.round((ds['time'][:]/24.0 - feat_def["seasonal_time"][0])/self.dtwint).astype(int)
        wti = np.maximum(0,np.minimum(len(feat_def["seasonal_time"])-1, wti))
        seasonal_mean_ds = feat_def["seasonal_mean_gh"][wti]
        Y = np.zeros((Nmem*Nt,Nlev,Nlat*Nlon))
        for i_mem in range(Nmem):
            Y[i_mem*Nt:(i_mem+1)*Nt,:,:] = get_gh(ds,self.dssource,i_mem=i_mem).reshape((Nt,Nlev,Nlat*Nlon)) - seasonal_mean_ds
        pc = np.zeros((Nmem*Nt,np.sum(Npc_per_level)))
        i_pc = 0
        for i_lev in range(Nlev):
            pc[:,i_pc:i_pc+Npc_per_level[i_lev]] = Y[:,i_lev,:].dot(feat_def["eofs"][i_lev,:,:])
            pc[:,i_pc:i_pc+Npc_per_level[i_lev]] *= 1.0/feat_def["singvals"][i_lev,:self.Npc]
            i_pc += Npc_per_level[i_lev]
            #pc[:,i_lev,:] = (Y[:,i_lev,:].dot(feat_def["eofs"][i_lev,:,:]))
            #pc[:,i_lev,:] *= 1.0/feat_def["singvals"][i_lev,:self.Npc]
        pc = pc.reshape((Nmem,Nt,np.sum(Npc_per_level)))
        return pc
    def project_onto_eofs(self,ds):
        # Take a dataset and project it onto the EOFs to determine the reduced coordinates.
        feat_def = pickle.load(open(join(self.ftdatadir,"feat_def"),"rb"))
        Nmem = get_ensemble_size(ds,self.dssource)
        Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
        Y = self.vectorize_ensemble(ds) 
        Y = Y.reshape((Nmem*Nt, Nlev*Nlat*Nlon))
        Y = Y - feat_def["mean"].flatten()
        pc = Y.dot(feat_def["left_singvecs"].reshape((Nlev*Nlat*Nlon,self.Npc)))
        pc = pc/feat_def["singvals"][:self.Npc]
        pc = pc.reshape((Nmem,Nt,self.Npc))
        return pc
    def set_physical_params(self,savedir,uthresh,wstart,wend):
        self.savedir = savedir
        self.wstart = wstart # Defines the beginning of SSW-fair-game winter (days since Nov. 1)
        self.wend = wend 
        if wend > self.wend_max:
            sys.exit("ERROR: wend too big")
        self.uthresh = uthresh
        return
    def ina_test(self,x):
        # Important: x must be 2-dimensional. Don't care whether the rows came from different ensembles or times.
        # Test whether a reanalysis dataset's components are in A
        Nx,xdim = x.shape
        ina = np.zeros(Nx,dtype=bool)
        nonwinter_idx = np.where((x[:,0] >= self.wstart) * (x[:,0] < self.wend) == 0)[0]
        ina[nonwinter_idx] = True
        return ina
    def inb_test(self,x):
        # Test whether a reanalysis dataset's components are in B
        Nx,xdim = x.shape
        inb = np.zeros(Nx, dtype=bool)
        winter_idx = np.where((x[:,0] >= self.wstart)*(x[:,0] < self.wend))[0]
        weak_wind_idx = np.where(self.uref_mean + self.uref_std*x[winter_idx,1] < self.uthresh)
        inb[winter_idx[weak_wind_idx]] = True
        return inb
    def compute_src_dest_tags(self,x):
        # Each member could have a different time horizon--it's fine. 
        Nmem,Nt = x.shape[:2]
        ina = self.ina_test(x.reshape((Nmem*Nt,self.Nfeat))).reshape((Nmem,Nt))
        inb = self.inb_test(x.reshape((Nmem*Nt,self.Nfeat))).reshape((Nmem,Nt))
        source_tag = 0.5*np.ones((Nmem,Nt))
        dest_tag = 0.5*np.ones((Nmem,Nt))
        # Source: move forward in time
        # Time zero, A is the default source
        source_tag[:,0] = 0*ina[:,0] + 1*inb[:,0] + 0.5*(ina[:,0]==0)*(inb[:,0]==0)*(x[:,0,0] > self.wstart) 
        for k in range(1,Nt):
            source_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + source_tag[:,k-1]*(ina[:,k]==0)*(inb[:,k]==0)
        # Dest: move backward in time
        # Time end, A is the default dest
        dest_tag[:,Nt-1] = 0*ina[:,Nt-1] + 1*inb[:,Nt-1] + 0.5*(ina[:,Nt-1]==0)*(inb[:,Nt-1]==0)*(x[:,-1,0] < self.wend)
        for k in np.arange(Nt-2,-1,-1):
            dest_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + dest_tag[:,k+1]*(ina[:,k]==0)*(inb[:,k]==0)
        #print("Overall fraction in B = {}".format(np.mean(inb)))
        #print("At time zero: fraction of traj in B = {}, fraction of traj headed to B = {}".format(np.mean(dest_tag[:,0]==1),np.mean((dest_tag[:,0]==1)*(inb[:,0]==0))))
        return source_tag,dest_tag
    def compute_arrival_times(self,ds):
        # Load the reduced dataset and compute the lead time from every point
        # also compute the lead time from every year.
        ina = self.ina_test(ds)
        inb = self.inb_test(ds)
        kmin = np.where(ds['trajtime'][:] >= self.wstart)[0][0]
        kmax = np.where(ds['trajtime'][:] < self.wend)[0][-1] + 1
        Ninit,Nens,Nt = ds['initidx'].size,ds['ensmemb'].size,ds['trajtime'].size
        taup = np.zeros((Ninit,Nens,Nt))  # Know the end time is in B
        for k in np.arange(Nt-2,-1,-1):
            dt = ds['trajtime'][:][k+1] - ds['trajtime'][:][k]
            taup[:,:,k] = 0*(ina[:,:,k] + inb[:,:,k]) + (dt+taup[:,:,k+1]) * (ina[:,:,k]==0)*(inb[:,:,k]==0)
        return taup
    def plot_lead_time(self,var1):
        ds = nc.Dataset(self.feature_filename,"r")
        Ninit,Nens,Nt = ds['initidx'].size,ds['ensmemb'].size,ds['trajtime'].size
        taup = self.compute_arrival_times(ds)
        if np.any(np.isnan(taup)):
            sys.exit("ERROR: some taups are nan")
        source_tag,dest_tag = self.compute_src_dest_tags(ds)
        qp = 1.0*(dest_tag.flatten()==1)
        lead_time = taup.flatten()*(qp != 0)/(qp + 1*(qp == 0))
        lead_time[qp==0] = np.nan
        theta_x = np.zeros((Ninit*Nens*Nt,2))
        theta_x[:,0] = np.outer(np.ones(Ninit*Nens),ds['trajtime'][:]).flatten()
        theta_x[:,1] = ds[var1][:].flatten()
        weight = np.ones(Ninit*Nens*Nt)/(Ninit*Nens*Nt)
        fun0name = "{} [{}]".format(self.meta['wintertime']['label'],self.meta['wintertime']['units'])
        fun1name = "{} [{}]".format(self.meta[var1]['label'],self.meta[var1]['units'])
        fig,ax = helper.plot_field_2d(lead_time,weight,theta_x,shp=[15,15],fieldname="ERA-20C lead time",fun0name=fun0name,fun1name=fun1name,contourflag=True,vmin=0,vmax=self.wend-self.wstart)
        ds.close()
        return fig,ax
    def plot_predictability(self,var1):
        ds = nc.Dataset(self.feature_filename,"r")
        Ninit,Nens,Nt = ds['initidx'].size,ds['ensmemb'].size,ds['trajtime'].size
        source_tag,dest_tag = self.compute_src_dest_tags(ds)
        if np.any((source_tag!=0)*(source_tag!=1) + (dest_tag!=0)*(dest_tag!=1)):
            sys.exit("ERROR: some sources and destinations are not defined")
        qp = 1.0*(dest_tag.flatten()==1)
        taup = self.compute_arrival_times(ds)
        if np.any(np.isnan(taup)):
            sys.exit("ERROR: some taups are nan")
        lead_time = taup.flatten()*(qp != 0)/(qp + 1*(qp == 0))
        lead_time[qp==0] = np.nan
        etab = self.wstart + taup[:,:,0]*(dest_tag[:,:,0]==1)
        etab[dest_tag[:,:,0]==0] = np.nan
        etab = etab.flatten()
        etab = etab[np.isnan(etab)==0]
        # Histogram it
        hist,bin_edges = np.histogram(etab,bins=int(Nt/18),range=(self.wstart,self.wend),density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        theta_x = np.zeros((Ninit*Nens*Nt,2))
        theta_x[:,0] = np.outer(np.ones(Ninit*Nens),ds['trajtime'][:]).flatten()
        theta_x[:,1] = ds[var1][:].flatten()
        weight = np.ones(Ninit*Nens*Nt)/(Ninit*Nens*Nt)
        fun0name = "{} [{}]".format(self.meta['wintertime']['label'],self.meta['wintertime']['units'])
        fun1name = "{} [{}]".format(self.meta[var1]['label'],self.meta[var1]['units'])
        fig,ax = plt.subplots(ncols=2,figsize=(12,6),sharey=True)
        _,_ = helper.plot_field_2d(qp,weight,theta_x,shp=[15,15],fieldname="ERA-20C committor",fun0name=fun0name,fun1name=fun1name,contourflag=True,fig=fig,ax=ax[0])
        _,_ = helper.plot_field_2d(lead_time,weight,theta_x,shp=[15,15],fieldname="ERA-20C lead time",fun0name=fun0name,fun1name=fun1name,contourflag=True,vmin=0,vmax=self.wend-self.wstart,fig=fig,ax=ax[1])
        pcoords = self.uthresh + hist*5*(theta_x[:,1].max()-theta_x[:,1].min())
        for i in range(2):
            # Now also put in the arrival times.
            ax[i].plot([self.wstart,self.wend],self.uthresh*np.ones(2),color='black',linestyle='--',linewidth=3)
            #ax[i].scatter(etab,self.uthresh*np.ones(len(etab)),s=144,color='cyan')
            #ax[i].plot(bin_centers,pcoords,color='black',zorder=5)
        ax[1].yaxis.set_visible(False)
        ds.close()
        return fig,ax,bin_centers,hist
    def plot_observable_timeseries(self,x,time_range,obs_key):
        # Plot a timeseries of a given observable
        # x is (time,xdim)
        # i_traj refers to one member of one ensemble
        funlib = self.observable_function_library()
        fig,ax = plt.subplots()
        time = funlib["time"]["fun"](x)
        tidx = np.where((time >= time_range[0])*(time <= time_range[1]))[0]
        obs = funlib[obs_key]["fun"](x)
        ax.plot(time,obs,color='black')
        ax.set_xlabel(funlib["time"]["label"])
        ax.set_ylabel(funlib[obs_key]["label"])
        return fig,ax
    def plot_forward_committor_2d(self,obs_keys):
        # given an observable "object", load all the data and compute the observable thereon. Also compute source and destination tags. Then plot against that observable. 
        # TODO: implement the obs object and compute it one dataset at a time. 
        x = np.load(join(self.ftdatadir,"features.npy"))
        src_tag,dest_tag = self.compute_src_dest_tags(x)
        #print("mean(dest_tag) = {}".format(dest_tag.mean()))
        #print("mean(dest_tag==1) = {}".format(np.mean(dest_tag==1)))
        x = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
        qp = 1.0*(dest_tag.flatten() == 1)
        Nx = len(qp)
        funlib = self.observable_function_library()
        theta_x = np.zeros((x.shape[0],2))
        theta_x[:,0] = funlib[obs_keys[0]]["fun"](x)
        theta_x[:,1] = funlib[obs_keys[1]]["fun"](x)
        fun0name = funlib[obs_keys[0]]["label"]
        fun1name = funlib[obs_keys[1]]["label"]
        #theta_x = x[:,[obs_idx[0],obs_idx[1]]]
        weight = np.ones(Nx)/Nx
        fig,ax = helper.plot_field_2d(qp,weight,theta_x,shp=[15,15],fieldname="ERA-20C committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
        return fig,ax
    def compute_rate(self):
        # Find which winters, and what fraction, undergo SSW
        x = np.load(join(self.ftdatadir,"features.npy"))
        Nx,Nt,xdim = x.shape
        inb = self.inb_test(x.reshape((Nx*Nt,xdim))).reshape((Nx,Nt))
        ssw_flag = np.any(inb==1, axis=1)
        rate = np.mean(ssw_flag)
        print("Next rate = {}".format(rate))
        return rate




