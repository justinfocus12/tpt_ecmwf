# Make a class to handle different, and heterogeneous, types of SSW data. Retain knowledge of multiple file sources at once, and mix and match them (with different durations and everything). Be able to work with any subset of the data for feature creation, training, and testing. Note that the minimal unit for training and testing is a file, i.e., an ensemble. The subset indices could be generated randomly externally. 
# Ultimately, create the information to compute any quantity of interest on new, unseen data: committors and lead times crucially. Also backward quantities and rates, given an initial distribution.
# the DGA_SSW object will have no knowledge of what year anything is; that will be implmented at a higher level. 
# Maybe even have variable lag time?
import numpy as np
import multiprocessing as MP
import netCDF4 as nc
import datetime
import time as timelib
import matplotlib
matplotlib.use('AGG')
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

# Functions from stack-overflow for using starmap with keyword arguments
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def reduced_svd(A):
    print(f"Process {os.getpid()} is about to compute a reduced SVD")
    return np.linalg.svd(A, full_matrices=False)

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

class WinterStratosphereFeatures:
    # Create a set of features, including out-of-sample extension. 
    def __init__(self,feature_file,winter_day0,spring_day0,delaytime_days=0,Npc_per_level_max=10,num_vortex_moments_max=4):
        self.feature_file = feature_file
        self.winter_day0 = winter_day0
        self.spring_day0 = spring_day0
        self.wtime = 24.0 * np.arange(self.winter_day0,self.spring_day0) # All times are in hours
        self.Ntwint = len(self.wtime)
        self.szn_hour_window = 5.0*24 # Number of days around which to average when unseasoning
        self.dtwint = self.wtime[1] - self.wtime[0]
        self.delaytime = delaytime_days*24.0 
        self.ndelay = int(self.delaytime/self.dtwint) + 1
        self.Npc_per_level_max = Npc_per_level_max # Determine from SVD if not specified
        self.num_vortex_moments_max = num_vortex_moments_max
        self.num_wavenumbers = 2 # How many wavenumbers to look at 
        self.lat_uref = 60 # Degrees North for CP07 definition of SSW
        self.lat_range_uref = self.lat_uref + 5.0*np.array([-1,1])
        self.pres_uref = 10 # hPa for CP07 definition of SSW
        return
    def compute_src_dest_tags(self,Y,feat_def,tpt_bndy,save_filename):
        # Compute where each trajectory started (A or B) and where it's going (A or B). Also maybe compute the first-passage times, forward and backward.
        Nmem,Nt,Nfeat = Y.shape
        ina = self.ina_test(Y.reshape((Nmem*Nt,Nfeat)),feat_def,tpt_bndy)
        ina = ina.reshape((Nmem,Nt))
        inb = self.inb_test(Y.reshape((Nmem*Nt,Nfeat)),feat_def,tpt_bndy)
        inb = inb.reshape((Nmem,Nt))
        src_tag = 0.5*np.ones((Nmem,Nt))
        dest_tag = 0.5*np.ones((Nmem,Nt))
        # Source: move forward in time
        # Time zero, A is the default src
        src_tag[:,0] = 0*ina[:,0] + 1*inb[:,0] + 0.5*(ina[:,0]==0)*(inb[:,0]==0)*(Y[:,0,0] > tpt_bndy['tthresh'][0]) 
        for k in range(1,Nt):
            src_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + src_tag[:,k-1]*(ina[:,k]==0)*(inb[:,k]==0)
        # Dest: move backward in time
        # Time end, A is the default dest
        dest_tag[:,Nt-1] = 0*ina[:,Nt-1] + 1*inb[:,Nt-1] + 0.5*(ina[:,Nt-1]==0)*(inb[:,Nt-1]==0)*(Y[:,-1,0] < tpt_bndy['tthresh'][1])
        for k in np.arange(Nt-2,-1,-1):
            dest_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + dest_tag[:,k+1]*(ina[:,k]==0)*(inb[:,k]==0)
        #print("Overall fraction in B = {}".format(np.mean(inb)))
        #print("At time zero: fraction of traj in B = {}, fraction of traj headed to B = {}".format(np.mean(dest_tag[:,0]==1),np.mean((dest_tag[:,0]==1)*(inb[:,0]==0))))
        result = {'src_tag': src_tag, 'dest_tag': dest_tag}
        pickle.dump(result,open(save_filename,'wb'))
        return src_tag,dest_tag
    def ina_test(self,y,feat_def,tpt_bndy):
        Ny,ydim = y.shape
        ina = np.zeros(Ny,dtype=bool)
        # Now look for midwinter times with strong wind and significant time since previous SSW
        i_time = self.fidx_Y['time_h']
        i_uref = np.array([self.fidx_Y['uref_dl%i'%(i_dl)] for i_dl in range(self.ndelay)])
        winter_flag = (y[:,i_time] >= tpt_bndy['tthresh'][0])*(y[:,i_time] < tpt_bndy['tthresh'][1])
        #print(f"winter_flag = {winter_flag}")
        nonwinter_flag = (winter_flag == 0)
        #print(f"nonwinter_flag = {nonwinter_flag}")
        nbuffer = int(round(tpt_bndy['sswbuffer']/self.dtwint))
        uref = y[:,i_uref] #self.uref_history(y,feat_def)
        weak_wind_flag = (np.min(uref[:,:self.ndelay-nbuffer-1], axis=1) < tpt_bndy['uthresh_b'])  # This has to be defined from the Y construction
        strong_wind_flag = (uref[:,-1] > tpt_bndy['uthresh_a'])
        ina = nonwinter_flag + winter_flag*(1 - weak_wind_flag)*strong_wind_flag
        return ina
    def inb_test(self,y,feat_def,tpt_bndy):
        # Test whether a reanalysis dataset's components are in B
        Ny,ydim = y.shape
        i_time = self.fidx_Y['time_h']
        i_uref = self.fidx_Y['uref_dl0'] #np.array([self.fidx_Y['uref_dl%i'%(i_dl)] for i_dl in range(self.ndelay)])
        inb = np.zeros(Ny, dtype=bool)
        winter_flag = (y[:,i_time] >= tpt_bndy['tthresh'][0])*(y[:,i_time] < tpt_bndy['tthresh'][1])
        nbuffer = int(round(tpt_bndy['sswbuffer']/self.dtwint))
        uref = y[:,i_uref] #self.uref_history(y,feat_def)[:,-1]
        weak_wind_flag = (uref < tpt_bndy['uthresh_b'])
        inb = winter_flag*weak_wind_flag
        return inb
    def get_ilev_ilat(self,ds):
        # Get the latitude and longitude indices
        ds_plev_hPa = ds['plev'][:]
        if ds['plev'].units == 'Pa':
            ds_plev_hPa *= 1.0/100
        i_lev = np.argmin(np.abs(self.pres_uref - ds_plev_hPa))
        i_lat = np.argmin(np.abs(self.lat_uref - ds['lat'][:]))
        return i_lev,i_lat
    def hours_since_nov1(self,ds):
        # Given the time from a dataset, convert the number to time in days since the most recent November 1
        dstime = ds['time']
        Nt = dstime.size
        date = nc.num2date(dstime[:],dstime.units,dstime.calendar)
        year = np.array([date[i].year for i in range(Nt)])
        month = np.array([date[i].month for i in range(Nt)])
        nov1_year = year*(month >= 11) + (year-1)*(month < 11)
        nov1_date = np.array([datetime.datetime(nov1_year[i], 11, 1) for i in range(Nt)])
        nov1_time = np.array([nc.date2num(nov1_date[i],dstime.units,dstime.calendar) for i in range(Nt)])
        #ensemble_size = ds['number'].size
        #dstime_adj = np.outer(np.ones(ensemble_size), (dstime - nov1_time)/24.0)
        dstime_adj = dstime - nov1_time 
        return dstime_adj # This is just one-dimensional. 
    def get_ensemble_source_size(self,ds):
        vbls = list(ds.variables.keys())
        if 'var129' in vbls: # This means it's from era20c OR eraint
            dssource = 'era20c'
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
            elif dssource in ['era20c','eraint']:
                gh[i_mem] = ds['var129'][:]/grav_accel
                u[i_mem] = ds['var131'][:]
            else:
                raise Exception("The dssource you gave me, %s, is not recognized"%(dssource))
        time,fall_year = self.time_since_nov1(ds['time'])
        return gh,u,time,ds['plev'][:],ds['lat'][:],ds['lon'][:],fall_year
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
        return T
    def get_meridional_heat_flux(self,gh,temperature,plev,lat,lon): # Returns average between 45N and 75N 
        Nx,Nlev,Nlat,Nlon = gh.shape
        _,vmer = self.compute_geostrophic_wind(gh,lat,lon)
        vbar = np.outer(np.mean(vmer,axis=3).flatten(),np.ones(Nlon)).reshape((Nx,Nlev,Nlat,Nlon))
        Tbar = np.outer(np.mean(temperature,axis=3).flatten(),np.ones(Nlon)).reshape((Nx,Nlev,Nlat,Nlon))
        vT = np.mean((vmer - vbar)*(temperature - Tbar), axis=3)
        imin,imax = np.argmin(np.abs(lat-75)),np.argmin(np.abs(lat-45))
        vT = np.sum(vT[:,:,imin:imax]*np.cos(lat[imin:imax]*np.pi/180), axis=2)/(Nlon*np.sum(np.cos(lat[imin:imax]*np.pi/180)))
        return vT
    def compute_qgpv(self,gh,lat,lon):
        # gh shape should be (Nx, Nlev,Nlat,Nlon)
        # Quasigeostrophic potential vorticity: just do horizontal component for now
        # QGPV = (g/f)*(laplacian(gh) - d(gh)/dy * beta/f) + f
        #      = (g/f)*(laplacian(gh) - 1/(earth radius)**2 * cos(lat)/sin(lat) * d(gh)/dlat) + f
        Nx,Nlev,Nlat,Nlon = gh.shape
        Omega = 2*np.pi/(3600*24*365)
        fcor = np.outer(2*Omega*np.sin(lat*np.pi/180), np.ones(lon.size))
        earth_radius = 6371e3 
        grav_accel = 9.80665
        dlat = np.pi/180 * (lat[1] - lat[0])
        dlon = np.pi/180 * (lon[1] - lon[0])
        gh_lon2 = (np.roll(gh,-1,axis=3) - 2*gh + np.roll(gh,1,axis=3))/dlon**2
        gh_lat2 = (np.roll(gh,-1,axis=2) - 2*gh + np.roll(gh,1,axis=2))/dlat**2
        gh_lat2[:,:,0,:] = gh_lat2[:,:,1,:]
        gh_lat2[:,:,-1,:] = gh_lat2[:,:,-2,:]
        gh_lat = (np.roll(gh,-1,axis=2) - np.roll(gh,1,axis=2))/(2*dlat)
        gh_lat[:,:,0,:] = (-3*gh[:,:,0,:] + 4*gh[:,:,1,:] - gh[:,:,2,:])/(2*dlat)
        gh_lat[:,:,-1,:] = (3*gh[:,:,-1,:] - 4*gh[:,:,-2,:] + gh[:,:,-3,:])/(2*dlat)
        cos = np.outer(np.cos(lat*np.pi/180), np.ones(len(lon)))
        sin = np.outer(np.sin(lat*np.pi/180), np.ones(len(lon)))
        # Make poles nan
        cos[np.abs(cos)<1e-3] = np.nan
        sin[np.abs(sin)<1e-3] = np.nan
        qgpv = fcor + grav_accel/(fcor*earth_radius**2)*(
                1.0/cos**2*gh_lon2 - (sin/cos + cos/sin)*gh_lat + gh_lat2)
        #qgpv = 1.0/earth_radius**2 * (
        #        1.0/cos**2*gh_lon2 + gh_lat2 - (sin/cos)*gh_lat  # Laplacian
        #        )
        #qgpv = fcor + 1.0/earth_radius**2 * (grav_accel/fcor)*(
        #        1.0/cos**2*gh_lon2 + gh_lat2 - (sin/cos)*gh_lat  # Laplacian
        #        - (cos/sin)*gh_lat # beta effect 
        #        )
        # Put NaN at the poles and the equator.
        #qgpv[:,:,:2,:] = np.nan
        #qgpv[:,:,-2:,:] = np.nan
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
    def unseason(self,t_field,field,field_szn_mean,field_szn_std,normalize=True):
        wti = ((t_field - self.wtime[0])/self.dtwint).astype(int)
        wti = np.maximum(0, np.minimum(len(self.wtime)-1, wti))
        field_unseasoned = field - field_szn_mean[wti]
        if normalize:
            field_unseasoned *= 1.0/field_szn_std[wti]
        return field_unseasoned
    def reseason(self,t_field,field_unseasoned,t_szn,field_szn_mean,field_szn_std):
        #print("t_field.shape = {}, field_unseasoned.shape = {}, t_szn.shape = {}, field_szn_mean.shape = {}, field_szn_std.shape = {}".format(t_field.shape, field_unseasoned.shape, t_szn.shape, field_szn_mean.shape, field_szn_std.shape))
        wti = ((t_field - self.wtime[0])/self.dtwint).astype(int)
        wti = np.maximum(0, np.minimum(len(self.wtime)-1, wti))
        #print("field_szn_std[wti].shape = {}, field_unseasoned.shape = {}, field_szn_mean[wti].shape = {}".format(field_szn_std[wti].shape,field_unseasoned.shape,field_szn_mean[wti].shape))
        field = field_szn_std[wti] * field_unseasoned + field_szn_mean[wti]
        return field
    def time_since_nov1(self,dstime):
        Nt = dstime.size
        date = nc.num2date(dstime[0],dstime.units,dstime.calendar)
        year = date.year 
        month = date.month
        nov1_year = year*(month >= 11) + (year-1)*(month < 11)
        nov1_date = datetime.datetime(nov1_year, 11, 1)
        nov1_time = nc.date2num(nov1_date,dstime.units,dstime.calendar)
        dstime_adj = nc.date2num(date,dstime.units,dstime.calendar) - nov1_time + dstime
        return dstime_adj,nov1_year # This is just one-dimensional. 
    def create_features(self,data_file_list,multiprocessing_flag=False):
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
        if multiprocessing_flag:
            num_workers = min(10,MP.cpu_count())
            gen = data_file_list #(nc.Dataset(ds_filename,"r") for ds_filename in data_file_list)
            pool = MP.Pool(num_workers)
            result = pool.map(self.get_u_gh_from_filename,gen)
            reading_mid = timelib.time() - reading_start
            print(f"Reading mid = {reading_mid}")
            gh = np.zeros((0,Nlev,Nlat,Nlon))
            u = np.zeros((0,Nlev,Nlat,Nlon))
            t_szn = np.zeros(0)
            for res in result:
                gh = np.concatenate((gh,res[0].reshape((res[0].shape[0]*res[0].shape[1],Nlev,Nlat,Nlon))),axis=0)
                u = np.concatenate((u,res[1].reshape((res[1].shape[0]*res[1].shape[1],Nlev,Nlat,Nlon))),axis=0)
                t_szn = np.concatenate((t_szn,res[2]))
            print(f"gh.shape = {gh.shape}")
            print(f"u.shape = {u.shape}")
            print(f"t_szn.shape = {t_szn.shape}")
        else:
            gh = np.zeros((0,Nlev,Nlat,Nlon)) # First dimension will have both time and ensemble members
            u = np.zeros((0,Nlev,Nlat,Nlon)) 
            t_szn = np.zeros(0)
            grid_shp = np.array([Nlev,Nlat,Nlon])
            for i_file in range(len(data_file_list)):
                print("Creating features: file {} out of {}".format(i_file,len(data_file_list)))
                ds = nc.Dataset(data_file_list[i_file],"r")
                gh_new,u_new,time,_,_,_,_ = self.get_u_gh(ds)
                Nmem,Nt = gh_new.shape[:2]
                shp_new = np.array(gh_new.shape)
                if np.any(shp_new[2:5] != grid_shp):
                    raise Exception("The file {} has a geopotential height field of shape {}, whereas it was supposed to have a shape {}".format(data_file_list[i_file],shp_new[2:5],grid_shp))
                gh = np.concatenate((gh,gh_new.reshape((Nmem*Nt,Nlev,Nlat,Nlon))),axis=0)
                u = np.concatenate((u,u_new.reshape((Nmem*Nt,Nlev,Nlat,Nlon))),axis=0)
                t_szn = np.concatenate((t_szn,self.hours_since_nov1(ds)))
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
        gh_szn_mean,gh_szn_std = self.get_seasonal_mean(t_szn,gh)
        gh_unseasoned = self.unseason(t_szn,gh,gh_szn_mean,gh_szn_std,normalize=False)
        cosine = np.cos(np.pi/180 * lat)
        weight = 1/np.sqrt(len(gh_unseasoned))*np.outer(np.sqrt(cosine),np.ones(Nlon)).flatten()
        eofs = np.zeros((Nlev,Nlat*Nlon,self.Npc_per_level_max))
        singvals = np.zeros((Nlev,self.Npc_per_level_max))
        tot_var = np.zeros(Nlev)
        svd_start = timelib.time()
        if multiprocessing_flag:
            with MP.Pool(num_workers) as pool:
                svd_results = pool.map(reduced_svd, ((gh_unseasoned[:,i_lev,:,:].reshape((len(gh),Nlat*Nlon))*weight).T for i_lev in range(Nlev)))
            for i_lev,svd_ilev in enumerate(svd_results):
                eofs[i_lev,:,:] = svd_ilev[0][:,:self.Npc_per_level_max]
                singvals[i_lev,:] = svd_ilev[1][:self.Npc_per_level_max]
                tot_var[i_lev] = np.sum(svd_ilev[1]**2)
        else:
            for i_lev in range(Nlev):
                print("svd'ing level %i out of %i"%(i_lev,Nlev))
                U,S,Vh = np.linalg.svd((gh_unseasoned[:,i_lev,:,:].reshape((len(gh),Nlat*Nlon))*weight).T, full_matrices=False)
                eofs[i_lev,:,:] = U[:,:self.Npc_per_level_max]
                singvals[i_lev,:] = S[:self.Npc_per_level_max]
                tot_var[i_lev] = np.sum(S**2)
        svd_duration = timelib.time() - svd_start
        print(f"with multiprocessing = {multiprocessing_flag}, svd_duration = {svd_duration}")
        # Temperature: first compute cap average, then deseasonalize
        temperature = self.get_temperature(gh,plev,lat,lon)
        vT = self.get_meridional_heat_flux(gh,temperature,plev,lat,lon) 
        i_lat_cap = np.argmin(np.abs(lat - 60))
        area_factor = np.outer(np.cos(lat*np.pi/180), np.ones(Nlon))
        temp_capavg = np.sum((temperature*area_factor)[:,:,:i_lat_cap,:], axis=(2,3))/np.sum(area_factor[:i_lat_cap,:])
        temp_capavg_szn_mean,temp_capavg_szn_std = self.get_seasonal_mean(t_szn,temp_capavg)
        vT_szn_mean,vT_szn_std = self.get_seasonal_mean(t_szn,vT)
        feat_def = {
                "t_szn": t_szn, "plev": plev, "lat": lat, "lon": lon,
                "i_lev_uref": i_lev_uref, "i_lat_uref": i_lat_uref,
                "uref_mean": uref_mean, "uref_std": uref_std,
                "uref_szn_mean": uref_szn_mean, "uref_szn_std": uref_szn_std,
                "waves_szn_mean": waves_szn_mean, "waves_szn_std": waves_szn_std,
                "wave_mag_szn_mean": wave_mag_szn_mean, "wave_mag_szn_std": wave_mag_szn_std,
                "gh_szn_mean": gh_szn_mean, "gh_szn_std": gh_szn_std,
                "eofs": eofs, "singvals": singvals, "tot_var": tot_var,
                "vtx_diags_szn_mean": vtx_diags_szn_mean, "vtx_diags_szn_std": vtx_diags_szn_std,
                "temp_capavg_szn_mean": temp_capavg_szn_mean, "temp_capavg_szn_std": temp_capavg_szn_std,
                "vT_szn_mean": vT_szn_mean, "vT_szn_std": vT_szn_std,
                }
        pickle.dump(feat_def,open(self.feature_file,"wb"))
        return
    def evaluate_features_database_parallel(self,file_list,feat_def,feat_filename,ens_start_filename,fall_year_filename,tmin,tmax):
        # Stack a bunch of forecasts together. They can start at different times, but must all have same length.
        #X_fallyear_list = pool.map(self.evaluate_features,
        ens_start_idx = np.zeros(len(file_list), dtype=int)
        fall_year_list = np.zeros(len(file_list), dtype=int)
        i_ens = 0
        # Now start up a pool of workers to read in all the files. 
        arg_gen = ((ds_filename, feat_def) for ds_filename in file_list)
        pool = MP.Pool(max(1, min(len(file_list)//20,MP.cpu_count())))
        result = pool.starmap(self.evaluate_features_from_filename,arg_gen)
        for i,res in enumerate(result):
            ens_start_idx[i] = i_ens
            Xnew,fall_year = res
            #print(f"Xnew.shape = {Xnew.shape}, fall_year = {fall_year}")
            dstime = Xnew[0,:,0] - Xnew[0,0,0]
            #print(f"dstime = {dstime}")
            fall_year_list[i] = fall_year
            ti_initial = np.where(dstime >= tmin)[0][0]
            ti_final = np.where(dstime <= tmax)[0][-1]
            #print(f"ti_initial = {ti_initial}, ti_final = {ti_final}")
            Xnew = Xnew[:,ti_initial:ti_final+1,:]
            if i == 0:
                X = Xnew.copy()
            else:
                X = np.concatenate((X,Xnew),axis=0)
            i_ens += Xnew.shape[0]
        # Save them in the directory
        np.save(feat_filename,X)
        np.save(ens_start_filename,ens_start_idx)
        np.save(fall_year_filename,fall_year_list)
        return X
    def evaluate_features_database(self,file_list,feat_def,feat_filename,ens_start_filename,fall_year_filename,tmin,tmax):
        # Stack a bunch of forecasts together. They can start at different times, but must all have same length.
        ens_start_idx = np.zeros(len(file_list), dtype=int)
        fall_year_list = np.zeros(len(file_list), dtype=int)
        i_ens = 0
        for i in range(len(file_list)):
            if i % 10 == 0:
                print("file %i out of %i: %s"%(i,len(file_list),file_list[i]))
            ens_start_idx[i] = i_ens
            ds = nc.Dataset(file_list[i],"r")
            Xnew,fall_year = self.evaluate_features(ds,feat_def)
            print(f"Xnew.shape = {Xnew.shape}, fall_year = {fall_year}")
            fall_year_list[i] = fall_year
            ti_initial = np.where(ds['time'][:] >= tmin)[0][0]
            ti_final = np.where(ds['time'][:] <= tmax)[0][-1]
            print(f"ds['time'][:] = {ds['time'][:]}")
            print(f"Xnew[0,:,0] = {Xnew[0,:,0]}")
            print(f"ti_initial = {ti_initial}, ti_final = {ti_final}")
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
    def evaluate_tpt_features(self,feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=False,seed=0):
        print(f" -------------- Inside evaluate_tpt_features: tpt_feat_filename = {tpt_feat_filename}, resample_flag = {resample_flag}, seed = {seed} --------------")
        # Evaluate a subset of the full features to use for clustering TPT.
        # A normalized version of these will be used for clustering.
        # The data set for clustering will have fewer time steps, due to time-delay embedding.
        X = np.load(feat_filename)
        print("Before resampling: X.shape = {}".format(X.shape))
        prng = np.random.RandomState(seed)
        if resample_flag:
            ens_start_idx = np.load(ens_start_filename)
            fall_year_list = np.load(fall_year_filename)
            fy_unique = np.unique(fall_year_list)
            #print("fy_unique = {}".format(fy_unique))
            fall_year_x = np.zeros(len(X), dtype=int)
            for i in range(len(ens_start_idx)):
                if i < len(ens_start_idx)-1:
                    ens_size = ens_start_idx[i+1] - ens_start_idx[i]
                else:
                    ens_size = len(X) - ens_start_idx[i]
                fall_year_x[ens_start_idx[i]:ens_start_idx[i]+ens_size] = fall_year_list[i]
                #print("ens_start_idx[i] = {}".format(ens_start_idx[i]))
            #print("fall_year_x: min={}, max={}, shape={}".format(fall_year_x.min(),fall_year_x.max(),fall_year_x.shape))
            fy_resamp = prng.choice(fy_unique,size=len(fy_unique),replace=True)
            #print("len(fy_resamp) = {}".format(len(fy_resamp)))
            idx_resamp = np.zeros(0, dtype=int)
            avg_match = 0
            for i in range(len(fy_resamp)):
                matches = np.where(fall_year_x == fy_resamp[i])[0]
                avg_match += len(matches)
                idx_resamp = np.concatenate((idx_resamp,matches))
            avg_match *= 1.0/len(fy_resamp)
            #print("avg_match = {}, len(fy_resamp) = {}, prod = {}".format(avg_match,len(fy_resamp), avg_match*len(fy_resamp)))
            X = X[idx_resamp]
        #print("len(idx_resamp) = {}".format(len(idx_resamp)))
        print("After resampling: X.shape = {}".format(X.shape))
        Nlev = len(feat_def['plev'])
        Nx,Ntx,xdim = X.shape
        print(f"Nx = {Nx}, Ntx = {Ntx}, xdim = {xdim}")
        # ------------- Define the cluster features Y ------------------
        # Y will have time-delay features built in. 
        # Y dimension: (time, uref, real + imag for each wave, pc for each level, vortex area, centroid latitude) all times the number of time lags
        Nty = Ntx - self.ndelay + 1
        print(f"self.ndelay = {self.ndelay}")
        ydim = 1 # Time
        ydim += self.ndelay # Zonal wind for some history
        ydim += 2*algo_params["Nwaves"] # real and imaginary parts of each 
        ydim += np.sum(algo_params["Npc_per_level"]) 
        ydim += algo_params["num_vortex_moments"] # Vortex moments
        ydim += np.sum(algo_params["captemp_flag"]) # Polar cap averaged temperature
        ydim += np.sum(algo_params["heatflux_flag"]) #*self.ndelay # Heat flux at each level and lag time
        print(f"ydim = {ydim}")
        Y = np.zeros((Nx,Nty,ydim))
        # Store information to unseason Y, simply as a set of seasonal means, one per column.
        szn_mean_Y = np.zeros((self.Ntwint,ydim-1))
        szn_std_Y = np.zeros((self.Ntwint,ydim-1))
        # ------------- Time ---------------
        Y[:,:,self.fidx_Y["time_h"]] = X[:,self.ndelay-1:,self.fidx_X["time_h"]]
        print(f"Y[0,0,0] = {Y[0,0,0]}")
        # ------------ Uref ------------------
        # Build time delays of u into Y
        i_feat_x = self.fidx_X["uref"]
        for i_dl in range(self.ndelay):
            i_feat_y = self.fidx_Y["uref_dl%i"%(i_dl)]
            Y[:,:,i_feat_y] = X[:,i_dl:i_dl+Nty,i_feat_x]
            print(f"t_szn.shape = {feat_def['t_szn'].shape}, uref_szn_mean.shape = {feat_def['uref_szn_mean'].shape}")
            szn_mean_Y[:,i_feat_y-1] = feat_def["uref_szn_mean"] 
            szn_std_Y[:,i_feat_y-1] = feat_def["uref_szn_std"]
            #offset_Y[i_feat_y-1] = feat_def["uref_mean"]
            #scale_Y[i_feat_y-1] = feat_def["uref_std"]
        # ----------- Waves -------------------
        for i_wave in np.arange(algo_params["Nwaves"]):
            i_feat_y = self.fidx_Y["real%i"%(i_wave)]
            i_feat_x = self.fidx_X["real%i"%(i_wave)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["waves_szn_mean"][:,2*i_wave]
            szn_std_Y[:,i_feat_y-1] = feat_def["waves_szn_std"][:,2*i_wave]
            i_feat_y = self.fidx_Y["imag%i"%(i_wave)]
            i_feat_x = self.fidx_X["imag%i"%(i_wave)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["waves_szn_mean"][:,2*i_wave+1]
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
            szn_mean_Y[:,i_feat_y-1] = feat_def["vtx_diags_szn_mean"][i_mom]
            szn_std_Y[:,i_feat_y-1] = feat_def["vtx_diags_szn_std"][i_mom]
        # ------- Polar cap temperature ------------
        for i_lev in range(Nlev):
            i_feat_y = self.fidx_Y["captemp_lev%i"%(i_lev)]
            i_feat_x = self.fidx_X["captemp_lev%i"%(i_lev)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["temp_capavg_szn_mean"][i_lev]
            szn_std_Y[:,i_feat_y-1] = feat_def["temp_capavg_szn_std"][i_lev]
        # ------- Heat flux ------------
        # TODO: make this a time integral
        for i_lev in range(Nlev):
            i_feat_y = self.fidx_Y["heatflux_lev%i"%(i_lev)]
            i_feat_x = self.fidx_X["heatflux_lev%i"%(i_lev)]
            Y[:,:,i_feat_y] = X[:,self.ndelay-1:,i_feat_x]
            szn_mean_Y[:,i_feat_y-1] = feat_def["vT_szn_mean"][i_lev]
            szn_std_Y[:,i_feat_y-1] = feat_def["vT_szn_std"][i_lev]
        tpt_feat = {"Y": Y, "szn_mean_Y": szn_mean_Y, "szn_std_Y": szn_std_Y}
        pickle.dump(tpt_feat, open(tpt_feat_filename,"wb"))
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
        flab[key] = r"Hours since Nov. 1"
        i_feat += 1
        # ---------- Reference zonal wind --------
        key = "uref"
        fidx[key] = i_feat
        flab[key] = r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]"
        i_feat += 1
        # -------------- Waves -------------------
        for i_wave in range(self.num_wavenumbers):
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
            flab[key] = r"Polar cap temp. at %i hPa [K]"%(feat_def["plev"]/100.0) 
            i_feat += 1
        # -------- Heat flux at 10 hPa -----------
        for i_lev in range(Nlev):
            key = "heatflux_lev%i"%(i_lev)
            fidx[key] = i_feat
            flab[key] = r"$\overline{v'T'}$ at 45-75$^\circ$N, %i hPa [K$\cdot$m/s]"%(feat_def["plev"][i_lev]/100.0)
            i_feat += 1
        # Save the file
        pickle.dump(fidx,open(fidx_X_filename,"wb"))
        self.fidx_X = fidx
        self.flab_X = flab
        return fidx
    def set_feature_indices_Y(self,feat_def,algo_params,fidx_Y_filename):
        # Build a mapping from feature names to indices in Y. (These will approximately match those in X, but will also include time-delay embeddings and perhaps more complicated combinations.) 
        fidx = dict()
        Nlev = len(feat_def["plev"])
        fidx["time_h"] = 0
        i_feat = 1
        # -------- Time-delayed uref ---------------
        for i_dl in range(self.ndelay):
            fidx["uref_dl%i"%(i_dl)] = i_feat
            i_feat += 1
        # --------- Waves (magnitude and phase) ----
        for i_wave in range(algo_params["Nwaves"]):
            fidx["real%i"%(i_wave)] = i_feat
            i_feat += 1
            fidx["imag%i"%(i_wave)] = i_feat
            i_feat += 1
        # ----- Principal components ---------------
        for i_lev in range(Nlev):
            for i_pc in range(algo_params["Npc_per_level"][i_lev]):
                fidx["pc%i_lev%i"%(i_pc,i_lev)] = i_feat
                i_feat += 1
        # -------- Vortex moments ----------------
        for i_mom in range(algo_params["num_vortex_moments"]):
            fidx["vxmom%i"%(i_mom)] = i_feat
            i_feat += 1
        # -------- Polar cap temperature ---------
        for i_lev in range(Nlev):
            if algo_params["captemp_flag"][i_lev]:
                fidx["captemp_lev%i"%(i_lev)] = i_feat
                i_feat += 1
        # -------- Heat flux at 10 hPa -----------
        for i_lev in range(Nlev):
            if algo_params["heatflux_flag"][i_lev]:
                fidx["heatflux_lev%i"%(i_lev)] = i_feat
                i_feat += 1
        # Save the file
        pickle.dump(fidx,open(fidx_Y_filename,"wb"))
        self.fidx_Y = fidx
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
        gh,u,time,plev,lat,lon,fall_year = self.get_u_gh(ds)
        time_ugh = timelib.time() - trun
        Nmem,Nt,Nlev,Nlat,Nlon = gh.shape
        area_factor = np.outer(np.cos(lat*np.pi/180), np.ones(Nlon))
        gh = gh.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        u = u.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        trun = timelib.time()
        _,vmer = self.compute_geostrophic_wind(gh,lat,lon) # for meridional wind
        time_vmer = timelib.time() - trun
        Nfeat = len(np.unique(np.array(self.fidx_X.values())))
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
        # ---------- Waves ---------------------
        trun = timelib.time()
        waves = self.get_wavenumbers(gh,i_lev_uref,self.lat_range_uref,lat,lon)
        time_waves = timelib.time() - trun
        for i_wave in range(self.num_wavenumbers):
            i_feat = self.fidx_X['real%i'%(i_wave)]
            X[:,i_feat] = waves[:,2*i_wave]
            i_feat = self.fidx_X['imag%i'%(i_wave)]
            X[:,i_feat] = waves[:,2*i_wave+1]
        # -------- EOFs ----------------------
        gh = self.unseason(X[:,0],gh,feat_def["gh_szn_mean"],feat_def["gh_szn_std"],normalize=False)
        for i_lev in range(Nlev):
            for i_pc in range(self.Npc_per_level_max):
                i_feat = self.fidx_X["lev%i_pc%i"%(i_lev,i_eof)]
                X[:,i_feat] = (gh[:,i_lev,:,:].reshape((Nmem*Nt,Nlat*Nlon)) @ (feat_def["eofs"][i_lev,:,i_pc])) / feat_def["singvals"][i_lev] 
        # --------- Temperature ---------------
        trun = timelib.time()
        temperature = self.get_temperature(gh,plev,lat,lon)
        i_lat_cap = np.argmin(np.abs(lat - 60))
        temp_capavg = np.sum((temperature*area_factor)[:,:,:i_lat_cap,:], axis=(2,3))/np.sum(area_factor[:i_lat_cap,:])
        print(f"temp_capavg.shape = {temp_capavg.shape}")
        time_temperature = timelib.time() - trun
        for i_lev in range(Nlev):
            i_feat = self.fidx_X["captemp_lev%i"%(i_lev)]
            X[:,i_feat] = temp_capavg[:,i_lev]
        # ---------- Heat flux ----------------
        trun = timelib.time()
        vT = self.get_meridional_heat_flux(gh,temperature,plev,lat,lon)
        time_vT = timelib.time() - trun
        print(f"Nlev = {Nlev}, vT.shape = {vT.shape}, i_feat = {i_feat}, X.shape = {X.shape}")
        for i_lev in range(Nlev):
            i_feat = self.fidx_X["heatflux_lev%i"%(i_lev)]
            X[:,i_feat] = vT[:,i_lev]
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
        obs_key_list = ["captemp_lev0","heatflux_lev0","heatflux_lev1","heatflux_lev2","uref","mag1","mag2","pc0_lev0","pc1_lev0","pc2_lev0","pc3_lev0","pc4_lev0","pc5_lev0"]
        for oki in range(len(obs_key_list)):
            obs_key = obs_key_list[oki]
            fig,ax = plt.subplots()
            ydata = funlib[obs_key]["fun"](X)
            if obs_key_list[oki].startswith("pc0"):
                ydata *= -1
            ax.plot(funlib["time_h"]["fun"](X),ydata,color='black')
            ax.set_xlabel("%s %i"%(funlib["time_h"]["label"],fall_year))
            ax.set_ylabel(funlib[obs_key]["label"])
            ax.axvspan(time[decel_time_range[0]],time[decel_time_range[1]],color='orange',zorder=-1)
            fig.savefig(join(savedir,"timeseries_%s_%s"%(save_suffix,obs_key)))
            plt.close(fig)
        # Plot polar cap evolution
        num_snapshots = 30
        i_lev_ref,i_lat_ref = self.get_ilev_ilat(ds)
        tidx = np.round(np.linspace(decel_time_range[0],decel_time_range[1],min(num_snapshots,decel_time_range[1]-decel_time_range[0]+2))).astype(int)
        gh,u,_,plev,lat,lon,fall_year = self.get_u_gh(ds)
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
                    "fun": lambda X: X[:,self.fidx_X[key]],
                    "label": self.flab_X[key],
                    }
        # Nonlinear functions
        for i_wave in range(self.num_wavenumbers):
            funlib["mag%i"%(i_wave)] = {
                    "fun": lambda X: np.sqrt(X[:,self.fidx_X["real%i"%(i_wave)]]**2 + X[:,self.fidx_X["imag%i"%(i_wave)]]**2),
                    "label": "Wave %i magnitude"%(i_wave),
                    }
            funlib["ph%i"%(i_wave)] = {
                    "fun": lambda X: np.arctan2(X[:,self.fidx_X["imag%i"%(i_wave)]]**2, X[:,self.fidx_X["real%i"%(i_wave)]]),
                    "label": "Wave %i phase"%(i_wave),
                    }
        return funlib
    def old_observable_function_library_X(self):
        # Build the database of observable functions
        feat_def = pickle.load(open(self.feature_file,"rb"))
        # Basic-state observables:
        funlib = dict()
        funlib = {
                "time_h": {"fun": lambda X: X[:,0],
                    "label": r"Hours since Nov. 1",},
                "time_d": {"fun": lambda X: X[:,0]/24.0,
                    "label": r"Days since Nov. 1",},
                "uref": {"fun": lambda X: X[:,1],
                    "label": r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]",},
                "mag1": {"fun": lambda X: np.sqrt(np.sum(X[:,2:4]**2, axis=1)),
                    "label": "Wave 1 magnitude",},
                "mag2": {"fun": lambda X: np.sqrt(np.sum(X[:,4:6]**2, axis=1)),
                    "label": "Wave 2 magnitude",},
                }
        if self.num_vortex_moments_max >= 1:
            funlib["area"] = {"fun": lambda X: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])],
                    "label": "Vortex area",}
        if self.num_vortex_moments_max >= 2:
            funlib["centerlat"] = {"fun": lambda X: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])+1],
                    "label": "Vortex center latitude",}
        if self.num_vortex_moments_max >= 3:
            funlib["asprat"] = {"fun": lambda X: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])+2],
                    "label": "Vortex aspect ratio",}
        if self.num_vortex_moments_max >= 4:
            funlib["kurt"] = {"fun": lambda X: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])+3],
                    "label": "Vortex excess kurtosis",}
        for i_lev in range(len(feat_def["plev"])):
            key = "lev%i_temp"%(i_lev)
            funlib[key] = {
                    "fun": lambda X,i_lev=i_lev: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])+self.num_vortex_moments_max+i_lev],
                    "label": "Cap temp. at %i hPa [K]"%(feat_def["plev"][i_lev]/100.0),
                    }
            key = "lev%i_vT"%(i_lev)
            funlib[key] = {
                    "fun": lambda X,i_lev=i_lev: X[:,2+2*self.num_wavenumbers+self.Npc_per_level_max*len(feat_def['plev'])+self.num_vortex_moments_max+len(feat_def['plev'])+i_lev],
                    "label": "$\overline{v'T'}$ at 45-75$^\circ$N, %i hPa [Km/s]"%(feat_def["plev"][i_lev]/100.0),
                    }
            for i_eof in range(self.Npc_per_level_max):
                key = "lev%i_pc%i"%(i_lev,i_eof)
                #print("key = {}".format(key))
                funlib[key] = {
                        "fun": lambda X,i_lev=i_lev,i_eof=i_eof: X[:,2+2*self.num_wavenumbers+i_lev*self.Npc_per_level_max+i_eof],
                        "label": "PC %i at %i hPa"%(i_eof+1, feat_def["plev"][i_lev]/100.0),
                        }
        return funlib
    def observable_function_library_Y(self,Nwaves=None,Npc_per_level=None,num_vortex_moments=None,heatflux_flag=None,captemp_flag=None):
        # Build the database of observable functions
        Nlev = len(feat_def['plev'])
        feat_def = pickle.load(open(self.feature_file,"rb"))
        if Nwaves is None:
            Nwaves = self.num_wavenumbers
        if Npc_per_level is None:
            Npc_per_level = self.Npc_per_level_max * np.ones(len(feat_def["plev"]), dtype=int)
        if num_vortex_moments is None:
            num_vortex_moments = self.num_vortex_moments_max
        if captemp_flag is None:
            captemp_flag = np.zeros(Nlev, dtype=bool)
        if heatflux_flag is None:
            heatflux_flag = np.zeros(Nlev, dtype=bool)
        print(f"Nwaves = {Nwaves}, Npc_per_level = {Npc_per_level}")
        print(f"Npc_per_level = {Npc_per_level}")
        print(f"Index for area = {1+self.ndelay+2*Nwaves+np.sum(Npc_per_level)}")
        funlib = {
                "time_h": {"fun": lambda Y: Y[:,0],
                    "label": r"Hours since Nov. 1",},
                "time_d": {"fun": lambda Y: Y[:,0]/24.0,
                    "label": r"Days since Nov. 1",},
                "uref": {"fun": lambda Y: self.uref_history(Y,feat_def)[:,-1],
                    "label": r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]",},
                "mag1": {"fun": lambda Y: self.wave_mph(Y,feat_def,1)[:,0],
                    "label": "Wave 1 magnitude",},
                "mag2": {"fun": lambda Y: self.wave_mph(Y,feat_def,2)[:,0],
                    "label": "Wave 2 magnitude",},
                "ph1": {"fun": lambda x: self.wave_mph(x,feat_def,1)[:,1],
                    "label": "Wave 1 phase",},
                "ph2": {"fun": lambda x: self.wave_mph(x,feat_def,2)[:,1],
                    "label": "Wave 2 phase",},
                }
        funlib = dict()
        i_feat_y = 0
        # ------ Time ---------
        funlib["time_h"] = {
                "fun": lambda Y: Y[:,i_feat_y],
                "label": r"Hours since Nov. 1"},
        funlib["time_d"] = {
                "fun": lambda Y: Y[:,i_feat_y]/24.0,
                "label": r"Days since Nov. 1"},
        i_feat_y += 1
        # ------ Zonal wind at 10 hPa and 60N -----------
        funlib["uref"] = {
                "fun": lambda Y: self.uref_history(Y,feat_def)[:,-1],
                "label": r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]",}
        # Time-delays for zonal wind
        for i_dl in range(self.ndelay):
            key = "dl%i_ubar"%(i_dl)
            funlib[key] = {
                    "fun": lambda Y: self.uref_history(Y,feat_def)[:,self.ndelay-1-i_dl],
                    "label": "$\overline{u}(60^\circ$N, 10 hPa), $t-$ %i days"%(i_dl),
                    }
            i_feat_y += 1
        # ------------ Wave magnitudes and phases -------------
        for i_wave in range(Nwaves):
            funlib["mag%i"%(i_wave)] = np.sqrt(Y[:,i_feat_y:i_feat_y+2]**2, axis=1)
            funlib["ph%i"%(i_wave)] = np.arctan2(Y[:,i_feat_y+1],Y[:,i_feat_y])
            i_feat_y += 2
        # ------------- PCs corresponding to EOFs --------------------
        for i_lev in range(len(feat_def["plev"])):
            for i_eof in range(Npc_per_level[i_lev]):
                funlib["lev%i_pc%i"%(i_lev,i_eof)] = {
                        "fun": lambda Y,i_lev=i_lev,i_eof=i_eof: Y[:,i_feat_y], #self.get_pc(Y,i_lev,i_eof,Nwaves,Npc_per_level),
                        "label": "PC %i at %i hPa"%(i_eof+1, feat_def["plev"][i_lev]/100.0),
                        }
                i_feat_y += 1
        # --------------- Vortex moments ---------------------------
        if num_vortex_moments >= 1:
            funlib["area"] = {
                    "fun": lambda Y: Y[:,i_feat_y],
                    "label": "Vortex area",}
            i_feat_y += 1
        if num_vortex_moments >= 2:
            funlib["centerlat"] = {
                    "fun": lambda Y: Y[:,i_feat_y],
                    "label": "Vortex center latitude",}
            i_feat_y += 1
        if num_vortex_moments >= 3:
            funlib["asprat"] = {"fun": lambda Y: Y[:,1+self.ndelay+2*Nwaves+np.sum(Npc_per_level)+2],
                "label": "Vortex aspect ratio",}
            i_feat_y += 1
        if num_vortex_moments >= 4:
            funlib["kurt"] = {"fun": lambda Y: Y[:,1+self.ndelay+2*Nwaves+np.sum(Npc_per_level)+3],
                "label": "Vortex excess kurtosis",}
            i_feat_y += 1
        for i_lev in range(len(feat_def["plev"])):
            key = "lev%i_temp"%(i_lev)
            funlib[key] = {
                    "fun": lambda Y,i_lev=i_lev: Y[:,1+self.ndelay+2*Nwaves+np.sum(Npc_per_level)+num_vortex_moments+i_lev],
                    "label": "Cap temp. at %i hPa [K]"%(feat_def["plev"][i_lev]/100.0),
                    }
            key = "lev%i_vT"%(i_lev)
            funlib[key] = {
                    "fun": lambda Y,i_lev=i_lev: Y[:,1+self.ndelay+2*Nwaves+np.sum(Npc_per_level)+num_vortex_moments+len(feat_def['plev'])+i_lev],
                    "label": "$\overline{v'T'}$ at 45-75$^\circ$N, %i hPa [Km/s]"%(feat_def["plev"][i_lev]/100.0),
                    }
        return funlib
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
        eof = feat_def["eofs"][i_lev,:,i_eof].reshape((Nlat,Nlon))
        fig,ax,_ = self.display_pole_field(eof,lat,lon)
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
