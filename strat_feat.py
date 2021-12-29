# Make a class to handle different, and heterogeneous, types of SSW data. Retain knowledge of multiple file sources at once, and mix and match them (with different durations and everything). Be able to work with any subset of the data for feature creation, training, and testing. Note that the minimal unit for training and testing is a file, i.e., an ensemble. The subset indices could be generated randomly externally. 
# Ultimately, create the information to compute any quantity of interest on new, unseen data: committors and lead times crucially. Also backward quantities and rates, given an initial distribution.
# the DGA_SSW object will have no knowledge of what year anything is; that will be implmented at a higher level. 
# Maybe even have variable lag time?
import numpy as np
import netCDF4 as nc
import datetime
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

class WinterStratosphereFeatures:
    # Create a set of features, including out-of-sample extension. 
    def __init__(self,feature_file,winter_day0,spring_day0,Npc_per_level_max=10):
        self.feature_file = feature_file
        self.winter_day0 = winter_day0
        self.spring_day0 = spring_day0
        self.wtime = 24.0 * np.arange(self.winter_day0,self.spring_day0) # All times are in hours
        self.Ntwint = len(self.wtime)
        self.szn_hour_window = 5.0*24 # Number of days around which to average when unseasoning
        self.dtwint = self.wtime[1] - self.wtime[0]
        self.Npc_per_level_max = Npc_per_level_max # Determine from SVD if not specified
        self.num_wavenumbers = 2 # How many wavenumbers to look at 
        self.lat_uref = 60 # Degrees North for CP07 definition of SSW
        self.lat_range_uref = self.lat_uref + 5.0*np.array([-1,1])
        self.pres_uref = 10 # hPa for CP07 definition of SSW
        return
    def compute_src_dest_tags(self,x,feat_def,tpt_bndy,save_filename):
        # Compute where each trajectory started (A or B) and where it's going (A or B). Also maybe compute the first-passage times, forward and backward.
        Nmem,Nt,Nfeat = x.shape
        ina = self.ina_test(x.reshape((Nmem*Nt,Nfeat)),feat_def,tpt_bndy).reshape((Nmem,Nt))
        inb = self.inb_test(x.reshape((Nmem*Nt,Nfeat)),feat_def,tpt_bndy).reshape((Nmem,Nt))
        src_tag = 0.5*np.ones((Nmem,Nt))
        dest_tag = 0.5*np.ones((Nmem,Nt))
        # Source: move forward in time
        # Time zero, A is the default src
        src_tag[:,0] = 0*ina[:,0] + 1*inb[:,0] + 0.5*(ina[:,0]==0)*(inb[:,0]==0)*(x[:,0,0] > tpt_bndy['tthresh'][0]) 
        for k in range(1,Nt):
            src_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + src_tag[:,k-1]*(ina[:,k]==0)*(inb[:,k]==0)
        # Dest: move backward in time
        # Time end, A is the default dest
        dest_tag[:,Nt-1] = 0*ina[:,Nt-1] + 1*inb[:,Nt-1] + 0.5*(ina[:,Nt-1]==0)*(inb[:,Nt-1]==0)*(x[:,-1,0] < tpt_bndy['tthresh'][1])
        for k in np.arange(Nt-2,-1,-1):
            dest_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + dest_tag[:,k+1]*(ina[:,k]==0)*(inb[:,k]==0)
        #print("Overall fraction in B = {}".format(np.mean(inb)))
        #print("At time zero: fraction of traj in B = {}, fraction of traj headed to B = {}".format(np.mean(dest_tag[:,0]==1),np.mean((dest_tag[:,0]==1)*(inb[:,0]==0))))
        result = {'src_tag': src_tag, 'dest_tag': dest_tag}
        pickle.dump(result,open(save_filename,'wb'))
        return src_tag,dest_tag
    def ina_test(self,x,feat_def,tpt_bndy):
        Nx,xdim = x.shape
        ina = np.zeros(Nx,dtype=bool)
        nonwinter_idx = np.where((x[:,0] >= tpt_bndy['tthresh'][0]) * (x[:,0] < tpt_bndy['tthresh'][1]) == 0)[0]
        ina[nonwinter_idx] = True
        return ina
    def inb_test(self,x,feat_def,tpt_bndy):
        # Test whether a reanalysis dataset's components are in B
        Nx,xdim = x.shape
        inb = np.zeros(Nx, dtype=bool)
        winter_idx = np.where((x[:,0] >= tpt_bndy['tthresh'][0])*(x[:,0] < tpt_bndy['tthresh'][1]))[0]
        uref = self.uref_obs(x[winter_idx],feat_def)
        weak_wind_idx = np.where(uref < tpt_bndy['uthresh'])
        inb[winter_idx[weak_wind_idx]] = True
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
        earth_radius = 6371e3 
        grav_accel = 9.80665
        dx = np.outer(earth_radius*np.cos(lat*np.pi/180), np.roll(lon,-1) - np.roll(lon,1)) # this counts both sides
        dy = np.outer((np.roll(lat,-1) - np.roll(lat,1))*earth_radius, np.ones(lon.size))
        gh_x = (np.roll(gh,-1,axis=3) - np.roll(gh,1,axis=3))/dx 
        gh_y = (np.roll(gh,-1,axis=2) - np.roll(gh,1,axis=2))/dy
        u = -gh_y/fcor * grav_accel
        v = gh_x/fcor * grav_accel
        return u,v
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
        cos[cos==0] = np.nan
        qgpv = 1.0/earth_radius**2 * (
                1.0/cos**2*gh_lon2 + gh_lat2 - (sin/cos)*gh_lat  # Laplacian
                )
        #qgpv = fcor + 1.0/earth_radius**2 * (grav_accel/fcor)*(
        #        1.0/cos**2*gh_lon2 + gh_lat2 - (sin/cos)*gh_lat  # Laplacian
        #        - (cos/sin)*gh_lat # beta effect 
        #        )
        # Put NaN at the poles.
        qgpv[:,:,:2,:] = np.nan
        qgpv[:,:,-2:,:] = np.nan
        return qgpv
    def compute_vortex_moments_sphere(self,gh,lat,lon,i_lev_subset=None):
        # Do the calculation in lat/lon coordinates. Regridding is too expensive
        Nsamp,Nlev_full,Nlat_full,Nlon = gh.shape
        if i_lev_subset is None:
            i_lev_subset = np.arange(Nlev_full)
        Nlev = len(i_lev_subset)
        i_lat_max = np.where(lat < 0.0)[0][0]  # All the way to the equator
        print(f"i_lat_max = {i_lat_max}")
        Nlat = i_lat_max # - 2
        stereo_factor = np.cos(lat[:i_lat_max]*np.pi/180)/(1 + np.sin(lat[:i_lat_max]*np.pi/180))
        X = np.outer(stereo_factor, np.cos(lon*np.pi/180)).flatten()
        Y = np.outer(stereo_factor, np.sin(lon*np.pi/180)).flatten()
        qgpv = self.compute_qgpv(gh,lat,lon)[:,i_lev_subset,:i_lat_max,:].reshape((Nsamp*Nlev,Nlat*Nlon))
        print(f"qgpv: nanfrac={np.mean(np.isnan(qgpv))}, min={np.nanmin(qgpv)}, max={np.nanmax(qgpv)}")
        # Assign an area to each grid cell. 
        dlat,dlon = np.pi/2*np.array([lat[1]-lat[0],lon[1]-lon[0]])
        area_factor = np.outer(np.cos(lat[:i_lat_max]*np.pi/180), np.ones(Nlon)).flatten()
        # Find vortex edge by ranking grid cells and finding the maximum slope of area fraction with respect to PV
        qgpv_order = np.argsort(qgpv,axis=1)
        area_fraction = np.cumsum(np.array([area_factor[qgpv_order[i]] for i in range(Nsamp*Nlev)]),axis=1)
        area_fraction = (area_fraction.T /area_fraction[:,-1]).T
        qgpv_sorted = np.array([qgpv[i,qgpv_order[i]] for i in np.arange(Nsamp*Nlev)])
        equiv_lat = np.arcsin(area_fraction)
        # Verify qgpv_sorted is monotonic with equiv_lat
        print(f"min diff = {np.nanmin(np.diff(qgpv_sorted, axis=1))}")
        if np.nanmin(np.diff(qgpv_sorted, axis=1)) < 0:
            raise Exception("qgpv_sorted must be monotonically increasing")
        window = 6
        dq_deqlat = (qgpv_sorted[:,window:] - qgpv_sorted[:,:-window])/(equiv_lat[:,window:] - equiv_lat[:,:-window])
        #idx_crit = np.nanargmax(dq_deqlat, axis=1) + window//2
        idx_crit = np.argmin(np.abs(area_fraction - 0.75), axis=1) #int(dA_dq.shape[1]/2) 
        qgpv_crit = qgpv_sorted[np.arange(Nsamp*Nlev),idx_crit]
        print(f"qgpv_crit: min={np.nanmin(qgpv_crit)}, max={np.nanmax(qgpv_crit)}")
        # Threshold and find moments
        q = (np.maximum(0, qgpv.T - qgpv_crit).T)
        print(f"q: frac>0 is {np.mean(q>0)},  min={np.nanmin(q)}, max={np.nanmax(q)}")
        # Zeroth moment: vortex area
        m00 = np.nansum(q*area_factor, axis=1)
        print(f"m00: min={np.nanmin(m00)}, max={np.nanmax(m00)}")
        area = m00 #/ qgpv_crit
        #area = np.sum((q>0)*area_factor, axis=1)
        # First moment: mean x and mean y
        m10 = np.nansum(area_factor*q*X, axis=1)/m00
        m01 = np.nansum(area_factor*q*Y, axis=1)/m00
        print(f"m10: min={np.nanmin(m10)}, max={np.nanmax(m10)}")
        print(f"m01: min={np.nanmin(m01)}, max={np.nanmax(m01)}")
        center = np.array([m10, m01]).T
        # Determine latitude and longitude of center
        center_lat = np.arcsin((1 - (center[:,0]**2 + center[:,1]**2))/(1 + (center[:,0]**2 + center[:,1]**2))) * 180/np.pi
        center_lon = np.arctan2(center[:,1],center[:,0]) * 180/np.pi
        # Reshape
        area = area.reshape((Nsamp,Nlev))
        center_latlon = np.array([center_lat,center_lon]).T
        #center = center.reshape((Nsamp,Nlev,2))
        print(f"area: min={np.nanmin(area)}, max={np.nanmax(area)}, mean={np.nanmean(area)}\ncenter(x): min={np.nanmin(center[:,0])}, max={np.nanmax(center[:,0])}, mean={np.nanmean(center[:,0])}")
        return area,center_latlon
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
    def create_features(self,data_file_list):
        # Use data in data_file_list as training, and dump the results into feature_file. Note this is NOT a DGA basis yet, just a set of features.
        # We will not YET use time-delay information, but might in the future. In which case we will just have somewhat fewer data per file. 
        # Mark the boundary of ensembles with a list of indices.
        ds0 = nc.Dataset(data_file_list[0],"r")
        Nlev,Nlat,Nlon = [ds0[v].size for v in ["plev","lat","lon"]]
        plev,lat,lon = [ds0[v][:] for v in ["plev","lat","lon"]]
        i_lev_uref,i_lat_uref = self.get_ilev_ilat(ds0)
        ds0.close()
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
        for i_lev in range(Nlev):
            print("svd'ing level %i out of %i"%(i_lev,Nlev))
            U,S,Vh = np.linalg.svd((gh_unseasoned[:,i_lev,:,:].reshape((len(gh),Nlat*Nlon))*weight).T, full_matrices=False)
            eofs[i_lev,:,:] = U[:,:self.Npc_per_level_max]
            singvals[i_lev,:] = S[:self.Npc_per_level_max]
            tot_var[i_lev] = np.sum(S**2)
        # TODO: remove EOFs that are highly correlated with uref or wavenumbers
        # Vortex moment diagnostics, only at reference level
        vtx_area,vtx_center_latlon = self.compute_vortex_moments_sphere(gh,lat,lon,i_lev_subset=[i_lev_uref])
        vtx_area_szn_mean,vtx_area_szn_std = self.get_seasonal_mean(t_szn,vtx_area[:,0])
        vtx_centerlat_szn_mean,vtx_centerlat_szn_std = self.get_seasonal_mean(t_szn,vtx_center_latlon[:,0])
        feat_def = {
                "t_szn": t_szn, "plev": plev, "lat": lat, "lon": lon,
                "i_lev_uref": i_lev_uref, "i_lat_uref": i_lat_uref,
                "uref_mean": uref_mean, "uref_std": uref_std,
                "uref_szn_mean": uref_szn_mean, "uref_szn_std": uref_szn_std,
                "waves_szn_mean": waves_szn_mean, "waves_szn_std": waves_szn_std,
                "wave_mag_szn_mean": wave_mag_szn_mean, "wave_mag_szn_std": wave_mag_szn_std,
                "gh_szn_mean": gh_szn_mean, "gh_szn_std": gh_szn_std,
                "eofs": eofs, "singvals": singvals, "tot_var": tot_var,
                "vtx_area_szn_mean": vtx_area_szn_mean, "vtx_area_szn_std": vtx_area_szn_std,
                "vtx_centerlat_szn_mean": vtx_centerlat_szn_mean, "vtx_centerlat_szn_std": vtx_centerlat_szn_std,
                }
        pickle.dump(feat_def,open(self.feature_file,"wb"))
        return
    def evaluate_features_database(self,file_list,feat_def,savedir,filename,tmin,tmax):
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
            fall_year_list[i] = fall_year
            ti_initial = np.where(ds['time'][:] >= tmin)[0][0]
            ti_final = np.where(ds['time'][:] <= tmax)[0][-1]
            Xnew = Xnew[:,ti_initial:ti_final+1,:]
            if i == 0:
                X = Xnew.copy()
            else:
                X = np.concatenate((X,Xnew),axis=0)
            i_ens += Xnew.shape[0]
        # Save them in the directory
        np.save(join(savedir,filename),X)
        np.save(join(savedir,"ens_start_idx"),ens_start_idx)
        np.save(join(savedir,"fall_year_list"),fall_year_list)
        return X
    def evaluate_cluster_features(self,feat_filename,ens_start_filename,fall_year_filename,feat_def,clust_feat_filename,Npc_per_level=None,Nwaves=None,resample_flag=False,seed=0):
        # Evaluate a subset of the full features to use for clustering.
        X = np.load(feat_filename)
        #print("Before resampling: X.shape = {}".format(X.shape))
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
            np.random.seed(seed)
            fy_resamp = np.random.choice(fy_unique,size=len(fy_unique),replace=True)
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
        #print("After resampling: X.shape = {}".format(X.shape))
        if Npc_per_level is None:
            Npc_per_level = self.Npc_per_level_max*np.ones(len(feat_def['plev']), dtype=int)
        if Nwaves is None:
            Nwaves = self.num_wavenumbers
        Nx,Nt,xdim = X.shape
        ydim = 2 + 2*Nwaves + np.sum(Npc_per_level) + 2
        Y = np.zeros((Nx,Nt,ydim))
        Y[:,:,:2] = X[:,:,:2]
        # TODO: build time-delay information into Y.
        i_feat_x = 2
        i_feat_y = 2
        for i_wave in np.arange(self.num_wavenumbers):
            if i_wave < Nwaves:
                Y[:,:,i_feat_y:i_feat_y+2] = X[:,:,i_feat_x:i_feat_x+2]
                i_feat_y += 2
            i_feat_x += 2
        for i_lev in range(len(feat_def['plev'])):
            for i_pc in range(self.Npc_per_level_max):
                if i_pc < Npc_per_level[i_lev]:
                    Y[:,:,i_feat_y] = X[:,:,i_feat_x]
                    i_feat_y += 1
                i_feat_x += 1
        # Read in vortex moment diagnostics
        window = 6
        for i in range(window):
            Y[:,window:,i_feat_y:i_feat_y+2] += X[:,window-i:Nt-i,i_feat_x:i_feat_x+2]/window
        Y[:,:window,i_feat_y:i_feat_y+2] = X[:,:window,i_feat_x:i_feat_x+2]
        #Y[:,:,i_feat_y:i_feat_y+2] = X[:,:,i_feat_x:i_feat_x+2]
        i_feat_y += 2
        i_feat_x += 2
        np.save(clust_feat_filename,Y)
        return 
    def evaluate_features(self,ds,feat_def):
        # Given a single ensemble in ds, evaluate the features and return a big matrix
        i_lev_uref,i_lat_uref = self.get_ilev_ilat(ds)
        gh,u,time,plev,lat,lon,fall_year = self.get_u_gh(ds)
        Nmem,Nt,Nlev,Nlat,Nlon = gh.shape
        gh = gh.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        u = u.reshape((Nmem*Nt,Nlev,Nlat,Nlon))
        Nfeat = 2 + 2*self.num_wavenumbers + Nlev*self.Npc_per_level_max + 2 # Last two for area and mean polar displacement of the vortex
        X = np.zeros((Nmem*Nt,Nfeat))
        # Time
        X[:,0] = np.outer(np.ones(Nmem),time).flatten()
        # Zonal-mean zonal wind
        #u,v = self.compute_geostrophic_wind(gh,lat,lon)
        uref = np.mean(u[:,i_lev_uref,i_lat_uref,:],axis=1)
        X[:,1] = self.unseason(X[:,0],uref,feat_def["uref_szn_mean"],feat_def["uref_szn_std"])
        # Wave amplitudes
        waves = self.get_wavenumbers(gh,i_lev_uref,self.lat_range_uref,lat,lon)
        X[:,2:2+2*self.num_wavenumbers] = self.unseason(X[:,0],waves,feat_def["waves_szn_mean"],feat_def["waves_szn_std"])
        # EOFs
        gh = self.unseason(X[:,0],gh,feat_def["gh_szn_mean"],feat_def["gh_szn_std"],normalize=False)
        i_feat = 2+2*self.num_wavenumbers
        for i_lev in range(Nlev):
            X[:,i_feat:i_feat+self.Npc_per_level_max] = (gh[:,i_lev,:,:].reshape((Nmem*Nt,Nlat*Nlon)) @ (feat_def["eofs"][i_lev,:,:self.Npc_per_level_max])) / feat_def["singvals"][i_lev]
            i_feat += self.Npc_per_level_max
        # Vortex moments
        vtx_area,vtx_center_latlon = self.compute_vortex_moments_sphere(gh,feat_def['lat'],feat_def['lon'],i_lev_subset=[i_lev_uref])
        #X[:,i_feat] = vtx_area[:,0] 
        #X[:,i_feat+1] = vtx_center_latlon[:,0] 
        X[:,i_feat] = self.unseason(X[:,0],vtx_area[:,0],feat_def["vtx_area_szn_mean"],feat_def["vtx_area_szn_std"])  # Vortex area
        X[:,i_feat+1] = self.unseason(X[:,0],vtx_center_latlon[:,0],feat_def["vtx_centerlat_szn_mean"],feat_def["vtx_centerlat_szn_std"])
        X = X.reshape((Nmem,Nt,Nfeat))
        return X,fall_year
    def plot_vortex_evolution(self,dsfile,savedir,save_suffix,i_mem=0):
        # Plot the holistic information about a single member of a single ensemble. Include some timeseries and some snapshots, perhaps along the region of maximum deceleration in zonal wind. 
        ds = nc.Dataset(dsfile,"r")
        print("self.num_wavenumbers = {}, self.Npc_per_level_max = {}".format(self.num_wavenumbers,self.Npc_per_level_max))
        funlib = self.observable_function_library()
        feat_def = pickle.load(open(self.feature_file,"rb"))
        X,fall_year = self.evaluate_features(ds,feat_def)
        X = X[i_mem]
        print("X.shape = {}".format(X.shape))
        # Determine the period of maximum deceleration
        time = X[:,0]
        decel_window = int(24*10.0/(time[1]-time[0]))
        uref = funlib["uref"]["fun"](X)
        decel10 = uref[decel_window:] - uref[:-decel_window]
        print("uref: min={}, max={}. decel10: min={}, max={}".format(uref.min(),uref.max(),decel10.min(),decel10.max()))
        start = np.argmin(decel10)
        print("start = {}".format(start))
        decel_time_range = [max(0,start-decel_window), min(len(time)-1, start+2*decel_window)]
        full_time_range = self.wtime[[0,-1]]
        # ----------- debugging -----------
        pc0 = funlib["lev0_pc0"]["fun"](X)
        pc1 = funlib["lev0_pc1"]["fun"](X)
        i_pc0 = 2 + 2*self.num_wavenumbers 
        i_pc1 = i_pc0 + 1
        print("max(abs(X[:,i_pc0] - X[:,i_pc1])) = {}".format(np.max(np.abs(X[:,i_pc0] - X[:,i_pc1]))))
        print("max(abs(pc0 - pc1)) = {}".format(np.max(np.abs(pc0-pc1))))
        print("direct vs fun: 0: {}, 1: {}".format(np.max(np.abs(X[:,i_pc0] - pc0)),np.max(np.abs(X[:,i_pc1] - pc1))))
        # ---------------------------------
        obs_key_list = ["area","displacement","uref","mag1","mag2","mag1_anomaly","mag2_anomaly","ph1","ph2","lev0_pc0","lev0_pc1","lev0_pc2","lev0_pc3","lev0_pc4","lev0_pc5"]
        for oki in range(len(obs_key_list)):
            obs_key = obs_key_list[oki]
            fig,ax = plt.subplots()
            ydata = funlib[obs_key]["fun"](X)
            if obs_key_list[oki].endswith("pc0"):
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
        i_lat_max = np.where(lat < 45)[0][0]
        gh[:,i_lat_max:,:] = np.nan
        qgpv[:,i_lat_max:,:] = np.nan
        for k in range(len(tidx)):
            i_time = tidx[k]
            fig,ax = self.show_ugh_onelevel_cartopy(gh[k],u[k],v[k],lat,lon)
            ax.set_title(r"$\Phi$, $u$ at day {}, {}".format(time[tidx[k]]/24.0,fall_year))
            fig.savefig(join(savedir,"vortex_gh_{}_day{}_yr{}".format(save_suffix,int(time[tidx[k]]/24.0),fall_year)))
            plt.close(fig)
            fig,ax = self.show_ugh_onelevel_cartopy(qgpv[k],u[k],v[k],lat,lon)
            ax.set_title(r"$\Phi$, $u$ at day {}, {}".format(time[tidx[k]]/24.0,fall_year))
            fig.savefig(join(savedir,"vortex_qgpv_{}_day{}_yr{}".format(save_suffix,int(time[tidx[k]]/24.0),fall_year)))
            plt.close(fig)
        return
    def wave_mph(self,x,feat_def,wn,widx=None,unseason_mag=False):
        # wn is wavenumber 
        # mph is magnitude and phase
        if wn <= 0:
            raise Exception("Need an integer wavenumber >= 1. You gave wn = {}".format(wn))
        if widx is None:
            widx = 2 + 2*wn*np.arange(2)
        wave = self.reseason(x[:,0],x[:,widx],feat_def['t_szn'],feat_def['waves_szn_mean'][:,2*(wn-1):2*(wn-1)+2],feat_def['waves_szn_std'][:,2*(wn-1):2*(wn-1)+2])
        phase = np.arctan(-wave[:,1]/(wn*wave[:,0]))
        mag = np.sqrt(np.sum(wave**2,axis=1))
        if unseason_mag:
            mag = self.unseason(x[:,0],mag,feat_def["wave_mag_szn_mean"][:,wn-1],feat_def["wave_mag_szn_std"][:,wn-1])
        return np.array([mag,phase]).T
    def uref_obs(self,x,feat_def):
        uref = self.reseason(x[:,0],x[:,1],feat_def["t_szn"],feat_def["uref_szn_mean"],feat_def["uref_szn_std"])
        #print("uref = {}".format(uref))
        #print("uref: min={}, max={}, mean={}".format(uref.min(),uref.max(),uref.mean()))
        return uref
    def observable_function_library(self,Nwaves=None,Npc_per_level=None):
        # Build the database of observable functions
        feat_def = pickle.load(open(self.feature_file,"rb"))
        if Nwaves is None:
            Nwaves = self.num_wavenumbers
        if Npc_per_level is None:
            Npc_per_level = self.Npc_per_level_max * np.ones(len(feat_def["plev"]), dtype=int)
        # TODO: build in PC projections and other stuff as observable functions
        funlib = {
                "time_h": {"fun": lambda x: x[:,0],
                    "label": r"Hours since Nov. 1",},
                "time_d": {"fun": lambda x: x[:,0]/24.0,
                    "label": r"Days since Nov. 1",},
                "uref": {"fun": lambda x: self.uref_obs(x,feat_def),
                    "label": r"$\overline{u}$ (10 hPa, 60$^\circ$N) [m/s]",},
                "mag1": {"fun": lambda x: self.wave_mph(x,feat_def,1)[:,0],
                    "label": "Wave 1 magnitude",},
                "mag2": {"fun": lambda x: self.wave_mph(x,feat_def,2)[:,0],
                    "label": "Wave 2 magnitude",},
                "mag1_anomaly": {"fun": lambda x: self.wave_mph(x,feat_def,1,unseason_mag=True)[:,0],
                    "label": "Wave 1 magnitude anomaly",},
                "mag2_anomaly": {"fun": lambda x: self.wave_mph(x,feat_def,2,unseason_mag=True)[:,0],
                    "label": "Wave 2 magnitude anomaly",},
                "ph1": {"fun": lambda x: self.wave_mph(x,feat_def,1)[:,1],
                    "label": "Wave 1 phase",},
                "ph2": {"fun": lambda x: self.wave_mph(x,feat_def,2)[:,1],
                    "label": "Wave 2 phase",},
                "area": {"fun": lambda x: self.get_vortex_area(x,0,Nwaves,Npc_per_level),
                    "label": "Vortex area",},
                "displacement": {"fun": lambda x: self.get_vortex_displacement(x,0,Nwaves,Npc_per_level),
                    "label": "Vortex displacement",},
                #"pc1": {"fun": lambda x: x[:,2+2*self.num_wavenumbers],
                #    "label": "PC 1"},
                #"pc2": {"fun": lambda x: x[:,2+2*self.num_wavenumbers+1],
                #    "label": "PC 2"},
                }
        for i_lev in range(len(feat_def["plev"])):
            for i_eof in range(self.Npc_per_level_max):
                key = "lev%i_pc%i"%(i_lev,i_eof)
                #print("key = {}".format(key))
                funlib[key] = {
                        "fun": lambda x,i_lev=i_lev,i_eof=i_eof: self.get_pc(x,i_lev,i_eof,Nwaves,Npc_per_level),
                        "label": "PC %i at $p=%i$ hPa"%(i_eof+1, feat_def["plev"][i_lev]/100.0),
                        }
        return funlib
    def get_pc(self,x,i_lev,i_eof,Nwaves=None,Npc_per_level=None):
        if Nwaves is None:
            Nwaves = self.num_wavenumbers
        if Npc_per_level is None:
            Npc_per_level = self.Npc_per_level_max * np.ones(len(feat_def["plev"]), dtype=int)
        idx = 2 + 2*Nwaves + np.sum(Npc_per_level[:i_lev]) + i_eof
        #eof = x[:,2 + 2*self.num_wavenumbers + i_lev*self.Npc_per_level_max + i_eof]
        eof = x[:,idx]
        return eof
    def get_vortex_area(self,x,i_lev,Nwaves,Npc_per_level):
        idx = 2 + 2*Nwaves + np.sum(Npc_per_level)
        area = self.reseason(x[:,0],x[:,idx],feat_def['t_szn'],feat_def['vtx_area_szn_mean'],feat_def['vtx_area_szn_std'])
        return area
    def get_vortex_displacement(self,x,i_lev,Nwaves,Npc_per_level):
        idx = 2 + 2*Nwaves + np.sum(Npc_per_level) + 1
        centerlat = self.reseason(x[:,0],x[:,idx],feat_def['t_szn'],feat_def['vtx_centerlat_szn_mean'],feat_def['vtx_centerlat_szn_std'])
        return centerlat 
    def show_ugh_onelevel_cartopy(self,gh,u,v,lat,lon): 
        # Display the geopotential height at a single pressure level
        fig,ax,data_crs = self.display_pole_field(gh,lat,lon)
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
    def display_pole_field(self,field,lat,lon):
        data_crs = ccrs.PlateCarree() 
        ax_crs = ccrs.Orthographic(-10,90)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection=ax_crs)
        im = ax.pcolormesh(lon,lat,field,shading='nearest',cmap='coolwarm',transform=data_crs)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=3, edgecolor='black')
        fig.colorbar(im,ax=ax)
        return fig,ax,data_crs
