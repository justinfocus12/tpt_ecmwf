#  This is a collection of functions to visualize aspects of the ensemble starting from one day. 
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
matplotlib.rc('font',**font)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os
from os import mkdir
from os.path import join,exists
from sklearn import linear_model
import cartopy
from cartopy import crs as ccrs
import pickle

codefolder = "/home/jf4241/ecmwf/s2s"
os.chdir(codefolder)

datadir_s2s = "/scratch/jf4241/ecmwf_data/s2s_data/2021-11-01"
datadir_era20c = "/scratch/jf4241/ecmwf_data/era20c_data/2021-11-03"
resdir = "/scratch/jf4241/ecmwf_data/s2s_results"
daydir = join(resdir,"2021-11-09")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"0")
if not exists(expdir): mkdir(expdir)
savedir = expdir


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
    if dssource == 's2s':
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
    gh = get_gh(ds,dssource,i_mem)
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
    
def plot_uref(ds,dssource,fig=None,ax=None):
    # Plot the zonal-mean zonal wind at 10 hPa and 60N, for all ensemble members.
    if fig is None or ax is None:
        fig,ax = plt.subplots()
    i_mem = 0
    i_lev = get_i_lev_hPa(ds,dssource,10) 
    i_lat = np.argmin(np.abs(ds['lat'][:] - 60))
    Nmem = get_ensemble_size(ds,dssource)
    for i_mem in range(Nmem):
        uref = np.zeros(ds['time'].size)
        u,v = compute_geostrophic_wind(ds,dssource,i_mem)
        uref = np.mean(u[:,i_lev,i_lat,:],axis=1)
        ax.plot(ds['time'][:]/24.0,uref,color='black')
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"$\overline{u}$ (10 hPa, 60$^\circ$N")
    return fig,ax

def vectorize_ensemble(ds,dssource):
    # Take a netcdf file with records in order and put it into a big matrix. This is where the ordering is defined. 
    # Output shape: (Nmem,Nt,xdim)
    # This can then be put into a bigger array with Nens as the first dimension
    Nmem = get_ensemble_size(ds,dssource)
    Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
    xdim = Nlev*Nlat*Nlon
    x = np.zeros((Nmem,Nt,xdim))
    for i_mem in range(Nmem):
        gh = get_gh(ds,dssource,i_mem=i_mem)
        x[i_mem,:,:] = gh.reshape((Nt,xdim))
    return x

def unvectorize_ensemble(x,dsname,dsref):
    # TODO
    # Given a vector (Nmem,Nt,xdim), form a netcdf appropriately
    ds = nc.Dataset(dsname,mode='w',format='NETCDF4')
    Nmem,Nt,xdim = x.shape
    ds.createDimension('memidx',Nmem)
    ds.createDimension('time',Nt)
    return

def find_eofs(dsfolder,dssource,flist,num_pc=None):
    # Given a folder for the datasets and a list of filenames, aggregate all the data together and create EOFs 
    # Leave some ghost functionality for time-lagged stuff
    ds0 = nc.Dataset(join(dsfolder,flist[0]),"r")
    lat,lon,plev = [ds0[v][:] for v in ['lat','lon','plev']]
    cosine = np.zeros((len(plev),len(lat),len(lon)))
    for i in range(len(plev)):
        cosine[i,:,:] = np.cos(np.pi/180 * np.outer(lat,np.ones(len(lon))))
    cosine = cosine.flatten()
    Nmem = get_ensemble_size(ds0,dssource)
    xdim = ds0['plev'].size*ds0['lat'].size*ds0['lon'].size
    ds0.close()
    X = np.zeros((Nmem,0,xdim))
    for i in range(len(flist)):
        if i % 10 == 0:
            print("Reading file {} out of {}".format(i,len(flist)))
        ds = nc.Dataset(join(dsfolder,flist[i]),"r")
        Xnew = vectorize_ensemble(ds,dssource)
        X = np.concatenate((X,Xnew),axis=1)
        ds.close()
    X = X.reshape((X.shape[0]*X.shape[1],xdim))
    Xmean = np.mean(X,axis=0)
    U,S,Vh = np.linalg.svd(((X-Xmean)*cosine).T/np.sqrt(len(X)), full_matrices=False) # Columns of U are spatial patterns; rows of Vh are coefficients
    print("S.shape = {}, S[:20] = {}".format(S.shape,S[:20]))
    # Truncate at 90% of variance
    running_var = np.cumsum(S**2)
    if num_pc is None:
        num_pc = np.where(running_var/running_var[-1] >= 0.9)[0][0]
    print("num_pc = {} out of {}".format(num_pc,len(S)))
    # In general, for a linear or nonlinear dimensionality reduction, we need a function to project the data. In the EOF case, that means U, S, (not V), and Xmean.
    # Save the EOFs as their own netcdf
    svd = {"left_singvecs": U[:,:num_pc].reshape((len(plev),len(lat),len(lon),num_pc)), "singvals": S, "mean": Xmean.reshape((len(plev),len(lat),len(lon))), "lat": lat, "lon": lon, "plev": plev, "dssource": dssource}
    pickle.dump(svd,open(join(savedir,"svd"),"wb"))
    return #U[:,:num_pc],S,Vh[:num_pc,:],Xmean

def plot_eofs(num_pc_display=3):
    # Plot the EOFs
    svd = pickle.load(open(join(savedir,"svd"),"rb"))
    num_pc = svd["left_singvecs"].shape[-1]
    i_lev = 0
    for i in range(num_pc_display):
        fig,ax,data_crs = display_pole_field(svd["left_singvecs"][i_lev,:,:,i],svd["lat"],svd["lon"])
        ax.set_title("EOF {}".format(i))
        fig.savefig(join(savedir,"eof{}".format(i)))
        plt.close(fig)
    fig,ax = plt.subplots()
    num_pc_plot = min(2*num_pc,len(svd["singvals"]))
    ax.plot(np.arange(num_pc_plot),svd["singvals"][:num_pc_plot],color='black')
    ax.set_yscale('log')
    ax.set_title("Singular values")
    fig.savefig(join(savedir,"singvals"))
    plt.close(fig)
    return

def project_onto_eofs(ds,dssource):
    # Take a dataset and project it onto the EOFs to determine the reduced coordinates.
    svd = pickle.load(open(join(savedir,"svd"),"rb"))
    Nmem = get_ensemble_size(ds,dssource)
    Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
    num_pc = svd["left_singvecs"].shape[-1]
    Y = vectorize_ensemble(ds,dssource) 
    Y = Y.reshape((Nmem*Nt, Nlev*Nlat*Nlon))
    Y = Y - svd["mean"].flatten()
    pc = Y.dot(svd["left_singvecs"].reshape((Nlev*Nlat*Nlon,num_pc)))
    pc = pc/svd["singvals"][:num_pc]
    pc = pc.reshape((Nmem,Nt,num_pc))
    return pc

def plot_pcs(dsfolder,dssource,flist,num_pc_display=3):
    for f in flist:
        ds = nc.Dataset(join(dsfolder,f),"r")
        pc = project_onto_eofs(ds,dssource)
        Nmem,Nt,xdim = pc.shape
        for i_pc in range(num_pc_display):
            fig,ax = plt.subplots()
            for i_mem in range(Nmem):
                ax.plot(ds['time'][:]/24,pc[i_mem,:,i_pc],color='red')
            ax.set_xlabel("Time [days]")
            ax.set_ylabel("PC")
            ax.set_ylabel("PC {}".format(i_pc))
            ax1 = ax.twinx()
            _,_ = plot_uref(ds,dssource,fig=fig,ax=ax1)
            ax1.tick_params(axis='y',color='red')
            fig.savefig(join(savedir,"pc{}_{}".format(i_pc,f.replace(".nc",""))))
            plt.close(fig)
        ds.close()
    return

def featurize_dataset(ds,dssource):
    # Convert a dataset into a feature space including both zonal-mean zonal wind and several PCs. 
    svd = pickle.load(open(join(savedir,"svd"),"rb"))
    num_pc = svd["left_singvecs"].shape[-1]
    Nfeat = num_pc + 1 # include zmzw
    Nmem = get_ensemble_size(ds,dssource)
    Nt,Nlev,Nlat,Nlon = [ds[v].size for v in ['time','plev','lat','lon']]
    pres = 10
    i_lev = get_i_lev_hPa(ds,dssource,pres)
    i_lat = np.argmin(np.abs(ds['lat'][:] - 60))
    # Set up the feature vector
    x = np.zeros((Nmem,Nt,Nfeat))
    # Put in zonal wind
    for i_mem in range(Nmem):
        u,v = compute_geostrophic_wind(ds,dssource)
        x[i_mem,:,0] = np.mean(u[:,i_lev,i_lat,:],axis=1)
    # Project PCs 
    x[:,:,1:num_pc+1] = project_onto_eofs(ds,dssource)
    # TODO: should this be netcdf as well? Or pandas array?
    return x

def main1():
    find_eofs_flag =    0
    plot_pcs_flag =     1
    plot_eofs_flag =    1
    dsfolder = datadir_era20c
    dssource = 'era20c'
    flist = []
    for fall_year in np.arange(1900,2007,20):
        flist += ["%i-11-01_to_%i-04-30.nc"%(fall_year,fall_year+1)]
    if find_eofs_flag:
        find_eofs(dsfolder,dssource,flist)
    if plot_pcs_flag:
        plot_pcs(dsfolder,dssource,flist)
    if plot_eofs_flag:
        plot_eofs(10)
    # Now plot winters colored red or blue
    return


def main0():
    #ds = nc.Dataset(join(datadir,"hc2012-11-03_rt2016-11-03.nc"),"r")
    fall_year = 2006
    ds = nc.Dataset(join(datadir_era20c,"%i-11-01_to_%i-04-30.nc"%(fall_year,fall_year+1)),"r")
    print("time = {}".format(ds['time']))
    dssource = 'era20c'
    fig,ax = plot_uref(ds,dssource)
    fig.savefig(join(savedir,"uref_era20c_fy%i"%(fall_year)))
    plt.close(fig)
    i_mem = 0
    pres = 10.0
    i_lev = get_i_lev_hPa(ds,dssource,pres)
    start_day = 110
    end_day = 120
    i_time_start = np.argmin(np.abs(ds['time'][:] - start_day*24))
    i_time_end = np.argmin(np.abs(ds['time'][:] - end_day*24))
    Nmem = get_ensemble_size(ds,dssource)
    u,v = compute_geostrophic_wind(ds,dssource,i_mem=i_mem)
    gh = get_gh(ds,dssource)
    for i_time in np.arange(i_time_start,i_time_end): #np.linspace(0,ds['time'].size-1,ds['time'].size).astype(int):
        #fig,ax = show_gh_onelevel_cartopy(ds,i_mem,i_time,i_lev)
        #fig.savefig(join(savedir,"gh_mem{}_t{}_p{}".format(i_mem,i_time,i_lev)))
        #plt.close(fig)
        fig,ax = show_ugh_onelevel_cartopy(gh[i_time,i_lev,:,:],u[i_time,i_lev,:,:],v[i_time,i_lev,:,:],ds['lat'][:],ds['lon'][:])
        ax.set_title(r"$\Phi$, $u$ at $p=$%i hPa, day %i"%(pres,ds['time'][i_time]))
        #fig.savefig(join(savedir,"u_{}_mem{}_t{}_p{}".format(i_mem,i_time,i_lev)))
        fig.savefig(join(savedir,"u_fy{}_it{}".format(fall_year,i_time)))
        plt.close(fig)
    ds.close()

if __name__ == "__main__":
    main1()
    main0()


