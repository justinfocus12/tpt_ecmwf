import numpy as np
import netCDF4 as nc
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import rcParams 
from matplotlib import ticker
rcParams["font.size"] = 18
rcParams["font.family"] = "monospace"
import cartopy
from cartopy import crs as ccrs


def display_pole_field(field,lat,lon,u=None,v=None,vmin=None,vmax=None,fig=None,ax=None):
    # field, u, and v must have shape (lat, lon)
    data_crs = ccrs.PlateCarree() 
    ax_crs = ccrs.Orthographic(-10,90)
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection=ax_crs)
    im = ax.pcolormesh(lon,lat,field,shading='nearest',cmap='coolwarm',transform=data_crs,vmin=vmin,vmax=vmax)
    # If there's a vector field, overlay that 
    if u is not None and v is not None:
        lon_subset = np.arange(0,lon.size,3) #np.linspace(0,lon.size-1,24).astype(int)
        lat_subset = np.arange(2,lat.size-1,3) #np.linspace(1,lat.size-2,60).astype(int)
        ax.quiver(
            lon[lon_subset],lat[lat_subset],
            u[lat_subset,:][:,lon_subset],
            v[lat_subset,:][:,lon_subset],
            transform=data_crs,color='black',zorder=5,
        )
    ax.add_feature(cartopy.feature.COASTLINE, zorder=3, edgecolor='black')
    fig.colorbar(im,ax=ax)
    return fig,ax,data_crs

def dullda():
    da = xr.DataArray(coords={"x": [0,1,2], "y": [-10.5,-9.5,-8.6,2.3]}, dims=["x","y"], data=np.arange(12).astype('float64').reshape((3,4)))
    return da

def preprocess_netcdf_basic(ds_in, src):
    # Depending on what dataset the netcdf came from, rename some dimensions 
    ds = ds_in.copy()
    if src == "s2": 
        ds = ds.rename({"number": "member"})
    if src == "e5":
        ds["z"][:] *= 1.0/9.806 # convert to geopotential height
        ds = ds.rename({"z": "gh"})
    return ds

def compute_eofs(ds, src, num_modes=10, months_of_interest=None):
    # TODO: finish this in a robust way to feed into the new POP analysis. 
    # Given a dask dataset ds, compute the EOFs of 10-day running average of each level separately. Also compute the seasonal mean and 
# Go through each level and get EOFs and PCs for each. 
    znm = (
        ds["z"]
        .isel(latitude=np.where(e5.latitude >= 20)[0])
        #.rolling(dim={"t_szn": 5}, min_periods=1).mean()
        .isel(t_szn=np.arange(0, szn_duration, 10).astype(int))
    )
    eofs_by_level = xr.DataArray(
        coords = {"level": e5.level, "mode": np.arange(num_eofs), "latitude": znm.latitude, "longitude": znm.longitude},
        dims = ["level","mode","latitude","longitude"],
        data = np.nan,
    )
    eigval = xr.DataArray(
        coords = {"level": e5.level, "mode": np.arange(num_eofs)}, dims=["level","mode"]
    )
    zmean = xr.DataArray(
        coords={"level": e5.level, "latitude": znm.latitude, "longitude": znm.longitude}, 
        dims=["level","latitude","longitude"]
    )
    zvariance = xr.DataArray(
        coords={"level": e5.level}, dims=["level"]
    )
    for lev in znm.level.data:
        print(f"Starting level {lev}")
        zmean.loc[dict(level=lev)] = znm.sel(level=lev).mean(dim=["fall_year","t_szn"])
        solver = Eof(
            znm.sel(level=lev)
            .stack({"time": ["fall_year","t_szn"]})
            .transpose("time","latitude","longitude").load(),
            weights=weights.transpose("latitude","longitude")
        )
        eofs_by_level.loc[dict(level=lev)] = solver.eofs().isel(mode=range(num_eofs))
        eigval.loc[dict(level=lev)] = solver.eigenvalues().isel(mode=range(num_eofs))
        zvariance.loc[dict(level=lev)] = solver.eigenvalues().sum()
    
    ds_eof = xr.Dataset({
        "eofs": eofs_by_level,
        "eigval": eigval,
        "zmean": zmean,
        "zvariance": zvariance,
    })

def old_compute_eofs(file_list, src, num_modes=10, months_of_interest=None):
    # Compute some EOFs of monthly mean data 
    # Each file must contain an integer number of whole months
    if months_of_interest is None:
        months_of_interest = np.arange(1, 13)
    gh_list = []
    for i_f,f in enumerate(file_list):
        if i_f % 20 == 0:
            print(f"starting file {i_f} out of {len(file_list)}")
        ds = preprocess_netcdf_basic(xr.open_dataset(f), src)
        ghmean = ds['gh'].resample(time="1M").mean()
        gh_list += [ghmean] # Using the fact that each file is a whole month
    ghm = xr.concat(gh_list, dim="time").sortby("time")
    ghm_climatology = ghm.groupby("time.month").mean(dim="time")
    ghma = ghm.groupby("time.month") - ghm_climatology # Anomaly
    # Restrict to latitudes 20 degrees and northward
    ghm_20N_w = ghma.where(
        (ghma.latitude >= 20) *
        ghma.month.isin(months_of_interest),
        drop=True
    ) 
    ghm_20N_w *= np.sqrt(np.maximum(0, np.cos(ghm_20N_w.latitude*np.pi/180)))
    X = ghm_20N_w.transpose("level","time","latitude","longitude").to_numpy().reshape((
        ghm_20N_w.level.size, ghm_20N_w.time.size, ghm_20N_w.latitude.size*ghm_20N_w.longitude.size, 
    ))
    
    num_modes = 10
    eofs = xr.DataArray(
        coords = {
            "level": ghm_20N_w.level, 
            "latitude": ghm_20N_w.latitude, 
            "longitude": ghm_20N_w.longitude, 
            "mode": 1 + np.arange(num_modes)},
        dims = ["level", "latitude", "longitude", "mode"],
        )
    variance_fraction = xr.DataArray(
        coords = {"level": ghm_20N_w.level, "mode": 1 + np.arange(num_modes),},
        dims = ["level","mode"],
        )
    for i_lev in range(eofs.level.size):
        U,S,Vh = np.linalg.svd(X[i_lev].T, full_matrices=False)
        eofs[dict(level=i_lev)] = U[:,:num_modes].reshape((eofs.latitude.size, eofs.longitude.size, eofs.mode.size))
        variance_fraction[dict(level=i_lev)] = S[:num_modes]**2 / np.sum(S**2)
    ds_eofs = xr.Dataset(
        data_vars = {"eofs": eofs, "variance_fraction": variance_fraction,},
        )
    ds_monclim = xr.Dataset(
        data_vars = {"gh": ghm_climatology}
        )
    return ds_eofs, ds_monclim 

