import numpy as np
import netCDF4 as nc
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import rcParams 
from matplotlib import ticker
rcParams["font.size"] = 40
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
    da = xr.DataArray(coords={"x": [0,1,2], "y": [-10.5,-9.5,-8.6,2.3]}, dims=["x","y"], data=np.arange(12).reshape((3,4)))
    return da

def compute_eofs(file_list, num_modes=10, months_of_interest=None):
    # Compute some EOFs of monthly mean data 
    # Each file must contain an integer number of whole months
    if months_of_interest is None:
        months_of_interest = np.arange(1, 13)
    z_list = []
    for i_f,f in enumerate(file_list):
        if i_f % 20 == 0:
            print(f"starting file {i_f} out of {len(file_list)}")
        zmean = xr.open_dataset(f)['z'].resample(time="1M").mean()
        z_list += [zmean] # Using the fact that each file is a whole month
    zm = xr.concat(z_list, dim="time").sortby("time")
    zm_climatology = zm.groupby("time.month").mean(dim="time")
    zma = zm.groupby("time.month") - zm_climatology # Anomaly
    # Restrict to latitudes 20 degrees and northward
    zm_20N_w = zma.where(
        (zma.latitude >= 20) *
        zma.month.isin(months_of_interest),
        drop=True
    ) 
    zm_20N_w *= np.sqrt(np.maximum(0, np.cos(zm_20N_w.latitude*np.pi/180)))
    X = zm_20N_w.transpose("level","time","latitude","longitude").to_numpy().reshape((
        zm_20N_w.level.size, zm_20N_w.time.size, zm_20N_w.latitude.size*zm_20N_w.longitude.size, 
    ))
    
    num_modes = 10
    eofs = xr.DataArray(
        coords = {
            "level": zm_20N_w.level, 
            "latitude": zm_20N_w.latitude, 
            "longitude": zm_20N_w.longitude, 
            "mode": 1 + np.arange(num_modes)},
        dims = ["level", "latitude", "longitude", "mode"],
        )
    variance_fraction = xr.DataArray(
        coords = {"level": zm_20N_w.level, "mode": 1 + np.arange(num_modes),},
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
        data_vars = {"z": zm_climatology}
        )
    return ds_eofs, ds_monclim 
