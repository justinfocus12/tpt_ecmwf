# Preprocess the ERA5 data to conform to same format
import numpy as np
import os
from os import mkdir
from os.path import exists,join
import netCDF4 as nc
from calendar import monthrange

datadir = "/scratch/jf4241/ecmwf_data/era5_data/2022-03-10"
if not exists(datadir):
    raise Exception(f"The data directory {datadir} does not exist")

# Loop through the files to recombine into netcdfs 1 per winter
for fall_year in np.arange(1950,2020):
    print(f"Starting fall year {fall_year}")
    # Load the fall and spring datasets
    ds_fall = nc.Dataset(join(datadir,"%i.nc"%(fall_year)),"r")
    ds_spring = nc.Dataset(join(datadir,"%i.nc"%(fall_year+1)),"r")
    # Determine the start time for the fall year and the end time for the spring year
    Nlat = ds_fall['latitude'].size
    Nlon = ds_fall['longitude'].size
    Nt_fall = np.sum([monthrange(fall_year,i)[1] for i in [10,11,12]])
    Nt_spring = np.sum([monthrange(fall_year+1,i)[1] for i in [1,2,3,4]])
    # Create the new dataset
    dsname = join(datadir,"%i-10-01_to_%i-04-30.nc"%(fall_year,fall_year+1))
    ds = nc.Dataset(dsname, "w", format="NETCDF4") 
    # Create time dimension
    time = ds.createDimension("time", Nt_fall+Nt_spring)
    ds.createVariable("time", "f4", ("time",))
    ds["time"][:Nt_fall] = ds_fall["time"][-Nt_fall:]
    ds["time"][Nt_fall:] = ds_spring["time"][:Nt_spring]
    ds["time"].units = ds_fall["time"].units
    ds["time"].calendar = ds_fall["time"].calendar
    # Create pressure dimension
    plev = ds.createDimension("plev", 1)
    ds.createVariable("plev", "f4", ("plev",))
    ds["plev"][0] = 10.0 # hPa 
    ds["plev"].units = "hPa"
    # Create latitude dimension
    lat = ds.createDimension("lat", Nlat)
    ds.createVariable("lat", "f4", ("lat",))
    ds["lat"][:] = ds_fall["latitude"][:]
    ds["lat"].units = ds_fall["latitude"].units
    # Create longitude dimensions
    lon = ds.createDimension("lon", Nlon)
    ds.createVariable("lon", "f4", ("lon",))
    ds["lon"][:] = ds_fall["longitude"][:]
    ds["lon"].units = ds_fall["longitude"].units
    # Create the variable for zonal wind
    ds.createVariable("var131", "f4", ("time","plev","lat","lon"))
    ds["var131"][:Nt_fall,0,:,:] = ds_fall["u"][-Nt_fall:,:,:]
    ds["var131"][Nt_fall:,0,:,:] = ds_spring["u"][:Nt_spring,:,:]
    ds["var131"].units = ds_fall["u"].units

    ds_fall.close()
    ds_spring.close()
    ds.close()

    print(f"Finished fall year {fall_year}")




