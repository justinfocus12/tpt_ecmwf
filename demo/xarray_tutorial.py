import numpy as np
import xarray as xr
import netCDF4 as nc

rng = np.random.default_rng(seed=0)

da = xr.DataArray(
        np.ones((3,4,2)),
        dims=('x','y','z'),
        name='a',
        coords={'z':[-1,1],'u':('x',[.1,1.2,2.3])},
        attrs={'attr': 'value'},
        )
#print(f"da = {da}")
#print(f"da.get_index('x') = {da.get_index('z')}")

# Make a dataset with different features 
X = xr.DataArray(
        np.zeros((2,9,4)), 
        dims = ("member","time","feature"),
        coords = {
            "feature": ["time_d","uref_dl0","uref_dl5","captemp"],
            "member": np.arange(2),
            "time": np.linspace(0,4,9)
            },
        )

X.data[:,:,0] = np.outer(np.ones(X.shape[0]),np.arange(X.shape[1]))
print(f"X.data = \n{X.data}")
# Try to look up the feature index corresponding to uref_dl5
time_d = X.loc[:,:,'time_d']
print(f"time_d = {time_d}")
time_d = X.sel(feature='time_d')
print(f"time_d = {time_d}")
# Try to change the time
X.loc[dict(feature='time_d',member=1)] *= 24.0
print(f"After adjusting, X.data = \n{X.data}")
# Save X to netcdf
X_ds = xr.Dataset({"X": X})
X_ds.to_netcdf("xarray_tutorial.nc")

# Now load it 
Y_ds = xr.open_dataset("xarray_tutorial.nc")
print(f"Y_ds = \n{Y_ds}")
Y = Y_ds["X"]
print(f"Y.data = \n{Y.data}")
Y.close()

del X_ds 
del X
del Y_ds
del Y
# Stacking and unstacking
A = xr.DataArray(
        np.arange(24).reshape((2,3,4)),
        dims=("x","y","z"),
        )
print(f"A before stacking = \n{A}")
A_stacked = A.stack(yz=("y","z"))
print(f"A_stacked = \n{A_stacked}")
print(f"The old way: \n{A.data.reshape((2,3*4))}")
print(f"A_roundtripped = \n{A_stacked.unstack()}")

xr.Dataset({"A": A}).to_netcdf("A.nc")

