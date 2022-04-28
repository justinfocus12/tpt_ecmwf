import numpy as np
import xarray as xr

rng = np.random.default_rng(seed=0)

da = xr.DataArray(
        np.ones((3,4,2)),
        dims=('x','y','z'),
        name='a',
        coords={'z':[-1,1],'u':('x',[.1,1.2,2.3])},
        attrs={'attr': 'value'},
        )
