import cdsapi
import numpy as np
from os.path import join
from calendar import monthrange


datadir = '/scratch/jf4241/ecmwf_data/era5_data/2022-03-10/'
winter_months = [1,2,3,4,10,11,12]

c = cdsapi.Client()

for year in range(1973,1979):
    target = join(datadir,"{:04}.nc".format(year))
    print(f"year = {year}, target = {target}")
    c.retrieve(
        'reanalysis-era5-pressure-levels-preliminary-back-extension',
        {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': 'u_component_of_wind',
        'pressure_level': '10',
        'year': '{:04}'.format(year),
        'month': ['{:02}'.format(month) for month in winter_months],
        'day': ['{:02}'.format(day) for day in range(1,32)],
        'time': '00:00',
        'format': 'netcdf',
        'area': [75, -180, 45, 180],
        },
        target,
        )
    print(f"Finished year {year}")

