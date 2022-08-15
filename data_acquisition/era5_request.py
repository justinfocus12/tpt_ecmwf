import cdsapi
import numpy as np
from os.path import join
from calendar import monthrange

datadir = '/scratch/jf4241/ecmwf_data/era5_data/2022-08-14/'

c = cdsapi.Client()


import cdsapi

c = cdsapi.Client()

# Download data one month at a time
year_list = np.arange(2010,2020)
month_list = np.arange(1,13)

for year in year_list:
    for month in month_list:
        print(f"Beginning to download year {year}")
        target_filename = join(datadir,f"{year:04}-{month:02}.nc")
        _,days_in_month = monthrange(year,month)
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential', 'temperature',
                    'u_component_of_wind',
                    'v_component_of_wind',
                ],
                'pressure_level': [
                    '10', '100', '500', '850',
                ],
                'year': [f"{year:04}"],
                'month': [f"{month:02}"], 
                'day': [f"{day:02}" for day in range(1,days_in_month+1)],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'grid': [2.5, 2.5],
                'format': 'netcdf',
            },
            target_filename)

