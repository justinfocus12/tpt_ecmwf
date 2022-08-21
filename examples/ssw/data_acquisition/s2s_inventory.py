import numpy as np
import os
from os import listdir,mkdir,system
from os.path import exists, join
import datetime
import sys


datadir = "/scratch/jf4241/ecmwf_data/s2s_data/raw_data/2022-08-20"


date_rt_first = datetime.datetime(2016, 7, 1)
num_days_rt = 366
weekdays_rt = [0,3] # Monday and Thursday
hc_back_extent = 20 # years in past 

target_list = []
target_flag = []

for k in range(366):
    date_rt = date_rt_first + datetime.timedelta(days = k)
    date_rt_str = f"{date_rt.year:04}-{date_rt.month:02}-{date_rt.day:02}"
    if date_rt.weekday() in weekdays_rt:
        for year_hc in np.arange(date_rt.year-hc_back_extent, date_rt.year)[2:]:
            valid_date = None
            try:
                date_hc = datetime.datetime(year_hc,date_rt.month,date_rt.day)
                valid_date = True
            except ValueError:
                valid_date = False
            if valid_date:
                date_hc_str = f"{date_hc.year:04}-{date_hc.month:02}-{date_hc.day:02}"
                target_ext = f"hc{date_hc_str}_rt{date_rt_str}.nc"
                target = join(datadir,target_ext)
                target_list += [target_ext]
                target_flag += [exists(target)]

