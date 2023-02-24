#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import numpy as np
import sys
import os
from os import mkdir,system
from os.path import exists,join
import datetime

server = ECMWFDataServer()

def digstr(n):
    if type(n) is not int:
        raise Exception("ERROR: n must be an integer")
    if n < 10:
        return "0%i"%(n)
    return "%i"%(n)

#task_id = int(sys.argv[1])
# Directories
datadir = "/scratch/jf4241/ecmwf_data/s2s_data/raw_data/2022-08-20"
if not exists(datadir): mkdir(datadir)
#savedir = "./gridmath"
#if not exists(savedir): mkdir(savedir)


# ----------- Dry run: one single request -------------
if False:
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": "2022-07-11",
        "expver": "prod",
        "hdate": "2002-07-11",
        "grid": "2.5/2.5",
        "levelist": "10/100/500/850",
        "levtype": "pl",
        "model": "glob",
        "number": "1/2/3/4/5/6/7/8/9/10",
        "origin": "ecmf",
        "param": "130/131/132/156",
        "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104",
        "stream": "enfh",
        "time": "00:00:00",
        "type": "pf",
        "target": join(datadir,"test.grib"),
    })
    sys.exit()

# ----------------------------------------------

# Set up the list of reforecasts, based on the current year
date_rt_first = datetime.datetime(2016, 7, 1)
num_days_rt = 366
weekdays_rt = [0,3] # Monday and Thursday
hc_back_extent = 20 # years in past 

for k in np.arange(143,144):
    date_rt = date_rt_first + datetime.timedelta(days = int(k))
    date_rt_str = f"{date_rt.year:04}-{date_rt.month:02}-{date_rt.day:02}"
    print(f"date_rt = {date_rt}; date_rt_str = {date_rt_str}")
    if date_rt.weekday() in weekdays_rt:
        for year_hc in np.arange(date_rt.year-hc_back_extent, date_rt.year)[::-1]:
            valid_date = None
            try:
                date_hc = datetime.datetime(year_hc,date_rt.month,date_rt.day)
                valid_date = True
            except ValueError:
                valid_date = False
            if valid_date:
                date_hc_str = f"{date_hc.year:04}-{date_hc.month:02}-{date_hc.day:02}"
                target = join(datadir,f"hc{date_hc_str}_rt{date_rt_str}.nc")
                retr = ({
                    "class": "s2",
                    "dataset": "s2s",
                    "date": date_rt_str,
                    "expver": "prod",
                    "grid": "2.5/2.5",
                    "hdate": date_hc_str,
                    "levelist": "10/100/500/850",
                    "levtype": "pl",
                    "model": "glob",
                    "number": "1/2/3/4/5/6/7/8/9/10",
                    "origin": "ecmf",
                    "param": "130/131/132/156",
                    "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104",
                    "stream": "enfh",
                    "time": "00:00:00",
                    "type": "pf",
                    "format": "netcdf",
                    "target": target,
                })
                server.retrieve(retr)

