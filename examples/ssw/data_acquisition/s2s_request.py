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


# ----------- Try a normal request -------------
if True:
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
    ##!/usr/bin/env python
    #server.retrieve({
    #    "class": "s2",
    #    "dataset": "s2s",
    #    "date": "2021-11-15",
    #    "expver": "prod",
    #    "hdate": "2001-11-15",
    #    "grid": "3.0/3.0",
    #    "levelist": "10/50/100/200/300/500/700/850/925/1000",
    #    "levtype": "pl",
    #    "model": "glob",
    #    "number": "1/2/3/4/5/6/7/8/9/10",
    #    "origin": "ecmf",
    #    "param": "156",
    #    "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104",
    #    "stream": "enfh",
    #    "time": "00:00:00",
    #    "type": "pf",
    #    "target": join(datadir,"test.grb"),
    #})
    
    sys.exit()

# ----------------------------------------------

winter_length = 180 # 6 months after Nov. 1
winter_start_month = 11
winter_start_day = 1 # That's November 1!
# Set up the list of reforecasts, based on the current year
year_rt = 2016 # So that we get the time range 1996-2016
weekdays_rt = [0,3] # Monday and Thursday
year_list_hc = np.arange(year_rt-20,year_rt)

# Go from November 1 to (November 1 + 180 days)
winter_start_date_rt = datetime.datetime(year_rt,winter_start_month,winter_start_day)
for k in range(45,winter_length):
    date_rt = winter_start_date_rt + datetime.timedelta(days = k)
    date_rt_str = "%s-%s-%s"%(digstr(date_rt.year),digstr(date_rt.month),digstr(date_rt.day))
    print("date_rt = {}; date_rt_str = {}".format(date_rt,date_rt_str))
    for j in range(len(year_list_hc)):
        year_hc = year_list_hc[j]
        if date_rt.month <= 6: 
            year_hc += 1
        if not (date_rt.month == 2 and date_rt.day == 29):
            date_hc = datetime.datetime(year_hc,date_rt.month,date_rt.day)
            #print("date_rt.weekday() = {}".format(date_rt.weekday()))
            if date_rt.weekday() in weekdays_rt:
                date_hc_str = "%s-%s-%s"%(digstr(date_hc.year),digstr(date_hc.month),digstr(date_hc.day))
                print("date_hc = {}; date_hc_str = {}".format(date_hc,date_hc_str))
                target = join(datadir,"hc%s_rt%s.grb"%(date_hc_str,date_rt_str))
                retr = ({
                    "class": "s2",
                    "dataset": "s2s",
                    "date": date_rt_str,
                    "expver": "prod",
                    "grid": "3.0/3.0",
                    "hdate": date_hc_str,
                    "levelist": "10/50/100/200/300/500/700/850/925/1000",
                    "levtype": "pl",
                    "model": "glob",
                    "number": "1/2/3/4/5/6/7/8/9/10",
                    "origin": "ecmf",
                    "param": "131/156",
                    "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104",
                    "stream": "enfh",
                    "time": "00:00:00",
                    "type": "pf",
                    "target": target,
                })
                server.retrieve(retr)
## Now convert to netcdf
#cmd = "cdo -f nc copy %s %s.nc"%(target,target)
#system(cmd)
#cmd = "rm %s"%(target)

