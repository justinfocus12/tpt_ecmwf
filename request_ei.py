#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import numpy as np
import os
from os import mkdir,system
from os.path import exists,join

server = ECMWFDataServer()
#task_id = int(sys.argv[1])
# Load cdo
#system("module load cdo/intel/1.9.10")
# Directories
datadir = "/scratch/jf4241/ecmwf_data/eraint_data/2022-02-10"
if not exists(datadir): mkdir(datadir)
#savedir = "./gridmath"
#if not exists(savedir): mkdir(savedir)
# Between years 1900-2008
for fall_year in np.arange(1979,2019):
    print("\n\n{}Beginning year {}{}\n".format("-"*10+" ",fall_year,"-"*10+" "))
    #target = join(datadir,"winter%i-%i.grb"%(fall_year,fall_year+1))
    target = join(datadir,"%i-10-01_to_%i-04-30.grb"%(fall_year,fall_year+1))
    retr = {
        "class": "ei",
        "dataset": "interim",
        "date": "%i-10-01/to/%i-04-30"%(fall_year,fall_year+1),
        "expver": "1",
        "grid": "3.0/3.0",
        "levelist": "10/50/100/200/300/500/700/850/925/1000",
        "levtype": "pl",
        #"param": "131.128",
        "param": "129.128/131.128",
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "target": target,
    }
    server.retrieve(retr)
## Now convert to netcdf
#cmd = "cdo -f nc copy %s %s.nc"%(target,target)
#system(cmd)
#cmd = "rm %s"%(target)

