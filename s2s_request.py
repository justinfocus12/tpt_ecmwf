# This script dowloads 20 years of S2S (November-April) from the public-facing ECMWF data catalogue.
# Notes:
# 1. "rt" usually means "real-time"; "hc" usually means "hindcast"
# 2. The SSW season is winter, and we label each winter by the year of its autumn. For example, winter 1999-2000 is labeled with the year 1999. 

from ecmwfapi import ECMWFDataServer
import numpy as np
import sys
import os
from os import mkdir,system
from os.path import exists,join
import datetime

# ------ Speficy directories and wintertime parameters -------
datadir = "/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23"
if not exists(datadir): mkdir(datadir)
# The SSW season ("winter") is defined from Oct. 1 to ~Apr. 30
winter_start_month = 10 # October
winter_length = 212 # 7 months after Oct. 1
winter_start_day = 1 # That's Oct. 1
# Set up the list of reforecasts, based on the current year
year_rt = 2016 # Year of real-time AUTUMN. Hindcasts go back 20 years, so we will get the time range 1996-2016
weekdays_rt = [0,3] # Monday and Thursday are when the hindcasts are launched
year_list_hc = np.arange(year_rt-20,year_rt) # List of years with hindcasts
# -----------------------------------------------------------

# Establish server connection and download each relevant hindcast. 
server = ECMWFDataServer()
winter_start_date_rt = datetime.datetime(year_rt,winter_start_month,winter_start_day)
for k in range(winter_length): # Loop through days of winter
    date_rt = winter_start_date_rt + datetime.timedelta(days = k)
    date_rt_str = f"{date_rt.year:04d}-{date_rt.month:02d}-{date_rt.day:02d}"
    for j in range(len(year_list_hc)): # Loop through hindcast years
        year_hc = year_list_hc[j]
        if date_rt.month <= 6: # If it's on Jan. 1 or later, the date corresponding to day k has a year which is 1 greater than the corresponding autumn of the same winter. 
            year_hc += 1 
        if not (date_rt.month == 2 and date_rt.day == 29): # We ignore all leap days to head off headaches. This reduces the number of data points by a small amount. 
            date_hc = datetime.datetime(year_hc,date_rt.month,date_rt.day)
            if date_rt.weekday() in weekdays_rt:
                date_hc_str = f"{date_hc.year:04d}-{date_hc.month:04d}-{date_hc.day:04d}"
                # Name the output filename to include both the hindcast and real-time dates.
                target = join(datadir,f"hc{date_hc_str}_rt{date_rt_str}.grb")
                # Specify the object to retrieve according to ECMWF API
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

