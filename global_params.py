
# ------------- Set directories for code and data ----------
codedir = "/home/jf4241/ecmwf/tpt_ecmwf"
datadirs = dict({
    "ei": "/scratch/jf4241/ecmwf_data/eraint_data/2022-02-10",
    "e2": "/scratch/jf4241/ecmwf_data/era20c_data/2022-02-10",
    "e5": "/scratch/jf4241/ecmwf_data/era5_data/2022-03-10",
    "s2s": "/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23",
    })
data_sources = list(datadirs.keys())
featdir = "/scratch/jf4241/ecmwf_data/features/2022-04-01" # Directory to store feature definitions
if not exists(featdir): mkdir(featdir)
feat_display_dir = join(featdir,"display0") # Directory to display features
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results" # Directory to store results of pipeline
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2022-04-01") # Directory of day of running code 
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"1") # Directory n for nth experiment on the current day
if not exists(expdir): mkdir(expdir)
# ---------------------------------------------------------

# -------------- Set the full year ranges and the full month ranges for each dataset ---------------
# For S2S, a single real-time model year performs hindcasts on the previous 20 years
year_rt_s2s = 2016
fall_years = dict({
    "ei": np.arange(1979,2018),
    "e2": np.arange(1900,2008),
    "e5": np.arange(1950,2020),
    "s2s": np.arange(year_rt_s2s-20,year_rt_s2s+1),
    })
# Define the season in which SSWs can occur (winter). The first day of winter_start_month will be the time origin in the following TPT analysis. 
winter_start_month = 10  
winter_end_month = 4
winter_length = sum([monthrange(1901,i)[1] for i in range(winter_start_month,13)]) + sum([monthrange(1902,i)[1] for i in range(winter_end_month+1)])

