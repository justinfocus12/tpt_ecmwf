# Plot zonal wind as derived from geopotential, and over all the winters
import pickle
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'serif', 'size': 12}
font = {'family': 'serif', 'size': 18}
bigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import netCDF4 as nc
import sys
import os
from os import mkdir
from os.path import join,exists
codedir = "/home/jf4241/ecmwf/era20c"
os.chdir(codedir)
datadir = "/scratch/jf4241/ecmwf_data/era20c_data/2021-11-03"
if not exists(datadir): mkdir(datadir)
ftdatadir = "/scratch/jf4241/ecmwf_data/era20c_data/featurized_data/2021-11-19"
if not exists(ftdatadir): mkdir(ftdatadir)
resultsdir = "/scratch/jf4241/ecmwf_data/era20c_results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2021-11-19")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"0")
if not exists(expdir): mkdir(expdir)
import helper
import tpt_era20c

wstart_min = 0.0
wend_max = 150.0
# Parameters to determine what to do
create_features_flag =         0
featurize_data_flag =          0
plot_flag =                    1

# Initialize generic tpt
fall_year_list = np.arange(1900,2007)
if create_features_flag:
    fall_year_list_subset = np.random.choice(np.arange(len(fall_year_list)),size=min(30,len(fall_year_list)),replace=False)
    tpt = tpt_era20c.TPTRA(datadir,ftdatadir,fall_year_list,wstart_min,wend_max)
    tpt.create_structured_features(flist=[tpt.data_file_list[i] for i in fall_year_list_subset])
    pickle.dump(tpt,open(join(expdir,"tpt"),"wb"))
tpt = pickle.load(open(join(expdir,"tpt"),"rb"))
if featurize_data_flag:
    tpt.featurize_database()
    pickle.dump(tpt,open(join(expdir,"tpt"),"wb"))
    tpt = pickle.load(open(join(expdir,"tpt"),"rb"))

# Now plot some 1d timeseries
fall_year = 2003
i_year = np.where(fall_year_list == fall_year)[0][0]
# To get the time range, bracket the 30-day period with maximum deceleration
tpt.plot_vortex_evolution(i_year,num_snapshots=30)

sys.exit()

uthresh_list = np.array([5,0,-5,-10,-15,-20,-25], dtype=float)
rate_list = np.zeros(len(uthresh_list))
for ui in range(len(uthresh_list)):
    wstart = wstart_min
    wend = wend_max
    # Physical parameters 
    uthresh = uthresh_list[ui]
    savedir = join(expdir,("wint{}-{}_uthresh{}".format(wstart,wend,uthresh)).replace(".","p"))
    if not exists(savedir): mkdir(savedir)
    print("savedir = {}".format(savedir))
    # Specialize TPT
    tpt.set_physical_params(savedir,uthresh=uthresh,wstart=wstart,wend=wend)
    #tpt.featurize_database() # Collects all the features into a giant numpy array
    
    obs_keys_list = [["time","uref"], ["time","mag1"], ["time","mag2"], ["mag1","mag2"]]
    for oki in range(len(obs_keys_list)):
        obs_keys = obs_keys_list[oki]
        fig,ax = tpt.plot_forward_committor_2d(obs_keys)
        fig.savefig(join(savedir,"qp_%s_%s"%(obs_keys[0],obs_keys[1])))
        plt.close(fig)

    rate_list[ui] = tpt.compute_rate()

# Save the rate list
rate_dict = {"uthresh_list": uthresh_list, "rate_list": rate_list}
pickle.dump(rate_dict,open(join(expdir,"rate"),"wb"))
#np.save(join(expdir,("rate_era20c_wint{}-{}".format(wstart,wend)).replace(".","p")),rate_list)
#np.save(join(expdir,("uthresh_era20c_wint{}-{}".format(wstart,wend)).replace(".","p")),uthresh_list)
print("rate list = {}".format(rate_list))
# Plot the threshold list
