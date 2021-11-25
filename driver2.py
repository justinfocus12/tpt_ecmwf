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
codedir = "/home/jf4241/ecmwf/s2s"
os.chdir(codedir)
datadir_e2 = "/scratch/jf4241/ecmwf_data/era20c_data/2021-11-03"
datadir_s2s = "/scratch/jf4241/ecmwf_data/s2s_data/2021-11-01"
featdir = "/scratch/jf4241/ecmwf_data/features/2021-11-23"
if not exists(featdir): mkdir(featdir)
feat_display_dir = join(featdir,"display2")
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2021-11-24")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"0")
if not exists(expdir): mkdir(expdir)
expdir_e2 = join(expdir,"era20c")
if not exists(expdir_e2): mkdir(expdir_e2)
expdir_s2s = join(expdir,"s2s")
if not exists(expdir_s2s): mkdir(expdir_s2s)
import helper
import strat_feat
import tpt_general

np.random.seed(1)

# Which years to use for each dataset
fall_years_e2 = np.arange(1900,2007)
fall_years_s2s = np.arange(1996,2017)

# Specify the data files
file_list_e2 = []
for i_fy in range(len(fall_years_e2)):
    file_list_e2 += [join(datadir_e2,"%s-11-01_to_%s-04-30.nc"%(fall_years_e2[i_fy],fall_years_e2[i_fy]+1))]
# TODO: specify files for s2s
ftidx_e2 = np.random.choice(np.arange(len(file_list_e2)),size=15,replace=False)
print("ftidx_e2 = {}".format(ftidx_e2))

winter_day0 = 0.0
spring_day0 = 150.0
Npc_per_level_max = 6
# Parameters to determine what to do
create_features_flag =         0
display_features_flag =        0
evaluate_database_e2 =         0

feature_file = join(featdir,"feat_def")
winstrat = strat_feat.WinterStratosphereFeatures(feature_file,winter_day0,spring_day0,Npc_per_level_max)

if create_features_flag:
    print("Creating features")
    winstrat.create_features([file_list_e2[i_ft] for i_ft in ftidx_e2])
if display_features_flag:
    # Show characteristics of the basis functions, e.g., EOFs and spectrum
    print("Showing EOFs")
    winstrat.show_multiple_eofs(feat_display_dir)
    # Show the basis functions evaluated on various samples
    for display_idx in np.arange(96,106):
        winstrat.plot_vortex_evolution(file_list_e2[display_idx],feat_display_dir,"fy{}".format(fall_years_e2[display_idx]))

# ------------------ Initialize the TPT object -------------------------------------
feat_def = pickle.load(open(winstrat.feature_file,"rb"))
tpt = tpt_general.WinterStratosphereTPT()
# ------------------ TPT direct estimates from ERA20C ------------------------------
if evaluate_database_e2:
    winstrat.evaluate_features_database(file_list_e2,feat_def,expdir_e2,"X",winstrat.wtime[-1])
tthresh0 = 10.0 # First day that SSW could happen
tthresh1 = 120.0 # Last day that SSW could happen
uthresh = 0.0
savedir = join(expdir,"tth%i-%i_uth%i"%(tthresh0,tthresh1,uthresh))
if not exists(savedir): mkdir(savedir)
tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh": uthresh,}
tpt.set_boundaries(tpt_bndy)
tpt.tpt_pipeline_dns(expdir_e2,savedir,winstrat,feat_def)




# ------------------- TPT direct estimates from S2S --------------------------------





# -------------------- DGA from S2S ------------------------------------------------
