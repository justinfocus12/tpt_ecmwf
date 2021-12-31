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
datadir_ei = "/scratch/jf4241/ecmwf_data/eraint_data/2021-12-12"
datadir_s2s = "/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23"
featdir = "/scratch/jf4241/ecmwf_data/features/2021-12-30"
if not exists(featdir): mkdir(featdir)
feat_display_dir = join(featdir,"display0")
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2021-12-31")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"0")
if not exists(expdir): mkdir(expdir)
expdir_e2 = join(expdir,"era20c")
if not exists(expdir_e2): mkdir(expdir_e2)
expdir_ei = join(expdir,"eraint")
if not exists(expdir_ei): mkdir(expdir_ei)
expdir_s2s = join(expdir,"s2s")
if not exists(expdir_s2s): mkdir(expdir_s2s)
import helper
import strat_feat
import tpt_general

# Which years to use for each dataset
fall_years_e2 = np.arange(1900,2007)
fall_years_ei = np.arange(1996,2017)
fall_years_s2s = np.arange(1996,2017)

# Specify the data files
file_list_e2 = []
for i_fy in range(len(fall_years_e2)):
    file_list_e2 += [join(datadir_e2,"%s-11-01_to_%s-04-30.nc"%(fall_years_e2[i_fy],fall_years_e2[i_fy]+1))]
file_list_ei = []
for i_fy in range(len(fall_years_ei)):
    file_list_ei += [join(datadir_ei,"%s-11-01_to_%s-04-30.nc"%(fall_years_ei[i_fy],fall_years_ei[i_fy]+1))]
file_list_s2s = [join(datadir_s2s,f) for f in os.listdir(datadir_s2s) if f.endswith(".nc")]
np.random.seed(1)
ftidx_e2 = np.random.choice(np.arange(len(file_list_e2)),size=15,replace=False)
dga_idx_s2s = np.random.choice(np.arange(len(file_list_s2s)),size=200,replace=False) # Subset of filed to use for DGA.

# ----------------- Constant parameters ---------------------
winter_day0 = 0.0
spring_day0 = 150.0
Npc_per_level_max = 6
# ------------------ Algorithmic parameters ---------------------
delaytime_days = 0.0 
num_clusters = 100
#Npc_per_level_single = 4
Npc_per_level = np.array([4,4,0,0,0,0,0,0,0,0]) #Npc_per_level_single*np.ones(len(feat_def["plev"]), dtype=int)  
pcstr = ""
for i_lev in range(len(Npc_per_level)):
    if Npc_per_level[i_lev] != 0:
        pcstr += f"lev{i_lev}pc{Npc_per_level[i_lev]}-"
if len(pcstr) > 1: pcstr = pcstr[:-1]
Nwaves = 0
paramdir_s2s = join(expdir_s2s, f"nclust{num_clusters}_nwaves{Nwaves}_delay{int(delaytime_days)}_{pcstr}")
if not exists(paramdir_s2s):
    mkdir(paramdir_s2s)

# ------------ Random seeds for bootstrap resampling ------------
num_seeds_e2 =  1
num_seeds_ei =  1
num_seeds_s2s = 1
seeddir_list_e2 = []
for i in range(num_seeds_e2):
    seeddir_list_e2 += [join(expdir_e2,"seed%i"%(i))]
seeddir_list_ei = []
for i in range(num_seeds_ei):
    seeddir_list_ei += [join(expdir_ei,"seed%i"%(i))]
seeddir_list_s2s = []
for i in range(num_seeds_s2s):
    seeddir_list_s2s += [join(paramdir_s2s,"seed%i"%(i))]

# Parameters to determine what to do
# Featurization
create_features_flag =         0
display_features_flag =        0
# era20c
evaluate_database_e2 =         0
tpt_featurize_e2 =             1
tpt_e2_flag =                  1
# eraint
evaluate_database_ei =         0
tpt_featurize_ei =             1
tpt_ei_flag =                  1
# s2s
evaluate_database_s2s =        0
tpt_featurize_s2s =            1
cluster_flag =                 1
build_msm_flag =               1
tpt_s2s_flag =                 1
# Summary statistics
plot_rate_flag =               1


feature_file = join(featdir,"feat_def")
winstrat = strat_feat.WinterStratosphereFeatures(feature_file,winter_day0,spring_day0,delaytime_days=delaytime_days,Npc_per_level_max=Npc_per_level_max)

if create_features_flag:
    print("Creating features")
    winstrat.create_features([file_list_e2[i_ft] for i_ft in ftidx_e2])
if display_features_flag:
    # Show characteristics of the basis functions, e.g., EOFs and spectrum
    print("Showing EOFs")
    winstrat.show_multiple_eofs(feat_display_dir)
    # Show the basis functions evaluated on various samples
    for display_idx in np.arange(96,98):
        winstrat.plot_vortex_evolution(file_list_e2[display_idx],feat_display_dir,"fy{}".format(fall_years_e2[display_idx]))

# ------------------ Initialize the TPT object -------------------------------------
feat_def = pickle.load(open(winstrat.feature_file,"rb"))
tpt = tpt_general.WinterStratosphereTPT()
# ----------------- Determine list of SSW definitions to consider --------------
tthresh0 = delaytime_days + 10 # First day that SSW could happen
tthresh1 = 145.0 # Last day that SSW could happen
sswbuffer = 0.0 # minimum buffer time between one SSW and the next
uthresh_a = 100.0 # vortex is in state A if it exceeds uthresh_a and it's been sswbuffer days since being inside B
uthresh_list = np.arange(5,-26,-15) #np.array([5.0,0.0,-5.0,-10.0,-15.0,-20.0])
# ------------------ TPT direct estimates from ERA20C --------------------------
feat_filename = join(expdir_e2,"X.npy")
ens_start_filename = join(expdir_e2,"ens_start_idx.npy")
fall_year_filename = join(expdir_e2,"fall_year_list.npy")
if evaluate_database_e2:
    print("Evaluating ERA20C database")
    winstrat.evaluate_features_database(file_list_e2,feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
if tpt_e2_flag: 
    print("Starting TPT on era20c")
    for i_seed in range(num_seeds_e2):
        print("\tSeed {} of {}".format(i_seed,num_seeds_e2))
        seeddir = seeddir_list_e2[i_seed]
        if not exists(seeddir): mkdir(seeddir)
        tpt_feat_filename = join(seeddir,"Y")
        if tpt_featurize_e2:
            winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,Npc_per_level=Npc_per_level,Nwaves=Nwaves,resample_flag=(i_seed>0),seed=i_seed)
        rates_e2 = np.zeros(len(uthresh_list))
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(seeddir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            summary_dns = tpt.tpt_pipeline_dns(tpt_feat_filename,savedir,winstrat,feat_def,Nwaves,Npc_per_level)
            rates_e2[i_uth] = summary_dns["rate"]
# ------------------ TPT direct estimates from ERA-Interim --------------------------
feat_filename = join(expdir_ei,"X.npy")
ens_start_filename = join(expdir_ei,"ens_start_idx.npy")
fall_year_filename = join(expdir_ei,"fall_year_list.npy")
if evaluate_database_ei:
    print("Evaluating ERA-Int database")
    winstrat.evaluate_features_database(file_list_ei,feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
if tpt_ei_flag: 
    print("Starting TPT on eraint")
    for i_seed in range(len(seeddir_list_ei)):
        print("\tSeed {} of {}".format(i_seed,num_seeds_ei))
        seeddir = seeddir_list_ei[i_seed]
        if not exists(seeddir): mkdir(seeddir)
        tpt_feat_filename = join(seeddir,"Y")
        if tpt_featurize_ei:
            winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,Npc_per_level=Npc_per_level,Nwaves=Nwaves,resample_flag=(i_seed>0),seed=i_seed)
        rates_ei = np.zeros(len(uthresh_list))
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(seeddir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            summary_dns = tpt.tpt_pipeline_dns(tpt_feat_filename,savedir,winstrat,feat_def,Nwaves,Npc_per_level)
            rates_ei[i_uth] = summary_dns["rate"]
#print(f"rates_e2 = {rates_e2}\nrates_ei = {rates_ei}")
# ------------------- DGA from S2S --------------------------------
#Npc_per_level = Npc_per_level_single*np.ones(len(feat_def["plev"]), dtype=int)  
#Npc_per_level[1:] = 0 # Only care about the top layer
feat_filename = join(expdir_s2s,"X.npy")
ens_start_filename = join(expdir_s2s,"ens_start_idx.npy")
fall_year_filename = join(expdir_s2s,"fall_year_list.npy")
if evaluate_database_s2s: # Expensive!
    print("Evaluating S2S database")
    winstrat.evaluate_features_database([file_list_s2s[i] for i in dga_idx_s2s],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])

print("Starting TPT on S2S")
for i_seed in np.arange(len(seeddir_list_s2s)):
    print("\tSeed {} of {}".format(i_seed,num_seeds_s2s))
    seeddir = seeddir_list_s2s[i_seed]
    if not exists(seeddir): mkdir(seeddir)
    tpt_feat_filename = join(seeddir,"Y")
    clust_filename = join(seeddir,"kmeans")
    msm_filename = join(seeddir,"msm")
    if tpt_featurize_s2s:
        winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,Npc_per_level=Npc_per_level,Nwaves=Nwaves,resample_flag=(i_seed>0),seed=i_seed)
    if cluster_flag:
        # TODO: work out cluster pre-transformation
        tpt.cluster_features(tpt_feat_filename,clust_filename,winstrat,num_clusters=num_clusters)  # In the future, maybe cluster A and B separately, which has to be done at each threshold
    if build_msm_flag:
        tpt.build_msm(tpt_feat_filename,clust_filename,msm_filename,winstrat)
    if tpt_s2s_flag:
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(seeddir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            summary_dga = tpt.tpt_pipeline_dga(tpt_feat_filename,clust_filename,msm_filename,feat_def,savedir,winstrat,Npc_per_level,Nwaves)

# ------------- Compare rates ---------------------
if plot_rate_flag:
    rate_list_e2 = np.zeros((num_seeds_e2,len(uthresh_list)))
    rate_list_ei = np.zeros((num_seeds_ei,len(uthresh_list)))
    rate_list_s2s = np.zeros((num_seeds_s2s,len(uthresh_list)))
    for i_uth in range(len(uthresh_list)):
        uthresh_b = uthresh_list[i_uth]
        for i_seed in range(num_seeds_e2):
            savedir = join(seeddir_list_e2[i_seed],"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            summary = pickle.load(open(join(savedir,"summary"),"rb"))
            rate_list_e2[i_seed,i_uth] = summary["rate"]
        for i_seed in range(num_seeds_ei):
            savedir = join(seeddir_list_ei[i_seed],"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            summary = pickle.load(open(join(savedir,"summary"),"rb"))
            rate_list_ei[i_seed,i_uth] = summary["rate"]
        for i_seed in range(num_seeds_s2s):
            savedir = join(seeddir_list_s2s[i_seed],"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            summary = pickle.load(open(join(savedir,"summary"),"rb"))
            rate_list_s2s[i_seed,i_uth] = summary["rate_tob"]
    fig,ax = plt.subplots()
    he2, = ax.plot(uthresh_list,rate_list_e2[0],color='cyan',linewidth=3,marker='o',label="ERA20C")
    hei, = ax.plot(uthresh_list,rate_list_ei[0],color='black',linewidth=3,marker='o',label="ERAI")
    hs2s, = ax.plot(uthresh_list,rate_list_s2s[0],color='red',linewidth=3,marker='o',label="S2S")
    if num_seeds_e2 > 1:
        ax.plot(uthresh_list,np.min(rate_list_e2[1:],axis=0),color='cyan',linestyle='--',alpha=0.5)
        ax.plot(uthresh_list,np.max(rate_list_e2[1:],axis=0),color='cyan',linestyle='--',alpha=0.5)
    if num_seeds_ei > 1:
        ax.plot(uthresh_list,np.min(rate_list_ei[1:],axis=0),color='black',linestyle='--',alpha=0.5)
        ax.plot(uthresh_list,np.max(rate_list_ei[1:],axis=0),color='black',linestyle='--',alpha=0.5)
    if num_seeds_s2s > 1:
        ax.plot(uthresh_list,np.min(rate_list_s2s[1:],axis=0),color='red',linestyle='--',alpha=0.5)
        ax.plot(uthresh_list,np.max(rate_list_s2s[1:],axis=0),color='red',linestyle='--',alpha=0.5)
    ax.set_xlabel("Zonal wind threshold",fontdict=font)
    ax.set_ylabel("Rate",fontdict=font)
    ax.legend(handles=[he2,hei,hs2s])
    fig.savefig(join(paramdir_s2s,"rate_tth%i-%i"%(tthresh0,tthresh1)))
    ax.set_yscale('log')
    fig.savefig(join(paramdir_s2s,"lograte_tth%i-%i"%(tthresh0,tthresh1)))
    ax.set_yscale('logit')
    fig.savefig(join(paramdir_s2s,"logitrate_tth%i-%i"%(tthresh0,tthresh1)))
    plt.close(fig)

