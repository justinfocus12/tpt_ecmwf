import pickle
import pandas
import numpy as np
import datetime
import time as timelib
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
datadirs = dict({
    "ei": "/scratch/jf4241/ecmwf_data/eraint_data/2022-02-10",
    "e2": "/scratch/jf4241/ecmwf_data/era20c_data/2022-02-10",
    "s2s": "/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23",
    })
sources = list(datadirs.keys())
featdir = "/scratch/jf4241/ecmwf_data/features/2022-02-27"
if not exists(featdir): mkdir(featdir)
feat_display_dir = join(featdir,"display1")
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2022-02-27")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"0")
if not exists(expdir): mkdir(expdir)
import helper
import strat_feat
import tpt_general

# COMPLETE listing of possible years to use for each dataset
fall_years = dict({
    "e2": np.arange(1900,2008),
    "ei": np.arange(1979,2018),
    "s2s": np.arange(1996,2017),
    })
# Listing of years that we will consider in DGA (must be a subset of the above)
subsets = dict({
    "e2": dict({"full_subset": fall_years["e2"]}),
    "ei": dict({"full_subset": fall_years["s2s"]}),
    "s2s": dict({"full_subset": fall_years["s2s"]}),
    }) 

file_lists = dict()
for key in ["e2","ei"]:
    file_lists[key] = [join(datadirs[key],"%s-10-01_to_%s-04-30.nc"%(fall_year,fall_year+1)) for fall_year in fall_years[key]]

# Build the s2s database, in an orderly fashion
s2sfiles = [join(datadirs["s2s"],f) for f in os.listdir(datadirs["s2s"]) if f.endswith(".nc")]
ref_date = datetime.date(1900,1,1)
time_s2s = np.zeros(len(s2sfiles))
for i,f in enumerate(s2sfiles):
    # template: hcYYYY-MM-DD_rtYYYY-MM-DD.nc
    base = os.path.basename(f)
    date_f = datetime.date(int(base[2:6]),int(base[7:9]),int(base[10:12]))
    time_s2s[i] = (date_f - ref_date).days
order = np.argsort(time_s2s)
file_lists["s2s"] = []
for i in range(len(s2sfiles)):
    file_lists["s2s"].append(s2sfiles[order[i]])

print(f"file_lists['s2s'][:10] = {file_lists['s2s'][:10]}")

# Which files to use to construct the climate average 
file_list_climavg = file_lists["ei"][:15]

# ------------ Subsetting for robustness tests ------------
# Two purposes here: error bars, and climate change signal. How should we mix them? 
# Method 1: divide years into disjoint subsets
#interval_length_lists = dict({
#    "e2": [53,107],
#    "ei": [20,40],
#    "s2s": [10,21],
#    })
#linestyle_lists = dict({
#    "e2": ['dotted','dashed','solid'],
#    "ei": ['dotted','dashed','solid'],
#    "s2s": ['dotted','dashed','solid'],
#    })
#subset_lists = dict()
#for key in ["e2","ei","s2s"]:
#    subset_lists[key] = []
#    for interval_length in interval_length_lists[key]:
#        starts = np.arange(fall_years[key][0],fall_years[key][-1],interval_length)
#        print(f"key = {key}, starts = {starts}")
#        for k0 in starts:
#            print(f"k0 = {k0}, k0 + interval_length = {k0 + interval_length}, fall_years[key][-1] = {fall_years[key][-1]}")
#            if k0 + interval_length - 1 <= fall_years[key][-1]:
#                subset_lists[key].append(np.arange(k0,k0+interval_length))
#    print(f"subset_lists[{key}] = {subset_lists[key]}")

# Method 2: resample with replacement
prng = np.random.RandomState(0) # This will be used for subsampling the years. 
subsets["num_bootstrap"] = 6
for key in sources:
    num_full_kmeans_seeds = 6 if key == "s2s" else 1
    subsets[key]["full_kmeans_seeds"] = np.arange(num_full_kmeans_seeds) # random-number generator seeds to use for KMeans. 
    Nyears = len(subsets[key]["full_subset"])
    subsets[key]["resampled_kmeans_seeds"] = num_full_kmeans_seeds + np.arange(subsets["num_bootstrap"])
    subsets[key]["resampled_subsets"] = np.zeros((subsets["num_bootstrap"],Nyears), dtype=int)
    for i_ss in range(subsets["num_bootstrap"]):
        subsets[key]["resampled_subsets"][i_ss] = prng.choice(subsets[key]["full_subset"], size=Nyears, replace=True)
    # Concatenate these
    subsets[key]["all_kmeans_seeds"] = np.concatenate((subsets[key]["full_kmeans_seeds"], subsets[key]["resampled_kmeans_seeds"]))
    subsets[key]["all_subsets"] = np.concatenate((
        np.outer(np.ones(num_full_kmeans_seeds, dtype=int),subsets[key]["full_subset"]),
        subsets[key]["resampled_subsets"]), axis=0)

print(f"s2s subsets: \n{subsets['s2s']['all_subsets']}")

# ----------------- Constant parameters ---------------------
winter_day0 = 0.0
spring_day0 = 180.0
Npc_per_level_max = 15
num_vortex_moments_max = 4 # Area, mean, variance, skewness, kurtosis. But it's too expensive. At least we need a linear approximation. 
heatflux_wavenumbers_per_level_max = 3 # 0: nothing. 1: zonal mean. 2: wave 1. 3: wave 2. 
# ----------------- Phase space definition parameters -------
delaytime_days = 20.0 # Both zonal wind and heat flux will be saved with this time delay. Must be shorter than tthresh0
# ----------------- Directories for this experiment --------
print(f"expdir = {expdir}, sources = {sources}")
expdirs = dict({key: join(expdir,key) for key in sources})
print(f"expdirs = {expdirs}")
for key in sources:
    if not exists(expdirs[key]): mkdir(expdirs[key])
# ------------------ Algorithmic parameters ---------------------
multiprocessing_flag = 0
num_clusters = 170
#Npc_per_level_single = 4
Npc_per_level = np.array([0,0,0,0,0,0,0,0,0,0]) #Npc_per_level_single*np.ones(len(feat_def["plev"]), dtype=int)  
captemp_flag = np.array([0,0,0,0,0,0,0,0,0,0], dtype=bool)
heatflux_wavenumbers = np.array([0,0,0,0,0,0,0,0,0,0], dtype=int)
num_vortex_moments = 0 # must be <= num_vortex_moments_max
pcstr = ""
hfstr = ""
tempstr = ""
for i_lev in range(len(Npc_per_level)):
    if Npc_per_level[i_lev] != 0:
        pcstr += f"lev{i_lev}pc{Npc_per_level[i_lev]}-"
    if heatflux_wavenumbers[i_lev] != 0:
        hfstr += f"lev{i_lev}wn{heatflux_wavenumbers[i_lev]}-"
    if captemp_flag[i_lev]:
        tempstr += f"{i_lev}-"
if len(pcstr) > 1: pcstr = pcstr[:-1]
Nwaves = 0
# Make a dictionary for all these parameters
algo_params = {"Nwaves": Nwaves, "Npc_per_level": Npc_per_level, "captemp_flag": captemp_flag, "heatflux_wavenumbers": heatflux_wavenumbers, "num_vortex_moments": num_vortex_moments}
fidx_X_filename = join(expdir,"fidx_X")
fidx_Y_filename = join(expdir,"fidx_Y")
paramdirs = dict({
    "s2s": join(expdir, "s2s", f"delay{int(delaytime_days)}_nwaves{Nwaves}_vxm{num_vortex_moments}_pc-{pcstr}_hf-{hfstr}_temp-{tempstr}_nclust{num_clusters}"),
    "e2": join(expdir, "e2", f"delay{int(delaytime_days)}_nwaves{Nwaves}_vxm{num_vortex_moments}_pc-{pcstr}_hf-{hfstr}_temp-{tempstr}"),
    "ei": join(expdir, "ei", f"delay{int(delaytime_days)}_nwaves{Nwaves}_vxm{num_vortex_moments}_pc-{pcstr}_hf-{hfstr}_temp-{tempstr}"),
    #"e2": join(expdir, "e2", f"delay{int(delaytime_days)}"),
    #"ei": join(expdir, "ei", f"delay{int(delaytime_days)}"),
    })
for key in sources:
    if not exists(paramdirs[key]): mkdir(paramdirs[key])


# Debugging: turn off each reanalysis individually
e2_flag = True
ei_flag = True

#subsetdirs = dict({key: [join(paramdirs[key],"%i-%i"%(subset[0],subset[-1]+1)) for subset in subset_lists[key]] for key in sources})
for key in sources:
    subsets[key]["full_dirs"] = [join(paramdirs[key],"full_seed%i"%(seed)) for seed in subsets[key]["full_kmeans_seeds"]]
    subsets[key]["resampled_dirs"] = [join(paramdirs[key],"resampled_%i"%(i_ss)) for i_ss in range(subsets["num_bootstrap"])]
    subsets[key]["all_dirs"] = np.concatenate((subsets[key]["full_dirs"],subsets[key]["resampled_dirs"]))

    

# Parameters to determine what to do
# Featurization
create_features_flag =         0
display_features_flag =        0
# era20c
evaluate_database_e2 =         0
tpt_featurize_e2 =             0
tpt_e2_flag =                  0
# eraint
evaluate_database_ei =         0
tpt_featurize_ei =             0
tpt_ei_flag =                  0
# s2s
evaluate_database_s2s =        0
tpt_featurize_s2s =            0
cluster_flag =                 0
build_msm_flag =               0
tpt_s2s_flag =                 0
transfer_results_flag =        0
plot_tpt_results_s2s_flag =    0
# Summary statistic
plot_rate_flag =               1
illustrate_dataset_flag =      0


feature_file = join(featdir,"feat_def")
winstrat = strat_feat.WinterStratosphereFeatures(feature_file,winter_day0,spring_day0,delaytime_days=delaytime_days,Npc_per_level_max=Npc_per_level_max,num_vortex_moments_max=num_vortex_moments_max,heatflux_wavenumbers_per_level_max=heatflux_wavenumbers_per_level_max)

if create_features_flag:
    print("Creating features")
    winstrat.create_features(file_list_climavg, multiprocessing_flag=multiprocessing_flag)
# ------------------ Initialize the TPT object -------------------------------------
feat_def = pickle.load(open(winstrat.feature_file,"rb"))
print(f"plev = {feat_def['plev']/100} hPa")
winstrat.set_feature_indices_X(feat_def,fidx_X_filename)
winstrat.set_feature_indices_Y(feat_def,fidx_Y_filename,algo_params)
tpt = tpt_general.WinterStratosphereTPT()
# ----------------- Display features ------------------------------------------
if display_features_flag:
    # Show characteristics of the basis functions, e.g., EOFs and spectrum
    print("Showing EOFs")
    winstrat.show_multiple_eofs(feat_display_dir)
    # Show the basis functions evaluated on various samples
    for display_idx in np.arange(1983,1985)-fall_years["ei"][0]:
        winstrat.plot_vortex_evolution(file_lists["ei"][display_idx],feat_display_dir,"fy{}".format(fall_years["ei"][display_idx]))

# ----------------- Determine list of SSW definitions to consider --------------
tthresh0 = 31 # First day that SSW could happen
tthresh1 = 31 + 30 + 31 + 31 + 28  # Last day that SSW could happen: end of February or March
sswbuffer = 0.0 # minimum buffer time between one SSW and the next
uthresh_a = 100.0 # vortex is in state A if it exceeds uthresh_a and it's been sswbuffer days since being inside B
uthresh_list = np.arange(5,-26,-5) #np.array([5.0,0.0,-5.0,-10.0,-15.0,-20.0])

# =============================================================
if ei_flag:
    # ------------------ TPT direct estimates from ERA-Interim --------------------------
    feat_filename = join(expdirs["ei"],"X.npy")
    ens_start_filename = join(expdirs["ei"],"ens_start_idx.npy")
    fall_year_filename = join(expdirs["ei"],"fall_year_list.npy")
    if evaluate_database_ei:
        print("Evaluating ERA-Interim database")
        eval_start = timelib.time()
        if multiprocessing_flag:
            winstrat.evaluate_features_database_parallel(file_lists["ei"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        else:
            winstrat.evaluate_features_database(file_lists["ei"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        eval_dur = timelib.time() - eval_start
        print(f"eval_dur = {eval_dur}")
    if tpt_ei_flag: 
        print("Starting TPT on eraint")
        #for i_subset,subset in enumerate(subsets["ei"]):
        for i_subset,subset in enumerate(subsets["ei"]["all_subsets"]):
            subsetdir = subsets["ei"]["all_dirs"][i_subset]
            print(f"subsetdir = {subsetdir}")
            if not exists(subsetdir): mkdir(subsetdir)
            tpt_feat_filename = join(subsetdir,"Y")
            if tpt_featurize_ei:
                winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=True,fy_resamp=subsets["ei"]["all_subsets"][i_subset])
            rates_ei = np.zeros(len(uthresh_list))
            for i_uth in range(len(uthresh_list)):
                uthresh_b = uthresh_list[i_uth]
                savedir = join(subsetdir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
                if not exists(savedir): mkdir(savedir)
                tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
                tpt.set_boundaries(tpt_bndy)
                summary_dns = tpt.tpt_pipeline_dns(tpt_feat_filename,savedir,winstrat,feat_def,algo_params,plot_field_flag=False)
                rates_ei[i_uth] = summary_dns["rate"]
# =============================================================================
# =============================================================================
if e2_flag:
    # ------------------ TPT direct estimates from ERA20C --------------------------
    feat_filename = join(expdirs["e2"],"X.npy")
    ens_start_filename = join(expdirs["e2"],"ens_start_idx.npy")
    fall_year_filename = join(expdirs["e2"],"fall_year_list.npy")
    if evaluate_database_e2:
        print("Evaluating ERA-20C database")
        eval_start = timelib.time()
        if multiprocessing_flag:
            winstrat.evaluate_features_database_parallel(file_lists["e2"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        else:
            winstrat.evaluate_features_database(file_lists["e2"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        eval_dur = timelib.time() - eval_start
        print(f"eval_dur = {eval_dur}")
    if tpt_e2_flag: 
        print("Starting TPT on era20c")
        for i_subset,subset in enumerate(subsets["e2"]["all_subsets"]):
            subsetdir = subsets["e2"]["all_dirs"][i_subset]
            print(f"\tsubsetdir = {subsetdir}")
            if not exists(subsetdir): mkdir(subsetdir)
            tpt_feat_filename = join(subsetdir,"Y")
            if tpt_featurize_e2:
                winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=True,fy_resamp=subsets["e2"]["all_subsets"][i_subset])
            rates_e2 = np.zeros(len(uthresh_list))
            for i_uth in range(len(uthresh_list)):
                uthresh_b = uthresh_list[i_uth]
                savedir = join(subsetdir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
                if not exists(savedir): mkdir(savedir)
                tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
                tpt.set_boundaries(tpt_bndy)
                summary_dns = tpt.tpt_pipeline_dns(tpt_feat_filename,savedir,winstrat,feat_def,algo_params,plot_field_flag=False)
                rates_e2[i_uth] = summary_dns["rate"]
# ================================================================================
# =======================================================================
# ------------------- DGA from S2S --------------------------------
feat_filename = join(expdirs["s2s"],"X.npy")
feat_filename_ra_dict = dict({key: join(expdirs[key],"X.npy") for key in ["ei","e2"]})
tpt_feat_filename_ra_dict = dict({key: join(subsets[key]["all_dirs"][0],"Y") for key in ["ei","e2"]})
#feat_filename_ra = join(expdirs["e2"],"X.npy")
#tpt_feat_filename_ra = join(subsetdirs["e2"][-1],"Y")
ens_start_filename = join(expdirs["s2s"],"ens_start_idx.npy")
fall_year_filename = join(expdirs["s2s"],"fall_year_list.npy")
if evaluate_database_s2s: # Expensive!
    print("Evaluating S2S database")
    eval_start = timelib.time()
    if multiprocessing_flag:
        winstrat.evaluate_features_database_parallel(file_lists["s2s"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
    else:
        winstrat.evaluate_features_database(file_lists["s2s"],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
    eval_dur = timelib.time() - eval_start
    print(f"eval_dur = {eval_dur}")
print("Starting TPT on S2S")
for i_subset,subset in enumerate(subsets["s2s"]["all_subsets"]):
    subsetdir = subsets["s2s"]["all_dirs"][i_subset]
    if not exists(subsetdir): mkdir(subsetdir)
    tpt_feat_filename = join(subsetdir,"Y")
    clust_filename = join(subsetdir,"kmeans")
    msm_filename = join(subsetdir,"msm")
    if tpt_featurize_s2s:
        winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=True,fy_resamp=subsets["s2s"]["all_subsets"][i_subset])
    if cluster_flag:
        tpt.cluster_features(tpt_feat_filename,clust_filename,winstrat,num_clusters=num_clusters,seed=subsets["s2s"]["all_kmeans_seeds"][i_subset])  # In the future, maybe cluster A and B separately, which has to be done at each threshold
    if build_msm_flag:
        tpt.build_msm(tpt_feat_filename,clust_filename,msm_filename,winstrat)
    if tpt_s2s_flag:
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            summary_dga = tpt.tpt_pipeline_dga(tpt_feat_filename,clust_filename,msm_filename,feat_def,savedir,winstrat,algo_params)
    if transfer_results_flag:
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            for key_ra in tpt_feat_filename_ra_dict.keys():
                tpt.transfer_tpt_results(tpt_feat_filename,clust_filename,feat_def,savedir,winstrat,algo_params,tpt_feat_filename_ra_dict[key_ra],key_ra)
    if plot_tpt_results_s2s_flag and (i_subset == 0): #len(subset) == max(interval_length_lists["s2s"]):
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            tpt.plot_results_data(feat_filename,tpt_feat_filename,feat_filename_ra_dict,tpt_feat_filename_ra_dict,feat_def,savedir,winstrat,algo_params,
                    spaghetti_flag=0*(i_uth==1 or i_uth==4),
                    fluxdens_flag=1*(i_uth==1 or i_uth==4),
                    verify_leadtime_flag=0*(i_uth==1 or i_uth==4),
                    current2d_flag=0*(i_uth==1 or i_uth==4),
                    comm_corr_flag=0*(i_uth==1 or i_uth==4),
                    )
            #tpt.plot_results_clust(feat_def,savedir,winstrat,algo_params)

# =============================================================================

# ------------- Compare rates ---------------------
if plot_rate_flag:
    rate_lists = dict({key: np.zeros((len(subsets[key]["all_subsets"]),len(uthresh_list))) for key in sources})
    rate_lists["s2s_naive"] = np.zeros((len(subsets["s2s"]["all_subsets"]),len(uthresh_list)))
    for i_uth in range(len(uthresh_list)):
        uthresh_b = uthresh_list[i_uth]
        for key in ["ei","e2","s2s"]:
            for i_subset,subset in enumerate(subsets[key]["all_subsets"]):
                savedir = join(subsets[key]["all_dirs"][i_subset],"tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer))
                summary = pickle.load(open(join(savedir,"summary"),"rb"))
                if key in ["e2","ei"]:
                    rate_lists[key][i_subset,i_uth] = summary["rate"]
                elif key == "s2s":
                    rate_lists[key][i_subset,i_uth] = summary["rate_tob"]
                    rate_lists["s2s_naive"][i_subset,i_uth] = summary["rate_naive"]
    # ------------ Bar plot -------------------
    colors = dict({"ei": "black", "e2": "dodgerblue", "s2s": "red"}) #, "s2s_naive": "cyan"})
    labels = dict({"ei": "ERA-Interim", "e2": "ERA-20C", "s2s": "S2S"}) #, "s2s_naive": "S2S unweighted"})
    df = pandas.DataFrame({
        "uthresh": uthresh_list,
        })
    for key in sources:
        df[key] = rate_lists[key][np.arange(len(subsets[key]["full_kmeans_seeds"]))].mean(axis=0)
        df[key+"_min"] = rate_lists[key].min(axis=0)
        df[key+"_max"] = rate_lists[key].max(axis=0)
    df.sort_values(by=["uthresh"],inplace=True)
    print(f"df = \n{df}")
    fig,ax = plt.subplots()
    yerrlo = np.array([df[key]-df[key+'_min'] for key in sources]) # 3xN
    yerrhi = np.array([df[key+'_max']-df[key] for key in sources]) # 3xN
    yerr = np.array([yerrlo,yerrhi]) #2x3xN
    yerr = np.transpose(yerr,(1,0,2)) #3x2xN
    # Build it up from ei to e2 to s2s
    df.plot(x="uthresh", xlabel="Zonal wind threshold", ylabel="SSW Rate", y=[key for key in sources], color=colors, kind="bar", ax=ax, rot=0, width=0.75, yerr=yerr, error_kw={"ecolor": "silver", "capsize": 2, "lw": 2, "capthick": 2})
    ax.legend([labels[key] for key in sources])
    fig.savefig(join(join(paramdirs["s2s"],"rate_bar")))
    plt.close(fig)
    # ------------ Line plot -------------------
    ylim = {'log': [1e-3,1.0], 'linear': [0.0,1.0]}
    loc = {'log': 'lower right', 'linear': 'upper left'}
    bndy_suffix = "tth%i-%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_a,sswbuffer)
    # Build this up one curve at a time
    label_needed = dict({key: True for key in colors.keys()})
    du = np.abs(uthresh_list[1] - uthresh_list[0])/10.0 # How far to offset the x axis positions for the three timeseries
    errorbar_offsets = dict({"ei": -du, "e2": du, "s2s": 0, "s2s_naive": du})
    for scale in ['linear','log']:
        fig,ax = plt.subplots()
        savefig_suffix = ""
        ax.set_xlabel("Zonal wind threshold",fontdict=font)
        ax.set_ylabel("Rate",fontdict=font)
        handles = []
        for key in ['ei','e2','s2s']:
            print(f"Starting to plot rate list for {key}")
            # ---------- Plot a single line with error bars ---------
            full_rate = rate_lists[key][np.arange(len(subsets[key]["full_kmeans_seeds"]))]
            full_rate_mean = full_rate.mean(axis=0)
            good_idx = np.where(full_rate_mean > 0)[0] if scale == 'log' else np.arange(len(uthresh_list))
            h = ax.scatter(uthresh_list[good_idx]+errorbar_offsets[key],full_rate_mean[good_idx],color=colors[key],linewidth=2,marker='o',linestyle='-',label=labels[key],alpha=1.0,s=36)
            handles += [h]
            for i_uth,uth in enumerate(uthresh_list[good_idx]):
                #ax.plot((uth+errorbar_offsets[key])*np.ones(2), np.array([full_rate[:,i_uth].min(),full_rate[:,i_uth].max()]), color=colors[key], linewidth=2.5)
                ax.plot((uth+errorbar_offsets[key])*np.ones(2), np.array([rate_lists[key][:,i_uth].min(), rate_lists[key][:,i_uth].max()]), color=colors[key], linewidth=2)
            savefig_suffix += key
            ax.legend(handles=handles,loc=loc[scale])
            ax.set_ylim(ylim[scale])
            ax.set_yscale(scale)
            fig.savefig(join(paramdirs["s2s"],"rate_%s_%s_%s"%(bndy_suffix,savefig_suffix,scale)))
            plt.close(fig)


if illustrate_dataset_flag:
    feat_filename_ra = join(expdirs["ei"],"X.npy")
    feat_filename_hc = join(expdirs["s2s"],"X.npy")
    ens_start_filename_ra = join(expdirs["ei"],"ens_start_idx.npy")
    ens_start_filename_hc = join(expdirs["s2s"],"ens_start_idx.npy")
    fall_year_filename_ra = join(expdirs["ei"],"fall_year_list.npy")
    fall_year_filename_hc = join(expdirs["s2s"],"fall_year_list.npy")
    tpt_feat_filename_ra = join(subsets["ei"]["full_dirs"][0],"Y")
    tpt_feat_filename_hc = join(subsets["s2s"]["full_dirs"][0],"Y")
    tthresh = np.array([tthresh0,tthresh1])*24.0
    winstrat.illustrate_dataset(
            uthresh_a,uthresh_list[[1,3,4]],tthresh,sswbuffer,
            file_lists["e2"],file_lists["s2s"],
            feat_filename_ra,feat_filename_hc,
            tpt_feat_filename_ra,tpt_feat_filename_hc,
            ens_start_filename_ra,ens_start_filename_hc,
            fall_year_filename_ra,fall_year_filename_hc,
            feat_def,feat_display_dir
            )
