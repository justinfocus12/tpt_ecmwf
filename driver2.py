import pickle
import pandas
import numpy as np
import datetime
from calendar import monthrange
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
    "e5": "/scratch/jf4241/ecmwf_data/era5_data/2022-03-10",
    "s2s": "/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23",
    })
sources = list(datadirs.keys())
featdir = "/scratch/jf4241/ecmwf_data/features/2022-03-13"
if not exists(featdir): mkdir(featdir)
feat_display_dir = join(featdir,"display0")
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2022-03-16")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"2")
if not exists(expdir): mkdir(expdir)
import helper
import strat_feat
import tpt_general

# COMPLETE listing of possible years to use for each dataset
fall_years = dict({
    "ei": np.arange(1979,2018),
    "e2": np.arange(1900,2008),
    "e5": np.arange(1950,2020),
    "s2s": np.arange(1996,2017),
    })
# Fully overlapping estimates
intersection = np.intersect1d(fall_years["e2"],fall_years["e5"])
print(f"intersection = {intersection}")
subsets = dict({
    "ei": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["ei"],}),
            "ra": dict({"full": intersection,}),
            "hc": dict({"full": np.intersect1d(fall_years["ei"],fall_years["s2s"])}),
            }),
        "num_bootstrap": 30, 
        "num_full_kmeans_seeds": 1,
        "rank": 0,
        }),
    "e2": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["e2"], 
                "color": "dodgerblue", 
                "label": f"ERA-20C {fall_years['e2'][0]}-{fall_years['e2'][-1]}",
                }),
            "ra": dict({"full": intersection, 
                "color": "dodgerblue", 
                "label": f"ERA-20C {intersection[0]}-{intersection[-1]}"
                }),
            "hc": dict({"full": np.intersect1d(fall_years["e2"],fall_years["s2s"]), 
                "color": "dodgerblue", 
                "label": f"ERA-20C {max(fall_years['e2'][0],fall_years['s2s'][0])}-{min(fall_years['e2'][-1],fall_years['s2s'][-1])}"
                }),
            }),
        "num_bootstrap": 30, 
        "num_full_kmeans_seeds": 1,
        "rank": 1,
        }),
    "e5": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["e5"], 
                "color": "black", 
                "label": f"ERA5 {fall_years['e5'][0]}-{fall_years['e5'][-1]}"
                }),
            "ra": dict({"full": intersection, 
                "color": "black", 
                "label": f"ERA5 {intersection[0]}-{intersection[-1]}"
                }),
            "hc": dict({"full": np.intersect1d(fall_years["e5"],fall_years["s2s"]), 
                "color": "orange", 
                "label": f"ERA5 {max(fall_years['e5'][0],fall_years['s2s'][0])}-{min(fall_years['e5'][-1],fall_years['s2s'][-1])}"
                }),
            }),
        "num_bootstrap": 30, 
        "num_full_kmeans_seeds": 1,
        "rank": 2,
        }),
    "s2s": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["s2s"],
                "color": "red",
                "label": f"S2S {fall_years['s2s'][0]}-{fall_years['s2s'][-1]}"
                }),
            }),
        "num_bootstrap": 30, 
        "num_full_kmeans_seeds": 5,
        "rank": 3,
        }),
    })

# Resample with replacement
for src in sources:
    for ovl in subsets[src]["overlaps"].keys():
        Nyears = len(subsets[src]["overlaps"][ovl]['full'])
        subsets[src]["overlaps"][ovl]["bootstrap"] = []
        prng = np.random.RandomState(0)
        for i_ss in range(subsets[src]["num_bootstrap"]):
            subsets[src]["overlaps"][ovl]["bootstrap"] += [prng.choice(subsets[src]["overlaps"][ovl]["full"], size=Nyears, replace=True)]

print(f"era5 subsets: \nself:\n\t{subsets['e5']['overlaps']['self']}\nra:\n\t{subsets['e5']['overlaps']['ra']}\nhc:\n\t{subsets['e5']['overlaps']['hc']}")


file_lists = dict()
for key in ["e2","ei","e5"]: 
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

# ----------------- Constant parameters ---------------------
winter_day0 = 0.0
spring_day0 = 180.0
Npc_per_level_max = 15
num_vortex_moments_max = 4 # Area, mean, variance, skewness, kurtosis. But it's too expensive. At least we need a linear approximation. 
heatflux_wavenumbers_per_level_max = 3 # 0: nothing. 1: zonal mean. 2: wave 1. 3: wave 2. 
# ----------------- Phase space definition parameters -------
delaytime_days = 15.0 # Both zonal wind and heat flux will be saved with this time delay. Must be shorter than tthresh0
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
tpt_param_string = f"delay{int(delaytime_days)}_nwaves{Nwaves}_vxm{num_vortex_moments}_pc-{pcstr}_hf-{hfstr}_temp-{tempstr}"
paramdirs = dict({
    "s2s": join(expdirs["s2s"], f"{tpt_param_string}_nclust{num_clusters}"),
    "e2": join(expdirs["e2"], tpt_param_string),
    "ei": join(expdirs["ei"], tpt_param_string),
    "e5": join(expdirs["e5"], tpt_param_string),
    })
for key in sources:
    if not exists(paramdirs[key]): mkdir(paramdirs[key])


# Debugging: turn off each reanalysis individually
ra_flags = dict({
    "ei": True,
    "e2": True,
    "e5": True,
    })

# Create directories for each subset, as well as a running master list of directories and subsets to loop through later. 
for src in sources:
    subsets[src]["all_dirs"] = []
    subsets[src]["all_subsets"] = []
    subsets[src]["all_kmeans_seeds"] = []
    for ovl in subsets[src]["overlaps"].keys():
        # Full samples, varying KMeans seed
        subsets[src]["overlaps"][ovl]["full_dirs"] = [join(paramdirs[src],"overlap%s_full_seed%i"%(ovl,seed)) for seed in range(subsets[src]["num_full_kmeans_seeds"])]
        subsets[src]["all_dirs"] += subsets[src]["overlaps"][ovl]["full_dirs"]
        subsets[src]["all_subsets"] += [subsets[src]["overlaps"][ovl]["full"] for i_ss in range(subsets[src]["num_full_kmeans_seeds"])]
        subsets[src]["all_kmeans_seeds"] += [seed for seed in range(subsets[src]["num_full_kmeans_seeds"])]
        # Bootstrap samples
        subsets[src]["overlaps"][ovl]["bootstrap_dirs"] = [join(paramdirs[src],"overlap%s_bootstrap%i"%(ovl,i_bs)) for i_bs in range(subsets[src]["num_bootstrap"])]
        subsets[src]["all_dirs"] += subsets[src]["overlaps"][ovl]["bootstrap_dirs"]
        subsets[src]["all_subsets"] += subsets[src]["overlaps"][ovl]["bootstrap"]
        subsets[src]["all_kmeans_seeds"] += [subsets[src]["num_full_kmeans_seeds"]+seed for seed in range(subsets[src]["num_bootstrap"])]

print(f"S2S subsets:\n{subsets['s2s']['overlaps']['self']}")


# Parameters to determine what to do
task_list = dict({
    "featurization": dict({
        "create_features_flag":               0,
        "display_features_flag":              0,
        }),
    "ei": dict({
        "evaluate_database_flag":             0,
        "tpt_featurize_flag":                 0,
        "tpt_flag":                           0,
        }),
    "e2": dict({
        "evaluate_database_flag":             0,
        "tpt_featurize_flag":                 0, 
        "tpt_flag":                           0,
        }),
    "e5": dict({
        "evaluate_database_flag":             0,
        "tpt_featurize_flag":                 0, 
        "tpt_flag":                           0,
        }),
    "s2s": dict({
        "evaluate_database_flag":             0,
        "tpt_featurize_flag":                 0,
        "cluster_flag":                       0,
        "build_msm_flag":                     0,
        "tpt_s2s_flag":                       0,
        "transfer_results_flag":              0,
        "plot_tpt_results_flag":              0,
        }),
    "comparison": dict({
        "plot_rate_flag":                     0,
        "illustrate_dataset_flag":            1,
        }),
    })


feature_file = join(featdir,"feat_def")
winstrat = strat_feat.WinterStratosphereFeatures(feature_file,winter_day0,spring_day0,delaytime_days=delaytime_days,Npc_per_level_max=Npc_per_level_max,num_vortex_moments_max=num_vortex_moments_max,heatflux_wavenumbers_per_level_max=heatflux_wavenumbers_per_level_max)

if task_list["featurization"]["create_features_flag"]:
    print("Creating features")
    winstrat.create_features(file_list_climavg, multiprocessing_flag=multiprocessing_flag)
# ------------------ Initialize the TPT object -------------------------------------
feat_def = pickle.load(open(winstrat.feature_file,"rb"))
print(f"plev = {feat_def['plev']/100} hPa")
winstrat.set_feature_indices_X(feat_def,fidx_X_filename)
winstrat.set_feature_indices_Y(feat_def,fidx_Y_filename,algo_params)
tpt = tpt_general.WinterStratosphereTPT()
# ----------------- Display features ------------------------------------------
if task_list["featurization"]["display_features_flag"]:
    # Show characteristics of the basis functions, e.g., EOFs and spectrum
    print("Showing EOFs")
    winstrat.show_multiple_eofs(feat_display_dir)
    # Show the basis functions evaluated on various samples
    for display_idx in np.arange(1983,1985)-fall_years["ei"][0]:
        winstrat.plot_vortex_evolution(file_lists["ei"][display_idx],feat_display_dir,"fy{}".format(fall_years["ei"][display_idx]))

# ----------------- Determine list of SSW definitions to consider --------------
tthresh0 = monthrange(1901,10)[1] # First day that SSW could happen is Nov. 1
tthresh1 = sum([monthrange(1901,i)[1] for i in [10,11,12]]) + sum([monthrange(1902,i)[1] for i in [1,2]]) #31 + 30 + 31 + 31 + 28  # Last day that SSW could happen: February 28
sswbuffer = 0.0 # minimum buffer time between one SSW and the next
uthresh_a = 100.0 # vortex is in state A if it exceeds uthresh_a and it's been sswbuffer days since being inside B
uthresh_list = np.arange(0,-36,-5) #np.array([5.0,0.0,-5.0,-10.0,-15.0,-20.0])
plottable_uthresh_list = [0,-15]
uthresh_dirname_fun = lambda uthresh_b: "tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer)

# =============================================================
# TPT direct estimates from ERA-Interim, ERA-20C, and ERA5
for key in ["ei","e2","e5"]:
    feat_filename = join(expdirs[key],"X.npy")
    ens_start_filename = join(expdirs[key],"ens_start_idx.npy")
    fall_year_filename = join(expdirs[key],"fall_year_list.npy")
    if task_list[key]["evaluate_database_flag"]:
        print(f"Evaluating {key} database")
        eval_start = timelib.time()
        if multiprocessing_flag:
            winstrat.evaluate_features_database_parallel(file_lists[key],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        else:
            winstrat.evaluate_features_database(file_lists[key],feat_def,feat_filename,ens_start_filename,fall_year_filename,winstrat.wtime[0],winstrat.wtime[-1])
        eval_dur = timelib.time() - eval_start
        print(f"eval_dur = {eval_dur}")
    if task_list[key]["tpt_flag"]: 
        print(f"Starting TPT on {key}")
        for i_subset,subset in enumerate(subsets[key]["all_subsets"]):
            subsetdir = subsets[key]["all_dirs"][i_subset]
            print(f"subsetdir = {subsetdir}")
            if not exists(subsetdir): mkdir(subsetdir)
            tpt_feat_filename = join(subsetdir,"Y")
            if task_list[key]["tpt_featurize_flag"]:
                winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=True,fy_resamp=subsets[key]["all_subsets"][i_subset])
                if key == "e5":
                    print(f"subsetdir = {subsetdir}, fy_resamp = \n{subsets[key]['all_subsets'][i_subset]}")
            for i_uth in range(len(uthresh_list)):
                uthresh_b = uthresh_list[i_uth]
                savedir = join(subsetdir,uthresh_dirname_fun(uthresh_b))
                if not exists(savedir): mkdir(savedir)
                tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
                tpt.set_boundaries(tpt_bndy)
                _ = tpt.tpt_pipeline_dns(tpt_feat_filename,savedir,winstrat,feat_def,algo_params,plot_field_flag=False)
# =============================================================================

# =======================================================================
#                DGA from S2S 
feat_filename = join(expdirs["s2s"],"X.npy")
feat_filename_ra_dict = dict({key: join(expdirs[key],"X.npy") for key in ["ei","e2"]})
tpt_feat_filename_ra_dict = dict({key: join(subsets[key]["overlaps"]["self"]["full_dirs"][0],"Y") for key in ["ei","e2"]})
ens_start_filename = join(expdirs["s2s"],"ens_start_idx.npy")
fall_year_filename = join(expdirs["s2s"],"fall_year_list.npy")
if task_list["s2s"]["evaluate_database_flag"]: # Expensive!
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
    if task_list["s2s"]["tpt_featurize_flag"]:
        winstrat.evaluate_tpt_features(feat_filename,ens_start_filename,fall_year_filename,feat_def,tpt_feat_filename,algo_params,resample_flag=True,fy_resamp=subsets["s2s"]["all_subsets"][i_subset])
    if task_list["s2s"]["cluster_flag"]:
        tpt.cluster_features(tpt_feat_filename,clust_filename,winstrat,num_clusters=num_clusters,seed=subsets["s2s"]["all_kmeans_seeds"][i_subset])  # In the future, maybe cluster A and B separately, which has to be done at each threshold
    if task_list["s2s"]["build_msm_flag"]:
        tpt.build_msm(tpt_feat_filename,clust_filename,msm_filename,winstrat)
    if task_list["s2s"]["tpt_s2s_flag"]:
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,uthresh_dirname_fun(uthresh_b))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            summary_dga = tpt.tpt_pipeline_dga(tpt_feat_filename,clust_filename,msm_filename,feat_def,savedir,winstrat,algo_params)
    if task_list["s2s"]["transfer_results_flag"] and (i_subset == 0):
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,uthresh_dirname_fun(uthresh_b))
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            for key_ra in tpt_feat_filename_ra_dict.keys():
                tpt.transfer_tpt_results(tpt_feat_filename,clust_filename,feat_def,savedir,winstrat,algo_params,tpt_feat_filename_ra_dict[key_ra],key_ra)
    if task_list["s2s"]["plot_tpt_results_flag"] and (i_subset == 0):
        for i_uth in range(len(uthresh_list)):
            uthresh_b = uthresh_list[i_uth]
            savedir = join(subsetdir,uthresh_dirname_fun(uthresh_b))
            if not exists(savedir): mkdir(savedir)
            tpt_bndy = {"tthresh": np.array([tthresh0,tthresh1])*24.0, "uthresh_a": uthresh_a, "uthresh_b": uthresh_b, "sswbuffer": sswbuffer*24.0}
            tpt.set_boundaries(tpt_bndy)
            tpt.plot_results_data(feat_filename,tpt_feat_filename,feat_filename_ra_dict,tpt_feat_filename_ra_dict,feat_def,savedir,winstrat,algo_params,
                    spaghetti_flag=0*(uthresh_b in plottable_uthresh_list),
                    fluxdens_flag=1*(uthresh_b in plottable_uthresh_list),
                    verify_leadtime_flag=0*(uthresh_b in plottable_uthresh_list),
                    current2d_flag=1*(uthresh_b in plottable_uthresh_list),
                    comm_corr_flag=0*(uthresh_b in plottable_uthresh_list),
                    )

# =============================================================================

# ------------- Compare rates ---------------------
# 2 plots: 
#   1. S2S, ERA20C (full), ERA5 (full), ERA5 (overlap with S2S)
#   2. ERA20C (overlap with ERA5), ERA5 (overlap with ERA20C) 

# Plot 1: specify which overlaps to use from each source
boxplot_keys_hc = dict({
    "e2": ["self",],
    "e5": ["self","hc",],
    "s2s": ["self",],
    })
boxplot_keys_ra = dict({
    "e2": ["ra",],
    "e5": ["ra",],
    })

if task_list["comparison"]["plot_rate_flag"]:
    for boxplot_keys in [boxplot_keys_hc,boxplot_keys_ra]:
        ylim = {'log': [1e-3,1.0], 'logit': [1e-3,0.8], 'linear': [0.0,1.0]}
        loc = {'log': 'lower right', 'logit': 'lower right', 'linear': 'upper left'}
        bndy_suffix = "tth%i-%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_a,sswbuffer)
        du = np.abs(uthresh_list[1] - uthresh_list[0])/8.0 # How far to offset the x axis positions for the three timeseries
        errorbar_offsets = dict({"e5-hc": -3/2*du, "e5-self": -du/2, "e2-self": du/2, "s2s-self": 3*du/2,
            "e5-ra": -1/2*du, "e2-ra": 1/2*du,})
        quantiles = np.array([0.05,0.25,0.75,0.95])
        for scale in ['linear','log']:
            fig,ax = plt.subplots()
            savefig_suffix = ""
            ax.set_xlabel("Zonal wind threshold [m/s]",fontdict=font)
            ax.set_ylabel("Rate",fontdict=font)
            handles = []
            # Build the rate dictionary
            rate_dict = dict({})
            for src in boxplot_keys:
                for ovl in boxplot_keys[src]:
                    rate_dict[f"{src}-{ovl}"] = np.zeros((1+subsets[src]['num_bootstrap'], len(uthresh_list))) # First entry for mean of full kmeans, rest of them for bootstrap
                    # First row
                    for i_km,dir_km in enumerate(subsets[src]["overlaps"][ovl]["full_dirs"]):
                        for i_uth,uthresh_b in enumerate(uthresh_list):
                            savedir = join(dir_km,uthresh_dirname_fun(uthresh_b))
                            summary = pickle.load(open(join(savedir,"summary"),"rb"))
                            if src == "s2s":
                                rate_dict[f"{src}-{ovl}"][0,i_uth] += summary["rate_tob"]/subsets[src]["num_full_kmeans_seeds"]
                            else:
                                print(f"summary = {summary}")
                                rate_dict[f"{src}-{ovl}"][0,i_uth] += summary["rate"]/subsets[src]["num_full_kmeans_seeds"]
                    # Bootstraps
                    for i_bs,dir_bs in enumerate(subsets[src]["overlaps"][ovl]["bootstrap_dirs"]):
                        for i_uth,uthresh_b in enumerate(uthresh_list):
                            savedir = join(dir_bs,uthresh_dirname_fun(uthresh_b))
                            summary = pickle.load(open(join(savedir,"summary"),"rb"))
                            if src == "s2s":
                                rate_dict[f"{src}-{ovl}"][1+i_bs,i_uth] = summary["rate_tob"]
                            else:
                                rate_dict[f"{src}-{ovl}"][1+i_bs,i_uth] = summary["rate"]
                    # Now plot them all 
                    good_idx = np.where(rate_dict[f"{src}-{ovl}"][0] > 0)[0] if scale == 'log' else np.arange(len(uthresh_list))
                    h = ax.scatter(uthresh_list[good_idx]+errorbar_offsets[f"{src}-{ovl}"],rate_dict[f"{src}-{ovl}"][0,good_idx],color=subsets[src]["overlaps"][ovl]["color"],linewidth=2,marker='o',linestyle='-',label=subsets[src]["overlaps"][ovl]["label"],alpha=1.0,s=16, zorder=1)
                    handles += [h]
                    xlabels = None if src == "s2s" else ['']*len(good_idx)
                    bootstraps = 2*rate_dict[f"{src}-{ovl}"][0,good_idx] - rate_dict[f"{src}-{ovl}"][1:,:][:,good_idx]
                    if scale == 'log' or scale == 'logit':
                        bootstraps = np.maximum(0.5*ylim[scale][0], np.minimum(0.5*(ylim[scale][1]+1), bootstraps))
                    bplot = ax.boxplot(
                            bootstraps, 
                            positions=uthresh_list[good_idx]+errorbar_offsets[f"{src}-{ovl}"], whis=(5,95), 
                            patch_artist=True, labels=xlabels, manage_ticks=False, widths=du, sym='x', showmeans=False, zorder=0, showfliers=False,
                            boxprops={"color": subsets[src]["overlaps"][ovl]["color"], "facecolor": "white"},
                            whiskerprops={"color": subsets[src]["overlaps"][ovl]["color"]},
                            medianprops={"color": subsets[src]["overlaps"][ovl]["color"]}, 
                            capprops={"color": subsets[src]["overlaps"][ovl]["color"]}, 
                            flierprops={"markerfacecolor": None, "markeredgecolor": subsets[src]["overlaps"][ovl]["color"]},
                            )
                    savefig_suffix += f"{src}{ovl}_"
                    ax.legend(handles=handles,loc=loc[scale])
                    ax.set_ylim(ylim[scale])
                    uthresh_list_sorted = np.sort(uthresh_list)
                    xlim = [1.5*uthresh_list_sorted[0]-0.5*uthresh_list_sorted[1], 1.5*uthresh_list_sorted[-1]-0.5*uthresh_list_sorted[-2]]
                    ax.set_xlim(xlim)
                    print(f"xlim = {xlim}; ax xlim = {ax.get_xlim()}")
                    if scale == 'logit':
                        ax.set_yscale('function',functions=(myinvlogit,mylogit))
                    else:
                        ax.set_yscale(scale)
                    fig.savefig(join(paramdirs["s2s"],"rate_%s_%s_%s"%(bndy_suffix,savefig_suffix,scale)))
                    plt.close(fig)


if task_list["comparison"]["illustrate_dataset_flag"]:
    feat_filename_ra = join(expdirs["e5"],"X.npy")
    feat_filename_hc = join(expdirs["s2s"],"X.npy")
    ens_start_filename_ra = join(expdirs["e5"],"ens_start_idx.npy")
    ens_start_filename_hc = join(expdirs["s2s"],"ens_start_idx.npy")
    fall_year_filename_ra = join(expdirs["e5"],"fall_year_list.npy")
    fall_year_filename_hc = join(expdirs["s2s"],"fall_year_list.npy")
    tpt_feat_filename_ra = join(subsets["e5"]["overlaps"]["self"]["full_dirs"][0],"Y")
    tpt_feat_filename_hc = join(subsets["s2s"]["overlaps"]["self"]["full_dirs"][0],"Y")
    tthresh = np.array([tthresh0,tthresh1])*24.0
    winstrat.illustrate_dataset(
            uthresh_a,uthresh_list[[1,3,4]],tthresh,sswbuffer,
            feat_filename_ra,feat_filename_hc,
            tpt_feat_filename_ra,tpt_feat_filename_hc,
            ens_start_filename_ra,ens_start_filename_hc,
            fall_year_filename_ra,fall_year_filename_hc,
            feat_def,feat_display_dir
            )
    fall_year_filename_ra_dict = dict({k: join(expdirs[k],"fall_year_list.npy") for k in ["e2","e5"]})
    feat_filename_ra_dict = dict({key: join(expdirs[key],"X.npy") for key in ["e2","e5"]})
    colors = {src: subsets[src]["overlaps"]["self"]["color"] for src in ["e2","e5"]}
    labels = {src: subsets[src]["overlaps"]["self"]["label"] for src in ["e2","e5"]}
    winstrat.plot_zonal_wind_every_year(
            feat_filename_ra_dict,fall_year_filename_ra_dict,
            feat_def,feat_display_dir,colors,labels,
            uthresh_a,uthresh_list,tthresh,
            )
