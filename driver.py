import pickle
import pandas
import numpy as np
from scipy.stats import binom as scipy_binom
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
# Auxiliary imports from other python files
import helper
import strat_feat
import tpt_general

# -------------- Set the list of tasks --------------
task_list = dict({
    "featurization": dict({
        "create_features_flag":               0,
        "display_features_flag":              0,
        }),
    "ei": dict({
        "tpt_featurize_flag":                 0,
        "tpt_flag":                           0,
        }),
    "e2": dict({
        "tpt_featurize_flag":                 0, 
        "tpt_flag":                           0,
        }),
    "e5": dict({
        "tpt_featurize_flag":                 0, 
        "tpt_flag":                           0,
        }),
    "s2s": dict({
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

# Download data? 
task_list["e5"]["download_flag"] =  0
task_list["ei"]["download_flag"] =  0
task_list["e2"]["download_flag"] =  0
task_list["s2s"]["download_flag"] = 0
# Evaluate databases?
task_list["e5"]["evaluate_database_flag"] =  0
task_list["ei"]["evaluate_database_flag"] =  0
task_list["e2"]["evaluate_database_flag"] =  0
task_list["s2s"]["evaluate_database_flag"] = 0
# ---------------------------------------------------------

# ------------- Set directories for code and data ----------
codedir = "/home/jf4241/ecmwf/tpt_ecmwf"
os.chdir(codedir)
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

# ------------------ Download each dataset as assigned --------------
download_scripts = dict({
    "ei": request_ei.py,
    "e2": request_e2.py,
    "e5": request_e5.py,
    "s2s": request_s2s.py,
    })
for src in data_sources:
    if task_list[src]["download_flag"]:
        # Call the corresponding bash script
# ------------------------------------------------------------------------
# ---------------- Work out the overlaps between each pair of datasets -------
intersection_ra = np.intersect1d(fall_years["e2"],fall_years["e5"]) # Intersection of all three reanalysis datasets (ERA-Interim is a strict subset of ERA-5)
num_bootstrap_global = 40 # Number of bootstrap resamplings to perform for confidence intervals on rates
num_full_kmeans_seeds_hc = 5 # Number of different seeds to average over for the DGA procedure below 
subsets = dict({
    "ei": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["ei"],
                "color": "red",
                "label": f"ERA-I {fall_years['ei'][0]}-{fall_years['ei'][-1]}",
                }),
            "hc": dict({"full": np.intersect1d(fall_years["ei"],fall_years["s2s"]),
                "color": "red",
                "label": f"ERA-I {max(fall_years['ei'][0],fall_years['s2s'][0])}-{min(fall_years['ei'][-1],fall_years['s2s'][-1])}",
                }),
            }),
        "num_bootstrap": num_bootstrap_global, 
        "dataset_type": "reanalysis",
        }),
    "e2": dict({
        "overlaps": dict({
            "self": dict({"full": fall_years["e2"], 
                "color": "dodgerblue", 
                "label": f"ERA-20C {fall_years['e2'][0]}-{fall_years['e2'][-1]}",
                }),
            "ra": dict({"full": intersection_ra, 
                "color": "dodgerblue", 
                "label": f"ERA-20C {intersection_ra[0]}-{intersection_ra[-1]}"
                }),
            "hc": dict({"full": np.intersect1d(fall_years["e2"],fall_years["s2s"]), 
                "color": "dodgerblue", 
                "label": f"ERA-20C {max(fall_years['e2'][0],fall_years['s2s'][0])}-{min(fall_years['e2'][-1],fall_years['s2s'][-1])}"
                }),
            }),
        "num_bootstrap": num_bootstrap_global, 
        "dataset_type": "reanalysis",
        }),
    "e5": dict({
        "generic_label": "ERA-5",
        "overlaps": dict({
            "self": dict({"full": fall_years["e5"], 
                "color": "gray", 
                "label": f"ERA-5 {fall_years['e5'][0]}-{fall_years['e5'][-1]}"
                }),
            "ra": dict({"full": intersection_ra, 
                "color": "black", 
                "label": f"ERA-5 {intersection_ra[0]}-{intersection_ra[-1]}"
                }),
            "ei": dict({"full": np.intersect1d(fall_years["e5"],fall_years["ei"]), 
                "color": "black", 
                "label": f"ERA-5 {fall_years['ei'][0]}-{fall_years['ei'][-1]}"
                }),
            "hc": dict({"full": np.intersect1d(fall_years["e5"],fall_years["s2s"]), 
                "color": "orange", 
                "label": f"ERA-5 {max(fall_years['e5'][0],fall_years['s2s'][0])}-{min(fall_years['e5'][-1],fall_years['s2s'][-1])}"
                }),
            }),
        "num_bootstrap": num_bootstrap_global, 
        "dataset_type": "reanalysis",
        }),
    "s2s": dict({
        "generic_label": "S2S",
        "overlaps": dict({
            "self": dict({"full": fall_years["s2s"],
                "color": "red",
                "label": f"S2S {fall_years['s2s'][0]}-{fall_years['s2s'][-1]}"
                }),
            }),
        "num_bootstrap": num_bootstrap_global, 
        "dataset_type": "hindcast",
        "num_full_kmeans_seeds": num_full_kmeans_seeds_hc,
        }),
    })

# Resample with replacement
for src in data_sources:
    for ovl in subsets[src]["overlaps"].keys():
        Nyears = len(subsets[src]["overlaps"][ovl]['full'])
        subsets[src]["overlaps"][ovl]["bootstrap"] = []
        prng = np.random.RandomState(0)
        for i_ss in range(subsets[src]["num_bootstrap"]):
            subsets[src]["overlaps"][ovl]["bootstrap"] += [prng.choice(subsets[src]["overlaps"][ovl]["full"], size=Nyears, replace=True)]

#print(f"era5 subsets: \nself:\n\t{subsets['e5']['overlaps']['self']}\nra:\n\t{subsets['e5']['overlaps']['ra']}\nhc:\n\t{subsets['e5']['overlaps']['hc']}")


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
spring_day0 = 190.0
Npc_per_level_max = 15
num_vortex_moments_max = 4 # Area, mean, variance, skewness, kurtosis. But it's too expensive. At least we need a linear approximation. 
heatflux_wavenumbers_per_level_max = 3 # 0: nothing. 1: zonal mean. 2: wave 1. 3: wave 2. 
# ----------------- Phase space definition parameters -------
delaytime_days = 20.0 # Both zonal wind and heat flux will be saved with this time delay. Must be shorter than tthresh0
# ----------------- Directories for this experiment --------
print(f"expdir = {expdir}, data_sources = {data_sources}")
expdirs = dict({key: join(expdir,key) for key in data_sources})
print(f"expdirs = {expdirs}")
for key in data_sources:
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
for key in data_sources:
    if not exists(paramdirs[key]): mkdir(paramdirs[key])

# For each subset of each data source, create
# 1. Directories 
# 2. Subset specifications, in the form of a list of years
# 3. A running master list of directories and subsets to loop through later. 
# 4. For the hindcast subsets, include a seed for KMeans
for src in data_sources:
    subsets[src]["all_dirs"] = []
    subsets[src]["all_subsets"] = []
    if subsets[src]["dataset_type"] == "reanalysis":
        num_full_subsets = 1
    elif subsets[src]["dataset_type"] == "hindcast":
        num_full_subsets = subsets[src]["num_full_kmeans_seeds"]
        subsets[src]["all_kmeans_seeds"] = []
    else:
        raise Exception(f"You gave me a dataset_type of {subsets[src]['dataset_type']}, but the only supported ones are 'reanalysis' and 'hindcast'")
    for ovl in subsets[src]["overlaps"].keys():
        # Full samples, varying KMeans seed in the case of DGA. 
        subsets[src]["overlaps"][ovl]["full_dirs"] = [join(paramdirs[src],f"overlap{ovl}_full_seed{seed}") for seed in range(num_full_subsets)]
        subsets[src]["all_dirs"] += subsets[src]["overlaps"][ovl]["full_dirs"]
        subsets[src]["all_subsets"] += [subsets[src]["overlaps"][ovl]["full"] for i_ss in range(num_full_subsets)]
        if subsets[src]["dataset_type"] == "hindcast":
            subsets[src]["all_kmeans_seeds"] += [seed for seed in range(num_full_subsets)]
        # Bootstrap samples
        subsets[src]["overlaps"][ovl]["bootstrap_dirs"] = [join(paramdirs[src],"overlap%s_bootstrap%i"%(ovl,i_bs)) for i_bs in range(subsets[src]["num_bootstrap"])]
        subsets[src]["all_dirs"] += subsets[src]["overlaps"][ovl]["bootstrap_dirs"]
        subsets[src]["all_subsets"] += subsets[src]["overlaps"][ovl]["bootstrap"]
        if subsets[src]["dataset_type"] == "hindcast":
            subsets[src]["all_kmeans_seeds"] += [subsets[src]["num_full_kmeans_seeds"]+seed for seed in range(subsets[src]["num_bootstrap"])]




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
tthresh0 = monthrange(1901,10)[1] + 1 # First day that SSW could happen is Nov. 1
tthresh1 = sum([monthrange(1901,i)[1] for i in [10,11,12]]) + sum([monthrange(1902,i)[1] for i in [1,2]]) # Last day that SSW could happen: February 28
sswbuffer = 0.0 # minimum buffer time between one SSW and the next
uthresh_a = 100.0 # vortex is in state A if it exceeds uthresh_a and it's been sswbuffer days since being inside B
uthresh_list = np.arange(0,-36,-5) #np.array([5.0,0.0,-5.0,-10.0,-15.0,-20.0])
plottable_uthresh_list = [0,-10,-15,-20]
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
#feat_filename_ra_dict = dict({key: join(expdirs[key],"X.npy") for key in ["e5","e2"]})
#tpt_feat_filename_ra_dict = dict({key: join(subsets[key]["overlaps"]["self"]["full_dirs"][0],"Y") for key in ["e5","e2"]})
keys_ra = dict({
    "e5": ["self","hc",],
    "e2": ["self",],
    })
keys_ra_current = ["e5-self"] # Only plot this subset for the overlays
colors_ra_dict = dict({})
labels_dict = dict({})
feat_filename_ra_dict = dict({})
fall_year_filename_ra_dict = dict({})
tpt_feat_filename_ra_dict = dict({})
labels = dict({})
for src in keys_ra.keys():
    for ovl in keys_ra[src]:
        srcovl = f"{src}-{ovl}"
        feat_filename_ra_dict[srcovl] = join(expdirs[src], "X.npy")
        fall_year_filename_ra_dict[srcovl] = join(expdirs[src], "fall_year_list.npy")
        tpt_feat_filename_ra_dict[srcovl] = join(subsets[src]["overlaps"][ovl]["full_dirs"][0], "Y")
        colors_ra_dict[srcovl] = subsets[src]["overlaps"][ovl]["color"]
        labels_dict[srcovl] = subsets[src]["overlaps"][ovl]["label"]
labels_dict["s2s-self"] = subsets["s2s"]["overlaps"]["self"]["label"]

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
            tpt.plot_results_data(
                    feat_filename,tpt_feat_filename,
                    feat_filename_ra_dict,tpt_feat_filename_ra_dict,
                    fall_year_filename_ra_dict,
                    feat_def,savedir,winstrat,algo_params,
                    spaghetti_flag=0*(uthresh_b in plottable_uthresh_list),
                    fluxdens_flag=1*(uthresh_b in plottable_uthresh_list),
                    verify_leadtime_flag=0*(uthresh_b in plottable_uthresh_list),
                    current2d_flag=0*(uthresh_b in plottable_uthresh_list),
                    comm_corr_flag=0*(uthresh_b in plottable_uthresh_list),
                    colors_ra_dict=colors_ra_dict,labels_dict=labels_dict,
                    keys_ra_current=keys_ra_current,
                    )

# =============================================================================

# ------------- Compare rates ---------------------
# 2 plots: 
#   1. S2S, ERA20C (full), ERA5 (full), ERA5 (overlap with S2S)
#   2. ERA20C (overlap with ERA5), ERA5 (overlap with ERA20C) 

# Plot 1: specify which overlaps to use from each source
boxplot_keys_hc = dict({
    "e5": ["hc","self",],
    "e2": ["self",],
    "s2s": ["self",],
    })
boxplot_keys_ra = dict({
    "e2": ["ra",],
    "e5": ["ra",],
    })
boxplot_keys_ei = dict({
    "e5": ["ei",],
    "ei": ["self",],
    })

# Two possibilities for error bars: bootstrap, or binomial confidence intervals.
binomial_flag = True
        
ylim = {'log': [1e-3,1.0], 'logit': [1e-3,0.8], 'linear': [0.0,1.0]}
loc = {'log': 'lower right', 'logit': 'lower right', 'linear': 'upper left'}
bndy_suffix = "tth%i-%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_a,sswbuffer)
du = np.abs(uthresh_list[1] - uthresh_list[0])/8.0 # How far to offset the x axis positions for the three timeseries
errorbar_offsets = dict({"e5-hc": -3/2*du, "e5-self": -du/2, "e2-self": du/2, "s2s-self": 3*du/2,
    "e5-ra": -1/2*du, "e5-ei": -1/2*du, "e2-ra": 1/2*du, "ei-self": du/2})
quantiles = np.array([0.05,0.25,0.75,0.95])

y0 = 0.1
base = 10
def mylogit(y):
    ilo = np.where(y < y0)
    iup = np.where(y >= y0)
    z = np.zeros_like(y)
    z[ilo] = np.log(y[ilo])/np.log(base)
    z[iup] = (y[iup]/y0 + np.log(y0) - 1)/np.log(base)
    return z
def myinvlogit(z):
    ilo = np.where(z < np.log(y0)/np.log(base))
    iup = np.where(z >= np.log(y0)/np.log(base))
    y = np.zeros_like(z)
    y[ilo] = np.exp(z[ilo]*np.log(base)) 
    y[iup] = y0*(z[iup]*np.log(base) + 1 - np.log(y0))
    return y


# Build the rate dictionary as we go
rate_dict = dict({})
nyears_dict = dict({})
conf_levels = [0.5,0.95]
if task_list["comparison"]["plot_rate_flag"]:
    for boxplot_keys in [boxplot_keys_hc,boxplot_keys_ra,boxplot_keys_ei]:
        for scale in ['log']: #['logit','linear','log']:
            fig,ax = plt.subplots()
            savefig_suffix = ""
            ax.set_xlabel("Zonal wind threshold [m/s]",fontdict=font)
            ax.set_ylabel("Rate",fontdict=font)
            handles = []
            for src in boxplot_keys:
                if subsets[src]["dataset_type"] == "reanalysis":
                    num_full_subsets = 1
                elif subsets[src]["dataset_type"] == "hindcast":
                    num_full_subsets = subsets[src]["num_full_kmeans_seeds"]
                    subsets[src]["all_kmeans_seeds"] = []
                else:
                    raise Exception(f"You gave me a dataset_type of {subsets[src]['dataset_type']}, but the only supported ones are 'reanalysis' and 'hindcast'")
                for ovl in boxplot_keys[src]:
                    srcovl = f"{src}-{ovl}"
                    rate_dict[srcovl] = np.zeros((1+subsets[src]['num_bootstrap'], len(uthresh_list))) # First entry for mean of full kmeans, rest of them for bootstrap
                    nyears_dict[srcovl] = len(subsets[src]["overlaps"][ovl]["full"])
                    # First row
                    for i_km,dir_km in enumerate(subsets[src]["overlaps"][ovl]["full_dirs"]):
                        for i_uth,uthresh_b in enumerate(uthresh_list):
                            savedir = join(dir_km,uthresh_dirname_fun(uthresh_b))
                            summary = pickle.load(open(join(savedir,"summary"),"rb"))
                            if src == "s2s":
                                rate_dict[srcovl][0,i_uth] += summary["rate_tob"]/num_full_subsets
                            else:
                                print(f"summary = {summary}")
                                rate_dict[srcovl][0,i_uth] += summary["rate"]/num_full_subsets
                    # Bootstraps
                    for i_bs,dir_bs in enumerate(subsets[src]["overlaps"][ovl]["bootstrap_dirs"]):
                        for i_uth,uthresh_b in enumerate(uthresh_list):
                            savedir = join(dir_bs,uthresh_dirname_fun(uthresh_b))
                            summary = pickle.load(open(join(savedir,"summary"),"rb"))
                            if src == "s2s":
                                rate_dict[srcovl][1+i_bs,i_uth] = summary["rate_tob"]
                            else:
                                rate_dict[srcovl][1+i_bs,i_uth] = summary["rate"]
            for src in boxplot_keys:
                for ovl in boxplot_keys[src]:
                    srcovl = f"{src}-{ovl}"
                    # Now plot them all 
                    #good_idx = np.where(rate_dict[srcovl][0] > 0)[0] if (scale == 'log' or scale == 'logit') else np.arange(len(uthresh_list))
                    good_idx = np.arange(len(uthresh_list))
                    print(f"scale = {scale}, good_idx = {good_idx}")
                    h = ax.scatter(uthresh_list[good_idx]+errorbar_offsets[srcovl],rate_dict[srcovl][0,good_idx],color=subsets[src]["overlaps"][ovl]["color"],linewidth=2,marker="_",linestyle='-',label=subsets[src]["overlaps"][ovl]["label"],alpha=1.0,s=32, zorder=1)
                    #handles += [h]
                    xlabels = None if src == "s2s" else ['']*len(good_idx)
                    bootstraps = 2*rate_dict[srcovl][0,good_idx] - rate_dict[srcovl][1:,:][:,good_idx]
                    if scale == 'log' or scale == 'logit':
                        bootstraps = np.maximum(0.5*ylim[scale][0], np.minimum(0.5*(ylim[scale][1]+1), bootstraps))
                    for i_conf in np.arange(len(conf_levels)):
                        if binomial_flag and (src != "s2s"):
                            # Calculate confidence intervals for the binomial distribution coefficients
                            num_events = rate_dict[srcovl][0,good_idx]*nyears_dict[srcovl]
                            if (src != "s2s") and np.max(np.abs(num_events - np.round(num_events))) > 1e-10:
                                raise Exception(f"ERROR: num_events = {num_events}. the number of events given the empirical rate does not seem to be an integer. src = {src}")
                            if "s2s" in boxplot_keys:
                                truth_prob = rate_dict["s2s-self"][0,good_idx]
                            else:
                                truth_prob = rate_dict[srcovl][0,good_idx]
                            quantile_lower = scipy_binom.ppf((1-conf_levels[i_conf])/2, nyears_dict[srcovl], truth_prob) / nyears_dict[srcovl]
                            quantile_upper = scipy_binom.ppf((1+conf_levels[i_conf])/2, nyears_dict[srcovl], truth_prob) / nyears_dict[srcovl]
                            # Manipulate so the confidence interval contains the point estimate
                            conf_lower = quantile_lower 
                            conf_upper = quantile_upper 
                            #conf_lower = rate_dict[srcovl][0,good_idx] - (quantile_upper - rate_dict["s2s-self"][0,good_idx])
                            #conf_upper = rate_dict[srcovl][0,good_idx] + (rate_dict["s2s-self"][0,good_idx] - quantile_lower)
                            print(f"conf_lower = {conf_lower}")
                        else:
                            conf_lower = np.quantile(bootstraps, (1-conf_levels[i_conf])/2, axis=0)
                            conf_upper = np.quantile(bootstraps, (1+conf_levels[i_conf])/2, axis=0)
                        if scale == 'logit':
                            print(f"At confidence level {conf_levels[i_conf]} for {src}, interval widths = ({conf_upper-conf_lower})")
                        # Create fake data points halfway in between lower and upper
                        conf_mid = 0.5*(conf_lower + conf_upper)
                        yerr = 0.5*(conf_upper - conf_lower)
                        for i_uth in good_idx:
                            h, = ax.plot(np.ones(2)*(uthresh_list[i_uth]+errorbar_offsets[srcovl]), [conf_lower[i_uth],conf_upper[i_uth]], color=subsets[src]["overlaps"][ovl]["color"],linewidth=3.0/(3**i_conf),zorder=0,label=subsets[src]["overlaps"][ovl]["label"],) #, marker='x')
                        if i_conf == 0:
                            handles += [h]
                        #ax.errorbar(uthresh_list[good_idx]+errorbar_offsets[srcovl],conf_mid,yerr=yerr,fmt='none',color=subsets[src]["overlaps"][ovl]["color"],linewidth=4/(2**i_conf),zorder=0,capthick=1)
                    savefig_suffix += f"{src}-{ovl}"
                    leg = ax.legend(handles=handles,loc=loc[scale])
                    #for legobj in leg.legendHandles:
                    #    print(f"legobj = {legobj}")
                    #    legobj.set_linewidth(4.0)
                    ax.set_ylim(ylim[scale])
                    uthresh_list_sorted = np.sort(uthresh_list)
                    xlim = [1.5*uthresh_list_sorted[0]-0.5*uthresh_list_sorted[1], 1.5*uthresh_list_sorted[-1]-0.5*uthresh_list_sorted[-2]]
                    ax.set_xlim(xlim)
                    if scale == 'logit':
                        ax.set_yscale('function',functions=(mylogit,myinvlogit))
                    else:
                        ax.set_yscale(scale)
                    fig.savefig(join(paramdirs["s2s"],"rate_%s_%s_%s_binom%i"%(bndy_suffix,savefig_suffix,scale,binomial_flag)))
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
    label_ra = subsets["e5"]["generic_label"]
    label_hc = subsets["s2s"]["generic_label"]
    tthresh = np.array([tthresh0,tthresh1])*24.0
    winstrat.illustrate_dataset(
            uthresh_a,uthresh_list[[0,2,3]],tthresh,sswbuffer,
            feat_filename_ra,feat_filename_hc,
            label_ra,label_hc,
            tpt_feat_filename_ra,tpt_feat_filename_hc,
            ens_start_filename_ra,ens_start_filename_hc,
            fall_year_filename_ra,fall_year_filename_hc,
            feat_def,feat_display_dir
            )
    fall_year_filename_ra_dict = dict({k: join(expdirs[k],"fall_year_list.npy") for k in ["e2","e5","ei"]})
    feat_filename_ra_dict = dict({key: join(expdirs[key],"X.npy") for key in ["e2","e5","ei"]})
    colors = {src: subsets[src]["overlaps"]["self"]["color"] for src in ["e2","e5","ei"]}
    labels = {src: subsets[src]["overlaps"]["self"]["label"] for src in ["e2","e5","ei"]}
    winstrat.plot_zonal_wind_every_year(
            feat_filename_ra_dict,fall_year_filename_ra_dict,
            feat_def,feat_display_dir,colors,labels,
            uthresh_a,uthresh_list,tthresh,
            )
