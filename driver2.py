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
feat_display_dir = join(featdir,"display2")
if not exists(feat_display_dir): mkdir(feat_display_dir)
resultsdir = "/scratch/jf4241/ecmwf_data/results"
if not exists(resultsdir): mkdir(resultsdir)
daydir = join(resultsdir,"2022-03-07")
if not exists(daydir): mkdir(daydir)
expdir = join(daydir,"1")
if not exists(expdir): mkdir(expdir)
import helper
import strat_feat
import tpt_general

# COMPLETE listing of possible years to use for each dataset
fall_years = dict({
    "ei": np.arange(1979,2018),
    "e2": np.arange(1900,2008),
    "s2s": np.arange(1996,2017),
    })
# Fully overlapping estimates
intersection = np.intersect1d(np.intersect1d(fall_years["e2"],fall_years["ei"]),fall_years["s2s"])
print(f"intersection = {intersection}")
subsets = dict({
    "ei": dict({
        "ra_overlap_full_subset": np.intersect1d(fall_years["ei"],fall_years["e2"]),
        "full_subset": fall_years["s2s"], #np.intersect1d(fall_years["e2"],fall_years["ei"]),
        "num_bootstrap": 20, 
        "num_full_kmeans_seeds": 1,
        }),
    "e2": dict({
        "ra_overlap_full_subset": np.intersect1d(fall_years["ei"],fall_years["e2"]),
        "full_subset": fall_years["e2"], #np.intersect1d(fall_years["e2"],fall_years["ei"]),
        "num_bootstrap": 20, 
        "num_full_kmeans_seeds": 1,
        }),
    "s2s": dict({
        "full_subset": fall_years["s2s"],
        "num_bootstrap": 20, 
        "num_full_kmeans_seeds": 5,
        }),
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

# Resample with replacement
for key in sources:
    prng = np.random.RandomState(0) # This will be used for subsampling the years. 
    prng_overlap = np.random.RandomState(1) # This will be used for subsampling the years. 
    subsets[key]["full_kmeans_seeds"] = np.arange(subsets[key]["num_full_kmeans_seeds"]) # random-number generator seeds to use for KMeans. 
    Nyears = len(subsets[key]["full_subset"])
    subsets[key]["resampled_kmeans_seeds"] = subsets[key]["num_full_kmeans_seeds"] + np.arange(subsets[key]["num_bootstrap"])
    subsets[key]["resampled_subsets"] = [] #[np.zeros(Nyears,dtype=int) for i_boot in range(subsets[key]["num_bootstrap"])] #np.zeros((subsets[key]["num_bootstrap"],Nyears), dtype=int)
    for i_ss in range(subsets[key]["num_bootstrap"]):
        subsets[key]["resampled_subsets"] += [prng.choice(subsets[key]["full_subset"], size=Nyears, replace=True)]
    # Concatenate these
    subsets[key]["all_kmeans_seeds"] = np.concatenate((subsets[key]["full_kmeans_seeds"], subsets[key]["resampled_kmeans_seeds"]))
    subsets[key]["all_subsets"] = [subsets[key]["full_subset"] for i_ss in range(subsets[key]["num_full_kmeans_seeds"])] + subsets[key]["resampled_subsets"]
    # Do a separate resampling for reanalysis for the period of overlap. 
    print(f"Before overlapping for key {key}, len(all_subsets) = {len(subsets[key]['all_subsets'])}")
    if key in ["ei","e2"]:
        Nyears_overlap = len(subsets[key]["ra_overlap_full_subset"])
        subsets[key]["ra_overlap_resampled_subsets"] = [] #((subsets[key]["num_bootstrap"],Nyears), dtype=int)
        for i_ss in range(subsets[key]["num_bootstrap"]):
            subsets[key]["ra_overlap_resampled_subsets"] += [prng_overlap.choice(subsets[key]["ra_overlap_full_subset"], size=Nyears_overlap, replace=True)]
        subsets[key]["ra_overlap_all_subsets"] = [subsets[key]["ra_overlap_full_subset"]] + subsets[key]["ra_overlap_resampled_subsets"]
        subsets[key]["all_subsets"] += subsets[key]["ra_overlap_all_subsets"]
        print(f"For {key}, num ra overlap resampled subsets = {len(subsets[key]['ra_overlap_all_subsets'])}. And len(all_subsets) = {len(subsets[key]['all_subsets'])}")
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
ra_flags = dict({
    "ei": True,
    "e2": True,
    })

#subsetdirs = dict({key: [join(paramdirs[key],"%i-%i"%(subset[0],subset[-1]+1)) for subset in subset_lists[key]] for key in sources})
for key in sources:
    subsets[key]["full_dirs"] = [join(paramdirs[key],"full_seed%i"%(seed)) for seed in subsets[key]["full_kmeans_seeds"]]
    subsets[key]["resampled_dirs"] = [join(paramdirs[key],"resampled_%i"%(i_ss)) for i_ss in range(subsets[key]["num_bootstrap"])]
    subsets[key]["all_dirs"] = np.concatenate((subsets[key]["full_dirs"],subsets[key]["resampled_dirs"]))
    if key in ["ei","e2"]:
        subsets[key]["ra_overlap_full_dirs"] = [join(paramdirs[key],"ra_overlap_full")]
        subsets[key]["ra_overlap_resampled_dirs"] = [join(paramdirs[key], "ra_overlap_resampled_%i"%(i_ss)) for i_ss in range(subsets[key]["num_bootstrap"])]
        subsets[key]["ra_overlap_all_dirs"] = np.concatenate((subsets[key]["ra_overlap_full_dirs"],subsets[key]["ra_overlap_resampled_dirs"]))
        subsets[key]["all_dirs"] = np.concatenate((subsets[key]["all_dirs"],subsets[key]["ra_overlap_all_dirs"]),axis=0)


    

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
        "plot_rate_flag":                     1,
        "illustrate_dataset_flag":            0,
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
tthresh0 = 31 # First day that SSW could happen
tthresh1 = 31 + 30 + 31 + 31 + 28  # Last day that SSW could happen: end of February or March
sswbuffer = 0.0 # minimum buffer time between one SSW and the next
uthresh_a = 100.0 # vortex is in state A if it exceeds uthresh_a and it's been sswbuffer days since being inside B
uthresh_list = np.arange(0,-36,-5) #np.array([5.0,0.0,-5.0,-10.0,-15.0,-20.0])
plottable_uthresh_list = [0,-15]
uthresh_dirname_fun = lambda uthresh_b: "tth%i-%i_uthb%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_b,uthresh_a,sswbuffer)

# =============================================================
# TPT direct estimates from ERA-Interim and ERA-20C 
for key in ["ei","e2"]:
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
tpt_feat_filename_ra_dict = dict({key: join(subsets[key]["all_dirs"][0],"Y") for key in ["ei","e2"]})
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
colors = dict({"ei": "black", "e2": "dodgerblue", "s2s": "red"}) #, "s2s_naive": "cyan"})
labels = dict({"ei": "ERA-Interim", "e2": "ERA-20C", "s2s": "S2S"}) 
if task_list["comparison"]["plot_rate_flag"]:
    rate_lists = dict({key: np.zeros((len(subsets[key]["all_subsets"]),len(uthresh_list))) for key in sources})
    for i_uth in range(len(uthresh_list)):
        uthresh_b = uthresh_list[i_uth]
        for key in ["ei","e2","s2s"]:
            for i_subset,subset in enumerate(subsets[key]["all_subsets"]):
                savedir = join(subsets[key]["all_dirs"][i_subset],uthresh_dirname_fun(uthresh_b))
                summary = pickle.load(open(join(savedir,"summary"),"rb"))
                if key in ["e2","ei"]:
                    rate_lists[key][i_subset,i_uth] = summary["rate"]
                elif key == "s2s":
                    rate_lists[key][i_subset,i_uth] = summary["rate_tob"]
    # ------------ Line plot -------------------
    ylim = {'log': [1e-3,1.0], 'logit': [1e-3,0.8], 'linear': [0.0,1.0]}
    loc = {'log': 'lower right', 'logit': 'lower right', 'linear': 'upper left'}
    bndy_suffix = "tth%i-%i_utha%i_buff%i"%(tthresh0,tthresh1,uthresh_a,sswbuffer)
    # Build this up one curve at a time
    label_needed = dict({key: True for key in colors.keys()})
    du = np.abs(uthresh_list[1] - uthresh_list[0])/8.0 # How far to offset the x axis positions for the three timeseries
    errorbar_offsets = dict({"ei": -du, "e2": du, "s2s": 0})
    quantiles = np.array([0.05,0.25,0.75,0.95])
    # Plot rates for full ranges 
    # Design custom logit function
    myscale = 10.0
    def mylogit(x):
        #print(f"x = {x}, type(x) = {type(x)}")
        return 1.0/(1 + np.exp(-x/myscale))
    def myinvlogit(x):
        return -myscale*np.log(1.0/x - 1)
    for scale in ['linear','log','logit']:
        fig,ax = plt.subplots()
        savefig_suffix = ""
        ax.set_xlabel("Zonal wind threshold [m/s]",fontdict=font)
        ax.set_ylabel("Rate",fontdict=font)
        handles = []
        for key in ['ei','e2','s2s']:
            print(f"Starting to plot rate list for {key}")
            # ---------- Plot a single line with error bars ---------
            full_rate = rate_lists[key][np.arange(len(subsets[key]["full_kmeans_seeds"]))]
            full_rate_mean = full_rate.mean(axis=0)
            good_idx = np.where(full_rate_mean > 0)[0] if (scale == 'log' or scale == 'logit') else np.arange(len(uthresh_list))
            # Plot the estimate from full dataset (mean of full kmeans seeds)
            h = ax.scatter(uthresh_list[good_idx]+errorbar_offsets[key],full_rate_mean[good_idx],color=colors[key],linewidth=2,marker='o',linestyle='-',label=f"{labels[key]} {subsets[key]['full_subset'][0]}-{subsets[key]['full_subset'][-1]}",alpha=1.0,s=16, zorder=1)
            handles += [h]
            # ---- box-and-whisker plot -------
            resamp_idx_range = subsets[key]["num_full_kmeans_seeds"] + np.arange(subsets[key]["num_bootstrap"])
            print(f"key = {key}, resamp_idx_range = {resamp_idx_range}, num_bootstrap = {subsets[key]['num_bootstrap']}, rate lists shape = {rate_lists[key].shape}")
            print(f"key = {key}, colors[key] = {colors[key]}")
            xlabels = None if key == "s2s" else ['']*len(good_idx)
            # Mask the resampled values appropriately
            bootstraps = 2*full_rate_mean[good_idx] - rate_lists[key][resamp_idx_range,:][:,good_idx]
            if scale == 'log' or scale == 'logit':
                bootstraps = np.maximum(0.5*ylim[scale][0], np.minimum(0.5*(ylim[scale][1]+1), bootstraps))
                print(f"bootstraps: min = {bootstraps.min()}, max = {bootstraps.max()}")
                print(f"full_rate_mean[good_idx]: min={full_rate_mean[good_idx].min()}, max={full_rate_mean[good_idx].max()}")
            bplot = ax.boxplot(
                    bootstraps, 
                    positions=uthresh_list[good_idx]+errorbar_offsets[key], whis=(5,95), 
                    patch_artist=True, labels=xlabels, manage_ticks=False, widths=du, sym='x', showmeans=False, zorder=0, showfliers=False,
                    boxprops={"color": colors[key], "facecolor": "white"},
                    whiskerprops={"color": colors[key]},
                    medianprops={"color": colors[key]}, capprops={"color": colors[key]}, 
                    flierprops={"markerfacecolor": None, "markeredgecolor": colors[key]}
                    )
            #for box in bplot['boxes']:
            #    box.set_facecolor(colors[key])

            # ---- manual line plot -----------
            #for i_uth in good_idx: 
            #    uth = uthresh_list[i_uth]
            #    ax.plot((uth+errorbar_offsets[key])*np.ones(2), np.array([rate_lists[key][:,i_uth].min(), rate_lists[key][:,i_uth].max()]), color=colors[key], linewidth=2)
            savefig_suffix += key
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
    # Plot rates for overlapping ranges for reanalysis
    for scale in ['linear','log']:
        fig,ax = plt.subplots()
        savefig_suffix = ""
        ax.set_xlabel("Zonal wind threshold [m/s]",fontdict=font)
        ax.set_ylabel("Rate",fontdict=font)
        handles = []
        for key in ['ei','e2']:
            print(f"Starting to plot overlapping rate list for {key}")
            # ---------- Plot a single line with error bars ---------
            full_rate_mean = rate_lists[key][subsets[key]["num_full_kmeans_seeds"]+subsets[key]["num_bootstrap"]] #+np.arange(len(subsets[key]["ra_overlap_resampled_subsets"]))]
            good_idx = np.where(full_rate_mean > 0)[0] if scale == 'log' else np.arange(len(uthresh_list))
            # Plot the estimate from full dataset (mean of full kmeans seeds)
            h = ax.scatter(uthresh_list[good_idx]+errorbar_offsets[key],full_rate_mean[good_idx],color=colors[key],linewidth=2,marker='o',linestyle='-',label=f"{labels[key]} {subsets[key]['ra_overlap_full_subset'][0]}-{subsets[key]['ra_overlap_full_subset'][-1]}",alpha=1.0,s=16, zorder=1)
            handles += [h]
            # ---- box-and-whisker plot -------
            resamp_idx_range = subsets[key]["num_full_kmeans_seeds"] + subsets[key]["num_bootstrap"] + 1 + np.arange(subsets[key]["num_bootstrap"]) 
            xlabels = None if key == "e2" else ['']*len(good_idx)
            bplot = ax.boxplot(
                    2*full_rate_mean[good_idx]-rate_lists[key][resamp_idx_range,:][:,good_idx], 
                    positions=uthresh_list[good_idx]+errorbar_offsets[key], whis=(5,95), 
                    patch_artist=True, labels=xlabels, manage_ticks=False, widths=du, sym='x', showmeans=False, zorder=0, showfliers=False,
                    boxprops={"color": colors[key], "facecolor": "white"},
                    whiskerprops={"color": colors[key]},
                    medianprops={"color": colors[key]}, capprops={"color": colors[key]}, 
                    flierprops={"markerfacecolor": None, "markeredgecolor": colors[key]}
                    )
            #for box in bplot['boxes']:
            #    box.set_facecolor(colors[key])

            # ---- manual line plot -----------
            #for i_uth in good_idx: 
            #    uth = uthresh_list[i_uth]
            #    ax.plot((uth+errorbar_offsets[key])*np.ones(2), np.array([rate_lists[key][:,i_uth].min(), rate_lists[key][:,i_uth].max()]), color=colors[key], linewidth=2)
            savefig_suffix += key
            ax.legend(handles=handles,loc=loc[scale])
            ax.set_ylim(ylim[scale])
            uthresh_list_sorted = np.sort(uthresh_list)
            xlim = [1.5*uthresh_list_sorted[0]-0.5*uthresh_list_sorted[1], 1.5*uthresh_list_sorted[-1]-0.5*uthresh_list_sorted[-2]]
            ax.set_xlim(xlim)
            ax.set_yscale(scale)
            fig.savefig(join(paramdirs["s2s"],"rate_ra_overlap_%s_%s_%s"%(bndy_suffix,savefig_suffix,scale)))
            plt.close(fig)


if task_list["comparison"]["illustrate_dataset_flag"]:
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
            feat_filename_ra,feat_filename_hc,
            tpt_feat_filename_ra,tpt_feat_filename_hc,
            ens_start_filename_ra,ens_start_filename_hc,
            fall_year_filename_ra,fall_year_filename_hc,
            feat_def,feat_display_dir
            )
    fall_year_filename_ra_dict = dict({k: join(expdirs[k],"fall_year_list.npy") for k in ["ei","e2"]})
    winstrat.plot_zonal_wind_every_year(
            feat_filename_ra_dict,fall_year_filename_ra_dict,
            feat_def,feat_display_dir,colors,labels,
            uthresh_a,uthresh_list,tthresh,
            )
