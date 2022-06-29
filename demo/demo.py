# Create a minimal demonstration of the DGA method. This does not save all the intermediate results, as the main code does, because it's too intensive. 
# Possibly allow for heterogeneous ensemble sizes and lengths 
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt 
import model_crommelin_seasonal
import feature_crommelin 
import sys
import os
from os import mkdir
from os.path import join,exists

# ----------- Create directory to save results ----------
topic_dir = "/scratch/jf4241/crommelin"
if not exists(topic_dir): mkdir(topic_dir)
day_dir = join(topic_dir,"2022-06-25")
if not exists(day_dir): mkdir(day_dir)
exp_dir = join(day_dir,"0")
if not exists(exp_dir): mkdir(exp_dir)
ra_dir = join(exp_dir,"reanalysis_data")
if not exists(ra_dir): mkdir(ra_dir)
ra_dir_contiguous = join(ra_dir,"contiguous")
if not exists(ra_dir_contiguous): mkdir(ra_dir_contiguous)
ra_dir_seasonal = join(ra_dir,"seasonal")
if not exists(ra_dir_seasonal): mkdir(ra_dir_seasonal)
hc_dir = join(exp_dir,"hindcast_data")
if not exists(hc_dir): mkdir(hc_dir)
featspec_dir = join(exp_dir,"featspec")
if not exists(featspec_dir): mkdir(featspec_dir)
results_dir = join(exp_dir,"results")
if not exists(results_dir): mkdir(results_dir)
results_dir_ra = join(results_dir,"ra")
if not exists(results_dir_ra): mkdir(results_dir_ra)
results_dir_hc = join(results_dir,"hc")
if not exists(results_dir_hc): mkdir(results_dir_hc)

# -------------- Specify which tasks to execute -------------
integrate_flag =             0
plot_integration_flag =      0
generate_hc_flag =           0
split_reanalysis_flag =      0
featurize_hc_flag =          0
featurize_ra_flag =          0
compute_climatology_flag =   0
illustrate_dataset_flag =    1
featurize_for_dga_ra_flag =  0
featurize_for_dga_hc_flag =  0
cluster_flag =               0
# ------------------------------------------------------------

# ----------- Set some physical parameters -----
dt_samp = 0.5
dt_szn = 0.74
szn_start = 300.0
szn_length = 250.0
year_length = 400.0
Nt_szn = int(szn_length / dt_szn)
szn_avg_window = 5.0
burnin_time = 500.0 


ndelay = 3

fundamental_param_dict = dict({"b": 0.5, "beta": 1.25, "gamma_limits": [0.15, 0.22], "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": year_length})
crom = model_crommelin_seasonal.SeasonalCrommelinModel(fundamental_param_dict)
# ----------------------------------------------

# ------------ Create reanalysis ---------------
traj_filename = join(ra_dir_contiguous,"crom_long.nc")
if integrate_flag:
    x0 = np.zeros((1,7))
    x0[0,6] = (1957 + 0.2)*fundamental_param_dict["year_length"]  # Starting time is about 20% of the way through 1957 (arbitrary)
    dt_save = 0.5
    tmax_save = 100*fundamental_param_dict["year_length"] + burnin_time
    t_save = np.arange(0,tmax_save,dt_samp)
    crom.integrate_and_save(x0,t_save,traj_filename,burnin_time=burnin_time)
    print(f"Done integrating")
if plot_integration_flag:
    crom.plot_integration(traj_filename,results_dir)
    print(f"Done plotting")
# -----------------------------------------------

# ---------- Split reanalysis into chunks representing seasons -------
if split_reanalysis_flag:
    crom.split_long_integration(traj_filename,ra_dir_seasonal,szn_start,szn_length)
# -----------------------------------------------

# ------------ Create hindcasts ----------------------
if generate_hc_flag:
    t_abs_range = fundamental_param_dict["year_length"]*np.array([1960,1970])
    crom.generate_hindcast_dataset(traj_filename,hc_dir,t_abs_range,dt_samp,ens_size=30,ens_duration=47,ens_gap=13,pert_scale=0.01)
# ----------------------------------------------

# ------------ Use reanalysis to define features --------
# For example, compute EOFs from the database of raw files
# -----------------------------------------------------


featspec_filename = join(featspec_dir,"featspec")
feat_crom = feature_crommelin.SeasonalCrommelinModelFeatures(featspec_filename,szn_start,szn_length,year_length,Nt_szn,szn_avg_window,dt_szn,delaytime=0)
# ------- Evaluate TPT features on reanalysis  --------
if featurize_ra_flag:
    rafiles = os.listdir(ra_dir_seasonal)
    raw_filename_list = [join(ra_dir_seasonal,f) for f in rafiles]
    save_filename = join(results_dir_ra,"X.nc")
    feat_crom.evaluate_features_database(raw_filename_list,save_filename)
# ----------------------------------------------

# ------ Evaluate TPT features on hindcasts --------
if featurize_hc_flag:
    hcfiles = os.listdir(hc_dir)
    raw_filename_list = [join(hc_dir,f) for f in hcfiles]
    save_filename = join(results_dir_hc,"X.nc")
    feat_crom.evaluate_features_database(raw_filename_list,save_filename)
# ----------------------------------------------

# ---------- Compute climatological statistics that will be neeed for the downstream DGA step. Use reanalysis for this, in order to define objective values --------
if compute_climatology_flag:
    in_filename = join(results_dir_ra,"X.nc")
    save_filename = join(results_dir_ra,"Xclim.nc")
    feat_crom.compute_climatology(in_filename,save_filename)
# -----------------------------------------------------------------------------
# ---------- Display features in reanalysis and hindcast data ---------------
if illustrate_dataset_flag:
    Xra_filename = join(results_dir_ra,"X.nc")
    Xhc_filename = join(results_dir_hc,"X.nc")
    Xclim_filename = join(results_dir_ra,"Xclim.nc")
    szns2illustrate = np.arange(1960,1970)
    feat_crom.illustrate_dataset(Xra_filename,Xhc_filename,results_dir,szns2illustrate,Xclim_filename)
# --------------------------------------------------------

# --------- Evaluate DGA features (i.e., to use for clustering) on reanalysis ----------------------
# Directly derivable from the X file 
if featurize_for_dga_ra_flag:
    input_filename = join(results_dir_ra,"X.nc")
    output_filename = join(results_dir_ra,"Y.nc")
    Xclim_filename = join(results_dir_ra,"Xclim.nc")
    feat_crom.evaluate_features_for_dga(input_filename,output_filename,Xclim_filename,ndelay=ndelay,inverse=False)
    # Invert to make sure
    input_filename = join(results_dir_ra,"Y.nc")
    output_filename = join(results_dir_ra,"XfromY.nc")
    feat_crom.evaluate_features_for_dga(input_filename,output_filename,Xclim_filename,ndelay=ndelay,inverse=True)
    
if featurize_for_dga_hc_flag:
    input_filename = join(results_dir_hc,"X.nc")
    output_filename = join(results_dir_hc,"Y.nc")
    Xclim_filename = join(results_dir_ra,"Xclim.nc")
    feat_crom.evaluate_features_for_dga(input_filename,output_filename,Xclim_filename,ndelay=ndelay,inverse=False)
    # Invert to make sure
    print(f"Beginning inversion of HC data")
    input_filename = join(results_dir_hc,"Y.nc")
    output_filename = join(results_dir_hc,"XfromY.nc")
    feat_crom.evaluate_features_for_dga(input_filename,output_filename,Xclim_filename,ndelay=ndelay,inverse=True)
# Now illustrate the normalized dataset
Xra_filename = join(results_dir_ra,"Y.nc")
Xhc_filename = join(results_dir_hc,"Y.nc")
Xclim_filename = join(results_dir_ra,"Xclim.nc")
szns2illustrate = np.arange(1960,1970)
feat_crom.illustrate_dataset(Xra_filename,Xhc_filename,results_dir,szns2illustrate,Xclim_filename,plot_climatology=False,suffix="_norm")
# Now illustrate the re-normalized dataset to make sure it matches the original
Xra_filename = join(results_dir_ra,"XfromY.nc")
Xhc_filename = join(results_dir_hc,"XfromY.nc")
Xclim_filename = join(results_dir_ra,"Xclim.nc")
szns2illustrate = np.arange(1960,1970)
feat_crom.illustrate_dataset(Xra_filename,Xhc_filename,results_dir,szns2illustrate,Xclim_filename,plot_climatology=True,suffix="_unnorm")
# --------------------------------------------------------------------------------------------------


# -----  Cluster TPT features day by day --------
if cluster_flag:
    pass
    
# ----------------------------------------------

# ------- Build the MSM on hindcast data ---------

# ----------------------------------------------

# ------- Perform TPT pipeline on hindcasts -----

# ----------------------------------------------

# ----- Plot TPT results ------------

# ------------------------------

# ----- Plot rates ----------------

# -------------------------------



