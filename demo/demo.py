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
day_dir = join(topic_dir,"2022-05-04")
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

integrate_flag =             0
plot_integration_flag =      0
generate_hc_flag =           0
split_reanalysis_flag =      0
calculate_climatology_flag = 0
featurize_hc_flag =          1

# ----------- Set some physical parameters -----
dt_samp = 0.5
dt_szn = 0.74
szn_start = 300.0
szn_length = 250.0
Nt_szn = int(szn_length / dt_szn)
szn_avg_window = 5.0

fundamental_param_dict = dict({"b": 0.5, "beta": 1.25, "gamma_limits": [0.15, 0.22], "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400})
crom = model_crommelin_seasonal.SeasonalCrommelinModel(fundamental_param_dict)
# ----------------------------------------------

# ------------ Create reanalysis ---------------
traj_filename = join(ra_dir_contiguous,"crom_long.nc")
if integrate_flag:
    x0 = np.zeros((1,7))
    x0[0,6] = (1957 + 0.2)*fundamental_param_dict["year_length"] 
    dt_save = 0.5
    tmax_save = 40500
    t_save = np.arange(0,tmax_save,dt_samp)
    crom.integrate_and_save(x0,t_save,traj_filename,burnin_time=500)
    print(f"Done integrating")
if plot_integration_flag:
    crom.plot_integration(traj_filename,results_dir)
    print(f"Done plotting")
# -----------------------------------------------

# ---------- Split reanalysis into chunks -------
if split_reanalysis_flag:
    crom.split_long_integration(traj_filename,ra_dir_seasonal,szn_start,szn_length)
# -----------------------------------------------

# ------------ Create hindcasts ----------------------
if generate_hc_flag:
    t_abs_range = fundamental_param_dict["year_length"]*np.array([1960,1970])
    crom.generate_hindcast_dataset(traj_filename,hc_dir,t_abs_range,dt_samp,ens_size=10,ens_duration=47,ens_gap=13,pert_scale=0.001)
# ----------------------------------------------

if calculate_climatology_flag:
    # Load dataset and compute etc.
    print("Done calculating climatology")

# ------------ Use reanalysis to define features --------

# -----------------------------------------------------

# ---------- Display features in reanalysis ---------------

# --------------------------------------------------------

# ------ Evaluate TPT features on reanalysis --------
featspec_filename = join(featspec_dir,"featspec")
feat_crom = feature_crommelin.SeasonalCrommelinModelFeatures(featspec_filename,szn_start,szn_length,Nt_szn,szn_avg_window,dt_szn,delaytime=0)
if featurize_hc_flag:
    hcfiles = os.listdir(hc_dir)
    raw_filename_list = [join(hc_dir,f) for f in hcfiles]
    save_filename = join(results_dir_ra,"X")
    feat_crom.evaluate_features_database(raw_filename_list,save_filename)
# ----------------------------------------------

# ------- Evaluate TPT features on hindcasts --------

# ----------------------------------------------

# -----  Cluster TPT features day by day --------

# ----------------------------------------------

# ------- Build the MSM on hindcast data ---------

# ----------------------------------------------

# ------- Perform TPT pipeline on hindcasts -----

# ----------------------------------------------

# ----- Plot TPT results ------------

# ------------------------------

# ----- Plot rates ----------------

# -------------------------------



