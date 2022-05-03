# Create a minimal demonstration of the DGA method. This does not save all the intermediate results, as the main code does, because it's too intensive. 
# Possibly allow for heterogeneous ensemble sizes and lengths 
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt 
import model_crommelin_seasonal
import sys
import os
from os import mkdir
from os.path import join,exists

# ----------- Create directory to save results ----------
topic_dir = "/scratch/jf4241/crommelin"
if not exists(topic_dir): mkdir(topic_dir)
day_dir = join(topic_dir,"2022-05-03")
if not exists(day_dir): mkdir(day_dir)
exp_dir = join(day_dir,"1")
if not exists(exp_dir): mkdir(exp_dir)
ra_dir = join(exp_dir,"reanalysis_data")
if not exists(ra_dir): mkdir(ra_dir)
hc_dir = join(exp_dir,"hindcast_data")
if not exists(hc_dir): mkdir(hc_dir)
results_dir = join(exp_dir,"results")
if not exists(results_dir): mkdir(results_dir)

integrate_flag =             1
plot_integration_flag =      1

# ------------ Create reanalysis ---------------
fundamental_param_dict = dict({"b": 0.5, "beta": 1.25, "gamma_limits": [0.15, 0.22], "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400})
crom = model_crommelin_seasonal.SeasonalCrommelinModel(fundamental_param_dict)
traj_filename = join(ra_dir,"crom_long.nc")
if integrate_flag:
    x0 = np.zeros((1,7))
    x0[0,6] = (1957 + 0.2)*fundamental_param_dict["year_length"] 
    dt_save = 0.5
    tmax_save = 40500
    Nt_save = int(tmax_save/dt_save) + 1
    t_save = np.linspace(0,tmax_save,Nt_save)
    crom.integrate_and_save(x0,t_save,traj_filename,burnin_time=500)
    print(f"Done integrating")
if plot_integration_flag:
    crom.plot_integration(traj_filename,results_dir)
    print(f"Done plotting")


# ----------------------------------------------

# ------------ Create hindcasts ----------------------

# ----------------------------------------------

# ------------ Use reanalysis to define features --------

# -----------------------------------------------------

# ---------- Display features in reanalysis ---------------

# --------------------------------------------------------

# ------ Evaluate TPT features on reanalysis --------

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



