# TPT class for GENERAL stratosphere data: hindcasts or reanalysis. No explicit feature-handling here, just take in numpy arrays and do the magic. Except for time and the feature that gives distances to A and B. 

import numpy as np
import netCDF4 as nc
import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'serif', 'size': 12}
font = {'family': 'serif', 'size': 18}
bigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import sys
import os
from os import mkdir
from os.path import join,exists
from sklearn import linear_model
import helper
import cartopy
from cartopy import crs as ccrs
import pickle

class WinterStratosphereTPT:
    def __init__(self):
        # physical_params is a dictionary that tells us (1) how to tell time, and (2) how to assess distance from A and B
        return
    def set_boundaries(self,tpt_bndy):
        self.tpt_bndy = tpt_bndy
        return
    # Everything above is just about defining the boundary value problem. Everything below will be analysis with respect to a specific data set. 
    def tpt_pipeline_dns(self,expdir,savedir,winstrat,feat_def):
        # Call winstrat and anything else we want. 
        # savedir: where all the results of this particular TPT will be saved.
        # winstrat: the object that was used to create features, and that can go on to evaluate other features.
        # feat_def: the feature definitions that came out of winstrat.
        X = np.load(join(expdir,"X.npy"))
        Nx,Nt,xdim = X.shape
        funlib = winstrat.observable_function_library()
        # ---- Plot committor in a few different coordinates -----
        src_tag,dest_tag = winstrat.compute_src_dest_tags(X,feat_def,self.tpt_bndy,"src_dest")
        qp = 1.0*(dest_tag == 1).flatten()
        weight = np.ones(Nx*Nt)/(Nx*Nt)
        keypairs = [['time_d','uref'],['time_d','mag1'],['time_d','lev0_pc0'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4'],['time_d','mag1_anomaly'],['time_d','mag2_anomaly']]
        for i_kp in range(len(keypairs)):
            fun0name,fun1name = [funlib[keypairs[i_kp][j]]["label"] for j in range(2)]
            theta_x = np.array([funlib[keypairs[i_kp][j]]["fun"](X.reshape((Nx*Nt,xdim))) for j in range(2)]).T
            fig,ax = helper.plot_field_2d(qp,weight,theta_x,shp=[15,15],fieldname="Committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
            fig.savefig(join(savedir,"qp_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
            plt.close(fig)
        # Compute density and backward committor
        # Compute rate
        # Compute lead time
        # Compute current (??)
        return
    def tpt_pipeline_dga(self,savedir,file_list,winstrat,feat_def):
        # Do the DGA pipeline. 
        return
    def compute_rate_direct(self,src_tag,dest_tag):
        # This is meant for full-winter trajectories
        src_tag = src_dest['src_tag']
        dest_tag = src_dest['dest_tag']
        ab_tag = (src_tag==0)*(dest_tag==1)
        absum = 1.0*(np.sum(ab_tag,axis=1) > 0)
        rate = np.mean(absum)
        return rate



