# This file contains the core functions to do TPT analysis
# Mostly having to do with manipulating MSMs to get quantities of interest

import numpy as np
import xarray as xr
import netCDF4 as nc
import model_crommelin_seasonal
import feature_crommelin 
from importlib import reload
import sys 
import os
from os import mkdir, makedirs
from os.path import join,exists
from importlib import reload
import pickle
import helper2
from sklearn.cluster import KMeans, MiniBatchKMeans
from datetime import datetime

def abtest(Xtpt, tpt_bndy):
    # Given a snapshot of an instance of the feat_tpt data structure, return ab_tag:
    # 0 means in A, 1 means in B, and 2 means neither.
    # The definition of A and B will be parameterized by a dictionary, tpt_bndy, which specifies
    # the time of season when blockings can happen as well as the thresholds for A and B.
    time_window_flag = 1.0*(
        Xtpt.sel(feature="t_szn") >= tpt_bndy["tthresh"][0])*(
        Xtpt.sel(feature="t_szn") <= tpt_bndy["tthresh"][1]
    )
    blocked_flag = 1.0*(Xtpt.sel(feature="x1") <= tpt_bndy["x1thresh"][0])
    zonal_flag = 1.0*(Xtpt.sel(feature="x1") >= tpt_bndy["x1thresh"][1])
    ab_tag = (
        abcode["A"]*((1*(time_window_flag == 0) + 1*zonal_flag) > 0) +
        abcode["B"]*(time_window_flag*blocked_flag) +
        abcode["D"]*(time_window_flag*(blocked_flag==0)*(zonal_flag==0))
    )
    return ab_tag

def cotton_eye_joe(Xtpt, tpt_bndy, ab_tag, mode):
    if mode == "timechunks":
        return cotton_eye_joe_timechunks(Xtpt, tpt_bndy, ab_tag)
    elif mode == "timesteps":
        return cotton_eye_joe_timesteps(Xtpt, tpt_bndy, ab_tag)
    else:
        raise Exception(f"You asked for a mode of {mode}, but I only accept 'timechunks' or 'timesteps'")

def cotton_eye_joe_timesteps(Xtpt, tpt_bndy, ab_tag):
    sintil = xr.DataArray(
        coords = dict({
            "ensemble": Xtpt.coords["ensemble"],
            "member": Xtpt.coords["member"],
            "t_sim": Xtpt.coords["t_sim"],
            "sense": ["since","until"],
            "state": ["A","B"]
        }),
        data = np.nan*np.ones((Xtpt["ensemble"].size, Xtpt["member"].size, Xtpt["t_sim"].size, 2, 2)),
        dims = ["ensemble","member","t_sim","sense","state"],
    )
    # Forward pass through time
    for i_time in np.arange(sintil["t_sim"].size):
        if i_time % 200 == 0:
            print(f"Forward pass: through time {i_time} out of {sintil['t_sim'].size}")
        for state in ["A","B"]:
            if i_time > 0:
                sintil[dict(t_sim=i_time)].loc[dict(sense="since",state=state)] = (
                    sintil.isel(t_sim=i_time-1).sel(sense="since",state=state).data +
                    sintil["t_sim"][i_time].data - sintil["t_sim"][i_time-1].data
                )
            state_flag = (ab_tag.isel(t_sim=i_time) == abcode[state])
            # Wherever the state is achieved at this time slice, set the time since to zero
            sintil[dict(t_sim=i_time)].loc[dict(sense="since",state=state)] = (
                (xr.zeros_like(sintil.isel(t_sim=i_time).sel(sense="since",state=state))).where(
                state_flag, sintil.isel(t_sim=i_time).sel(sense="since",state=state))
            )
    # Backward pass through time
    for i_time in np.arange(sintil["t_sim"].size-1,-1,-1):
        if i_time % 200 == 0:
            print(f"Backward pass: through time {i_time} out of {sintil['t_sim'].size}")
        for state in ["A","B"]:
            if i_time < sintil["t_sim"].size-1:
                sintil[dict(t_sim=i_time)].loc[dict(sense="until",state=state)] = (
                    sintil.isel(t_sim=i_time+1).sel(sense="until",state=state).data +
                    sintil["t_sim"][i_time+1].data - sintil["t_sim"][i_time].data
                )
            state_flag = (ab_tag.isel(t_sim=i_time) == abcode[state])
            sintil[dict(t_sim=i_time)].loc[dict(sense="until",state=state)] = (
                (xr.zeros_like(sintil.isel(t_sim=i_time).sel(sense="until",state=state))).where(
                state_flag, sintil.isel(t_sim=i_time).sel(sense="until",state=state))
            )
    return sintil


# Function to find the time since and until hitting A and B
def cotton_eye_joe_timechunks(Xtpt, tpt_bndy, ab_tag):
    sintil = xr.DataArray(
        coords = dict({
            "ensemble": Xtpt.coords["ensemble"],
            "member": Xtpt.coords["member"],
            "t_sim": Xtpt.coords["t_sim"],
            "sense": ["since","until"],
            "state": ["A","B"]
        }),
        data = np.nan*np.ones((Xtpt["ensemble"].size, Xtpt["member"].size, Xtpt["t_sim"].size, 2, 2)),
        dims = ["ensemble","member","t_sim","sense","state"],
    )
    t_sim = Xtpt["t_sim"].data
    print(f"t_sim.shape = {t_sim.shape}")
    Nt = t_sim.size
    # Forward pass through time
    for ensemble in Xtpt.coords["ensemble"]:
        for member in Xtpt.coords["member"]:
            for state in ["A","B"]:
                indicator = (ab_tag.sel(ensemble=ensemble,member=member) == abcode[state]).data.astype(int)
                tsince = np.nan*np.ones(Nt)
                tuntil = np.nan*np.ones(Nt)
                # Fill in zeros inside the set
                tsince[indicator==1] = 0.0
                tuntil[indicator==1] = 0.0
                # Find the crossover points
                idx_exit = np.where(np.diff(indicator) == -1)[0] + 1 # First step outside of state
                idx_entry = np.where(np.diff(indicator) == 1)[0] + 1 # First entry to state
                # Take care of boundary cases
                if (not indicator[0]) and len(idx_entry) > 0:
                    tuntil[:idx_entry[0]] = t_sim[idx_entry[0]] - t_sim[:idx_entry[0]]
                    idx_entry = idx_entry[1:]
                if (not indicator[Nt-1]) and len(idx_exit) > 0:
                    tsince[idx_exit[-1]:] = t_sim[idx_exit[-1]:] - t_sim[idx_exit[-1]-1]
                    idx_exit = idx_exit[:-1]
                # Now the middle components: time intervals between exits and entries
                if len(idx_entry) > 0 and len(idx_exit) > 0:
                    for k in range(len(idx_exit)):
                        i0,i1 = idx_exit[k],idx_entry[k]
                        tsince[i0:i1] = t_sim[i0:i1] - t_sim[i0-1]
                        tuntil[i0:i1] = t_sim[i1] - t_sim[i0:i1]
                sintil.loc[dict(ensemble=ensemble,member=member,state=state,sense="since")] = tsince
                sintil.loc[dict(ensemble=ensemble,member=member,state=state,sense="until")] = tuntil
    return sintil

