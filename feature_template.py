# Provide a parent class for features of any given system. Most important: definitions of A and B, time-delay embedding, and maps from features to indices. 
# The data will be in the form of Xarray DataSets, with a different variable for each feature, as well as appropriate metadata. 
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn.cluster import KMeans
from numpy import save,load
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
import sys
from abc import ABC,abstractmethod
import tpt_utils

class SeasonalFeatures(ABC):
    def __init__(self):
        self.set_ab_code()
        super().__init__()
        return
    @abstractmethod
    def set_ab_code(self, *args, **kwargs):
        """
        Create a mapping from sets of interest (e.g., A, B, D) to 
        integers for an efficient encoding. 
        """
        pass
    @abstractmethod
    def set_ab_boundaries(self, *args, **kwargs):
        """
        Sets the specific boundaries of A and B
        """
        pass
    def set_event_seasonal_params(self,espd):
        # Set the time parameters for the season: its starting date, its duration (the season may overlap two calendar years), and the size of the time window over which to take seasonal averages.
        # TODO: Adjust for leap years somehow. Maybe we fix a universal start data, like Jan 1, and specify time relative to that. 
        self.szn_start = espd["szn_start"] 
        self.szn_length = espd["szn_length"] 
        self.year_length = espd["year_length"]
        self.Nt_szn = espd["Nt_szn"] # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.dt_szn = self.szn_length/self.Nt_szn
        self.t_szn_edge = np.linspace(0,self.szn_length,self.Nt_szn+1) #+ self.szn_start
        self.t_szn_cent = 0.5*(self.t_szn_edge[:-1] + self.t_szn_edge[1:])
        return
    def time_conversion_from_absolute(self,t_abs):
        """
        Parameters
        -----------
        t_abs: array_like
            Absolute time in days since year 0, day 0

        Returns 
        -------
        szn_start_year: the calendar year of the most recent season beginning
        t_cal: the calendar year of the current time
        t_szn: the time since the most recent season beginning
        ti_szn: the time index of the 
        """
        year = (t_abs / self.year_length).astype(int)
        t_cal = t_abs - year*self.year_length
        szn_start_year = year*(t_cal >= self.szn_start) + (year-1)*(t_cal < self.szn_start)
        t_szn = t_abs - (szn_start_year*self.year_length + self.szn_start)
        ti_szn = (t_szn / self.dt_szn).astype(int)
        return szn_start_year,t_cal,t_szn,ti_szn
    @abstractmethod
    def ab_test(self,Xtpt):
        """
        Parameters
        ----------
        Xtpt: xarray.DataArray
            dimensions (t_init, member, time, feature)
        tpt_bndy: dict
            Parameters that specify the parameters in the definition of A and B

        Returns
        -------
        ab_tag: xarray.DataArray
            dimensions (t_init, member, time)
            The integer corresponding to the state space location of each coordinate. E.g., wherever the input data Xtpt is in A, ab_tag contains ab_code["A"]
        """
        pass
    def cotton_eye_joe(self, Xtpt, ab_tag, mode):
        if mode == "timechunks":
            return self.cotton_eye_joe_timechunks(Xtpt, ab_tag)
        elif mode == "timesteps":
            return self.cotton_eye_joe_timesteps(Xtpt, ab_tag)
        else:
            raise Exception(f"You asked for a mode of {mode}, but I only accept 'timechunks' or 'timesteps'")
    
    def cotton_eye_joe_timesteps(self, Xtpt, ab_tag):
        cej_coords = {key: Xtpt.coords[key].to_numpy() for key in list(Xtpt.coords.keys())}
        cej_coords.pop("feature")
        cej_coords["sense"] = ["since","until"]
        cej_coords["state"] = ["A","B"]
        cej = xr.DataArray(
            coords = cej_coords,
            dims = list(cej_coords.keys()),
            data = np.nan,
        )
        # Forward pass through time
        for i_time in np.arange(cej["t_sim"].size):
            if i_time % 200 == 0:
                print(f"Forward pass: through time {i_time} out of {cej['t_sim'].size}")
            for state in ["A","B"]:
                if i_time > 0:
                    cej[dict(t_sim=i_time)].loc[dict(sense="since",state=state)] = (
                        cej.isel(t_sim=i_time-1).sel(sense="since",state=state).data +
                        cej["t_sim"][i_time].data - cej["t_sim"][i_time-1].data
                    )
                state_flag = (ab_tag.isel(t_sim=i_time) == self.ab_code[state])
                # Wherever the state is achieved at this time slice, set the time since to zero
                cej[dict(t_sim=i_time)].loc[dict(sense="since",state=state)] = (
                    (xr.zeros_like(cej.isel(t_sim=i_time).sel(sense="since",state=state))).where(
                    state_flag, cej.isel(t_sim=i_time).sel(sense="since",state=state))
                )
        # Backward pass through time
        for i_time in np.arange(cej["t_sim"].size-1,-1,-1):
            if i_time % 200 == 0:
                print(f"Backward pass: through time {i_time} out of {cej['t_sim'].size}")
            for state in ["A","B"]:
                if i_time < cej["t_sim"].size-1:
                    cej[dict(t_sim=i_time)].loc[dict(sense="until",state=state)] = (
                        cej.isel(t_sim=i_time+1).sel(sense="until",state=state).data +
                        cej["t_sim"][i_time+1].data - cej["t_sim"][i_time].data
                    )
                state_flag = (ab_tag.isel(t_sim=i_time) == self.ab_code[state])
                cej[dict(t_sim=i_time)].loc[dict(sense="until",state=state)] = (
                    (xr.zeros_like(cej.isel(t_sim=i_time).sel(sense="until",state=state))).where(
                    state_flag, cej.isel(t_sim=i_time).sel(sense="until",state=state))
                )
        return cej
    # Function to find the time since and until hitting A and B
    def cotton_eye_joe_timechunks(self, Xtpt, ab_tag):
        # TODO: expand dims of E5 to include t_init and member
        cej_coords = {key: Xtpt.coords[key].to_numpy() for key in list(Xtpt.coords.keys())}
        cej_coords.pop("feature")
        cej_coords["sense"] = ["since","until"]
        cej_coords["state"] = ["A","B"]
        cej = xr.DataArray(
            coords = cej_coords,
            dims = list(cej_coords.keys()),
            data = np.nan,
        )
        t_sim = Xtpt["t_sim"].data
        print(f"t_sim.shape = {t_sim.shape}")
        Nt = t_sim.size
        # Forward pass through time
        for t_init in Xtpt.coords["t_init"]:
            for member in Xtpt.coords["member"]:
                for state in ["A","B"]:
                    indicator = (ab_tag.sel(t_init=t_init,member=member) == self.ab_code[state]).data.astype(int)
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
                    cej.loc[dict(t_init=t_init,member=member,state=state,sense="since")] = tsince
                    cej.loc[dict(t_init=t_init,member=member,state=state,sense="until")] = tuntil
        return cej
    def estimate_empirical_committor(self, cej):
        # Make points NaN if hanging endpoints
        cej_infed = xr.where(np.isnan(cej)==0, cej, np.inf)
        comm_emp = 1.0*(cej_infed.sel(state="B") < cej_infed.sel(state="A")) + 0.5*(cej_infed.sel(state="B") == cej_infed.sel(state="A"))
        comm_emp.loc[dict(sense="since")] = 1 - comm_emp.sel(sense="since")
        return comm_emp
    def estimate_rate(self, cej, comm_emp):
        # This will be a lower bound on the rate, because of hanging endpoints. 
        a2b_flag = 1.0 * (comm_emp.sel(sense="since") == self.ab_code["A"]) * (comm_emp.sel(sense="until") == self.ab_code["B"])
        rate_lowerbound = (1.0*a2b_flag.diff(dim="t_sim")==1).sum(dim="t_sim") / (a2b_flag["t_sim"][-1] - a2b_flag["t_sim"][0])
        return rate_lowerbound
    def get_seasonal_statistics(self, feat_da, t_szn):
        # Get the statistics in each box of the season. This will be used to normalize the dataset later.
        szn_stats = dict()
        field = feat_da.stack(sample=("t_init","member","t_sim")).transpose("sample","feature").to_numpy()
        weights = np.ones_like(field)
        features = t_szn.stack(sample=("t_init","member","t_sim")).to_numpy().reshape(-1,1)
        szn_stats_dict,edges,centers = (
            tpt_utils.project_field(
                field, weights, features, 
                bounds = self.t_szn_edge[[0,-1]].reshape((2,1)),
                shp = (self.Nt_szn,)
            )
        )
        szn_stats = xr.Dataset(
            data_vars = dict({
                key: xr.DataArray(
                    coords={"t_szn_cent": centers[0], "feature": feat_da["feature"],},
                    data=szn_stats_dict[key],
                    dims=["t_szn_cent", "feature"],
                ) 
                for key in list(szn_stats_dict.keys())
            }),
        )
        return szn_stats
    def unseason(self, feat_msm, szn_stats, t_obs, max_delay):
        feat_msm_normalized = np.nan*xr.ones_like(feat_msm)
        szn_window = (t_obs.sel(feature="t_szn")/self.dt_szn).astype(int)
        szn_start_year = t_obs.sel(feature="year_szn_start").astype(int)
        for i_win in range(self.Nt_szn):
            feat_msm_normalized = xr.where(
                szn_window==i_win, 
                (feat_msm - szn_stats["mean"].isel(t_szn_cent=i_win,drop=True)) / szn_stats["std"].isel(t_szn_cent=i_win,drop=True), 
                feat_msm_normalized
            )
        # --------------- Mark the trajectories that originated in an earlier time window and will reach another time window ---------------
        traj_ending_flag = (
            (szn_window == szn_window.isel(t_sim=-1,drop=True)) *
            (szn_start_year == szn_start_year.isel(t_sim=-1,drop=True))
        ) > 0
        traj_beginning_flag = (
            (szn_window == szn_window.isel(t_sim=0,drop=True)) *
            (szn_start_year == szn_start_year.isel(t_sim=0,drop=True)) +
            (feat_msm_normalized["t_sim"] < feat_msm_normalized["t_sim"][max_delay+2]) # If a trajectory is not old enough yet, may as well be Nan
        ) > 0
        # -----------------------------------------------------------------------------------------------
        return szn_window,szn_start_year,traj_beginning_flag,traj_ending_flag,feat_msm_normalized
    def cluster(self, feat_msm_normalized, t_obs, szn_window, traj_beginning_flag, traj_ending_flag, km_seed): 
        km_assignment = -np.ones((t_obs["t_init"].size,t_obs["member"].size,t_obs["t_sim"].size), dtype=int)
        km_centers = []
        km_n_clusters = -np.ones(self.Nt_szn, dtype=int)
        for i_win in range(self.Nt_szn):
            if i_win % 10 == 0:
                print(f"Starting K-means number {i_win} out of {self.Nt_szn}")
            idx_in_window = np.where(
                (szn_window.data==i_win) * 
                (np.isnan(feat_msm_normalized).sum(dim="feature") == 0)
            ) # All the data in this time window
            # idx_for_clustering is all the data that we're allowed to use to build the KMeans object
            if i_win == 0:
                idx_for_clustering = np.where(
                    (szn_window.data==i_win) *
                    (traj_ending_flag.data == 0) *
                    (np.isnan(feat_msm_normalized).sum(dim="feature") == 0)
                )
            elif i_win == self.Nt_szn-1:
                idx_for_clustering = np.where(
                    (szn_window.data==i_win)*
                    (traj_beginning_flag.data == 0) *
                    (np.isnan(feat_msm_normalized).sum(dim="feature") == 0)
                )            
            else:
                idx_for_clustering = np.where(
                    (szn_window.data==i_win)*
                    (traj_ending_flag.data == 0)*
                    (traj_beginning_flag.data == 0) * 
                    (np.isnan(feat_msm_normalized).sum(dim="feature") == 0)
                )
            if len(idx_for_clustering[0]) == 0:
                raise Exception(f"At window {i_win}, there are no indices fit to cluster")
            km_n_clusters[i_win] = min(200,max(1,len(idx_for_clustering[0]//2)))
            km = KMeans(n_clusters=km_n_clusters[i_win],random_state=km_seed).fit(
                    feat_msm_normalized.data[idx_for_clustering])
            km_assignment[idx_in_window] = km.predict(feat_msm_normalized.data[idx_in_window]) 
            km_centers += [km.cluster_centers_]
        km_assignment_da = xr.DataArray(
            coords={"t_init": t_obs["t_init"], "member": t_obs["member"], "t_sim": t_obs["t_sim"]},
            dims=["t_init","member","t_sim"],
            data=km_assignment.copy()    #np.zeros((feat_tpt["ensemble"].size,feat_tpt["member"].size,feat_tpt["t_sim"].size), dtype=int)
        )    
        return km_assignment_da,km_centers,km_n_clusters
    def construct_transition_matrices(self, km_assignment, km_n_clusters, szn_window, szn_start_year):
        time_dim = list(szn_window.dims).index("t_sim")
        nontime_dims = np.setdiff1d(np.arange(len(szn_window.dims)), [time_dim])
        P_list = []
        for i_win in range(self.Nt_szn-1):
            if i_win % 10 == 0: print(f"i_win = {i_win}")
            P = np.zeros((km_n_clusters[i_win],km_n_clusters[i_win+1]))
            # Count the trajectories that passed through both box i during window i_win, and box j during window i_win+1. 
            # Maybe some trajectories will be double counted. 
            idx_pre = np.where(szn_window.data==i_win)
            idx_post = np.where(szn_window.data==i_win+1)
            overlap = np.where(
                np.all(
                    np.array([
                        (np.subtract.outer(idx_pre[dim], idx_post[dim]) == 0) 
                        for dim in nontime_dims
                    ]), axis=0
                ) * (
                    np.subtract.outer(
                        szn_start_year.data[idx_pre], szn_start_year.data[idx_post]
                    ) == 0
                )          
            )
            idx_pre_overlap = tuple([idx_pre[dim][overlap[0]] for dim in range(len(idx_pre))])
            idx_post_overlap = tuple([idx_post[dim][overlap[1]] for dim in range(len(idx_pre))])
            km_pre = km_assignment.data[idx_pre_overlap]
            km_post = km_assignment.data[idx_post_overlap]
            tinit_member_year_identifier = np.concatenate((
                np.array(idx_pre_overlap)[nontime_dims,:], 
                [szn_start_year.data[idx_pre_overlap]]
            ), axis=0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    traj_idx, = np.where((km_pre==i)*(km_post==j))
                    P[i,j] = np.unique(tinit_member_year_identifier[:,traj_idx], axis=1).shape[1]
            min_rowsum = np.min(np.sum(P, axis=1))
            min_colsum = np.min(np.sum(P, axis=0))
            if min_rowsum == 0 or min_colsum == 0:
                raise Exception(f"Under-filled transition matrices between seasonal windows {i_win} and {i_win+1}. min_rowsum = {min_rowsum} and min_colsum = {min_colsum}")
            # Normalize the matrix
            P = np.diag(1.0/np.sum(P, axis=1)).dot(P)
            P_list += [P]
        return P_list
    def broadcast_field_msm2dataarray(self, msm, field_msm, szn_stats, density_flag=False):
        field_da = np.zeros(msm["szn_window"].shape)
        for i_win in range(msm["Nt_szn"]):
            idx_in_window = np.where(msm["szn_window"].data == i_win)
            for i_clust in range(msm["km_n_clusters"][i_win]):
                idx_in_cluster = np.where(msm["km_assignment"].data[idx_in_window] == i_clust)
                idx_in_window_and_cluster = tuple([idx_in_window[dim][idx_in_cluster] for dim in range(len(idx_in_window))])
                field_da[idx_in_window_and_cluster] = field_msm[i_win][i_clust]
                if density_flag and (len(idx_in_window_and_cluster[0]) > 0):
                    field_da[idx_in_window_and_cluster] *= 1.0/len(idx_in_window_and_cluster[0])
        da_broadcast = xr.DataArray(
            coords = msm["szn_window"].coords,
            dims = msm["szn_window"].dims, 
            data = field_da,
        )
        return da_broadcast

