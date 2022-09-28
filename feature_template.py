# Provide a parent class for features of any given system. Most important: definitions of A and B, time-delay embedding, and maps from features to indices. 
# The data will be in the form of Xarray DataSets, with a different variable for each feature, as well as appropriate metadata. 
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn.cluster import KMeans
from numpy import save,load
from scipy.linalg import expm,logm
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
        cej_infed = xr.where(np.isnan(cej)==0, cej, np.inf)
        comm_emp = 1.0*(cej_infed.sel(state="B",drop=True) < cej_infed.sel(state="A",drop=True)) + 0.5*(cej_infed.sel(state="B",drop=True) == cej_infed.sel(state="A",drop=True))
        comm_emp.loc[dict(sense="since")] = 1 - comm_emp.sel(sense="since")
        return comm_emp
    def estimate_rate(self, ab_tag, comm_emp):
        # This will be a lower bound on the rate, because of hanging endpoints. 
        a2b_flag = 1.0 * (comm_emp.sel(sense="since") == 1) * (comm_emp.sel(sense="until") == 1)
        a2all_flag = 1.0 * (comm_emp.sel(sense="since") == 1) * (ab_tag == self.ab_code["D"]) 
        # --- Old: divide by time --------`
        #rate_lowerbound = (1.0*a2b_flag.diff(dim="t_sim")==1).sum(dim="t_sim") / (a2b_flag["t_sim"][-1] - a2b_flag["t_sim"][0])
        # ---- New: divide by number of exits from A
        num_a2b = ((1.0*a2b_flag).diff(dim="t_sim")==1).sum(dim="t_sim")
        num_a2all = ((1.0*a2all_flag).diff(dim="t_sim")==1).sum(dim="t_sim")
        rate_lowerbound =  num_a2b / num_a2all
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
                cov_mat_flag = True,
                bounds = self.t_szn_edge[[0,-1]].reshape((2,1)),
                shp = (self.Nt_szn,)
                )
            )
        data_vars = dict({
            key: xr.DataArray(
                coords={"t_szn_cent": centers[0], "feature": feat_da["feature"],},
                data=szn_stats_dict[key],
                dims=["t_szn_cent", "feature"],
                ) 
            for key in ["weightsum","sum","mean","std","q25","q75","min","max"]
            })
        data_vars["cov_mat"] = xr.DataArray(
                coords = {"t_szn_cent": centers[0], "feat0": feat_da["feature"].to_numpy(), "feat1": feat_da["feature"].to_numpy(),},
                data = szn_stats_dict["cov_mat"],
                dims = ["t_szn_cent", "feat0", "feat1"],
                )
        szn_stats = xr.Dataset(data_vars=data_vars)
        return szn_stats
    def build_lim(self, feat_lim, szn_stats_e5, t_obs, max_delay, demean=True, max_lag=3):
        # Build a LIM for the timeseries. 
        lags = np.arange(1,max_lag+1) # Lags to average over
        # First, move the 'feature' index to the end. All the slicing operations will use Numpy for this. 
        feat_lim = feat_lim.transpose("t_init","member","t_sim","feature")
        szn_window = (t_obs.sel(feature="t_szn")/self.dt_szn).astype(int)
        szn_start_year = t_obs.sel(feature="year_szn_start").astype(int)
        szn_mean = szn_stats_e5["mean"]
        Glim = xr.DataArray( # Green's function for LIM
                coords={"t_szn_cent": szn_mean["t_szn_cent"], "feature_new": feat_lim["feature"].values, "feature_old": feat_lim["feature"].values},
                dims=["t_szn_cent","feature_new","feature_old"],
                data=np.nan
                )
        Qlim = xr.DataArray( # Noise covariance matrix
                coords={"t_szn_cent": szn_mean["t_szn_cent"], "feature_new": feat_lim["feature"].values, "feature_old": feat_lim["feature"].values},
                dims=["t_szn_cent","feature_new","feature_old"],
                data=np.nan
                )
        for i_win in range(self.Nt_szn-1):
            # To find idx for the LIM, restrict to indices where there are at least enough lags ahead.
            window_flag = (szn_window == i_win)*(szn_window.shift(t_sim=-1) == i_win+1)
            for i_lag in range(1,len(lags)):
                window_flag = window_flag * np.isfinite(szn_window.shift(t_sim=-lags[i_lag]))
            idx = np.where((window_flag).data)
            # Construct the pair of data matrices, with mean subtracted (this works because `feature' is the last coordinate)
            X = (feat_lim.values[idx] - (1*demean)*szn_mean.isel(t_szn_cent=i_win).values).T
            nfeat,nsamp = X.shape
            if not (nfeat == feat_lim.feature.size and nsamp == len(idx[0])):
                raise Exception("Your dimensions are off for the data matrices. X.shape = {X.shape}")
            # Replace nan with zero 
            nsamp = np.all(np.isfinite(X), axis=0).sum()
            X[np.isnan(X)] = 0
            C_00 = X @ X.T / (nsamp - 1)
            # Loop over various lags
            B = np.nan*np.ones((len(lags), Glim["feature_new"].size, Glim["feature_old"].size)).astype(complex)
            for i_lag in range(len(lags)):
                if i_win + lags[i_lag] < self.Nt_szn:
                    idx_lag = [idx[d].copy() for d in range(len(idx))]
                    idx_lag[feat_lim.dims.index("t_sim")] += lags[i_lag]
                    idx_lag = tuple(idx_lag)
                    Y = (feat_lim.values[idx_lag] - (1*demean)*szn_mean.isel(t_szn_cent=i_win+lags[i_lag]).values).T 
                    Y[np.isnan(Y)] = 0
                    # Construct the transition matrix
                    C_10 = Y @ X.T / (nsamp - 1)
                    G = C_10 @ np.linalg.inv(C_00)
                    if len(lags) > 1: B[i_lag] = logm(G)/lags[i_lag]
            # Average Green's function over the lags
            if len(lags) > 1:
                G = expm(np.nanmean(B, axis=0)).real
            residual = G @ X - Y
            Q = residual @ residual.T / (nsamp - 1) # (feature_new, feature)
            Glim[dict(t_szn_cent=i_win)] = G
            Qlim[dict(t_szn_cent=i_win)] = Q
        lim = xr.Dataset(data_vars={"szn_mean": szn_mean, "G": Glim, "Q": Qlim})
        return lim
    def unseason(self, feat_msm, szn_stats, t_obs, max_delay, divide_by_std_flag=False, whiten_flag=False):
        Nfeat = feat_msm.feature.size
        feat_msm_stacked = (
                feat_msm
                .stack({"sample": ("t_init","t_sim","member")})
                #.transpose("sample","feature")
                )
        feat_msm_normalized = (
                np.nan*xr.ones_like(feat_msm_stacked)
                .rename({"feature": "feature_norm"})
                .assign_coords({"feature_norm": np.arange(Nfeat)})
                )
        t_obs_stacked = (
                t_obs 
                .stack({"sample": ("t_init","t_sim","member")})
                #.transpose("sample","feature")
                )
        szn_window = (t_obs_stacked.sel(feature="t_szn")/self.dt_szn).astype(int)
        szn_start_year = t_obs_stacked.sel(feature="year_szn_start").astype(int)
        # Construct a dataarray containing the normalizing matrix 
        norm_mat = xr.DataArray(coords={"feature_norm": np.arange(Nfeat), "feature": feat_msm["feature"]}, dims=["feature_norm","feature"])
        for i_win in range(self.Nt_szn):
            if i_win % 50 == 0:
                print(f"i_win = {i_win} out of {self.Nt_szn}")
            idx, = np.where(szn_window.to_numpy() == i_win)
            demean = (feat_msm_stacked.isel(sample=idx) - szn_stats["mean"].isel(t_szn_cent=i_win))
            if whiten_flag:
                cov_mat = szn_stats["cov_mat"].isel(t_szn_cent=i_win,drop=True).to_numpy()
                eigval,eigvec = np.linalg.eigh(cov_mat)
                order = np.argsort(eigval)[::-1]
                eigval = eigval[order]
                eigvec = eigvec[:,order]
                norm_mat[:] = np.diag(1/np.sqrt(eigval)).dot(eigvec.T)
            elif divide_by_std_flag:
                norm_mat[:] = np.diag(1/szn_stats["std"].isel(t_szn_cent=i_win,drop=True).to_numpy())
            else:
                norm_mat[:] = np.eye(feat_msm['feature'].size)
            data_standardized = xr.dot(norm_mat, demean)
            #print(f"feat_msm.shape = {feat_msm.shape}")
            #print(f"szn_stats mean.shape = {szn_stats['mean'].shape}")
            #print(f"ds shape = {data_standardized.shape}")
            feat_msm_normalized[dict(sample=idx)] = data_standardized
            #feat_msm_normalized = xr.where(
            #    szn_window==i_win, 
            #    data_standardized,
            #    #(feat_msm - szn_stats["mean"].isel(t_szn_cent=i_win,drop=True)) / szn_stats["std"].isel(t_szn_cent=i_win,drop=True), 
            #    feat_msm_normalized
            #)
            #print(f"data standard dims = \n{data_standardized.dims}\n fmn dims = \n{feat_msm_normalized.dims}; fmn coords = \n{feat_msm_normalized.coords}")
            num_in_window = np.sum(szn_window==i_win)
        szn_window = szn_window.unstack(dim="sample").transpose("t_init","member","t_sim") #.transpose(dim_order_nofeat)
        szn_start_year = szn_start_year.unstack(dim="sample").transpose("t_init","member","t_sim")
        print(f"two")
        feat_msm_normalized = feat_msm_normalized.rename({"feature_norm": "feature"}).unstack(dim="sample").transpose("t_init","member","t_sim","feature")
        print(f"fmn coords = {feat_msm_normalized.coords}")
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
    def cluster(self, feat_msm_normalized, t_obs, szn_window, traj_beginning_flag, traj_ending_flag, km_seed, num_clusters): 
        print(f"shapes: ")
        print(f"feat_msm_normalized: {feat_msm_normalized.shape}")
        print(f"t_obe: {t_obs.shape}")
        print(f"szn_window: {szn_window.shape}")
        km_assignment = -np.ones((t_obs["t_init"].size,t_obs["member"].size,t_obs["t_sim"].size), dtype=int)
        km_centers = []
        km_n_clusters = -np.ones(self.Nt_szn, dtype=int)
        for i_win in range(self.Nt_szn):
            if i_win % 30 == 0:
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
            km_n_clusters[i_win] = min(num_clusters,max(1,len(idx_for_clustering[0]//2)))
            km_input = feat_msm_normalized.data[idx_for_clustering]
            km = KMeans(n_clusters=km_n_clusters[i_win],random_state=km_seed).fit(
                    km_input)
            km_assignment[idx_in_window] = km.predict(feat_msm_normalized.data[idx_in_window]) 
            km_centers += [km.cluster_centers_]
        km_assignment_da = xr.DataArray(
            coords={"t_init": t_obs["t_init"], "member": t_obs["member"], "t_sim": t_obs["t_sim"]},
            dims=["t_init","member","t_sim"],
            data=km_assignment.copy()    #np.zeros((feat_tpt["ensemble"].size,feat_tpt["member"].size,feat_tpt["t_sim"].size), dtype=int)
        )    
        return km_assignment_da,km_centers,km_n_clusters
    def construct_transition_matrices_stacked(self, km_assignment, km_n_clusters, szn_window, szn_start_year):
        # TODO: make a second list of matrices to test for memory...and maybe a third list, etc. etc.
        kmass = km_assignment.stack(traj=("t_init","member"))
        swindow = szn_window.stack(traj=("t_init","member"))
        ssy = szn_start_year.stack(traj=("t_init","member"))
        P_list = []
        for i_win in range(self.Nt_szn-1):
            if i_win % 30 == 0: print(f"Starting transition matrix for i_win = {i_win}")
            P = np.zeros((km_n_clusters[i_win],km_n_clusters[i_win+1]))
            idx_pre = np.where(swindow.data==i_win)
            idx_post = np.where(szn_window.data==i_win+1)

        time_dim = list(szn_window.dims).index("t_sim")
        nontime_dims = np.setdiff1d(np.arange(len(szn_window.dims)), [time_dim])
        P_list = []
        for i_win in range(self.Nt_szn-1):
            if i_win % 30 == 0: print(f"Starting transition matrix for i_win = {i_win}")
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
    def construct_transition_matrices(self, km_assignment, km_n_clusters, szn_window, szn_start_year):
        # TODO: make a second list of matrices to test for memory...and maybe a third list, etc. etc.
        # TODO: stack this dataset for faster identification of indices
        time_dim = list(szn_window.dims).index("t_sim")
        nontime_dims = np.setdiff1d(np.arange(len(szn_window.dims)), [time_dim])
        P_list = []
        for i_win in range(self.Nt_szn-1):
            if i_win % 30 == 0: print(f"Starting transition matrix for i_win = {i_win}")
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
    def broadcast_field_msm2dataarray(self, msm, field_msm, density_flag=False):
        field_da = np.nan*np.ones(msm["szn_window"].shape)
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

