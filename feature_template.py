# Provide a parent class for features of any given system. Most important: definitions of A and B, time-delay embedding, and maps from features to indices. 
# The data will be in the form of Xarray DataSets, with a different variable for each feature, as well as appropriate metadata. 
import numpy as np
import xarray as xr
import netCDF4 as nc
from numpy import save,load
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
import sys
from abc import ABC,abstractmethod

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
        comm_emp = 1.0*(cej.sel(state="B") < cej.sel(state="A"))
        comm_emp.loc[dict(sense="since")] = 1 - comm_emp.sel(sense="since")
        return comm_emp
    def estimate_rate(self, cej, comm_emp):
        # This will be a lower bound on the rate, because of hanging endpoints. 
        a2b_flag = 1.0 * (comm_emp.sel(sense="since") == self.ab_code["A"]) * (comm_emp.sel(sense="until") == self.ab_code["B"])
        rate_lowerbound = (1.0*a2b_flag.diff(dim="t_sim")==1).sum(dim="t_sim") / (a2b_flag["t_sim"][-1] - a2b_flag["t_sim"][0])
        return rate_lowerbound




