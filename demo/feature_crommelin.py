# Methods to compute relevant observable functions from the Crommelin model. 
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
from feature_template import SeasonalFeatures


class SeasonalCrommelinModelFeatures(SeasonalFeatures):
    def __init__(self):
        super().__init__()
        return
    def set_ab_code(self):
        self.ab_code = {"A": 0, "B": 1, "D": 2}
        return
    def set_ab_boundaries(self, t_thresh0, t_thresh1, x1_thresh0, x1_thresh1):
        # Sets the time boundaries (relative to the beginning of the season) during which the event must occur
        # Set B happens whenever x1 dips below x1_thresh0
        # Set A happens if either (1) the time is outside the season thresholds, or (2) x1 goes above x1_thresh1
        self.tpt_bndy = {"t_thresh": [t_thresh0,t_thresh1], "x1_thresh": [x1_thresh0,x1_thresh1]}
        return
    # ------------------- Below is a collection of observable functions ----------------
    def identity_observable(self,ds,q):
        da = ds["X"]
        return da
    def orography_cycle(self,ds,q):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        gamma_t: numpy.ndarray
            The gamma parameter corresponding to the given time of year. It varies sinusoidaly. Same is (Nx,m_max) where m_max is the maximum zonal wavenumber.
        gamma_tilde_t: numpy.ndarray
            The gamma_tilde parameter corresponding to the given time of year. Same shape as gamma_t.
        """
        t_abs = ds["X"].sel(feature="t_abs")
        gamma = xr.DataArray(
                coords={
                    "member": ds.coords['member'],
                    "t_sim": ds.coords['t_sim'],
                    "param": ["g","g_dot"],
                    "tilde": [0,1], # 0 is for gamma; 1 is for gamma_tilde
                    "m": [1,2],
                    },
                data = np.zeros((ds['member'].size, ds['t_sim'].size, 2, 2, 2)),
                dims = ["member","t_sim","param","tilde","m"],
                )
        cosine = np.cos(2*np.pi*t_abs/q["year_length"])
        sine = np.sin(2*np.pi*t_abs/q["year_length"])
        for i_m in range(gamma["m"].size):
            for tilde in [0,1]:
                limits = q["gamma_tilde_limits"][:,i_m] if tilde else q["gamma_limits"][:,i_m]
                amplitude = (limits[1] - limits[0])/2
                offset = (limits[1] + limits[0])/2
                gamma.loc[dict(param="g",tilde=tilde,m=gamma["m"][i_m])] = cosine * amplitude + offset
                gamma.loc[dict(param="g_dot",tilde=tilde,m=gamma["m"][i_m])] = -sine * amplitude
        return gamma
    def energy_enstrophy_coeffs(self,ds,q):
        # Return the coefficients for each energy and enstrophy reservoir given a dataset's attributes
        b = q["b"]
        coeffs_energy = dict({
            "E01": 0.5,
            "E02": 2.0,
            "E11": (b**2 + 1)/2,
            "E12": (b**2 + 4)/2,
            })
        coeffs_enstrophy = dict({
            "Om01": 1.0/(2*b**2),
            "Om02": 8.0/b**2,
            "Om11": (b**2 + 1)**2/(2*b**2),
            "Om12": (b**2 + 4)**2/(2*b**2),
            })
        return coeffs_energy,coeffs_enstrophy
    def energy_observable(self,ds,q):
        # ds is the dataset, including metadata
        # Return a dataarray with all the energy components
        cE,_ = self.energy_enstrophy_coeffs(ds,q)
        da_E = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "reservoir": ["E01","E02","E11","E12","Etot",]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 5)),
                dims = ["member","t_sim","reservoir"],
                )
        da_E.loc[dict(reservoir="E01")] = cE["E01"]*ds["X"].sel(feature="x1")**2
        da_E.loc[dict(reservoir="E02")] = cE["E02"]*ds["X"].sel(feature="x4")**2
        da_E.loc[dict(reservoir="E11")] = cE["E11"]*(ds["X"].sel(feature=["x2","x3"])**2).sum(dim=["feature"])
        da_E.loc[dict(reservoir="E12")] = cE["E12"]*(ds["X"].sel(feature=["x5","x6"])**2).sum(dim=["feature"])
        da_E.loc[dict(reservoir="Etot")] = da_E.sel(reservoir=["E01","E02","E11","E12"]).sum(dim=["reservoir"])
        return da_E
    def energy_tendency_observable(self,ds,q,da_E=None):
        # Compute the tendency of the total energy.
        da_Edot = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "Eflow": ["dissipation","forcing","quadratic","Etot"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 4)),
                dims = ["member","t_sim","Eflow"],
                )
        if da_E is None:
            da_E = self.energy_observable(ds,q)
        da_Edot.loc[dict(Eflow="dissipation")] = -2*q["C"]*da_E.sel(reservoir="Etot")
        da_Edot.loc[dict(Eflow="forcing")] = q["C"]*(
                ds["X"].sel(feature="x1")*q["xstar"][0] + 
                4*ds["X"].sel(feature="x4")*q["xstar"][3]
                )
        da_Edot.loc[dict(Eflow="quadratic")] = 0.0
        da_Edot.loc[dict(Eflow="Etot")] = da_Edot.sel(Eflow=["dissipation","forcing","quadratic"]).sum(dim=["Eflow"])
        return da_Edot
    def energy_tendency_observable_findiff(self,ds,q,da_E=None):
        # Approximate the tendency of energy by taking a finite difference. The dataset must be time-ordered.
        if da_E is None:
            da_E = self.energy_observable(ds,q)
        da_Edot_findiff = da_E.differentiate('t_sim')
        return da_Edot_findiff
    def energy_exchange_observable(self,ds,q,da_E=None):
        # Compute the energy transfers 
        # q is the metadata
        cE,_ = self.energy_enstrophy_coeffs(ds,q)
        if da_E is None:
            da_E = self.energy_observable(ds,q)
        da_Ex = xr.DataArray(
                coords = {
                    "member": ds.coords["member"], 
                    "t_sim": ds.coords["t_sim"], 
                    "source": ["E01","E02","E11","E12","forcing","dissipation",],
                    "sink": ["E01","E02","E11","E12","forcing","dissipation",],
                    },
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 6, 6)),
                dims = ["member","t_sim","source","sink"],
                )
        gamma = self.orography_cycle(ds,q)
        # First, the input from forcing
        da_Ex.loc[dict(source="forcing",sink="E01")] = 2*cE["E01"]*q["C"]*ds["X"].sel(feature="x1")*q["xstar"][0]
        da_Ex.loc[dict(source="forcing",sink="E02")] = 2*cE["E02"]*q["C"]*ds["X"].sel(feature="x4")*q["xstar"][3]
        # Second, the leakage due to dissipation 
        for key in ["E01","E02","E11","E12"]:
            da_Ex.loc[dict(source=key,sink="dissipation")] = 2*q["C"]*da_E.sel(reservoir=key)
        da_Ex.loc[dict(source="E11",sink="dissipation")] += 2*gamma.sel(tilde=0,m=1,param="g")*ds["X"].sel(feature="x1")*ds["X"].sel(feature="x3")*cE["E11"]
        da_Ex.loc[dict(source="E12",sink="dissipation")] += 2*gamma.sel(tilde=0,m=2,param="g")*ds["X"].sel(feature="x4")*ds["X"].sel(feature="x6")*cE["E12"]
        da_Ex.loc[dict(source="E11",sink="E02")] += 2*q["epsilon"]*cE["E02"]*ds["X"].sel(feature="x4")*(
                ds["X"].sel(feature="x2")*ds["X"].sel(feature="x6") - 
                ds["X"].sel(feature="x3")*ds["X"].sel(feature="x5")
                )
        da_Ex.loc[dict(source="E11",sink="E12")] += 2*q["delta"][1]*cE["E12"]*ds["X"].sel(feature="x4")*(
                ds["X"].sel(feature="x2")*ds["X"].sel(feature="x6") - 
                ds["X"].sel(feature="x3")*ds["X"].sel(feature="x5")
                )
        return da_Ex
    def enstrophy_observable(self,ds,q):
        # ds is the dataset, including metadata
        # Return a dataarray with all the energy components
        _,cOm = self.energy_enstrophy_coeffs(ds,q)
        dsOm = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "feature": ["Om01","Om02","Om11","Om12","Omtot"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 5)),
                dims = ["member","t_sim","feature"],
                )
        dsOm.loc[dict(feature="Om01")] = cOm["Om01"]*ds["X"].sel(feature="x1")**2
        dsOm.loc[dict(feature="Om02")] = cOm["Om02"]*ds["X"].sel(feature="x4")**2
        dsOm.loc[dict(feature="Om11")] = cOm["Om11"]*(ds["X"].sel(feature=["x2","x3"])**2).sum(dim=["feature"])
        dsOm.loc[dict(feature="Om12")] = cOm["Om12"]*(ds["X"].sel(feature=["x5","x6"])**2).sum(dim=["feature"])
        dsOm.loc[dict(feature="Omtot")] = dsOm.sel(feature=["Om01","Om02","Om11","Om12"]).sum(dim=["feature"])
        return dsOm
    def phase_observable(self,ds,q):
        # Return the phases of waves 1 and 2
        dsph = xr.DataArray(
                coords = {"member": ds.coords["member"], "t_sim": ds.coords["t_sim"], "wavenumber": ["ph11","ph12"]},
                data = np.zeros((ds["member"].size, ds["t_sim"].size, 2)),
                dims = ["member","t_sim","wavenumber"],
                )
        dsph.loc[dict(wavenumber="ph11")] = np.arctan2(-ds["X"].sel(feature="x3"), ds["X"].sel(feature="x2"))
        dsph.loc[dict(wavenumber="ph12")] = np.arctan2(-ds["X"].sel(feature="x6"), ds["X"].sel(feature="x5"))
        return dsph
    def ab_test(self, Xtpt):
        time_window_flag = 1.0*(
            Xtpt.sel(feature="t_szn") >= self.tpt_bndy["t_thresh"][0])*(
            Xtpt.sel(feature="t_szn") <= self.tpt_bndy["t_thresh"][1]
        )
        blocked_flag = 1.0*(Xtpt.sel(feature="x1") <= self.tpt_bndy["x1_thresh"][0])
        zonal_flag = 1.0*(Xtpt.sel(feature="x1") >= self.tpt_bndy["x1_thresh"][1])
        ab_tag = (
            self.ab_code["A"]*((1*(time_window_flag == 0) + 1*zonal_flag) > 0) + 
            self.ab_code["B"]*(time_window_flag*blocked_flag) + 
            self.ab_code["D"]*(time_window_flag*(blocked_flag==0)*(zonal_flag==0))
        )
        return ab_tag    

