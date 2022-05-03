# Code to run the model from Crommelin 2003, adapted from Charney & Devore 1979

import numpy as np
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
import xarray as xr
import netCDF4 as nc
import sys
import os
from os import mkdir
from os.path import join,exists

class CrommelinModel:
    def __init__(self,fundamental_param_dict):
        self.xdim = 6 
        self.dt_sim = 0.1
        self.set_params(fundamental_param_dict)
        return
    def set_params(self,fpd):
        """
        Parameters
        ----------
        fpd: dict
            fundamental parameter dictionary, with values for b, beta, gamma, x1star, r, and C
        """
        n_max = 1
        m_max = 2
        self.q = dict({})
        self.q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        self.q["C"] = fpd["C"]
        self.q["xstar"] = np.array([fpd["x1star"],0,0,fpd["r"]*fpd["x1star"],0,0])
        self.q["alpha"] = np.zeros(m_max)
        self.q["beta"] = np.zeros(m_max)
        self.q["gamma"] = np.zeros(m_max)
        self.q["gamma_tilde"] = np.zeros(m_max)
        self.q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            self.q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (fpd["b"]**2 + m**2 - 1)/(fpd["b"]**2 + m**2)
            self.q["beta"][i_m] = fpd["beta"]*fpd["b"]**2/(fpd["b"]**2 + m**2)
            self.q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (fpd["b"]**2 - m**2 + 1)/(fpd["b"]**2 + m**2)
            self.q["gamma_tilde"][i_m] = fpd["gamma"]*4*m/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/np.pi
            self.q["gamma"][i_m] = fpd["gamma"]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/(np.pi*(fpd["b"]**2 + m**2))
        # The nonlinear system has a constant term F, a linear term Lx, and a quadratic term B(x,x). Construct them each in turn. 
        # 1. Constant term
        F = self.q["C"]*self.q["xstar"]
        # 2. Matrix for linear term
        L = -self.q["C"]*np.eye(self.xdim)
        L[0,2] = self.q["gamma_tilde"][0]
        L[1,2] = self.q["beta"][0]
        L[2,0] = -self.q["gamma"][0]
        L[2,1] = -self.q["beta"][0]
        L[3,5] = self.q["gamma_tilde"][1]
        L[4,5] = self.q["beta"][1]
        L[5,4] = -self.q["beta"][1]
        L[5,3] = -self.q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((self.xdim,self.xdim,self.xdim))
        B[1,0,2] = -self.q["alpha"][0]
        B[1,3,5] = - self.q["delta"][0]
        B[2,0,1] = self.q["alpha"][0]
        B[2,3,4] = self.q["delta"][0]
        B[3,1,5] = self.q["epsilon"]
        B[3,2,4] = -self.q["epsilon"]
        B[4,0,5] = -self.q["alpha"][1]
        B[4,3,2] = -self.q["delta"][1]
        B[5,0,4] = self.q["alpha"][1]
        B[5,3,1] = self.q["delta"][1]
        self.q["forcing_term"] = F
        self.q["linear_term"] = L
        self.q["bilinear_term"] = B
        return
    def tendency(self,x):
        """
        Parameters
        ----------
        x: numpy.ndarray
            shape (Nx,xdim) the current state of the dynamical system
        Returns
        -------
        xdot: numpy.ndarray
            shape (Nx,xdim) the time derivative of x
        """
        if not (x.ndim == 2 and x.shape[1] == self.xdim):
            raise Exception(f"Shape problem: you gave me (t,x) such that t.shape = {t.shape} and x.shape = {x.shape}")
        Nx = x.shape[0]
        xdot = np.zeros((Nx,self.xdim))
        # ----------- Do the calculation with the precomputed terms -------------
        xdot += self.q["forcing_term"] 
        xdot += x.dot(self.q["linear_term"].T)
        for j in range(self.xdim):
            xdot[:,j] += np.sum(x * (x.dot(self.q["bilinear_term"][j].T)), axis=1)
        return xdot
    def integrate(self,x0,t_save):
        """
        Parameters
        ----------
        x0: numpy.ndarray
            Shape is (Nx,xdim) where Nx = (number of parallel integrations) and xdim = (dimensionality of the state vector)
        t_save: numpy.ndarray
            The time since the beginning of the simulation; NOT an absolute time. If the system is time-dependent, that must be incorporated into the state vector.
            Shape is (Nt,) where Nt-1 is the number of timesteps.
        Returns
        -------
        x: numpy.ndarray
            The fully integrated system. Shape is (Nx,Nt,xdim)
        """
        if not (t_save.ndim == 1 and t_save[0] == 0 and x0.ndim == 2 and x0.shape[1] == self.xdim):
            raise Exception(f"Shape problem: for integration from initial condition, you gave me x0.shape = {x0.shape} and t_save.shape = {t_save.shape}")
        Nt = t_save.size
        Nx = x0.shape[0] #Number of initial conditions
        dt_save_min = np.min(np.diff(t_save))
        if dt_save_min < self.dt_sim:
            raise Exception(f"Sampling problem: you're asking for time outputs as frequent as {dt_save_min}, whereas the computational timestep is {self.dt_sim}")
        x = np.zeros((Nx,Nt,self.xdim))
        x_old = x0.copy()
        t_old = 0.0
        i_save = 0 # Index of most recently saved state
        x[:,i_save,:] = x0.copy()
        while t_old < t_save[Nt-1]:
            k1 = self.tendency(x_old)
            k2 = self.tendency(x_old+self.dt_sim*k1/2)
            k3 = self.tendency(x_old+self.dt_sim*k2/2)
            k4 = self.tendency(x_old+self.dt_sim*k3)
            x_new = x_old + 1.0/6*self.dt_sim*(k1 + 2*k2 + 2*k3 + k4)
            t_new = t_old + self.dt_sim
            # If this timestep has crossed a save time, linearly interpolate to the save time. Use the equation
            # (x_new - x_old) / self.dt_sim = (x[i_save+1] - x[i_save])/(t_save[i_save+1] - t_save[i_save])
            # to solve for x[i_save+1].
            if (t_old <= t_save[i_save+1]) and (t_new >= t_save[i_save+1]):
                x[:,i_save+1] = x[:,i_save] + (t_save[i_save+1]-t_save[i_save])*(x_new - x_old)/self.dt_sim
                i_save += 1
            # Update new to old
            x_old = x_new
            t_old = t_new
        return x
    def integrate_and_save(self,x0,t_save,traj_filename,burnin_time=0):
        if not (traj_filename.endswith(".nc")):
            raise Exception("The traj_filename must end with .nc")
        x = self.integrate(x0,t_save)
        if burnin_time > 0:
            ti0 = np.where(t_save > burnin_time)[0][0]
            x = x[:,ti0:]
            t_save = t_save[ti0:]
        ds = xr.Dataset(
                data_vars={"X": xr.DataArray(coords={'member': np.arange(x.shape[0]), 'time': t_save, 'feature': [f"x{i}" for i in range(1,self.xdim+1)], 'time': t_save}, data=x),}
                )
        ds.to_netcdf(traj_filename)
        return 
    def plot_integration(self,traj_filename,savefolder):
        ds = xr.open_dataset(traj_filename)
        # Plot x1 and x4 vs. time
        fig,ax = plt.subplots(nrows=2,figsize=(12,12),sharex=True)
        ax[0].plot(ds.coords['time'].data,ds['X'].sel(feature='x1').data.flatten(),color='black')
        ax[0].set_ylabel(r"$x_1$")
        ax[1].plot(ds.coords['time'].data,ds['X'].sel(feature='x4').data.flatten(),color='black')
        ax[1].set_ylabel(r"$x_4$")
        fig.savefig(join(savefolder,"t_x1x4"))
        plt.close(fig)
        # Plot x4 vs. x1
        fig,ax = plt.subplots()
        ax.plot(ds['X'].sel(feature='x1').data.flatten(), ds['X'].sel(feature='x4').data.flatten(), color='black')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_4$")
        fig.savefig(join(savefolder,"x1_x4"))
        plt.close(fig)
        return
