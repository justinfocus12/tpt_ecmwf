import numpy as np
from numpy.random import Generator, PCG64
import matplotlib
#matplotlib.use('AGG')
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
import pickle

class OscillatorModel:
    def __init__(self,fundamental_param_dict):
        self.xdim = 3 # position, velocity, and time 
        self.noise_dim = 2
        self.dt_sim = 0.1
        self.fundamental_param_dict = fundamental_param_dict
        self.set_params(self.fundamental_param_dict)
        self.rng = Generator(PCG64(seed=0))
        return
    def set_params(self,fpd):
        """
        Parameters
        ----------
        fpd: dict
            fundamental parameter dictionary, with values for b, beta, gamma, x1star, r, and C
        """
        self.q = dict({"fpd": fpd})
        self.q["period"] = fpd["period"]
        self.q["omega"] = 2*np.pi/self.q["period"]
        # Linear matrix
        L = np.zeros((self.xdim-1,self.xdim-1))
        L[0,1] = 1.0 # dx/dt = v
        L[0,0] = -0.2
        L[1,0] = -self.q["omega"]**2
        L[1,1] = -0.2
        self.q["L"] = L
        # Diffusion matrix
        S = np.zeros((self.xdim,self.noise_dim))
        S[0,0] = 1.0
        S[1,1] = 1.0
        self.q["sigma"] = S
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
        # ------------ Do the calculation as written -----------------
        xdot[:,:-1] = x[:,:-1].dot(self.q["L"].T) 
        xdot[:,-1] = 1.0
        return xdot
    def noise_term(self,x):
        Nx = x.shape[0]
        eta = self.rng.standard_normal((Nx,self.noise_dim))
        sdw = eta.dot(self.q["sigma"].T)
        return sdw
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
            sdw1 = self.noise_term(x_old)
            x_new = x_old + self.dt_sim*k1 + np.sqrt(self.dt_sim) * sdw1
            t_new = t_old + self.dt_sim
            # If this timestep has crossed a save time, linearly interpolate to the save time. Use the equation
            # (x_new - x_old) / self.dt_sim = (x[i_save+1] - x[i_save])/(t_save[i_save+1] - t_save[i_save])
            # to solve for x[i_save+1].
            if (t_old <= t_save[i_save+1]) and (t_new >= t_save[i_save+1]):
                x[:,i_save+1] = x[:,i_save] + (t_save[i_save+1]-t_save[i_save])*(x_new - x_old)/self.dt_sim
                i_save += 1
            if int(t_new/1000) > int(t_old/1000):
                print(f"Integrated through time {t_new} out of {t_save[Nt-1]}")
            # Update new to old
            x_old = x_new
            t_old = t_new
        return x




