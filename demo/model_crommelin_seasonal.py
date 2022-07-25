# Code to run the model from Crommelin 2003, adapted from Charney & Devore 1979

import numpy as np
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

class SeasonalCrommelinModel:
    def __init__(self,fundamental_param_dict):
        self.xdim = 7 # six standard dimensions plus time 
        self.dt_sim = 0.1
        self.fundamental_param_dict = fundamental_param_dict
        self.set_params(self.fundamental_param_dict)
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
        self.q = dict({"fpd": fpd})
        self.q["year_length"] = fpd["year_length"]
        self.q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        self.q["C"] = fpd["C"]
        self.q["b"] = fpd["b"]
        self.q["gamma_limits_fpd"] = fpd["gamma_limits"]
        self.q["xstar"] = np.array([fpd["x1star"],0,0,fpd["r"]*fpd["x1star"],0,0])
        self.q["alpha"] = np.zeros(m_max)
        self.q["beta"] = np.zeros(m_max)
        self.q["gamma_limits"] = np.zeros((2,m_max))
        self.q["gamma_tilde_limits"] = np.zeros((2,m_max))
        self.q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            self.q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (fpd["b"]**2 + m**2 - 1)/(fpd["b"]**2 + m**2)
            self.q["beta"][i_m] = fpd["beta"]*fpd["b"]**2/(fpd["b"]**2 + m**2)
            self.q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (fpd["b"]**2 - m**2 + 1)/(fpd["b"]**2 + m**2)
            for j in range(2):
                self.q["gamma_tilde_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/np.pi
                self.q["gamma_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/(np.pi*(fpd["b"]**2 + m**2))
        # Construct the forcing, linear matrix, and quadratic matrix, but leaving out the time-dependent parts 
        # 1. Constant term
        F = self.q["C"]*self.q["xstar"]
        # 2. Matrix for linear term
        L = -self.q["C"]*np.eye(self.xdim-1)
        #L[0,2] = self.q["gamma_tilde"][0]
        L[1,2] = self.q["beta"][0]
        #L[2,0] = -self.q["gamma"][0]
        L[2,1] = -self.q["beta"][0]
        #L[3,5] = self.q["gamma_tilde"][1]
        L[4,5] = self.q["beta"][1]
        L[5,4] = -self.q["beta"][1]
        #L[5,3] = -self.q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((self.xdim-1,self.xdim-1,self.xdim-1))
        B[1,0,2] = -self.q["alpha"][0]
        B[1,3,5] = -self.q["delta"][0]
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
    def date_format(self,t_abs,decimal=False):
        # Given a single time t_abs, format the date.
        year = int(t_abs/self.q["year_length"])
        t_cal = t_abs - year*self.q["year_length"]
        if decimal:
            t_str = f"{year:04d}-{t_cal:03d}"
        else:
            t_str = f"{year:.0f}-{t_cal:.0f}"
        return t_str,year
    def orography_cycle(self,t_abs):
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
        cosine = np.cos(2*np.pi*t_abs/self.q["year_length"])
        sine = np.sin(2*np.pi*t_abs/self.q["year_length"])
        gamma_t = np.outer(cosine, (self.q["gamma_limits"][1,:] - self.q["gamma_limits"][0,:])/2) + (self.q["gamma_limits"][0,:] + self.q["gamma_limits"][1,:])/2
        gammadot_t = np.outer(-sine, (self.q["gamma_limits"][1,:] - self.q["gamma_limits"][0,:])/2)
        gamma_tilde_t = np.outer(cosine, (self.q["gamma_tilde_limits"][1,:] - self.q["gamma_tilde_limits"][0,:])/2) + (self.q["gamma_tilde_limits"][0,:] + self.q["gamma_tilde_limits"][1,:])/2
        gammadot_tilde_t = np.outer(-sine, (self.q["gamma_tilde_limits"][1,:] - self.q["gamma_tilde_limits"][0,:])/2) + (self.q["gamma_tilde_limits"][0,:] + self.q["gamma_tilde_limits"][1,:])/2
        gamma_fpd_t = cosine * (self.q["gamma_limits_fpd"][1] - self.q["gamma_limits_fpd"][0])/2 
        gammadot_fpd_t = -sine * (self.q["gamma_limits_fpd"][1] - self.q["gamma_limits_fpd"][0])/2
        return gamma_t,gamma_tilde_t,gamma_fpd_t,gammadot_t,gammadot_tilde_t,gammadot_fpd_t
    def calendar_day(self,t_abs):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        t_cal: numpy.ndarray
            The calendar time, found by taking t_abs mod (year length)
        """
        t_cal = np.mod(t_abs,self.q["year_length"])
        return t_cal
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
        xdot = self.tendency_forcing(x) + self.tendency_dissipation(x) + self.tendency_quadratic(x)
        #gamma_t,gamma_tilde_t,gamma_fpd_t = self.orography_cycle(x[:,6])
        #xdot[:,0] = gamma_tilde_t[:,0]*x[:,2] - self.q["C"]*(x[:,0] - self.q["xstar"][0])
        #xdot[:,1] = -(self.q["alpha"][0]*x[:,0] - self.q["beta"][0])*x[:,2] - self.q["C"]*x[:,1] - self.q["delta"][0]*x[:,3]*x[:,5]
        #xdot[:,2] = (self.q["alpha"][0]*x[:,0] - self.q["beta"][0])*x[:,1] - gamma_t[:,0]*x[:,0] - self.q["C"]*x[:,2] + self.q["delta"][0]*x[:,3]*x[:,4]
        #xdot[:,3] = gamma_tilde_t[:,1]*x[:,5] - self.q["C"]*(x[:,3] - self.q["xstar"][3]) + self.q["epsilon"]*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
        #xdot[:,4] = -(self.q["alpha"][1]*x[:,0] - self.q["beta"][1])*x[:,5] - self.q["C"]*x[:,4] - self.q["delta"][1]*x[:,3]*x[:,2]
        #xdot[:,5] = (self.q["alpha"][1]*x[:,0] - self.q["beta"][1])*x[:,4] - gamma_t[:,1]*x[:,3] - self.q["C"]*x[:,5] + self.q["delta"][1]*x[:,3]*x[:,1]
        #xdot[:,6] = 1.0 # Time 
        # ----------- Do the calculation with the precomputed terms -------------
        #xdot[:,1:] += self.q["forcing_term"] 
        #xdot[:,1:] += x.dot(self.q["linear_term"].T)
        #for j in range(self.xdim):
        #    xdot[:,j] += np.sum(x * (x.dot(self.q["bilinear_term"][j].T)), axis=1)
        return xdot
    def tendency_forcing(self,x):
        Nx,xdim = x.shape
        xdot = np.zeros((Nx,xdim))
        xdot[:,:-1] = np.outer(np.ones(Nx),self.q["forcing_term"])
        xdot[:,-1] = 1.0
        return xdot 
    def tendency_dissipation(self,x):
        Nx,xdim = x.shape
        diss = np.zeros((Nx,xdim))
        diss[:,:-1] = x[:,:-1].dot(self.q["linear_term"].T)
        # Modify the time-dependent components
        gamma_t,gamma_tilde_t,gamma_fpd_t,gammadot_t,gammadot_tilde_t,gammadot_fpd_t = self.orography_cycle(x[:,6])
        diss[:,0] += gamma_tilde_t[:,0]*x[:,2]
        diss[:,2] -= gamma_t[:,0]*x[:,0]
        diss[:,3] += gamma_tilde_t[:,1]*x[:,5]
        diss[:,5] -= gamma_t[:,1]*x[:,3]
        return diss
    def tendency_quadratic(self,x):
        """
        Compute the tendency according to only the nonlinear terms, in order to check conservation of energy and enstrophy.
        """
        Nx = x.shape[0]
        xdot = np.zeros((Nx,self.xdim))
        # ----------- Do the calculation with the precomputed terms -------------
        for j in range(self.xdim-1):
            xdot[:,j] += np.sum(x[:,:-1] * (x[:,:-1].dot(self.q["bilinear_term"][j].T)), axis=1)
        return xdot
    def energy_tendency_dissipation(self,x):
        energy,_ = self.energy_enstrophy(x)
        Edot_autonomous = -2*self.q["C"]*energy["total"]
        return Edot_autonomous 
    def energy_tendency_forcing(self,x):
        return self.q["C"]*(x[:,0]*self.q["xstar"][0] + 4*x[:,3]*self.q["xstar"][3])
    def energy_enstrophy(self,x):
        # Compute the energy and enstrophy (normalized by area) of a given state.
        xsq = x**2
        energy = dict({
            "01": 0.5*xsq[:,0],
            "02": 2*xsq[:,3],
            "11": (self.q["b"]**2 + 1)/2*(xsq[:,1] + xsq[:,2]),
            "12": (self.q["b"]**2 + 4)/2*(xsq[:,4] + xsq[:,5]),
            })
        energy["total"] = energy["01"] + energy["02"] + energy["11"] + energy["12"]
        enstrophy = dict({
            "01": xsq[:,0]/(2*self.q["b"]**2),
            "02": xsq[:,3]*8/(self.q["b"]**2),
            "11": (self.q["b"]**2 + 1)**2/(2*self.q["b"]**2)*(xsq[:,1] + xsq[:,2]),
            "12": (self.q["b"]**2 + 4)**2/(2*self.q["b"]**2)*(xsq[:,4] + xsq[:,5]),
            })
        enstrophy["total"] = enstrophy["01"] + enstrophy["02"] + enstrophy["11"] + enstrophy["12"]
        return energy,enstrophy
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
            if int(t_new/1000) > int(t_old/1000):
                print(f"Integrated through time {t_new} out of {t_save[Nt-1]}")
            # Update new to old
            x_old = x_new
            t_old = t_new
        return x
    def integrate_and_save(self,x0,t_save,traj_filename,metadata_filename=None,burnin_time=0):
        """
        Parameters
        ----------
        x0,t_save: same as in self.integrate
        traj_filename: str
            Filename (including full path, and .nc extension) in which to save the final output of self.integrate as an Xarray database
        metadata_filename: str
            Filename (including full path) in which to save the physical parameters
        burnin_time: float
            Time to clip off the beginning of the trajectory to allow it to approach its attractor

        Returns 
        -------
        None

        Side effects
        ------------
        Saves trajectory to traj_filename as an xarray.Dataset
        """
        if not (traj_filename.endswith(".nc")):
            raise Exception("The traj_filename must end with .nc")
        x = self.integrate(x0,t_save)
        if burnin_time > 0:
            ti0 = np.where(t_save > burnin_time)[0][0]
            x = x[:,ti0:]
            t_save = t_save[ti0:] - t_save[ti0]
        ds = xr.Dataset(
                data_vars={"X": xr.DataArray(coords={'member': np.arange(x.shape[0]), 't_sim': t_save, 'feature': [f"x{i}" for i in range(1,self.xdim)] + ["t_abs"]}, data=x),}
                )
        ds.to_netcdf(traj_filename)
        if metadata_filename is not None:
            pickle.dump(self.q, open(metadata_filename, "wb"))
        # TODO: pickle dump the parameter file for reading
        return 
    def split_long_integration(self,traj_filename,savefolder,szn_start,szn_length):
        """
        Parameters
        ----------
        traj_filename: str
            name of the .nc file in which the long (multi-year) trajectory is stored
        savefolder: str
            name of the folder in which to store the single-season trajectories
        szn_start: float
            calendar day for the beginning of the season 
        szn_length: float
            length of the season
        """
        traj = xr.open_dataset(traj_filename)['X']
        print(f"traj = \n{traj}")
        traj_coords_local = {dim: traj.coords[dim].data for dim in traj.dims}
        print(f"traj = \n{traj}")
        t_abs = traj.sel(feature='t_abs',member=0).data.flatten()
        year = (t_abs/self.q["year_length"]).astype(int)
        t_cal = t_abs - year*self.q["year_length"]
        # Does the first season start in the first year, or the second year? 
        if t_cal[0] <= szn_start:
            year_start = year[0]
        else:
            year_start = year[0] + 1
        t_abs_start = year_start*self.q["year_length"] + szn_start
        t_sim_start = t_abs_start - t_abs[0]
        t_sim_end = t_sim_start + szn_length
        print(f"t_sim_start = {t_sim_start}, t_sim_end = {t_sim_end}")
        i_start = np.where(traj.t_sim[:] > t_sim_start)[0][0]
        print(f"t_sim[i_start] = {traj.t_sim[i_start]}")
        szn_filename_list = []
        while t_sim_end < traj.t_sim[-1]:
            traj_szn = traj.sel(t_sim=slice(t_sim_start,t_sim_end))
            traj_coords_local['t_sim'] = traj_szn.t_sim.data 
            traj_coords_local['t_sim'] -= traj_coords_local['t_sim'][0]
            traj_szn = xr.Dataset({'X': xr.DataArray(data=traj_szn.data, coords=traj_coords_local)})
            t0_str,t0_year = self.date_format(t_sim_start + t_abs[0])
            t1_str,t1_year = self.date_format(t_sim_end + t_abs[0])
            # Add a unique identifier to traj_szn as metadata
            traj_szn.attrs['szn_id'] = f"{t0_year:04d}-{t1_year:04d}"
            szn_filename = join(savefolder,f"ra{t0_str}_to_{t1_str}.nc")
            szn_filename_list += [szn_filename]
            traj_szn.to_netcdf(szn_filename)
            # Advance t_start and t_end
            t_sim_start += self.q["year_length"]
            t_sim_end += self.q["year_length"]
        return szn_filename_list
    def plot_integration(self,traj_filename,savefolder):
        """
        Plot the output of integrate_and_save above
        Parameters
        ----------
        traj_filename: str
            The filename storing the integration (with .nc extension)
        savefolder: str
            The folder in which to save the result
        """
        ds = xr.open_dataset(traj_filename)
        # Plot x1 and x4 vs. time (but only for 4 cycles)
        ti0 = 0
        ti1 = np.where(ds.coords['t_sim'].data < 4*self.q["year_length"])[0][-1]
        fig,ax = plt.subplots(nrows=3,figsize=(12,18),sharex=True)
        ax[0].plot(ds['X'].sel(feature='t_abs').data.flatten()[ti0:ti1],ds['X'].sel(feature='x1').data.flatten()[ti0:ti1],color='black')
        ax[0].set_ylabel(r"$x_1$")
        ax[1].plot(ds['X'].sel(feature='t_abs').data.flatten()[ti0:ti1],ds['X'].sel(feature='x4').data.flatten()[ti0:ti1],color='black')
        ax[1].set_ylabel(r"$x_4$")
        gamma_t,gamma_tilde_t,gamma_fpd_t,_,_,_ = self.orography_cycle(ds['X'].sel(feature='t_abs',member=0).data.flatten())
        ax[2].plot(ds['X'].sel(feature='t_abs').data.flatten()[ti0:ti1],gamma_fpd_t[ti0:ti1],color='black')
        ax[2].set_ylabel(r"$\gamma(t)$")
        fig.savefig(join(savefolder,"t_x1x4gamma"))
        plt.close(fig)
        # Plot x4 vs. x1
        fig,ax = plt.subplots()
        ax.plot(ds['X'].sel(feature='x1').data.flatten(), ds['X'].sel(feature='x4').data.flatten(), color='black')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_4$")
        fig.savefig(join(savefolder,"x1_x4"))
        plt.close(fig)
        # Plot x1 vs. the seasonal cycle
        fig,ax = plt.subplots()
        ax.plot(gamma_fpd_t,ds['X'].sel(feature='x1').data.flatten(), color='black')
        ax.set_xlabel(r"$\gamma(t)$")
        ax.set_ylabel(r"$x_1(t)$")
        fig.savefig(join(savefolder,"gamma_x1"))
        plt.close(fig)
        # Plot x1 vs. the time of year
        t_cal = self.calendar_day(ds['X'].sel(feature='t_abs',member=0).data.flatten())
        fig,ax = plt.subplots()
        ax.plot(t_cal,ds['X'].sel(feature='x1').data.flatten(), color='black')
        ax.set_xlabel(r"Calendar time")
        ax.set_ylabel(r"$x_1(t)$")
        fig.savefig(join(savefolder,"gamma_x1"))
        plt.close(fig)
        return
    def generate_hindcast_dataset(self,ra_filename,hc_dir,t_abs_range,dt_samp,ens_size=10,ens_duration=47,ens_gap=3,pert_scale=0.01):
        """
        Parameters
        ----------
        ra_file: str
            Path to read reanalysis data from (including .nc extension)
        hc_dir: str
            Path to directory to store hindcasts 
        t_abs_range: array or list with 2 elements
            Absolute timespan over which to generate trajectories
            How many years should we compute hindcasts for? 
        ens_size: int
            How many parallel ensembles to launch at each initialization date
        ens_duration: float
            How long to run each member before terminating
        ens_gap: float
            Length of time between each successive ensemble (3.5 on average for biweekly forecasts)
        """
        ra = xr.open_dataset(ra_filename)
        t_sim_ra = ra.coords['t_sim'].data
        t_abs_ra = ra['X'].sel(feature='t_abs',member=0).data.flatten()
        if not (t_abs_range[0] >= t_abs_ra[0] and t_abs_range[1] <= t_abs_ra[-1]):
            raise Exception("You gave me a time range that falls outside the simulation time: t_abs_range = {t_abs_range}, whilst the simulation time ranges from {t_abs_ra[0]} to {t_abs_ra[-1]}")
        # Each hindcast ensemble will be stored in a different file. 
        t_abs = t_abs_range[0]
        while t_abs < t_abs_range[1]:
            i_time_ra = np.where(t_abs_ra >= t_abs)[0][0]
            x0_ra = ra['X'].isel(t_sim=i_time_ra,member=0).data
            x0 = np.outer(np.ones(ens_size),x0_ra)
            # Add perturbations to the non-time variables
            pert = pert_scale * np.random.randn(ens_size,self.xdim-1)
            x0[:,:-1] += pert
            # Integrate 
            t0_str,t0_year = self.date_format(t_abs)
            t1_str,t1_year = self.date_format(t_abs+ens_duration)
            ens_filename = join(hc_dir,f"hc{t0_str}_to_{t1_str}.nc")
            t_ens = np.arange(0,ens_duration,dt_samp)
            self.integrate_and_save(x0,t_ens,ens_filename)
            # Advance the initialization time
            t_abs = t_abs_ra[i_time_ra] + ens_gap
            if int(t_abs/1000) > (t_abs_ra[i_time_ra]/1000):
                print(f"Hindcast integration {(t_abs - t_abs_range[0])/np.ptp(t_abs_range)*100} percent done")
        return
