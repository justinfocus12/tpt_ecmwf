# Code to run the model from Crommelin 2003, adapted from Charney & Devore 1979

import numpy as np

class CrommelinModel:
    def __init__(self,fundamental_param_dict):
        self.q = self.set_params(fundamental_param_dict)
        self.xdim = 6 
        self.dt_sim = 0.001 
        return
    def set_params(self,fpd):
        n_max = 1
        m_max = 2
        self.q = dict({})
        self.q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        self.q["C"] = fpd["C"]
        for vbl in "alpha beta delta gamma gamma_tilde".split(" "):
            self.q[vbl] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            self.q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (fpd["b"]**2 + m**2 - 1)/(fpd["b"]**2 + m**2)
            self.q["beta"][i_m] = fpd["beta"]*fpd["b"]**2/(fpd["b"]**2 + m**2)
            self.q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (fpd["b"]**2 - m**2 + 1)/(fpd["b"]**2 + m**2)
            self.q["gamma_tilde"][i_m] = fpd["gamma"]*4*m/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/np.pi
            self.q["gamma"][i_m] = fpd["gamma"]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/(np.pi*(fpd["b"]**2 + m**2))
        return
    def tendency(self,t,x):
        # t: shape Nx
        # x: shape (Nx,xdim)
        # Check shapes
        if not (t.ndim == 1 and x.ndim == 2 and x.shape[0] == t.shape[0] and x.shape[1] == self.xdim):
            raise Exception(f"Shape problem: you gave me (t,x) such that t.shape = {t.shape} and x.shape = {x.shape}")
        Nx = x.shape[0]
        xdot = np.zeros((Nx,self.xdim))
        xdot[:,0] = self.q["gamma_tilde"][0]*x[:,2] - self.q["C"]*(x[:,0] - self.q["xstar"][0])
        xdot[:,1] = -(self.q["alpha"][0]*x[:,0] - self.q["beta"][0])*x[:,2] - self.q["C"]*x[:,1] - self.q["delta"][0]*x[:,3]*x[:,5]
        xdot[:,2] = (self.q["alpha"][0]*x[:,0] - self.q["beta"][0])*x[:,1] - self.q["gamma"][0]*x[:,0] - self.q["C"]*x[:,2] + self.q["delta"][0]*x[:,3]*x[:,4]
        xdot[:,3] = self.q["gamma_tilde"][1]*x[:,5] - self.q["C"]*(x[:,3] - self.q["xstar"][3]) + self.q["epsilon"]*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
        xdot[:,4] = -(self.q["alpha"][1]*x[:,0] - self.q["beta"][1])*x[:,5] - self.q["C"]*x[:,4] - self.q["delta"][1]*x[:,3]*x[:,2]
        xdot[:,5] = (self.q["alpha"][1]*x[:,0] - self.q["beta"][1])*x[:,4] - self.q["gamma"][1]*x[:,3] - self.q["C"]*x[:,5] + self.q["delta"][1]*x[:,3]*x[:,1]
        return xdot
    def integrate(self,x0,t_save):
        # Integrate forward with Runge-Kutta. Here x0 is a whole list of initial conditions, but all must be integrated with same time horizon.
        if not (t_save.ndim == 2 and x0.ndim == 2 and x0.shape[1] == self.xdim):
            raise Exception(f"Shape problem: for integration from initial condition, you gave me x0.shape = {x0.shape} and t_save.shape = {t_save.shape}")
        Nt_save = len(t_save)
        Nx = x0.shape[0] #Number of initial conditions
        dt_save_min = np.min(np.diff(t_save))
        if dt_save < self.dt_sim:
            raise Exception(f"Sampling problem: you're asking for time outputs as frequent as {dt_save_min}, whereas the computational timestep is {self.dt_sim}")
        x = np.zeros((Nx,Nt_save,self.xdim))
        x_old = x0.copy()
        t_old = t_save[0]
        i_save = 0 # Index of most recently saved state
        x[:,i_save,:] = x0.copy()
        while t_old < t_save[Nt_save-1]:
            k1 = self.tendency(t_old,x_old)
            k2 = self.tendency(t_old+self.dt_sim/2, x_old+self.dt_sim*k1/2)
            k3 = self.tendency(t_old+self.dt_sim/2, x_old+self.dt_sim*k2/2)
            k4 = self.tendency(t_old+self.dt_sim, x_old+self.dt_sim*k3)
            x_new = x_old + 1.0/6*self.dt_sim*(k1 + 2*k2 + 2*k3 + k4)
            t_new = t_old + self.dt_sim
            # If this timestep has crossed a save time, linearly interpolate to the save time. Use the equation
            # (x_new - x_old) / self.dt_sim = (x[i_save+1] - x[i_save])/(t_save[i_save+1] - t_save[i_save])
            # to solve for x[i_save+1].
            if (t_old <= t_save[i_save+1]) and (t_new >= t_save[i_save+1]):
                x[i_save+1] = x[i_save] + (t_save[i_save+1]-t_save[i_save])*(x_new - x_old)/self.dt_sim
                i_save += 1
            # Update new to old
            x_old = x_new
            t_old = t_new
        return x





