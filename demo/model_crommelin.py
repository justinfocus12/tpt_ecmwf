# Code to run the model from Crommelin 2003, adapted from Charney & Devore 1979

import numpy as np

class CrommelinModel:
    def __init__(self,param_dict):
        self.q= param_dict
        self.xdim = 6 
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




