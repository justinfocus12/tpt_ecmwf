# Class definition for a time-dependent (discret-time, discrete-space) Markov Chain
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os.path import join,exists
savedir = "/home/users/gj116206/s2s/tpt_hcfc_results/2021-09-08/0"
codedir = "/home/users/gj116206/s2s/tpt_hcfc"
os.chdir(codedir)

class TimeDependentMarkovChain:
    def __init__(self,P_list,t):
        self.P_list = P_list
        self.Nx = np.array([P.shape[0] for P in P_list] + [P_list[-1].shape[1]]) # Nx is a vvector of state space dimensions
        self.t = t
        self.Nt = len(t)
        if self.Nt != len(self.Nx):
            sys.exit("ERROR: need Nx and t to have the same length. len(Nx) = {}, len(t) = {}".format(len(self.Nx),len(self.t)))
        return
    def direct_numerical_simulation(self,path_fun,init_dist,Nsim=1000):
        # Starting from a given initial distribution, estimate ensemble average of path_fun
        path_fun_reals = np.zeros(Nsim)
        for i in range(Nsim):
            x = np.zeros(self.Nt, dtype=int)
            x[0] = np.random.choice(np.arange(self.Nx[0]),size=1,p=init_dist)
            for j in range(1,self.Nt):
                x[j] = np.random.choice(np.arange(self.Nx[j]),size=1,p=self.P_list[j-1][x[j-1],:])
            path_fun_reals[i] = path_fun(x)
        return path_fun_reals
    def path_fun_for_dga(self,F,G,x):
        # apply the path functional to a given trajectory x
        Q = 0.0
        for T in range(1,len(self.t)):
            QT = 1.0
            for i in range(T-1):
                QT *= F[i][x[i],x[i+1]]
            QT *= G[T-1][x[T]]
            Q += QT
        return Q
    def dynamical_galerkin_approximation(self,F,G):
        # The functional is F_{0,1}(x0,x1)*F(x1,x2)*...*F(x(T-1),x(T))*G(x(T))
        # F is a (Nt-1)-length list of matrices, and G is a (Nt)-length list of vectors. 
        # F and G are both (Nt-1)-length lists of matrices
        Q = [G[i].copy() for i in range(self.Nt)]
        for i in np.arange(self.Nt-2,-1,-1):
            Q[i] += (self.P_list[i]*F[i]) @ (Q[i+1])
        return Q
    def propagate_density_forward(self,init_dens):
        Q = [np.zeros(self.Nx[i]) for i in range(self.Nt)]
        Q[0] = init_dens
        for i in np.arange(1,self.Nt):
            Q[i] = self.P_list[i-1].T @ (Q[i-1]) 
        return Q
    def estimate_leadtime(self,k0,k1,x0,bndy_idx):
        # Starting from a time index k0, estimate the lead time to get to a time-dependent set, as a full distribution from k0+1 to self.Nt-1
        subPlist = [self.P_list[k].copy() for k in np.arange(k0,k1-1)]
        for k in np.arange(k0,k1-1):
            subPlist[k-k0][bndy_idx[k],:] = 0.0
        submc = TimeDependentMarkovChain(subPlist,self.t[k0:k1])
        init_dens = np.zeros(self.Nx[k0])
        init_dens[x0] = 1.0
        Q = submc.propagate_density_forward(init_dens)
        sumQ = np.array([np.sum(Q[k]) for k in range(len(Q))])
        #print("sumQ = {}".format(sumQ))
        taudens = -np.diff(sumQ)
        taudens *= 1.0/np.sum(taudens)
        #print("taudens = {}".format(taudens))
        #print("taudens.size = {}, and k0 = {} out of {}".format(taudens.size, k0, self.Nt))
        return taudens


if __name__ == "__main__":
    # Make a nice test case for this thing
    np.random.seed(1)
    Nx = np.array([4,2,5])
    t = np.array([0.1,0.2,0.3])
    P_list = []
    for i in range(len(t)-1):
        P_list += [np.random.rand(Nx[i],Nx[i+1])]
        P_list[i] = (np.diag(1/np.sum(P_list[i],axis=1))) @ P_list[i]#Make it a true probability transition matrix
    G = []
    F = []
    bndy_idx = []
    bndy_vals = []
    for i in range(len(t)):
        G += [np.random.randn(Nx[i])]
        F += [np.random.randn(Nx[i],Nx[i+1])]
        bndy_idx += [np.array([0,Nx[i]-1])]
        bndy_vals += [np.array([0,1])]
    # Define the path functional
    def path_fun(x):
        Q = 0.0
        for T in range(1,len(t)):
            QT = 1.0
            for i in range(T-1):
                QT *= F[i][x[i],x[i+1]]
            QT *= G[T-1][x[T-1],x[T]]
            Q += QT
        return Q
    mc = TimeDependentMarkovChain(P_list,t)
    init_dist = np.random.rand(Nx[0])
    init_dist *= 1.0/np.sum(init_dist)
    Q_dga = mc.dynamical_galerkin_approximation(F,G)
    Q_dga = np.sum(Q_dga[0]*init_dist)
    Nsim_log = 7
    Nsim = int(10**Nsim_log)
    Nsim_list = (10**np.linspace(0,Nsim_log,20)).astype(int)
    Q_dns = mc.direct_numerical_simulation(path_fun,init_dist,Nsim=Nsim)
    Q_dns_list = np.zeros(len(Nsim_list))
    for i in range(len(Nsim_list)):
        Q_dns_list[i] = np.mean(Q_dns[:Nsim_list[i]])
    fig,ax = plt.subplots()
    ax.plot(Nsim_list,np.abs(Q_dns_list-Q_dga))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of sample paths")
    ax.set_ylabel("Error between DNS and DGA")
    ax.set_title("DGA vs. DNS")
    fig.savefig(join(savedir,"dga_dns_test"),bbox_inches="tight",pad_inches=0.2)
    plt.close(fig)





        
