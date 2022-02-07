# TPT class for GENERAL stratosphere data: hindcasts or reanalysis. No explicit feature-handling here, just take in numpy arrays and do the magic. Except for time and the feature that gives distances to A and B. 

import numpy as np
import netCDF4 as nc
import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'serif', 'size': 12}
font = {'family': 'serif', 'size': 18}
bigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import scipy.sparse as sps
import sys
import os
from os import mkdir
from os.path import join,exists
from sklearn import linear_model
from sklearn.cluster import KMeans, MiniBatchKMeans
import helper
import cartopy
from cartopy import crs as ccrs
import pickle
import tdmc_obj

class WinterStratosphereTPT:
    def __init__(self):
        # physical_params is a dictionary that tells us (1) how to tell time, and (2) how to assess distance from A and B
        return
    def set_boundaries(self,tpt_bndy):
        self.tpt_bndy = tpt_bndy
        return
    # Everything above is just about defining the boundary value problem. Everything below will be analysis with respect to a specific data set. 
    def compute_rate_direct(self,src_tag,dest_tag):
        # This is meant for full-winter trajectories
        ab_tag = (src_tag==0)*(dest_tag==1)
        print(f"ab_tag.shape = {ab_tag.shape}")
        #absum = 1.0*(np.sum(ab_tag,axis=1) > 0)
        absum = 1.0*(np.sum(np.diff(1.0*ab_tag,axis=1)==1, axis=1))
        print(f"absum = {absum}")
        rate = np.mean(absum)
        print(f"rate = {rate}")
        return rate
    def tpt_pipeline_dns(self,tpt_feat_filename,savedir,winstrat,feat_def,algo_params,plot_field_flag=True):
        # savedir: where all the results of this particular TPT will be saved.
        # winstrat: the object that was used to create features, and that can go on to evaluate other features.
        # feat_def: the feature definitions that came out of winstrat.
        tpt_feat = pickle.load(open(tpt_feat_filename, "rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Ny,Nt,ydim = Y.shape
        #print(f"in DNS pipeline: ydim = {ydim}")
        funlib = winstrat.observable_function_library_Y(algo_params)
        # ---- Plot committor in a few different coordinates -----
        src_tag,dest_tag = winstrat.compute_src_dest_tags(Y,feat_def,self.tpt_bndy,"src_dest")
        #print(f"src_tag[:,0] = {src_tag[:,0]}")
        ina_Y = winstrat.ina_test(Y[:,0,:],feat_def,self.tpt_bndy)
        #print(f"ina_Y = {ina_Y}")
        qp = 1.0*(dest_tag == 1).flatten()
        qm = 1.0*(src_tag == 0).flatten()
        pi = 1.0*np.ones(Ny*Nt)/(Ny*Nt)
        #keypairs = [['time_d','area'],['time_d','displacement'],['time_d','uref'],['time_d','mag1'],['time_d','lev0_pc0'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4']]
        if plot_field_flag:
            keypairs = [['time_d','uref_dl0']]
            for i_kp in range(len(keypairs)):
                fun0name,fun1name = [funlib[keypairs[i_kp][j]]["label"] for j in range(2)]
                theta0_x = funlib[keypairs[i_kp][0]]["fun"](Y.reshape((Ny*Nt,ydim)))
                theta1_x = funlib[keypairs[i_kp][1]]["fun"](Y.reshape((Ny*Nt,ydim)))
                theta_x = np.array([theta0_x,theta1_x]).T
                # Plot forward committor
                fig,ax = helper.plot_field_2d(qp,pi,theta_x,shp=[15,15],fieldname="Committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                fig.savefig(join(savedir,"qp_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                # Plot density
                fig,ax = helper.plot_field_2d(pi,np.ones(Ny*Nt),theta_x,shp=[15,15],fieldname="Density",fun0name=fun0name,fun1name=fun1name,contourflag=True,logscale=True,avg_flag=False)
                fig.savefig(join(savedir,"pi_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                # Plot backward committor
                fig,ax = helper.plot_field_2d(qm,pi,theta_x,shp=[15,15],fieldname="Backward committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                fig.savefig(join(savedir,"qm_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
        # Compute density and backward committor
        # Compute rate
        rate = self.compute_rate_direct(src_tag,dest_tag)
        # Compute lead time
        # Compute current (??)
        summary = {"rate": rate}
        pickle.dump(summary,open(join(savedir,"summary"),"wb"))
        return summary 
    def cluster_features(self,tpt_feat_filename,clust_filename,winstrat,num_clusters=100,resample_flag=False,seed=0):
        # Read in a feature array from feat_filename and build clusters. Save cluster centers. Save them out in clust_filename.
        tpt_feat = pickle.load(open(tpt_feat_filename,"rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Nx,Nt,ydim = Y.shape
        Y = Y.reshape((Nx*Nt,ydim))
        # cluster based on the non-time features. 
        Y_unseasoned = winstrat.unseason(Y[:,0],Y[:,1:],szn_mean_Y,szn_std_Y,normalize=True)
        kmtime = []
        kmlist = []
        for ti in range(winstrat.Ntwint):
            idx = np.where(np.abs(Y[:,0] - winstrat.wtime[ti]) < winstrat.szn_hour_window/2)[0]
            if len(idx) == 0:
                print("WARNING, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(ti,Y[:,0].min(),Y[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
            else:
                kmtime += [winstrat.wtime[ti]]
                #km = MiniBatchKMeans(min(len(idx),num_clusters)).fit((Y[idx,1:] - offset_Y)/scale_Y)
                km = MiniBatchKMeans(min(len(idx),num_clusters)).fit(Y_unseasoned[idx])
                kmlist += [km]
        kmdict = {"kmlist": kmlist, "kmtime": kmtime}
        pickle.dump(kmdict,open(clust_filename,"wb"))
        print("n_clusters: {}".format(np.array([km.n_clusters for km in kmlist])))
        return
    def build_msm(self,tpt_feat_filename,clust_filename,msm_filename,winstrat):
        nnk = 4 # Number of nearest neighbors for filling in empty positions
        tpt_feat = pickle.load(open(tpt_feat_filename,"rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Ny,Nt,ydim = Y.shape
        Y_unseasoned = winstrat.unseason(Y[:,:,0].reshape(Ny*Nt),Y[:,:,1:].reshape((Ny*Nt,ydim-1)),szn_mean_Y,szn_std_Y,normalize=True).reshape((Ny,Nt,ydim-1))
        kmdict = pickle.load(open(clust_filename,"rb"))
        kmlist,kmtime = kmdict["kmlist"],kmdict["kmtime"]
        P = []
        #centers = np.concatenate((np.zeros(km.n_clusters,1),km.cluster_centers_), axis=1)
        for ti in range(len(kmtime)-1): #winstrat.Ntwint-1):
            if ti % 30 == 0:
                print("MSM timestep {} out of {}".format(ti,winstrat.Ntwint))
            P += [np.zeros((kmlist[ti].n_clusters,kmlist[ti+1].n_clusters))]
            #centers[:,0] = winstrat.wtime[ti]
            #print("wtime[0] = {}".format(winstrat.wtime[0]))
            #print("X[0,:3,0] = {}".format(X[0,:3,0]))
            #print("X[:,:,0] range = {},{}".format(X[:,:,0].min(),X[:,:,0].max()))
            idx0 = np.where(np.abs(Y[:,:-1,0] - kmtime[ti]) < winstrat.dtwint/2) # (idx1_x,idx1_t) where idx1_x < Nx and idx1_t < Nt-1
            idx1 = np.where(np.abs(Y[:,1:,0] - kmtime[ti+1]) < winstrat.dtwint/2) # (idx1_x,idx1_t) where idx1_x < Nx and idx1_t < Nt-1
            if len(idx0[0]) > 0 and len(idx1[0]) > 0:
                overlap = np.where(np.subtract.outer(idx0[0],idx1[0]) == 0)
                labels0 = kmlist[ti].predict(Y_unseasoned[idx0[0][overlap[0]],idx0[1][overlap[0]]])
                labels1 = kmlist[ti+1].predict(Y_unseasoned[idx1[0][overlap[1]],idx1[1][overlap[1]]])
                #print("labels0.shape = {}".format(labels0.shape))
                #print("labels1.shape = {}".format(labels1.shape))
                #print("len(idx0[0]) = {}, len(idx1[0]) = {}".format(len(idx0[0]),len(idx1[0])))
                #print("len(overlap[0]) = {}".format(len(overlap[0])))
                # Overlaps between idx0[0] and idx1[0] give tell us they're on the same trajectory
                for i in range(kmlist[ti].n_clusters):
                    for j in range(kmlist[ti+1].n_clusters):
                        P[ti][i,j] += np.sum((labels0==i)*(labels1==j))
                                #(labels[idx0[0][overlap[0]],idx0[1][overlap[0]]] == i) *
                                #(labels[idx1[0][overlap[1]],idx1[1][overlap[1]]] == j)
                                #)
            # Make sure every row and column has a nonzero entry. 
            rowsums = np.array(P[ti].sum(1)).flatten()
            #print("rowsums: min={}, max={}".format(rowsums.min(),rowsums.max()))
            idx_rs0 = np.where(rowsums==0)[0]
            for i in idx_rs0:
                nnki = min(nnk,kmlist[ti].n_clusters)
                knn = np.argpartition(np.sum((kmlist[ti].cluster_centers_ - kmlist[ti].cluster_centers_[i])**2, axis=1), nnki)[:nnki]
                P[ti][i,nnki] = 1.0/nnki
                #j = np.argmin(np.sum((km.cluster_centers_ - km.cluster_centers_[i])**2, axis=1))
                #P[ti][i,j] = 1.0
            colsums = np.array(P[ti].sum(0)).flatten()
            idx_cs0 = np.where(colsums==0)[0]
            for j in idx_cs0:
                knn = np.argpartition(np.sum((kmlist[ti+1].cluster_centers_ - kmlist[ti+1].cluster_centers_[j])**2, axis=1), nnk)[:nnk]
                P[ti][nnk,j] = 1.0/nnk
            # Now normalize rows
            P[ti] = np.diag(1.0 / np.array(P[ti].sum(1)).flatten()) @ P[ti]
            # Check row sums
            rowsums = np.array(P[ti].sum(1))
            if np.any(np.abs(rowsums - 1.0) > 1e-10):
                raise Exception("The rowsums of P[{}] range from {} to {}".format(ti,rowsums.min(),rowsums.max()))
            colsums = P[ti].sum(0)
            if np.any(colsums == 0):
                raise Exception("The colsums of P range from {} to {}".format(colsums.min(),colsums.max()))
        pickle.dump(P,open(msm_filename,"wb"))
        return
    def compute_tdep_density(self,P_list,init_dens,time):
        init_dens *= 1.0/np.sum(init_dens)
        mc = tdmc_obj.TimeDependentMarkovChain(P_list,time)
        dens = mc.propagate_density_forward(init_dens)
        return dens
    def compute_backward_committor(self,P_list,time,ina,inb,dens):
        P_list_bwd = []
        for i in np.arange(len(time)-2,-1,-1):
            P_list_bwd += [(P_list[i] * np.outer(dens[i], 1.0/dens[i+1])).T]
            rowsums = np.sum(P_list_bwd[-1],axis=1)
        mc = tdmc_obj.TimeDependentMarkovChain(P_list_bwd,time)
        #print("len(mc.Nx) = {}, mc.Nx[0] = {}, mc.Nx[-1] = {}\nlen(mc.P_list) = {}, mc.P_list[0].shape = {}, mc.P_list[-1].shape = {}".format(len(mc.Nx),mc.Nx[0],mc.Nx[-1],len(mc.P_list),mc.P_list[0].shape,mc.P_list[-1].shape))
        G = []
        F = []
        for i in np.arange(mc.Nt-1,-1,-1):
            G += [1.0*ina[i]]
            if i < mc.Nt-1: 
                #if len(inb[i+1]) != len(inb[i]):
                #    print("i = {}, len(inb[i+1]) = {}, len(inb[i]) = {}, mc.Nx[i+1] = {}, mc.Nx[i] = {}".format(i,len(inb[i+1]),len(inb[i]),mc.Nx[i+1],mc.Nx[i]))
                Fnew = np.outer(1.0*(ina[i+1]==0)*(inb[i+1]==0), np.ones(len(inb[i])))
                F += [Fnew.copy()]
                #if Fnew.shape[0] != P_list_bwd[i].shape[0] or Fnew.shape[1] != P_list_bwd[i].shape[1]:
                #    raise Exception("At index {}, Fnew.shape = {}, P_list_bwd.shape = {}".format(i,Fnew.shape,P_list_bwd[i].shape))
        #print("len(F) = {}, F shapes {}...{}".format(len(F),F[0].shape,F[-1].shape))
        #print("len(G) = {}, G shapes {}...{}".format(len(G),G[0].shape,G[-1].shape))
        #print("len(P_list) = {}, P_list shapes {}...{}".format(len(P_list),P_list[0].shape,P_list[-1].shape))
        #print("len(P_list_bwd) = {}, P_list_bwd shapes {}...{}".format(len(P_list_bwd),P_list_bwd[0].shape,P_list_bwd[-1].shape))
        #print("len(ina) = {}, ina shapes {}...{}".format(len(ina),ina[0].shape,ina[-1].shape))
        #print("len(inb) = {}, inb shapes {}...{}".format(len(inb),inb[0].shape,inb[-1].shape))
        qm = mc.dynamical_galerkin_approximation(F,G)
        qm.reverse()
        return qm
    def compute_forward_committor(self,P_list,time,ina,inb):
        mc = tdmc_obj.TimeDependentMarkovChain(P_list,time)
        G = []
        F = []
        for i in range(mc.Nt):
            G += [1.0*inb[i]]
            if i < mc.Nt-1: F += [1.0*np.outer((ina[i]==0)*(inb[i]==0), np.ones(mc.Nx[i+1]))]
        qp = mc.dynamical_galerkin_approximation(F,G)
        return qp
    def compute_integral_to_B(self,P_list,time,ina,inb,qp,dam_centers,maxmom=2):
        # For example, lead time. damfun must be a function of the full state space, but computable on cluster centers. 
        # G = (1_B) * exp(lam*damfun)
        # F = (1_D) * exp(lam*damfun)
        nclust_list = [len(dam_centers[ti]) for ti in range(len(dam_centers))]
        int2b = []
        if maxmom >= 1:
            mc = tdmc_obj.TimeDependentMarkovChain(P_list,time)
            G = []
            F = []
            I = []
            j = 0 # counts cluster
            for i in range(mc.Nt):
                ncl0 = nclust_list[i]
                if i < mc.Nt-1: 
                    ncl1 = nclust_list[i+1]
                    F += [np.outer(1.0*(ina[i]==0)*(inb[i]==0), np.ones(mc.Nx[i+1]))]
                    I += [0.5*np.add.outer(dam_centers[i], dam_centers[i+1])]
                    G += [(P_list[i] * F[i] * I[i]) @ (qp[i+1])]
                else:
                    G += [np.zeros(ncl0)] #mc.Nx[mc.Nt-1])]
                j += ncl0
            int2b += [mc.dynamical_galerkin_approximation(F,G)]
        if maxmom >= 2:
            G = []
            j = 0
            for i in range(mc.Nt):
                ncl0 = nclust_list[i]
                if i < mc.Nt-1:
                    ncl1 = nclust_list[i+1]
                    G += [(P_list[i] * F[i] * I[i]**2) @ (qp[i+1]) + 2*(P_list[i] * F[i] * I[i]) @ (int2b[0][i+1])]
                else:
                    G += [np.zeros(ncl0)]
            int2b += [mc.dynamical_galerkin_approximation(F,G)]
        if maxmom >= 3:
            G = []
            j = 0
            for i in range(mc.Nt):
                ncl0 = nclust_list[i]
                if i < mc.Nt-1:
                    ncl1 = nclust_list[i+1]
                    G += [(P_list[i] * F[i] * I[i]**3) @ (qp[i+1]) + 3*(P_list[i] * F[i] * I[i]**2) @ (int2b[0][i+1]) + 3*(P_list[i] * F[i] * I[i]) @ (int2b[1][i+1])]
                else:
                    G += [np.zeros(ncl0)]
            int2b += [mc.dynamical_galerkin_approximation(F,G)]
        if maxmom >= 4:
            G = []
            j = 0
            for i in range(mc.Nt):
                ncl0 = nclust_list[i]
                if i < mc.Nt-1:
                    ncl1 = nclust_list[i+1]
                    G += [(P_list[i] * F[i] * I[i]**4) @ (qp[i+1]) + 4*(P_list[i] * F[i] * I[i]**3) @ (int2b[0][i+1]) + 6*(P_list[i] * F[i] * I[i]**2) @ (int2b[1][i+1]) + 4*(P_list[i] * F[i] * I[i]) @ (int2b[2][i+1])]
                else:
                    G += [np.zeros(ncl0)]
            int2b += [mc.dynamical_galerkin_approximation(F,G)]
        return int2b
    def conditionalize_int2b(self,P_list,time,int2b,qp): #,damkey):
        # Having already computed an integral, turn it conditional
        int2b_cond = {}
        nclust_list = [len(qp[ti]) for ti in range(len(qp))]
        qp = np.concatenate(tuple(qp))
        maxmom = len(int2b)
        #print("maxmom = {}".format(maxmom))
        #sys.exit()
        if maxmom >= 1:
            m1 = np.concatenate(tuple(int2b[0]))
            cond_m1 = m1*(qp != 0)/(qp + 1.0*(qp == 0))
            cond_m1[qp == 0] = np.nan
            int2b_cond['mean'] = []
            int2b_cond['m1'] = [] # This is pedantic: it's the same
            j = 0
            for k in range(len(time)):
                nclust = nclust_list[k]
                int2b_cond['mean'] += [cond_m1[j:j+nclust]]
                int2b_cond['m1'] += [cond_m1[j:j+nclust]]
                j += nclust
        if maxmom >= 2:
            m2 = np.concatenate(tuple(int2b[1]))
            cond_m2 = m2*(qp != 0)/(qp + 1.0*(qp == 0))
            cond_m2[qp == 0] = np.nan
            cond = (cond_m2 - cond_m1**2) #*(qp != 0)/(qp + 1.0*(qp == 0))
            if np.min(cond) < 0: sys.exit("error: we got a negative variance. min = {}, max = {}".format(np.min(m2-m1**2),np.max(m2-m1**2)))
            cond[qp == 0] = np.nan
            cond = np.sqrt(cond)
            int2b_cond['std'] = []
            int2b_cond['m2'] = []
            j = 0
            for k in range(len(time)):
                nclust = nclust_list[k]
                int2b_cond['std'] += [cond[j:j+nclust]]
                int2b_cond['m2'] += [cond_m2[j:j+nclust]]
                j += nclust
        if maxmom >= 3:
            m3 = np.concatenate(tuple(int2b[2]))
            cond_m3 = m3*(qp != 0)/(qp + 1.0*(qp == 0))
            cond_m3[qp == 0] = np.nan
            cond = (cond_m3 - 3*cond_m1*(cond_m2 - cond_m1**2) - cond_m1**3) #*(qp != 0)/(qp + 1.0*(qp == 0))
            cond[qp == 0] = np.nan
            #cond = np.sign(cond)*np.abs(cond)**(1.0/3)
            int2b_cond['skew'] = []
            int2b_cond['m3'] = []
            j = 0
            for k in range(len(time)):
                nclust = nclust_list[k]
                int2b_cond['skew'] += [cond[j:j+nclust]]
                int2b_cond['m3'] += [cond_m3[j:j+nclust]]
                j += nclust
        if maxmom >= 4:
            m4 = np.concatenate(tuple(int2b[3]))
            cond_m4 = m4*(qp != 0)/(qp + 1.0*(qp == 0))
            cond_m4[qp == 0] = np.nan
            cond = (cond_m4 - 4*cond_m3*cond_m1 + 6*cond_m2*cond_m1**2 - 3*cond_m1**4)/(cond_m2 - cond_m1**2)**2
            cond[qp == 0] = np.nan
            int2b_cond['kurt'] = []
            int2b_cond['m4'] = []
            j = 0
            for k in range(len(time)):
                nclust = nclust_list[k]
                int2b_cond['kurt'] += [cond[j:j+nclust]]
                int2b_cond['m4'] += [cond_m4[j:j+nclust]]
                j += nclust
        return int2b_cond 
    def tpt_pipeline_dga(self,tpt_feat_filename,clust_filename,msm_filename,feat_def,savedir,winstrat,algo_params,plot_field_flag=True):
        # Label each cluster as in A or B or elsewhere
        tpt_feat = pickle.load(open(tpt_feat_filename,"rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Ny,Nt,ydim = Y.shape
        funlib = winstrat.observable_function_library_Y(algo_params)
        #uref_y = funlib["uref"]["fun"](Y.reshape((Ny*Nt,ydim)))
        #print("uref_y: min={}, max={}, mean={}".format(uref_y.min(),uref_y.max(),uref_y.mean()))
        kmdict = pickle.load(open(clust_filename,"rb"))
        kmlist,kmtime = kmdict["kmlist"],kmdict["kmtime"]
        kmtime_d = np.array(kmtime)/24.0
        ina = []
        inb = []
        centers = []
        for ti in range(len(kmtime)):
            #centers_t = np.concatenate((kmtime[ti]*np.ones((kmlist[ti].n_clusters,1)), offset_Y+scale_Y*kmlist[ti].cluster_centers_), axis=1)
            #centers_t = np.concatenate((kmtime[ti]*np.ones((kmlist[ti].n_clusters,1)), kmlist[ti].cluster_centers_), axis=1)
            # Re-season the centers
            centers_t = winstrat.reseason(kmtime[ti]*np.ones(kmlist[ti].n_clusters),kmlist[ti].cluster_centers_,None,szn_mean_Y,szn_std_Y)
            centers += [np.zeros((centers_t.shape[0],centers_t.shape[1]+1))]
            centers[-1][:,0] = kmtime[ti]
            centers[-1][:,1:] = centers_t
            ina += [winstrat.ina_test(centers[-1],feat_def,self.tpt_bndy)]
            inb += [winstrat.inb_test(centers[-1],feat_def,self.tpt_bndy)]
        #km = pickle.load(open(clust_filename,"rb"))
        P_list = pickle.load(open(msm_filename,"rb"))
        # Check rowsums
        minrowsums = np.inf
        mincolsums = np.inf
        for i in range(len(P_list)):
            rowsums = np.array(P_list[i].sum(1)).flatten()
            minrowsums = min(minrowsums,np.min(rowsums))
            mincolsums = min(mincolsums,np.min(P_list[i].sum(0)))
            #print("rowsums: min={}, max={}".format(rowsums.min(),rowsums.max()))
        init_dens = np.array([np.sum(kmlist[0].labels_ == i) for i in range(kmlist[0].n_clusters)], dtype=float)
        # Density and committors
        init_dens *= 1.0/np.sum(init_dens)
        init_dens = np.maximum(init_dens, np.max(init_dens)*1e-4)
        pi = self.compute_tdep_density(P_list,init_dens,kmtime)
        piflat = np.concatenate(pi)
        print("ina[0].shape = {}, inb[0].shape = {}, P_list[0].shape = {}".format(ina[0].shape,inb[0].shape,P_list[0].shape))
        qm = self.compute_backward_committor(P_list,kmtime,ina,inb,pi)
        qmflat = np.concatenate(qm)
        qp = self.compute_forward_committor(P_list,kmtime,ina,inb)
        qpflat = np.concatenate(qp)
        # Integral to B
        dam_centers = [np.ones(kmlist[ti].n_clusters) for ti in range(len(kmtime))]
        #dam_centers[0][:] = (kmtime[1] - kmtime[0])/2
        #dam_centers[-1][:] = (kmtime[-1] - kmtime[-2])/2
        #for ti in range(1,len(kmtime)-1):
        #    dam_centers[ti][:] = (kmtime[ti+1] - kmtime[ti-1])/2
        int2b = {}
        int2b_cond = {}
        int2b['time'] = self.compute_integral_to_B(P_list,kmtime_d,ina,inb,qp,dam_centers,maxmom=3)
        int2b_cond['time'] = self.conditionalize_int2b(P_list,kmtime_d,int2b['time'],qp)
        # Rate
        # To compute the rate, must sum over all fluxes going into B.
        flux = []
        rate_froma = 0
        rate_tob = 0
        flux_froma = []
        flux_tob = []
        flux_dens_tob = np.zeros(len(kmtime))
        for ti in range(len(kmtime)-1):
            flux += [(P_list[ti].T * pi[ti] * qm[ti]).T * qp[ti+1]]
            #print(f"ti = {ti}, P_list[ti].shape = {P_list[ti].shape}, pi[ti].shape = {pi[ti].shape}, ina[ti].shape = {ina[ti].shape}, qp[ti].shape = {qp[ti].shape}")
            flux_froma += [(P_list[ti].T * pi[ti] * ina[ti]).T * qp[ti+1]]
            flux_tob += [(P_list[ti].T * pi[ti] * qm[ti]).T * inb[ti+1]]
            rate_froma += np.sum(flux_froma[-1])
            rate_tob += np.sum(flux_tob[-1])
            flux_dens_tob[ti] = np.sum(flux_tob[-1])
        print(f"flux_dens_tob: min={flux_dens_tob.min()},max={flux_dens_tob.max()}")
        # Smooth out flux_dens_tob
        hist,t_hist = np.histogram(kmtime_d, weights=flux_dens_tob, bins=15)
        t_hist = (t_hist[1:] + t_hist[:-1])/2
        #ti_froma = np.where(winstrat.wtime > self.tpt_bndy['tthresh'][0])[0][0]
        if np.abs(rate_froma - rate_tob) > 0.1*max(rate_froma, rate_tob):
            raise Exception(f"Rate discrepancy: froma is {rate_froma}, tob is {rate_tob}")
        print(f"Rate: froma={rate_froma}, tob={rate_tob}")
        #rate = np.sum(flux[ti_froma])
        pickle.dump(qp,open(join(savedir,"qp"),"wb"))
        pickle.dump(qm,open(join(savedir,"qm"),"wb"))
        pickle.dump(pi,open(join(savedir,"pi"),"wb"))
        pickle.dump(int2b,open(join(savedir,"int2b"),"wb"))
        pickle.dump(int2b_cond,open(join(savedir,"int2b_cond"),"wb"))
        pickle.dump(ina,open(join(savedir,"ina"),"wb"))
        pickle.dump(inb,open(join(savedir,"inb"),"wb"))
        pickle.dump(centers,open(join(savedir,"centers"),"wb"))
        # Do the time-dependent Markov Chain analysis
        summary = {"rate_froma": rate_froma, "rate_tob": rate_tob,}
        pickle.dump(summary,open(join(savedir,"summary"),"wb"))
        # Plot 
        if plot_field_flag:
            centers_all = np.concatenate(centers, axis=0)
            #keypairs = [['time_d','area'],['time_d','centerlat'],['time_d','uref'],['time_d','asprat'],['time_d','kurt'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4']][:5]
            keypairs = [['time_d','uref_dl0']]
            keypairs += [['time_d','pc%i_lev0'%(i_pc)] for i_pc in range(algo_params['Npc_per_level'][0])]
            keypairs += [['time_d','captemp_lev%i'%(i_lev)] for i_lev in np.where(algo_params["captemp_flag"])[0]]
            keypairs += [['time_d','heatflux_lev%i'%(i_lev)] for i_lev in np.where(algo_params["captemp_flag"])[0]]
            #keypairs += [['time_d','vxmom%i'] for i in range(algo_params['num_vortex_moments'])]
            #keypairs += [['pc1_lev0','pc3_lev0']]
            #keypairs += [['uref_dl0','uref_dl%i'%(i_dl)] for i_dl in range(1,min(5,winstrat.ndelay))]
            for i_kp in range(len(keypairs)):
                fun0name,fun1name = [funlib[keypairs[i_kp][j]]["label"] for j in range(2)]
                theta_x = np.array([funlib[keypairs[i_kp][j]]["fun"](centers_all) for j in range(2)]).T
                Jth = self.project_current(theta_x,kmtime,centers,flux)

                # Plot density
                fig,ax = helper.plot_field_2d(piflat,np.ones(len(centers_all)),theta_x,shp=[15,15],fieldname="Density",fun0name=fun0name,fun1name=fun1name,contourflag=True,avg_flag=False,logscale=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    print(f"flux_dens_tob: min={flux_dens_tob.min()}, max={flux_dens_tob.max()}")
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"pi_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                # Plot committor
                fig,ax = helper.plot_field_2d(qpflat,piflat,theta_x,shp=[15,15],fieldname="Committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"qp_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                # Plot backward committor
                fig,ax = helper.plot_field_2d(qmflat,piflat,theta_x,shp=[15,15],fieldname="Backward committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"qm_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                # Plot integrals to B
                fig,ax = helper.plot_field_2d(np.concatenate(int2b_cond['time']['mean']),piflat,theta_x,shp=[15,15],fieldname="Lead time",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"lt_mean_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                fig,ax = helper.plot_field_2d(np.concatenate(int2b_cond['time']['std']),piflat,theta_x,shp=[15,15],fieldname="Lead time std.",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"lt_std_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
                fig,ax = helper.plot_field_2d(np.concatenate(int2b_cond['time']['skew']),piflat,theta_x,shp=[15,15],fieldname="Lead time skew",fun0name=fun0name,fun1name=fun1name,contourflag=True)
                self.plot_current_overlay(theta_x,Jth,np.ones(len(centers_all)),fig,ax)
                if keypairs[i_kp][0] == 'time_d' and keypairs[i_kp][1] == 'uref':
                    ax.plot(t_hist,self.tpt_bndy['uthresh_b']+hist/np.max(hist)*10,color='black',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_b'],color='black',linestyle='--',zorder=5)
                    ax.axhline(y=self.tpt_bndy['uthresh_a'],color='black',linestyle='--',zorder=5)
                fig.savefig(join(savedir,"lt_skew_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
                plt.close(fig)
        return summary 
    def project_current(self,theta_flat,time,centers,flux):
        # For a given vector-valued observable theta, evaluate the current operator J dot grad(theta)
        # theta is a list of (Mt x d) arrays, where d is the dimensionality of theta and Mt is the number of clusters at time step t
        klast = np.where(time < self.tpt_bndy['tthresh'][1])[0][-1] + 2
        thdim = theta_flat.shape[1]
        nclust_list = np.array([centers[ti].shape[0] for ti in range(len(centers))])
        Jth = np.zeros((np.sum(nclust_list),thdim))
        Jmag = np.zeros(np.sum(nclust_list))  # Magnitude of the current
        i1 = 0
        for k in range(klast):
            i2 = i1 + nclust_list[k]
            if k > 0:
                bwd_weight = 0.5*(k < len(time)-1) + 1.0*(k == len(time)-1)
                i0 = i1 - nclust_list[k-1]
                for j in range(thdim):
                    Jth[i1:i2,j] += bwd_weight*np.sum(flux[k-1]*np.add.outer(-theta_flat[i0:i1,j], theta_flat[i1:i2,j]), axis=0)
            if k < klast-1:
                fwd_weight = 0.5*(k > 0) + 1.0*(k == 0)
                i3 = i2 + nclust_list[k+1]
                for j in range(thdim):
                    #print("shapes: Jth[i1:i2,j]: {}, flux[k]: {}, theta_flat[i1:i2,j]: {}, theta_flat[i2:i3,j]: {}".format(Jth[i1:i2,j].shape,flux[k].shape,theta_flat[i1:i2,j].shape, theta_flat[i2:i3,j].shape))
                    Jth[i1:i2,j] += fwd_weight*np.sum(flux[k]*np.add.outer(-theta_flat[i1:i2,j], theta_flat[i2:i3,j]), axis=1)
            i1 = i2
        return Jth
    def plot_current_overlay(self,theta_x,Jth,weight,fig,ax):
        # Plot a field on a (time,var1) plane.
        # field must be a list of arrays in the same shape as the state space
        Nx = len(theta_x)
        shp = 20*np.ones(2, dtype=int)
        _,dth,thaxes,_,J0_proj,_,_,_,bounds = helper.project_field(Jth[:,0],weight,theta_x,shp=shp,avg_flag=False)
        _,_,_,_,J1_proj,_,_,_,_ = helper.project_field(Jth[:,1],weight,theta_x,shp=shp,bounds=bounds,avg_flag=False)
        Jmag = np.sqrt(J0_proj**2 + J1_proj**2)
        minmag,maxmag = np.nanmin(Jmag),np.nanmax(Jmag)
        coeff1 = 3.0/maxmag
        dsmin,dsmax = np.max(shp)/200,np.max(shp)/15
        coeff0 = dsmax / (np.exp(-coeff1 * maxmag) - 1)
        ds = coeff0 * (np.exp(-coeff1 * Jmag) - 1)
        #ds = dsmin + (dsmax - dsmin)*(Jmag - minmag)/(maxmag - minmag)
        normalizer = ds*(Jmag != 0)/(np.sqrt((J0_proj/(dth[0]))**2 + (J1_proj/(dth[1]))**2) + (Jmag == 0))
        J0_proj *= normalizer*(1 - np.isnan(J0_proj))
        J1_proj *= normalizer*(1 - np.isnan(J1_proj))
        th01,th10 = np.meshgrid(thaxes[0],thaxes[1],indexing='ij') 
        ax.quiver(th01,th10,J0_proj,J1_proj,angles='xy',scale_units='xy',scale=1.0,color='black',width=1.5,headwidth=4.4,units='dots',zorder=4)
        # Maybe plot a flux distribution across a surface
        return th01,th10,J0_proj,J1_proj



