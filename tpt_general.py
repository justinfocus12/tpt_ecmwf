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
    def interpolate_field_clust2data(self,kmdict,Ny,Nt,field_clust,density_flag=False,idx_a=None,idx_b=None,val_a=None,val_b=None):
        # field_clust is a list of arrays, one array for each time step in the winter, and has a value for each cluster center. This function interpolates the field onto data points (in feature space). 
        kmlist,kmtime,kmidx = kmdict["kmlist"],kmdict["kmtime"],kmdict["kmidx"]
        #print(f"n_clusters: {[km.n_clusters for km in kmlist]}")
        #print(f"len(idx): {[len(idx_cl) for idx_cl in kmidx]}")
        field_Y = np.zeros(Ny*Nt)
        for i_time in range(len(kmlist)):
            idx_Y = np.array(kmidx[i_time])
            if len(idx_Y) > 0:
                #print(f"idx_Y: min={idx_Y.min()}, max={idx_Y.max()}, len={len(idx_Y)}")
                #print(f"nclust = {kmlist[i_time].n_clusters}")
                for i_cl in range(kmlist[i_time].n_clusters):
                    idx_cl = np.array(np.where(kmlist[i_time].labels_ == i_cl)[0])
                    if len(idx_cl) > 0:
                        #print(f"idx_cl: min={idx_cl.min()}, max={idx_cl.max()}, len={len(idx_cl)}")
                        field_Y[idx_Y[idx_cl]] = field_clust[i_time][i_cl] 
                        if density_flag:
                            field_Y[idx_Y[idx_cl]] *= 1.0/len(idx_cl) # So the change of measure normalizes
        # TODO: spruce this up with continuous interpolation
        if idx_a is not None and val_a is not None:
            field_Y[idx_a] = val_a
        if idx_b is not None and val_b is not None:
            field_Y[idx_b] = val_b
        return field_Y.reshape((Ny,Nt))
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
        kmidx = [] # At each time step, which snapshots have the same time. 
        for ti in range(winstrat.Ntwint):
            idx = np.where(np.abs(Y[:,winstrat.fidx_Y['time_h']] - winstrat.wtime[ti]) < winstrat.szn_hour_window/2)[0]
            if len(idx) == 0:
                print("WARNING, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(ti,Y[:,0].min(),Y[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
            else:
                kmidx.append(idx)
                kmtime.append(winstrat.wtime[ti])
                km = MiniBatchKMeans(min(len(idx),num_clusters),random_state=0).fit(Y_unseasoned[idx])
                print(f"ti = {ti}, km.n_clusters = {km.n_clusters}, len(idx) = {len(idx)}")
                kmlist.append(km)
        kmdict = {"kmlist": kmlist, "kmtime": kmtime, "kmidx": kmidx,}
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
        pickle.dump(flux,open(join(savedir,"flux"),"wb"))
        pickle.dump(int2b,open(join(savedir,"int2b"),"wb"))
        pickle.dump(int2b_cond,open(join(savedir,"int2b_cond"),"wb"))
        pickle.dump(ina,open(join(savedir,"ina"),"wb"))
        pickle.dump(inb,open(join(savedir,"inb"),"wb"))
        pickle.dump(centers,open(join(savedir,"centers"),"wb"))
        np.save(join(savedir,"kmtime"),kmtime)
        # Do the time-dependent Markov Chain analysis
        summary = {"rate_froma": rate_froma, "rate_tob": rate_tob,}
        pickle.dump(summary,open(join(savedir,"summary"),"wb"))
        
        # ------------------------------------------------------------------------------------
        # Interpolate quantities of interest onto all the points. Enforce boundary conditions. 
        # ------------------------------------------------------------------------------------
        ina_Y = winstrat.ina_test(Y.reshape((Ny*Nt,ydim)),feat_def,self.tpt_bndy)
        inb_Y = winstrat.inb_test(Y.reshape((Ny*Nt,ydim)),feat_def,self.tpt_bndy)
        idx_a = np.where(ina_Y)[0]
        idx_b = np.where(inb_Y)[0]
        np.save(join(savedir,"ina_Y"),ina_Y)
        np.save(join(savedir,"inb_Y"),inb_Y)
        # Committor
        qp_Y = self.interpolate_field_clust2data(kmdict,Ny,Nt,qp,density_flag=False,idx_a=idx_a,idx_b=idx_b,val_a=0.0,val_b=1.0)
        np.save(join(savedir,"qp_Y"),qp_Y)
        # Backward committor
        qm_Y = self.interpolate_field_clust2data(kmdict,Ny,Nt,qm,density_flag=False,idx_a=idx_a,idx_b=idx_b,val_a=1.0,val_b=0.0)
        np.save(join(savedir,"qm_Y"),qm_Y)
        # Change of measure
        pi_Y = self.interpolate_field_clust2data(kmdict,Ny,Nt,pi,density_flag=True)
        np.save(join(savedir,"pi_Y"),pi_Y)
        # Lead time 
        for mom_name in int2b_cond['time'].keys():
            lt_Y = self.interpolate_field_clust2data(kmdict,Ny,Nt,int2b_cond['time'][mom_name],density_flag=False,idx_a=idx_a,idx_b=idx_b)
            np.save(join(savedir,"lt_%s_Y"%(mom_name)),lt_Y)
        return summary 
    def plot_results_data(self,feat_filename,tpt_feat_filename,feat_filename_ra,tpt_feat_filename_ra,feat_def,savedir,winstrat,algo_params,spaghetti_flag=True,fluxdens_flag=True,current2d_flag=True):
        # Load the reanalysis data for comparison
        Xra = np.load(feat_filename_ra)[:,winstrat.ndelay-1:,:]
        tpt_feat_ra = pickle.load(open(tpt_feat_filename_ra,"rb"))
        Yra = tpt_feat_ra["Y"]
        Nyra,Ntyra,ydim = Yra.shape
        Nxra,Ntxra,xdim = Xra.shape
        if not (Nyra == Nxra and Ntyra == Ntxra):
            raise Exception(f"ERROR: Xra and Yra have shapes {Xra.shape} and {Yra.shape} respectively. The first two dimensions should match")
        ina_yra = winstrat.ina_test(Yra.reshape((Nyra*Ntyra,ydim)),feat_def,self.tpt_bndy).reshape(Nyra*Ntyra)
        inb_yra = winstrat.inb_test(Yra.reshape((Nyra*Ntyra,ydim)),feat_def,self.tpt_bndy).reshape(Nyra*Ntyra)
        # Get source and destination
        src_tag,dest_tag = winstrat.compute_src_dest_tags(Yra,feat_def,self.tpt_bndy)
        print(f"src_tag.shape = {src_tag.shape}, dest_tag.shape = {dest_tag.shape}, Yra.shape = {Yra.shape}")
        # Restrict reanalysis to the midwinter
        winter_flag_ra = (Yra[:,:,winstrat.fidx_Y['time_h']] >= self.tpt_bndy['tthresh'][0])*(Yra[:,:,winstrat.fidx_Y['time_h']] <= self.tpt_bndy['tthresh'][1])
        winter_flag_ra = winter_flag_ra.reshape(Nyra*Ntyra)
        Yra = Yra.reshape((Nyra*Ntyra,ydim))
        Xra = Xra.reshape((Nyra*Ntyra,xdim))
        src_tag = src_tag.reshape(Nyra*Ntyra)
        dest_tag = dest_tag.reshape(Nyra*Ntyra)
        # Plot fields using the data points rather than the clusters
        funlib_Y = winstrat.observable_function_library_Y(algo_params)
        funlib_X = winstrat.observable_function_library_X()
        tpt_feat = pickle.load(open(tpt_feat_filename, "rb"))
        Y,szn_mean_Y,szn_std_Y,idx_resamp = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y","idx_resamp"]]
        X = np.load(feat_filename)[:,winstrat.ndelay-1:,:][idx_resamp]
        Ny,Nty,ydim = Y.shape
        Nx,Ntx,xdim = X.shape
        if not (Ntx==Nty and Nx==Ny):
            raise Exception(f"ERROR: X.shape = {X.shape} and Y.shape = {Y.shape}")
        # Keep track of which entries are the beginning of a short trajectory
        Y = Y.reshape((Ny*Nty,ydim))
        X = X.reshape((Ny*Nty,xdim))
        traj_rank = np.outer(np.ones(Ny), np.arange(Nty)).flatten()
        winter_flag = (Y[:,winstrat.fidx_Y['time_h']] >= self.tpt_bndy['tthresh'][0])*(Y[:,winstrat.fidx_Y['time_h']] <= self.tpt_bndy['tthresh'][1])
        idx_winter = np.where(winter_flag)[0]
        traj_start_positions = np.where(traj_rank == 0)[0]
        winter_starts = np.intersect1d(idx_winter,traj_start_positions)
        winter_fully_idx = np.where(np.all(winter_flag.reshape((Ny,Nty)), axis=1))[0] # Indices within the (Ny,Nty,ydim)-shaped array
        print(f"winter_fully_idx = {winter_fully_idx}")
        qp_Y = np.load(join(savedir,"qp_Y.npy")).reshape(Ny*Nty)
        qm_Y = np.load(join(savedir,"qm_Y.npy")).reshape(Ny*Nty)
        pi_Y = np.load(join(savedir,"pi_Y.npy")).reshape(Ny*Nty)
        lt_mean = np.load(join(savedir,"lt_mean_Y.npy")).reshape(Ny*Nty)
        lt_std = np.load(join(savedir,"lt_std_Y.npy")).reshape(Ny*Nty)
        lt_skew = np.load(join(savedir,"lt_skew_Y.npy")).reshape(Ny*Nty)
        if fluxdens_flag:
            # ----------- Plot flux distribution of entry times ----------
            theta_normal = funlib_X['uref']['fun'](X)
            theta_normal_label = funlib_X['uref']['label']
            theta_tangential = funlib_X['time_d']['fun'](X)
            theta_tangential_label = funlib_X['time_d']['label']
            theta_mid_list = np.array([5,0,-5,-10,-15,-20,-25], dtype=float)
            theta_lower_list = theta_mid_list - 1.0
            theta_upper_list = theta_mid_list + 1.0
            fig,ax = self.plot_flux_distributions_1d(qm_Y.reshape((Ny,Nty))[winter_fully_idx],qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],theta_normal.reshape((Ny,Nty))[winter_fully_idx],theta_tangential.reshape((Ny,Nty))[winter_fully_idx],theta_normal_label,theta_tangential_label,theta_lower_list,theta_upper_list)
            fig.savefig(join(savedir,"fluxdens_J-uref_d-timed"))
            plt.close(fig)
            theta_normal = funlib_X['time_d']['fun'](X)
            theta_normal_label = funlib_X['time_d']['label']
            theta_tangential = funlib_X['uref']['fun'](X)
            theta_tangential_label = funlib_X['uref']['label']
            theta_mid_list = np.array([50,75,100,125], dtype=float)
            theta_lower_list = theta_mid_list - 5.0
            theta_upper_list = theta_mid_list + 5.0
            fig,ax = self.plot_flux_distributions_1d(qm_Y.reshape((Ny,Nty))[winter_fully_idx],qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],theta_normal.reshape((Ny,Nty))[winter_fully_idx],theta_tangential.reshape((Ny,Nty))[winter_fully_idx],theta_normal_label,theta_tangential_label,theta_lower_list,theta_upper_list)
            fig.savefig(join(savedir,"fluxdens_J-timed_d-uref"))
            plt.close(fig)
        if current2d_flag:
            # ------------- Current plots (Y) --------------------
            keypairs = []
            keypairs += [['uref_dl0','uref_dl%i'%(i_dl)] for i_dl in np.arange(5,winstrat.ndelay,5)]
            keypairs += [['time_d','uref_dl0']]
            for key0,key1 in keypairs:
                print(f"Plotting current on key pair {key0},{key1}")
                theta_x = np.array([funlib_Y[key0]["fun"](Y), funlib_Y[key1]["fun"](Y)]).T
                theta_x_ra = np.array([funlib_Y[key0]["fun"](Yra), funlib_Y[key1]["fun"](Yra)]).T
                # A -> B
                fig,ax = helper.plot_field_2d((qp_Y*qm_Y)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$A\to B$",fun0name=funlib_Y[key0]["label"],fun1name=funlib_Y[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],qm_Y.reshape((Ny,Nty))[winter_fully_idx],qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = ((src_tag==0)*(dest_tag==1)*winter_flag_ra).reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax)
                fig.savefig(join(savedir,"J_%s_%s_ab"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
                # A -> A
                fig,ax = helper.plot_field_2d(((1-qp_Y)*qm_Y)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$A\to A$",fun0name=funlib_Y[key0]["label"],fun1name=funlib_Y[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],qm_Y.reshape((Ny,Nty))[winter_fully_idx],1-qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = ((src_tag==0)*(dest_tag==0)*winter_flag_ra).reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax)
                fig.savefig(join(savedir,"J_%s_%s_aa"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
                # A or B -> A or B
                fig,ax = helper.plot_field_2d(np.ones(Ny*Nty)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$\mathrm{Steady-state}$",fun0name=funlib_Y[key0]["label"],fun1name=funlib_Y[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],np.ones((Ny,Nty))[winter_fully_idx],np.ones((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = winter_flag_ra.reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax,debug=False)
                fig.savefig(join(savedir,"J_%s_%s"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
            # ------------------ Current plots outside the TPT observables ------------
            funlib_X = winstrat.observable_function_library_X()
            keypairs = []
            #keypairs += [['time_d','captemp_lev%i'%(i_lev)] for i_lev in range(Nlev)]
            #keypairs += [['time_d','heatflux_lev%i'%(i_lev)] for i_lev in range(Nlev)]
            #keypairs += [['time_d','vxmom%i'%(i_mom)] for i_mom in range(winstrat.num_vortex_moments_max)]
            keypairs += [['time_d','uref']]
            #keypairs += [['time_d','pc%i_lev0'%(i_pc)] for i_pc in range(algo_params['Npc_per_level'][0])]
            keypairs += [['uref','pc%i_lev0'%(i_pc)] for i_pc in range(6)]
            keypairs += [['pc1_lev0','pc%i_lev0'%(i_pc)] for i_pc in range(2,6)]
            for key0,key1 in keypairs:
                print(f"Plotting current on key pair {key0},{key1}")
                theta_x = np.array([funlib_X[key0]["fun"](X), funlib_X[key1]["fun"](X)]).T
                theta_x_ra = np.array([funlib_X[key0]["fun"](Xra), funlib_X[key1]["fun"](Xra)]).T
                print(f"theta_x_ra.shape = {theta_x_ra.shape}")
                # A -> B
                fig,ax = helper.plot_field_2d((qp_Y*qm_Y)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$A\to B$",fun0name=funlib_X[key0]["label"],fun1name=funlib_X[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],qm_Y.reshape((Ny,Nty))[winter_fully_idx],qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = ((src_tag==0)*(dest_tag==1)*winter_flag_ra).reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax)
                fig.savefig(join(savedir,"J_%s_%s_ab"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
                # A -> A
                fig,ax = helper.plot_field_2d(((1-qp_Y)*qm_Y)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$A\to A$",fun0name=funlib_X[key0]["label"],fun1name=funlib_X[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],qm_Y.reshape((Ny,Nty))[winter_fully_idx],1-qp_Y.reshape((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = ((src_tag==0)*(dest_tag==0)*winter_flag_ra).reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax)
                fig.savefig(join(savedir,"J_%s_%s_aa"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
                # (A or B) -> (A or B)
                fig,ax = helper.plot_field_2d(np.ones((Ny*Nty))[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=r"$\mathrm{Steady-state}$",fun0name=funlib_X[key0]["label"],fun1name=funlib_X[key1]["label"],avg_flag=False,logscale=True,cmap=plt.cm.YlOrBr)
                _,_,_,_ = self.plot_current_overlay_data(theta_x.reshape((Ny,Nty,2))[winter_fully_idx],np.ones((Ny,Nty))[winter_fully_idx],np.ones((Ny,Nty))[winter_fully_idx],pi_Y.reshape((Ny,Nty))[winter_fully_idx],fig,ax)
                reactive_flag = winter_flag_ra.reshape((Nyra,Ntyra))
                self.plot_trajectory_segments(theta_x_ra.reshape((Nyra,Ntyra,2)),reactive_flag,fig,ax)
                fig.savefig(join(savedir,"J_%s_%s"%(key0.replace("_",""),key1.replace("_",""))))
                plt.close(fig)
        if spaghetti_flag:
            # ----------- New plot: short histories of zonal wind, colored by committor. Maybe this will inform new coordinates --------------
            fig,ax = plt.subplots()
            prng = np.random.RandomState(1)
            ss = prng.choice(winter_starts,size=min(len(winter_starts),500),replace=False) 
            print(f"qp on ss: min={np.min(qp_Y[ss])}, max={np.max(qp_Y[ss])}")
            time_full = np.arange(winstrat.ndelay)*winstrat.dtwint/24.0
            time_full -= time_full[-1]
            uref_idx = np.array([winstrat.fidx_Y["uref_dl%i"%(i_dl)] for i_dl in range(winstrat.ndelay)])[::-1]
            uref = Y[:,uref_idx]
            for i_y in ss:
                time_i = time_full + Y[i_y,winstrat.fidx_Y['time_h']]/24.0
                ax.plot(time_i,uref[i_y],color=plt.cm.coolwarm(qp_Y[i_y]),alpha=0.4)
            ax.set_xlabel(funlib_Y["time_d"]["label"])
            ax.set_ylabel(funlib_Y["uref_dl0"]["label"])
            fig.savefig(join(savedir,"qp_uref_spaghetti"))
            plt.close(fig)
            print("saved spaghetti, now moving on...")
            # ------------ New plot: vertical profiles of zonal wind colored by committor. -------------
            Nlev = len(feat_def['plev'])
            fig,ax = plt.subplots()
            prng = np.random.RandomState(1)
            ss = prng.choice(winter_starts,size=min(len(winter_starts),1000)) # Make sure it's always at the beginning
            ubar_idx = np.array([winstrat.fidx_X["ubar_60N_lev%i"%(i_lev)] for i_lev in range(Nlev)])
            ubar = X[:,ubar_idx]
            for i_x in ss:
                ax.plot(ubar[i_x],-7*np.log(feat_def["plev"]/feat_def["plev"][-1]),color=plt.cm.coolwarm(qp_Y[i_x]),alpha=0.1,linewidth=2)
            ax.set_xlabel(r"$\overline{u}$ [m/s]")
            ax.set_ylabel(r"Pseudo-height [km]")
            fig.savefig(join(savedir,"qp_uprofile_spaghetti"))
            plt.close(fig)
            # ------------ New plot: vertical profiles of heat flux colored by committor. -------------
            Nlev = len(feat_def['plev'])
            fig,ax = plt.subplots()
            prng = np.random.RandomState(1)
            ss = prng.choice(winter_starts,size=min(len(winter_starts),1000)) # The Nty makes sure it's always at the beginning
            vT_idx = np.array([winstrat.fidx_X["heatflux_lev%i"%(i_lev)] for i_lev in range(Nlev)])
            vT = X[:,vT_idx]
            for i_x in ss:
                ax.plot(vT[i_x],-7*np.log(feat_def["plev"]/feat_def["plev"][-1]),color=plt.cm.coolwarm(qp_Y[i_x]),alpha=0.1,linewidth=2)
            ax.set_xlabel(r"$\overline{v'T'}$ [K$\cdot$m/s]")
            ax.set_ylabel(r"Pseudo-height [km]")
            fig.savefig(join(savedir,"qp_vTprofile_spaghetti"))
            plt.close(fig)
        return
    def plot_results_clust(self,feat_def,savedir,winstrat,algo_params):
        qp = pickle.load(open(join(savedir,"qp"),"rb"))
        qm = pickle.load(open(join(savedir,"qm"),"rb"))
        pi = pickle.load(open(join(savedir,"pi"),"rb"))
        flux = pickle.load(open(join(savedir,"flux"),"rb"))
        int2b = pickle.load(open(join(savedir,"int2b"),"rb"))
        int2b_cond = pickle.load(open(join(savedir,"int2b_cond"),"rb"))
        ina = pickle.load(open(join(savedir,"ina"),"rb"))
        inb = pickle.load(open(join(savedir,"inb"),"rb"))
        centers = pickle.load(open(join(savedir,"centers"),"rb"))
        kmtime = np.load(join(savedir,"kmtime.npy"))
        # ---------- Flatten some fields ------
        qpflat = np.concatenate(qp)
        qmflat = np.concatenate(qm)
        piflat = np.concatenate(pi)
        # ------------ Plot -------------------
        funlib = winstrat.observable_function_library_Y(algo_params)
        # Plot distribution of uref across different committor level sets
        # Jab dot grad (qp) d(uref)
        theta_normal_flat = qpflat
        theta_normal_label = r"$q_B^+$"
        theta_lower_list = [0.2,0.45,0.7]
        theta_upper_list = [0.3,0.55,0.8]
        theta_tangential_flat = funlib["uref_dl0"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["uref_dl0"]["label"]
        fig,ax = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux,theta_lower_list=theta_lower_list,theta_upper_list=theta_upper_list)
        fig.savefig(join(savedir,"Jab-qp_d-uref"))
        plt.close(fig)
        # Jab dot grad (qp) d(time)
        theta_normal_flat = qpflat
        theta_normal_label = r"$q_B^+$"
        theta_lower_list = [0.2,0.45,0.7]
        theta_upper_list = [0.3,0.55,0.8]
        theta_tangential_flat = funlib["time_d"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["time_d"]["label"]
        fig,ax = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux,theta_lower_list=theta_lower_list,theta_upper_list=theta_upper_list)
        fig.savefig(join(savedir,"Jab-qp_d-timed"))
        plt.close(fig)
        # Jab dot grad (-leadtime) d(time)
        theta_normal_flat = -np.concatenate(int2b['time'][0])
        theta_normal_label = r"$-\eta_B^+$"
        theta_tangential_flat = funlib["uref_dl0"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["uref_dl0"]["label"]
        fig,ax = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux)
        fig.savefig(join(savedir,"Jab-tbd_d-uref"))
        plt.close(fig)
        # Jab dot grad (time) d(uref)
        theta_normal_flat = funlib["time_d"]["fun"](np.concatenate(centers,axis=0))
        theta_normal_label = funlib["time_d"]["label"]
        theta_tangential_flat = funlib["uref_dl0"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["uref_dl0"]["label"]
        fig,ax = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux)
        fig.savefig(join(savedir,"Jab-timed_d-uref"))
        plt.close(fig)
        # Jab dot grad (uref) d(time)
        theta_normal_flat = funlib["uref_dl0"]["fun"](np.concatenate(centers,axis=0))
        theta_normal_label = funlib["uref_dl0"]["label"]
        theta_tangential_flat = funlib["time_d"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["time_d"]["label"]
        fig,ax = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux)
        fig.savefig(join(savedir,"Jab-uref_d-timed"))
        plt.close(fig)
        centers_all = np.concatenate(centers, axis=0)
        #keypairs = [['time_d','area'],['time_d','centerlat'],['time_d','uref'],['time_d','asprat'],['time_d','kurt'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4']][:5]
        keypairs = [['time_d','uref_dl0']]
        #keypairs += [['time_d','pc%i_lev0'%(i_pc)] for i_pc in range(algo_params['Npc_per_level'][0])]
        #keypairs += [['time_d','captemp_lev%i'%(i_lev)] for i_lev in np.where(algo_params["captemp_flag"])[0]]
        #keypairs += [['time_d','heatflux_lev%i'%(i_lev)] for i_lev in np.where(algo_params["captemp_flag"])[0]]
        #keypairs += [['time_d','vxmom%i'] for i in range(algo_params['num_vortex_moments'])]
        keypairs += [['pc1_lev0','pc3_lev0']]
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
        return 
    def reactive_flux_density_levelset(self,thmid,Jth,Jweight,theta_lower_list,theta_upper_list):
        # For all data points with a value of theta between theta_lower and theta_upper, find the reactive flux density on that level set. 
        # Assume the current has already been projected onto the observable and is given by Jth
        # theta_centers is a list of numpy arrays, one array for each time step.
        close_idx = []
        reactive_flux = []
        for i_theta in range(len(theta_lower_list)):
            theta_lower = theta_lower_list[i_theta]
            theta_upper = theta_upper_list[i_theta]
            close_idx_new = np.where((thmid >= theta_lower)*(thmid <= theta_upper))[0]
            close_idx.append(close_idx_new)
            reactive_flux.append((Jweight[close_idx_new]*Jth[close_idx_new,:].T).T)
            print(f"At level number {i_theta} out of {len(theta_lower_list)}, bounds = ({theta_lower},{theta_upper}). There are {len(close_idx_new)} new close indices")
        return close_idx,reactive_flux
    def plot_flux_distributions_1d(self,qm,qp,pi,theta_normal,theta_tangential,theta_normal_label,theta_tangential_label,theta_lower_list=None,theta_upper_list=None):
        # theta_normal and theta_tangential must both be scalar fields. 
        dth_tangential = (np.nanmax(theta_tangential) - np.nanmin(theta_tangential))/10
        theta_vec = np.transpose(np.array([theta_normal,theta_tangential]), (1,2,0))
        print(f"theta_vec.shape = {theta_vec.shape}")
        Jth,thmid,Jweight = self.project_current_data(theta_vec,qm,qp,pi)
        if theta_lower_list is None or theta_upper_list is None:
            theta_normal_min = np.nanmin(theta_normal)
            theta_normal_max = np.nanmax(theta_normal)
            theta_edges = np.linspace(theta_normal_min-1e-10,theta_normal_max+1e-10,4+1)
            theta_lower_list = theta_edges[:-1]
            theta_upper_list = theta_edges[1:]
        num_levels = len(theta_lower_list)
        close_idx,reactive_flux = self.reactive_flux_density_levelset(thmid[:,0],Jth,Jweight,theta_lower_list,theta_upper_list)
        fig,ax = plt.subplots()
        handles = []
        for i_thlev in range(num_levels):
            # Make a histogram of the reactive flux density distribution
            idx = close_idx[i_thlev]
            if len(idx) > 1:
                idx = np.array(idx)
                bins = max(int((np.nanmax(thmid[idx,1]) - np.nanmin(thmid[idx,1]))/dth_tangential), 3)
                weights = reactive_flux[i_thlev]
                x = thmid[idx,1]
                #print(f"weights.shape = {weights.shape}, x.shape = {x.shape}")
                hist,bin_edges = np.histogram(thmid[idx,1],weights=reactive_flux[i_thlev][:,0],bins=bins)
                print(f"At level {i_thlev}, bin range = {bin_edges[[0,-1]]}")
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                h, = ax.plot(bin_centers,hist,color=plt.cm.coolwarm((i_thlev+1)/(num_levels)),label=r"$%.2f$"%((theta_lower_list[i_thlev]+theta_upper_list[i_thlev])/2),marker='o')
                handles.append(h)
        ax.legend(handles=handles)
        ax.set_xlabel(theta_tangential_label,fontdict=font)
        ax.set_ylabel(r"d[%s]"%(theta_normal_label),fontdict=font)
        ax.axhline(y=0,color='black')
        return fig,ax
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
    def plot_trajectory_segments(self,theta_x,reactive_flag,fig,ax,zorder=10,numtraj=20,debug=False):
        # Plot trajectory segments in 2D given by theta_x. only plot the segments with reactive_flag on
        if theta_x.shape[2] != 2:
            raise Exception(f"ERROR: you gave me a data set of shape {theta_x.shape}, but I need dimension 2 to have size 2")
        prng = np.random.RandomState(2)
        ss = prng.choice(np.arange(len(theta_x)),size=numtraj,replace=False)
        print(f"reactive_flag.shape = {reactive_flag.shape}")
        print(f"theta_x.shape = {theta_x.shape}")
        for i in ss:
            ridx = np.where(reactive_flag[i])[0]
            xx = theta_x[i,ridx,:]
            #xx[np.where(reactive_flag[i]==0)[0],:] = np.nan
            #if debug: print(f"At index {i}, reactive_flag[i] = {reactive_flag[i]}")
            ax.plot(xx[:,0],xx[:,1],color='cyan',linewidth=1,zorder=zorder)
        if debug: sys.exit()
        return
    def project_current_data(self,theta_x,qm,qp,pi):
        Nx,Nt,thdim = theta_x.shape
        Jth = np.zeros((Nx,Nt-1,thdim))
        thmid = np.zeros((Nx,Nt-1,thdim))
        Jweight = np.zeros((Nx,Nt-1))
        for i_time in range(Nt-1):
            for i_th in range(thdim):
                Jth[:,i_time,i_th] = qm[:,i_time]*qp[:,i_time+1]*(theta_x[:,i_time+1,i_th] - theta_x[:,i_time,i_th])
                thmid[:,i_time,i_th] = 0.5*(theta_x[:,i_time,i_th] + theta_x[:,i_time+1,i_th])
                Jweight[:,i_time] = 0.5*(pi[:,i_time] + pi[:,i_time+1])
        thmid = thmid.reshape((Nx*(Nt-1),thdim))
        Jth = Jth.reshape((Nx*(Nt-1),thdim))
        Jweight = Jweight.flatten()
        return Jth,thmid,Jweight
    def plot_current_overlay_data(self,theta_x,qm,qp,pi,fig,ax):
        # Plot the current using data directly.
        # theta_x is a (Nx,Nt,2) array with trajectories separated from each other. 
        # Keep everything to a lag time of 1.0
        # 1. Forward component
        Jth,thmid,Jweight = self.project_current_data(theta_x,qm,qp,pi)
        print(f"Jth.shape = {Jth.shape}")
        shp,dth,thaxes,_,J0_proj,_,_,_,bounds = helper.project_field(Jth[:,0],Jweight.flatten(),thmid,avg_flag=False)
        _,_,_,_,J1_proj,_,_,_,_ = helper.project_field(Jth[:,1],Jweight.flatten(),thmid,shp=shp,bounds=bounds,avg_flag=False)
        Jmag = np.sqrt(J0_proj**2 + J1_proj**2)
        minmag,maxmag = np.nanmin(Jmag),np.nanmax(Jmag)
        coeff1 = 1.0/maxmag
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



