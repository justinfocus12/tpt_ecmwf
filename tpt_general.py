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
    def tpt_pipeline_dns(self,expdir,savedir,winstrat,feat_def):
        # Call winstrat and anything else we want. 
        # savedir: where all the results of this particular TPT will be saved.
        # winstrat: the object that was used to create features, and that can go on to evaluate other features.
        # feat_def: the feature definitions that came out of winstrat.
        X = np.load(join(expdir,"X.npy"))
        Nx,Nt,xdim = X.shape
        funlib = winstrat.observable_function_library()
        # ---- Plot committor in a few different coordinates -----
        src_tag,dest_tag = winstrat.compute_src_dest_tags(X,feat_def,self.tpt_bndy,"src_dest")
        qp = 1.0*(dest_tag == 1).flatten()
        weight = np.ones(Nx*Nt)/(Nx*Nt)
        keypairs = [['time_d','uref'],['time_d','mag1'],['time_d','lev0_pc0'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4'],['time_d','mag1_anomaly'],['time_d','mag2_anomaly']]
        for i_kp in range(len(keypairs)):
            fun0name,fun1name = [funlib[keypairs[i_kp][j]]["label"] for j in range(2)]
            theta_x = np.array([funlib[keypairs[i_kp][j]]["fun"](X.reshape((Nx*Nt,xdim))) for j in range(2)]).T
            fig,ax = helper.plot_field_2d(qp,weight,theta_x,shp=[15,15],fieldname="Committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
            fig.savefig(join(savedir,"qp_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
            plt.close(fig)
        # Compute density and backward committor
        # Compute rate
        rate = self.compute_rate_direct(src_tag,dest_tag)
        # Compute lead time
        # Compute current (??)
        result = {"qp": qp, "rate": rate}
        pickle.dump(result,open(join(savedir,"result"),"wb"))
        return result
    def cluster_features(self,clust_feat_filename,clust_filename,winstrat,num_clusters=100):
        # Read in a feature array from feat_filename and build clusters. Save cluster centers. Save them out in clust_filename.
        Y = np.load(clust_feat_filename)
        Nx,Nt,ydim = Y.shape
        Y = Y.reshape((Nx*Nt,ydim))
        # cluster based on the non-time features. 
        kmlist = []
        for ti in range(winstrat.Ntwint):
            idx = np.where(np.abs(Y[:,0] - winstrat.wtime[ti]) < winstrat.szn_hour_window/2)[0]
            if len(idx) == 0:
                raise Exception("Problem, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(ti,Y[:,0].min(),Y[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
            km = MiniBatchKMeans(num_clusters).fit(Y[idx,1:])
            kmlist += [km]
        pickle.dump(kmlist,open(clust_filename,"wb"))
        return
    def build_msm(self,clust_feat_filename,clust_filename,msm_filename,winstrat):
        nnk = 4 # Number of nearest neighbors for filling in empty positions
        Y = np.load(clust_feat_filename)
        Nx,Nt,ydim = Y.shape
        kmlist = pickle.load(open(clust_filename,"rb"))
        P = []

        #centers = np.concatenate((np.zeros(km.n_clusters,1),km.cluster_centers_), axis=1)
        for ti in range(winstrat.Ntwint-1):
            P += [np.zeros((kmlist[ti].n_clusters,kmlist[ti+1].n_clusters))]
            #centers[:,0] = winstrat.wtime[ti]
            #print("wtime[0] = {}".format(winstrat.wtime[0]))
            #print("X[0,:3,0] = {}".format(X[0,:3,0]))
            #print("X[:,:,0] range = {},{}".format(X[:,:,0].min(),X[:,:,0].max()))
            idx0 = np.where(np.abs(Y[:,:-1,0] - winstrat.wtime[ti]) < winstrat.dtwint/2) # (idx1_x,idx1_t) where idx1_x < Nx and idx1_t < Nt-1
            idx1 = np.where(np.abs(Y[:,1:,0] - winstrat.wtime[ti+1]) < winstrat.dtwint/2) # (idx1_x,idx1_t) where idx1_x < Nx and idx1_t < Nt-1
            if len(idx0[0]) > 0 and len(idx1[0]) > 0:
                overlap = np.where(np.subtract.outer(idx0[0],idx1[0]) == 0)
                labels0 = kmlist[ti].predict(Y[idx0[0][overlap[0]],idx0[1][overlap[0]],1:])
                labels1 = kmlist[ti+1].predict(Y[idx1[0][overlap[1]],idx1[1][overlap[1]],1:])
                print("labels0.shape = {}".format(labels0.shape))
                print("labels1.shape = {}".format(labels1.shape))
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
        G = []
        F = []
        for i in np.arange(mc.Nt-1,-1,-1):
            G += [1.0*ina[i]]
            if i < mc.Nt-1: F += [np.outer(1.0*(ina[i+1]==0)*(inb[i+1]==0), np.ones(mc.Nx[i]))]
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
    def tpt_pipeline_dga(self,feat_filename,clust_feat_filename,clust_filename,msm_filename,feat_def,savedir,winstrat):
        # Label each cluster as in A or B or elsewhere
        X = np.load(feat_filename)
        Y = np.load(clust_feat_filename)
        Nx,Nt,xdim = X.shape
        Nx,Nt,ydim = Y.shape
        funlib = winstrat.observable_function_library()
        uref_x = funlib["uref"]["fun"](X.reshape((Nx*Nt,xdim)))
        uref_y = funlib["uref"]["fun"](Y.reshape((Nx*Nt,ydim)))
        print("uref_x: min={}, max={}, mean={}".format(uref_x.min(),uref_x.max(),uref_x.mean()))
        print("uref_y: min={}, max={}, mean={}".format(uref_y.min(),uref_y.max(),uref_y.mean()))
        kmlist = pickle.load(open(clust_filename,"rb"))
        ina = []
        inb = []
        centers = []
        for ti in range(winstrat.Ntwint):
            centers_t = np.concatenate((winstrat.wtime[ti]*np.ones((kmlist[ti].n_clusters,1)),kmlist[ti].cluster_centers_), axis=1)
            centers += [centers_t]
            ina += [winstrat.ina_test(centers_t,feat_def,self.tpt_bndy)]
            inb += [winstrat.inb_test(centers_t,feat_def,self.tpt_bndy)]
        km = pickle.load(open(clust_filename,"rb"))
        P_list = pickle.load(open(msm_filename,"rb"))
        # Check rowsums
        minrowsums = np.inf
        mincolsums = np.inf
        for i in range(len(P_list)):
            rowsums = np.array(P_list[i].sum(1)).flatten()
            minrowsums = min(minrowsums,np.min(rowsums))
            mincolsums = min(mincolsums,np.min(P_list[i].sum(0)))
            #print("rowsums: min={}, max={}".format(rowsums.min(),rowsums.max()))
        print("minrowsums = {}, mincolsums = {}".format(minrowsums,mincolsums))
        init_dens = np.array([np.sum(km[0].labels_ == i) for i in range(km[0].n_clusters)], dtype=float)
        init_dens *= 1.0/np.sum(init_dens)
        init_dens = np.maximum(init_dens, np.max(init_dens)*1e-4)
        pi = self.compute_tdep_density(P_list,init_dens,winstrat.wtime)
        print("pi[0]: min={}, max={}".format(pi[0].min(),pi[0].max()))
        piflat = np.concatenate(pi)
        print("piflat: min={}, max={}".format(piflat.min(),piflat.max()))
        qm = self.compute_backward_committor(P_list,winstrat.wtime,ina,inb,pi)
        qmflat = np.concatenate(qm)
        print("qmflat: min={}, max={}".format(qmflat.min(),qmflat.max()))
        qp = self.compute_forward_committor(P_list,winstrat.wtime,ina,inb)
        qpflat = np.concatenate(qp)
        print("qpflat.shape = {}".format(qpflat.shape))
        print("qp: min={}, max={}, frac in (.2,.8) = {}".format(qpflat.min(),qpflat.max(),np.mean((qpflat>.2)*(qpflat<.8))))
        pickle.dump(qp,open(join(savedir,"qp"),"wb"))
        pickle.dump(qm,open(join(savedir,"qm"),"wb"))
        pickle.dump(pi,open(join(savedir,"pi"),"wb"))
        pickle.dump(ina,open(join(savedir,"ina"),"wb"))
        pickle.dump(inb,open(join(savedir,"inb"),"wb"))
        pickle.dump(centers,open(join(savedir,"centers"),"wb"))
        # Do the time-dependent Markov Chain analysis
        result_dga = {"qp": qp, "pi": pi,}
        # Plot 
        centers_all = np.concatenate(centers, axis=0)
        uref_call = funlib["uref"]["fun"](centers_all)
        print("uref_call: min={}, max={}, mean={}".format(uref_call.min(),uref_call.max(),uref_call.mean()))
        weight = np.ones(len(centers_all))/(len(centers_all))
        keypairs = [['time_d','uref'],['time_d','mag1'],['time_d','mag1_anomaly'],['time_d','mag2'],['time_d','mag2_anomaly'],['time_d','lev0_pc1']]
        #keypairs = [['time_d','uref'],['time_d','mag1'],['time_d','lev0_pc0'],['time_d','lev0_pc1'],['time_d','lev0_pc2'],['time_d','lev0_pc3'],['time_d','lev0_pc4'],['time_d','mag1_anomaly'],['time_d','mag2_anomaly']]
        for i_kp in range(len(keypairs)):
            fun0name,fun1name = [funlib[keypairs[i_kp][j]]["label"] for j in range(2)]
            theta_x = np.array([funlib[keypairs[i_kp][j]]["fun"](centers_all) for j in range(2)]).T
            fig,ax = helper.plot_field_2d(piflat,np.ones(len(centers_all)),theta_x,shp=[15,15],fieldname="Density",fun0name=fun0name,fun1name=fun1name,contourflag=True,avg_flag=False,logscale=True)
            fig.savefig(join(savedir,"pi_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
            plt.close(fig)
            fig,ax = helper.plot_field_2d(qpflat,piflat,theta_x,shp=[15,15],fieldname="Committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
            fig.savefig(join(savedir,"qp_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
            plt.close(fig)
            fig,ax = helper.plot_field_2d(qmflat,piflat,theta_x,shp=[15,15],fieldname="Backward committor",fun0name=fun0name,fun1name=fun1name,contourflag=True)
            fig.savefig(join(savedir,"qm_%s_%s"%(keypairs[i_kp][0],keypairs[i_kp][1])))
            plt.close(fig)
        return result_dga
    def compute_rate_direct(self,src_tag,dest_tag):
        # This is meant for full-winter trajectories
        ab_tag = (src_tag==0)*(dest_tag==1)
        absum = 1.0*(np.sum(ab_tag,axis=1) > 0)
        rate = np.mean(absum)
        return rate



