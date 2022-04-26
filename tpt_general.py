# TPT class for GENERAL stratosphere data: hindcasts or reanalysis. No explicit feature-handling here, just take in numpy arrays and do the magic. Except for time and the feature that gives distances to A and B. 

import numpy as np
import pandas
import netCDF4 as nc
import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
matplotlib.rcParams['savefig.dpi'] = 140
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
        for i in range(1): #len(ab_tag)):
            print(f"For {i}th reanalysis, ")
            print(f"\tsrc_tag[{i}] = \n\t\t{src_tag[i].astype(int)}")
            print(f"\tdest_tag[{i}] = \n\t\t{dest_tag[i].astype(int)}")
            print(f"\tab_tag[{i}] = \n\t\t{ab_tag[i].astype(int)}")
        #absum = 1.0*(np.sum(ab_tag,axis=1) > 0)
        absum = 1.0*(np.sum(np.diff(1.0*ab_tag,axis=1)==1, axis=1))
        print(f"absum = {absum}")
        rate = np.mean(absum)
        print(f"rate = {rate}")
        #sys.exit()
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
        src_tag,dest_tag,time2dest = winstrat.compute_src_dest_tags(Y,feat_def,self.tpt_bndy,"src_dest")
        print(f"src_tag[:,0] = {src_tag[:,0]}")
        print(f"dest_tag[:,0] = {dest_tag[:,0]}")
        ina_Y = winstrat.ina_test(Y[:,90,:],feat_def,self.tpt_bndy)
        print(f"ina_Y = {ina_Y}")
        inb_Y = winstrat.inb_test(Y[:,90,:],feat_def,self.tpt_bndy)
        print(f"inb_Y = {inb_Y}")
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
        print(f"rate = {rate}")
        # Compute lead time
        # Compute current (??)
        summary = {"rate": rate}
        pickle.dump(summary,open(join(savedir,"summary"),"wb"))
        return summary 
    def interpolate_field_clust2data(self,kmdict,Y_unseasoned,Ny,Nt,field_clust,density_flag=False,idx_a=None,idx_b=None,val_a=None,val_b=None):
        # field_clust is a list of arrays, one array for each time step in the winter, and has a value for each cluster center. This function interpolates the field onto data points (in feature space). 
        kmlist,kmtime,kmidx = kmdict["kmlist"],kmdict["kmtime"],kmdict["kmidx"]
        #print(f"n_clusters: {[km.n_clusters for km in kmlist]}")
        #print(f"len(idx): {[len(idx_cl) for idx_cl in kmidx]}")
        field_Y = np.zeros(Ny*Nt)
        for i_time in range(len(kmlist)):
            idx_Y = np.array(kmidx[i_time])
            if len(idx_Y) > 0:
                # ------- Option 1: Interpolate from cluster centers smoothly onto data points -------
                #Dsq_y_c = Y_unseasoned[idx_Y].dot(kmlist[i_time].cluster_centers_.T) + np.add.outer(np.sum(Y_unseasoned[idx_Y]**2, axis=1),np.sum(kmlist[i_time].cluster_centers_**2, axis=1))
                #weights = np.exp(-0.1*Dsq_y_c)
                #weights = np.diag(1/np.sum(weights,axis=1)).dot(weights)
                #field_Y[idx_Y] = weights.dot(field_clust[i_time])
                # ------------ Option 2: Just use nearest cluster (same as above procedure, in limit that weights is a single 1 in every row) -------------
                for i_cl in range(kmlist[i_time].n_clusters):
                    idx_cl = np.array(np.where(kmlist[i_time].labels_ == i_cl)[0])
                    if len(idx_cl) > 0:
                        field_Y[idx_Y[idx_cl]] = field_clust[i_time][i_cl] 
                        if density_flag:
                            field_Y[idx_Y[idx_cl]] *= 1.0/len(idx_cl) # So the change of measure normalizes
        if idx_a is not None and val_a is not None:
            field_Y[idx_a] = val_a
        if idx_b is not None and val_b is not None:
            field_Y[idx_b] = val_b
        return field_Y.reshape((Ny,Nt))
    def transfer_tpt_results(self,
            tpt_feat_filename,clust_filename,feat_def,savedir,winstrat,algo_params,
            tpt_feat_filename_ra,key_ra):
        tpt_qoi_keys = "qp qm pi ina inb lt_mean lt_std".split(" ")
        int2b_cond = pickle.load(open(join(savedir,"int2b_cond"),"rb"))
        qtpt = dict()
        for qk in tpt_qoi_keys:
            if qk == "lt_mean":
                qtpt[qk] = int2b_cond["time"]["mean"]
            elif qk == "lt_std":
                qtpt[qk] = int2b_cond["time"]["std"]
            else:
                qtpt[qk] = pickle.load(open(join(savedir,qk),"rb"))
        centers = pickle.load(open(join(savedir,"centers"),"rb"))
        # Now load reanalysis results
        print(f"tpt_feat_filename_ra = {tpt_feat_filename_ra}")
        tpt_feat_ra = pickle.load(open(tpt_feat_filename_ra,"rb"))
        Yra,szn_mean_Yra,szn_std_Yra = [tpt_feat_ra[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        print(f"Yra.shape = {Yra.shape}, centers[0][0].shape = {centers[0][0].shape}")
        Nyra,Ntyra,ydim = Yra.shape
        Yra = Yra.reshape((Nyra*Ntyra,ydim))
        # Now find cluster labels of them all 
        Yra_unseasoned = winstrat.unseason(Yra[:,0],Yra[:,1:],szn_mean_Yra,szn_std_Yra,delayed=True)
        kmdict = pickle.load(open(clust_filename,"rb"))
        kmlist,kmtime,kmidx = kmdict["kmlist"],kmdict["kmtime"],kmdict["kmidx"]
        # Identify the cluster time step that each Y corresponds to
        kmtime_idx = np.zeros(Nyra*Ntyra, dtype=int)
        labels_Y = np.zeros(Nyra*Ntyra, dtype=int)
        q_Yra = dict({qk: np.nan*np.ones(Nyra*Ntyra) for qk in tpt_qoi_keys})
        for i_time in range(len(kmlist)):
            idx_Y = np.where(np.abs(Yra[:,winstrat.fidx_Y['time_h']] - kmtime[i_time]) < winstrat.dtwint/2)[0] #szn_hour_window/2)[0]
            if len(idx_Y) == 0:
                #print("WARNING, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(i_time,Yra[:,0].min(),Yra[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
                pass
            else:
                kmtime_idx[idx_Y] = i_time
                labels_Y[idx_Y] = kmlist[i_time].predict(Yra_unseasoned[idx_Y])
                for i_cl in range(kmlist[i_time].n_clusters):
                    idx_Y_cl = idx_Y[np.where(labels_Y[idx_Y] == i_cl)[0]]
                    if len(idx_Y_cl) > 0:
                        for qk in tpt_qoi_keys:
                            q_Yra[qk][idx_Y_cl] = qtpt[qk][i_time][i_cl]
        # Check to see if we left any out
        for qk in tpt_qoi_keys:
            if np.any(np.isnan(q_Yra[qk])):
                print(f"WARNING: key {qk} has a nan fraction of {np.mean(np.isnan(q_Yra[qk]))}")
            else:
                print(f"All values were filled for key {qk}")
            pickle.dump(q_Yra[qk],open(join(savedir,f"{qk}_{key_ra}"),"wb"))
        # Check min and max of each interpolated quantity
        for qk in tpt_qoi_keys:
            print(f"interpolated field {qk}: min = {np.nanmin(q_Yra[qk])}, max = {np.nanmax(q_Yra[qk])}")
        return
    def out_of_sample_extension(self,winstrat,clust,f_clust,tpt_feat_test):
        # Given a function defined on cluster centers, evaluate the function on a "test set" tpt_feat_test. 
        Y,szn_mean_Y,szn_std_Y = [tpt_feat_test[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Y_unseasoned = winstrat.unseason(Y[:,0],Y[:,1:],szn_mean_Y,szn_std_Y,normalize=True,delayed=True)
        Nx,Nt,ydim = Y_unseasoned.shape
        Y_unseasoned = Y_unseasoned.reshape((Nx*Nt,ydim))
        kmlist,kmtime = [clust[v] for v in ["kmlist","kmtime"]]
        f_Y = np.nan*np.ones(Nx*Nt)
        for ti in range(winstrat.Ntwint):
            idx_Y = np.where(np.abs(Y[:,winstrat.fidx_Y['time_h']] - winstrat.wtime[ti]) < winstrat.szn_hour_window/2)[0]
            if len(idx) == 0:
                #print("WARNING, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(ti,Y[:,0].min(),Y[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
                pass
            else:
                labels = kmlist[ti].predict(Y_unseasoned[idx_Y])
                for i_cl in range(kmlist[ti].n_clusters):
                    idx_cl = np.where(labels == i_cl)[0]
                    if len(idx_cl) > 0:
                        f_Y[idx_Y[idx_cl]] = f_clust[ti][i_cl]
        return f_Y
    def cluster_features(self,tpt_feat_filename,clust_filename,winstrat,num_clusters=100,resample_flag=False,seed=0):
        # Read in a feature array from feat_filename and build clusters. Save cluster centers. Save them out in clust_filename.
        tpt_feat = pickle.load(open(tpt_feat_filename,"rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        #print(f"min(szn_std_Y) = {np.min(szn_std_Y)}")
        #print(f"which feature has zero szn_std? {np.where(np.min(szn_std_Y,axis=0)==0)[0]}")
        Nx,Nt,ydim = Y.shape
        #print(f"Y.shape = {Y.shape}")
        #print(f"fidx_Y = {winstrat.fidx_Y}")
        Y = Y.reshape((Nx*Nt,ydim))
        # cluster based on the non-time features. 
        Y_unseasoned = winstrat.unseason(Y[:,0],Y[:,1:],szn_mean_Y,szn_std_Y,normalize=True,delayed=True)
        kmtime = []
        kmlist = []
        kmidx = [] # At each time step, which snapshots have the same time. 
        for ti in range(winstrat.Ntwint):
            #idx = np.where(np.abs(Y[:,winstrat.fidx_Y['time_h']] - winstrat.wtime[ti]) < winstrat.szn_hour_window/2)[0]
            idx = np.where(np.abs(Y[:,winstrat.fidx_Y['time_h']] - winstrat.wtime[ti]) < winstrat.dtwint/2)[0]
            if len(idx) == 0:
                #print("WARNING, we don't have any data in time slot {}. Y time: min={}, max={}. wtime: min={}, max={}.".format(ti,Y[:,0].min(),Y[:,0].max(),winstrat.wtime.min(),winstrat.wtime.max()))
                pass
            else:
                kmidx.append(idx)
                kmtime.append(winstrat.wtime[ti])
                km = MiniBatchKMeans(min(len(idx),num_clusters),random_state=seed).fit(Y_unseasoned[idx])
                if ti % 30 == 0:
                    print(f"ti = {ti}, km.n_clusters = {km.n_clusters}, len(idx) = {len(idx)}")
                kmlist.append(km)
        kmdict = {"kmlist": kmlist, "kmtime": np.array(kmtime), "kmidx": kmidx,}
        pickle.dump(kmdict,open(clust_filename,"wb"))
        #print("n_clusters: {}".format(np.array([km.n_clusters for km in kmlist])))
        return
    def build_msm(self,tpt_feat_filename,clust_filename,msm_filename,winstrat):
        nnk = 4 # Number of nearest neighbors for filling in empty positions
        tpt_feat = pickle.load(open(tpt_feat_filename,"rb"))
        Y,szn_mean_Y,szn_std_Y = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y"]]
        Ny,Nt,ydim = Y.shape
        Y_unseasoned = winstrat.unseason(Y[:,:,0].reshape(Ny*Nt),Y[:,:,1:].reshape((Ny*Nt,ydim-1)),szn_mean_Y,szn_std_Y,normalize=True,delayed=True).reshape((Ny,Nt,ydim-1))
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
        ina_Y = winstrat.ina_test(Y.reshape((Ny*Nt,ydim)),feat_def,self.tpt_bndy)
        inb_Y = winstrat.inb_test(Y.reshape((Ny*Nt,ydim)),feat_def,self.tpt_bndy)
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
            centers_t = winstrat.reseason(kmtime[ti]*np.ones(kmlist[ti].n_clusters),kmlist[ti].cluster_centers_,None,szn_mean_Y,szn_std_Y,delayed=True)
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
        ti_feb1 = np.argmin(np.abs(kmtime/24 - (31 + 30 + 31 + 31)))
        ti_mar1 = np.argmin(np.abs(kmtime/24 - (31 + 30 + 31 + 31 + 28)))
        print(f"flux_dens_tob: min={flux_dens_tob.min()},max={flux_dens_tob.max()}, frac>0 = {np.mean(flux_dens_tob>0)}")
        print(f"feb time indices: {ti_feb1}, {ti_mar1}. flux in feb. = {np.sum(flux_dens_tob[ti_feb1:ti_mar1])}")
        print(f"flux_dens_tob = {flux_dens_tob}")
        # Smooth out flux_dens_tob
        hist,t_hist = np.histogram(kmtime_d, weights=flux_dens_tob, bins=5)
        t_hist = (t_hist[1:] + t_hist[:-1])/2
        fig,ax = plt.subplots()
        ax.bar(t_hist,hist)
        ax.set_title("Sloppy")
        fig.savefig(join(savedir,"szn_dist_sloppy"))
        plt.close(fig)

        #ti_froma = np.where(winstrat.wtime > self.tpt_bndy['tthresh'][0])[0][0]
        if np.abs(rate_froma - rate_tob) > 0.1*max(rate_froma, rate_tob):
            raise Exception(f"Rate discrepancy: froma is {rate_froma}, tob is {rate_tob}")
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
        summary = {"rate_froma": rate_froma, "rate_tob": rate_tob, "rate_naive": np.mean(np.any(inb_Y.reshape((Ny,Nt)), axis=1))}
        pickle.dump(summary,open(join(savedir,"summary"),"wb"))
        print(f"Rate: froma={summary['rate_froma']}, tob={summary['rate_tob']}, naive={summary['rate_naive']}")
        
        # ------------------------------------------------------------------------------------
        # Interpolate quantities of interest onto all the points. Enforce boundary conditions. 
        # ------------------------------------------------------------------------------------
        
        idx_a = np.where(ina_Y)[0]
        idx_b = np.where(inb_Y)[0]
        np.save(join(savedir,"ina_Y"),ina_Y)
        np.save(join(savedir,"inb_Y"),inb_Y)
        Y_unseasoned = winstrat.unseason(Y.reshape((Ny*Nt,ydim))[:,0],Y.reshape((Ny*Nt,ydim))[:,1:],szn_mean_Y,szn_std_Y,normalize=True,delayed=True).reshape((Ny*Nt,ydim-1))
        # Committor
        qp_Y = self.interpolate_field_clust2data(kmdict,Y_unseasoned,Ny,Nt,qp,density_flag=False,idx_a=idx_a,idx_b=idx_b,val_a=0.0,val_b=1.0)
        np.save(join(savedir,"qp_Y"),qp_Y)
        # Backward committor
        qm_Y = self.interpolate_field_clust2data(kmdict,Y_unseasoned,Ny,Nt,qm,density_flag=False,idx_a=idx_a,idx_b=idx_b,val_a=1.0,val_b=0.0)
        np.save(join(savedir,"qm_Y"),qm_Y)
        # Change of measure
        pi_Y = self.interpolate_field_clust2data(kmdict,Y_unseasoned,Ny,Nt,pi,density_flag=True)
        np.save(join(savedir,"pi_Y"),pi_Y)
        # Lead time 
        for mom_name in int2b_cond['time'].keys():
            lt_Y = self.interpolate_field_clust2data(kmdict,Y_unseasoned,Ny,Nt,int2b_cond['time'][mom_name],density_flag=False,idx_a=idx_a,idx_b=idx_b)
            np.save(join(savedir,"lt_%s_Y"%(mom_name)),lt_Y)
        return summary 
    def plot_results_data(self,
            feat_filename,tpt_feat_filename,
            feat_filename_ra_dict,tpt_feat_filename_ra_dict,
            fall_year_filename_ra_dict,
            feat_def,savedir,winstrat,algo_params,
            spaghetti_flag=True,fluxdens_flag=True,
            current2d_flag=True,verify_leadtime_flag=True,
            comm_corr_flag=True,
            colors_ra_dict=None,labels_dict=None,
            keys_ra_current=None):
        # Get DGA rate 
        rate = pickle.load(open(join(savedir,"summary"),"rb"))["rate_tob"]
        # Load the reanalysis data for comparison
        keys_ra = list(feat_filename_ra_dict.keys()) # e2 and e5
        if keys_ra_current is None:
            keys_ra_current = [k for k in keys_ra]
        print(f"keys_ra = {keys_ra}")
        ra = dict({key: dict({}) for key in keys_ra})
        print(f"ra.keys() = {ra.keys()}")
        for k in keys_ra:
            tpt_feat_ra = pickle.load(open(tpt_feat_filename_ra_dict[k],"rb"))
            ra[k]["Y"] = tpt_feat_ra["Y"]
            ra[k]["idx_resamp"] = tpt_feat_ra["idx_resamp"]
            ra[k]["Ny"],ra[k]["Nty"],ra[k]["ydim"] = ra[k]["Y"].shape
            ra[k]["X"] = np.load(feat_filename_ra_dict[k])[:,winstrat.ndelay-1:,:][ra[k]["idx_resamp"]]
            ra[k]["fall_years"] = np.load(fall_year_filename_ra_dict[k])[ra[k]["idx_resamp"]]
            ra[k]["Nx"],ra[k]["Ntx"],ra[k]["xdim"] = ra[k]["X"].shape
            if not (ra[k]["Ny"] == ra[k]["Nx"] and ra[k]["Nty"] == ra[k]["Ntx"]):
                raise Exception(f"ERROR: Xra and Yra have shapes {ra[k]['X'].shape} and {ra[k]['Y'].shape} respectively. The first two dimensions should match")
            ra[k]["ina"] = winstrat.ina_test(ra[k]["Y"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["ydim"])),feat_def,self.tpt_bndy).reshape(ra[k]["Ny"],ra[k]["Nty"])
            ra[k]["inb"] = winstrat.inb_test(ra[k]["Y"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["ydim"])),feat_def,self.tpt_bndy).reshape(ra[k]["Ny"],ra[k]["Nty"])
            # Get source and destination
            ra[k]["src_tag"],ra[k]["dest_tag"],ra[k]["time2dest"] = winstrat.compute_src_dest_tags(ra[k]["Y"],feat_def,self.tpt_bndy)
            ra[k]["rate"] = np.mean(np.any((ra[k]["src_tag"]==0)*(ra[k]["dest_tag"]==1), axis=1))
            print(f"For reanalysis key {k}, rate = {ra[k]['rate']}")
            # Restrict reanalysis to the midwinter
            ra[k]["winter_flag"] = ((ra[k]["Y"][:,:,winstrat.fidx_Y['time_h']] >= self.tpt_bndy['tthresh'][0])*(ra[k]["Y"][:,:,winstrat.fidx_Y['time_h']] <= self.tpt_bndy['tthresh'][1]))#.flatten()
            #ra[k]["Y"] = ra[k]["Y"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["ydim"]))
            #ra[k]["X"] = ra[k]["X"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["xdim"]))
            ra[k]["color"] = colors_ra_dict[k]
            ra[k]["label"] = labels_dict[k]
        # Plot fields using the data points rather than the clusters
        funlib_Y = winstrat.observable_function_library_Y(algo_params)
        funlib_X = winstrat.observable_function_library_X()
        tpt_feat = pickle.load(open(tpt_feat_filename, "rb"))
        Y,szn_mean_Y,szn_std_Y,idx_resamp = [tpt_feat[v] for v in ["Y","szn_mean_Y","szn_std_Y","idx_resamp"]]
        X = np.load(feat_filename)[:,winstrat.ndelay-1:,:][idx_resamp]
        print(f"X.shape = {X.shape}")
        print(f"fidx_X[heatflux_lev9_wn0] = {winstrat.fidx_X['heatflux_lev9_wn0']}")
        Ny,Nty,ydim = Y.shape
        Nx,Ntx,xdim = X.shape
        if not (Ntx==Nty and Nx==Ny):
            raise Exception(f"ERROR: X.shape = {X.shape} and Y.shape = {Y.shape}")
        # Keep track of which entries are the beginning of a short trajectory
        #Y = Y.reshape((Ny*Nty,ydim))
        #X = X.reshape((Ny*Nty,xdim))
        traj_rank = np.outer(np.ones(Ny), np.arange(Nty)).flatten()
        winter_flag = (Y[:,:,winstrat.fidx_Y['time_h']] >= self.tpt_bndy['tthresh'][0])*(Y[:,:,winstrat.fidx_Y['time_h']] <= self.tpt_bndy['tthresh'][1])
        idx_winter = np.where(winter_flag)
        #traj_start_positions = np.where(traj_rank == 0)[0]
        #winter_starts = np.intersect1d(idx_winter,traj_start_positions)
        winter_fully_idx = np.where(np.all(winter_flag, axis=1))[0] # Indices within the (Ny,Nty,ydim)-shaped array
        print(f"winter_fully_idx.shape = {winter_fully_idx.shape}")
        qp_Y = np.load(join(savedir,"qp_Y.npy"))#.reshape(Ny*Nty)
        qm_Y = np.load(join(savedir,"qm_Y.npy"))#.reshape(Ny*Nty)
        pi_Y = np.load(join(savedir,"pi_Y.npy"))#.reshape(Ny*Nty)
        lt_mean_Y = np.load(join(savedir,"lt_mean_Y.npy"))#.reshape(Ny*Nty)
        lt_std_Y = np.load(join(savedir,"lt_std_Y.npy"))#.reshape(Ny*Nty)
        lt_skew_Y = np.load(join(savedir,"lt_skew_Y.npy"))#.reshape(Ny*Nty)
        # Now load the same quantities of interest interpolated onto reanalysis
        for k in keys_ra:
            for qk in "qp qm pi lt_mean lt_std".split(" "):
                ra[k][qk] = pickle.load(open(join(savedir,f"{qk}_{k}"),"rb"))
        if comm_corr_flag:
            Nlev = len(feat_def['plev'])
            qp_range = np.array([0.25,0.75])
            lt_range = np.array([0.0,180.0])
            # -------- Perform sparse regression over time-lagged zonal wind ------------
            delay_idx = np.arange(0,winstrat.ndelay,10)
            lasso_coeffs = np.nan*np.ones((winstrat.Ntwint, len(delay_idx)))
            lasso_scores = np.nan*np.ones(winstrat.Ntwint)
            U = np.array([funlib_Y["uref_dl%i"%(i_dl)]["fun"](Y.reshape((Ny*Nty,ydim))) for i_dl in delay_idx]).reshape((len(delay_idx),Ny,Nty))
            print(f"U.shape = {U.shape}")
            for i_time in range(winstrat.Ntwint):
                idx_Y = np.where(
                        (np.abs(Y[:,:,winstrat.fidx_Y['time_h']] - winstrat.wtime[i_time]) < 3*winstrat.dtwint) * 
                        (qp_Y >= qp_range[0]) * (qp_Y <= qp_range[1]) * 
                        (lt_mean_Y >= lt_range[0]) * (lt_mean_Y <= lt_range[1])
                        )
                if len(idx_Y[0]) > 15:
                    #lm = linear_model.Lasso(alpha=0.25)
                    lm = linear_model.LinearRegression()
                    lm.fit(U[:,idx_Y[0],idx_Y[1]].T,qp_Y[idx_Y[0],idx_Y[1]],sample_weight=pi_Y[idx_Y[0],idx_Y[1]]/np.sum(pi_Y[idx_Y[0],idx_Y[1]]))
                    lasso_coeffs[i_time,:] = lm.coef_
                    lasso_scores[i_time] = lm.score(U[:,idx_Y[0],idx_Y[1]].T,qp_Y[idx_Y[0],idx_Y[1]],sample_weight=pi_Y[idx_Y[0],idx_Y[1]]/np.sum(pi_Y[idx_Y[0],idx_Y[1]]))
            fig,ax = plt.subplots(nrows=2,figsize=(6,12),sharex=True)
            handles = []
            for i_dl in range(len(delay_idx)):
                h, = ax[0].plot(winstrat.wtime/24.0,lasso_coeffs[:,i_dl],color=plt.cm.coolwarm(i_dl/len(delay_idx)),label=r"$t-%i$ days"%(delay_idx[i_dl]))
                handles += [h]
            ax[0].axhline(0,color='black',linestyle='--')
            ax[0].legend(handles=handles)
            ax[1].plot(winstrat.wtime/24.0,lasso_scores,color='black')
            ax[1].set_xlabel(funlib_Y["time_d"]["label"])
            fig.savefig((join(savedir,"corr_lasso_u_qp%.2f-%.2f_lt%i-%i"%(qp_range[0],qp_range[1],lt_range[0],lt_range[1]))).replace(".","p"))
            plt.close(fig)
            # --------- Correlate committor (within a certain mid-range) with zonal wind profile ------
            u_qp_corr = np.nan*np.ones((winstrat.Ntwint, Nlev))
            U = np.array([X[:,:,winstrat.fidx_X["ubar_60N_lev%i"%(i_lev)]] for i_lev in range(Nlev)])
            for i_time in range(winstrat.Ntwint):
                idx_Y = np.where(
                        (np.abs(Y[:,:,winstrat.fidx_Y['time_h']] - winstrat.wtime[i_time]) < 3*winstrat.dtwint) * 
                        (qp_Y >= qp_range[0]) * (qp_Y <= qp_range[1]) * 
                        (lt_mean_Y >= lt_range[0]) * (lt_mean_Y <= lt_range[1])
                        )
                if len(idx_Y[0]) > 15:
                    for i_lev in range(Nlev):
                        Ui = U[i_lev,idx_Y[0],idx_Y[1]]
                        qpi = qp_Y[idx_Y[0],idx_Y[1]]
                        pii = pi_Y[idx_Y[0],idx_Y[1]]
                        Ui_mean = np.sum(Ui*pii)/np.sum(pii)
                        qpi_mean = np.sum(qpi*pii)/np.sum(pii)
                        u_qp_corr[i_time,i_lev] = (np.sum((Ui-Ui_mean)*(qpi-qpi_mean)*pii))/(np.sqrt(np.sum((Ui-Ui_mean)**2*pii)*np.sum((qpi-qpi_mean)**2*pii)))
            fig,ax = plt.subplots()
            ax.set_title(r"Corr($q_B^+, \overline{u}(z)$)")
            handles = []
            for i_lev in range(Nlev):
                h, = ax.plot(winstrat.wtime/24.0,u_qp_corr[:,i_lev],color=plt.cm.coolwarm(i_lev/(Nlev-1)),label=r"%i hPa"%(feat_def["plev"][i_lev]/100))
                handles += [h]
            xlim = ax.get_xlim()
            ax.set_xlim([xlim[0],xlim[1]+0.25*(xlim[1]-xlim[0])])
            ax.set_ylim([-1,1])
            ax.axhline(0,linestyle='--',color='black')
            ax.legend(handles=handles,loc='lower right')
            ax.set_xlabel(funlib_X['time_d']['label'])
            ax.set_ylabel("Correlation coefficient")
            fig.savefig((join(savedir,"corr_u_qp%.2f-%.2f_lt%i-%i"%(qp_range[0],qp_range[1],lt_range[0],lt_range[1]))).replace(".","p"))
            plt.close(fig)
            # ---------- Correlate committor with zonal wind time delays -------------
            u_qp_corr = np.nan*np.ones((winstrat.Ntwint, winstrat.ndelay))
            U = np.array([funlib_Y["uref_dl%i"%(i_dl)]["fun"](Y.reshape((Ny*Nty,ydim))) for i_dl in range(winstrat.ndelay)]).reshape((winstrat.ndelay,Ny,Nty))
            for i_time in range(winstrat.Ntwint):
                idx_Y = np.where(
                        (np.abs(Y[:,:,winstrat.fidx_Y['time_h']] - winstrat.wtime[i_time]) < 3*winstrat.dtwint) * 
                        (qp_Y >= qp_range[0]) * (qp_Y <= qp_range[1]) *
                        (lt_mean_Y >= lt_range[0]) * (lt_mean_Y <= lt_range[1])
                        )
                if len(idx_Y[0]) > 15:
                    for i_dl in range(winstrat.ndelay):
                        Ui = U[i_dl,idx_Y[0],idx_Y[1]]
                        qpi = qp_Y[idx_Y[0],idx_Y[1]]
                        pii = pi_Y[idx_Y[0],idx_Y[1]]
                        Ui_mean = np.sum(Ui*pii)/np.sum(pii)
                        qpi_mean = np.sum(qpi*pii)/np.sum(pii)
                        u_qp_corr[i_time,i_dl] = (np.sum((Ui-Ui_mean)*(qpi-qpi_mean)*pii))/(np.sqrt(np.sum((Ui-Ui_mean)**2*pii)*np.sum((qpi-qpi_mean)**2*pii)))
            fig,ax = plt.subplots()
            ax.set_title(r"Corr($q_B^+, \overline{u}(t-\Delta t)$)")
            handles = []
            for i_dl in np.arange(0,winstrat.ndelay,5):
                h, = ax.plot(winstrat.wtime/24.0,u_qp_corr[:,i_dl],color=plt.cm.coolwarm(i_dl/(winstrat.ndelay-1)),label=r"$t-%i$ days"%(i_dl))
                handles += [h]
            xlim = ax.get_xlim()
            ax.set_xlim([xlim[0],xlim[1]+0.25*(xlim[1]-xlim[0])])
            ax.set_ylim([-1,1])
            ax.axhline(0,linestyle='--',color='black')
            ax.legend(handles=handles,loc='lower right')
            ax.set_xlabel(funlib_X['time_d']['label'])
            ax.set_ylabel("Correlation coefficient")
            fig.savefig((join(savedir,"corr_udelays_qp%.2f-%.2f_lt%i-%i"%(qp_range[0],qp_range[1],lt_range[0],lt_range[1]))).replace(".","p"))
            plt.close(fig)
            # ------------ Correlate committor with DIFFERENCES in zonal wind -----------
            u_qp_corr = np.nan*np.ones((winstrat.Ntwint, winstrat.ndelay))
            U = np.array([funlib_Y["uref_dl%i"%(i_dl)]["fun"](Y.reshape((Ny*Nty,ydim))) for i_dl in range(winstrat.ndelay)]).reshape((winstrat.ndelay,Ny,Nty))
            for i_time in range(winstrat.Ntwint):
                idx_Y = np.where(
                        (np.abs(Y[:,:,winstrat.fidx_Y['time_h']] - winstrat.wtime[i_time]) < 3*winstrat.dtwint) * 
                        (qp_Y >= qp_range[0]) * (qp_Y <= qp_range[1]) *
                        (lt_mean_Y >= lt_range[0]) * (lt_mean_Y <= lt_range[1])
                        )
                if len(idx_Y[0]) > 15:
                    for i_dl in range(1,winstrat.ndelay):
                        Ui = (U[0,idx_Y[0],idx_Y[1]] - U[i_dl,idx_Y[0],idx_Y[1]])/(i_dl*winstrat.dtwint)
                        qpi = qp_Y[idx_Y[0],idx_Y[1]]
                        pii = pi_Y[idx_Y[0],idx_Y[1]]
                        Ui_mean = np.sum(Ui*pii)/np.sum(pii)
                        qpi_mean = np.sum(qpi*pii)/np.sum(pii)
                        u_qp_corr[i_time,i_dl] = (np.sum((Ui-Ui_mean)*(qpi-qpi_mean)*pii))/(np.sqrt(np.sum((Ui-Ui_mean)**2*pii)*np.sum((qpi-qpi_mean)**2*pii)))
            fig,ax = plt.subplots()
            ax.set_title(r"Corr($q_B^+,\Delta\overline{u}/\Delta t$)")
            handles = []
            for i_dl in np.arange(0,winstrat.ndelay,5):
                h, = ax.plot(winstrat.wtime/24.0,u_qp_corr[:,i_dl],color=plt.cm.coolwarm(i_dl/(winstrat.ndelay-1)),label=r"$t-%i$ days"%(i_dl))
                handles += [h]
            xlim = ax.get_xlim()
            ax.set_xlim([xlim[0],xlim[1]+0.25*(xlim[1]-xlim[0])])
            ax.set_ylim([-1,1])
            ax.axhline(0,linestyle='--',color='black')
            ax.legend(handles=handles,loc='lower right')
            ax.set_xlabel(funlib_X['time_d']['label'])
            ax.set_ylabel("Correlation coefficient")
            fig.savefig((join(savedir,"corr_dudt_qp%.2f-%.2f_lt%i-%i"%(qp_range[0],qp_range[1],lt_range[0],lt_range[1]))).replace(".","p"))
            plt.close(fig)
        if fluxdens_flag:
            reactive_code = [0,1] #= ((src_tag==0)*(dest_tag==1)*winter_flag_ra).reshape((Nyra,Ntyra))
            # ----------- Plot seasonal distribution with multiple resolutions ---------
            # New method: simply look at the discretized flux crossing the boundary.
            theta_normal_label = funlib_X['uref']['label']
            theta_tangential_label = funlib_X['time_d']['label']
            theta_lower = -self.tpt_bndy['uthresh_b'] - 1.0
            theta_upper = -self.tpt_bndy['uthresh_b'] + 1.0
            infoth = dict({
                "theta_normal": -funlib_X['uref']['fun'](X.reshape((Ny*Nty,xdim))).reshape((Ny,Nty)),
                "theta_tangential": funlib_X['time_d']['fun'](X.reshape((Ny*Nty,xdim))).reshape((Ny,Nty)), 
                })
            bin_edges_list = [
                    np.cumsum([0,31,30,31,31,28,31]) + 0.5,
                    np.cumsum([0,
                        10,10,11,
                        10,10,10,
                        10,10,11,
                        10,10,11,
                        9,9,10,
                        10,10,11,
                        ]) + 0.5
                    #np.cumsum([0,
                    #    8,7,8,8,
                    #    8,7,8,7,
                    #    8,7,8,8,
                    #    8,7,8,8,
                    #    7,7,7,7,
                    #    8,7,8,8,
                    #    ]) + 0.5
                    ]
            # Make a vertical stack of panels, one for each reanalysis dataset
            fig,ax = plt.subplots(nrows=1+len(ra), figsize=(6,3*(1+len(ra))),sharex=True,sharey=False)
            ax[0].set_title(r"$U_{10}^{\mathrm{(th)}}=%i$ m/s"%(self.tpt_bndy['uthresh_b']),fontdict=font)
            # First, DGA
            info = dict({
                "qm": qm_Y, 
                "qp": qp_Y, 
                "pi": pi_Y, 
                "rate": rate,
                "label": labels_dict["s2s-self"],
                })
            hist_color_list = ['red','black']
            for i_be,bin_edges in enumerate(bin_edges_list):
                _,_,hist = self.plot_flux_distributions_multiresolution(
                    info,infoth, # Can be either reanalysis or DGA data
                    theta_normal_label,theta_tangential_label,
                    reactive_code,
                    theta_lower,theta_upper,
                    bin_edges,hist_color_list[i_be],fill=('solid' if i_be==0 else 'hatch'),
                    info_type="dga",fig=fig,ax=ax[0],label=info["label"] if i_be==0 else None)
            # Second, the reanalysis
            for i_k,k in enumerate(ra.keys()):
                print(f"ra.keys = {ra.keys()}")
                info = dict({
                    "src_tag": ra[k]['src_tag'],
                    "dest_tag": ra[k]['dest_tag'],
                    "rate": ra[k]['rate'],
                    "label": ra[k]['label'],
                    })
                infoth = dict({
                    "theta_normal": -funlib_X['uref']['fun'](ra[k]["X"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["xdim"]))).reshape((ra[k]["Ny"],ra[k]["Nty"])),
                    "theta_tangential": funlib_X['time_d']['fun'](ra[k]["X"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["xdim"]))).reshape((ra[k]["Ny"],ra[k]["Nty"])),
                    })
                hist_color_list = [ra[k]['color'],'black']
                #if k == 'e5-self': hist_color_list[0] = 'gray'
                for i_be,bin_edges in enumerate(bin_edges_list):
                    print(f"Starting to plot histogram for {k} at level {self.tpt_bndy['uthresh_b']}. The reanalysis rate is {ra[k]['rate']}")
                    _,_,hist = self.plot_flux_distributions_multiresolution(
                        info,infoth, # Can be either reanalysis or DGA data
                        theta_normal_label,theta_tangential_label,
                        reactive_code,
                        theta_lower,theta_upper,
                        bin_edges,hist_color_list[i_be],fill=('solid' if i_be==0 else 'hatch'),
                        info_type="ra",fig=fig,ax=ax[1+i_k],label=info["label"] if i_be==0 else None)
            # Format the axis labels by naming months
            for i_ax in range(1+len(ra)):
                ax[i_ax].set_xticks(bin_edges_list[0])
                ax[i_ax].set_xlabel("")
                if i_ax == len(ra):
                    ax[i_ax].set_xticklabels(['Oct. 1', 'Nov. 1', 'Dec. 1', 'Jan. 1', 'Feb. 1', 'Mar. 1', 'Apr. 1'])
                else:
                    ax[i_ax].set_xticklabels(['']*7)
                ax[i_ax].set_ylabel("SSW rel. freq.")
            xlim = [self.tpt_bndy['tthresh'][0]/24.0-5, self.tpt_bndy['tthresh'][1]/24.0+5]
            ylim = [0, max([axi.get_ylim()[1] for axi in ax])]
            for i_ax in range(len(ax)):
                ax[i_ax].set_xlim(xlim)
                ax[i_ax].set_ylim(ylim)
            fig.savefig(join(savedir,"szn_dist"))
            plt.close(fig)

        if verify_leadtime_flag:
            fig,ax = plt.subplots()
            for k in keys_ra_current:
                good_idx = np.where((np.isnan(ra[k]["lt_mean"].flatten())==0)*(ra[k]["dest_tag"].flatten()==1))[0]
                ax.scatter(ra[k]["lt_mean"].flatten()[good_idx],ra[k]["time2dest"].flatten()[good_idx]/24.0, color=ra[k]["color"], alpha=0.5, marker='.', s=36*ra[k]["qp"].flatten()[good_idx])
            ax.set_xlabel(r"$\eta_B^+$")
            ax.set_ylabel(r"Time to $B$")
            fig.savefig(join(savedir,"verification_leadtime"))
            plt.close(fig)
        if current2d_flag:
            # ------------- Current plots --------------------
            xticks = np.cumsum([31,30,31,31,28,31])
            xticklabels = ['Nov. 1', 'Dec. 1', 'Jan. 1', 'Feb. 1', 'Mar. 1', 'Apr. 1']
            keypairs = []
            #keypairs += [['uref_inc_0','uref_inc_2']]
            keypairs += [['time_d','uref']]
            #keypairs += [['uref_dl0','uref_dl%i'%(i_dl)] for i_dl in np.arange(5,winstrat.ndelay,5)]
            #keypairs += [['heatflux_lev1_total','heatflux_lev4_total']]
            #keypairs += [['time_d','captemp_lev0']]
            #keypairs += [['time_d','heatflux_lev4_wn%i'%(i_wn)] for i_wn in range(winstrat.heatflux_wavenumbers_per_level_max)]
            #keypairs += [['heatflux_lev4_wn1','heatflux_lev4_wn2']]
            #keypairs += [['time_d','pc%i_lev0'%(i_pc)] for i_pc in range(algo_params['Npc_per_level'][0])]
            #keypairs += [['uref','pc%i_lev0'%(i_pc)] for i_pc in range(6)]
            #keypairs += [['pc1_lev0','pc%i_lev0'%(i_pc)] for i_pc in range(2,6)]
            for key0,key1 in keypairs:
                print(f"Plotting current on key pair {key0},{key1}")
                y_flag = (key0 in funlib_Y.keys() and key1 in funlib_Y.keys())
                x_flag = (key0 in funlib_X.keys() and key1 in funlib_X.keys())
                if not (y_flag or x_flag):
                    print(f"WARNING: {key0} and {key1} are not both in funlib_X.keys or funlib_Y.keys")
                else:
                    rath = dict({key: dict({}) for key in keys_ra_current}) # Supplemental dictionary for this specific projection and current direction. This might be modified by the function.
                    if y_flag:
                        theta_x = np.array([funlib_Y[key0]["fun"](Y.reshape((Ny*Nty,ydim))), funlib_Y[key1]["fun"](Y.reshape((Ny*Nty,ydim)))]).T.reshape((Nx,Ntx,2))
                        #theta_x_ra = np.array([funlib_Y[key0]["fun"](Yra), funlib_Y[key1]["fun"](Yra)]).T
                        for k in keys_ra_current:
                            rath[k]["theta"] = np.array([
                                funlib_Y[key0]['fun'](ra[k]["Y"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["ydim"]))),
                                funlib_Y[key1]['fun'](ra[k]["Y"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["ydim"])))]).T.reshape((ra[k]["Ny"],ra[k]["Nty"],2))
                        lab0 = funlib_Y[key0]["label"]
                        lab1 = funlib_Y[key1]["label"]
                    else: 
                        theta_x = np.array([funlib_X[key0]["fun"](X.reshape((Nx*Ntx,xdim))), funlib_X[key1]["fun"](X.reshape((Nx*Ntx,xdim)))]).T.reshape((Nx,Ntx,2))
                        #theta_x_ra = np.array([funlib_X[key0]["fun"](Xra), funlib_X[key1]["fun"](Xra)]).T
                        for k in keys_ra_current:
                            rath[k]["theta"] = np.array([
                                funlib_X[key0]['fun'](ra[k]["X"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["xdim"]))),
                                funlib_X[key1]['fun'](ra[k]["X"].reshape((ra[k]["Ny"]*ra[k]["Nty"],ra[k]["xdim"])))]).T.reshape((ra[k]["Ny"],ra[k]["Nty"],2))
                        lab0 = funlib_X[key0]["label"]
                        lab1 = funlib_X[key1]["label"]
                    if key0 == "time_d":
                        xlim = self.tpt_bndy['tthresh']/24.0
                        print(f"xlim = {xlim}")
                    else:
                        xlim = np.array([np.nanmin(theta_x[idx_winter[0],idx_winter[1],0]),np.nanmax(theta_x[idx_winter[0],idx_winter[1],0])])
                    ylim = np.array([np.nanmin(theta_x[idx_winter[0],idx_winter[1],1]),np.nanmax(theta_x[idx_winter[0],idx_winter[1],1])])
                    ylim[1] += 0.15*(ylim[1] - ylim[0])
                    # A -> B
                    reactive_code = [0,1]
                    comm_bwd = qm_Y*(reactive_code[0] == 0) + (1-qm_Y)*(reactive_code[0] == 1)
                    comm_fwd = qp_Y*(reactive_code[1] == 1) + (1-qp_Y)*(reactive_code[1] == 0)
                    print(f"shapes: qp_Y: {qp_Y.shape}, pi_Y: {pi_Y.shape}, theta_x: {theta_x.shape}")
                    print(f"idx_winter: 0: min={idx_winter[0].min()}, max={idx_winter[0].max()}")
                    # Plot the committor
                    fieldname = r"Committor probability"
                    fig,ax = plt.subplots()
                    if key0 == "time_d" and key1 == "uref":
                        ax.axhline(self.tpt_bndy['uthresh_b'],linestyle='--',color='purple',zorder=5)
                    helper.plot_field_2d((comm_fwd)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=fieldname,fun0name=lab0,fun1name=lab1,avg_flag=True,logscale=False,cmap=plt.cm.coolwarm,contourflag=True,fig=fig,ax=ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"qp_lin_%s_%s_ab_build1"%(key0.replace("_",""),key1.replace("_",""))))
                    plt.close(fig)
                    # Plot the backward committor
                    fieldname = r"Backward committor probability"
                    fig,ax = plt.subplots()
                    if key0 == "time_d" and key1 == "uref":
                        ax.axhline(self.tpt_bndy['uthresh_b'],linestyle='--',color='purple',zorder=5)
                    helper.plot_field_2d((comm_bwd)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=fieldname,fun0name=lab0,fun1name=lab1,avg_flag=True,logscale=False,cmap=plt.cm.coolwarm,contourflag=True,fig=fig,ax=ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"qm_lin_%s_%s_ab_build1"%(key0.replace("_",""),key1.replace("_",""))))
                    plt.close(fig)
                    # Plot the lead time 
                    fig,ax = plt.subplots()
                    fieldname = r"Lead time [days]"
                    if key0 == "time_d" and key1 == "uref":
                        ax.axhline(self.tpt_bndy['uthresh_b'],linestyle='--',color='purple',zorder=5)
                    helper.plot_field_2d((-lt_mean_Y)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=fieldname,fun0name=lab0,fun1name=lab1,avg_flag=True,logscale=False,cmap=plt.cm.coolwarm,contourflag=True,fig=fig,ax=ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"lt_%s_%s_ab_build1"%(key0.replace("_",""),key1.replace("_",""))))
                    plt.close(fig)
                    # Plot the A->B density and current
                    fieldname = r"$A\to B$ (winters with SSW)"
                    fig,ax = plt.subplots()
                    if key0 == "time_d" and key1 == "uref":
                        handles,seg_labels = self.plot_trajectory_segments(ra,rath,reactive_code,fig,ax)
                        #ax.legend(handles=handles)
                        sample_suffix = '-'.join(seg_labels)
                        ax.axhline(self.tpt_bndy['uthresh_b'],linestyle='--',color='purple',zorder=5)
                    ax.set_xlabel(lab0,fontdict=font)
                    ax.set_ylabel(lab1,fontdict=font)
                    ax.set_title(fieldname,fontdict=font)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_ab_%s_build0"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    helper.plot_field_2d((comm_bwd*comm_fwd)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=fieldname,fun0name=lab0,fun1name=lab1,avg_flag=False,logscale=False,cmap=plt.cm.YlOrRd,contourflag=True,fig=fig,ax=ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_ab_%s_build1"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    _,_,_,_ = self.plot_current_overlay_data(theta_x,comm_bwd,comm_fwd,pi_Y,fig,ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_ab_%s_build2"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    plt.close(fig)
                    # Plot the A -> A density and current
                    fieldname = r"$A\to A$ (winters without SSW)"
                    reactive_code = [0,0]
                    comm_bwd = qm_Y*(reactive_code[0] == 0) + (1-qm_Y)*(reactive_code[0] == 1)
                    comm_fwd = qp_Y*(reactive_code[1] == 1) + (1-qp_Y)*(reactive_code[1] == 0)
                    fig,ax = plt.subplots()
                    if key0 == "time_d" and key1 == "uref":
                        handles,seg_labels = self.plot_trajectory_segments(ra,rath,reactive_code,fig,ax)
                        sample_suffix = '-'.join(seg_labels)
                        #ax.legend(handles=handles)
                        ax.axhline(self.tpt_bndy['uthresh_b'],linestyle='--',color='purple',zorder=5)
                    ax.set_xlabel(lab0,fontdict=font)
                    ax.set_ylabel(lab1,fontdict=font)
                    ax.set_title(fieldname,fontdict=font)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_aa_%s_build0"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    helper.plot_field_2d((comm_bwd*comm_fwd)[idx_winter],pi_Y[idx_winter],theta_x[idx_winter],fieldname=fieldname,fun0name=lab0,fun1name=lab1,avg_flag=False,logscale=False,cmap=plt.cm.YlOrRd,contourflag=True,fig=fig,ax=ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_aa_%s_build1"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    _,_,_,_ = self.plot_current_overlay_data(theta_x,comm_bwd,comm_fwd,pi_Y,fig,ax)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel("")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.savefig(join(savedir,"J_%s_%s_aa_%s_build2"%(key0.replace("_",""),key1.replace("_",""),sample_suffix)))
                    plt.close(fig)
        if spaghetti_flag:
            # ----------- New plot: short histories of zonal wind, colored by committor. Maybe this will inform new coordinates --------------
            fig,ax = plt.subplots()
            time_full = np.arange(winstrat.ndelay)*winstrat.dtwint/24.0
            time_full -= time_full[-1]
            uref_idx = np.array([winstrat.fidx_Y["uref_dl%i"%(i_dl)] for i_dl in range(winstrat.ndelay)])[::-1]
            uref = Y[:,:,uref_idx]
            print(f"Y.shape = {Y.shape}")
            print(f"uref.shape = {uref.shape}")
            prng = np.random.RandomState(1)
            ss = prng.choice(np.arange(len(idx_winter[0])),size=min(len(idx_winter[0]),100),replace=False) 
            for i_y in ss:
                time_i = time_full + Y[idx_winter[0][i_y],idx_winter[1][i_y],winstrat.fidx_Y['time_h']]/24.0
                ax.plot(time_i,uref[idx_winter[0][i_y],idx_winter[1][i_y]],color=plt.cm.coolwarm(qp_Y[idx_winter[0][i_y],idx_winter[1][i_y]]),alpha=0.75)
            ax.set_xlabel(funlib_Y["time_d"]["label"])
            ax.set_ylabel(funlib_Y["uref_dl0"]["label"])
            fig.savefig(join(savedir,"qp_uref_spaghetti"))
            plt.close(fig)
            print("saved spaghetti, now moving on...")
            ## ------------ New plot: vertical profiles of zonal wind colored by committor. -------------
            #Nlev = len(feat_def['plev'])
            #fig,ax = plt.subplots()
            #prng = np.random.RandomState(1)
            #ss = prng.choice(winter_starts,size=min(len(winter_starts),1000)) # Make sure it's always at the beginning
            #ubar_idx = np.array([winstrat.fidx_X["ubar_60N_lev%i"%(i_lev)] for i_lev in range(Nlev)])
            #ubar = X[:,ubar_idx]
            #for i_x in ss:
            #    ax.plot(ubar[i_x],-7*np.log(feat_def["plev"]/feat_def["plev"][-1]),color=plt.cm.coolwarm(qp_Y[i_x]),alpha=0.1,linewidth=2)
            #ax.set_xlabel(r"$\overline{u}$ [m/s]")
            #ax.set_ylabel(r"Pseudo-height [km]")
            #fig.savefig(join(savedir,"qp_uprofile_spaghetti"))
            #plt.close(fig)
            # ------------ New plot: vertical profiles of heat flux colored by committor. -------------
            Nlev = len(feat_def['plev'])
            fig,ax = plt.subplots()
            prng = np.random.RandomState(1)
            ss = prng.choice(np.arange(len(idx_winter[0])),size=min(len(idx_winter[0]),100),replace=False) 
            vT = np.array([funlib_X["heatflux_lev%i_total"%(i_lev)]["fun"](X[idx_winter[0],idx_winter[1],:]) for i_lev in range(Nlev)]).T
            print(f"vT.shape = {vT.shape}; X.shape = {X.shape}; max(ss) = {ss.max()}")
            for i_x in ss:
                ax.plot(vT[i_x],-7*np.log(feat_def["plev"]/feat_def["plev"][-1]),color=plt.cm.coolwarm(qp_Y[idx_winter[0][i_y],idx_winter[1][i_y]]),alpha=0.1,linewidth=2)
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
        fig,ax,_ = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux,theta_lower_list=theta_lower_list,theta_upper_list=theta_upper_list)
        fig.savefig(join(savedir,"Jab-qp_d-uref"))
        plt.close(fig)
        # Jab dot grad (qp) d(time)
        theta_normal_flat = qpflat
        theta_normal_label = r"$q_B^+$"
        theta_lower_list = [0.2,0.45,0.7]
        theta_upper_list = [0.3,0.55,0.8]
        theta_tangential_flat = funlib["time_d"]["fun"](np.concatenate(centers,axis=0))
        theta_tangential_label = funlib["time_d"]["label"]
        fig,ax,_ = self.plot_flux_distributions_1d(centers,theta_normal_flat,theta_tangential_flat,theta_normal_label,theta_tangential_label,kmtime,flux,theta_lower_list=theta_lower_list,theta_upper_list=theta_upper_list)
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
        return close_idx,reactive_flux
    def reactive_flux_density_levelset_strict(self,theta_level,theta_x,qm,qp,pi):
        # Only count trajectories crossing that surface.
        Nx,Nt = theta_x.shape
        rflux = np.zeros((Nx,Nt-1))
        for i_time in range(Nt-1):
            idx_crossing = np.where((theta_x[:,i_time] < theta_level)*(theta_x[:,i_time+1] > theta_level))[0]
            rflux[idx_crossing,i_time] = qm[idx_crossing,i_time]*pi[idx_crossing,i_time]*qp[idx_crossing,i_time+1]
        return rflux 
    def plot_flux_distributions_multiresolution(self,
            info,infoth, # Can be either reanalysis or DGA data
            theta_normal_label,theta_tangential_label,
            reactive_code,
            theta_lower,theta_upper,
            bin_edges,hist_color,fill='solid',
            info_type="dga",
            fig=None,ax=None,label=None):
        # Meant to plot histograms of reactive flux density with different resolutions. 
        theta_lower_list = [theta_lower]
        theta_upper_list = [theta_upper]
        infoth["thmid_tangential"] = 0.5*(infoth["theta_tangential"][:,1:] + infoth["theta_tangential"][:,:-1])
        infoth["thmid_normal"] = 0.5*(infoth["theta_normal"][:,1:] + infoth["theta_normal"][:,:-1])
        if info_type == "dga":
            # In this case, possibly perform this in the space of cluster centers
            rflux = self.reactive_flux_density_levelset_strict((theta_lower+theta_upper)/2, infoth["theta_normal"], info["qm"], info["qp"], info["pi"])
            theta_tan_rflux = (infoth["theta_tangential"][:,:-1] + infoth["theta_tangential"][:,1:])/2
            # Make a histogram of the reactive flux density distribution
            hist,bin_edges = np.histogram(theta_tan_rflux.flatten(), weights=rflux.flatten(), bins=bin_edges)
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            hist *= 1.0/np.sum(hist*np.diff(bin_edges))
            #hist *= info["rate"] 
        else: # Reanalysis
            infoth["delta_theta_normal"] = infoth["theta_normal"][:,1:] - infoth["theta_normal"][:,:-1]
            infoth["reactive_flag"] = (info["src_tag"] == reactive_code[0])*(info["dest_tag"] == reactive_code[1])
            print(f"reactive_flag: {np.sum(infoth['reactive_flag'],axis=1)}")
            theta_mid = 0.5*(theta_lower_list[0] + theta_upper_list[0])
            idx_fwd = np.where((infoth["theta_normal"][:,:-1] < theta_mid)*(infoth["theta_normal"][:,1:] >= theta_mid)*(infoth["reactive_flag"][:,:-1]+infoth["reactive_flag"][:,1:]))
            idx_bwd = np.where((infoth["theta_normal"][:,:-1] >= theta_mid)*(infoth["theta_normal"][:,1:] < theta_mid)*(infoth["reactive_flag"][:,:-1]+infoth["reactive_flag"][:,1:]))
            idx = (np.concatenate((idx_fwd[0], idx_bwd[0])),
                    np.concatenate((idx_fwd[1], idx_bwd[1])))
            signs_ra = np.concatenate((np.ones(len(idx_fwd[0])),-np.ones(len(idx_bwd[0])))) #* np.sign(theta_normal_comm_corr)
            theta_tangential_ra = 0.5*(infoth["theta_tangential"][idx[0],idx[1]] + infoth["theta_tangential"][idx[0],idx[1]+1])
            infoth["num_rxn"] = np.abs(np.sum(signs_ra))
            if infoth["num_rxn"] != 0:
                hist,bin_edges = np.histogram(theta_tangential_ra, weights=signs_ra,bins=bin_edges)
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                hist *= 1.0/(np.sum(hist)*np.diff(bin_edges)) 
                #hist *= info["rate"]
        # Make the plot
        if fig is None or ax is None: fig,ax = plt.subplots()
        x_edges = []
        y_edges = []
        for i_bin in range(len(bin_centers)):
            x_edges += [bin_edges[i_bin],bin_edges[i_bin+1]]
            y_edges += [hist[i_bin],hist[i_bin]]
        if fill == 'solid':
            h = ax.fill_between(x_edges,y_edges,y2=0,color=hist_color,label=label)
        elif fill == 'hatch':
            ax.fill_between(x_edges,y_edges,y2=0,color=hist_color,label=label,hatch='//',facecolor="none")
            ax.plot(bin_edges[i_bin:i_bin+2],hist[i_bin]*np.ones(2),color=hist_color)
            #left_jump = 0 if i_bin==0 else hist[i_bin-1]
            #right_jump = 0 if i_bin==len(bin_centers)-1 else hist[i_bin+1]
            #ax.plot(bin_edges[i_bin]*np.ones(2), [left_jump,hist[i_bin]],color=hist_color,linewidth=2.5)
            #ax.plot(bin_edges[i_bin+1]*np.ones(2), [right_jump,hist[i_bin]],color=hist_color)
        ax.set_xlabel(theta_tangential_label,fontdict=font)
        if fill == 'solid': ax.legend(handles=[h],loc='upper left')
        ax.set_ylabel(r"",fontdict=font)
        ax.axhline(y=0,color='black')
        return fig,ax,hist
    def plot_flux_distributions_1d(self,
            qm,qp,pi,
            theta_normal,theta_tangential,
            ra,rath,
            theta_normal_label,theta_tangential_label,
            reactive_code,rate,
            theta_lower_list=None,theta_upper_list=None,
            timeseries_like=False,invert_flag=False,
            fig=None,ax=None,dashed_flag=False,desired_bins=10,
            dga_flag=True):
        # theta_normal and theta_tangential are scalar fields. 
        dth_tangential = (np.nanmax(theta_tangential) - np.nanmin(theta_tangential))/desired_bins
        theta_vec = np.transpose(np.array([theta_normal,theta_tangential]), (1,2,0))
        print(f"shapes: theta_vec->{theta_vec.shape}, qm->{qm.shape}, qp->{qp.shape}, pi->{pi.shape}")
        Jth,thmid,Jweight = self.project_current_data(theta_vec,qm,qp,pi)
        if theta_lower_list is None or theta_upper_list is None:
            theta_normal_min = np.nanmin(theta_normal)
            theta_normal_max = np.nanmax(theta_normal)
            theta_edges = np.linspace(theta_normal_min-1e-10,theta_normal_max+1e-10,4+1)
            theta_lower_list = theta_edges[:-1]
            theta_upper_list = theta_edges[1:]
        # ---------- Reanalysis --------------
        theta_normal_comm_corr = np.nansum((theta_normal - theta_normal.mean())*(qp - qp.mean()))
        keys_ra = list(ra.keys())
        for k in keys_ra:
            rath[k]["delta_theta_normal"] = rath[k]["theta_normal"][:,1:] - rath[k]["theta_normal"][:,:-1]
            rath[k]["thmid_normal"] = 0.5*(rath[k]["theta_normal"][:,1:] + rath[k]["theta_normal"][:,:-1])
            rath[k]["thmid_tangential"] = 0.5*(rath[k]["theta_tangential"][:,1:] + rath[k]["theta_tangential"][:,:-1])
            rath[k]["reactive_flag"] = (ra[k]["src_tag"] == reactive_code[0])*(ra[k]["dest_tag"] == reactive_code[1])
        num_levels = len(theta_lower_list)
        close_idx,reactive_flux = self.reactive_flux_density_levelset(thmid[:,0],Jth,Jweight,theta_lower_list,theta_upper_list)
        if fig is None or ax is None:
            fig,ax = plt.subplots()
        linestyle = '--' if dashed_flag else '-'
        handles = []
        bins = max(int(round((np.nanmax(thmid[:,1]) - np.nanmin(thmid[:,1]))/dth_tangential)), 3)
        dth_tangential = (np.nanmax(thmid[:,1]) - np.nanmin(thmid[:,1]))/bins
        for i_thlev in range(num_levels):
            # Make a histogram of the reactive flux density distribution
            idx = close_idx[i_thlev]
            if len(idx) > 1:
                idx = np.array(idx)
                weights = reactive_flux[i_thlev]
                x = thmid[idx,1]
                #print(f"weights.shape = {weights.shape}, x.shape = {x.shape}")
                hist,bin_edges = np.histogram(thmid[idx,1],weights=reactive_flux[i_thlev][:,0],bins=bins,range=(np.nanmin(thmid[:,1]),np.nanmax(thmid[:,1])))
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                hist *= 1.0*rate/(np.sum(np.abs(hist))) #*dth_tangential)
                if invert_flag: hist *= -1
                # Naive histogram
                hist_nv,bin_edges_nv = np.histogram(thmid[idx,1],bins=bins,weights=np.ones(len(idx)),range=(np.nanmin(thmid[:,1]),np.nanmax(thmid[:,1])))
                bin_centers_nv = (bin_edges_nv[1:] + bin_edges_nv[:-1])/2
                hist_nv *= 1.0/(np.abs(np.sum(hist_nv))*dth_tangential)
                hist_nv *= np.sign(np.sum(hist))/np.sign(np.sum(hist_nv))
                # ------------- Reanalysis --------------------
                theta_mid = 0.5*(theta_lower_list[i_thlev] + theta_upper_list[i_thlev])
                for k in keys_ra:
                    idx_ra_fwd = np.where((rath[k]["theta_normal"][:,:-1] < theta_mid)*(rath[k]["theta_normal"][:,1:] >= theta_mid)*(rath[k]["reactive_flag"][:,:-1]+rath[k]["reactive_flag"][:,1:]))
                    idx_ra_bwd = np.where((rath[k]["theta_normal"][:,:-1] >= theta_mid)*(rath[k]["theta_normal"][:,1:] < theta_mid)*(rath[k]["reactive_flag"][:,:-1]+rath[k]["reactive_flag"][:,1:]))
                    idx_ra = (np.concatenate((idx_ra_fwd[0], idx_ra_bwd[0])),
                            np.concatenate((idx_ra_fwd[1], idx_ra_bwd[1])))
                    signs_ra = np.concatenate((np.ones(len(idx_ra_fwd[0])),-np.ones(len(idx_ra_bwd[0])))) #* np.sign(theta_normal_comm_corr)
                    theta_tangential_ra = 0.5*(rath[k]["theta_tangential"][idx_ra[0],idx_ra[1]] + rath[k]["theta_tangential"][idx_ra[0],idx_ra[1]+1])
                    rath[k]["num_rxn"] = np.sum(signs_ra)*np.sign(theta_normal_comm_corr) 
                    print(f"rath[{k}]['num_rxn'] = {rath[k]['num_rxn']}")
                    # Put together into a histogram
                    if rath[k]["num_rxn"] != 0:
                        hist_ra,bin_edges_ra = np.histogram(theta_tangential_ra, weights=signs_ra,range=(np.nanmin(thmid[:,1]),np.nanmax(thmid[:,1])),bins=bins)
                        if invert_flag:
                            hist_ra *= -1
                        #hist_ra *= num_rxn_ra/(Nxra*dth_tan_ra*np.sum(hist_ra))
                        rath[k]["bin_centers"] = (bin_edges_ra[1:] + bin_edges_ra[:-1])/2
                        rath[k]["hist"] = hist_ra*ra[k]["rate"]/(np.abs(np.sum(hist_ra))) #*dth_tangential)
                        print(f"Just made bins and hist for {k}")
                # ---------------------------------------------
                # Store all the needed info in a Pandas dataframe
                df = pandas.DataFrame({
                    "Time": bin_centers.astype(int),
                    })
                for k in keys_ra:
                    df[k] = rath[k]["hist"]
                if timeseries_like:
                    max_bin_height = 0.6*np.min(np.diff(theta_lower_list))
                    offset_horz = (theta_lower_list[i_thlev] + theta_upper_list[i_thlev])/2
                    x1 = offset_horz * np.ones(bins)
                    for k in keys_ra:
                        if rath[k]["num_rxn"] != 0:
                            print(f"rath[{k}] keys = {rath[k].keys()}")
                            print(f"rath[{k}] hist = {rath[k]['hist']}")
                            x2 = x1 + max_bin_height*rath[k]["hist"]/np.max(np.abs(rath[k]["hist"]))
                            ax.plot(x2, rath[k]["bin_centers"], color=ra[k]["color"], linestyle=linestyle)
                            ax.plot(x1, rath[k]["bin_centers"], color='black', linestyle=linestyle)

                    x2 = x1 + max_bin_height*hist/np.max(hist)
                    ax.plot(x2, bin_centers, color='red', linestyle=linestyle)
                    ax.plot(x1, bin_centers, color='black', linestyle=linestyle)
                    #ax.fill_betweenx(bin_centers, x1, x2, facecolor='red', edgecolor='black',alpha=0.5)
                else:
                    # Make a bar plot
                    df = pandas.DataFrame({
                        "Time": bin_centers.astype(int),
                        })
                    for k in keys_ra:
                        df[k] = rath[k]["hist"]
                    df["s2s"] = hist
                    color = {k: ra[k]["color"] for k in keys_ra}
                    color["s2s"] = "red"
                    # Plot a subset according to the input specs
                    bars2plot = [key for key in keys_ra if rath[key]]
                    if dga_flag: bars2plot += ["s2s"]
                    df.plot(ax=ax, x="Time", y=bars2plot, kind="bar", color=color, rot=0, fontsize=12)
                    leg = [ra[k]["label"] for k in keys_ra]
                    if dga_flag:
                        leg += ["S2S 1996-2016"]
                    ax.legend(leg) #["ERA-Interim","ERA-20C","S2S"])
                    #for k in keys_ra:
                    #    if rath[k]["num_rxn"] != 0:
                    #        hra, = ax.plot(rath[k]["bin_centers"],rath[k]["hist"],color=ra[k]["color"],label=ra[k]["label"],marker='.', linestyle=linestyle)
                    #        handles.append(hra)
                    #h, = ax.plot(bin_centers,hist,color='red',label="S2S",marker='.',linestyle=linestyle)
                    #handles.append(h)
                    #hnv, = ax.plot(bin_centers_nv,hist_nv,color='orange',label="S2S unweighted",marker='.',linestyle=linestyle)
                    #handles.append(hnv)
        #ax.legend(handles=handles)
        if timeseries_like:
            ax.set_xlabel(theta_normal_label,fontdict=font)
            ax.set_ylabel(theta_tangential_label,fontdict=font)
        else:
            ax.set_xlabel(theta_tangential_label,fontdict=font)
            ax.set_ylabel(r"SSW frequency",fontdict=font)
            ax.axhline(y=0,color='black')
        #ax.set_ylim([-0.005,0.03])
        return fig,ax,bins
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
    def plot_trajectory_segments(self,ra,rath,reactive_code,fig,ax,zorder=10,numtraj=2,debug=False):
        # Plot trajectory segments in 2D given by theta_x. only plot the segments with reactive_flag on
        for k in rath.keys():
            if rath[k]["theta"].shape[2] != 2:
                raise Exception(f"ERROR: you gave me a data set of shape {theta_x.shape}, but I need dimension 2 to have size 2")
            reactive_flag = (ra[k]["src_tag"] == reactive_code[0])*(ra[k]["dest_tag"] == reactive_code[1])*(ra[k]["ina"] == 0)*(ra[k]["inb"] == 0)
            any_rxn_idx = np.where(np.any(reactive_flag, axis=1))[0]
            print(f"len(any_rxn_idx) = {len(any_rxn_idx)}")
            prng = np.random.RandomState(6)
            ss = prng.choice(any_rxn_idx,size=min(numtraj,len(any_rxn_idx)),replace=False)
            print(f"For reanalysis {k}, len(ss) = {len(ss)}")
            handles = []
            labels = []
            for i in ss:
                # Plot a dotted line underneath, and a heavy line where reactive.
                xx = rath[k]["theta"][i,:,:]
                label = f"{ra[k]['fall_years'][i]}"
                h, = ax.plot(xx[:,0],xx[:,1],color="black",linewidth=0.7,zorder=zorder,linestyle='--',label=label)
                handles += [h]
                labels += [label]
                ridx = np.sort(np.where(reactive_flag[i])[0])
                if ridx[-1] < len(xx)-1:
                    ridx = np.concatenate((ridx,[ridx[-1]+1]))
                ax.plot(xx[ridx,0],xx[ridx,1],color="black",linewidth=1.5,zorder=zorder)
        if len(labels) == 0:
            print("ERROR: no realized trajectories to plot")
        return handles,labels
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
        coeff1 = 4.0/maxmag
        dsmin,dsmax = np.max(shp)/200,np.max(shp)/15
        coeff0 = dsmax / (np.exp(-coeff1 * maxmag) - 1)
        ds = coeff0 * (np.exp(-coeff1 * Jmag) - 1)
        #ds = dsmin + (dsmax - dsmin)*(Jmag - minmag)/(maxmag - minmag)
        normalizer = ds*(Jmag != 0)/(np.sqrt((J0_proj/(dth[0]))**2 + (J1_proj/(dth[1]))**2) + (Jmag == 0))
        J0_proj *= normalizer*(1 - np.isnan(J0_proj))
        J1_proj *= normalizer*(1 - np.isnan(J1_proj))
        th01,th10 = np.meshgrid(thaxes[0],thaxes[1],indexing='ij') 
        ax.quiver(th01,th10,J0_proj,J1_proj,angles='xy',scale_units='xy',scale=1.0,color='black',width=2.0,headwidth=4.4,units='dots',zorder=4)
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



