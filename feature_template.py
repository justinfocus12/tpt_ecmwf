# Provide a parent class for features of any given system. Most important: definitions of A and B, time-delay embedding, and maps from features to indices. 
import numpy as np
from numpy import save,load
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
import sys
from abc import ABC,abstractmethod

class TPTFeatures(ABC):
    def __init__(self,feature_file,szn_start,szn_end,Nt_szn,szn_avg_window):
        self.feature_file = feature_file # File location to store parameters to specify features, e.g., from seasonal averages
        # szn_(start,end) denote the beginning and end times of the universe for this model.
        self.szn_start = szn_start 
        self.szn_end = szn_end 
        self.Nt_szn = Nt_szn # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.dt_szn = (self.szn_end - self.szn_start)/(self.Nt_szn + 1)
        self.t_szn = np.linspace(self.szn_start,self.szn_end,self.Nt_szn+1)
        self.szn_avg_window = szn_avg_window
        # Delay time will be a different number for each feature. And we will not assume a uniform time sampling. 
        return
    @abstractmethod
    def ina_test(self,y,feat_def,tpt_bndy):
        pass
    @abstractmethod
    def inb_test(self,y,feat_def,tpt_bndy):
        pass
    def compute_src_dest_tags(self,Y,feat_def,tpt_bndy,save_filename=None):
        # Compute where each trajectory started (A or B) and where it's going (A or B). 
        Ny,Nt,ydim = Y.shape
        ina = self.ina_test(Y.reshape((Ny*Nt,ydim)),feat_def,tpt_bndy).reshape((Ny,Nt))
        inb = self.inb_test(Y.reshape((Ny*Nt,ydim)),feat_def,tpt_bndy).reshape((Ny,Nt))
        src_tag = 0.5*np.ones((Ny,Nt))
        dest_tag = 0.5*np.ones((Ny,Nt))
        # Source: move forward in time
        # Time zero, A is the default src
        src_tag[:,0] = 0*ina[:,0] + 1*inb[:,0] + 0.5*(ina[:,0]==0)*(inb[:,0]==0)*(Y[:,0,0] > tpt_bndy['tthresh'][0]) 
        for k in range(1,Nt):
            src_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + src_tag[:,k-1]*(ina[:,k]==0)*(inb[:,k]==0)
        # Dest: move backward in time
        # Time end, A is the default dest
        dest_tag[:,Nt-1] = 0*ina[:,Nt-1] + 1*inb[:,Nt-1] + 0.5*(ina[:,Nt-1]==0)*(inb[:,Nt-1]==0)*(Y[:,-1,0] < tpt_bndy['tthresh'][1])
        for k in np.arange(Nt-2,-1,-1):
            dest_tag[:,k] = 0*ina[:,k] + 1*inb[:,k] + dest_tag[:,k+1]*(ina[:,k]==0)*(inb[:,k]==0)
            time2dest[:,k] = 0*ina[:,k] + 0*inb[:,k] + (Y[:,k+1,self.fidx_Y['time_h']] - Y[:,k,self.fidx_Y['time_h']] + time2dest[:,k+1])*(ina[:,k]==0)*(inb[:,k]==0)
        #print("Overall fraction in B = {}".format(np.mean(inb)))
        #print("At time zero: fraction of traj in B = {}, fraction of traj headed to B = {}".format(np.mean(dest_tag[:,0]==1),np.mean((dest_tag[:,0]==1)*(inb[:,0]==0))))
        result = {'src_tag': src_tag, 'dest_tag': dest_tag, 'time2dest': time2dest}
        if save_filename is not None:
            pickle.dump(result,open(save_filename,'wb'))
        return src_tag,dest_tag,time2dest
    def get_seasonal_stats(self,t_field,field):
        # Get a smoothed seasonal mean and standard deviation for a field 
        if not (t_field.ndim == 1 and (not np.isscalar(field)) and field.shape[0] == t.size):
            raise Exception(f"Shape error: you gave me t_field.shape = {t_field.shape}, and field.shape = {field.shape}. First dimensions must match")
        field_szn_mean_shape = np.array(field.shape).copy()
        field_szn_mean_shape[0] = self.Nt_szn
        field_szn_mean = np.zeros(field_szn_mean_shape)
        field_szn_std = np.zeros(field_szn_mean_shape)
        field_szn_counts = np.zeros(self.Nt_szn)
        for i_time in range(self.Nt_szn):
            idx = np.where(np.abs(t_field - 0.5*(self.t_szn[i_time] + self.t_szn[i_time+1])) <= self.szn_avg_window/2)[0]
            field_szn_mean[i_time] = np.mean(field[idx], axis=0)
            field_szn_std[i_time] = np.std(field[idx], axis=0)
            field_szn_counts[i_time] = len(idx)
        # Where no data are available, make the statistics nan
        field_szn_mean[np.where(field_szn_counts==0)[0]] = np.nan
        field_szn_std[np.where(field_szn_counts==0)[0]] = np.nan
        return field_szn_mean,field_szn_std,field_szn_counts
    def unseason(self,t_field,field,field_szn_mean,field_szn_std,normalize=True,delayed=False):
        if not (t_field.ndim == 1 and (not np.isscalar(field)) and field.shape[0] == t_field.size):
            raise Exception(f"Shape error: you gave me t_field.shape = {t_field.shape} and field.shape = {field.shape}. First dimensions must match")
        tidx = ((t_field - self.t_szn[0])/self.dt_szn).astype(int)
        field_unseasoned = field - field_szn_mean[tidx]
        if normalize:
            field_unseasoned *= 1.0/field_szn_std[wti]
        return field_unseasoned
    def reseason(self,t_field,field_unseasoned,field_szn_mean,field_szn_std,normalize=True):
        if not (t_field.ndim == 1 and (not np.isscalar(field)) and field.shape[0] == t_field.size):
            raise Exception(f"Shape error: you gave me t_field.shape = {t_field.shape} and field.shape = {field.shape}. First dimensions must match")
        tidx = ((t_field - self.t_szn[0])/self.dt_szn).astype(int)
        if normalize:
            mult_factor = field_szn_std[tidx]
        else:
            mult_factor = np.ones_like(field_szn_std[tidx])
        field = mult_factor * field_unseasoned + field_szn_mean
        return field
    @abstractmethod
    def evaluate_features_database(self,*args,**kwargs):
        # Should evaluate a database of raw model output and convert to basic features. Could be the identity map if the model is simple. More likely, some things will be averaged out. This will be a long and slow execution if the database is large. 
        #Result: save out a .npy file
        pass
    def evaluate_tpt_features(self):
        # TODO





