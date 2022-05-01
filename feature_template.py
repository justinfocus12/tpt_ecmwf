# Provide a parent class for features of any given system. Most important: definitions of A and B, time-delay embedding, and maps from features to indices. 
# The data will be in the form of Xarray DataSets, with a different variable for each feature, as well as appropriate metadata. 
import numpy as np
import xarray as xr
import netCDF4 as nc
from numpy import save,load
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join,exists
import sys
from abc import ABC,abstractmethod

class TPTFeatures(ABC):
    def __init__(self,featspec_file,szn_start,szn_end,Nt_szn,szn_avg_window):
        self.featspec_file = featspec_file # Filename of Xarray DataBase that stores parameters to specify features, e.g., from seasonal averages
        # szn_(start,end) denote the beginning and end times of the universe for this model.
        self.szn_start = szn_start 
        self.szn_end = szn_end 
        self.Nt_szn = Nt_szn # Number of time windows within the season (for example, days). Features will be averaged over each time interval to construct the MSM. 
        self.dt_szn = (self.szn_end - self.szn_start)/(self.Nt_szn + 1)
        self.t_szn_edge = np.linspace(self.szn_start,self.szn_end,self.Nt_szn+1)
        self.t_szn_cent = 0.5*(self.t_szn_edge[:-1] + self.t_szn_edge[1:])
        self.szn_avg_window = szn_avg_window
        # Delay time will be a different number for each feature. And we will not assume a uniform time sampling. 
        return
    # For ina_test and inb_test, Y should be an xarray DataSet. Each component array has dimensions (member,simtime). One of the observables will be physical time.
    @abstractmethod
    def ina_test(self,Y,featspec,tpt_bndy):
        pass
    @abstractmethod
    def inb_test(self,Y,featspec,tpt_bndy):
        pass
    def compute_src_dest_tags(self,Y,featspec,tpt_bndy,save_filename=None):
        # Compute where each trajectory started (A or B) and where it's going (A or B). 
        """
        Parameters
        ----------
        Y: xarray.DataArray
            Dimensions must be ('feature','member','simtime')
        featspec: xarray.DataSet
            A metadata structure that specifies how feaures are computed and evaluated. This will depend on the application.
        tpt_bndy: dict
            Metadata specifying the boundaries of the current TPT problem. 
        save_filename: str
            netcdf file in which to save the seasonal statistics.
        Returns
        ----------
        src_dest: xarray.Dataset 
            Two DataArrays: src, encoding where the system last came from, and dest, encoding where it is headed to next. Code 0 means A, code 1 means B.
        """
        Ny,Nt = [Y[v].size for v in ['member','simtime']]
        Nfeat = Y.Nfeat
        ina = self.ina_test(Y,featspec,tpt_bndy)
        inb = self.inb_test(Y,featspec,tpt_bndy)
        src_tag = xr.Variable(dims=('member','simtime'), data=0.5*np.ones((Ny,Nt)))
        dest_tag = xr.Variable(dims=('member','simtime'), data=0.5*np.ones((Ny,Nt)))
        # Source: move forward in time
        # Time zero, A is the default src
        src_tag.data[:,0] = 0*ina[:,0] + 1*inb[:,0] + 0.5*(ina[:,0]==0)*(inb[:,0]==0)*(Y[:,0,0] > tpt_bndy['tthresh'][0]) 
        for k in range(1,Nt):
            src_tag.data[:,k] = 0*ina[:,k] + 1*inb[:,k] + src_tag.data[:,k-1]*(ina[:,k]==0)*(inb[:,k]==0)
        # Dest: move backward in time
        # Time end, A is the default dest
        dest_tag.data[:,Nt-1] = 0*ina[:,Nt-1] + 1*inb[:,Nt-1] + 0.5*(ina[:,Nt-1]==0)*(inb[:,Nt-1]==0)*(Y[:,-1,0] < tpt_bndy['tthresh'][1])
        for k in np.arange(Nt-2,-1,-1):
            dest_tag.data[:,k] = 0*ina[:,k] + 1*inb[:,k] + dest_tag.data[:,k+1]*(ina[:,k]==0)*(inb[:,k]==0)
        result = xr.Dataset({"src_tag": src_tag, "dest_tag": dest_tag})
        # Structure the result into an xarray dataset
        if save_filename is not None:
            result.to_netcdf(save_filename)
        return result
    def get_seasonal_stats(self,field):
        """
        Parameters
        ----------
        field: xarray.DataArray
            Some time-dependent field with possibly many realizations.
            Must have dimensions ('feature','member'), where Y.feature.coords must include 't_szn' and Y.member.coords is a list of integers to index the realization.
        featspec: xarray.DataSet
            A metadata structure that specifies how feaures are computed and evaluated. This will depend on the application.
        tpt_bndy: dict
            Metadata specifying the boundaries of the current TPT problem. 
        save_filename: str
            netcdf file in which to save the seasonal statistics.

        Returns
        -------
        field_szn_mean: xarray.DataArray
            Dimensions ('feature','t_szn'). The value of any feature at a given time is the mean of the feature over all realizations of field in the same time window.
        """
        if not ('feature' in field.dims and 'member' in field.dims and "t_szn" in field.coords['feature']):
            raise Exception("You must pass a field with dimensions including 'feature' and 'member', and with 't_szn' as a coordinate of 'feature'. You passed a field with dimensions \n{field.dims}")
        # Construct DataArray for the seasonal mean
        coords = {key: field.coords[key] for key in field.dims if key != 'member'}
        coords['t_szn'] = self.t_szn_cent
        field_szn_mean_shape = np.zeros([coords[key].size for key in coords.keys()])
        field_szn_stats = xr.Dataset({stat: xr.DataArray(
            coords=coords, data=np.nan*np.ones([coords[key].size for key in coords.keys()]))} 
            for stat in ["mean", "std"]})
        # Step through time and average the corresponding samples whose timesteps match within the given time window
        field_t_szn_window = ((field.sel(dict(feature='t_szn')) - self.t_szn_edge[0])/self.dt_szn).astype(int)
        for i_szn in range(self.Nt_szn):
            # select all the points where the field's t_szn falls in the appropriate time window
            selection =  xr.where((field.sel(feature='t_szn')>=self.t_szn_edge[i_szn])*(field.sel(feature='t_szn')<=self.t_szn_edge[i_szn+1]))
            # Take statistics over all data points falling in that window
            field_szn_stats["mean"] = selection.mean(dim=["member","t_szn"], skipna=True)
            field_szn_stats["std"] = selection.std(dim=["member","t_szn"], skipna=True)
        return field_szn_stats
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
    def resample_cycles(self,szn_id_fname,szn_id_resamp):
        szn_id = np.load(szn_id_fname)
        # Resample trajectories to only come from a certain set of cycles 
        idx_resamp = []
        for i in range(len(szn_id_resamp)):
            match_idx = np.where(szn_id == szn_id_resamp[i])[0]
            idx_resamp += [np.sort(match_idx)]
        return np.array(idx_resamp)
    @abstractmethod
    def evaluate_tpt_features(self,X_fname,Y_fname,szn_id_resamp,featspec,*args,**kwargs):
        # In Y_fname, save a dictionary with key-value mappings 
        # Y: array of observables, including time
        # id_szn: the unique ID associated with the season in which it was launched (e.g., the year)
        pass
    @abstractmethod
    def set_feature_indices_X(self,featspec): 
        pass
    @abstractmethod
    def set_feature_indices_Y(self,featspec):
        pass
    @abstractmethod
    def observable_function_library_X(self):
        pass
    @abstractmethod
    def observable_function_library_X(self):
        pass
    def overlay_hc_ra(self,
            tpt_bndy_list,featspec,
            Y_fname_ra,Y_fname_hc,
            obs_name, # Tells us which observable function to use
            label_ra,label_hc,
            feat_display_dir):
        # Evaluate the observable functions 
        funlib = self.observable_function_library_Y()
        Yra_dict = pickle.load(open(Y_fname_ra,"rb"))
        Yhc_dict = pickle.load(open(Y_fname_hc,"rb"))
        Yra = Yra_dict["Y"]
        id_szn_ra = Yra_dict["id_szn"]
        Nyra,Ntyra,_ = Yra.shape
        Yhc = Yhc_dict["Y"]
        id_szn_hc = Yhc_dict["id_szn"]
        Nyhc,Ntyhc,_ = Yhc.shape
        fra = funlib[obs_name]["fun"](Yra)
        fhc = funlib[obs_name]["fun"](Yhc)
        # Evaluate the climatology quantiles
        quantile_ranges = [0.4, 0.8, 1.0]
        lower = np.zeros((len(quantile_ranges),self.Nt_szn))
        upper = np.zeros((len(quantile_ranges),self.Nt_szn))
        t_ra = Y[:,:,self.fidx_Y["time"]]
        tidx_szn = ((t_ra - self.t_szn[0])/self.dt_szn).astype(int)
        for ti in range(self.Nt_szn):
            idx = np.where(tidx_szn == ti)
            for qi in range(len(quantile_ranges)):
                lower[qi,ti] = np.quantile(fra[idx[0],idx[1]], 0.5-0.5*quantile_ranges[qi])
                upper[qi,ti] = np.quantile(fra[idx[0],idx[1]], 0.5+0.5*quantile_ranges[qi])
        # Determine reanalysis years that cross each threshold but not the next one
        src_tag_list = []
        dest_tag_list = []
        rxn_tag_list = []
        ina_list = []
        inb_list = []
        for i_bndy,tpt_bndy in enumerate(tpt_bndy_list):
            src_tag,dest_tag = self.compute_src_dest_tags(Y,featspec,tpt_bndy)
            ina_list += [self.ina_test(Yra,featspec,tpt_bndy)]
            inb_list += [self.inb_test(Yra,featspec,tpt_bndy)]
            src_tag_list += [src_tag]
            dest_tag_list += [dest_tag]
            rxn_tag_list += [np.where(np.any((src_tag==0)*(dest_tag==1), axis=1))[0]]
        for i_bndy in np.arange(len(tpt_bndy_list)-1, 0, -1):
            rxn_tag_list[i_bndy] = np.setdiff1d(rxn_tag_list[i_bndy],rxn_tag_list[i_bndy-1])
        for i_bndy,tpt_bndy in enumerate(tpt_bndy_list):
            if len(rxn_tag_list[i_bndy]) > 0:
                fig,ax = plt.subplots()
                # Plot the climatological envelopes
                for qi in range(len(quantile_ranges)):
                    ax.fill_between(self.t_szn_cent, lower[qi], upper[qi], color=plt.cm.binary(0.2 + 0.6*(1-quantile_ranges[qi])), zorder=-1)
                # Plot the reanalysis, colored when it hits A or B
                idx = rxn_tag_list[i_bndy][0]
                for i_time in range(self.Nt_szn-1):
                    if ina_list[i_bndy][idx,i_time]: 
                        color = 'dodgerblue'
                    elif inb_list[i_bndy][idx,i_time]:
                        color = 'red'
                    else:
                        color = 'black'
                    ax.plot(self.t_szn_cent[i_time:i_time+2], Yra[idx][i_time:i_time+2], color=color)
                # Plot two hindcast trajectories 
                i_cycle = 

            



        # Plot the hindcasts as bundles overlying the reanalysis trajectories. Color differently the curves entering set B and A. 





