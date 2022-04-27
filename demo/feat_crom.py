# Methods to compute relevant observable functions from the Crommelin model. 


class CrommelinModelFeatures:
    def __init__(self,feature_file,szn_start,szn_end,delaytime=0):
        self.feature_file = feature_file # File location to store parameters to specify features, e.g., from seasonal averages
        # szn_(start,end) denote the beginning and end of the time window during which the event of interest can happen
        self.szn_start = szn_start 
        self.szn_end = szn_end 
        self.delaytime = delaytime # The length of time-delay embedding to use as features in the model. 
        return
    def ina_test(self,y,feat_def,tpt_bndy):
        i_time = self.fidx_Y['time_h']
        i_uref = np.array([self.fidx_Y['uref_dl%i'%(i_dl)] for i_dl in range(self.ndelay)])
        Ny,ydim = y.shape
        ina = np.zeros(Ny,dtype=bool)
        szn_flag = (y[:,i_time] >= tpt_bndy["tthresh"][0])*(y[:,i_time] < tpt_bndy["tthresh"][1])
        uref = y[:,i_uref]
        strong_wind_flag = (np.min(uref[:,:1+nbuffer], axis=1) >= tpt_bndy["uthresh_a"])
        ina = (1-szn_flag) + szn_flag*strong_wind_flag
        return ina
    def inb_test(self,y,feat_def,tpt_bndy):
        # Test whether a reanalysis dataset's components are in B
        Ny,ydim = y.shape
        i_time = self.fidx_Y['time_h']
        i_uref = self.fidx_Y['uref_dl0'] #np.array([self.fidx_Y['uref_dl%i'%(i_dl)] for i_dl in range(self.ndelay)])
        inb = np.zeros(Ny, dtype=bool)
        winter_flag = (y[:,i_time] >= tpt_bndy['tthresh'][0])*(y[:,i_time] < tpt_bndy['tthresh'][1])
        nbuffer = int(round(tpt_bndy['sswbuffer']/self.dtwint))
        uref = y[:,i_uref] #self.uref_history(y,feat_def)[:,-1]
        weak_wind_flag = (uref < tpt_bndy['uthresh_b'])
        inb = winter_flag*weak_wind_flag
        return inb

        

