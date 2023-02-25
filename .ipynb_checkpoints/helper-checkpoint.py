import numpy as np
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
import sys
from sklearn import linear_model

def mantexp(num):
    # Generate the mantissa an exponent
    if num == 0:
        return 0,0
    exponent = int(np.log10(np.abs(num)))
    mantissa = num/(10**exponent)
    if np.abs(mantissa) < 1:
        mantissa *= 10
        exponent -= 1
    return mantissa,exponent
def generate_sci_fmt(xmin,xmax,numdiv=100):
    # Print to two sig figs
    eps = (xmax-xmin)/numdiv
    #print("eps = {}".format(eps))
    general_sci_fmt = lambda num,pos: sci_not_precision(num,eps)
    return general_sci_fmt
def sci_not_precision(num,eps):
    # Specify a number to an accuracy of epsilon/10
    #print("num = {}".format(num))
    if np.abs(num) < eps*1e-3: 
        #print("num = {}; returning 0".format(num))
        return "0"
    Mn,En = mantexp(num)
    Me,Ee = mantexp(eps)
    # Need Ee+1 places past the decimal point
    #digs = np.abs(Ee) # Wrong for large eps
    # Need enough digits to distinguish num from num+eps
    digs = max(0,En-Ee)
    #print("digs = {}".format(digs))
    num_to_prec = eval(("{:."+str(digs)+"e}").format(num))
    #print("num_to_prec = {}".format(num_to_prec))
    # Now format it accordingly
    Mn,En = mantexp(num_to_prec)
    if np.abs(En) > 2:
        #sci = ("{:."+str(digs)+"f}\\times 10^{}").format(Mn,En)
        sci = ("{:."+str(digs)+"f}").format(Mn)
        #sci = "%s\\times 10^{%d}"%(sci,En)
        sci = "%se%d"%(sci,En)
    else:
        #sci = ("{:."+str(digs)+"f}").format(num_to_prec)
        sci = ("{:."+str(digs)+"f}").format(num_to_prec)
    return sci #num_to_prec 
def fmt(num,pos):
    return '{:.1f}'.format(num)
def fmt2(num,pos):
    return '{:.2f}'.format(num)
def fmt3(num,pos):
    return '{:.3f}'.format(num)
def sci_fmt(num,lim):
    return '{:.1e}'.format(num)
def sci_fmt_short(num,lim):
    return '{:.0e}'.format(num)
def sci_fmt_latex0(num):
    # Convert a number to scientific notation
    exponent = int(np.log10(np.abs(num)))
    mantissa = num/(10**exponent)
    if np.abs(mantissa) < 1:
        mantissa += np.sign(mantissa)
        exponent -= 1
    if exponent != 0:
        sci = "%.0f\\times 10^{%d}" % (mantissa,exponent)
    else:
        sci = r"%.0f" % mantissa
    return sci
def sci_fmt_latex1(num):
    # Convert a number to scientific notation
    exponent = int(np.log10(np.abs(num)))
    mantissa = num/(10**exponent)
    if np.abs(mantissa) < 1:
        mantissa += np.sign(mantissa)
        exponent -= 1
    if exponent != 0:
        sci = "%.1f\\times 10^{%d}" % (mantissa,exponent)
    else:
        sci = r"%.1f" % mantissa
    return sci
def sci_fmt_latex(num):
    # Convert a number to scientific notation
    exponent = int(np.log10(np.abs(num)))
    mantissa = num/(10**exponent)
    if np.abs(mantissa) < 1:
        mantissa += np.sign(mantissa)
        exponent -= 1
    if exponent != 0:
        sci = "%.2f\\times 10^{%d}" % (mantissa,exponent)
    else:
        sci = r"$%.2f$" % mantissa
    return sci

def mean_uncertainty(X,num_blocks=10):
    # Given a list of numbers X, return the mean and the uncertainty in the mean
    N = len(X)
    block_size = int(N/num_blocks)
    N = block_size*num_blocks # might be less than len(X), but not by more than block_size-1
    block_means = np.zeros(num_blocks)
    idx = np.arange(N).reshape((num_blocks,block_size))
    for i in range(num_blocks):
        block_means[i] = np.mean(X[idx[i]])
    unc = np.std(block_means)
    print("mean(X) = {}. min(block_means) = {}. max(block_means) = {}, unc = {}".format(np.mean(X),np.min(block_means),np.max(block_means),unc))
    return unc

def weighted_quantile(x,w,q):
    # For an array x with weights w, find the quantile q
    order = np.argsort(x)
    qx = np.cumsum(w[order])/np.sum(w)
    if q <= 0:
        return x[order[0]]
    if q >= 1:
        return x[order[-1]]
    else:
        i0 = np.where(qx < q)[0][-1]
        i1 = i0 + 1
        xst = x[order[i0]] + (q - qx[i0])/(qx[i1] - qx[i0])*(x[order[i1]] - x[order[i0]])
    return xst

def maximize_entropy_given_moments(x,mom):
    # Given a discrete range of values x, and a list of known moments mom, construct the maximum-entropy PDF
    # mom lists the moments 1,...,K
    moments = np.concatenate(([1.0],mom))
    print("moments = {}".format(moments))
    maxmom = len(mom)
    X = np.array([x**k for k in np.arange(maxmom+1)]).T # shape is (maxmom+1, len(x))
    def F(lam):
        expo = np.exp(-1-X.dot(lam))
        fun = np.sum(lam*moments) + np.sum(expo)
        jac = moments - X.T.dot(expo)
        #hess = - (X.T * expo).dot(X)
        return fun,jac
    lam0 = np.zeros(maxmom+1)
    res = minimize(F,lam0,method='Newton-CG',jac=True)
    pme = np.exp(-1-X.dot(res.x))
    return pme
    


def both_grids(bounds,shp):
    # This time shp is the number of cells
    Nc = np.prod(shp-1)   # Number of centers
    Ne = np.prod(shp) # Number of edges
    center_grid = np.array(np.unravel_index(np.arange(Nc),shp-1)).T
    edge_grid = np.array(np.unravel_index(np.arange(Ne),shp)).T
    dx = (bounds[:,1] - bounds[:,0])/(shp - 1)
    center_grid = bounds[:,0] + dx * (center_grid + 0.5)
    edge_grid = bounds[:,0] + dx * edge_grid
    return center_grid,edge_grid,dx

def project_field(field,weight,theta_x,shp=None,avg_flag=True,bounds=None):
    if np.min(weight) < 0:
        sys.exit("Negative weights")
    # Given a vector-valued observable function evaluation theta_x, find the mean 
    # and standard deviation of the field across remaining dimensions
    # Also return some integrated version of the standard deviation
    thdim = theta_x.shape[1]
    if shp is None: shp = 20*np.ones(thdim,dtype=int) # number of INTERIOR
    if bounds is None:
        bounds = np.array([np.nanmin(theta_x,0)-1e-10,np.nanmax(theta_x,0)+1e-10]).T
    cgrid,egrid,dth = both_grids(bounds, shp+1)
    thaxes = [np.linspace(bounds[i,0]+dth[i]/2,bounds[i,1]-dth[i]/2,shp[i]) for i in range(thdim)]
    data_bins = ((theta_x - bounds[:,0])/dth).astype(int)
    for d in range(len(shp)):
        data_bins[:,d] = np.maximum(data_bins[:,d],0)
        data_bins[:,d] = np.minimum(data_bins[:,d],shp[d]-1)
    data_bins_flat = np.ravel_multi_index(data_bins.T,shp) # maps data points to bin
    Ncell = np.prod(shp)
    filler = np.nan if avg_flag else 0.0
    field_mean = filler*np.ones(Ncell)
    field_std = filler*np.ones(Ncell)
    for i in range(Ncell):
        idx = np.where(data_bins_flat == i)[0]
        if len(idx) > 0 and not np.all(np.isnan(field[idx])):
            weightsum = np.sum(weight[idx]*(1-np.isnan(field[idx])))
            field_mean[i] = np.nansum(field[idx]*weight[idx])
            #if avg_flag and (weightsum == 0):
            #    sys.exit("Doh! supposed to average, but weights are zero!")
            if avg_flag and (weightsum != 0):
                field_mean[i] *= 1/weightsum
                field_std[i] = np.sqrt(np.nansum((field[idx]-field_mean[i])**2*weight[idx]))
                field_std[i] *= 1/np.sqrt(weightsum)
                field_range = np.nanmax(field[idx])-np.nanmin(field[idx])
                #if (len(idx) > 1) and (field_mean[i] < np.min(field[idx])-0.05*field_range or field_mean[i] > np.max(field[idx])+0.05*field_range):
                if (field_mean[i] < np.min(field[idx])) and np.abs((field_mean[i] - np.min(field[idx]))/np.min(field[idx])) > 0.05:
                        sys.exit("Doh! Too low! field_mean[i]={}, min(field[idx])={}".format(field_mean[i],np.min(field[idx])))
                if (field_mean[i] > np.max(field[idx])) and np.abs((field_mean[i] - np.max(field[idx]))/np.max(field[idx])) > 0.05:
                        sys.exit("Doh! Too high! field_mean[i]={}, max(field[idx])={}".format(field_mean[i],np.max(field[idx])))
                    #sys.exit("Doh! Average is outside the bounds! len(idx)={}\n field_mean[i] = {}\n field[idx] in ({},{})\n weights in ({},{})\n".format(len(idx),field_mean[i],np.min(field[idx]),np.max(field[idx]),np.min(weight[idx]),np.max(weight[idx])))
    field_std_L2 = np.sqrt(np.nansum(field_std**2)/Ncell) #*np.prod(dth))
    field_std_Linf = np.nanmax(field_std)*np.prod(dth)
    return shp,dth,thaxes,cgrid,field_mean,field_std,field_std_L2,field_std_Linf,bounds

def plot_field_1d(theta,u,weight,shp=[20,],uname="",thetaname="",avg_flag=True,std_flag=False,fig=None,ax=None,color='black',label="",linestyle='-',linewidth=1,orientation='horizontal',units=1.0,unit_symbol="",eq_ax=False,density_flag=False):
    shp = np.array(shp)
    # Plot a 1d scatterplot of a field average across remaining dimensions
    print("avg_flag = {}".format(avg_flag))
    shp,dth,thaxes,cgrid,u_mean,u_std,u_std_L2,u_std_Linf,_ = project_field(u,weight,theta.reshape(-1,1),shp,avg_flag=avg_flag)
    if density_flag: 
        u_mean *= dth[0]/units
        u_std *= dth[0]/units
    print("shp0 = {}, dth={}".format(shp,dth*units))
    print("thaxes in ({},{})".format(thaxes[0][0]*units,thaxes[0][-1]*units))
    print("u in ({},{}), u_mean in ({},{})".format(np.nanmin(u),np.nanmax(u),np.nanmin(u_mean),np.nanmax(u_mean)))
    if (fig is None) or (ax is None):
        fig,ax = plt.subplots(figsize=(6,6),constrained_layout=True)
    if orientation=='horizontal':
        handle, = ax.plot(units*thaxes[0],u_mean,marker='o',linestyle=linestyle,color=color,label=label,linewidth=linewidth)
        if std_flag:
            ax.plot(units*thaxes[0],u_mean-u_std,color=color,linestyle='--',linewidth=linewidth)
            ax.plot(units*thaxes[0],u_mean+u_std,color=color,linestyle='--',linewidth=linewidth)
        xlab = thetaname
        if len(unit_symbol) > 0: xlab += " ({})".format(unit_symbol)
        ax.set_xlabel(xlab,fontdict=font)
        ax.set_ylabel(uname,fontdict=font)
        ax.set_xlim([np.min(units*theta),np.max(units*theta)])
    else:
        handle, = ax.plot(u_mean,units*thaxes[0],marker='o',linestyle=linestyle,color=color,label=label,linewidth=linewidth)
        if std_flag:
            ax.plot(u_mean-u_std,units*thaxes[0],color=color,linestyle='--')
            ax.plot(u_mean+u_std,units*thaxes[0],color=color,linestyle='--')
        ylab = thetaname
        if len(unit_symbol) > 0: ylab += " ({})".format(unit_symbol)
        print("ylab = {}".format(ylab))
        ax.set_ylabel(ylab,fontdict=font)
        ax.set_xlabel(uname,fontdict=font)
        ax.set_ylim([np.min(units*theta),np.max(units*theta)])
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    fmt_x = generate_sci_fmt(xlim[0],xlim[1])
    fmt_y = generate_sci_fmt(ylim[0],ylim[1])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',labelsize=10)
    #ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    if eq_ax:
        xylim = np.array([ax.get_xlim(),ax.get_ylim()])
        xylim = np.array([np.min(xylim[:,0]),np.max(xylim[:,1])])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.plot(xylim,xylim,color='black',linestyle='--')
    return fig,ax,handle

def plot_field_2d(field,weight,theta_x,shp=[20,20],cmap=plt.cm.coolwarm,fieldname="",fun0name="",fun1name="",avg_flag=True,std_flag=False,logscale=False,ss=None,units=np.ones(2),unit_symbols=["",""],cbar_orientation='horizontal',cbar_location='top',fig=None,ax=None,vmin=None,vmax=None,cbar_pad=0.2,fmt_x=None,fmt_y=None,contourflag=True):
    # The function inside TPT should just extend this one
    shp = np.array(shp)
    shp,dth,thaxes,cgrid,field_mean,field_std,field_std_L2,field_std_Linf,_ = project_field(field,weight,theta_x,shp,avg_flag=avg_flag)
    if std_flag:
        if fig is None or ax is None:
            fig,ax = plt.subplots(ncols=2,figsize=(12,6))
        ax0,ax1 = ax[0],ax[1]
    else:
        if fig is None or ax is None:
            fig,ax = plt.subplots(figsize=(6,6))
        ax0 = ax
    th01,th10 = np.meshgrid(units[0]*thaxes[0],units[1]*thaxes[1],indexing='ij')
    thaxes_edge = [np.concatenate(([1.5*th[0]-0.5*th[1]],(th[1:]+th[:-1])/2,[1.5*th[-1]-0.5*th[-2]])) for th in thaxes]
    th01_edge,th10_edge = np.meshgrid(units[0]*thaxes_edge[0],units[1]*thaxes_edge[1],indexing='ij')

    if logscale:
        realidx = np.where(np.isnan(field_mean)==0)[0]
        if len(realidx) > 0:
            posidx = realidx[np.where(field_mean[realidx] > 0)[0]]
        #field_mean[posidx] = np.log10(field_mean[posidx])
        field_mean[np.setdiff1d(np.arange(np.prod(shp)),posidx)] = np.nan
    locator = ticker.LogLocator(numticks=10) if logscale else ticker.MaxNLocator()
    if contourflag:
        im = ax0.contourf(th01,th10,field_mean.reshape(shp),cmap=cmap,locator=locator,zorder=1,vmin=vmin,vmax=vmax)
    else:
        im = ax0.pcolormesh(th01_edge,th10_edge,field_mean.reshape(shp),cmap=cmap,zorder=1,vmin=vmin,vmax=vmax)
    xlim = np.nanmin(units[0]*thaxes[0]),np.nanmax(units[0]*thaxes[0])
    #print("xlim = {}".format(xlim))
    ylim = np.nanmin(units[1]*thaxes[1]),np.nanmax(units[1]*thaxes[1])
    ax0.set_xlim([np.nanmin(units[0]*thaxes[0]),np.nanmax(units[0]*thaxes[0])])
    ax0.set_ylim([np.nanmin(units[1]*thaxes[1]),np.nanmax(units[1]*thaxes[1]) + 0.15*units[1]*np.ptp(thaxes[1])])
    #print("eps = {} - {}".format(np.nanmax(field_mean),np.nanmin(field_mean)))
    cbar_fmt = generate_sci_fmt(np.nanmin(field_mean),np.nanmax(field_mean),20)
    # -------------------
    # New colorbar code
    ax0_left,ax0_bottom,ax0_width,ax0_height = ax0.get_position().bounds
    if cbar_orientation == 'vertical':
        sys.exit("Not doing vertical colorbars right now")
    elif cbar_orientation == 'horizontal':
        if cbar_location == 'bottom':
            cbaxes = fig.add_axes([0.2,0.00,0.8,0.01])
        elif cbar_location == 'top':
            cbaxes = fig.add_axes([ax0_left+0.1*ax0_width,ax0_bottom+0.97*ax0_height,0.8*ax0_width,0.03*ax0_height])
        if not logscale:
            cbarmin,cbarmax = np.nanmin(field_mean),np.nanmax(field_mean)
            if vmin is not None: cbarmin = max(cbarmin,vmin)
            if vmax is not None: cbarmax = min(cbarmax,vmax)

            cbar = plt.colorbar(im, ax=ax0, cax=cbaxes, orientation=cbar_orientation, format=ticker.FuncFormatter(cbar_fmt), ticks=np.linspace(cbarmin,cbarmax,4))
        else:
            cbar = plt.colorbar(im, ax=ax0, cax=cbaxes, ticks=10.0**(np.linspace(np.log10(np.nanmin(field_mean)),np.log10(np.nanmax(field_mean)),4).astype(int)), orientation='horizontal')
        cbar.ax.tick_params(labelsize=15)
    
    ax0.tick_params(axis='x',labelsize=14)
    ax0.tick_params(axis='y',labelsize=14)
    xlim,ylim = ax0.get_xlim(),ax0.get_ylim()
    fmt_x = generate_sci_fmt(xlim[0],xlim[1])
    fmt_y = generate_sci_fmt(ylim[0],ylim[1])
    ax0.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    if std_flag:
        if contourflag:
            im = ax1.contourf(th01,th10,field_std.reshape(shp),cmap=plt.cm.magma)
        else:
            im = ax1.pcolor(th01_edge,th10_edge,field_std.reshape(shp),cmap=plt.cm.magma)
        ax1.tick_params(axis='x',labelsize=10)
        ax1.tick_params(axis='y',labelsize=10)
    ax0.set_title("{}".format(fieldname),fontdict=font,y=1.0) #,loc='left')
    xlab = fun0name
    if len(unit_symbols[0]) > 0: xlab += " [{}]".format(unit_symbols[0])
    ylab = fun1name
    if len(unit_symbols[1]) > 0: ylab += " [{}]".format(unit_symbols[1])
    ax0.set_xlabel("{}".format(xlab),fontdict=font)
    ax0.set_ylabel("{}".format(ylab),fontdict=font)
    if std_flag: 
        cbar = fig.colorbar(im,ax=ax[1],format=ticker.FuncFormatter(fmt),orientation=cbar_orientation,pad=0.2,ticks=np.linspace(np.nanmin(field_std),np.nanmax(field_std),4))
        cbar.ax.tick_params(labelsize=10)
        ax1.set_title(r"Std; $L^2=%.2e$"%(field_std_L2),fontdict=font)
        ax1.set_xlabel("{}".format(xlab),fontdict=font)
        ax1.set_ylabel("{}".format(ylab),fontdict=font)
    return fig,ax 
def reweight_data(x,theta_fun,algo_params,theta_pdf):
    # theta_fun is a CV space; theta_pdf is a density function on that CV space (need not be normalized)
    # Given a reference dataset meant to be pi-distributed, resample
    # ref_data could be transformed
    Nx = len(x)
    theta_x = theta_fun(x,algo_params)
    theta_weights = theta_pdf(theta_x)
    shp,dth,thaxes,cgrid,field_mean,field_std,field_std_L2,field_std_Linf,bounds = project_field(np.ones(Nx),np.ones(Nx),theta_x,avg_flag=False)
    lower_bounds = np.array([th[0] for th in thaxes])
    data_bins = ((theta_x - lower_bounds)/dth).astype(int)
    data_bins_flat = np.ravel_multi_index(data_bins.T,shp).T
    empirical_weights = field_mean[data_bins_flat]
    print("empirical weights: min={}, max={}, mean={}, std={}".format(np.min(empirical_weights),np.max(empirical_weights),np.mean(empirical_weights),np.std(empirical_weights)))
    sample_weights = theta_weights*(empirical_weights!=0) / (empirical_weights + 1*(empirical_weights==0))
    #sample_weights = 1*(empirical_weights==0) / (empirical_weights + 1*(empirical_weights==0))
    sample_weights *= 1.0/np.sum(sample_weights)
    return sample_weights


def compare_fields(theta0,theta1,u0,u1,weights0,weights1,shp=None,avg_flag=True,subset_flag=True):
    # u_emp is some timeseries that is a function following the long trajectory. u0 is its computed (conditional) expectation
    # theta_fun is some CV space that we will grid up and compare u0 to u1 averaged over each box
    N0 = len(theta0)
    N1 = len(theta1)
    if subset_flag:
        ss0 = np.random.choice(np.arange(N0),size=min(N0,10000),replace=True)
        ss1 = np.random.choice(np.arange(N1),size=min(N1,10000),replace=True)
    else:
        ss0 = np.arange(N0)
        ss1 = np.arange(N1)
    if shp is None: shp = 10*np.ones(2,dtype=int)
    shp = np.array(shp)
    shp,dth,thaxes,cgrid,u0_grid,u0_std,u0_std_L2,u0_std_Linf,bounds = project_field(u0[ss0],weights0[ss0]/np.sum(weights0[ss0]),theta0[ss0],avg_flag=avg_flag,shp=shp)
    _,_,_,_,u1_grid,_,_,_,_ = project_field(u1[ss1],weights1[ss1]/np.sum(weights1[ss1]),theta1[ss1],avg_flag=avg_flag,shp=shp,bounds=bounds)
    return shp,dth,thaxes,cgrid,u0_grid,u1_grid

def compare_plot_fields_1d(theta0,theta1,u0,u1,weights0,weights1,theta_name="",u_names=["",""],theta_units=1.0,theta_unit_symbol="",avg_flag=True,logscale=False,shp=None):
    N0 = len(theta0)
    N1 = len(theta1)
    shp,dth,thaxes,cgrid,u0_grid,u1_grid = compare_fields(theta0,theta1,u0,u1,weights0,weights1,shp=shp,avg_flag=avg_flag)
    # Also get the weights for each bin
    _,_,_,_,w0_grid,w1_grid = compare_fields(theta0,theta1,np.ones(N0),np.ones(N1),weights0,weights1,shp=shp,avg_flag=False)
    # Compute some total error metric
    fig,ax = plt.subplots(ncols=2,figsize=(12,6))
    scatter_subset = np.where((u0_grid>0)*(u1_grid>0))[0] if logscale else np.arange(len(u0_grid))
    total_error = np.sqrt(np.nansum((u0_grid-u1_grid)**2*w1_grid)/np.nansum(w1_grid))
    h = ax[0].scatter(u0_grid[scatter_subset],u1_grid[scatter_subset],marker='o',color='black',s=50*w1_grid[scatter_subset]/np.max(w1_grid[scatter_subset]),label=r"Avg. Error = %3.3e"%(total_error))
    ax[0].legend(handles=[h],prop={'size':12})
    umin = min(np.nanmin(u0_grid[scatter_subset]),np.nanmin(u1_grid[scatter_subset]))
    umax = max(np.nanmax(u0_grid[scatter_subset]),np.nanmax(u1_grid[scatter_subset]))
    ax[0].set_xlim([umin,umax])
    ax[0].set_ylim([umin,umax])
    ax[0].plot([umin,umax],[umin,umax],linestyle='--',color='black')
    ax[0].set_xlabel(u_names[0],fontdict=font)
    ax[0].set_ylabel(u_names[1],fontdict=font)
    ax[0].set_title("DGA fidelity")
    handle0, = ax[1].plot(thaxes[0],u0_grid,marker='o',color='red',label=u_names[0])
    handle1, = ax[1].plot(thaxes[0],u1_grid,marker='o',color='black',label=u_names[1])
    #handle0, = ax[1].plot(thaxes[0][scatter_subset],u0_grid[scatter_subset],marker='o',color='red',label=u_names[0])
    #handle1, = ax[1].plot(thaxes[0][scatter_subset],u1_grid[scatter_subset],marker='o',color='black',label=u_names[1])
    ax[1].legend(handles=[handle0,handle1],prop={'size': 12})
    ax[1].set_xlabel(theta_name,fontdict=font)
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    return fig,ax

def compare_plot_fields_2d(theta0,theta1,u0,u1,weights0,weights1,theta_names=["",""],u_names=["",""],theta_units=np.ones(2),theta_unit_symbols=["",""],avg_flag=True,logscale=False,shp=None):
    N0 = len(theta0)
    N1 = len(theta1)
    ss0 = np.random.choice(np.arange(N0),size=min(N0,500000),replace=False)
    ss1 = np.random.choice(np.arange(N1),size=min(N1,500000),replace=False)
    shp,dth,thaxes,cgrid,u0_grid,u1_grid = compare_fields(theta0,theta1,u0,u1,weights0,weights1,shp=shp,avg_flag=avg_flag)
    # Get the weights for each grid box
    _,_,_,_,w0_grid,w1_grid = compare_fields(theta0,theta1,np.ones(N0),np.ones(N1),weights0,weights1,shp=shp,avg_flag=False)
    fig,ax = plt.subplots(ncols=3,figsize=(18,6),sharey=False)
    scatter_subset = np.where((u0_grid>0)*(u1_grid>0))[0] if logscale else np.arange(len(u0_grid))
    total_error = np.sqrt(np.nansum((u0_grid-u1_grid)**2*w1_grid)/np.nansum(w1_grid))
    h = ax[0].scatter(u0_grid[scatter_subset],u1_grid[scatter_subset],marker='o',color='black',s=50*w1_grid[scatter_subset]/np.max(w1_grid[scatter_subset]),label=r"Avg. Error = %3.3e"%(total_error))
    ax[0].legend(handles=[h],prop={'size':12})
    umin = min(np.nanmin(u0_grid[scatter_subset]),np.nanmin(u1_grid[scatter_subset]))
    umax = max(np.nanmax(u0_grid[scatter_subset]),np.nanmax(u1_grid[scatter_subset]))
    #chmin = 1e-8
    #chmax = 1.0
    ax[0].set_xlim([umin,umax])
    ax[0].set_ylim([umin,umax])
    ax[0].plot([umin,umax],[umin,umax],linestyle='--',color='black')
    ax[0].set_xlabel(u_names[0],fontdict=font)
    ax[0].set_ylabel(u_names[1],fontdict=font)
    ax[0].set_title(r"DGA fidelity",fontdict=font)
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    _,_ = plot_field_2d(u0[ss0],weights0[ss0]/np.sum(weights0[ss0]),theta0[ss0],avg_flag=avg_flag,cmap=plt.cm.coolwarm,fun0name=theta_names[0],fun1name=theta_names[1],fieldname=u_names[0],std_flag=False,logscale=0*logscale,ss=ss0,units=theta_units,fig=fig,ax=ax[1],shp=shp)
    _,_ = plot_field_2d(u1[ss1],weights1[ss1]/np.sum(weights1[ss1]),theta1[ss1],avg_flag=avg_flag,cmap=plt.cm.coolwarm,fun0name=theta_names[0],fun1name=theta_names[1],fieldname=u_names[1],std_flag=False,logscale=0*logscale,ss=ss1,units=theta_units,fig=fig,ax=ax[2],shp=shp)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[2].tick_params(axis='both', which='major', labelsize=15)
    return fig,ax

def gamma_mom(moment_list):
    e1,e2 = moment_list[0],moment_list[1]
    # Given the mean and mean-square
    # Assume the moment list starts at 1
    alpha = e1**2/(e2 - e1**2)
    beta = alpha/e1
    print("e1 = {}, alpha = {}, beta = {}, alpha/beta = {}".format(e1,alpha,beta,alpha/beta))
    return alpha,beta

def gamma_mom_overdet(moment_list):
    # Assume the moment list starts at 1
    nmom = len(moment_list)
    ratios = np.zeros(nmom)
    ratios[0] = moment_list[0]
    ratios[1:] = moment_list[1:]/moment_list[:-1]
    print("ratios = {}".format(ratios))
    lm = linear_model.LinearRegression()
    lm.fit(np.arange(nmom).reshape((nmom,1)),ratios)
    beta = 1.0/lm.coef_[0]
    alpha = beta*lm.intercept_
    return alpha,beta