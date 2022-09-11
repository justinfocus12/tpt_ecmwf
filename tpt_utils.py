# All new set of functions to take statistics of multi-dimensional data
import numpy as np
import matplotlib
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2
smallfont = {'family': 'serif', 'size': 12}
font = {'family': 'serif', 'size': 18}
bigfont = {'family': 'serif', 'size': 40}
giantfont = {'family': 'serif', 'size': 80}
ggiantfont = {'family': 'serif', 'size': 120}
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

def project_field(field, weights, features, cov_mat_flag=False, shp=None, bounds=None):
    """
    Parameters
    ----------
    field: numpy.array, shape (Nx,Nf)
        Function values to project. This does multiple functions at a time to save computation
    weights: numpy.array, shape (Nx,Nf)
        Weights of the function values 
    features: numpy.array, shape (Nx, d)
        Coordinates in d-dimensional space 
    shp: numpy.array of int, shape (d,)
        Number of bins in each dimension
    
    Returns 
    -------
    field_projected: numpy.ndarray, shape (Nf,shp)
        Weighted average (if avg == True) or weighted sum (if avg == False) of the field in each bin
    """
    # First, get rid of all nans in features
    Nx,dim = features.shape
    Nf = field.shape[1]
    if not (Nx == field.shape[0] == weights.shape[0] and field.shape[1] == weights.shape[1]):
        raise Exception("Inconsistent shape inputs to project_field. shapes: field {field.shape}, weights {weights.shape}, features {features.shape}")
    if np.any(weights < 0):
        raise Exception("Some weights are negative.")
    if shp is None: shp = 20*np.ones(dim,dtype=int) # number of INTERIOR
    if bounds is None:
        bounds = np.array([np.nanmin(features,0),np.nanmax(features,0)])
        if np.any(bounds[1] <= bounds[0]):
            raise Exception("The bounding box is zero along some dimension: bounds=\n{bounds}")
        padding = 0.01*(bounds[1] - bounds[0])
        bounds[0] -= padding
        bounds[1] += padding
    # Determine grid box size
    dx = (bounds[1] - bounds[0])/shp
    # Take only the indices for which the following three conditions hold
    goodidx = np.where(
            np.all(np.isnan(features)==0, axis=1) *  # All features are defined
            np.all(features >= bounds[0], axis=1) *  # The features are above the lower boundary
            np.all(features < bounds[1], axis=1)     # The features are below the upper boundary
            )[0]
    # Determine which grid box each data point falls into
    grid_cell = ((features[goodidx] - bounds[0])/dx).astype(int)
    grid_cell_flat = np.ravel_multi_index(tuple(grid_cell.T), shp)
    # Loop through the grid cells and average all the data with the corresponding index
    Ngrid = np.prod(shp)
    field_proj_stats = dict({key: np.zeros((Ngrid,Nf)) for key in ["weightsum","sum","mean","std","q25","q75","min","max"]})
    if cov_mat_flag:
        cov_mat = np.zeros((Ngrid, Nf, Nf))
    for i_flat in range(Ngrid):
        idx, = np.where(grid_cell_flat == i_flat)
        weights_idx = weights[goodidx[idx],:]
        field_idx = field[goodidx[idx],:]
        field_proj_stats["weightsum"][i_flat,:] = np.nansum(weights_idx,axis=0)
        field_proj_stats["sum"][i_flat,:] = np.nansum(field_idx*weights_idx,axis=0)
        if cov_mat_flag:
            cov_mat[i_flat] = np.ma.getdata(np.ma.cov(
                    np.ma.masked_array(field_idx, mask=np.isnan(field_idx)), rowvar=False
                    ))
        good_fun_idx = np.where((field_proj_stats["weightsum"][i_flat,:] != 0)*(np.all(np.isnan(field_idx),axis=0)==0))[0]
        bad_fun_idx = np.setdiff1d(np.arange(Nf),good_fun_idx)
        for key in ["mean","std","q25","q75","min","max"]:
            field_proj_stats[key][i_flat,bad_fun_idx] = np.nan
        field_proj_stats["mean"][i_flat,good_fun_idx] = field_proj_stats["sum"][i_flat,good_fun_idx]/field_proj_stats["weightsum"][i_flat,good_fun_idx]
        field_proj_stats["std"][i_flat,good_fun_idx] = np.sqrt(np.nansum((field_idx[:,good_fun_idx] - field_proj_stats["mean"][i_flat,good_fun_idx])**2 * weights_idx[:,good_fun_idx], axis=0) / field_proj_stats["weightsum"][i_flat,good_fun_idx])
        if len(field_idx) > 0 and len(good_fun_idx) > 0:
            if np.any(np.all(np.isnan(weights_idx[:,good_fun_idx]), axis=0)):
                raise Exception("The good fun idx are not all good")
            field_proj_stats["min"][i_flat,good_fun_idx] = np.nanmin(field_idx[:,good_fun_idx],axis=0)
            field_proj_stats["max"][i_flat,good_fun_idx] = np.nanmax(field_idx[:,good_fun_idx],axis=0)
            # quantiles
            order = np.argsort(field_idx[:,good_fun_idx],axis=0)
            for i_fun in good_fun_idx:
                cdf = np.nancumsum(weights_idx[order[:,i_fun],i_fun])
                cdf *= 1.0/cdf[-1]
                if np.any(cdf >= 0.25):
                    field_proj_stats["q25"][i_flat,i_fun] = field_idx[order[np.where(cdf >= 0.25)[0][0],i_fun],i_fun]
                else:
                    raise Exception(f"cdf has nothing greater than 0.25: cdf = {cdf}")
                field_proj_stats["q75"][i_flat,i_fun] = field_idx[order[np.where(cdf >= 0.75)[0][0],i_fun],i_fun]
    for key in ["weightsum","sum","mean","std","q25","q75","min","max"]:
        field_proj_stats[key] = field_proj_stats[key].reshape(np.concatenate((shp,[Nf])))
    if cov_mat_flag:
        field_proj_stats["cov_mat"] = cov_mat.reshape(np.concatenate((shp,[Nf,Nf])))
    # Make a nice formatted grid, too
    edges = tuple([np.linspace(bounds[0,i],bounds[1,i],shp[i]+1) for i in range(dim)])
    centers = tuple([(edges[i][:-1] + edges[i][1:])/2 for i in range(dim)])
    return field_proj_stats, edges, centers

def plot_field_1d(
        field, weights, feature, 
        density_flag = False,
        nbins=None, bounds=None, orientation="horizontal", 
        fig=None, ax=None, 
        field_name=None, feat_name=None,
        quantile_flag=True, minmax_flag=True,
        ):
    # TODO: draw standard deviation (and more) envelopes around the 1D plot, to illustrate some of the uncertainty.
    if not (
            field.ndim == weights.ndim == feature.ndim == 1 and 
            field.size == weights.size == feature.size
            ):
        raise Exception(f"field, weights, and feature all must be 1D. But their shapes are {field.shape}, {weights.shape}, and {feature.shape}")
    if bounds is None:
        bounds = np.array([np.nanmin(feature), np.nanmax(feature)])
        padding = 0.01*(bounds[1] - bounds[0])
        bounds[0] -= padding
        bounds[1] += padding
    bounds_proj = bounds.reshape((2,1))
    features = feature.reshape((feature.size,1))
    if nbins is None:
        nbins = 20
    shp = np.array((nbins,))
    field_proj,edges,centers = project_field(field.reshape(-1,1), weights.reshape(-1,1), features, shp=shp, bounds=bounds_proj)
    # pull out the field of interest
    if density_flag:
        field2plot = field_proj["weightsum"][:,0] 
        field2plot *= 1.0/ np.sum(field2plot * (edges[0][1:] - edges[0][:-1]))
    else:
        field2plot = field_proj["mean"][:,0]
    # Now plot it
    if fig is None or ax is None:
        fig,ax = plt.subplots()
    if orientation == "horizontal":
        ax.plot(centers[0],field2plot, marker='.',color='black')
        if quantile_flag:
            ax.fill_between(centers[0],field_proj["q25"][:,0],field_proj["q75"][:,0],color=plt.cm.binary(0.5),zorder=-1)
        if minmax_flag:
            ax.fill_between(centers[0],field_proj["min"][:,0],field_proj["max"][:,0],color=plt.cm.binary(0.25),zorder=-2)
            #ax.fill_between(centers[0],field_proj["min"][:,0],field_proj["max"][:,0],color=plt.cm.binary(0.3),zorder=-2)
        if feat_name is not None:
            ax.set_xlabel(feat_name)
        if field_name is not None:
            ax.set_ylabel(field_name)
    else:
        ax.plot(field2plot,centers[0],marker='.',color='black')
        if quantile_flag:
            ax.fill_betweenx(centers[0],field_proj["q25"][:,0],field_proj["q75"][:,0],color=plt.cm.binary(0.5),zorder=-1)
        if minmax_flag:
            ax.fill_betweenx(centers[0],field_proj["min"][:,0],field_proj["max"][:,0],color=plt.cm.binary(0.25),zorder=-2)
        if feat_name is not None:
            ax.set_ylabel(feat_name)
        if field_name is not None:
            ax.set_xlable(field_name)
    return fig,ax
        
def plot_field_2d(
        field, weights, features,
        shp=None, bounds=None, 
        fig=None, ax=None,
        field_name=None, feat_names=None,
        stat_name="mean",
        logscale=False, cmap=None, 
        pcolor_flag=True, contour_flag=False,
        vmin=None, vmax=None
        ):
    if not (
            (field.ndim == weights.ndim == 1) and 
            (features.ndim == 2) and 
            (field.shape[0] == weights.shape[0] == features.shape[0])
            ):
        raise Exception(f"Inconsistent shapes. field: ({field.shape}), features: ({features.shape}), weights: ({weights.shape})")
    if (fig is None or ax is None):
        fig,ax = plt.subplots()
    if shp is None: 
        shp = np.array([20,20])
    if cmap is None:
        cmap = plt.cm.coolwarm
    field_proj,edges,centers = project_field(field.reshape(-1,1), weights.reshape(-1,1), features, shp=shp, bounds=bounds)
    # Plot in 2d
    xy,yx = np.meshgrid(edges[0], edges[1], indexing='ij')
    if logscale:
        eps = 1e-15
        field_proj[stat_name][np.where(field_proj[stat_name] < eps)] = np.nan
        if vmin is None: vmin = eps
        vmin = max(np.nanmin(field_proj[stat_name]), vmin)
        print(f"vmin = {vmin}")
        if vmax is None: vmax = np.nanmax(field_proj[stat_name])
        im = ax.pcolormesh(
                xy,yx,field_proj[stat_name][:,:,0],
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap
                )
        fig.colorbar(im, ax=ax)
    else:
        im = ax.pcolormesh(xy,yx,field_proj[stat_name][:,:,0],cmap=cmap,vmin=vmin,vmax=vmax)
        fig.colorbar(im, ax=ax)
    if feat_names is not None:
        ax.set_xlabel(feat_names[0])
        ax.set_ylabel(feat_names[1])
    if field_name is not None:
        ax.set_title(field_name)
    return fig,ax,centers



