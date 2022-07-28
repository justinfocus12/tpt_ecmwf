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

def project_field(field, weights, features, shp=None, bounds=None):
    """
    Parameters
    ----------
    field: numpy.array, shape (Nx,)
        Function values to project
    weights: numpy.array, shape (Nx,)
        Weights of the function values 
    features: numpy.array, shape (Nx, d)
        Coordinates in d-dimensional space 
    shp: numpy.array of int, shape (d,)
        Number of bins in each dimension
    
    Returns 
    -------
    field_projected: numpy.ndarray, shape (shp)
        Weighted average (if avg == True) or weighted sum (if avg == False) of the field in each bin
    """
    # First, get rid of all nans in features
    Nx,dim = features.shape
    if not (Nx == field.size == weights.size):
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
    # Determine which grid box each data point falls into
    goodidx = np.where(np.all(np.isnan(features)==0, axis=1))[0]
    grid_cell = ((features[goodidx] - bounds[0])/dx).astype(int)
    print(f"grid_cell: \nmin = \n{np.min(grid_cell,axis=0)}\nmax={np.max(grid_cell,axis=0)}")
    grid_cell_flat = np.ravel_multi_index(tuple(grid_cell.T), shp)
    # TODO: fix this bug
    # Loop through the grid cells and average all the data with the corresponding index
    Ngrid = np.prod(shp)
    field_proj_stats = dict({key: np.zeros(Ngrid) for key in ["weightsum","sum","mean","std","q25","q75","min","max"]})
    for i_flat in range(Ngrid):
        idx, = np.where(grid_cell_flat == i_flat)
        weights_idx = weights[goodidx[idx]]
        field_idx = field[goodidx[idx]]
        field_proj_stats["weightsum"][i_flat] = np.nansum(weights_idx)
        field_proj_stats["sum"][i_flat] = np.nansum(field_idx*weights_idx)
        if field_proj_stats["weightsum"][i_flat] == 0 or np.all(np.isnan(field_idx)):
            for key in ["mean","std","q25","q75","min","max"]:
                field_proj_stats[key][i_flat] = np.nan
        else:
            field_proj_stats["mean"][i_flat] = field_proj_stats["sum"][i_flat]/field_proj_stats["weightsum"][i_flat]
            field_proj_stats["std"][i_flat] = np.sqrt(np.nansum((field_idx - field_proj_stats["mean"][i_flat])**2 * weights_idx) / field_proj_stats["weightsum"][i_flat])
            field_proj_stats["min"][i_flat] = np.nanmin(field_idx)
            field_proj_stats["max"][i_flat] = np.nanmax(field_idx)
            # quantiles
            order = np.argsort(field_idx)
            cdf = np.cumsum(weights_idx[order])
            cdf *= 1.0/cdf[-1]
            field_proj_stats["q25"][i_flat] = field_idx[order[np.where(cdf >= 0.25)[0][0]]]
            field_proj_stats["q75"][i_flat] = field_idx[order[np.where(cdf >= 0.75)[0][0]]]
    for key in ["weightsum","sum","mean","std","q25","q75","min","max"]:
        field_proj_stats[key] = field_proj_stats[key].reshape(shp)
    # Make a nice formatted grid, too
    edges = tuple([np.linspace(bounds[0,i],bounds[1,i],shp[i]+1) for i in range(dim)])
    centers = tuple([(edges[i][:-1] + edges[i][1:])/2 for i in range(dim)])
    return field_proj_stats, edges, centers

def plot_field_1d(
        field, weights, feature, 
        nbins=None, bounds=None, orientation="horizontal", 
        fig=None, ax=None, 
        field_name=None, feat_name=None,
        quantile_flag=True,
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
    print(f"shapes: field = {field.shape}, weights = {weights.shape}, features = {features.shape}")
    field_proj,edges,centers = project_field(field, weights, features, shp=shp, bounds=bounds_proj)
    # Now plot it
    if fig is None or ax is None:
        fig,ax = plt.subplots()
    if orientation == "horizontal":
        ax.plot(centers[0],field_proj["mean"],marker='.',color='black')
        if quantile_flag:
            ax.fill_between(centers[0],field_proj["q25"],field_proj["q75"],color=plt.cm.binary(0.6),zorder=-1)
            ax.fill_between(centers[0],field_proj["min"],field_proj["max"],color=plt.cm.binary(0.3),zorder=-2)
        if feat_name is not None:
            ax.set_xlabel(feat_name)
        if field_name is not None:
            ax.set_ylabel(field_name)
    else:
        ax.plot(field_proj["mean"].flatten(),centers[0],marker='.',color='black')
        if quantile_flag:
            ax.fill_betweenx(centers[0],field_proj["q25"],field_proj["q75"],color=plt.cm.binary(0.6),zorder=-1)
            ax.fill_betweenx(centers[0],field_proj["min"],field_proj["max"],color=plt.cm.binary(0.3),zorder=-2)
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
        ):
    if not (
            (field.ndim == weights.ndim == 1) and 
            (features.ndim == 2) and 
            (field.size == weights.size == features.shape[0])
            ):
        raise Exception(f"Inconsistent shapes. field: ({field.shape}), features: ({features.shape}), weights: ({weights.shape})")
    if (fig is None or ax is None):
        fig,ax = plt.subplots()
    if shp is None: 
        shp = np.array([20,20])
    field_proj,edges,centers = project_field(field, weights, features, shp=shp, bounds=bounds)
    print(f"field_proj shape = {field_proj['mean'].shape}")
    # Plot in 2d
    xy,yx = np.meshgrid(edges[0], edges[1], indexing='ij')
    im = ax.pcolormesh(xy,yx,field_proj['mean'],cmap=plt.cm.coolwarm)
    if feat_names is not None:
        ax.set_xlabel(feat_names[0])
        ax.set_ylabel(feat_names[1])
    if field_name is not None:
        ax.set_title(field_name)
    return fig,ax



