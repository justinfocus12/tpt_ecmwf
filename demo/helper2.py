# All new set of functions to take statistics of multi-dimensional data
import numpy as np
import matplotlib
import matplotlib.colors as colors
matplotlib.rcParams['font.size'] = 18
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
    field_proj_stats = dict({key: np.zeros(Ngrid) for key in ["weightsum","sum","mean","std"]})
    for i_flat in range(Ngrid):
        idx, = np.where(grid_cell_flat == i_flat)
        weights_idx = weights[goodidx[idx]]
        field_idx = field[goodidx[idx]]
        field_proj_stats["weightsum"][i_flat] = np.nansum(weights_idx)
        field_proj_stats["sum"][i_flat] = np.nansum(field_idx*weights_idx)
        if field_proj_stats["weightsum"][i_flat] == 0:
            field_proj_stats["mean"][i_flat] = np.nan
            field_proj_stats["std"][i_flat] = np.nan
        else:
            field_proj_stats["mean"][i_flat] = field_proj_stats["sum"][i_flat]/field_proj_stats["weightsum"][i_flat]
            field_proj_stats["std"][i_flat] = np.sqrt(np.nansum((field_idx - field_proj_stats["mean"][i_flat])**2 * weights_idx) / field_proj_stats["weightsum"][i_flat])
    for key in ["weightsum","sum","mean","std"]:
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
        ):
    # TODO: draw standard deviation (and more) envelopes around the 1D plot, to illustrate some of the uncertainty.
    if not (
            field.ndim == weights.ndim == feature.ndim == 1 and 
            field.size == weights.size == feature.size
            ):
        raise Exception("field, weights, and feature all must be 1D. But their shapes are {field.shape}, {weights.shape}, and {feature.shape}")
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
        ax.plot(centers[0],field_proj["mean"].flatten(),marker='.',color='black')
        if feat_name is not None:
            ax.set_xlabel(feat_name)
        if field_name is not None:
            ax.set_ylabel(field_name)
    else:
        ax.plot(field_proj["mean"].flatten(),centers[0],marker='.',color='black')
        if feat_name is not None:
            ax.set_ylabel(feat_name)
        if field_name is not None:
            ax.set_xlable(field_name)
    return fig,ax
        
