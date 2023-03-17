import numpy as np
import numba
import bisect
from .constants import *
import functools
import numba
from scipy.interpolate import interp1d, griddata, InterpolatedUnivariateSpline


def sample_from_powerlaw(alpha, xmin=0.1, xmax=1, nsample=int(1e4)):
    """
    Sample values from power law $x ~ x^alpha$
    Args:
    ----
        alpha: power-law index (float)
        xmin, xmax:  optional, minium and maximu values (float)
        nsample: number of samples (integer)
    Returns:
    -------
        random draws

    Examples:
    --------
        > x = sample_from_powerlaw(alpha, xmin=0.1, xmax=1, nsample=int(1e3))
    """
    x= np.linspace(xmin, xmax, int(1e6))
    pdf=x**alpha
    cdf=np.cumsum(pdf)
    cdf=cdf/np.nanmax(cdf)
    return random_draw(x, cdf, nsample=int(nsample))


@numba.njit
def random_draw(x_grid, cdf, nsample=10):
    """
    1D- Random draw by using numpy search on a grid
    Args:
    ----
        x_grid: grid of values ( array)
        cdf:  corresponding values from the CDF
        nsample: optional, number of samples
    Returns:
    -------
        random draws

    Examples:
    --------
        > x = np.arange(0, 10)
        > cdf = x**3/(x[-1]**3)
        > res= random_draw(x, cdf)
    """
    values = np.random.rand(nsample)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf


def make_spt_number(spt):
    """
    Returns a number from a spectral type
    Args:
    ----
        spt: spectral type (float or string)
    Returns:
    -------
        spectral type in float 

    Examples:
    --------
    """
    if isinstance(spt, str):
        if spt.lower().startswith('m'):
            return 10+float(spt[1:])
        if spt.lower().startswith('l'):
            return 20+float(spt[1:])
        if spt.lower().startswith('t'):
            return 30+float(spt[1:])
        if spt.lower().startswith('y'):
            return 40+float(spt[1:])
    else:
        return spt

@numba.vectorize("float64(float64, float64)", target='cpu')
def get_distance(absmag, appmag):
    """
    Distance from absolute magnitude and apparent magnitude
    Args:
    ----
        absmag: absolute magnitude (float or array)
        app: apparent magnitude (float or array)
    Returns:
    -------
        distance (float or array)

    Examples:
    --------

    """
    return 10.**(-(absmag-appmag)/5. + 1.)



@numba.jit(nopython=True)
#@numba.jit(nopython=True, parallel=True) --> fails due to numba issues
def trapzl(y, x):
    """
    Fast trapezoidal integration

    Args:
    ----
        y: y values on a grid  (list or array)
        x: x values on a grid  (list or array)
    Returns:
    -------
        distance (float or array)

    Examples:
    --------
        > x = np.arange(0, 10)
        > y = np.ones_like(x)
        > res= trapzl(y, x)

    """
    s = 0
    for i in numba.prange(1, len(x)):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2


def dropnans(x):
    """
    """
    return x[~np.isnan(x)]

def group_by(xvalues, yvalues, grid= np.arange(0, 1, 1000)):
    """
    Group values on a 2D-grid by computing the median and standard deviation
    Works like an interpolation
    Args:
    ----
        xvalues: grid of x-values ( array, size N)
        yvalues:  grid of corresponding y-values ( array, size N)
        grid: optional, group of new values to group (array, any size)

    Returns:
    -------
        dictionary of values and standard deviations

    Examples:
    --------
        > x = np.random.uniform(0, 10, 1000)
        > y = np.random.uniform(20, 100, 1000)
        > z = np.linspace(0, 10)
        > grouped =group_by(y, yvalues, grid= np.arange(0, 1, 1000))

    """
    res=np.ones_like(grid)*np.nan
    std=np.ones_like(grid)*np.nan
    for idx, g in enumerate(grid):
        if idx < len(grid)-1:
            bools=np.logical_and(xvalues>=grid[idx], xvalues<grid[idx+1])
        else:
            bools=xvalues>=grid[-1]
        np.place(res, grid==[grid[idx]], np.nanmedian(yvalues[bools]) )
        np.place(std, grid==[grid[idx]], np.nanstd(yvalues[bools]))
    return {'grid': grid, 'median': res, 'std': std}
    
def k_clip_fit(x, y, sigma_y, sigma = 5, n=6):
    """
    Fit a polymomial by sigma clipping
    Args:
    ----
        x: grid of x-values ( array, N)
        y:  grid of y-values (array, N)
        sigma_y: uncertainties in y (arry, N)
        sigma: optional, number of sigmas to clip
        n: optional, order of the polynomial
    Returns:
    -------
        returns a tuple of the list of retained y-values and polynomial object

    Examples:
    --------
        > x =  np.random.uniform(0, 10, 1000)
        > y = x**3
        > yerr= x = np.random.uniform(0, 1, 1000)
        > vals, pol= k_clip_fit(x, y, yerr, sigma = 5, n=6)

    """
    not_clipped = np.ones_like(y).astype(bool)
    n_remove = 1
    #use median sigma
    #median_sigma= np.nanmedian(sigma_y)
    while n_remove > 0:
        best_fit = np.poly1d(np.polyfit(x[not_clipped], y[not_clipped], n))
        
        norm_res = (np.abs(y - best_fit(x)))/(sigma_y)
        remove = np.logical_and(norm_res >= sigma, not_clipped == 1)
        n_remove = sum(remove)
        not_clipped[remove] = 0   
    return  not_clipped, best_fit

def apply_polynomial_relation_gpt(pol, x, xerr=0.0, nsample=100):
    x = np.array(x)
    size = 0

    if x.size == 1:
        size = -1

    x = np.random.normal(x, xerr, (int(nsample), len(np.atleast_1d(x))))
    
    results = [
        (coeffs * (x - xshift)[:, None] ** np.arange(len(coeffs))) 
        for k in pol.keys()
        for xshift, coeffs, scatter in [
            (pol[k]['xshift'], pol[k]['coeffs'], pol[k]['yerr'])
        ]
        for lowlim, uplim in [
            (float(k.split('_')[0]) - xshift, float(k.split('_')[-1]) - xshift)
        ]
        if np.logical_and(x - xshift >= lowlim, x - xshift <= uplim).any()
    ]

    ans = np.nansum(results, axis=1)
    res = np.nanmedian(
        [
            np.random.normal(a, scatter)
            for a, scatter in zip(ans, [pol[k]['yerr'] for k in pol.keys()])
        ],
        axis=0,
    )

    if size == -1:
        return np.nanmedian(res.flatten()), np.nanstd(res.flatten())

    return np.nanmedian(res, axis=0), np.nanstd(res, axis=0)

def inverse_polynomial_relation_gpt(pol, y, xgrid, nsample=1000, interpolation='griddata'):
    ygrid, yunc = apply_polynomial_relation(pol, xgrid, xerr=0.0, nsample=nsample)

    # Remove nans
    nans = np.logical_or(np.isnan(ygrid), np.isnan(yunc))

    # Reshape
    rand_y = np.random.normal(ygrid[~nans], yunc[~nans], size=(int(nsample), len(yunc[~nans]))).flatten()
    rand_x = np.broadcast_to(xgrid[~nans], (int(nsample), len(yunc[~nans]))).flatten()

    # Interpolation methods
    interpolation_methods = {
        'griddata': lambda y: griddata(rand_y, rand_x, y, fill_value=np.nan, method='linear', rescale=True),
        'spline': lambda y: (
            lambda f: f(y)
        )(
            InterpolatedUnivariateSpline(
                *zip(*sorted(zip(rand_y[~np.isnan(rand_y)], rand_x[~np.isnan(rand_y)]))),
                ext='const', k=1
            )
        )
    }

    return interpolation_methods[interpolation](y)


def apply_polynomial_relation(pol, x, xerr=0.0, nsample=100):
    """
    1D- Random draw by using numpy search on a grid
    Args:
    ----
        x_grid: grid of values ( array)
        cdf:  corresponding values from the CDF
        nsample: optional, number of samples
    Returns:
    -------
        random draws

    Examples:
    --------
        > x = np.arange(0, 10)
        > cdf = x**3/(x[-1]**3)
        > res= random_draw(x, cdf)

    """
    x=np.array(x)
    #handle cases where x is a float
    size=0
    if x.size==1:
        x=np.array([x, x]).astype(float)
        xerr=np.array([xerr, xerr]).astype(float)
        size=-1
    x=np.random.normal(x, xerr, (int(nsample), len(x)))
    #if xerr is None
    res= []
    unc= []
    #loop through each coefficient
    for k in pol.keys():
        #compute the shift to x values
        xshift= pol[k]['xshift']
        coeffs=pol[k]['coeffs']
        scatter=pol[k]['yerr']
        
        #shit x
        x_s=x-xshift
        
        #mask the low limit and high limit
        lowlim= float(k.split('_')[0])-xshift
        uplim= float(k.split('_')[-1])-xshift
    
        
        #compute masked arrays
        masked_x= np.ma.masked_outside(x_s, lowlim, uplim, copy=True)
      
        #compute polynomials
        ans= np.nansum([coeffs[i]*(masked_x**i) for i in range(len(coeffs))], axis=0)
        
        #update the result inside bounds
        masked_ans=np.ma.masked_array(ans, mask=masked_x.mask)
        
        #resample with uncertainties
        vals=np.random.normal(masked_ans.filled(fill_value=np.nan), scatter )
        
    
        
        res.append(vals)

    res=np.nanmean(res, axis=0)
    if size ==-1:
        return np.nanmedian(res.flatten()), np.nanstd(res.flatten())
    if size !=-1:
        return np.nanmean(res, axis=0), np.nanstd(res, axis=0)


def inverse_polynomial_relation(pol, y, xgrid, nsample=1000, interpolation='griddata'):
    """
    1D- Random draw by using numpy search on a grid
    Args:
    ----
        x_grid: grid of values ( array)
        cdf:  corresponding values from the CDF
        nsample: optional, number of samples
    Returns:
    -------
        random draws

    Examples:
    --------
        > x = np.arange(0, 10)
        > cdf = x**3/(x[-1]**3)
        > res= random_draw(x, cdf)

    """
    ygrid, yunc= apply_polynomial_relation(pol, xgrid, xerr=0.0, nsample=nsample)

    #remove nans
    nans= np.logical_or(np.isnan(ygrid), np.isnan(yunc))
    
    #reshape
    rand_y= np.random.normal(ygrid[~nans], yunc[~nans], size=(int(nsample), len(yunc[~nans]))).flatten()
    rand_x= np.random.normal(xgrid[~nans], np.zeros_like(xgrid[~nans]), size=(int(nsample), len(yunc[~nans]))).flatten()
    
    #f=interp1d(rand_y, rand_x, assume_sorted = False, fill_value = np.nan, bounds_error=False)

    if interpolation=='griddata':
        return  griddata(rand_y, rand_x, y, fill_value=np.nan, method='linear', rescale=True)
    if interpolation=='spline':
        #1 degree spline, will extrapolate to zeros 
        rand_y= rand_y[~np.isnan(rand_y)]
        rand_x= rand_x[~np.isnan(rand_y)]
        mask=np.argsort(rand_y)
        f=InterpolatedUnivariateSpline(rand_y[mask], rand_x[mask], ext='const', k=1)
        return f(y)