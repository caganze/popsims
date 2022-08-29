import numpy as np
import numba
import bisect
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
import astropy.coordinates as astro_coord
import functools
from scipy.interpolate import interp1d, griddata, InterpolatedUnivariateSpline

Rsun=8300.
Zsun=27.

#default coordinate frame
#sharma coordinate frame https://www.galah-survey.org/dr3/the_catalogues/#ages-masses-distances-and-other-parameters-estimated-by-bstep
v_sun = astro_coord.CartesianDifferential([11.1, 248., 7.25]*u.km/u.s) #almost the same as my coordinate 
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)
 
def sample_from_powerlaw(alpha, xmin=0.1, xmax=1, nsample=int(1e4)):
    """
    """
    x= np.linspace(xmin, xmax, int(1e6))
    pdf=x**alpha
    cdf=np.cumsum(pdf)
    return random_draw(x, cdf, nsample=int(nsample))

def random_draw(x_grid, cdf, nsample=10):
    """
    """
    cdf = cdf / cdf[-1]
    values = np.random.rand(nsample)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf

def make_spt_number(spt):
    """
    """
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

def dropnans(x):
    """
    """
    return x[~np.isnan(x)]

def group_by(xvalues, yvalues, grid= np.arange(0, 1, 1000)):
    """
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
        
    return  not_clipped
def random_draw_old(xvals, cdfvals, nsample=10):
    """
    """
    @numba.vectorize("int32(float64)")
    def invert_cdf(i):
        return bisect.bisect(cdfvals, i)-1
    x=np.random.rand(nsample)
    idx=invert_cdf(x)
    return np.array(xvals)[idx]

def apply_polynomial_relation(pol, x, xerr=0.0, nsample=100):
    
    """
    
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


def plot_annotated_heatmap(ax, data, gridpoints, columns, cmap='viridis', 
                           annotate=False, vmin=0.0, vmax=1.0, textsize=14, alpha=0.1):
    #plot an annotated heatmap
    data= data.dropna()
    xcol, ycol, zcol= columns
    step1= np.ptp(data[xcol])/gridpoints
    step2= np.ptp(data[ycol])/gridpoints
    
    #print (step1)
    
    xgrid= np.linspace(data[xcol].min(), data[xcol].max(), gridpoints)
    ygrid= np.linspace(data[ycol].min(), data[ycol].max(), gridpoints)
    
    
    mask = np.zeros((len(xgrid), len(ygrid)))
    values = np.zeros((len(xgrid), len(ygrid)))
    #for annotation
    for i in range(len(xgrid)):
        #loop over matrix
        for j in range(len(ygrid)):
            if (i == len(xgrid)-1) | (j == len(ygrid)-1) :
                pass
            else:
                maskx= np.logical_and(data[xcol] > xgrid[i], data[xcol] <= xgrid[i]+step1)
                masky=np.logical_and(data[ycol] > ygrid[j], data[ycol] <=ygrid[j]+step2)
                zmedian= np.nanmean(data[zcol][np.logical_and(maskx, masky)])
                lenz= len(data[np.logical_and.reduce([maskx, masky])])

                if lenz == 0:
                    values[j][i] = np.nan
                    mask[j][i] = 1
                else:
                    values[j][i] = zmedian
                    if annotate == 'third_value':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{zmedian:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                    if annotate== 'number':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{lenz:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                
    values2 = np.ma.array(values, mask=mask)
    cax = ax.pcolormesh(xgrid, ygrid, values2, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
    #plt.axis('tight')
    ymin, ymax = plt.ylim()

    ax.minorticks_on()

    ax.set_ylim(ymax, ymin)
    return 


def alias(aliases):
    """
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            for name, alias in aliases.items():
                if name not in kwargs and alias in kwargs:
                    kwargs[name] = kwargs[alias]
            return func(**kwargs)
        return wrapper
    return decorator