
################################
#includes some useful relations, for full package, see splat
##############################

from scipy.interpolate import interp1d
import numpy as np
#import splat.empirical as spe

from .core_tools import apply_polynomial_relation, inverse_polynomial_relation


PECAUT_TEFF_SPT_RELATIONS={'pecaut': {'bibcode': '2013ApJS..208....9P', 'url': 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt', \
'method': 'interpolate', 'range': [0.0, 29.0], 'fitunc': 108.0, 
'spt': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, \
16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 34.5, 35.0, 35.5, 36.0, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0], \
'values': [5280.0, 5240.0, 5170.0, 5140.0, 5040.0, 4990.0, 4830.0, 4700.0, 4600.0, 4540.0, 4410.0, 4330.0, 4230.0, 4190.0, 4070.0, 4000.0, 3940.0, 3870.0, 3800.0, 3700.0, 3650.0, 3550.0, 3500.0, 3410.0, 3250.0, 3200.0, 3100.0, \
3030.0, 3000.0, 2850.0, 2710.0, 2650.0, 2600.0, 2500.0, 2440.0, 2400.0, 2320.0, 2250.0, 2100.0, 1960.0, 1830.0, 1700.0, 1590.0, 1490.0, 1410.0, 1350.0, 1300.0, 1260.0, 1230.0, 1200.0, 1160.0, 1120.0, 1090.0, 1050.0, 1010.0, 960.0, \
840.0, 770.0, 700.0, 610.0, 530.0, 475.0, 420.0, 390.0, 350.0, 325.0, 250.0], \
'rms': [108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,
       108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,
       108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,
       108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,
       108., 108., 108., 108., 108., 108., 108., 108., 108., 108., 108.,
       108.]}}

LITERATURE_LOOKUP_TABLES={}

LITERATURE_POLYNOMIALS={'kirkpatrick2021':{'x=spt,y=teff': {'20_28.75': {'coeffs': [2.2375e+03, -1.4496e+02, 4.0301e+00],'xshift':20, 'yerr':134}, '28.75_34.75':{ 'coeffs': [1.4379e+03, -1.8309e+01], 'xshift':20, 'yerr':79}, '34.75_42':{ 'coeffs': [5.1413e+03,-3.6865e+02, 6.7301e+00],'xshift':20, 'yerr':79}},
                                          'x=j_2mass,y=j_mko': {'10_20': {'coeffs': [7.0584e-01, 9.3542e-01],'xshift': 0., 'yerr':0.17}},
                                          'x=h_2mass,y=teff':{'9.5_25.': {'coeffs': [1.2516e+04,-1.5666e+03,6.7502e+01, -9.2430e-01,-1.9530e-03],'xshift': 0.0,'yerr':88.1 }},
                                          'x=spt,y=j_mko':{'20_42': {'coeffs': [1.1808e+01, 3.3790e-01, -1.9013e-01, 7.1759e-02,-9.9829e-03,6.3147e-04, -1.8672e-05, 2.1526e-07],'xshift': 20.0,'yerr': 0.6 }},
                                          'x=spt,y=h_2mass':{'20_42': {'coeffs': [1.1808e+01, 3.3790e-01, -1.9013e-01, 7.1759e-02, -9.9829e-03, 6.3147e-04,-1.8672e-05, 2.1526e-07],'xshift': 20.0, 'yerr': 0.57}},
                                          'x=j_mko,y=spt': {'14.2_24': {'coeffs': [-7.7784e+01, 1.3260e+01, -6.1185e-01,9.6221e-03], 'xshift':0.0, 'yerr':0.53}},
                                          'x=h_2mass,y=spt':{'14.5_24.0': {'coeffs': [-6.9184e+01, 1.1863e+01,-5.4084e-01,8.4661e-03],'xshift':0.0,'yerr':0.51}}},
                                          
                       'dupuy2012': {'x=spt,y=y_mko': {'16_39': {'coeffs': np.flip([-.00000252638, .000285027, -.0126151, .279438, -3.26895, 19.5444, -35.1560]),'xshift':10.0,'yerr':0.4}},
                                    'x=spt,y=j_mko': {'16_39': {'coeffs': np.flip([-.00000194920, .000227641, -.0103332, .232771, -2.74405, 16.3986, -28.3129]),'xshift':10.0,'yerr':0.4}},
                                    'x=spt,y=h_mko': {'16_39': {'coeffs': np.flip([-.00000224083, .000251601, -.0110960, .245209, -2.85705, 16.9138, -29.7306]),'xshift':10.0,'yerr':0.4}},
                                    'x=spt,y=ks_mko': {'16_39': {'coeffs':np.flip([-.00000104935, .000125731, -.00584342, .135177, -1.63930, 10.1248, -15.2200]),'xshift':10.0,'yerr':0.38}},
                                    'x=spt,y=j_2mass': {'16_39': {'coeffs': np.flip([-.000000784614, .000100820, -.00482973, .111715, -1.33053, 8.16362, -9.67994]),'xshift':10.0,'yerr':0.4}},
                                    'x=spt,y=h_2mass': {'16_39': {'coeffs': np.flip([-.00000111499, .000129363, -.00580847, .129202, -1.50370, 9.00279, -11.7526]),'xshift':10.0,'yerr':0.4}},
                                    'x=spt,y=ks_2mass': {'16_39': {'coeffs': np.flip([1.06693e-4, -6.42118e-3, 1.34163e-1, -8.67471e-1, 1.10114e1]),'xshift':10.0,'yerr':0.43}}},

                       'freeser2022': {'x=spt,y=r_ps1': {'20_26': {'coeffs': [1.911e1, -3.775e-1, 3.201e-2],'xshift':10.0,'yerr':0.32}},
                                    'x=spt,y=i_ps1': {'20_32': {'coeffs': [5.176e1, -9.916, 9.562e-1, -3.728e-2, 5.207e-4] ,'xshift':10.0,'yerr':0.36}},
                                    'x=spt,y=z_ps1': {'20_38': {'coeffs': [5.642e1, -1.289e1, 1.452, -7.393e-2, 1.739e-3, -1.503e-5 ],'xshift':10.0,'yerr':0.34}},
                                    'x=spt,y=y_ps1': {'20_38': {'coeffs':[6.358e1, -1.523e1, 1.707, -8.697e-2, 2.048e-3, -1.776e-5 ],'xshift':10.0,'yerr':0.36}},
                                    'x=spt,y=y_mko': {'20_39': {'coeffs': [3.314e1, -4.581, 1.611, 2.863e-2, -2.646e-3, 8.018e-5, -8.210e-7],'xshift':10.0,'yerr':0.36}},
                                    'x=spt,y=j_mko': {'20_39': {'coeffs': [3.197e-2, 7.465, -1.630, 1.638e-1, -8.147e-3, 1.951e-4, -1.787e-6],'xshift':10.0,'yerr':0.36}},
                                    'x=spt,y=h_mko': {'20_39': {'coeffs': [3.756e1,-7.055e0, 5.951e-1, -1.096e-2, -7.656e-4, 3.653e-5,-4.289e-7], 'xshift':10.0,'yerr':0.46}},
                                    'x=spt,y=k_mko': {'20_39': {'coeffs': [8.884e1,-2.728e0, 3.784e0, -2.701e-1,1.065e-2, -2.216e-4, 1.915-6], 'xshift':10.0,'yerr':0.42}},
                                    'x=spt,y=j_2mass': {'20_39': {'coeffs': [-6.495e1,3.158e1, -5.253e0, 4.463e-1, -2.020e-2, 4.629e-4, -4.208-6], 'xshift':10.0,'yerr':0.34}},
                                    'x=spt,y=h_2mass':{'20_39': {'coeffs': [-8.744e0,1.042e1, -2.701e-1, 1.065e-2, -9.965e-3,2.439e-4, -2.239-6], 'xshift':10.0,'yerr':0.26}},
                                    'x=spt,y=k_2mass':{'20_39': {'coeffs': [2.734e1, -4.828e0, 4.802e-1, -1.931e-2, 2.801e-4], 'xshift':10.0,'yerr':0.28}}},

                        'zhang2013': {'x=spt,y=r_sdss': {'11_28':{'coeffs': np.flip([-1.6751e-4,8.4503e-3,-1.5169e-1,1.1111,-1.8452,9.8326]), 'xshift':10, 'yerr':0.7}},
                                    'x=spt,y=i_sdss': {'11_28':{'coeffs': np.flip([-1.0634e-4,5.3482e-3,-9.6139e-2,7.1114e-1,-1.0400,8.6772]), 'xshift':10, 'yerr':0.7}},
                                    'x=spt,y=z_sdss': {'11_28':{'coeffs': np.flip([-6.2600e-5,2.9652e-3,-4.9888e-2,3.3379e-1,-4.7257e-2,7.8303]), 'xshift':10, 'yerr':0.6}}},

                        'zhang2017': {'x=spt,y=j_mko': {'10_27':{'coeffs': np.flip([8.53625e-4,-1.76459e-2,3.17384e-1,8.64788]), 'xshift':10, 'yerr':0.4}},
                                    'x=spt,y=h_mko': {'10_27':{'coeffs': np.flip([2.90020e-4,-4.54248e-3,2.71013e-1,8.19731]), 'xshift':10, 'yerr':0.4}},
                                    'x=spt,y=j_2mass': {'10_27':{'coeffs': np.flip([8.48172e-4,-1.75984e-2,3.16187e-1,8.68342]), 'xshift':10, 'yerr':0.4}},
                                    'x=spt,y=h_2mass': {'10_27':{'coeffs': np.flip([4.32261e-4,-7.53663e-3,2.81607e-1,8.18494]), 'xshift':10, 'yerr':0.4}}},

                        'gonzales2018':{'x=spt,y=teff': {'10_27': {'coeffs': np.flip([-117, 3721.]), 'xshift':10, 'yerr': 108}}}}


                                                   

kirkpatrick2020LF={'bin_center': np.array([ 525,  675,  825,  975, 1125, 1275, 1425, 1575, 1725, 1875, 2025]),
    'values': np.array([4.24, 2.8 , 1.99, 1.72, 1.11, 1.95, 0.94, 0.81, 0.78, 0.5 , 0.72]),
    'unc': np.array([0.7 , 0.37, 0.32, 0.3 , 0.25, 0.3 , 0.22, 0.2 , 0.2 , 0.17, 0.18])}


def teff_to_spt_pecaut(teff):
    """Convert effective temperature to spectral type using Pecaot et al. (2013) relation.
    Args:
        teff (float): Effective temperature.

    Returns:
        float: Spectral type.

    """
    rel=PECAUT_TEFF_SPT_RELATIONS['pecaut']
    teffsc=np.random.normal(teff, 108)
    fx=interp1d(rel['values'], rel['spt'], assume_sorted = False, fill_value = np.nan, bounds_error=False)
    return fx(teffsc)
  
def spt_to_teff_pecaut(spt):
    """
    Convert spectral type to effective temperature using Pecaot et al. (2013) relation.

    Copy code
    Args:
        spt (float): Spectral type.

    Returns:
        float: Effective temperature.

    """
    rel=PECAUT_TEFF_SPT_RELATIONS['pecaut']
    fx=interp1d(rel['spt'], rel['values'], assume_sorted = False, fill_value = np.nan,  bounds_error=False)
    return np.random.normal(fx(spt), 108)


def scale_to_local_lf(teffs):
    """
    This function takes in an array of Teff values and scales it to the local luminosity function (LF) as determined by Kirkpatrick et al. (2020).

    Copy code
    Parameters
    ----------
    teffs: array-like
        Array of Teff values to be scaled to the local LF.

    Returns
    -------
    res: list
        A list containing the median scale, standard deviation of the scale, and the sum of the predicted values multiplied by the median scale.
    """
    binedges= np.append(kirkpatrick2020LF['bin_center']-75, kirkpatrick2020LF['bin_center'][-1]+75)
    preds=np.histogram(teffs, bins=binedges, normed=False)[0]
    
    obs=np.array(kirkpatrick2020LF['values'])
    unc=np.array(kirkpatrick2020LF['unc'])
    
    obs_monte_carlo= np.random.normal(obs, unc, (10000, len(obs)))
    pred_monte= np.ones_like(obs_monte_carlo)*(preds)
    unc_monte=  np.ones_like(obs_monte_carlo)*(unc)

    scale=(np.nansum((obs_monte_carlo*pred_monte)/(unc_monte**2), axis=1)\
           /np.nansum(((pred_monte**2)/(unc_monte**2)), axis=1))*(10**-3)
    
    
    res=[np.nanmedian(scale), np.nanstd(scale), \
                                     np.sum(preds*np.nanmedian(scale))]

    return res

def  spt_to_teff_kirkpatrick(spt):
    return apply_polynomial_relation(LITERATURE_POLYNOMIALS['kirkpatrick2021']['x=spt,y=teff'], spt)
 

def teff_to_spt_kirkpatrick(teff, nsample=1000):
    sptgrid=np.linspace(10, 43, 1000)
    return inverse_polynomial_relation(LITERATURE_POLYNOMIALS['kirkpatrick2021']['x=spt,y=teff'], teff, sptgrid, \
        nsample=nsample, interpolation='griddata')


def teff_to_mag_kirkpatrick(teff, nsample=1000):
    sptgrid=np.linspace(10, 43, 1000)
    return inverse_polynomial_relation(LITERATURE_POLYNOMIALS['kirkpatrick2021']['x=h_2mass,y=teff'], teff, sptgrid, \
        nsample=nsample, interpolation='griddata')

    
def teff_to_spt_subdwarf(teff):
    values= np.empty_like(teff)
    mask=np.logical_or(teff <1600, teff>2700) #use dwarfs for higher teff 
    values[mask]=teff_to_spt_pecaut(teff).flatten()[mask]
    pol=LITERATURE_POLYNOMIALS['gonzales2018']['x=spt,y=teff']
    sptgrid=np.linspace(10, 43, 1000)
    values[~mask]=inverse_polynomial_relation(pol, teff, sptgrid, \
        nsample=100, interpolation='griddata')[~mask]
    return values

def interpolated_local_lf():
    """
    """
    binedges= np.append(kirkpatrick2020LF['bin_center']-75, kirkpatrick2020LF['bin_center'][-1]+75)
    obs=np.array(kirkpatrick2020LF['values'])
    unc=np.array(kirkpatrick2020LF['unc'])
    return interp1d( binedges, obs, assume_sorted = False, fill_value = np.nan, bounds_error=False)

def polynomial_relation(x, xkey, ykey, ref, nsample=1000, xerr=0.0):
    """
    This function calculates the polynomial relation using the given input values.

    Parameters:
    x (float or array): The independent variable
    xkey (str): The independent variable key in the literature polynomial
    ykey (str): The dependent variable key in the literature polynomial
    ref (str): The reference for the literature polynomial
    nsample (int): The number of samples for Monte Carlo simulation (default=1000)
    xerr (float): The error in the independent variable (default=0.0)

    Returns:
    float or array: The polynomial relation value(s)
    """
    pol= LITERATURE_POLYNOMIALS[ref]['x={},y={}'.format(xkey, ykey)]
    return apply_polynomial_relation(pol, x, xerr=0.0, nsample=nsample)

def absolute_mag_from_spt(spt, mag='j', syst='mko', ref='kirkpatrick2021',  nsample=1000):
    """Returns absolute magnitude from polynomial relation.
    Args:
        spt (float): Spectral type.
        mag (str): Magnitude band to use (default 'j').
        syst (str): Magnitude system to use (default 'mko').
        ref (str): Reference for polynomial relation (default 'kirkpatrick2021').
        nsample (int): Number of samples to use for Monte Carlo (default 1000).

    Returns:
        float: Absolute magnitude.

    """
    #retuns abs mag from polynomial relation
    pol= LITERATURE_POLYNOMIALS[ref]['x=spt,y={}_{}'.format(mag, syst)]   
    return apply_polynomial_relation(pol, spt, xerr=0.0, nsample=nsample)