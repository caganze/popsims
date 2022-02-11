
from scipy.interpolate import griddata
import astropy.units as u
import numba
import pandas as pd
import numpy as np
import os

from .relations import teff_to_spt, scale_to_local_lf, spt_to_teff
from .tools import sample_from_powerlaw
from .config import *

DATA_FOLDER=os.environ['POPSIMS_DATA_FOLDER']


def read_bintemplates():
    df=pd.read_pickle(DATA_FOLDER+'/binary_lookup_table.pkl')
    return [df.prim.values, df.sec.values, df.sys.values]

def get_system_type(pr, sc, interpolators):
    """
    Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.

    """
    #where secondary are nans set to primaries
    sc[np.isnan(sc)]=pr[np.isnan(sc)]
    #interpolate
    interpoints=np.array([interpolators[0], interpolators[1] ]).T
    comb=griddata(interpoints, interpolators[-1] , (pr, sc), method='linear')
    return comb


def evolutionary_model_interpolator(mass, age, model):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
    model_filename=DATA_FOLDER+'/evmodels//'+model.lower()+'.csv'
    evolutiomodel=pd.read_csv( model_filename)

    #use the full cloud treatment for saumon models
    if model=='saumon2008':
         evolutiomodel=evolutiomodel[evolutiomodel.cloud=='hybrid']
 
    #make age, teff, mass logarithm scale
    valuest=np.log10(evolutiomodel.temperature.values)
    #valueslogg=evolutiomodel.gravity.values
    valueslumn=evolutiomodel.luminosity.values

    valuesm=np.log10(evolutiomodel.mass.values)
    valuesag=np.log10(evolutiomodel.age.values)

    evolpoints=np.array([valuesm, valuesag ]).T

    teffs=griddata(evolpoints, valuest , (np.log10(mass), np.log10(age)), method='linear')
    lumn=griddata(evolpoints, valueslumn , (np.log10(mass), np.log10(age)), method='linear')

    return {'mass': mass*u.Msun, 'age': age*u.Gyr, 'temperature': 10**teffs*u.Kelvin, 
    'luminosity': lumn*u.Lsun}




def simulate_spts(nsample=int(1e4), model_name='baraffe2003', save=False, mass_age_range=None, filename=None ):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """

    #use hybrid models that predit the T dwarf bump for Saumon Models
    if model_name=='saumon2008':
        cloud='hybrid'
    else:
        cloud=False

    #automatically set maxima and minima to avoid having too many nans
    #mass age and age,  min, max
    #all masses should be 0.01
    default_values={'baraffe2003': [0.01, 0.1, 0.01, 8.0],
    'marley2019': [0.01, 0.08, 0.001, 8.0], 'saumon2008':[0.01, 0.09, 0.003, 8.0], 
    'phillips2020':[0.01, 0.075, 0.001, 8.0 ],'burrows2001':[0.01, 0.075, 10, 12]}

    if  mass_age_range is None:
        mass_age_range= default_values[model_name]
        #raise ValueError('mass and age range is none')
    # masses for singles [this can be done with pymc but nvm]
    #m_singles = spsim.simulateMasses(nsim,range=[ mass_age_range[0], ranges[1]],distribution='power-law',alpha=0.6)
    m_singles=sample_from_powerlaw(-0.6, xmin= mass_age_range[0], xmax=mass_age_range[1], nsample=nsample)
    ages_singles= np.random.uniform(mass_age_range[2], mass_age_range[-1], nsample)

    #parameters for binaries
    qs=sample_from_powerlaw(4, xmin= 0., xmax=1., nsample=nsample)
    m_prims = sample_from_powerlaw(-0.6, xmin= mass_age_range[0], xmax=mass_age_range[1], nsample=nsample)
    m_sec=m_prims*qs
    ages_bin=np.random.uniform(mass_age_range[2], mass_age_range[-1], nsample)

    #interpolate evolurionary models
    single_evol=evolutionary_model_interpolator(m_singles, ages_singles, model_name)
    primary_evol=evolutionary_model_interpolator(m_prims,ages_bin, model_name)
    secondary_evol=evolutionary_model_interpolator(m_sec,ages_bin, model_name)

    #temperatures
    teffs_singl =single_evol['temperature'].value
    teffs_primar=primary_evol['temperature'].value
    teffs_second=secondary_evol['temperature'].value

    #spectraltypes
    spts_singl =teff_to_spt(teffs_singl)
    spt_primar=teff_to_spt(teffs_primar)
    spt_second=teff_to_spt(teffs_second)

    #compute binary spectral types
    xy=np.vstack([np.round(np.array(spt_primar), decimals=0), np.round(np.array(spt_second), decimals=0)]).T
    spt_binr=get_system_type(xy[:,0], xy[:,1], read_bintemplates())



    values={ 'sing_evol': single_evol, 'sing_spt':spts_singl,
        		 'prim_evol': primary_evol, 'prim_spt':spt_primar,
        		 'sec_evol': secondary_evol, 'sec_spt': spt_second,
        		'binary_spt': spt_binr }

    if save and (filename is not None):
        import pickle
        with open(filename, 'wb') as file:
           pickle.dump(values,file)
    if save and (filename is not None):
        raise ValueError('file name empty')

    return values

def make_systems(bfraction=0.2, **kwargs):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """

    mods=simulate_spts(**kwargs)
    
    #singles
    singles=mods['sing_evol']
    #singles['abs_2MASS_J']= get_abs_mag(mods['sing_spt'], '2MASS J')[0]
        #bolometric corrections for 2MASS J
    #bcs_sings=fillipazzo_bolometric_correction(mods['sing_spt'], filt='2MASS_J', 
    #                                    mask=None)

    #singles['bolometric_cor_2MASS_J']=bcs_sings
    #singles['abs_2MASS_J']=get_mag_from_luminosity(singles['luminosity'].value,\
    #                                                bcs_sings, log=False)
    singles['is_binary']= np.zeros_like(mods['sing_spt']).astype(bool)
    singles['spt']=mods['sing_spt']
    singles['prim_spt']=mods['sing_spt']
    singles['sec_spt']=np.ones_like(mods['sing_spt'])*np.nan

    #print (np.isnan(singles['temperature']).all())
    
    #binary
    binaries={}
    binaries['age']=mods['prim_evol']['age']
    binaries['mass']=mods['prim_evol']['mass']+mods['sec_evol']['mass']
    binaries['pri_mass']=mods['prim_evol']['mass']
    binaries['sec_mass']=mods['sec_evol']['mass']
    
    binaries['luminosity']=np.log10(10**(mods['prim_evol']['luminosity']).value+\
    10**(mods['sec_evol']['luminosity']).value)
    #binaries['temperature']=mods['prim_evol']['temperature']
    binaries['spt']=np.random.normal(mods['binary_spt'], 0.3)
    binaries['prim_spt']=mods['prim_spt']
    binaries['sec_spt']=mods['sec_spt']
    binaries['prim_luminosity']=10**(mods['prim_evol']['luminosity']).value
    binaries['sec_luminosity']=10**(mods['sec_evol']['luminosity']).value

    binaries['is_binary']=np.ones_like(mods['sec_spt']).astype(bool)

    
    #bolometric corrections for 2MASS J
    #bcs_bins=fillipazzo_bolometric_correction(binaries['spt'], filt='2MASS_J', 
    #                                    mask=None)
    #binaries['bolometric_cor_2MASS_J']=bcs_bins
    

    #magnitudes ugh
    """
    ignore 2mass photometry

    js_singles, j_single_unc=get_abs_mag(mods['sing_spt'],'2MASS J')
    hs_singles, h_single_unc=get_abs_mag(mods['sing_spt'],'2MASS H')

    singles['abs_2MASS_J']=np.random.normal(js_singles, j_single_unc)
    singles['abs_2MASS_H']=np.random.normal(hs_singles, h_single_unc)

    js_primns, junc_prims=get_abs_mag(mods['prim_spt'], '2MASS J')
    js_prims_to_use=np.random.normal(js_primns, junc_prims)

    hs_primns, hunc_prims=get_abs_mag(mods['prim_spt'], '2MASS H')
    hs_prims_to_use=np.random.normal(hs_primns, junc_prims)

    js_secs, junc_secs=get_abs_mag(mods['sec_spt'], '2MASS J')
    js_secs_to_use=np.random.normal(js_secs, junc_secs)

    hs_secs, hunc_secs=get_abs_mag(mods['sec_spt'], '2MASS H')
    hs_secs_to_use=np.random.normal(hs_secs, hunc_secs)

    #print (np.isnan(js_prims_to_us).any())

    binaries['abs_2MASS_J']= -2.5*np.log10(10**(-0.4*js_prims_to_use)+ 10**(-0.4*js_secs_to_use))
    binaries['abs_2MASS_H']=  -2.5*np.log10(10**(-0.4*hs_prims_to_use)+ 10**(-0.4*hs_secs_to_use))
    """

    #assign teff from absolute mag
    #binaries['temperature']=get_teff_from_mag_ignore_unc(binaries['abs_2MASS_H'])
    binaries['temperature']=spt_to_teff(binaries['spt'])
    #binaries['temperature']=
    #print (np.isnan(binaries['temperature']).all())

    
    #compute numbers to choose based on binary fraction
    ndraw= int(len(mods['sing_spt'])/(1-bfraction))-int(len(mods['sing_spt']))
    #ndraw=int(len(mods['sing_spt'])* bfraction)

    
    #random list of binaries to choose
    random_int=np.random.choice(np.arange(len(binaries['spt'])), ndraw)
    
    chosen_binaries={}
    for k in binaries.keys():
        chosen_binaries[k]=binaries[k][random_int]
    
    #add scale to the local lf
    res=pd.concat([pd.DataFrame(singles), pd.DataFrame(chosen_binaries)])
    scl=scale_to_local_lf(res.temperature.values)
    #print (scl
    res['scale']=scl[0]
    res['scale_unc']=scl[1]
    res['scale_times_model']=scl[-1]

    #combine the to dictionaries 
    #print (np.isnan(res['temperature']).all())

    return res


def make_systems_nocombined_light(**kwargs):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
   
    model_vals=simulate_spts(**kwargs)
    ndraw=int(len(model_vals['sing_spt'])* binary_fraction)


    nans=np.isnan(model_vals['binary_spt'])
    
    choices={'spt': np.random.choice(model_vals['binary_spt'][~nans], ndraw),
            'teff': np.random.choice(model_vals['prim_evol']['temperature'].value[~nans], ndraw), 
            'age': np.random.choice(model_vals['prim_evol']['age'].value[~nans],ndraw),
            'mass': np.random.choice(model_vals['prim_evol']['mass'].value[~nans]+model_vals['sec_evol']['mass'].value[~nans],ndraw)}


    vs={'system_spts': np.concatenate([model_vals['sing_spt'], choices['spt']]), 
            'system_teff':  np.concatenate([(model_vals['sing_evol']['temperature']).value, choices['teff']]),
            'system_age':  np.concatenate([(model_vals['sing_evol']['age']).value,  choices['age']]),
            'system_mass': np.concatenate([(model_vals['sing_evol']['mass']).value,  choices['mass']])}

    return vs