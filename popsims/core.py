import splat.simulate as spsim
import splat.evolve as spev
import splat.empirical as splat_teff_to_spt

from .config import DATA_FOLDER, POLYNOMIALS, EVOL_MODELS_FOLDER
from .tools import teff_to_spt
#import pymc3 as pm
from scipy.interpolate import griddata
#import theano.tensor as tt
#from theano.compile.ops import as_op
import astropy.units as u
import numba
import pandas as pd
import numpy as np

def read_bintemplates():
    df=pd.read_pickle(DATA_FOLDER+'/binary_lookup_table.pkl.gz')
    return [df.prim.values, df.sec.values, df.sys.values]

def get_system_type(pr, sc, interpolators):
    """
    use the lookup table to get a spectral type for the binary
    using a linear interpolation to avoid nans
    pr: primary type (float, M0=10)
    sc: secondary type float, M0=10)
    interpolatotrs: (3, N) array of loats (0: primary, 1: secondary, 2: system)
    """
    #where secondary are nans set to primaries
    sc[np.isnan(sc)]=pr[np.isnan(sc)]
    #interpolate
    interpoints=np.array([interpolators[0], interpolators[1] ]).T
    comb=griddata(interpoints, interpolators[-1] , (pr, sc), method='linear')
    return comb


def evolutionary_model_interpolator(mass, age, model):
    """
    Evolutionary model interpolator
    input: mass, age
    model: model name
    """

    model_filename=EVOL_MODELS_FOLDER+'//'+model.lower()+'.csv'
    evolutiomodel=pd.read_csv( model_filename)

    #use the full cloud treatment for saumon models
    if model=='saumon2008':
         evolutiomodel=evolutiomodel[evolutiomodel.cloud=='hybrid']
 
    #make age, teff, mass logarithm scale
    valuest=np.log10(evolutiomodel.temperature.values)
    valueslogg=evolutiomodel.gravity.values
    valueslumn=evolutiomodel.luminosity.values

    valuesm=np.log10(evolutiomodel.mass.values)
    valuesag=np.log10(evolutiomodel.age.values)

    evolpoints=np.array([valuesm, valuesag ]).T

    teffs=griddata(evolpoints, valuest , (np.log10(mass), np.log10(age)), method='linear')
    lumn=griddata(evolpoints, valueslumn , (np.log10(mass), np.log10(age)), method='linear')


    return {'mass': mass*u.Msun, 'age': age*u.Gyr, 'temperature': 10**teffs*u.Kelvin, 
    'luminosity': lumn*u.Lsun}




def simulate_spts(**kwargs):
    """
    Simulate parameters from mass function,
    mass ratio distribution and age distribution
    """
    recompute=kwargs.get('recompute', False)
    model_name=kwargs.get('name','baraffe2003')

    #use hybrid models that predit the T dwarf bump for Saumon Models
    if model_name=='saumon2008':
        cloud='hybrid'
    else:
        cloud=False

    #automatically set maxima and minima to avoid having too many nans
    #mass age and age,  min, max
    #all masses should be 0.01
    acceptable_values={'baraffe2003': [0.01, 0.1, 0.01, 8.0],
    'marley2019': [0.01, 0.08, 0.001, 8.0], 'saumon2008':[0.01, 0.09, 0.003, 8.0], 
    'phillips2020':[0.01, 0.075, 0.001, 8.0 ],'burrows2001':[0.01, 0.075, 10, 12]}
    
    if recompute:

        nsim = kwargs.get('nsample', 1e5)

        ranges=kwargs.get('range',None)
        
        # masses for singles [this can be done with pymc but nvm]
        m_singles = spsim.simulateMasses(nsim,range=[ranges[0], ranges[1]],distribution='power-law',alpha=0.6)
        #ages for singles
        ages_singles= spsim.simulateAges(nsim,range=[ranges[2], ranges[3]], distribution='uniform')

        #parameters for binaries
        #binrs=simulate_binary(int(nsim), [ranges[0], ranges[1]], [ranges[2], ranges[3]])
        qs=spsim.simulateMassRatios(nsim,distribution='power-law',q_range=[0.1,1.0],gamma=4)
        m_prims = spsim.simulateMasses(nsim,range=[ranges[0], ranges[1]],distribution='power-law',alpha=0.6)
        m_sec=m_prims*qs
        ages_bin= spsim.simulateAges(nsim,range=[ranges[2], ranges[3]], distribution='uniform')

        #single_evol=spev.modelParameters(mass=m_singles,age=ages_singles, set=model_name, cloud=cloud)
        single_evol=evolutionary_model_interpolator(m_singles, ages_singles, model_name)

        #primary_evol=spev.modelParameters(mass=binrs[0],age=binrs[-1], set=model_name, cloud=cloud)
        primary_evol=evolutionary_model_interpolator(m_prims,ages_bin, model_name)

        #secondary_evol=spev.modelParameters(mass=binrs[1],age=binrs[-1], set=model_name, cloud=cloud)
        secondary_evol=evolutionary_model_interpolator(m_sec,ages_bin, model_name)
        #save luminosities

        #temperatures
        teffs_singl =single_evol['temperature'].value
        teffs_primar=primary_evol['temperature'].value
        teffs_second=secondary_evol['temperature'].value

        #spectraltypes
        spts_singl =teff_to_spt(teffs_singl)

        #the singles will be fine, remove nans from systems 
        spt_primar=teff_to_spt(teffs_primar)
        spt_second=teff_to_spt(teffs_second)

        xy=np.vstack([np.round(np.array(spt_primar), decimals=0), np.round(np.array(spt_second), decimals=0)]).T

        spt_binr=get_system_type(xy[:,0], xy[:,1], read_bintemplates())

   
        values={ 'sing_evol': single_evol, 'sing_spt':spts_singl,
        		 'prim_evol': primary_evol, 'prim_spt':spt_primar,
        		 'sec_evol': secondary_evol, 'sec_spt': spt_second,
        		'binary_spt': spt_binr }

        import pickle
        with open(DATA_FOLDER+'/mass_age_spcts_with_bin{}.pkl'.format(model_name), 'wb') as file:
           pickle.dump(values,file)
    else:
        values=pd.read_pickle(DATA_FOLDER+'/mass_age_spcts_with_bin{}.pkl'.format(model_name))

    return values

def make_systems(**kwargs):
    """
    choose a random sets of primaries and secondaries 
    and a sample of single systems based off a preccomputed-evolutionary model grid 
    and an unresolved binary fraction

    """
    #recompute for different evolutionary models
    model=kwargs.get('model_name', 'baraffe2003')
    binary_fraction=kwargs.get('bfraction', 0.2)

    model_vals=simulate_spts(name=model, **kwargs)


    #nbin= int(len(model_vals['sing_spt'])*binary_fraction) #number of binaries
    ndraw= int(len(model_vals['sing_spt'])/(1-binary_fraction))-int(len(model_vals['sing_spt']))


    nans=np.isnan(model_vals['binary_spt'])
    
    choices={'spt': np.random.choice(model_vals['binary_spt'][~nans], ndraw),
            'teff': np.random.choice(model_vals['prim_evol']['temperature'].value[~nans], ndraw), 
            'age': np.random.choice(model_vals['prim_evol']['age'].value[~nans],ndraw)}


    vs={'system_spts': np.concatenate([model_vals['sing_spt'], choices['spt']]), 
            'system_teff':  np.concatenate([(model_vals['sing_evol']['temperature']).value, choices['teff']]),
            'system_age':  np.concatenate([(model_vals['sing_evol']['age']).value,  choices['age']])}

    return vs