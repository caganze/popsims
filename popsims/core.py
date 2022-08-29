
from scipy.interpolate import griddata
import astropy.units as u
import numba
import pandas as pd
import numpy as np
import os
import sys

from .relations import  scale_to_local_lf, teff_to_spt_kirkpatrick, \
spt_to_teff_kirkpatrick, teff_to_spt_pecaut, spt_to_teff_pecaut
from .tools import sample_from_powerlaw

#DATA_FOLDER=os.environ['POPSIMS_DATA_FOLDER']

CODE_FOLDER=os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(CODE_FOLDER)

DATA_FOLDER= CODE_FOLDER+'/data/'


def read_bintemplates(filename=None):
    #must be a pickle file with columns prim, sec, sys all floats
    if filename == None:
        df=pd.read_csv(DATA_FOLDER+'/binary_lookup_table_small.csv')
    else:
        file=pd.read_pickle(filename)
    return [df.prim.values, df.sec.values, df.sys.values]

def get_system_type(pr, sc, interpolators):
    """
    Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
    ----
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
    -------
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
    -------
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
    model_filename=DATA_FOLDER+'/models/evolutionary/'+model.lower()+'.csv'
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