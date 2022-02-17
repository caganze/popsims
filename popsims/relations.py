
################################
#includes some useful relations, for full package, see splat
##############################

from scipy.interpolate import interp1d
import numpy as np
import splat.empirical as spe

TEFF_SPT_RELATIONS={'pecaut': {'bibcode': '2013ApJS..208....9P', 'url': 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt', \
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
       108.]}, \

'kirkpatrick2021': {'bibcode': None, 'method':''}}

kirkpatrick2020LF={'bin_center': np.array([ 525,  675,  825,  975, 1125, 1275, 1425, 1575, 1725, 1875, 2025]),
    'values': np.array([4.24, 2.8 , 1.99, 1.72, 1.11, 1.95, 0.94, 0.81, 0.78, 0.5 , 0.72]),
    'unc': np.array([0.7 , 0.37, 0.32, 0.3 , 0.25, 0.3 , 0.22, 0.2 , 0.2 , 0.17, 0.18])}

def teff_to_spt(teff):
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
    rel=TEFF_SPT_RELATIONS['pecaut']
    teffsc=np.random.normal(teff, 108)
    fx=interp1d(rel['values'], rel['spt'], assume_sorted = False, fill_value = np.nan, bounds_error=False)
    return fx(teffsc)
  
def spt_to_teff(spt):
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
    rel=TEFF_SPT_RELATIONS['pecaut']
    fx=interp1d(rel['spt'], rel['values'], assume_sorted = False, fill_value = np.nan,  bounds_error=False)
    return np.random.normal(fx(spt), 108)


def scale_to_local_lf(teffs):
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
        IOError
    """
    #change to convetion L0=0
    spt=spt-20
    res= np.nan
    rms=79
    if spt<0:
        res, rms=spe.typeToTeff(float(spt+20), ref='mamajek')
        res=res.value
        rms=rms.value
        
    if np.logical_and(spt>=0, spt<=8.75):
        res= 2.2375e+03-1.4496e+02*spt+4.0301e+00*(spt**2)
        rms=134
    if np.logical_and(spt>=8.75, spt<=14.75):
        res= 1.4379e+03-1.8309e+01*spt
    if np.logical_and(spt>=14.75, spt<=22):
        res= 5.1413e+03-3.6865e+02*spt+6.7301e+00*(spt**2)
    
    return res, rms

def teff_to_spt_kirkpatrick(teff):
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
        IOError
    """
    
    #spectral type grid
    sp_grid=np.arange(10, 43)
    teff_grid, unc= np.vstack([list(spt_to_teff_kirkpatrick(x)) for x in sp_grid]).T
    
    
    #randomize
    rand_teffs= np.random.normal(teff_grid, unc, size=(10_000, len(unc)) ).flatten()
    rand_types= np.vstack([sp_grid for x in range(0, 10_000)]).flatten()
   
    #return interpolated value and uncertainties
    f=interp1d(rand_teffs, rand_types, assume_sorted = False, fill_value = np.nan, bounds_error=False)

    return f(teff)

def interpolated_local_lf():
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
        IOError
    """

    binedges= np.append(kirkpatrick2020LF['bin_center']-75, kirkpatrick2020LF['bin_center'][-1]+75)
    obs=np.array(kirkpatrick2020LF['values'])
    unc=np.array(kirkpatrick2020LF['unc'])

    return interp1d( binedges, obs, assume_sorted = False, fill_value = np.nan, bounds_error=False)
