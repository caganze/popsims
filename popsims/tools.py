import numpy as np
import numba
import bisect
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
 
def sample_from_powerlaw(alpha, xmin=0.1, xmax=1, nsample=int(1e4)):
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
    x= np.linspace(xmin, xmax, int(1e6))
    pdf=x**alpha
    cdf=np.cumsum(pdf)
    return random_draw(x, cdf, nsample=int(nsample))

def random_draw(x_grid, cdf, nsample=10):
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
    cdf = cdf / cdf[-1]
    values = np.random.rand(nsample)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf

def make_spt_number(spt):
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
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

def dropnans(x):
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
    return x[~np.isnan(x)]

def group_by(xvalues, yvalues, grid= np.arange(0, 1, 1000)):
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

def random_draw_old(xvals, cdfvals, nsample=10):
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
    @numba.vectorize("int32(float64)")
    def invert_cdf(i):
        return bisect.bisect(cdfvals, i)-1
    x=np.random.rand(nsample)
    idx=invert_cdf(x)
    return np.array(xvals)[idx]

def compute_uvw_from_pm(ra_J2000, dec_J2000, parallax, rv, mu_ra, mu_dec, e_parallax, e_rv, e_mu_ra, e_mu_dec, correct_lsr=True):
    #from Dino's code
    """
    Compute the Galactic UVW space velocity based on the formulation in Johnson and Soderblom (1987).
    Parameters
    ----------
    ra          :   float
                    RA of the source in degrees
    dec         :   float
                    Dec of the source in degrees
    parallax    :   float
                    the parallax in mas
    rv          :   float
                    the radial velocity in km/s
    mu_ra       :   float
                    the proper motion in right ascension in mas/yr
    mu_dec      :   float
                    the proper motion in declination in mas/yr
    e_parallax  :   float
                    the error of parallax in mas
    e_rv        :   float
                    the error of radial velocity in km/s
    e_mu_ra     :   float
                    the error of proper motion in right ascension in mas/yr
    e_mu_dec    :   float
                    the error of proper motion in declination in mas/yr
    Optional Parameters
    -------------------
    correct_lsr :   bool
                    If True: uvw corrected to the LSR
    Returns
    -------
    uvw         :   array-like
                    UVW velocities in km/s
    e_uvw       :   array-like
                    errors of UVW velocities in km/s
    """
    ## convert proper motions and parallax from mas to arcsec
    parallax   /= 1000
    mu_ra      /= 1000
    mu_dec     /= 1000

    e_parallax /= 1000
    e_mu_ra    /= 1000
    e_mu_dec   /= 1000

    ## convert ra and dec into radians (the paper uses equinox 1950)
    coord_J2000 = SkyCoord(ra_J2000*u.deg, dec_J2000*u.deg, unit='deg', frame='icrs')

    coord_J1950 = coord_J2000.transform_to(FK5(equinox='J1950.0'))

    ra          = coord_J1950.ra.value
    dec         = coord_J1950.dec.value

    ## degree to radian conversion
    deg_to_rad  = np.pi/180

    ## define the A matrix
    A_ra      = np.array([  [   +np.cos(ra*deg_to_rad),     +np.sin(ra*deg_to_rad), 0],
                            [   +np.sin(ra*deg_to_rad),     -np.cos(ra*deg_to_rad), 0],
                            [                        0,                         0, -1]])

    A_dec     = np.array([  [   +np.cos(dec*deg_to_rad),     0, -np.sin(dec*deg_to_rad)],
                            [                         0,    -1,                       0],
                            [   -np.sin(dec*deg_to_rad),     0, -np.cos(dec*deg_to_rad)]])

    A         = A_ra.dot(A_dec)

    #A0         = np.array([[   +np.cos(ra*deg_to_rad)*np.cos(dec*deg_to_rad), -np.sin(ra*deg_to_rad), -np.cos(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
    #                   [   +np.sin(ra*deg_to_rad)*np.cos(dec*deg_to_rad), +np.cos(ra*deg_to_rad), -np.sin(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
    #                   [   +np.sin(dec*deg_to_rad)                      ,                      0, +np.cos(dec*deg_to_rad)                     ]])

    ## define RA and Dec for the North Galactic Pole (NGP) in degrees
    ra_ngp  = 192.25
    dec_ngp = 27.4
    theta0  = 123 # the position angle of NGP relative to great semi-circle of the North Celetial Pole and the zero Galactic longitude
    
    T1      = np.array([[  +np.cos(theta0*deg_to_rad), +np.sin(theta0*deg_to_rad),  0],
                        [  +np.sin(theta0*deg_to_rad), -np.cos(theta0*deg_to_rad),  0],
                        [                           0,                          0,  +1]])

    T2      = np.array([[-np.sin(dec_ngp*deg_to_rad),  0, +np.cos(dec_ngp*deg_to_rad)],
                        [                          0, -1,                           0],
                        [+np.cos(dec_ngp*deg_to_rad),  0, +np.sin(dec_ngp*deg_to_rad)]])

    T3      = np.array([[  +np.cos(ra_ngp*deg_to_rad), +np.sin(ra_ngp*deg_to_rad),  0],
                        [  +np.sin(ra_ngp*deg_to_rad), -np.cos(ra_ngp*deg_to_rad),  0],
                        [                           0,                          0, +1]])

    ## define the T matrix
    T       = T1.dot(T2.dot(T3))

    ## B matrix = TA
    B       = T.dot(A)

    ## uvw matrix
    k       = 1.4959787 * 10**8 / 365.24219879 / 24 /3600 #4.74057 # AU/tropical yr (km/s)
    uvw     = B.dot(np.array([  [rv], 
                                [k * mu_ra  / parallax], 
                                [k * mu_dec / parallax]]))

    ## solar uvw from Schonrich et al. (2010)
    uvw_solar = np.array([  [11.1], [12.24], [7.25] ])

    C       = B**2
    e_uvw2  = C.dot(np.array([  [ e_rv**2], 
                                [ (k/parallax)**2 * ( e_mu_ra**2  + ( mu_ra  * e_parallax / parallax )**2 )], 
                                [ (k/parallax)**2 * ( e_mu_dec**2 + ( mu_dec * e_parallax / parallax )**2 )]    ])) \
                    + 2 * mu_ra * mu_dec * k**2 * e_parallax**2 / parallax**4 * \
                    np.array([  [ B[0][1]*B[0][2] ], 
                                [ B[1][1]*B[1][2] ], 
                                [ B[2][1]*B[2][2] ] ])

    if correct_lsr: uvw += uvw_solar

    return uvw, np.sqrt(e_uvw2)

def plot_annotated_heatmap(ax, data, gridpoints, columns, cmap='viridis', 
                           annotate=False, vmin=0.0, vmax=1.0, textsize=14):
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
    cax = ax.pcolormesh(xgrid, ygrid, values2, vmin=vmin, vmax=vmax, cmap=cmap)
    #plt.axis('tight')
    ymin, ymax = plt.ylim()

    ax.minorticks_on()

    ax.set_ylim(ymax, ymin)
    return 

def compute_pm_from_uvw(ra_J2000, dec_J2000, parallax, us, vs, ws, correct_lsr=False):
    """
    Compute the Galactic UVW space velocity based on the formulation in Johnson and Soderblom (1987).
    Parameters
    ----------
    ra          :   float
                    RA of the source in degrees
    dec         :   float
                    Dec of the source in degrees
    parallax    :   float
                    the parallax in pc
    us, vs, ws  :   float
                    the UVW velocities in km/s
    Optional Parameters
    -------------------
    correct_lsr :   bool
                    If True: uvw corrected to the LSR
    Returns
    -------
    array: rv (km/s), proper motion in (mas/yr), vtan (km/s)
    """
    ## convert proper motions and parallax from mas to arcsec
    parallax   /= 1000
    #mu_ra     /= 1000
    #mu_dec        /= 1000

    #e_parallax /= 1000
    #e_mu_ra    /= 1000
    #e_mu_dec   /= 1000

    ## convert ra and dec into radians (the paper uses equinox 1950)
    coord_J2000 = SkyCoord(ra_J2000*u.deg, dec_J2000*u.deg, unit='deg', frame='icrs')

    coord_J1950 = coord_J2000.transform_to(FK5(equinox='J1950.0'))

    ra          = coord_J1950.ra.value
    dec         = coord_J1950.dec.value

    ## degree to radian conversion
    deg_to_rad  = np.pi/180

    ## define the A matrix
    A_ra      = np.array([  [   +np.cos(ra*deg_to_rad),     +np.sin(ra*deg_to_rad), 0],
                            [   +np.sin(ra*deg_to_rad),     -np.cos(ra*deg_to_rad), 0],
                            [                        0,                         0, -1]])

    A_dec     = np.array([  [   +np.cos(dec*deg_to_rad),     0, -np.sin(dec*deg_to_rad)],
                            [                         0,    -1,                       0],
                            [   -np.sin(dec*deg_to_rad),     0, -np.cos(dec*deg_to_rad)]])

    A         = A_ra.dot(A_dec)

    #A0         = np.array([[   +np.cos(ra*deg_to_rad)*np.cos(dec*deg_to_rad), -np.sin(ra*deg_to_rad), -np.cos(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
    #                   [   +np.sin(ra*deg_to_rad)*np.cos(dec*deg_to_rad), +np.cos(ra*deg_to_rad), -np.sin(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
    #                   [   +np.sin(dec*deg_to_rad)                      ,                      0, +np.cos(dec*deg_to_rad)                     ]])

    ## define RA and Dec for the North Galactic Pole (NGP) in degrees
    ra_ngp  = 192.25
    dec_ngp = 27.4
    theta0  = 123 # the position angle of NGP relative to great semi-circle of the North Celetial Pole and the zero Galactic longitude

    T1      = np.array([[  +np.cos(theta0*deg_to_rad), +np.sin(theta0*deg_to_rad),  0],
                        [  +np.sin(theta0*deg_to_rad), -np.cos(theta0*deg_to_rad),  0],
                        [                           0,                          0,  +1]])

    T2      = np.array([[-np.sin(dec_ngp*deg_to_rad),  0, +np.cos(dec_ngp*deg_to_rad)],
                        [                          0, -1,                           0],
                        [+np.cos(dec_ngp*deg_to_rad),  0, +np.sin(dec_ngp*deg_to_rad)]])

    T3      = np.array([[  +np.cos(ra_ngp*deg_to_rad), +np.sin(ra_ngp*deg_to_rad),  0],
                        [  +np.sin(ra_ngp*deg_to_rad), -np.cos(ra_ngp*deg_to_rad),  0],
                        [                           0,                          0, +1]])

    ## define the T matrix
    T       = T1.dot(T2.dot(T3))

    ## B matrix = TA
    B       = T.dot(A)
    uvw= np.vstack([us, vs, ws])
    uvw_solar = np.array([  [11.1], [12.24], [7.25] ])

    if correct_lsr: uvw += uvw_solar

    motion=np.linalg.solve(B, uvw)

    ## uvw matrix
    k       = 1.4959787 * 10**8 / 365.24219879 / 24 /3600 #4.74057 # AU/tropical yr (km/s)
    rv=motion[0]
    mu_ra=motion[1]/(k*parallax)
    mu_dec=motion[-1]/(k*parallax)
    vtan=np.sqrt(k* (mu_ra**2+ mu_dec**2))*parallax
    return np.array([rv, mu_ra, mu_dec, vtan]).flatten()