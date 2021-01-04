
################################
#population simulations routines
#includes copied functions from splat
##############################
import numpy as np
import numba
import scipy
from .config import DATA_FOLDER, POLYNOMIALS
from .tools import random_draw
import scipy.integrate as integrate
from scipy.interpolate import interp1d

#some constants
MAG_KEYS=['WFIRST_WFIJ', 'WFIRST_WFIH', 'WFIRST_WFIK', 'WFIRST_WFIY', 'WFIRST_WFIZ']
HS=[250, 300, 350, 500]
Rsun=8300.
Zsun=27.
R0=2600.
class Pointing(object):
    ## a pointing object making it easier to draw samples
    def __init__(self, **kwargs):
        #only input is the direction
        self.coord=kwargs.get('coord', None)
        self.survey=kwargs.get('survey', None)
        self.name=kwargs.get('name', None)
        self._mag_limits=kwargs.get('mag_limits', None)
        self.area=kwargs.get('area', None)
        self._dist_limits={}
        self._volume={}
        self.distance_cdf={}
        self.dens_profile=kwargs.get('density', 'exp')

        #compute volumes after initialization
        if self.coord is not None:
            for h in HS:
               self.distance_cdf.update({h:interpolated_cdf(self.coord.galactic.l.radian, \
                   self.coord.galactic.b.radian, h, kind=self.dens_profile)})

    @property
    def volume(self):
        return self._volume

    @property
    def distance_limits(self):
        return self._dist_limits

    @property
    def mag_limits(self):
        return self._mag_limits

    @mag_limits.setter
    def mag_limits(self, new_lts):
        self._mag_limits=new_lts
        #compute distance limits for each 
        for k in new_lts.keys():
            ds={}
            for s in np.arange(15, 41):
                dmin= get_distance(POLYNOMIALS[k][0](s), new_lts[k][0])
                dmax= get_distance(POLYNOMIALS[k][0](s), new_lts[k][1])
                ds.update({s: [dmin, dmax]})

            self._dist_limits.update({k: ds})


    def draw_distances(self, dmin, dmax, h, nsample=1e3):
        ##draw random distances in this direction for a specific spectral type
        d=np.logspace(np.log10(dmin), np.log10(dmax), int(nsample))
        #print (d, dmin, dmax)
        cdfvals=self.distance_cdf[h](d)
        #dgrid=np.concatenate([[0], np.logspace(-1, np.log10(dmax+50), int(1e4))])
        #cdfvals=np.array([volume_calc(self.coord.galactic.l.radian, self.coord.galactic.b.radian, 0, dx, h,kind=self.dens_profile) \
        #    for dx in dgrid])
        return random_draw(d, cdfvals/np.nanmax(cdfvals), int(nsample))

    

def interpolated_cdf(l, b, h, **kwargs):
    #interpolated cdf up a further distance to avoid using each time I have to draw a distance
    d=np.concatenate([[10**-2], np.logspace(-1, 6, int(1e4))])
    #print (d)
    cdfvals=np.array([volume_calc(l,b,0, dx, h, **kwargs) for dx in d])
    #remove nans and infinities
    cdfvals= cdfvals/np.nanmax(cdfvals)
    bools= np
    return interp1d(d, cdfvals)

def exponential_density(r, z, H, R0):
    """
        expoential galactic density porfile
    """
    zpart=np.exp(-abs(z-Zsun)/H)
    rpart=np.exp(-(r-Rsun)/R0)
    return zpart*rpart

def spheroid_density(r, z):
    """
        Spheroid density profile
    """
    q = 0.64 #: halo axial ratio
    n = 2.77#: spherical power law index
    return  (Rsun/(((r)**2+((z)/q)**2)**0.5))**n

@numba.vectorize("float64(float64, float64)", target='cpu')
def get_distance(absmag, rel_mag):
    return 10.**(-(absmag-rel_mag)/5. + 1.)

def volume_calc(l,b,dmin, dmax, h, kind='exp'):
    nsamp=1000
    fh=0.0051
    ds = np.linspace(dmin,dmax,nsamp)
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    rh0=exponential_density(rd, zd, h, R0)
    if kind =='spheroid':
         rh0=spheroid_density(rd, zd)
    if kind=='both':
        rh0= rh0+fh*spheroid_density(rd, zd)
    val=integrate.trapz(rh0*(ds**2), x=ds)
    return val