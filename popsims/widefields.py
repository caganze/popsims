
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
from .core import make_systems,  POLYNOMIALS
from .galaxy import exponential_density, spheroid_density, Rsun, Zsun, get_uvw
from scipy.interpolate import interp1d
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import pymc3 as pm


def galactic_density(rd, zd, Hthin):
    fh=0.0051
    ft=0.12
    #only change thin disk scaleheight, keep thick disk and halo fixed
    thin=exponential_density(rd, zd, Hthin, 2600)
    thick=exponential_density(rd, zd, 900, 2600)
    halo=spheroid_density(rd, zd)
    return {'thin': thin, 'thick': ft*thick , 'halo': fh*halo}

def logtotal_galaxy_density_likelihood(r, z, d,Hthin):
    return np.log((d**2)*galactic_density(r, z, Hthin))

def transform_tocylindrical(l, b, ds):
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    return (rd, zd)

def create_mixed_population(r=None, ds=None, z=None, bfraction=None,\
                     model=None, fh=None, nsample=None, field=None, 
                     dmax=None, l=None, b=None, absmag_keys=['WFIRST_WFIJ']):
    #create population 
    if  field is not None:
         tr=field.sample_points(nsample=nsample, Hthin=350, dmax= dmax)
         df=make_systems(model=model, bfraction=bfraction)
         df['d']=tr['d'].flatten()
         df['r']=tr['r'].flatten()
         df['z']=tr['z'].flatten()
         df['l']=tr['l'].flatten()
         df['b']=tr['b'].flatten()

    else:
        df=make_systems(model=model, bfraction=bfraction)
        df=df.sample(n=len(ds))
        #assign distances
        df['d']=ds
        df['r']=r
        df['z']=z
        df['l']=l
        df['b']=b
    
    df['population']='disk'
    df['population'].loc[df.sample(frac=fh).index.values]='halo'
    disk_bool=(df['population']=='disk').values
    halo_bool= ~disk_bool
   
    us, vs, ws=get_uvw(df.age[disk_bool], kind='dwarf')
    hus, hvs, hws=get_uvw(df.age[halo_bool], z= df.z[halo_bool], kind='subdwarf')
    
    df['U']=None
    df['V']=None
    df['W']=None
    
    
    df['U'][disk_bool]=us
    df['V'][disk_bool]=vs
    df['W'][disk_bool]=ws
    
    df['U'][halo_bool]=hus
    df['V'][halo_bool]=hvs
    df['W'][halo_bool]=hws
   
    dff=get_magnitudes(df.spt, df.d, keys=absmag_keys).join(df)
    #proper motion 
    return dff


class WideField(object):
    def __init__(self, **kwargs):
        #only input is the direction
        self.name=kwargs.get('name', None)
        self.area=None
        self.density_likelihood=kwargs.get('density', 'exp')
        self._footprinttype=kwargs.get('footprinttype', 'discrete')
        self._volume=None
        self._footprint=None
        
    @property
    def volume(self):
        self._volume
    
    @property
    def footprint(self):
        ##a set of n images of area x each centered around positions 
        return self._footprint
    
    @footprint.setter
    def footprint(self, footprint):
        ##a set of n images of area x each centered around positions 
        
        if self._footprinttype=='discrete':
            self._footprint={'l':footprint['l'],'b':footprint['b'], \
                              'FOV': footprint['FOV']}
    
        if self._footprinttype=='uniform':
            self._footprint={'l_bounds':footprint['l'],\
                             'b_bounds':footprint['b'],\
                             'FOV': footprint['FOV']}
            
        self.area=self._footprint['FOV']

            
    def sample_points(self, nsample=1000, Hthin=300, dmin=0, dmax=5000):
        """
        sample the galaxy given a scale height
        """
        #add an option for sampling a uniform distribution for scale-heights
        with pm.Model() as model:
            
            if self._footprinttype=='discrete':
                l=pm.Uniform('l',  lower=-2*np.pi, upper=2*np.pi, testval=np.pi/2, \
                             observed=self._footprint['l'])
                b=pm.Uniform('b', lower=-np.pi, upper=np.pi, testval=np.pi/3, \
                             observed=self._footprint['b'])
                
            if  self._footprinttype=='uniform':
                #sphere point-picking
                
                l=pm.Uniform('l', lower=self._footprint['l_bounds'][0], 
                             upper=self._footprint['l_bounds'][1])
                
                b=pm.Uniform('b', lower=self._footprint['b_bounds'][0], 
                             upper=self._footprint['b_bounds'][1])
                
            d=pm.Uniform('d', lower=dmin, upper=dmax, testval=dmin+(dmax-dmin)/2.)

            x=pm.Deterministic('x',  Rsun-d*np.cos(b)*np.cos(l))
            y=pm.Deterministic('y', -d*np.cos(b)*np.sin(l))
            r=pm.Deterministic('r', (x**2+y**2)**0.5 )
            z=pm.Deterministic('z', Zsun+ d * np.sin(b))

            like = pm.Potential('likelihood', \
                                logtotal_galaxy_density_likelihood(r, z, d,  Hthin))

            trace = pm.sample(draws=int(nsample), cores=4, tune=int(nsample/20),
                discard_tuned_samples=True, step=pm.Metropolis())

        return trace