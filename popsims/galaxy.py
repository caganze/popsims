
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
from scipy.interpolate import interp1d
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

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
        

def get_uvw(age, kind='dwarf',z=None):
    #velocity paremeters
    #returns simple gaussians from velocity dispersions
    
    v10 = 41.899
    tau1 = 0.001
    beta = 0.307

    v10_v = 28.823
    tau_v = 0.715
    beta_v = 0.430

    v10_w = 23.381
    tau_w = 0.001
    beta_w = 0.445

    k = 74.
    sigma_u = v10*((age+tau1)/(10.+tau1))**beta
    sigma_v =  v10_v*((age+tau_v)/(10.+tau_v))**beta_v
    sigma_w =  v10_w*((age+tau_w)/(10.+tau_w))**beta_w

    voff = -1.*(sigma_v**2)/k
    
    us=np.random.normal(loc=0, scale=sigma_u, size=len(age))
    vs =np.random.normal(loc=voff, scale=sigma_v, size=len(age))
    ws =np.random.normal(loc=0.0, scale=sigma_w, size=len(age))
    
    if kind=='subdwarf':
        us=np.zeros(len(age))
        vs=np.zeros(len(age))
        ws=np.zeros(len(age))
        
        #0-4 kpc
        bools0=np.logical_and(np.abs(z) >=0, np.abs(z)>=4000)
        bools1=np.logical_and(np.abs(z) >4000, np.abs(z)>=8000)
        bools2=np.abs(z)>8000
        
        us[bools0]=np.random.normal(loc=-52+270, scale=-242 +270, size=len(z[bools0]))
        vs[bools0]=np.random.normal(loc=-242+270, scale=-103 +270, size=len(z[bools0]))
        ws[bools0]=np.random.normal(loc=0+270, scale=67 +270, size=len(z[bools0]))
        
        us[bools1]=np.random.normal(loc=-12+270, scale=131+270, size=len(z[bools1]))
        vs[bools1]=np.random.normal(loc=-282+270, scale=-111 +270, size=len(z[bools1]))
        ws[bools1]=np.random.normal(loc=-37+270, scale=85 +270, size=len(z[bools1]))
        
        us[bools2]=np.random.normal(loc=-1+270, scale=172+270, size=len(z[bools2]))
        vs[bools2]=np.random.normal(loc=-328+270, scale=-119 +270, size=len(z[bools2]))
        ws[bools2]=np.random.normal(loc=-32+270, scale=106 +270, size=len(z[bools2]))
        
    
    return us, vs, ws

def get_magnitudes(spt, d):
    res={}
    for k in POLYNOMIALS.keys():
        absmag= np.random.normal((POLYNOMIALS[k][0])(spt), 
                                 POLYNOMIALS[k][1])
        res.update({k: absmag+5*np.log10(d/10.0) })
    return pd.DataFrame(res)

def create_population(model, coord, kind='disk'):
    #create population 
    df=pd.DataFrame()
    df['spt']=model['system_spts']
    df['teff']=model['system_teff']
    df['age']=model['system_age']
    if kind=='disk':
        p= Pointing(coord=coord, name='wfirst0')
        us, vs, ws=get_uvw(model['system_age'], kind='dwarf')
        df['d']=p.draw_distances(10, 7e4, 350, nsample=len(df.age))
    if kind=='halo':
        p= Pointing(coord=coord, density='spheroid')
        df['d']=p.draw_distances(10, 7e4, 350, nsample=len(df.age))
        crx=SkyCoord(p.coord, distance=df['d'].values*u.pc).cartesian
        us, vs, ws=get_uvw(model['system_age'], z= crx.z.value, kind='subdwarf')
    df['u']=us
    df['v']=vs
    df['w']=ws
    dff=get_magnitudes(df.spt, df.d).join(df)
    #proper motion 
    a, b= p.coord.icrs.ra.radian, p.coord.icrs.dec.radian
    T=np.matrix([[-0.06699, -0.87276, -0.48354],
    [0.49273, -0.45035, 0.74458],
    [-0.86760, -0.18837,0.46020]])

    A=np.matrix([[np.cos(a)*np.cos(b), -np.sin(a), -np.cos(a)*np.sin(b)],
       [np.sin(a)*np.cos(b) ,np.cos(a), -np.sin(a)*np.cos(b)],
       [np.sin(b), 0,np.cos(b)]])
    B= A @ T
    
    props_dfs=np.linalg.solve( B, np.vstack([us, vs, ws]))
    
    dff['rv']=props_dfs[0]
    dff['mu_alpha']=props_dfs[1]/dff.d
    dff['mu_delta']=props_dfs[-1]/(4.74057*dff.d)
    dff['vtan']=np.sqrt(4.74* (dff['mu_alpha']**2+ dff['mu_delta']**2))*dff.d
    return dff

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