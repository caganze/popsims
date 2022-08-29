################################
#population simulation routines for thin disk, 
#thick and halo populations 
##############################
import numpy as np
import numba
import scipy
from .tools import random_draw
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.coordinates as astro_coord
import astropy.units as u
import warnings
import collections
from tqdm import tqdm
from abc import ABCMeta
from abc import abstractproperty, abstractmethod
from functools import reduce
import gala.coordinates as gc
import gala.dynamics as gd

Rsun=8300.
Zsun=27.

#sharma coordinate frame https://www.galah-survey.org/dr3/the_catalogues/#ages-masses-distances-and-other-parameters-estimated-by-bstep
v_sun = astro_coord.CartesianDifferential([11.1, 248., 7.25]*u.km/u.s) #almost the same as my coordinate 
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)

@numba.vectorize("float64(float64, float64)", target='cpu')
def get_distance(absmag, appmag):
    """
    Distance from absolute magnitude and apparent magnitude
    Args:
    ----
        absmag: absolute magnitude (float or array)
        app: apparent magnitude (float or array)
    Returns:
    -------
        distance (float or array)

    Examples:
    --------

    """
    return 10.**(-(absmag-rel_mag)/5. + 1.)

@numba.jit(nopython=True)
def trapzl(y, x):
    """
    Fast trapezoidal integration

    Args:
    ----
        y: y values on a grid  (list or array)
        x: x values on a grid  (list or array)
    Returns:
    -------
        distance (float or array)

    Examples:
    --------
        > x = np.arange(0, 10)
        > y = np.ones_like(x)
        > res= trapzl(y, x)

    """
    s = 0
    for i in range(1, len(x)):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2

def exponential_density(r, z, H,L):
    """
    Exponential density profile

    Args:
    ----
        r: galactocentric radius  (list or array)
        z: galactocentric height  (list or array)
        H: scaleheight (list or array)
        L: scalelength (list or array)
    Returns:
    -------
        density (float or array)

    Examples:
    --------
        > x = exponential_density(8700, 100, 300,2600)

    """
    zpart=np.exp(-abs(z-Zsun)/H)
    rpart=np.exp(-(r-Rsun)/L)
    return zpart*rpart

def spheroid_density(r, z, q, n):
    """
    sperhoid density profile

    Args:
    ----
        r: galactocentric radius  (list or array)
        z: galactocentric height  (list or array)
        q: flattening parameter
        n: power-law exponent
    Returns:
    -------
        density (float or array)

    Examples:
    --------
        > x = spheroid_density(8700, 100, 0.64, n=2.77)

    """
    return  (Rsun/(((r)**2+((z)/q)**2)**0.5))**n

def transform_tocylindrical(l, b, ds):
    """
    sperhoid density profile

    Args:
    ----
        r: galactocentric radius  (list or array)
        z: galactocentric height  (list or array)
        q: flattening parameter
        n: power-law exponent
    Returns:
    -------
        density (float or array)

    Examples:
    --------
        > x = spheroid_density(8700, 100, 0.64, n=2.77)

    """
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    return (rd, zd)

class GalacticComponent(object):
    """
    A meta class for galactic components

    """
    __metaclass__=ABCMeta
    def __init__(self, parameters):
        #initialize from dictionary
        for key in parameters.keys():
            setattr(self, key, parameters[key]) 

    def __add__(self, other):
        #defining method for addition by adding the stellar density
        old_dens= self.stellar_density
        new_dens= other.stellar_density
        all_parameters= {**self.__dict__, **other.__dict__}
        new= GalacticComponent(all_parameters)
        new.stellar_density= self._combine_densities(old_dens, new_dens)
        return new

    def __radd__(self, other):
        return self.__add__(other)
    
    #overload multiplication
    def __mul__(self, number):
        new= GalacticComponent(self.__dict__)
        new.stellar_density= lambda r, z: number*self.stellar_density(r, z)
        return new
    
    def __rmul__(self, number):
        return self.__mul__(number)

    @abstractmethod
    def stellar_density(self, r, z):
        pass

    @staticmethod
    def _combine_densities(*fs):
        def compose(f, g):
            return lambda r, z : f(r, z)+ g(r, z)
        return reduce(compose, fs)

    def sample_distances(self, dmin, dmax, nsample, dsteps=200, l=None, b=None):
        """
        Draw distances from a likelihood d^2\rho(r, z) by inverse-sampling

        Args:
        ----
            dmin: minium of the distance(astropy quantity)
            dmax: maximum distance (astropy quantity)
            nsample: number of draws
            l, b: (optional), galactic latitudes (astropy quantities).
                 if set to None, will randomly pick directions
            dsteps: (optional): number of steps in trapezoidal integration (int)
        Returns:
        -------
            distances: array of distances (astropy quantity)

        Examples:
        --------
            > d = GalacticComponent.sample_distances(10*u.pc, 1000*u.pc, 1000 )

        """
        #if l and b are none sample from random direction using sphere-point picking 
        if l is None:
            l= 2*np.pi*np.random.uniform(0, 1)
            b= np.arccos(2*np.random.uniform(0, 1)-1)-np.pi/2

        d=np.logspace(np.log10(dmin), np.log10(dmax),dsteps)
        #compute r and z
        cdf=np.array([self.volume( l, b, d[0], dx, dsteps=dsteps) for dx in d])
        cdf[0]=cdf[1] #avoid zero at the start of the array
        #interpolate over cdf to get a smoother function
        f=interp1d(d, cdf)
        #increase the resolution
        d=np.logspace(np.log10(dmin), np.log10(dmax),int(nsample))
        cdfvals=f(d)
        return random_draw(d, cdfvals/np.nanmax(cdfvals), int(nsample))

    def volume(self, l, b, dmin, dmax, dsteps=1000):
        """
        Compute volume by integrating d^2\rho(r, z) by inverse-sampling

        Args:
        ----
            dmin: minium of the distance(astropy quantity)
            dmax: maximum distance (astropy quantity)
            l, b: galactic latitude and longitude (astropy quantities).
            dsteps: (optional): number of steps in trapezoidal integration (int)
        Returns:
        -------
            distances: array of distances (astropy quantity)

        Examples:
        --------
            > d = GalacticComponent.sample_distances(10*u.pc, 1000*u.pc, 1000 )

        """
        ds = np.logspace(np.log10(dmin), np.log10(dmax),dsteps)
        rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
        zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
        rh0=self.stellar_density( rd, zd)
        #val=integrate.trapz(rh0*(ds**2), x=ds)
        val= trapzl(rh0*(ds**2), ds)
        return val

    def density_gradient(self):
        """"
        Uses jax autodiff to get the gradient of the density

        """
        pass

    @staticmethod
    def from_gala_potential():
        """
        Initialize galactic density model from gala potential object
        """
        pass

class Disk(GalacticComponent):
    def __init__(self, H=300, L=2600):
        super().__init__({'H': H, 'L': L})

    def stellar_density(self, r, z):
        """
        Compute the stellar density at a particular position

        Args:
        ----
            r: galacto-centric radius ( astropy.quantity )
            z: galacto-centric height (astropy.quantity )

        Returns:
        -------
            unit-less stellar density

        Examples:
        --------
            > d = Disk.stellar_density(100*u.pc, -100*u.pc)
        """
        #add a raise error if r <0

        return exponential_density(r, z, self.H, self.L)

class Halo(GalacticComponent):
    def __init__(self, q= 0.64, n=2.77):
        super().__init__({'q': q, 'n': n})
    def stellar_density(self, r, z):
        """
        Compute the stellar density at a particular position

        Args:
        ----
            r: galacto-centric radius ( astropy.quantity )
            z: galacto-centric height (astropy.quantity )

        Returns:
        -------
            unit-less stellar density

        Examples:
        --------
            > d = Disk.stellar_density(100*u.pc, -100*u.pc)
        """
        return spheroid_density(r, z, self.q, self.n)


def get_velocities(ra, dec, d, population='thin_disk', age=None):
    """
       Draw velocities from a Gaussians assuming a velocity dispersion

        Args:
        ----
            ra, dec: right ascenscion and declination ( astropy.quantity )
                     used to compute sky motions

            age (optional): age distribution used for thin disk 

            d (optional): distance (astropy.quantity
                            used for binning halo stars )
            population: string

                "thin_disk" Aumer & Binney
                "thick_disk": Bensby et al. 2013
                "halo": carollo et al. 2007

        Returns:
        -------
           a dictionaries with galaxy kinematics (UVW, vr, vphi, vz) and sky motions

        Examples:
        --------
            > 
    """
    vels={}
    #CHECK THAT ALL RA, DEC, D, AGE ARE THE SAME SIZE
    s= SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=d*u.pc )
    r, z= transform_tocylindrical(s.galactic.l.radian, s.galactic.b.radian, d)

    vels['r']=r
    vels['z']= z
    vels['l']=s.galactic.l.radian
    vels['b']=s.galactic.b.radian

    if population=='thin_disk':
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

        vels['U']=us
        vels['V']=vs
        vels['W']=ws

        #compute sky coordinates
        #compute_pm_from_uvw(ra_J2000, dec_J2000, parallax, us, vs, ws, correct_lsr=False):
        prop=get_proper_motion_from_uvw(ra, dec, d, us, vs, ws)          
        for k in prop.keys(): vels[k]= prop[k]

        #add vr, vphi, vz
        cylind= get_vrphiz_from_radec_distance(ra, dec, d, vels['mu_alpha_cosdec'],\
         vels['mu_delta'], vels['RV'])
        for k in cylind.keys(): vels[k]= cylind[k]

    if population=='halo':
        #values from carollo et al. not exactly right but close
        #these are actually vr, vphi, vz
        #vr=np.random.normal(loc=3, scale=150, size=len(age))
        #vphi=np.random.normal(loc=7, scale=95, size=len(age))
        #vz=np.random.normal(loc=3, scale=85, size=len(age))
        #vels={'Vr': vr, 'Vphi':vphi,  'Vz':vz }

        #compute sky coordinates
        #compure r and z 
        #prop= get_proper_motion_cylindrical(ra,dec, d, vr, vphi/(r*u.pc.to(u.km)), vz)
        #for k in prop.keys(): vels[k]= prop[k]
        #uvw=get_uvw_from_radec_distance(ra, dec, d,  vels['mu_alpha_cosdec'],\
        # vels['mu_delta'], vels['RV'])
        #for k in uvw.keys(): vels[k]= uvw[k]

        #compute vr and vphi for different abs z

        abs_z= np.abs(z)

        us, vs, ws= np.empty_like(z), np.empty_like(z), np.empty_like(z)

        #I'm also averaging over metallicities, future version should be different
        bool0=abs_z <=1000
        us[bool0]= np.random.normal(loc=-2.5, scale=58, size=len(z[bool0]))
        vs[bool0]= np.random.normal(loc=-31, scale=52.5, size=len(z[bool0]))
        ws[bool0]= np.random.normal(loc=2, scale=38, size=len(z[bool0]))

        bool1= np.logical_and(abs_z >=1000, abs_z <=2000)
        us[bool1]= np.random.normal(loc=-10, scale=136, size=len(z[bool1]))
        vs[bool1]= np.random.normal(loc=-181, scale=105, size=len(z[bool1]))
        ws[bool1]= np.random.normal(loc=-4, scale=78, size=len(z[bool1]))


        bool2= np.logical_and(abs_z >=2000, abs_z <=3000)
        us[bool2]= np.random.normal(loc=-29, scale=163, size=len(z[bool2]))
        vs[bool2]= np.random.normal(loc=-209, scale=121, size=len(z[bool2]))
        ws[bool2]= np.random.normal(loc=4, scale=95, size=len(z[bool2]))

        bool3= abs_z >=3000
        us[bool3]= np.random.normal(loc=-37, scale=152, size=len(z[bool3]))
        vs[bool3]= np.random.normal(loc=-237, scale=136, size=len(z[bool3]))
        ws[bool3]= np.random.normal(loc=2, scale=109, size=len(z[bool3]))

        #vels={'U': us, 'V':vs,  'W':ws }
        vels['U']=us
        vels['V']=vs
        vels['W']=ws
        #compute sky coordinates
        prop=get_proper_motion_from_uvw(ra, dec, d, us, vs, ws)
        for k in prop.keys(): vels[k]= prop[k]

        #add vr, vphi, vz
        cylind= get_vrphiz_from_radec_distance(ra, dec, d, vels['mu_alpha_cosdec'],\
         vels['mu_delta'], vels['RV'])
        for k in cylind.keys(): vels[k]= cylind[k]
        
    if population=='thick_disk':
        #use Bensby et al
        v_assym=46
        uvw_lsr=[0, 0, 0]
        us=np.random.normal(loc=uvw_lsr[0], scale=67,size=len(age))
        vs=np.random.normal(loc=uvw_lsr[1]-v_assym, scale=38,size=len(age))
        ws=np.random.normal(loc=uvw_lsr[-1], scale=35,size=len(age))
        #vels={'U': us, 'V':vs,  'W':ws }
        vels['U']=us
        vels['V']=vs
        vels['W']=ws

        #compute sky coordinates
        prop=get_proper_motion_from_uvw(ra, dec, d, us, vs, ws)
        for k in prop.keys(): vels[k]= prop[k]

        #add vr, vphi, vz
        cylind= get_vrphiz_from_radec_distance(ra, dec, d, vels['mu_alpha_cosdec'],\
         vels['mu_delta'], vels['RV'])
        for k in cylind.keys(): vels[k]= cylind[k]

    return  pd.DataFrame(vels)



def get_proper_motion_from_uvw(ra, dec, d, U, V, W):
    #ra in degree
    #dec in degree
    #d in parsec
    #UVW in km/s
    s=SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=d*u.pc).transform_to( astro_coord.Galactic)
    
    
    #this is centered around the sun
    c= astro_coord.Galactic(u=s.cartesian.x,
                            v= s.cartesian.y, 
                            w=s.cartesian.z,
                            U=U*u.km/u.s,
                            V=V*u.km/u.s,
                            W=W*u.km/u.s, representation_type= 'cartesian')
    #transform to sky 
    cx=c.transform_to(astro_coord.ICRS)
    
    return {'RV': cx.radial_velocity.to(u.km/u.s).value ,\
            'mu_alpha_cosdec':(cx.pm_ra_cosdec).to(u.mas/u.yr).value,\
            'mu_delta': cx.pm_dec.to(u.mas/u.yr).value}

def get_proper_motion_cylindrical(ra,dec, d, vr, vphi, vz):
    #ra in degree
    #dec in degree
    #d in pc
    #vr in km/s
    #vphi in rad/s
    #vz in km/s
    c=astro_coord.CylindricalDifferential(d_rho=vr*u.km/u.s,\
                                      d_phi=(vphi*u.rad/u.s).to(u.deg/u.s),\
                                      d_z=vz*u.km/u.s)

    co=astro_coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, \
                       distance=d*u.pc).transform_to(galcen_frame ).cylindrical
    
    c.to_cartesian(co)
    co.to_cartesian()
    xyz = astro_coord.SkyCoord(x=co.to_cartesian().x, 
                               y=co.to_cartesian().y, \
                               z=co.to_cartesian().z, frame=galcen_frame)
    vxyz = [c.to_cartesian(co).x.to(u.km/u.s).value,\
     c.to_cartesian(co).y.to(u.km/u.s).value, \
     c.to_cartesian(co).z.to(u.km/u.s).value]*u.km/u.s
    
    w = gd.PhaseSpacePosition(pos=xyz.cartesian.xyz, vel=vxyz)
    gal_c = w.to_coord_frame(astro_coord.ICRS)
    
    return {'RV': gal_c.radial_velocity.to(u.km/u.s).value ,\
                      'mu_alpha_cosdec':(gal_c.pm_ra_cosdec).to(u.mas/u.yr).value,\
                      'mu_delta': gal_c.pm_dec.to(u.mas/u.yr).value}

def get_vrphiz_from_radec_distance(ra, dec, distance, pmra_cosdec, pmdec, rv):
    #ra: in degree
    #dec : in degree
    #distance: in pc
    #pmracosdec : pmra in pc
    #pm_dec: pm dec/
    #rv in km/s
    #returns vr, vphi, vz in km/s
    c= astro_coord.ICRS(ra=ra*u.degree,dec=dec*u.degree,
                  distance=distance*u.pc,
                  pm_ra_cosdec=pmra_cosdec*u.mas/u.yr,
                  pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=rv*u.km/u.s)
    cg= c.transform_to(galcen_frame)
    cg.representation_type= 'cylindrical'
    
    return  {'Vr': cg.d_rho.to(u.km/u.s).value, \
     'Vphi': (cg.d_phi*cg.rho).to(u.km/u.s, equivalencies=u.dimensionless_angles()).value,\
     'Vz':cg.d_z.to(u.km/u.s).value}
    
def get_uvw_from_radec_distance(ra, dec, distance, pmra_cosdec, pmdec, rv):
    #ra: in degree
    #dec : in degree
    #distance: in pc
    #pmracosdec : pmra in pc
    #pm_dec: pm dec
    #rv in km/s
    #returns U V W in km/s
    c= astro_coord.ICRS(ra=ra*u.degree,dec=dec*u.degree,
                  distance=distance*u.pc,
                  pm_ra_cosdec=pmra_cosdec*u.mas/u.yr,
                  pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=rv*u.km/u.s)
    
    cg= c.transform_to(galcen_frame).transform_to(astro_coord.Galactic)
    cg.representation_type= 'cartesian'
    
    return {'U': cg.U.to(u.km/u.s).value, 'V':cg.V.to(u.km/u.s).value, \
            'W':cg.W.to(u.km/u.s).value}


   
def avr_aumer(sigma,  direction='vertical', verbose=False):
    """
    """
    verboseprint = print if verbose else lambda *a, **k: None
    result=None
    beta_dict={'radial': [0.307, 0.001, 41.899],
                'total': [ 0.385, 0.261, 57.15747],
                'azimuthal':[0.430, 0.715, 28.823],
                'vertical':[0.445, 0.001, 23.831],
                }

    verboseprint("Assuming Aumer & Binney 2009 Metal-Rich Fits and {} velocity ".format(direction))

    beta, tau1, sigma10=beta_dict[direction]
       
    result=((sigma/sigma10)**(1/beta))*(10+tau1)-tau1

    return result

def avr_yu(sigma, verbose=False, disk='thin', direction='vertical', height='above', nsample=1e4):
    """
    """
    verboseprint = print if verbose else lambda *a, **k: None
    #the dictionary has thin disk and thick disk
    #thin disk  AVR is for [Fe<H] <-0.2 and two different fits for 
    #|z| > 270 pc and |z|<270
    _, tau1, sigma10= 0.385, 0.261, 57.15747
    
    beta_dict={'thin':{'vertical': [[0.54, 0.13], [0.48, 0.14]],
              'azimuthal':[[0.30, 0.09],[0.4, 0.12]],
              'radial': [ [0.28, 0.08], [0.36, 0.28]]},
               'thick':{'vertical': [[0.56, 0.14], [0.51, 0.15]],
              'azimuthal':[[0.34, 0.12],[0.42, 0.14]],
              'radial': [ [0.34, 0.17], [0.39, 0.13]]}}
    
    beta=beta_dict[disk][direction][0]
    if  height=='below':
         beta=beta_dict[disk][direction][1]
    if height=='median':
        vals=np.array([beta_dict[disk][direction][0], beta_dict[disk][direction][1]])
        beta=[(vals[:,0]).mean(), (vals[:,1]**2).sum()**0.5]
    verboseprint("Assuming Yu & Liu 2018, {} disk {} velocities ".format(disk, direction))
    if np.isscalar(sigma):
        betas=(np.random.normal(beta[0], beta[-1], int(nsample)))
        #sigmas= sigma**(np.random.normal(beta[0], beta[-1], 10000))
        #sigmas=((sigma/sigma10)**(1/betas))*(10+tau1)-tau1
        sigmas= sigma**(betas)
        return np.nanmedian(sigmas), np.nanstd(sigmas)
    else:
        betas=(np.random.normal(beta[0], beta[-1], (int(nsample), len(sigma))))
        #sigmas= sigma**(np.random.normal(beta[0], beta[-1], 10000))
        #sigmas=((sigma/sigma10)**(1/betas))*(10+tau1)-tau1
        sigmas= sigma**(betas)
        #sigmas= sigma**(np.random.normal(beta[0], beta[-1], (10000, len(sigma))))
        return np.vstack([np.nanmedian(sigmas, axis=0), np.nanstd(sigmas, axis=0)])

def avr_sanders(sigma, verbose=False, direction='vertical'):
    """
    """
    #return the age from an age-velocity dispersion 
    verboseprint = print if verbose else lambda *a, **k: None
    beta_dict={'radial': 0.3, 'vertical': 0.4}
    beta=beta_dict[direction]
    verboseprint("Assuming Sanders et al. 2018 Power for  velocity {}".format(direction))
    return sigma**(beta)



#compute age velocity dispersion relations, compare to sharma
def avr_sharma(sigma,  direction='vertical',  z=None, met=None, verbose=False, nsample=1000):
    verboseprint = print if verbose else lambda *a, **k: None
    result=None
    sigma=np.array(sigma).flatten()
    
    beta_dict={'beta':{'radial': [(0.251, 0.006), 0.1,  (39.4, 0.3)],
                'vertical':[(0.441, 0.007), 0.1, (21.1, 0.2)]},
                'gamma_z': {'vertical': (0.2, 0.01), 'radial': (0.12, 0.01)},
                'gamma_met': {'vertical': (-0.52, 0.01), 'radial': (-0.12, 0.01)}
                }

    limits_of_validity= {'vertical':{'sigmav':[0, 22] , 'z': [0, 2.1], 'met': [-1, 0]},
                        'radial': {'sigmav': [0, 40], 'z': [0, 2.1], 'met': [-1, 0]}}
    
    limits=limits_of_validity[direction]
    verboseprint("Assuming Sharma et al. 2021 Metal-Rich Fits and {} velocity valid  {} ".format(direction, limits ))
                            
    #propagate uncertainties via monte-carlo
    beta, tau1, sigma10=beta_dict['beta'][direction]
    gamma_z= beta_dict['gamma_z'][direction]
    gamma_met= beta_dict['gamma_met'][direction]
   
    
    #case for floats
    if sigma.size <1:
        return np.array([[], []])
    
    if sigma.size==1:
        #make it an array to avoid repeating stuff
        sigma=np.concatenate([sigma, sigma]).flatten()
        z= np.array([z, z])
        met=np.array([met, met])

    
    #case for arrays
    if sigma.size >1:
        beta_norm= np.random.normal(*beta, (int(nsample), len(sigma)))
        sigma10_norm= np.random.normal(*sigma10, (int(nsample), len(sigma)))
        gamma_z_norm= np.random.normal(*gamma_z, (int(nsample), len(sigma)))
        gamma_met_norm= np.random.normal(*gamma_met, (int(nsample), len(sigma)))
        
        #truncate based on limits
        bools=np.logical_and.reduce([
            np.logical_and(sigma >=limits['sigmav'][0],sigma <=limits['sigmav'][-1]),
            np.logical_and( met >=limits['met'][0], met <=limits['met'][-1]),
            np.logical_and(z >=limits['z'][0], z <=limits['z'][-1])]).flatten().astype(int).astype(float)
        
        #replace false by nans
        bools[bools==0]=np.nan
        
        fz= (1+gamma_z_norm*np.abs(z*bools))
        fmet= (1+gamma_met_norm*met*bools)
        result=((sigma*bools/ (sigma10_norm*fz*fmet))**(1/ beta_norm))*(10+tau1)-tau1
        
        return np.nanmedian(result, axis=0), np.nanstd(result, axis=0) 

def avr_just(sigma, verbose=False, direction='vertical'):
    """
    """
    #return the age from an age-velocity dispersion 
    verboseprint = print if verbose else lambda *a, **k: None
    beta_dict={'radial': None, 'vertical': 0.375, 'azimuthal': None}
    beta=beta_dict[direction]
    verboseprint("Just et al. 2010 power law for  velocity {}".format(direction))
    sigma0, t0, tp, alpha=(25, 0.17, 12, 0.375)
    return ((sigma/sigma0)**(1/alpha))*(tp+t0)-t0

def scaleheight_to_vertical_disp(hs):
    """
    """
    shape=277 #shape parameter
    sigma_68=1.
    return np.sqrt((np.array(hs))/shape)*20