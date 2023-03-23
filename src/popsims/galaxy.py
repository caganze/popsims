################################
#population simulation routines for thin disk, 
#thick and halo populations 
##############################

from .core_tools import random_draw, get_distance,  trapzl
from .constants import Rsun, Zsun, galcen_frame


import numpy as np
import numba
import scipy
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import pandas as pd
import astropy.coordinates as astro_coord
import astropy.units as u
import warnings
import collections
from tqdm import tqdm
from abc import ABCMeta
from abc import abstractproperty, abstractmethod
from functools import reduce
#import gala.coordinates as gc
#import gala.dynamics as gd

#@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def cylindrical_to_cartesian(r, z, phi):
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
        > x = cylindrical_to_cartesian(r, z, phi)

    """
    x= r*np.cos(phi)
    y= r*np.sin(phi)
    z= z
    return (x, y, z)

class GalacticComponent(object):
    """
    A meta class for galactic components.
    """
    __metaclass__ = ABCMeta

    def __init__(self, parameters):
        """
        Initialize a GalacticComponent object with the given parameters.

        Args:
            parameters (dict): A dictionary containing the parameters of the galactic component.
        """
        # Initialize from dictionary
        for key in parameters.keys():
            setattr(self, key, parameters[key])

    def __add__(self, other):
        """
        Add two GalacticComponent objects by combining their stellar densities.

        Args:
            other (GalacticComponent): The other GalacticComponent object to add.

        Returns:
            GalacticComponent: A new GalacticComponent object with the combined stellar densities.
        """
        # Defining method for addition by adding the stellar density
        old_dens = self.stellar_density
        new_dens = other.stellar_density
        all_parameters = {**self.__dict__, **other.__dict__}
        new = GalacticComponent(all_parameters)
        new.stellar_density = self._combine_densities(old_dens, new_dens)
        return new

    def __radd__(self, other):
        return self.__add__(other)

    # Overload multiplication
    def __mul__(self, number):
        """
        Multiply the stellar density of the GalacticComponent object by a scalar.

        Args:
            number (float): The scalar to multiply the stellar density by.

        Returns:
            GalacticComponent: A new GalacticComponent object with the scaled stellar density.
        """
        new = GalacticComponent(self.__dict__)
        new.stellar_density = lambda r, z: number * self.stellar_density(r, z)
        return new

    def __rmul__(self, number):
        return self.__mul__(number)

    @abstractmethod
    def stellar_density(self, r, z):
        """
        Abstract method for computing the stellar density of the GalacticComponent object at a given
        position (r, z).

        Args:
            r (float): The radial distance from the Galactic center, in units of distance (e.g., parsecs, meters).
            z (float): The vertical distance from the Galactic plane, in units of distance (e.g., parsecs, meters).

        Returns:
            float: The stellar density at the given position (r, z), in units of density (e.g., stars/pc^3).
        """
        pass

    @staticmethod
    def _combine_densities(*fs):
        """
        Combine the densities of multiple GalacticComponent objects.

        Args:
            *fs: A variable number of lambda functions representing the densities of the GalacticComponent objects.

        Returns:
            function: A lambda function representing the combined densities of the input GalacticComponent objects.
        """
        def compose(f, g):
            return lambda r, z: f(r, z) + g(r, z)

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
        if dmin==0: #avoid weird issues
            dmin=0.1
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
        raise NotImplementedError

    def plot_countours(self, rmin=0.5, rmax=10000, zmin=-2000, zmax=2000, npoints=1000, log=False, grid=None, cmap='cividis'):
        """
        plot contours of density in cylindrical coordinates
        """
        import matplotlib.pyplot as plt
        from .plot_style import  plot_style
        plot_style()

        rs= np.linspace(rmin,rmax, npoints)
        zs= np.linspace(zmin,zmax, npoints)

        if grid is None:
            grid = np.meshgrid(rs, zs)

        dens=self.stellar_density(grid[0], grid[1])

        if log:
            dens=np.log(self.stellar_density(grid[0], grid[1]))
        
        fig, ax=plt.subplots()

        h = plt.contourf(rs, zs, dens, cmap=cmap)
        h = plt.contour(rs, zs, dens, cmap= 'cubehelix')

        ax.set(xlabel='r (pc)', ylabel='z (pc)')
        return ax

class Uniform(GalacticComponent):
    """
    Uniform stellar density
    """
    def __init__(self, rho=1):
        super().__init__({'rho': rho})

    def stellar_density(self, r, z):
        return self.rho
    
class M31Halo(GalacticComponent):
    """
    power-law stellar density for M31's halo by Ibata et al. 2014
    """
    def __init__(self, q=1.11, gamma=-3):
        super().__init__({'q': q, 'gamma': gamma})

    def stellar_density(self, r, z):
        """
        Compute the stellar density at a particular position

        Args:
        ----
            x, y, z: galacto-centric x, y,z ( astropy.quantity )
        Returns:
        -------
            unit-less stellar density

        Examples:
        --------
            > d = Disk.stellar_density(100*u.pc, -100*u.pc)
        """
        #add a raise error if r <0
        
        s= (r**2+(z/self.q)**2)**0.5

        return s**self.gamma

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
        #values from carollo et al. 
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
    """
    Determine the proper motion of a star from its galactic velocity components.

    This function takes the position, distance, and galactic velocity components of a star and uses
    transformations to determine the proper motion of the star in the sky. The function takes the
    right ascension, declination, and distance of the star in degrees and parsecs, respectively, and
    the U, V, and W velocity components of the star in km/s. The function returns the radial velocity,
    right ascension proper motion, and declination proper motion of the star in km/s and mas/yr,
    respectively.

    Args:
        ra: The right ascension of the star, as a floating point value in degrees.
        dec: The declination of the star, as a floating point value in degrees.
        d: The distance of the star, as a floating point value in parsecs.
        U: The U velocity component of the star, as a floating point value in km/s.
        V: The V velocity component of the star, as a floating point value in km/s.
        W: The W velocity component of the star, as a floating point value in km/s.

    Returns:
        A dictionary containing the radial velocity, right ascension proper motion, and declination proper
        motion of the star, as floating point values in km/s and mas/yr, respectively.
    """
    # Function implementation goes here
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
    """
    Calculates the proper motion in right ascension and declination given right ascension, declination, distance, vr, vphi, and vz velocities.

    Copy code
    Args:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        d (float): Distance in pc.
        vr (float): Vr velocity in km/s.
        vphi (float): Vphi velocity in rad/s.
        vz (float): Vz velocity in km/s.

    Returns:
        dict: Dictionary with keys 'RV', 'mu_alpha_cosdec', and 'mu_delta' corresponding to the calculated radial velocity in km/s and proper motion in right ascension and declination in mas/yr.

    """
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
    """Calculates the vr, vphi, and vz velocities in km/s given right ascension, declination, distance, proper motion in right ascension and declination, and radial velocity in km/s.

    Copy code
    Args:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        distance (float): Distance in pc.
        pmra_cosdec (float): Proper motion in right ascension in mas/yr.
        pmdec (float): Proper motion in declination in mas/yr.
        rv (float): Radial velocity in km/s.

    Returns:
        dict: Dictionary with keys 'Vr', 'Vphi', and 'Vz' corresponding to the calculated velocities in km/s.

    """
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
    """Calculates the U, V, and W velocities in km/s given right ascension, declination, distance, proper motion in right ascension and declination, and radial velocity in km/s.

    Copy code
    Args:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        distance (float): Distance in pc.
        pmra_cosdec (float): Proper motion in right ascension in mas/yr.
        pmdec (float): Proper motion in declination in mas/yr.
        rv (float): Radial velocity in km/s.

    Returns:
        dict: Dictionary with keys 'U', 'V', and 'W' corresponding to the calculated velocities in km/s.
    """
    c= astro_coord.ICRS(ra=ra*u.degree,dec=dec*u.degree,
                  distance=distance*u.pc,
                  pm_ra_cosdec=pmra_cosdec*u.mas/u.yr,
                  pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=rv*u.km/u.s)
    
    cg= c.transform_to(galcen_frame).transform_to(astro_coord.Galactic)
    cg.representation_type= 'cartesian'
    
    return {'U': cg.U.to(u.km/u.s).value, 'V':cg.V.to(u.km/u.s).value, \
            'W':cg.W.to(u.km/u.s).value}


def avr_aumer(sigma, direction='vertical', verbose=False):

    """Calculates the Aumer & Binney 2009 metal-rich age-velocity relation.   
    Args:
        sigma (float): Sigma value to use in the calculation.
        direction (str, optional): Direction to use in the calculation. Can be 'radial', 'total', 'azimuthal', or 'vertical'. Defaults to 'vertical'.
        verbose (bool, optional): Set to True to print additional information about the calculation. Defaults to False.

    Returns:
        float: Result of the calculation.

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
    Determine the age of a population based on its velocity dispersion.

    This function takes the velocity dispersion of a population and uses the power law relation from
    Yu & Liu (2018) to determine the age of the population. The function can also take a number of
    optional parameters, such as a verbose flag to enable verbose output, a disk parameter to specify
    whether the population belongs to the thin or thick disk, a direction parameter to specify the
    direction of the velocity dispersion (radial, azimuthal, or vertical), and a height parameter to
    specify whether the population is above or below a certain height.

    Args:
        sigma: The velocity dispersion of the population, as a floating point value or array of values.
        verbose: An optional boolean flag that can be used to enable verbose output.
        disk: An optional string that specifies whether the population belongs to the thin or thick disk.
        direction: An optional string that specifies the direction of the velocity dispersion.
        height: An optional string that specifies whether the population is above or below a certain height.
        nsample: An optional integer that specifies the number of samples to use in the Monte Carlo simulation.

    Returns:
        The age of the population, as a floating point value or a tuple of two floating point values
        (mean and standard deviation) if the input sigma is an array of values.
    """
    # Function implementation goes here
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
    Determine the age of a population based on its velocity dispersion.

    This function takes the velocity dispersion of a population and uses the power law relation from
    Sanders et al. (2018) to determine the age of the population. The function can also take an optional
    verbose parameter that can be used to enable verbose output, and a direction parameter that
    specifies the direction of the velocity dispersion (radial or vertical).

    Args:
        sigma: The velocity dispersion of the population, as a floating point value.
        verbose: An optional boolean flag that can be used to enable verbose output.
        direction: An optional string that specifies the direction of the velocity dispersion.

    Returns:
        The age of the population, as a floating point value.
    """
    # Function implementation goes here
    #return the age from an age-velocity dispersion 
    verboseprint = print if verbose else lambda *a, **k: None
    beta_dict={'radial': 0.3, 'vertical': 0.4}
    beta=beta_dict[direction]
    verboseprint("Assuming Sanders et al. 2018 Power for  velocity {}".format(direction))
    return sigma**(beta)

def avr_sharma(sigma, direction='vertical', z=None, met=None, verbose=False, nsample=1000):
    """
    Determine the age of a population based on its velocity dispersion, metallicity, and position.

    This function takes the velocity dispersion, metallicity, and position of a population and uses the
    power law relation from Sharma et al. (2021) to determine the age of the population. The function can
    also take an optional verbose parameter that can be used to enable verbose output, and a direction
    parameter that specifies the direction of the velocity dispersion (radial or vertical). The function
    also supports Monte Carlo uncertainty propagation, which can be enabled by setting the nsample parameter.

    Args:
        sigma: The velocity dispersion of the population, as a floating point value or array of values.
        direction: An optional string that specifies the direction of the velocity dispersion.
        z: The position of the population in the z-direction, as a floating point value or array of values.
        met: The metallicity of the population, as a floating point value or array of values.
        verbose: An optional boolean flag that can be used to enable verbose output.
        nsample: An optional integer that specifies the number of samples to use for Monte Carlo uncertainty
            propagation.

    Returns:
        A tuple containing the median age and uncertainty of the population, as floating point values.
    """
    # Function implementation goes here

    #compute age velocity dispersion relations based on sharma et al.
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
    Determine the age of a population based on its velocity dispersion.

    This function takes the velocity dispersion of a population and uses the power law relation from
    Just et al. (2010) to determine the age of the population. The function can also take an optional
    verbose parameter that can be used to enable verbose output, and a direction parameter that
    specifies the direction of the velocity dispersion (radial, vertical, or azimuthal).

    Args:
        sigma: The velocity dispersion of the population, as a floating point value.
        verbose: An optional boolean flag that can be used to enable verbose output.
        direction: An optional string that specifies the direction of the velocity dispersion.

    Returns:
        The age of the population, as a floating point value.
    #generated with chatGPT
    """
    # Function implementation goes here
    #return the age from an age-velocity dispersion 
    verboseprint = print if verbose else lambda *a, **k: None
    beta_dict={'radial': None, 'vertical': 0.375, 'azimuthal': None}
    beta=beta_dict[direction]
    verboseprint("Just et al. 2010 power law for  velocity {}".format(direction))
    sigma0, t0, tp, alpha=(25, 0.17, 12, 0.375)
    return ((sigma/sigma0)**(1/alpha))*(tp+t0)-t0

def scaleheight_to_vertical_disp(hs):
    """
    Convert the scale height of a population to its vertical velocity dispersion.
    This function takes the scale height of a population and uses a conversion factor to determine the
    corresponding vertical velocity dispersion. The conversion factor is based on the shape parameter
    of the population and the assumption that the 68% dispersion of the population is 1.
    
    Args:
        hs (float): The scale height of the population, in units of distance (e.g., parsecs, meters).
        
    Returns:
        float: The vertical velocity dispersion of the population, in units of velocity (e.g., km/s, m/s).
    """
    # Function implementation goes here
    shape = 277  # shape parameter
    sigma_68 = 1.
    return np.sqrt((np.array(hs)) / shape) * 20