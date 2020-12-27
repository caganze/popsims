
################################
#population simulations routines
#includes copied functions from splat
##############################
import numpy as np
import numba
import scipy

class Pointing(object):
    ## a pointing object making it easier to draw samples
    def __init__(self, **kwargs):
        #only input is the direction
        self.coord=kwargs.get('coord', None)
        self.survey=kwargs.get('survey', None)
        self.name=kwargs.get('name', None)
        self.mags={}
        self.mag_limits={}
        self.dist_limits={}
        self.volumes={}
        self.exposure_time=None
        self.exposure_times=None
        self.observation_date=None
        self.snr1=None
        self.number_of_sources={}
        self.mags_unc={}
        self.imag_exptime=None

        #compute volumes after initialization
        if self.name is not None:
            df=STARS[STARS.pointing.str.lower()==self.name.lower()]
            self.exposure_time=(df['exposure_time']).values.mean()
            self.exposure_times=(df['exposure_time']).values
            self.observation_date=(df['observation_date']).values
            self.snr1=df.snr1.values
            self.imag_exptime=np.nanmean(df.expt_f140w.values)
            for k in ['F110', 'F140', 'F160']:
                self.mags[k]=df[k].values
                self.mags_unc
                self.mag_limits[k]= get_mag_limit(self, k, self.mags[k])
                self.number_of_sources[k]= len(self.mags[k])

            del df

    def compute_volume(self, h):
        self.dist_limits=compute_distance_limits(self.mag_limits)
        for h in HS:
        	volumes={}
        if pnt.dist_limits:
            lb=[pnt.coord.galactic.l.radian,pnt.coord.galactic.b.radian]
            for k in SPGRID:
                volumes[k]= np.array(custom_volume(lb[0],lb[1],  pnt.dist_limits[k][1], pnt.dist_limits[k][0], h))
        return volumes

            self.volumes[h]=computer_volume(self, h)


def galactic_density(r, z, H, R0):
	"""
	expoential galactic density porfile
    """
    zpart=np.exp(-abs(z-Zsun)/H)
    rpart=np.exp(-(r-Rsun)/R0)
    return zpart*rpart

@numba.vectorize("float64(float64, float64)", target='cpu')
def get_distance(absmag, rel_mag):
    return 10.**(-(absmag-rel_mag)/5. + 1.)
    
def compute_volume(l,b,dmin, dmax, h):
    nsamp=1000
    ds = np.linspace(dmin,dmax,nsamp)
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    rh0=density_function(rd, zd,h=h )
    val=integrate.trapz(rh0*(ds**2), x=ds)
    return val