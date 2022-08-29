##purpose: simulate a brown population
### 
###

#imports
from .galaxy import * 
from .core import *
from .tools import *
from .relations import teff_to_spt_subdwarf

POLYNOMIALS=np.load(DATA_FOLDER+'/abs_mag_relations.npy', allow_pickle=True)[0]

class Population(object):
    def __init__(self, **kwargs):
        self.imfpower= kwargs.get('imf_power', -0.6)
        self.binaryfraction= kwargs.get('binary_fraction', 0.2)
        self.binaryq= kwargs.get('binary_q', 4)
        self.evolmodel= kwargs.get('evolmodel', 'burrows1997')
        self.metallicity=kwargs.get('metallicity', 'dwarfs')
        self.agerange= kwargs.get('age_range', [0.01, 14.])
        self.massrange= kwargs.get('mass_range', [0.01, 14.])

    def _sample_ages(self, nsample):
        return np.random.uniform(*self.agerange, int(nsample))

    def _sample_masses(self, nsample):
        return sample_from_powerlaw(self.imfpower, xmin=  self.massrange[0], xmax= self.massrange[1], nsample=int(nsample))

    def _interpolate_evolutionary_model(self, mass, age):
        return evolutionary_model_interpolator(mass, age, self.evolmodel)

    def simulate(self, nsample):
        #single stars
        m_singles=self._sample_masses(nsample)
        ages_singles= self._sample_ages(nsample)

        #binaries
        qs=sample_from_powerlaw(self.binaryq, xmin= 0., xmax=1., nsample=nsample)
        m_prims = self._sample_masses(nsample)
        m_sec=m_prims*qs
        ages_bin=self._sample_ages(nsample)

        #interpolate evolurionary models
        single_evol=self._interpolate_evolutionary_model(m_singles, ages_singles)
        primary_evol=self._interpolate_evolutionary_model(m_prims,ages_bin)
        secondary_evol=self._interpolate_evolutionary_model(m_sec,ages_bin)

        #temperatures
        teffs_singl =single_evol['temperature'].value
        teffs_primar=primary_evol['temperature'].value
        teffs_second=secondary_evol['temperature'].value

        #spectraltypes
        spts_singl=teff_to_spt_kirkpatrick(teffs_singl)
        spt_primar=teff_to_spt_kirkpatrick(teffs_primar)
        spt_second=teff_to_spt_kirkpatrick(teffs_second)

        #use pecaut for teff <2000 k
        spts_singl[teffs_singl >2000]= teff_to_spt_pecaut(teffs_singl[teffs_singl>2000])
        spt_primar[teffs_primar >2000]= teff_to_spt_pecaut(teffs_primar[teffs_primar>2000])
        spt_second[teffs_second>2000]= teff_to_spt_pecaut(teffs_second[teffs_second>2000])


        #compute combined binary spectral types
        xy=np.vstack([np.round(np.array(spt_primar), decimals=0), np.round(np.array(spt_second), decimals=0)]).T
        spt_binr=get_system_type(xy[:,0], xy[:,1], read_bintemplates())


        values={ 'sing_evol': single_evol, 'sing_spt':spts_singl,
                     'prim_evol': primary_evol, 'prim_spt':spt_primar,
                     'sec_evol': secondary_evol, 'sec_spt': spt_second,
                    'binary_spt': spt_binr }

        #make systems
        return make_systems(values, self.binaryfraction)


def make_systems(mods, bfraction):
    """
    """
    
    #singles
    singles=mods['sing_evol']
    singles['is_binary']= np.zeros_like(mods['sing_spt']).astype(bool)
    singles['spt']=mods['sing_spt']
    singles['prim_spt']=mods['sing_spt']
    singles['sec_spt']=np.ones_like(mods['sing_spt'])*np.nan

    #print (np.isnan(singles['temperature']).all())
    
    #binary
    binaries={}
    binaries['age']=mods['prim_evol']['age']
    binaries['mass']=mods['prim_evol']['mass']+mods['sec_evol']['mass']
    binaries['pri_mass']=mods['prim_evol']['mass']
    binaries['sec_mass']=mods['sec_evol']['mass']
    
    binaries['luminosity']=np.log10(10**(mods['prim_evol']['luminosity']).value+\
    10**(mods['sec_evol']['luminosity']).value)
    #binaries['temperature']=mods['prim_evol']['temperature']
    binaries['spt']=np.random.normal(mods['binary_spt'], 0.3)
    binaries['prim_spt']=mods['prim_spt']
    binaries['sec_spt']=mods['sec_spt']
    binaries['prim_luminosity']=10**(mods['prim_evol']['luminosity']).value
    binaries['sec_luminosity']=10**(mods['sec_evol']['luminosity']).value

    binaries['is_binary']=np.ones_like(mods['sec_spt']).astype(bool)

    #assign teff from absolute mag
    #binaries['temperature']=get_teff_from_mag_ignore_unc(binaries['abs_2MASS_H'])
    mask= binaries['spt'] >20.
    binaries['temperature']= np.ones_like( binaries['spt'])*np.nan
    binaries['temperature'][mask]=spt_to_teff_kirkpatrick(binaries['spt'])[0][mask]
    binaries['temperature'][~mask]=spt_to_teff_pecaut(binaries['spt'])[~mask]

    #compute numbers to choose based on binary fraction
    ndraw= int(len(mods['sing_spt'])/(1-bfraction))-int(len(mods['sing_spt']))
    #ndraw=int(len(mods['sing_spt'])* bfraction)

    
    #random list of binaries to choose
    random_int=np.random.choice(np.arange(len(binaries['spt'])), ndraw)
    
    chosen_binaries={}
    for k in binaries.keys():
        chosen_binaries[k]=binaries[k][random_int]

    #add scale to the local lf
    res=pd.concat([pd.DataFrame(singles), pd.DataFrame(chosen_binaries)])
    scl=scale_to_local_lf(res.temperature.values)
    #print (scl
    res['scale']=scl[0]
    res['scale_unc']=scl[1]
    res['scale_times_model']=scl[-1]

    #combine the to dictionaries 
    #print (np.isnan(res['temperature']).all())

    return res

def pop_mags(x, d=None, keys=[], object_type='dwarfs', get_from='spt', reference=None, pol=None):
    """
    Compute magnitudes from pre-computed absolute mag relations
    """
    res={}
    if pol is None: pol=POLYNOMIALS['absmags_{}'.format(get_from)][object_type]
    if reference is not None: pol=POLYNOMIALS['references'][reference]
    for k in keys:
        print (keys)
        #sometimes sds don't have absolute magnitudes defined 
        if k not in pol.keys():
            warnings.warn("{} relation not available for {} ".format(k,object_type))

        fit=pol[k]['fit']
        scat=0.1#pol[k]['scatter']
        #print (k, 'scatter', scat)
        rng=pol[k]['range']
        mag_key=pol[k]['y']
        offset=pol[k]['x0']
        #put constraints on spt range
        mask= np.logical_and(x >rng[0], x <=rng[-1])
        absmag= np.random.normal(fit(x-offset),scat)
        #forget about scatter for now
        #absmag= fit(x-offset)

        masked_abs_mag= np.ma.masked_array(data=absmag, mask=~mask)
        #make it nans outside the range
        if d is not None: 
            res.update({ k: masked_abs_mag.filled(np.nan)+5*np.log10(d/10.0) })
        res.update({'abs_'+ k: masked_abs_mag.filled(np.nan)})


    return pd.DataFrame(res)

def pop_colors(x, d=None, keys=[], object_type='dwarfs', get_from='spt', reference=None, pol=None):
    """
    Compute colors from pre-computed absolute mag relations
    """
    res={}
    if pol is None: pol=POLYNOMIALS['colors_{}'.format(get_from)][object_type]
    if reference is not None: pol=POLYNOMIALS['references'][reference]
    for k in keys:
        #sometimes sds don't have absolute magnitudes defined 
        if k not in pol.keys():
            warnings.warn("{} relation not available for {} ".format(k,object_type))

        #if pol[k]['method']=='polynmial':
        fit=pol[k]['fit']
        #if pol[k]['method']=='spline':
        #     fit=pol[k]['fit']

        scat=pol[k]['scatter']
        
        rng=pol[k]['range']
        mag_key=pol[k]['y']
        offset=pol[k]['x0']
        #put constraints on spt range
        mask= np.logical_and(x >rng[0], x <=rng[-1])
        absmag= np.random.normal(fit(x-offset),scat)

        masked_abs_mag= np.ma.masked_array(data=absmag, mask=~mask)

        res.update({ mag_key:masked_abs_mag.filled(np.nan)})

    return pd.DataFrame(res)



def simulate_population(disk, coord, dmin, dmax, nsample=1e5, poptype='dwarfs', galtype='thin_disk', age_range=[0.01, 8], \
                        mass_range=[0.01, 0.1], mag_keys=None, evolmodel='burrows1997', popargs={},
                        get_from='spt'):

    #simulate a full population 
    #disk = disk or halo object 
    #coord is the footprint of survey, must be astropy obejct
    #dmin, dmax= minimum and maximum distances
    # popargs= additional argument for initializing population object
    #galtype: prescription for assigning velocities

    pop=Population(evolmodel=evolmodel, age_range=age_range, mass_range=mass_range, **popargs)
    #draw samples 
    samples= pop.simulate(nsample)

    #assign spt from subdwarfs teff relations
    if poptype=='subdwarfs':
        samples['spt']=teff_to_spt_subdwarf(samples.temperature.values)
    
    #draw distances
    ds= [disk.sample_distances( dmin, dmax, int(len(samples)/len(coord)), l=sx.l.radian, b=sx.b.radian,dsteps=1000)\
         for sx in tqdm(coord.galactic)]
    
    samples['distance']=np.random.choice(np.array(ds).flatten(), len(samples), replace=True)
    samples['ra']=np.random.choice(coord.ra.degree, len(samples), replace=True)
    samples['dec']=np.random.choice(coord.dec.degree, len(samples), replace=True)
    
    #add magnitudes from spectral types for dwarfs or from teff for subdwarfs
    mags=None
    if get_from =='spt':
        mags= pop_mags(samples.spt.values, d=samples['distance'], keys=mag_keys, \
                                 object_type=poptype, get_from=get_from, reference=None)
    print (mags)
    print (mag_keys)
    #always simulate subdwarfs from temperature relation
    if get_from =='teff':
        mags= pop_mags(samples.temperature.values, d=samples['distance'], keys=mag_keys, \
                                 object_type=poptype, get_from=get_from, reference=None)

    for c in mags.columns:
        samples[c]= mags[c]
        
    #add kinematics 
    vels= get_velocities(samples['ra'].values, samples['dec'].values,\
                         samples['distance'].values, samples.age.values, population=galtype)
    for c in vels.columns:
        samples[c]= vels[c]
    
    #ADD REDUCED PROPER MOTIONS
    for k in mag_keys:
        samples['redH_{}'.format(k)]=samples['abs_{}'.format(k)]+\
         5*np.log10((samples.mu_alpha_cosdec**2+ samples.mu_delta**2)**0.5)+5

    return samples.reset_index(drop=True)

def pop_mags_from_color(color, d=None, keys=[], object_type='dwarfs'):
    """
    """
    res={}
    for k in tqdm(keys):
        pol=POLYNOMIALS['colors'][object_type]
        fit=np.poly1d(pol[k]['fit'])
        scat=pol[k]['scatter']
        rng=pol[k]['range']
        mag_key=pol[k]['y']
        absmag= np.random.normal(fit(color),scat)
        #forget about scatter
        #make it nans outside the range
        absmag[np.logical_and(color <rng[0], color <rng[-1])]=np.nan
        if d is not None: res.update({ mag_key: absmag+5*np.log10(d/10.0) })
        res.update({'abs_'+ mag_key: absmag})

    return pd.DataFrame(res)