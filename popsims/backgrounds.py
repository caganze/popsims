##purpose: simulate a background population
### only works for LSST for now :(
###

from .galaxy import * 
from .core import *
from .tools import *
from .simulator import get_velocities
from astropy.io import ascii
import glob
import scipy.interpolate as interp 
#link to isochrones
#look into this code later https://github.com/mfouesneau/ezpadova


def download_isochrones():
	pass

def read_isochrones(survey='lsst', filters=[]):
    d=ascii.read(glob.glob(DATA_FOLDER+'/*parsec_isochrones_{}.txt'.format(survey))[0]).to_pandas()
    return d


def assign_kinematics(samples, mag_keys, population):
    #add kinematics 
    #add random ras and decs
    ras= (2*np.pi*np.random.uniform(0, 1, len(samples)))*u.radian.to(u.degree)
    decs= (np.arccos(2*np.random.uniform(0, 1, len(samples))-1)-np.pi/2)*u.radian.to(u.degree)
    vels= get_velocities(ras, decs,\
                         samples['distance'].values, samples.age.values,\
                         population=population)
    #print (vels.columns)
    for c in vels.columns:
        samples[c]= vels[c]
    
    #ADD REDUCED PROPER MOTIONS
    for k in mag_keys:
        samples['redH_{}'.format(k)]=samples['abs_{}'.format(k)]+\
         5*np.log10((samples.mu_alpha_cosdec**2+ samples.mu_delta**2)**0.5)+5
        
    return samples

def assign_absmag(df, mag_keys):
    for mag_key in mag_keys:
        absmag= df['abs_{}'.format(mag_key)]
        df[mag_key]=absmag+5*np.log10(df.distance/10.0)
    return df

def assign_galactic(df, mag_keys, nsample=1e5):
    #can assign distances for various scaleheights
    #choose a proportional disk population H= 300 pc
    #choose a thick disk population H=900 pc 
    disk= Disk()
    tdisk=Disk(H=900, L=3600)
    h= Halo()
    
    #resample the data by replacement 
    #sample by age 
    #choose objects
    df_thin= df.query('age < 8').sample(n=int(nsample*0.8908), replace=True)
    df_thick=df.query('age >= 8 and age <10').sample(n= int(nsample*0.12), replace=True)
    df_halo=df.query('age >= 10').sample(n=int(nsample*0.0025), replace=True)
                       
    #sample distances in random directions
    df_thin['distance']=disk.sample_distances( 0.1, 10000, len(df_thin))
    df_thick['distance']=tdisk.sample_distances( 0.1, 10000, len(df_thick))
    df_halo['distance']=h.sample_distances( 0.1, 10000, len(df_halo))
                       
    df_thin['population']= 'thin disk'
    df_thick['population']= 'thick disk'
    df_halo['population']= 'halo'
    
    #give them magnitudes
    
    df_thin=assign_absmag(df_thin, mag_keys)
    df_thick=assign_absmag(df_thick, mag_keys)
    df_halo=assign_absmag(df_halo, mag_keys)
    
                       
    #give them kinematics
    df_thin=assign_kinematics(df_thin, mag_keys, 'thin_disk')
    df_thick=assign_kinematics(df_thick, mag_keys, 'thick_disk')
    df_halo=assign_kinematics(df_halo, mag_keys, 'halo')
    
    df_comb= pd.concat([df_thin, df_thick, df_halo])
    
    return df_comb

def sample_stars_from_isochrones(nsample, imf='kroupa', age_power=0., \
                                 mass_range=(0.01, 100), age_range=(0.01, 13), met_range=(-2, 0.5)):
    
    #read isochrones
    isos=read_isochrones(survey='lsst')

    #get masses from kroupa imf to do: implement Kroupa IMF
    if imf=='kroupa':
    	#masses= np.empty(int(nsample))
    	m0=sample_from_powerlaw(-0.3, xmin=0.03, xmax= 0.08, nsample=int(nsample))
    	m1=sample_from_powerlaw(-1.3, xmin=0.08, xmax= 0.5, nsample=int(nsample))
    	m2=sample_from_powerlaw(-2.3, xmin=0.5, xmax= 100 , nsample=int(nsample))
    	m= np.concatenate([m0, m1, m2]).flatten()
    	mask= np.logical_and(m> mass_range[0], m< mass_range[1])
    	masses= np.random.choice(m[mask], int(nsample))
    else:
    	masses= sample_from_powerlaw(imf, xmin=mass_range[0], xmax= mass_range[-1], nsample=int(nsample))
    #masses= sample_from_powerlaw(imf, xmin=mass_range[0], xmax= mass_range[-1], nsample=int(nsample))
    
    #ages
    logage= np.log10(sample_from_powerlaw(age_power, xmin=age_range[0], xmax= age_range[-1], nsample=int(nsample))*1e9)


    #metallicities
    metals= np.random.uniform(met_range[0], met_range[-1], int(nsample))
    
    us, gs, rs, imags, zs, ys =np.empty_like(logage), np.empty_like(logage), np.empty_like(logage), \
    np.empty_like(logage), np.empty_like(logage), np.empty_like(logage)
    
    for met in np.unique(isos.MH):
        for age in np.unique(isos.logAge):

            subset= isos[np.logical_and(isos.MH==met, isos.logAge == age)]

            iz= subset.imag.values-subset.zmag.values

            inter_z=  interp.interp1d(np.log10(subset.Mass.values), subset.zmag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            inter_i=  interp.interp1d(np.log10(subset.Mass.values), subset.imag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            inter_u=  interp.interp1d(np.log10(subset.Mass.values), subset.umag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            inter_g=  interp.interp1d(np.log10(subset.Mass.values), subset.gmag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            inter_r=  interp.interp1d(np.log10(subset.Mass.values), subset.rmag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            inter_y=  interp.interp1d(np.log10(subset.Mass.values), subset.ymag.values, assume_sorted = False, fill_value = np.nan, bounds_error=False)
            
            #assign in bins of 0.1 in metallicity and 0.5 on log age
            delta_age= np.abs(np.unique(isos.logAge)[0]-np.unique(isos.logAge)[1])
            delta_met= np.abs(np.unique(isos.MH)[0]-np.unique(isos.MH)[1])
            mask= np.logical_and.reduce([logage >=age,  logage <age+delta_age,\
                                        metals >= met, metals < met+delta_met])

            #draw z mags according from masseS                       
            #print (inter_u(masses)[mask], masses, met, age, logage)
            us[mask]= inter_u(np.log10(masses))[mask]
            gs[mask]= inter_g(np.log10(masses))[mask]
            rs[mask]= inter_r(np.log10(masses))[mask]
            imags[mask]= inter_i(np.log10(masses))[mask]
            zs[mask]= inter_z(np.log10(masses))[mask]
            ys[mask]= inter_y(np.log10(masses))[mask]
            #jk
                                        
    return pd.DataFrame({'mass': masses, \
                         'logage': logage,\
                         'abs_LSST_U': us,
                         'abs_LSST_G' : gs,
                         'abs_LSST_R': rs,
                         'abs_LSST_I': imags,
                         'abs_LSST_Z':  zs,
                         'abs_LSST_Y': ys})


def simulate_lsst_background_stars(nsample, imf=-0.5, age_power=0, mass_range=(0.01, 1), age_range=(0.01, 13), met_range=(-2, 0.5) ):
 	df=sample_stars_from_isochrones(nsample, imf=imf, age_power=age_power, \
                                 mass_range=mass_range, age_range=age_range, met_range=met_range)
 	df['age']=(10**df.logage)/(1e9)
 	mag_keys=['LSST_U', 'LSST_G', 'LSST_R', 'LSST_I', 'LSST_Z', 'LSST_Y']
 	df_full= assign_galactic(df, mag_keys)
 	return df_full
                                        

