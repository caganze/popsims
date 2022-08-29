
################################
#Contains scripts for generating and visualizing a brand new absolute magnitude relations
##############################
import numpy as np
from astropy.io import ascii
from popsims import DATA_FOLDER


def load_templates(dwarfs=False):
    return

def load_photometry():
    return

def load_dwarf_models():
    return 

def load_sd_models():
    return 

def get_correct_filter_name(flt):
    #reformat from by 
    f= flt.split('_')
    if f[-1]=='ps1':
        return 'PANSTARRS_'+(f[0]).upper()
    else:
        return  flt

def get_colors(sp, flt1, flt2):
    #measuring filtermags in for two filters and comparing that to target filters
    mag, mag_unc = filterMag(sp.wave, sp.flux, sp.noise, sp.flux.unit, flt1)
    #calculate the mag of the standard in J and H
    mag2, mag2_unc = filterMag(sp.wave, sp.flux, sp.noise, sp.flux.unit, flt2)
    #calculate the offset 
    return mag2-mag, (mag2_unc**2+ mag_unc**2)**0.5

def compute_colors(sp, fl_pairs):
    res={}
    for p in fl_pairs.keys():
        flt=get_correct_filter_name(fl_pairs[p]['ref'])
        res.update({p+'and'+flt: \
                    get_colors(sp, p,flt)})
    return res

def compute_color_corrections(spectra, flt_pair_dict):
    #compute color corrections
    color_corrections=[ compute_colors(sp, flt_pair_dict) for sp in tqdm(spectra)]
    return pd.DataFrame.from_records(color_corrections)

