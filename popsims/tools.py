import splat
import numpy as np
import numba
import bisect

@np.vectorize      
def teff_to_spt(teff):
    rel=splat.SPT_TEFF_RELATIONS['pecaut']
    spt_sorted_idx=np.argsort(rel['values'])
    scatter=108
    teffsc=np.random.normal(teff, scatter)
    return np.interp(teffsc, np.array(rel['values'])[spt_sorted_idx], np.array(rel['spt'])[spt_sorted_idx])

@numba.jit
def make_spt_number(spt):
    ##make a spt a number
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

def dropnans(x):
    return x[~np.isnan(x)]


def random_draw(xvals, cdfvals, nsample=10):
    """
    randomly drawing from a discrete distribution
    """
    @numba.vectorize("int32(float64)")
    def invert_cdf(i):
        return bisect.bisect(cdfvals, i)-1
    x=np.random.rand(nsample)
    idx=invert_cdf(x)
    return np.array(xvals)[idx]

