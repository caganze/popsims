
################################
#population simulations routines
#includes copied functions from splat
##############################
import numpy as np
import numba
import scipy

def galactic_density(r, z, h, l):
	"""
	expoential galactic density porfile
    """
    zpart=np.exp(-abs(z-Zsun)/h)
    rpart=np.exp(-(r-Rsun)/l)
    return zpart*rpart
