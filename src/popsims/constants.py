###
# This module contains a list of constants
##

##coordinate system
import astropy.coordinates as astro_coord
import numpy as np
import astropy.units as u

Rsun=8300.
Zsun=27.
#default coordinate frame
#sharma coordinate frame https://www.galah-survey.org/dr3/the_catalogues/#ages-masses-distances-and-other-parameters-estimated-by-bstep
v_sun = astro_coord.CartesianDifferential([11.1, 248., 7.25]*u.km/u.s) 
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)