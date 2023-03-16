
################################
# test popsims.py functions
##############################
import pandas as pd
from popsims import evolutionary_model_interpolator
import numpy as np

def test_evolutionary_model_interpolator():
	mass= np.random.uniform(0.01, 0.1, 10)
	age= np.random.uniform(0.01, 13, 10)
	vals=evolutionary_model_interpolator( mass, age, 'saumon2008',  subset=('cloud', 'hybrid'))
	teffs=vals['temperature'].value
	teffs=teffs[~np.isnan(teffs)]
	assert (( teffs< 5000).all())