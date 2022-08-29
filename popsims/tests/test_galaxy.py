
################################
# test galaxy.py functions 
##############################
import popsims
from popsims.galaxy import *
import numpy as np
import pytest

def test_trapzl():
	x = np.arange(0, 10)
	y = np.ones_like(x)
	res= trapzl(y, x)
	assert np.isclose(res, 9, rtol=1e-1)

def test_exponential_density():
	assert exponential_density(8300, 27, 300,2600) ==1.0

def test_spheroid_density():
	assert spheroid_density(8300, 0, 1,1) ==1.0

def test_transform_tocylindrical():
	assert transform_tocylindrical(0, 0, 8300)[0]== 0.0

def test_galactic_component():
	g=Disk()
	g2= Halo()
	g3= 0.1*g+0.2*g2
	assert g3 != None