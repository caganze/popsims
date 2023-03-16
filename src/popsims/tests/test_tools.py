
################################
# test core_tools.py functions 
##############################

from popsims.core_tools import *
import numpy as np
import pytest

def test_sample_from_powerlaw():
	x = sample_from_powerlaw(3, xmin=0, xmax=2, nsample=int(1e3))
	assert np.trapz(np.linspace(0, 2, 1000), x)< (2.5**4/4)

def test_random_draw():
	x = np.arange(0, 10)
	cdf = x**3/(x[-1]**3)
	res= random_draw(x, cdf)
	assert len(x[x>5])< len(x[x<5])

def test_make_spt_number():
	assert make_spt_number('L0.0')==20

def test_get_distance():
	assert get_distance(10, 20)==1000.

def test_trapzl():
	x = np.arange(0, 10)
	y = np.ones_like(x)
	res= trapzl(y, x)
	assert np.isclose(res, 9, rtol=1e-5)

@pytest.mark.skip(reason="example not implemented in docstring")
def test_group_by():
	pass

@pytest.mark.skip(reason="example not implemented in docstring")
def test_k_clip_fit():
	pass

@pytest.mark.skip(reason="example not implemented in docstring")
def apply_polynomial_relation():
	pass

@pytest.mark.skip(reason="example not implemented in docstring")
def test_inverse_polynomial_relation():
	pass
    