#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import unicode_literals
from setuptools import setup, find_packages
from popsims import __version__

name = "popsims"
version = __version__
setup(
  name=name,
  url= "https://github.com/caganze/popsims.git",
  version=version,
  packages = find_packages(exclude=['docs','tests'], include=['data']),

  # Project uses reStructuredText, so ensure that the docutils get
  # installed or upgraded on the target machine
  install_requires = [
  'astropy', 
  'gala',
  'matplotlib',
  'numba',
  'numpy', 
  'pandas==1.3.4',
  'scikit-learn',
  'scipy',
  'seaborn', 
  'easyshapey',
  'tqdm',
  'scipy'],

  package_dir = {'popsims': 'popsims'},    
  package_data = {'popsims': ['docs/*','data/*','build/*']},
  include_package_data=True,

  zip_safe = True,
  use_2to3 = False,
  classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Astronomy',
  ],

  # metadata for upload to PyPI
  author = "Christian Aganze",
  author_email = "caganze@ucsd.edu",
  description = "UCD Population Simulations",
  license = "MIT",
  keywords = ['astronomy','astrophysics','ultracool dwarfs','low mass stars', 'brown dwarfs', 'monte-carlo']

  )