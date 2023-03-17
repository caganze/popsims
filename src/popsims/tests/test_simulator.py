
################################
# simulator.py functions 
#############################

import popsims
from popsims.simulator import Population, pop_mags
import pandas as pd

def test_population():
    p=Population(evolmodel= 'baraffe2003',
                  imf_power= -0.6,
                  binary_fraction=0.2,
                  age_range=[0.01, 14],
                  mass_range=[0.01, 1],
                  nsample=1000)
    p.simulate()
    df=p.to_dataframe(['mass', 'age', 'temperature'])
    assert len(p.mass) ==1000
    assert len(df)==1000
    print (df)