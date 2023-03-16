
################################
# test binaries.py functions 
##############################
import pandas as pd
def test_binaries():
    from popsims.binaries import BINARIES 
    #must be a pickle file with columns prim, sec, sys all floats
    tbl=pd.DataFrame(BINARIES)

    assert len(tbl) >0