# -*- coding: utf-8 -*-
"""
make_addload.py

Created on Fri Sep 13 15:37:23 2024

@author: elean

Create addload DataFrame combining multiple sources of load growth

"""

# Importing libraries
import os
import pandas as pd
# import add_cec2127_ev_load as cec
import add_cec2127_2_ev_load as cec2
import add_nrelefs_be_load as nrel

# Inputs
# cec2127_sc_num_ls = [1, 3, 4] # 1st Assessment
cec2127_2_sc_num_ls = [1, 2, 3] # 2nd Assessment
# nrelefs_sc_num_ls = range(2, 5)
nrelefs_sc_num_ls = [4] # only use high building electrification scenario

# Output filepath
output_dir = r'..\data'

def make_addload(ev_sc_num_ls, be_sc_num_ls, save=True):
    '''
    Create addload DataFrame combining multiple sources of load growth
    '''
    # Incremental load from electric vehicles
    # addload_ev = cec.make_addload_cec2127_ev(ev_sc_num_ls, save=False) # 1st Assessment
    addload_ev = cec2.make_addload_cec2127_2_ev(ev_sc_num_ls, save=False) # 2nd Assessment
    # Incremental load from building electrification
    addload_be = nrel.make_addload_nrelefs_be(be_sc_num_ls, save=False)
    # Combine
    addload = pd.merge(addload_ev, addload_be, how='outer', on=['feeder_id', 'month', 'hour', 'mhid'])
    
    # Optional: Save to file
    if save:
        output_path = os.path.join(output_dir, 'addload.csv')
        addload.to_csv(output_path, index=False)
    
    return addload

#%%

if __name__ == "__main__":
    # make_addload(cec2127_sc_num_ls, nrelefs_sc_num_ls, save=True) # 1st Assessment
    make_addload(cec2127_2_sc_num_ls, nrelefs_sc_num_ls, save=True) # 2nd Assessment