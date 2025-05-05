# -*- coding: utf-8 -*-
"""
make_feedload.py

Created on Mon Nov 25 12:51:56 2024

@author: elean

Copied from feat/local
"""
# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np

# Inputs
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'

def make_feedload(ica_file, save=True):
    '''
    Creates month-hour profile for feedload (baseload for each feeder) based on ICA data
    '''
    # Use ICA data as the source of "baseline" load for each circuit
    circload = gpd.read_file(ica_file, layer="FeederLoadProfile")
    circload = circload.rename(columns={'FeederID':'feeder_id'})
    circload = circload.drop(columns='geometry')
    
    # Create month-hour DataFrame
    mh = pd.DataFrame({'month': list(np.repeat(np.array(range(1,13)), 24)), 'hour': list(range(24))*12})
    mh['mhid'] = range(1, len(mh)+1)
    
    # Reformat circuit load DataFrame
    circload['month'] = circload['MonthHour'].str[:2].astype(int)
    circload['hour'] = circload['MonthHour'].str[3:5].astype(int)
    circload = circload.rename(columns={'Light': 'l_kW', 'High': 'h_kW'})
    circload = circload.drop(columns=['MonthHour'])
    
    # Round sums to 3 decimal places
    circload['l_kW'] = round(circload['l_kW'], 3)
    circload['h_kW'] = round(circload['h_kW'], 3)
    
    # Merge with mh DataFrame, drop and rename columns
    feedload = pd.merge(mh, circload, on=['month', 'hour'], how='outer')

    # Save
    if save:
        feedload.to_csv(r'..\data\feedload.csv', index=False)
    
    return feedload

#%%

if __name__ == "__main__":
    feedload = make_feedload(ica_file)