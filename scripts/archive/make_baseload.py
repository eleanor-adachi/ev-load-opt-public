# -*- coding: utf-8 -*-
"""
make_baseload.py

Created on Mon Oct  7 13:07:25 2024

@author: elean
"""
# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np

# Inputs
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'

def make_baseload(ica_file, save=True):
    '''
    Creates month-hour profile for baseload based on ICA data
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
    
    # Calculate system-wide month-hour baseload profiles
    baseload = circload.groupby(['month', 'hour']).agg(
        {'l_kW':'sum', 
         'h_kW':'sum', 
         'feeder_id':'count'
         })
    
    # # Convert sums from kW to MW
    # baseload['l_MW'] = round(baseload['l_kW']/1000, 3)
    # baseload['h_MW'] = round(baseload['h_kW']/1000, 3)
    
    # Round sums to 3 decimal places
    baseload['l_kW'] = round(baseload['l_kW'], 3)
    baseload['h_kW'] = round(baseload['h_kW'], 3)
    
    # Merge with mh DataFrame, drop and rename columns
    baseload = pd.merge(mh, baseload, on=['month', 'hour'], how='outer')
    baseload = baseload.rename(columns={'feeder_id':'feeder_count'})
    # baseload = baseload.drop(columns=['l_kW', 'h_kW'])
    
    # Save
    if save:
        baseload.to_csv(r'..\data\baseload.csv', index=False)
    
    return baseload

#%%

if __name__ == "__main__":
    baseload = make_baseload(ica_file)