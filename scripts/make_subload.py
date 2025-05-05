# -*- coding: utf-8 -*-
"""
make_subload.py

Created on Sun Feb 23 23:52:38 2025

@author: elean

"""
# Import libraries
import geopandas as gpd
import os
import pandas as pd
import numpy as np
from make_feedload import make_feedload

# Inputs
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'
feedload_file = r'..\data\feedload.csv'


def create_sub_map(ica_file):
    '''
    Create mapping of substation ID to substation name.

    Parameters
    ----------
    ica_file : str
        File path for ICADisplay file.

    Returns
    -------
    DataFrame.

    '''
    # Create substation name-ID mapping based on FeederDetail layer
    feedinfo = gpd.read_file(ica_file, layer="FeederDetail")
    # feedinfo['SubID'] = feedinfo['FeederID'].apply(lambda x: x[:5]).astype(int)
    feedinfo['sub_id'] = feedinfo['FeederID'].apply(lambda x: x[:5]).astype(int)
    # feed_map = feedinfo[['Substation', 'SubID']]
    feed_map = feedinfo[['Substation', 'sub_id']]
    feed_map = feed_map.rename(columns={'Substation':'SubName_FeederDetail'})
    feed_map = feed_map.drop_duplicates() # necessary since original feedinfo was indexed by feeder ID; substations have unique names
    
    # Create substation name-ID mapping based on Substations layer
    subinfo = gpd.read_file(ica_file, layer="Substations")
    sub_map = subinfo[['SUBNAME', 'SUBSTATIONID']].drop_duplicates()
    # sub_map = sub_map.rename(columns={
    #     'SUBNAME': 'SubName_Substations',
    #     'SUBSTATIONID': 'SubID' # note, automatically read as int
    #     })
    sub_map = sub_map.rename(columns={
        'SUBNAME': 'SubName_Substations',
        'SUBSTATIONID': 'sub_id' # note, automatically read as int
        })
    # sub_map = sub_map.drop_duplicates(subset='SubID') # necessary because some substations have multiple names
    sub_map = sub_map.drop_duplicates(subset='sub_id') # necessary because some substations have multiple names
    
    # Merge
    # final_map = pd.merge(feed_map, sub_map, how='outer', on='SubID')
    # final_map['SubName'] = final_map.apply(
    #     lambda row: row['SubName_FeederDetail'] 
    #     if pd.notnull(row['SubName_FeederDetail']) 
    #     else row['SubName_Substations'], axis=1
    #     )
    final_map = pd.merge(feed_map, sub_map, how='outer', on='sub_id')
    final_map['sub_name'] = final_map.apply(
        lambda row: row['SubName_FeederDetail'] 
        if pd.notnull(row['SubName_FeederDetail']) 
        else row['SubName_Substations'], axis=1
        )
    final_map = final_map.drop(columns=['SubName_FeederDetail', 'SubName_Substations'])
    
    return final_map


def make_subload(ica_file, save=True):
    '''
    Creates month-hour profile for subload (baseload for each substation) based on ICA data
    '''
    # STEP 1: Create substation load profiles from ICA data, if available
    # Use ICA data as the source of "baseline" load for each substation
    subload = gpd.read_file(ica_file, layer="SubstationLoadProfile")
    subload = subload.drop(columns='geometry')
    
    # Create month-hour DataFrame
    mh = pd.DataFrame({'month': list(np.repeat(np.array(range(1,13)), 24)), 'hour': list(range(24))*12})
    mh['mhid'] = range(1, len(mh)+1)
    
    # Reformat subload
    subload['SubID'] = subload['SubID'].astype(int)
    subload['month'] = subload['MonthHour'].str[:2].astype(int)
    subload['hour'] = subload['MonthHour'].str[3:5].astype(int)
    # subload = subload.rename(columns={'Light': 'l_kW', 'High': 'h_kW_raw'})
    subload = subload.rename(columns={
        'Light':'l_kW', 'High':'h_kW_raw', 'SubName':'sub_name', 'SubID':'sub_id'
        })
    subload = subload.drop(columns=['MonthHour'])
    
    # Round load values to 3 decimal places
    subload['l_kW'] = round(subload['l_kW'], 3)
    subload['h_kW_raw'] = round(subload['h_kW_raw'], 3)
    
    # Merge with mh DataFrame, drop and rename columns
    subload = pd.merge(mh, subload, on=['month', 'hour'], how='right')
    
    # STEP 2: Aggregate feeder load profiles
    # Create feeder-substation map
    feedinfo = gpd.read_file(ica_file, layer="FeederDetail")
    feed_sub_map = feedinfo[['FeederID']]
    feed_sub_map = feed_sub_map.rename(columns={'FeederID':'feeder_id'})
    # feed_sub_map['SubID'] = feed_sub_map['feeder_id'].apply(lambda x: x[:5]).astype(int)
    feed_sub_map['sub_id'] = feed_sub_map['feeder_id'].apply(lambda x: x[:5]).astype(int)
    feed_sub_map['feeder_id'] = feed_sub_map['feeder_id'].astype(int)
    
    # Read or make feedload
    if os.path.exists(feedload_file):
        feedload = pd.read_csv(feedload_file)
    else:
        feedload = make_feedload(ica_file, save=False)
    
    # Merge feedload with feeder-substation mapping
    # Note, 38 feeders are missing a substation in the mapping
    feedload = pd.merge(feedload, feed_sub_map, how='left', on='feeder_id')    
    
    # Create subload by summing up feedload
    # subload_feedsum = feedload.groupby(['SubID','mhid'])['h_kW'].sum().reset_index()
    subload_feedsum = feedload.groupby(['sub_id','mhid'])['h_kW'].sum().reset_index()
    subload_feedsum = subload_feedsum.rename(columns={'h_kW':'h_kW_feedsum'})
    
    # Create sub_map
    sub_map = create_sub_map(ica_file)
    
    # Merge subload_feedsum, sub_map and mh    
    # subload_feedsum = pd.merge(subload_feedsum, sub_map, how='left', on='SubID')
    subload_feedsum = pd.merge(subload_feedsum, sub_map, how='left', on='sub_id')
    subload_feedsum = pd.merge(mh, subload_feedsum, on='mhid', how='right')
    
    # STEP 3: Merge and compare
    # subload = pd.merge(subload, subload_feedsum, how='outer', on=['SubID','SubName','month','hour','mhid'])
    subload = pd.merge(subload, subload_feedsum, how='outer', on=['sub_id','sub_name','month','hour','mhid'])
    subload['h_kW'] = subload.apply(
        lambda row: row['h_kW_feedsum'] if pd.isnull(row['h_kW_raw']) 
        else row['h_kW_raw'], axis=1
        )

    # Save
    if save:
        subload.to_csv(r'..\data\subload.csv', index=False)
    
    return subload

#%%

if __name__ == "__main__":
    subload = make_subload(ica_file, save=True)