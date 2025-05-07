# -*- coding: utf-8 -*-
"""
add_cec2127_2_ev_load.py

Created on Fri Feb 21 15:01:56 2025

@author: elean

Adapted from EVcharging.Rmd created by Anna Brockway

Creates EV charging profiles based on California Energy Commission's 
Assembly Bill 2127 Second Electric Vehicle Charging Infrastructure Assessment

Underlying data for figure results provided by Adam Davis (Adam.Davis@energy.ca.gov)

List of scenarios:
    1. Primary/baseline (Figure 20)
    2. High home access (Figure E-3)
    3. Low home access (Figure E-4)

Key assumptions:
    - Number of chargers is proportional to amount of EV charging demand (MW)
    - Compared to 2030, EV charging demand will be 2.5x in 2040 and 4x in 2050*
    
*Growth estimated from Figure 15 in CARB's Mobile Source Strategy

"""

# Importing libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import datetime as dt

# Inputs
cec2127_2_tableresults = r'..\data\raw_data\ev\cec_ab2127_2\AB 2127 2nd Assessment Tables.xlsx'
tab_col_dict = {
    'Table D-4': 'MUD',
    'Table D-5': 'Work and Public',
    'Table D-6': 'DCFC',
    }
pge_county_ls = [
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", 
    "Contra Costa", "El Dorado", "Fresno", "Glenn", "Humboldt", "Kern", 
    "Kings", "Lake", "Lassen", "Madera", "Marin", "Mariposa", "Mendocino", 
    "Merced", "Monterey", "Napa", "Nevada", "Placer", "Plumas", "Sacramento", 
    "San Benito", "San Francisco", "San Joaquin", "San Luis Obispo", 
    "San Mateo", "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", 
    "Sierra", "Siskiyou", "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", 
    "Trinity", "Tulare", "Tuolumne", "Yolo", "Yuba"
    ]
cec2127_2_figureresults = r'..\data\raw_data\ev\cec_ab2127_2\AB2127 Load Curve Data 2nd Assessment_partial.xlsx'
sc_tab_dict = {
    1: 'Figure 20',  # primary/baseline
    2: 'Figure E-3', # higher home access
    3: 'Figure E-4', # lower home access
    }
sc_num_ls = [1, 2, 3] # selected scenarios
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'
# EV charging growth ratios
ratio_from_2030 = {
    2030: 1,
    2040: 2.5,
    2050: 4
    }

# Output filepath
output_dir = r'..\data'

#%% Get fraction of PG&E EV chargers

# 39.07% of chargers in California are expected to be located in counties served by PG&E
# based on this we will assume that 39.07% of total power demand for EV charging will occur in PG&E's territory

def calc_frac_pge(pge_county_ls):
    '''Calculate fraction of chargers in PG&E service territory'''
    # import data
    chrgrsbycty_dict = pd.read_excel(cec2127_2_tableresults, sheet_name=list(tab_col_dict.keys()), skiprows=1, engine='openpyxl')
    
    # combine into one DataFrame, using 2030 charger estimates
    chrgrsbycty = pd.DataFrame(columns=tab_col_dict.values())
    for key, value in tab_col_dict.items():
        temp_df = chrgrsbycty_dict[key].copy()
        temp_df = temp_df[['County', 2030]]
        temp_df = temp_df.rename(columns={2030: value})
        if chrgrsbycty.empty:
            chrgrsbycty = temp_df.copy()
        else:
            chrgrsbycty = pd.merge(chrgrsbycty, temp_df, how='inner', on='County')
    
    # filter for PG&E
    chrgrsbycty_PGE = chrgrsbycty[chrgrsbycty["County"].str.title().isin(pge_county_ls)]
    
    # calculate total chargers for all of California and for PG&E
    total_chrgrs = chrgrsbycty.set_index('County').sum().sum()
    total_chrgrs_PGE = chrgrsbycty_PGE.set_index('County').sum().sum()
    
    # calculate fraction of chargers in PG&E service territory
    frac_pge = total_chrgrs_PGE / total_chrgrs
    
    return frac_pge

#%% Create hourly average EV charging profiles for PG&E

# TODO: Continue
def read_cec2127_2_EVprofiles(sc_num_ls):
    '''
    Read EV profiles from CEC AB 2127 2nd Assessment and calculate residential 
    and commercial totals for each scenario.
    
    Parameters
    ----------
    sc_num_ls : list
        List of scenario numbers.

    Returns
    -------
    Dictionary of DataFrames indexed by scenario number.
    '''
    raw_EVprofiles = {}
    
    # Get EV charging profile for each scenario
    for sc_num in sc_num_ls:
        tab = sc_tab_dict[sc_num]
        df = pd.read_excel(cec2127_2_figureresults, sheet_name=tab, header=2)
        # check if time is dt.time; Python reads 24:00:00 to 25:00:00 as dt.datetime
        df['is_time'] = df['time'].apply(lambda x: isinstance(x, dt.time))
        # keep rows where is_time is True
        df = df[ df['is_time']==True ]
        # add Hour column
        df['Hour'] = df['time'].apply(lambda x: x.hour)
        # calculate EVres and EVcom
        df['EVres'] = df[['Residential Level 1', 'Residential Level 2']].sum(axis=1)
        df['EVcom'] = df[['Public and Work Level 2', 'DC Fast']].sum(axis=1)
        raw_EVprofiles[sc_num] = df
        
    return raw_EVprofiles


def make_cec2127_2_hourly_ev(sc_num_ls):
    '''
    Create hourly residential and commercial EV charging profiles for each 
    scenario based on CEC AB 2127 2nd Assessment.

    Parameters
    ----------
    sc_num_ls : list
        List of scenario numbers.

    Returns
    -------
    Dictionary of DataFrames indexed by scenario number.

    '''
    raw_EVprofiles = read_cec2127_2_EVprofiles(sc_num_ls)
    EVprofiles = {}
    
    # Make hourly EV charging profile for each scenario
    for sc_num in sc_num_ls:
        raw_profile = raw_EVprofiles[sc_num]
        df = raw_profile.groupby('Hour')[['EVres','EVcom']].mean().reset_index()
        EVprofiles[sc_num] = df
    
    return EVprofiles


def make_pge_cec2127_2_hourly_ev(sc_num_ls, ratio_from_2030, pge_county_ls, save=False):
    '''
    Creates addload DataFrame for hourly average EV charging load in PG&E's 
    service territory in 2030, 2040, and 2050 based on CEC AB 2127 2nd Assessment.
    
    Parameters
    ----------
    sc_num_ls : list
        List of scenario numbers.
    ratio_from_2030 : dict
        Dictionary of years and ratios of EV charging growth.
    pge_county_ls : list
        List of counties in PG&E service territory.

    Returns
    -------
    DataFrame of EVres and EVcom load growth for each scenario and month-hour
    '''
    # Calculate PG&E fraction of statewide EV load
    frac_pge = calc_frac_pge(pge_county_ls)
    # Process CEC AB 2127 2nd Assessment EV load profiles
    EVprofiles = make_cec2127_2_hourly_ev(sc_num_ls)
    
    # Create DataFrame for PG&E EV charging load profiles with hourly averages
    # Represents total increase in EVres and EVcom load across the service territory
    pge_hourly_ev = pd.DataFrame(columns=['sc_num', 'year', 'month', 'hour', 'ldinc_EVres_kW', 'ldinc_EVcom_kW'])
                                 
    # Iterate across scenarios, years, months
    for sc_num in sc_num_ls:
        EVprofile = EVprofiles[sc_num].copy()
        for yr in [2030, 2040, 2050]:
            for month in range(1, 13): # all 12 months (identical profile for each month)
                # get load growth ratio
                growth_ratio = ratio_from_2030[yr]
                # residential - convert from MW to kW
                pge_hourly_ev0_res = growth_ratio * frac_pge * 1000 * EVprofile.pivot_table(values='EVres', index='Hour', aggfunc='mean')
                pge_hourly_ev0_res = pge_hourly_ev0_res.reset_index()
                pge_hourly_ev0_res = pge_hourly_ev0_res.rename(columns={'Hour':'hour', 'EVres':'ldinc_EVres_kW'})
                # commercial - convert from MW to kW
                pge_hourly_ev0_com = growth_ratio * frac_pge * 1000 * EVprofile.pivot_table(values='EVcom', index='Hour', aggfunc='mean')
                pge_hourly_ev0_com = pge_hourly_ev0_com.reset_index()
                pge_hourly_ev0_com = pge_hourly_ev0_com.rename(columns={'Hour':'hour', 'EVcom':'ldinc_EVcom_kW'})
                # combine
                pge_hourly_ev0 = pd.merge(pge_hourly_ev0_res, pge_hourly_ev0_com, how='outer', on='hour')
                # add other columns
                pge_hourly_ev0['sc_num'] = sc_num
                pge_hourly_ev0['year'] = yr
                pge_hourly_ev0['month'] = month
                # reorder
                pge_hourly_ev0 = pge_hourly_ev0[ pge_hourly_ev.columns ]
                # concat
                if pge_hourly_ev.empty:
                    pge_hourly_ev = pge_hourly_ev0
                else:
                    pge_hourly_ev = pd.concat([pge_hourly_ev, pge_hourly_ev0], axis=0)
    
    # Optional: Save to file
    if save:
        output_path = os.path.join(output_dir, 'addload_cec2127_2_allpge.csv')
        pge_hourly_ev.to_csv(output_path, index=False)
    
    # Return DataFrame of results
    return pge_hourly_ev


def make_addload_cec2127_2_ev(sc_num_ls, save=False):
    '''
    Creates addload DataFrame for hourly average EV charging load for each feeder 
    in PG&E's service territory in 2030, 2040, and 2050 based on CEC AB 2127 2nd Assessment
    
    Parameters
    ----------
    sc_num_ls : list
        List of scenario numbers.

    Returns
    -------
    DataFrame of EVres and EVcom load growth for each feeder, scenario, and month-hour
    '''
    print('Calculating incremental load from electric vehicles based on CEC AB 2127 2nd Assessment...')
    
    # Get PG&E-wide hourly load
    pge_hourly_ev = make_pge_cec2127_2_hourly_ev(sc_num_ls, ratio_from_2030, pge_county_ls)
    
    # Get fraction of residential and commercial customers for each feeder
    circinfo = gpd.read_file(ica_file, layer="FeederDetail")
    circinfo = circinfo.drop(columns='geometry')
    circinfo['res_frac'] = circinfo['ResCust'] / circinfo['ResCust'].sum()
    circinfo['com_frac'] = circinfo['ComCust'] / circinfo['ComCust'].sum()
    
    # Create month-hour DataFrame
    mh = pd.DataFrame({'month': list(np.repeat(np.array(range(1,13)), 24)), 'hour': list(range(24))*12})
    mh['mhid'] = range(1, len(mh)+1)
    
    # Create addload (similar to circloadincr)
    addload = pd.merge(circinfo[['FeederID', 'res_frac', 'com_frac']], mh, how='cross')
    addload = addload.rename(columns={'FeederID':'feeder_id'})
    
    # Iterate across scenarios and years
    for sc_num in sc_num_ls:
        print('Scenario: ',sc_num)
        for yr in [2030, 2040, 2050]:
            print('Year: ',yr)
            # Get PG&E hourly profiles for scenario and year
            pge_sc_yr = pge_hourly_ev[ (pge_hourly_ev['year']==yr) & (pge_hourly_ev['sc_num']==sc_num) ]
            pge_sc_yr = pge_sc_yr[['month', 'hour', 'ldinc_EVres_kW', 'ldinc_EVcom_kW']]
            # Merge into circloadincr
            addload = pd.merge(addload, pge_sc_yr, how='outer', on=['month', 'hour'])
            # Calculate EVres and EVcom for each feeder and round to 3 decimals
            addload[f'ldinc_EVres{sc_num}_{yr}_kW'] = round(addload['ldinc_EVres_kW'] * addload['res_frac'], 3)
            addload[f'ldinc_EVcom{sc_num}_{yr}_kW'] = round(addload['ldinc_EVcom_kW'] * addload['com_frac'], 3)
            # Drop PG&E-wide columns
            addload = addload.drop(columns=['ldinc_EVres_kW', 'ldinc_EVcom_kW'])
    
    # Drop res_frac and com_frac
    addload = addload.drop(columns=['res_frac', 'com_frac'])
    # Sort
    addload = addload.sort_values(['feeder_id', 'mhid'])
    
    # Optional: Save to file
    if save:
        output_path = os.path.join(output_dir, 'addload_cec2127_2_ev.csv')
        addload.to_csv(output_path, index=False)
    
    # Return DataFrame
    return addload