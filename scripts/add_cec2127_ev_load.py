# -*- coding: utf-8 -*-
"""
add_cec2127_ev_load.py

Created on Sun Aug  4 19:19:51 2024

@author: elean

Adapted from EVcharging.Rmd created by Anna Brockway

Creates EV charging profiles based on California Energy Commission's 1st 
Electric Vehicle Charging Infrastructure Assessment for Assembly Bill 2127

Go to https://www.energy.ca.gov/data-reports/reports/electric-vehicle-charging-infrastructure-assessment-ab-2127
Scroll down to "Reports"
Under "Assembly Bill 2127 Electric Vehicle Charging Infrastructure Assessment - 
Analyzing Charging Needs to Support Zero-Emission Vehicles in 2030", download:
    1. Spreadsheet of AB 2127 Commission Report Figure Results
    2. Spreadsheet of AB 2127 Commission Report Table Results

List of scenarios:
    1. "Standard" (weekday) from Figure 13_Weekday
    2. "Unconstrained Residential Load" from Figure D-1
    3. "Low Residential Access" from Figure D-2
    4. "High Residential Access" from Figure D-3
    5. "Low Energy Demand" from Figure D-4
    6. "High Energy Demand" from Figure D-5
    7. "Low Range PEVs" from Figure D-6
    8. "Gas Station Model" from Figure D-7
    9. "EV Happy Hour" from Figure D-8
    10. "Level 1 Charging" from Figure D-9
    11. "Lazy PHEVs" from Figure D-10
    12. "Widespread Topping Off" from Figure D-11

Key assumptions:
    - Number of chargers is proportional to amount of EV charging demand (MW)
    - Compared to 2030, EV charging demand will double in 2040 and quadruple in 2050

"""

# Importing libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import os

# Inputs
cec2127_tableresults = r'..\data\raw_data\ev\cec_ab2127\TN238851_20210714T100913_AB 2127 Commission Report Table Results.xlsx'
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
cec2127_figureresults = r'..\data\raw_data\ev\cec_ab2127\TN238852_20210714T100913_AB 2127 Commission Report Figure Results.xlsx'
sc_tab_dict = {
    1: 'Figure 13_Weekday',
    2: 'Figure D-1',
    3: 'Figure D-2',
    4: 'Figure D-3',
    5: 'Figure D-4',
    6: 'Figure D-5',
    7: 'Figure D-6',
    8: 'Figure D-7',
    9: 'Figure D-8',
    10: 'Figure D-9', # note, only tab that contins Public Lv 1 or Work Lv 1
    11: 'Figure D-10',
    12: 'Figure D-11'
    }
# sc_num_ls = range(1,13) # all scenarios
sc_num_ls = [1, 3, 4] # selected scenarios
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'
# EV charging growth ratios
ratio_from_2030 = {
    2030: 1,
    2040: 2,
    2050: 4
    }

# Output filepath
output_dir = r'..\data'

#%% Get fraction of PG&E EV chargers

# 39.019% of chargers in California are expected to be located in counties served by PG&E
# based on this we will assume that 39.019% of total power demand for EV charging will occur in PG&E's territory

def calc_frac_pge(pge_county_ls):
    # import data
    chrgrsbycty = pd.read_excel(cec2127_tableresults, sheet_name=14, skiprows=2, engine='openpyxl')
    chrgrsPGE = chrgrsbycty[chrgrsbycty["County"].str.title().isin(pge_county_ls)]
    
    chrgrsPGE_total = chrgrsPGE[["MUD","Work","Public","DCFC","Total"]].sum()
    chrgrsPGE_total['County'] = 'TOTAL_PGE'
    
    chrgrsPGE = chrgrsPGE.reset_index(drop=True)
    chrgrsPGE.loc[len(chrgrsPGE), :] = chrgrsPGE_total
    
    # comparing total number of chargers in PG&E's counties vs statewide
    chrgrsfrac = pd.concat([chrgrsbycty.tail(1), chrgrsPGE.tail(1)]).melt(id_vars="County")
    chrgrsfrac = chrgrsfrac.pivot(index="variable", columns="County", values="value")
    chrgrsfrac["FRAC_PGE"] = chrgrsfrac["TOTAL_PGE"] / chrgrsfrac["TOTAL"]
    
    frac_pge = chrgrsfrac.loc["Total", "FRAC_PGE"]
    return frac_pge

#%% Create hourly average EV charging profiles for PG&E

def read_cec2127_EVprofiles():
    '''
    Read CEC EV profiles and calculate residential and commercial totals for each scenario
    Returns a dictionary of DataFrames indexed by scenario number
    '''
    raw_EVprofiles = {}
    filter_text = 'Post-Processed 10-Minute Interval Load (MW)'
    
    # Get EV charging profile for each scenario
    for sc_num in sc_num_ls:
        tab = sc_tab_dict[sc_num]
        raw_df = pd.read_excel(cec2127_figureresults, sheet_name=tab, header=[0,2])
        col_filter = list(map(lambda x: filter_text in x, raw_df.columns.get_level_values(0)))
        df = raw_df[raw_df.columns[col_filter]]
        df = df.droplevel(0, axis=1)
        df = df.rename(
            columns={
                'Residential Lv 1':'homelvl1', 'Residential Lv 2':'homelvl2', 
                'Public Lv 1':'publiclvl1', 'Public Lv 2':'publiclvl2', 
                'Work Lv 1':'worklvl1', 'Work Lv 2':'worklvl2', 
                'DC Fast':'DCfast', 'Total':'EVtot'
                }
            )
        # drop rows where Hour is NA
        df = df.dropna(subset='Hour')
        # drop last row
        df = df.iloc[:-1, :]
        # replace Hour column
        df['Hour'] = np.repeat(np.array(range(24)), 6)
        # calculate EVres and EVcom
        df['EVres'] = df[['homelvl1', 'homelvl2']].sum(axis=1)
        if tab == 'Figure D-9':
            df['EVcom'] = df[['DCfast', 'publiclvl1', 'publiclvl2', 'worklvl1', 'worklvl2']].sum(axis=1)
        else:
            df['EVcom'] = df[['DCfast', 'publiclvl2', 'worklvl2']].sum(axis=1)
        raw_EVprofiles[sc_num] = df
        
    return raw_EVprofiles


def make_pge_hourly_ev(sc_num_ls, ratio_from_2030, pge_county_ls, save=False):
    '''
    Creates addload DataFrame for hourly average EV charging load in PG&E's 
    service territory in 2030, 2040, and 2050 based on CEC AB 2127 EV charging 
    infrastructure assessment
    '''
    # Calculate PG&E fraction of statewide EV load
    frac_pge = calc_frac_pge(pge_county_ls)
    # Process CEC AB 2127 EV load profiles
    raw_EVprofiles = read_cec2127_EVprofiles()
    
    # Create DataFrame for PG&E EV charging load profiles with hourly averages
    # Represents total increase in EVres and EVcom load across the service territory
    pge_hourly_ev = pd.DataFrame(columns=['sc_num', 'year', 'month', 'hour', 'ldinc_EVres_kW', 'ldinc_EVcom_kW'])
                                 
    # Iterate across scenarios, years, months
    for sc_num in sc_num_ls:
        EVprofile = raw_EVprofiles[sc_num].copy()
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
        output_path = os.path.join(output_dir, 'addload_cec2127_allpge.csv')
        pge_hourly_ev.to_csv(output_path, index=False)
    
    # Return DataFrame of results
    return pge_hourly_ev


def make_addload_cec2127_ev(sc_num_ls, save=False):
    '''
    Creates addload DataFrame for hourly average EV charging load in PG&E's 
    service territory in 2030, 2040, and 2050 based on CEC AB 2127 EV charging 
    infrastructure assessment
    '''
    print('Calculating incremental load from electric vehicles based on CEC AB 2127 EV charging infrastructure assessment...')
    
    # Get PG&E-wide hourly load
    pge_hourly_ev = make_pge_hourly_ev(sc_num_ls, ratio_from_2030, pge_county_ls)
    
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
        output_path = os.path.join(output_dir, 'addload_cec2127_ev.csv')
        addload.to_csv(output_path, index=False)
    
    # Return DataFrame
    return addload