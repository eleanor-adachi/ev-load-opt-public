# -*- coding: utf-8 -*-
"""
add_nrelefs_be_load.py

Created on Tue Mar 26 16:50:36 2024

@author: elean

Aggregates NREL electrification scenarios

Adapted from nrelelecscenarios.Rmd  and load_v4_EA.py
Switched source of base load from pickles/results_cmh.csv to ICA "FeederLoadProfile" layer

List of scenarios:
    1. Low
    2. Reference
    3. Medium
    4. High
    5. Technical Potential

"""
# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Inputs
energy_file = r'..\data\raw_data\be\energy.csv'
ica_file = r'..\data\raw_data\ica\ICADisplay.gdb.zip'
sc_num_ls = range(2, 5) # include 2, 3, and 4 (Reference, Medium and High)

# Output filepath
nrelefs_output = r'..\data\addload_nrelefs_be.csv'

def create_sc_df():
    '''Create Scenario DataFrame for NREL Electrification Futures Study'''
    sc_df = pd.DataFrame([
        ['1: Low', 1, 'Low'], ['2: Reference', 2, 'Reference'], ['3: Medium', 3, 'Medium'],
        ['4: High', 4, 'High'], ['5: Technical Potential', 5, 'Technical Potential']], 
        columns=['sc', 'sc_num', 'sc_name']
        )
    return sc_df

def calc_pct_chg_2024(save_figs=False):
    '''
    Calculate percent increase or decrease in load between 2024 and a set of 
    future years based on data from NREL Electrification Futures Study
    '''
    # Directly from NREL docs
    # Download "Final Energy Demand.gzip" (i.e., energy.csv.gzip) from https://data.nrel.gov/submissions/92
    # Unzip and save as energy.csv
    nrelenergy = pd.read_csv(energy_file, header=0, index_col=0)
    nrelenergy = nrelenergy.reset_index(drop=True)
    
    # Create scenario DataFrame
    sc_df = create_sc_df()
    
    # Extract residential electrification columns from NREL EFS data
    nrelenergy_sub = nrelenergy[(nrelenergy['STATE'] == "CALIFORNIA") & 
                                (nrelenergy['SECTOR'] == "RESIDENTIAL") & 
                                (nrelenergy['SUBSECTOR'].isin(["RESIDENTIAL SPACE HEATING", "RESIDENTIAL WATER HEATING"]))].copy()
    nrelenergy_sub['sc'] = nrelenergy_sub['SCENARIO'].replace({'LOW ELECTRICITY GROWTH - MODERATE TECHNOLOGY ADVANCEMENT': '1: Low',
                                                              'REFERENCE ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT': '2: Reference',
                                                              'MEDIUM ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT': '3: Medium',
                                                              'HIGH ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT': '4: High',
                                                              'ELECTRIFICATION TECHNICAL POTENTIAL - MODERATE TECHNOLOGY ADVANCEMENT': '5: Technical Potential'})
    nrelenergy_sub = nrelenergy_sub.astype({'YEAR': 'int', 'MMBTU': 'float'})
    nrelenergy_sub['sc'] = pd.Categorical(nrelenergy_sub['sc'], categories=['1: Low', '2: Reference', '3: Medium', '4: High', '5: Technical Potential'])
    nrelenergy_sub = nrelenergy_sub.drop(columns=['DEMAND_TECHNOLOGY'])
    
    # Sum across subsectors
    nrelenergy_sub = nrelenergy_sub.groupby(['SUBSECTOR', 'sc', 'YEAR', 'FINAL_ENERGY']).agg({'MMBTU': 'sum'}).reset_index()
    nrelenergy_sub = pd.merge(nrelenergy_sub, sc_df, how='inner', on='sc')
    
    # Plot projected energy demand in California by source (?)
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=nrelenergy_sub, x='YEAR', y='MMBTU', hue='FINAL_ENERGY', style='SUBSECTOR')
    plt.xlabel('')
    plt.ylabel('Projected energy demand in California (MWh)')
    plt.legend(title='Energy source', loc='lower center')
    if save_figs:
        plt.savefig(r'..\results\nrelelec_res_shwh.png', dpi=300)
    
    # Plot projected energy demand in California by scenario (?)
    nrelenergy_sub_agg = nrelenergy_sub.drop(columns=['SUBSECTOR']).groupby(['sc', 'YEAR', 'FINAL_ENERGY']).agg({'MMBTU': 'sum'}).reset_index()
    plt.figure(figsize=(9, 4.1))
    sns.lineplot(data=nrelenergy_sub_agg, x='YEAR', y='MMBTU', hue='FINAL_ENERGY', style='sc')
    plt.xlabel('')
    plt.ylabel('Projected energy demand in California (MWh)')
    plt.ylim(0, 100)
    if save_figs:
        plt.savefig(r'..\results\nrelelec_res.png', dpi=300)
    
    # Aggregate projected *electricity* demand by scenario and year
    nrelenergy_sub_agg2 = nrelenergy_sub.drop(columns=['SUBSECTOR']).groupby(['sc', 'YEAR', 'FINAL_ENERGY']).agg({'MMBTU': 'sum'}).reset_index()
    nrelenergy_sub_agg2 = pd.merge(nrelenergy_sub_agg2, sc_df, how='inner', on='sc')
    nrelenergy_pctchgelec = nrelenergy_sub_agg2[nrelenergy_sub_agg2['FINAL_ENERGY'] == 'ELECTRICITY'].copy()
    
    # Calculate percent increase in electricity demand between 2024 and given year
    # set rep=34 if not filtering year, rep=4 if filtering
    MMBTU_2024 = np.repeat(nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == 1) & (nrelenergy_pctchgelec['YEAR'] == 2024)]['MMBTU'].values, 34)
    MMBTU_2024 = np.concatenate((MMBTU_2024, np.repeat(nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == 2) & (nrelenergy_pctchgelec['YEAR'] == 2024)]['MMBTU'].values, 34)))
    MMBTU_2024 = np.concatenate((MMBTU_2024, np.repeat(nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == 3) & (nrelenergy_pctchgelec['YEAR'] == 2024)]['MMBTU'].values, 34)))
    MMBTU_2024 = np.concatenate((MMBTU_2024, np.repeat(nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == 4) & (nrelenergy_pctchgelec['YEAR'] == 2024)]['MMBTU'].values, 34)))
    MMBTU_2024 = np.concatenate((MMBTU_2024, np.repeat(nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == 5) & (nrelenergy_pctchgelec['YEAR'] == 2024)]['MMBTU'].values, 34)))
    nrelenergy_pctchgelec['MMBTU_2024'] = MMBTU_2024
    nrelenergy_pctchgelec['chgfrom2024_pct'] = (nrelenergy_pctchgelec['MMBTU'] - nrelenergy_pctchgelec['MMBTU_2024']) / nrelenergy_pctchgelec['MMBTU_2024'] * 100
    
    # run if YEAR above not filtered to the years we will use
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=nrelenergy_pctchgelec, x='YEAR', y='chgfrom2024_pct', hue='sc')
    plt.xlabel('Year')
    plt.ylabel('Change from 2024 (%)')
    if save_figs:
        plt.savefig(r'..\results\nrelelec_res_pctchg_2024.png', dpi=300)
    
    # nrelenergy_pctchgelec_pivot = nrelenergy_pctchgelec.pivot_table(index='sc', columns='YEAR', values='chgfrom2024_pct')
    nrelenergy_pctchgelec = nrelenergy_pctchgelec[[
        'sc', 'sc_num', 'sc_name', 'YEAR', 'FINAL_ENERGY', 'MMBTU', 'MMBTU_2024', 
        'chgfrom2024_pct'
        ]]
    return nrelenergy_pctchgelec


def make_addload_nrelefs_be(sc_num_ls, save=False):
    '''
    Calculate the increase in loads due to Building Electrification (BE)
    for each feeder based on data from NREL Electrification Futures Study
    '''
    print('Calculating incremental load from Building Electrification based on NREL Electrification Futures Study...')
    
    # Use ICA data as the source of "baseline" load for each circuit
    circload = gpd.read_file(ica_file, layer="FeederLoadProfile")
    circload = circload.drop(columns='geometry')
    circload = circload.rename(columns={'FeederID':'feeder_id'})
    
    # Create month-hour DataFrame
    mh = pd.DataFrame({'month': list(np.repeat(np.array(range(1,13)), 24)), 'hour': list(range(24))*12})
    mh['mhid'] = range(1, len(mh)+1)
    
    # Reformat circuit load DataFrame & add month-hour data
    circload['feeder_id'] = circload['feeder_id'].astype(str).str.zfill(9)
    circload['month'] = circload['MonthHour'].str[:2].astype(int)
    circload['hour'] = circload['MonthHour'].str[3:5].astype(int)
    circload = circload.rename(columns={'Light': 'l_kW', 'High': 'h_kW'})
    circload = pd.merge(mh, circload, on=['month', 'hour'], how='outer')
    
    # Calculate percent increase or decrease in load between 2024 and future years
    nrelenergy_pctchgelec = calc_pct_chg_2024()
                
    # Create addload (similar to circloadincr)
    addload = circload[['feeder_id', 'month', 'hour', 'mhid', 'h_kW']]
    
    # Iterate across scenarios and years
    for sc_num in sc_num_ls:
        print('Scenario: ',sc_num)
        for yr in [2030, 2040, 2050]:
            print('Year: ',yr)
            # Get percent load increase for the scenario and year
            pct = nrelenergy_pctchgelec[(nrelenergy_pctchgelec['sc_num'] == sc_num) & (nrelenergy_pctchgelec['YEAR'] == yr)]['chgfrom2024_pct'].values[0]
            # Calculate increase in BE for each feeder and round to 3 decimals
            addload[f'ldinc_BE{sc_num}_{yr}_kW'] = round(addload['h_kW'] * (pct/100), 3)
    
    # Drop h_kW
    addload = addload.drop(columns=['h_kW'])
    # Sort
    addload = addload.sort_values(['feeder_id', 'mhid'])
    
    # Optional: Save to file
    if save:
        addload.to_csv(nrelefs_output, index=False)
    
    return addload