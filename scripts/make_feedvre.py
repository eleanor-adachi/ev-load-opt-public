# -*- coding: utf-8 -*-
"""
make_feedvre.py

Created on Mon Dec 16 11:50:51 2024

@author: elean

Makes 288 month-hour profile for each feeder's portion of Variable Renewable Energy 
(VRE) for the 12 month period that ends with the end date

Copied from feat/local

"""

# Import libraries
import pandas as pd
import datetime as dt

# Import module
from make_pge_vre import make_pge_vre, make_load_profile

# Inputs
# end_date = dt.datetime(2023, 10, 31)
# prod_yr_xl_dict = {
#     2022: r'..\data\raw_data\vre\productionandcurtailmentsdata_2022.xlsx',
#     2023: r'..\data\raw_data\vre\productionandcurtailmentsdata_2023.xlsx',
#     2024: r'..\data\raw_data\vre\production-and-curtailments-data-2024.xlsx'
#     }
# load_yr_xl_dict = {
#     2022: r'..\data\raw_data\vre\historicalemshourlyload-2022.xlsx',
#     2023: r'..\data\raw_data\vre\historicalemshourlyloadfor2023.xlsx'
#     }
prod_xl =  r'..\data\raw_data\vre\productionandcurtailmentsdata_2023.xlsx'
load_xl = r'..\data\raw_data\vre\historicalemshourlyloadfor2023.xlsx'
feedload_file = r'..\data\feedload.csv'

# def make_feedvre(prod_yr_xl_dict, load_yr_xl_dict, start_date=None, end_date=None, save=True):
#     '''
#     Creates month-hour profile for each feeder's portion of Variable Renewable Energy 
#     (VRE) based on CAISO production and curtailments data and historical EMS 
#     hourly load data over a specified time period
#     '''
#     # Make 288 month-hour VRE profile for entire PG&E service territory
#     pge_vre = make_pge_vre(prod_yr_xl_dict, load_yr_xl_dict, end_date=end_date, save=False)
    
#     # Read feedload
#     feedload = pd.read_csv(feedload_file)
    
#     # Pivot feedload; total load for each feeder
#     feedload_totals = pd.pivot_table(feedload, index='feeder_id', values='h_kW', aggfunc='sum')
#     feedload_totals = feedload_totals.reset_index()
    
#     # Calculate total load across all PG&E feeders
#     total_load = feedload['h_kW'].sum()
    
#     # Merge pge_vre and feedload_totals
#     feedvre = pd.merge(pge_vre, feedload_totals, how='cross')
#     feedvre = feedvre.sort_values(['feeder_id', 'month', 'hour'])
    
#     # Calculate each feeder's portion of solar_kW and wind_kW
#     feedvre['solar_kW'] = feedvre['solar_kW'] * feedvre['h_kW'] / total_load
#     feedvre['wind_kW'] = feedvre['wind_kW'] * feedvre['h_kW'] / total_load
    
#     # Drop unnecessary columns
#     feedvre = feedvre.drop(columns='h_kW')
    
#     # Save
#     if save:
#         feedvre.to_csv(r'..\data\feedvre.csv', index=False)
    
#     return feedvre
def make_feedvre(prod_xl, load_xl, year_ls=[2030,2040,2050], save=True):
    '''
    Creates month-hour profile for each feeder's portion of Variable Renewable Energy 
    (VRE) based on CAISO production and curtailments data and historical EMS 
    hourly load data
    '''
    # Make 288 month-hour VRE profile for entire PG&E service territory
    pge_vre = make_pge_vre(prod_xl, load_xl, year_ls=year_ls, save=False)
    
    # Make 288 month-hour load profile for PG&E and CAISO
    load_profile = make_load_profile(load_xl)
    # Calculate total load across PG&E service territory
    pge_total_load = load_profile['PGE'].sum() * 1000 # in kWh
    
    # Read feedload
    feedload = pd.read_csv(feedload_file)
    
    # Pivot feedload; total load for each feeder
    feedload_totals = pd.pivot_table(feedload, index='feeder_id', values='h_kW', aggfunc='sum')
    feedload_totals = feedload_totals.reset_index()
    
    # Calculate total load across all PG&E feeders
    total_load = feedload['h_kW'].sum()
    
    # Merge pge_vre and feedload_totals
    feedvre = pd.merge(pge_vre, feedload_totals, how='cross')
    feedvre = feedvre.sort_values(['feeder_id', 'month', 'hour'])
    
    # Calculate each feeder's portion of solar and wind generation for each year
    for year in year_ls:
        solar_yr_col = 'solar_%d_kW' % year
        wind_yr_col = 'wind_%d_kW' % year
        # feedvre[solar_yr_col] = feedvre[solar_yr_col] * feedvre['h_kW'] / total_load
        # feedvre[wind_yr_col] = feedvre[wind_yr_col] * feedvre['h_kW'] / total_load
        feedvre[solar_yr_col] = feedvre[solar_yr_col] * feedvre['h_kW'] / pge_total_load
        feedvre[wind_yr_col] = feedvre[wind_yr_col] * feedvre['h_kW'] / pge_total_load
    
    # Drop unnecessary columns
    feedvre = feedvre.drop(columns='h_kW')
    
    # Save
    if save:
        feedvre.to_csv(r'..\data\feedvre.csv', index=False)
    
    return feedvre


#%%

if __name__ == "__main__":
    # feedvre = make_feedvre(prod_yr_xl_dict, load_yr_xl_dict, end_date=end_date)
    feedvre = make_feedvre(prod_xl, load_xl)