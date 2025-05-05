# -*- coding: utf-8 -*-
"""
make_pge_vre.py

Created on Tue Oct  1 15:30:26 2024

@author: elean

Makes 288 month-hour profile for PG&E's portion of Variable Renewable Energy 
(VRE) for the 12 month period that ends with the end date

Production data downloaded from: https://www.caiso.com/about/our-business/managing-the-evolving-grid
Load data downloaded from: https://www.caiso.com/library/historical-ems-hourly-load
VRE capacity projections based on CAISO installed renewable resources and CPUC 2023 Preferred System Plan
- CAISO installed renewable resources: https://www.caiso.com/documents/key-statistics-jan-2024.pdf
- CPUC 2023 Preferred System Plan: https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/integrated-resource-plan-and-long-term-procurement-plan-irp-ltpp/2023-irp-cycle-events-and-materials/2024-01-12-presentation-summarizing-updated-servm-and-resolve-analysis.pdf

"""

# Importing libraries
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

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
vre_cap_dict = {
    'solar_GW': {
        2023: 18.52,
        2030: 33.32,
        2040: 53.52,
        2050: 76.02,
        },
    'wind_GW': {
        2023: 8.36,
        2030: 13.36,
        2040: 19.86,
        2050: 21.16,
        },
    }

# def make_load_profile(load_yr_xl_dict, start_date=None, end_date=None):
#     '''
#     Creates month-hour load profile based on sum of CAISO's historical EMS hourly load
#     over a specified time period
#     '''
#     # Set end date as last day of month BEFORE previous month if none specified
#     if end_date is None:
#         today = dt.date.today()
#         first_of_month = today.replace(day=1)
#         end_date = first_of_month - relativedelta(months=1) - dt.timedelta(days=1)
#     # Create date range
#     if start_date is None:
#         # default is 12-month period
#         months = pd.date_range(end=end_date, periods=12, freq='MS')
#     else:
#         months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
#     # Get years in date range
#     yr_ls = list(set(map(lambda x: x.year, months)))
    
#     # Read CAISO data, filter for date range, and combine into one DataFrame
#     master_df = pd.DataFrame()
#     for yr in yr_ls:
#         xl_file = load_yr_xl_dict[yr]
#         raw_df = pd.read_excel(xl_file)
#         raw_df = raw_df.rename(columns={'HR': 'HE'}) # Hour represents Hour Ending (HE)
#         raw_df['hour'] = raw_df['HE'].apply(lambda x: x-1) # subtract one hour to get Hour Beginning
#         raw_df['month_dt'] = raw_df['Date'].apply(lambda x: dt.datetime(x.year, x.month, 1))
#         raw_df = raw_df[ raw_df['month_dt'].isin(months) ]
#         if master_df.empty:
#             master_df = raw_df.copy()
#         else:
#             master_df = pd.concat([master_df, raw_df])
    
#     # Make 288 month-hour load profile for PG&E and CAISO
#     master_df['month'] = master_df['month_dt'].apply(lambda x: x.month)
#     load_profile = master_df.groupby(['month', 'hour'], as_index=False)[['PGE', 'CAISO']].sum()
    
#     return load_profile

# def make_load_profile(load_xl):
#     '''
#     Creates month-hour load profile based on sum of CAISO's historical EMS hourly load
#     '''
def make_load_profile(load_xl):
    '''
    Creates month-hour load profile based on average of CAISO's historical EMS hourly load
    '''
    # Read CAISO EMS data
    raw_df = pd.read_excel(load_xl)
    
    # Reformat
    df = raw_df.copy()
    df = df.rename(columns={'HR': 'HE'}) # Hour represents Hour Ending (HE)
    df['hour'] = df['HE'].apply(lambda x: x-1) # subtract one hour to get Hour Beginning
    df['month_dt'] = df['Date'].apply(lambda x: dt.datetime(x.year, x.month, 1))
    df['month'] = df['month_dt'].apply(lambda x: x.month)
    
    # Make 288 month-hour load profile for PG&E and CAISO
    # load_profile = df.groupby(['month', 'hour'], as_index=False)[['PGE', 'CAISO']].sum()
    load_profile = df.groupby(['month', 'hour'], as_index=False)[['PGE', 'CAISO']].mean()
    
    return load_profile
    

# def make_vre_profile(prod_yr_xl_dict, start_date=None, end_date=None, save=False):
#     '''
#     Creates month-hour profile for Variable Renewable Energy (VRE) based on 
#     averaging CAISO production and curtailments data over a specified time period
#     '''
#     # Set end date as last day of month BEFORE previous month if none specified
#     if end_date is None:
#         today = dt.date.today()
#         first_of_month = today.replace(day=1)
#         end_date = first_of_month - relativedelta(months=1) - dt.timedelta(days=1)
#     # Create date range
#     if start_date is None:
#         # default is 12-month period
#         months = pd.date_range(end=end_date, periods=12, freq='MS')
#     else:
#         months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
#     # Get years in date range
#     yr_ls = list(set(map(lambda x: x.year, months)))
    
#     # Read CAISO data, filter for date range, and combine into one DataFrame
#     master_df = pd.DataFrame()
#     for yr in yr_ls:
#         xl_file = prod_yr_xl_dict[yr]
#         raw_df = pd.read_excel(xl_file, sheet_name='Production')
#         raw_df = raw_df.rename(columns={'Hour': 'HE'}) # Hour represents Hour Ending (HE)
#         raw_df['hour'] = raw_df['HE'].apply(lambda x: x-1) # subtract one hour to get Hour Beginning
#         raw_df['month_dt'] = raw_df['Date'].apply(lambda x: dt.datetime(x.year, x.month, 1))
#         raw_df = raw_df[ raw_df['month_dt'].isin(months) ]
#         if master_df.empty:
#             master_df = raw_df.copy()
#         else:
#             master_df = pd.concat([master_df, raw_df])
    
#     # Make 288 month-hour VRE profile
#     master_df['month'] = master_df['month_dt'].apply(lambda x: x.month)
#     vre_profile = master_df.groupby(['month', 'hour'], as_index=False)[['Solar', 'Wind']].mean()
    
#     # Rename columns and convert from MW to kW
#     vre_profile = vre_profile.rename(columns={'Solar':'solar_MW', 'Wind':'wind_MW'})
#     vre_profile['solar_kW'] = round(vre_profile['solar_MW']*1000, 3)
#     vre_profile['wind_kW'] = round(vre_profile['wind_MW']*1000, 3)
#     vre_profile = vre_profile.drop(columns=['solar_MW', 'wind_MW'])
    
#     # Save
#     if save:
#         vre_profile.to_csv(r'..\data\vre_profile.csv', index=False)
    
#     return vre_profile
def make_vre_profile(prod_xl, save=False):
    '''
    Creates month-hour profile for Variable Renewable Energy (VRE) based on 
    averaging CAISO production and curtailments data
    '''
    # Read CAISO production data
    raw_df = pd.read_excel(prod_xl, sheet_name='Production')
    
    # Reformat
    df = raw_df.copy()
    df = df.rename(columns={'Hour': 'HE'}) # Hour represents Hour Ending (HE)
    df['hour'] = df['HE'].apply(lambda x: x-1) # subtract one hour to get Hour Beginning
    df['month_dt'] = df['Date'].apply(lambda x: dt.datetime(x.year, x.month, 1))
    df['month'] = df['month_dt'].apply(lambda x: x.month)
    
    # Make 288 month-hour VRE profile
    vre_profile = df.groupby(['month', 'hour'], as_index=False)[['Solar', 'Wind']].mean()
    
    # Rename columns and convert from MW to kW
    vre_profile = vre_profile.rename(columns={'Solar':'solar_MW', 'Wind':'wind_MW'})
    vre_profile['solar_kW'] = round(vre_profile['solar_MW']*1000, 3)
    vre_profile['wind_kW'] = round(vre_profile['wind_MW']*1000, 3)
    vre_profile = vre_profile.drop(columns=['solar_MW', 'wind_MW'])
    
    # Save
    if save:
        vre_profile.to_csv(r'..\data\vre_profile.csv', index=False)
    
    return vre_profile

# def make_pge_vre(prod_yr_xl_dict, load_yr_xl_dict, start_date=None, end_date=None, save=True):
#     '''
#     Creates month-hour profile for PG&E's portion of Variable Renewable Energy 
#     (VRE) based on CAISO production and curtailments data and historical EMS 
#     hourly load data over a specified time period
#     '''
#     # Get statewide 288 month-hour VRE profile
#     vre_profile = make_vre_profile(prod_yr_xl_dict, start_date=start_date, end_date=end_date)
    
#     # Get PG&E and CAISO 288 month-hour load profile
#     load_profile = make_load_profile(load_yr_xl_dict, start_date=start_date, end_date=end_date)
    
#     # Merge load profile into VRE profile
#     pge_vre = pd.merge(vre_profile, load_profile, how='inner', on=['month', 'hour'])
#     pge_vre = pge_vre.rename(columns={'solar_kW':'caiso_solar_kW', 'wind_kW':'caiso_wind_kW'})
    
#     # Multiply VRE by PGE load fraction and round to 3 decimals
#     pge_vre['solar_kW'] = pge_vre['caiso_solar_kW'] * pge_vre['PGE'] / pge_vre['CAISO']
#     pge_vre['wind_kW'] = pge_vre['caiso_wind_kW'] * pge_vre['PGE'] / pge_vre['CAISO']
#     pge_vre['solar_kW'] = round(pge_vre['solar_kW'], 3)
#     pge_vre['wind_kW'] = round(pge_vre['wind_kW'], 3)
    
#     # Drop columns
#     pge_vre = pge_vre.drop(columns=['caiso_solar_kW', 'caiso_wind_kW', 'PGE', 'CAISO'])
    
#     # Save
#     if save:
#         pge_vre.to_csv(r'..\data\pge_vre.csv', index=False)
    
#     return pge_vre
def make_pge_vre(prod_xl, load_xl, year_ls=[2030,2040,2050], save=True):
    '''
    Creates month-hour profile for PG&E's portion of Variable Renewable Energy 
    (VRE) based on CAISO production and curtailments data and historical EMS 
    hourly load data and scales up based on VRE growth projections
    '''
    # Get statewide 288 month-hour VRE profile
    vre_profile = make_vre_profile(prod_xl)
    
    # Get PG&E and CAISO 288 month-hour load profile
    load_profile = make_load_profile(load_xl)
    # NEW
    pge_frac = load_profile['PGE'].sum() / load_profile['CAISO'].sum()
    
    # Merge load profile into VRE profile
    pge_vre = pd.merge(vre_profile, load_profile, how='inner', on=['month', 'hour'])
    pge_vre = pge_vre.rename(columns={'solar_kW':'caiso_solar_kW', 'wind_kW':'caiso_wind_kW'})
    
    # Multiply VRE by PGE load fraction and round to 3 decimals
    # pge_vre['solar_2023_kW'] = pge_vre['caiso_solar_kW'] * pge_vre['PGE'] / pge_vre['CAISO']
    # pge_vre['wind_2023_kW'] = pge_vre['caiso_wind_kW'] * pge_vre['PGE'] / pge_vre['CAISO']
    pge_vre['solar_2023_kW'] = pge_vre['caiso_solar_kW'] * pge_frac
    pge_vre['wind_2023_kW'] = pge_vre['caiso_wind_kW'] * pge_frac
    pge_vre['solar_2023_kW'] = round(pge_vre['solar_2023_kW'], 3)
    pge_vre['wind_2023_kW'] = round(pge_vre['wind_2023_kW'], 3)
    
    for year in year_ls:
        solar_growth_ratio = vre_cap_dict['solar_GW'][year] / vre_cap_dict['solar_GW'][2023]
        wind_growth_ratio = vre_cap_dict['wind_GW'][year] / vre_cap_dict['wind_GW'][2023]
        pge_vre[ 'solar_%d_kW' % year ] =  pge_vre['solar_2023_kW'] * solar_growth_ratio
        pge_vre[ 'wind_%d_kW' % year ] =  pge_vre['wind_2023_kW'] * wind_growth_ratio
    
    # Drop columns
    pge_vre = pge_vre.drop(columns=['solar_2023_kW', 'wind_2023_kW', 'caiso_solar_kW', 'caiso_wind_kW', 'PGE', 'CAISO'])
    
    # Save
    if save:
        pge_vre.to_csv(r'..\data\pge_vre.csv', index=False)
    
    return pge_vre

#%%

if __name__ == "__main__":
    # pge_vre = make_pge_vre(prod_yr_xl_dict, load_yr_xl_dict, end_date=end_date)
    pge_vre = make_pge_vre(prod_xl, load_xl, save=True)