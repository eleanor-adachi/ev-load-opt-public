# -*- coding: utf-8 -*-
"""
make_feedlim.py

Created on Fri Sep 13 21:19:23 2024
Adapted from make_circlim

@author: elean

https://iopscience.iop.org/article/10.1088/2634-4505/ac949c/meta
Supplementary Information
- Steps from "Circuit hosting capacity" on page 11:
"(i) Find the average hosting capacity value for each circuit segment within a 
circuit by averaging hosting capacity values across month-hours;
(ii) Calculate the maximum, median, minimum and the 90th, 75th, 25th, and 10th
percentiles of the averaged circuit segment values for each circuit;
(iii) Select the circuit segments that generated those values as proxy measures 
of circuit capacity for new load; and,
(iv) Use the month-hour hosting capacity values of the identified circuit 
segments as measures for the overall circuit, corresponding to different 
options for how load might be allocated."
- From page 17: "In all cases, we use the “high” load (90th percentile) reported 
by PG&E rather than the “low” load (10th percentile)."

L: the limit for load on the circuit
G: the limit for generation on the circuit
B: the smaller of the thermal or voltage limits
T: the limit due to thermal constraints
V: the limit due to voltage constraints

mn: minimum limit across circuit segments
10: 10th percentile limit across circuit segments
q1: 25th percentile limit across circuit segments
md: median limit across circuit segments
q3: 75th percentile limit across circuit segments
90: 90th percentile limit across circuit segments
mx: maximum limit across circuit segments

Total runtime: 1 hour 20 minutes

"""

import datetime as dt
import os
import numpy as np
import pandas as pd

# NOTE: Disable if at all unsure
pd.set_option('mode.chained_assignment', None)

source_dir = r'C:\Users\elean\OneDrive_relink\OneDrive\Documents\UC Berkeley\ERG Masters Project\EV infrastructure\ICA\df'
file_ls = os.listdir(source_dir)
save_dir = r'..\data'

# Testing
# file_ls = file_ls[:13]

# Example of Type 1 (no Loading_Scenario or Hourly_ICA_OF columns, etc)
# file = '012011101.csv'

# Example of Type 2
# file = '012111102.csv'
# file = '012240402.csv'


def find_limits_by_type(df, limit_col):
    avg_df = df[
        [limit_col, 'LineSectionID']
        ].groupby('LineSectionID').mean().reset_index()
    avg_df['rank'] = avg_df[limit_col].rank(method='first')
    # select_ls = [1] + list(map(lambda x: round(x*avg_df['rank'].max()), [0.25, 0.5, 0.75, 1])) # minimum and percentiles
    # select_labels = ['mn', 'q1', 'md', 'q3', 'mx']
    select_ls = [1] + list(map(lambda x: round(x*avg_df['rank'].max()), [0.1, 0.25, 0.5, 0.75, 0.9, 1])) # minimum and percentiles
    select_labels = ['mn', '10', 'q1', 'md', 'q3', '90', 'mx']
    select_dict = dict(zip(select_ls, select_labels))
    select_df = avg_df[ avg_df['rank'].isin(select_ls) ]
    select_df['LimType'] = select_df['rank'].map(select_dict)
    select_df.drop(columns='rank', inplace=True)
    return select_df

def process_file_type_1(df, feeder_id):
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Filter for Load
    load_df = df[ df['LoadOrGen'] == 'L' ]
    
    # Find circuit segments for mn, q1, md, q3, and mx by thermal, voltage, and both (min of thermal and voltage)
    t_df = find_limits_by_type(load_df, 'IC_Thermal_kW')
    v_df = find_limits_by_type(load_df, 'IC_Voltage_kW')
    b_df = pd.merge(t_df, v_df, how='inner', on='LimType', suffixes=('_t','_v'))
    b_df['IC_Both_kW'] = b_df.apply(
        lambda row: min(row['IC_Thermal_kW'],row['IC_Voltage_kW']), axis=1)
    # WARNING: There are cases where kW values are equal for different segments, selection in those cases is arbitrary
    b_df['LineSectionID'] = b_df.apply(
        lambda row: row['LineSectionID_t'] if row['IC_Thermal_kW']==row['IC_Both_kW']
        else row['LineSectionID_v'], axis=1)
    b_df['Limit'] = b_df.apply(
        lambda row: 'Thermal' if row['IC_Thermal_kW']==row['IC_Both_kW']
        else 'Voltage', axis=1)
    
    # Find thermal limit kW for each month and hour of selected thermal segments
    t_df = t_df.melt(
        id_vars=['LineSectionID', 'LimType'], 
        value_vars=['IC_Thermal_kW'], value_name='Avg_Limit_kW'
        )
    t_df['Limit'] = t_df['variable'].apply(lambda x: x.split('_')[1])
    t_df.drop(columns='variable', inplace=True)
    t_hr_df = pd.merge(
        t_df[['LineSectionID','LimType','Limit']],
        load_df[['LineSectionID','Month','Hour','IC_Thermal_kW']],
        how='left',on='LineSectionID')
    t_hr_df['LimType'] = t_hr_df['LimType'].apply(lambda x: 't_'+x)
    t_hr_df.rename(columns={'IC_Thermal_kW':'Limit_kW'}, inplace=True)
    
    # Find voltage limit kW for each month and hour of selected voltage segments
    v_df = v_df.melt(
        id_vars=['LineSectionID', 'LimType'], 
        value_vars=['IC_Voltage_kW'], value_name='Avg_Limit_kW'
        )
    v_df['Limit'] = v_df['variable'].apply(lambda x: x.split('_')[1])
    v_df.drop(columns='variable', inplace=True)
    v_hr_df = pd.merge(
        v_df[['LineSectionID','LimType','Limit']],
        load_df[['LineSectionID','Month','Hour','IC_Voltage_kW']],
        how='left',on='LineSectionID')
    v_hr_df['LimType'] = v_hr_df['LimType'].apply(lambda x: 'v_'+x)
    v_hr_df.rename(columns={'IC_Voltage_kW':'Limit_kW'}, inplace=True)
    
    # Find thermal limit kW for each month and hour of selected both segments
    b_df = b_df.melt(
        id_vars=['LineSectionID', 'LimType', 'Limit'], 
        value_vars=['IC_Both_kW'], value_name='Avg_Limit_kW'
        )
    bt_df = b_df[ b_df['Limit']=='Thermal' ]
    bt_hr_df = pd.merge(
        bt_df[['LineSectionID','LimType']],
        load_df[['LineSectionID','Month','Hour','IC_Thermal_kW']],
        how='left',on='LineSectionID')
    bt_hr_df.rename(columns={'IC_Thermal_kW':'Limit_kW'}, inplace=True)
    # Find voltage limit kW for each month and hour of selected both segments
    bv_df = b_df[ b_df['Limit']=='Voltage' ]
    bv_hr_df = pd.merge(
        bv_df[['LineSectionID','LimType']],
        load_df[['LineSectionID','Month','Hour','IC_Voltage_kW']],
        how='left',on='LineSectionID')
    bv_hr_df.rename(columns={'IC_Voltage_kW':'Limit_kW'}, inplace=True)
    # Combine thermal and voltage limits for selected both segments
    b_hr_df = pd.concat([bt_hr_df, bv_hr_df], axis=0)
    b_hr_df['Limit'] = 'Both'
    b_hr_df['LimType'] = b_hr_df['LimType'].apply(lambda x: 'b_'+x)
    
    # Combine all limits and add feeder ID and loading scenario
    lim_hr_df = pd.concat([t_hr_df, v_hr_df, b_hr_df], axis=0)
    lim_hr_df['FeederID'] = feeder_id
    lim_hr_df['Loading_Scenario'] = np.nan
    
    # Remove null values and convert to int
    lim_hr_df = lim_hr_df[ pd.notnull(lim_hr_df['Limit_kW']) ]
    lim_hr_df['Limit_kW'] = lim_hr_df['Limit_kW'].astype(int)
    
    return lim_hr_df


def find_limits_by_scenario(df, feeder_id, loading):
    # Drop duplicates
    df = df.drop_duplicates()
    
    # If multiple values for Hourly_Load, keep the highest
    df = df.sort_values(
        ['LoadOrGen', 'Loading_Scenario', 'LineSectionID', 'Month', 'Hour', 'Hourly_Load']
        )
    df = df.drop_duplicates(
        subset=['LoadOrGen', 'Loading_Scenario', 'LineSectionID', 'Month', 'Hour'], 
        keep='last'
        )
    
    # Filter for Load and by Loading Scenario
    load_df = df[ df['LoadOrGen'] == 'L' ]
    load_df = load_df[ load_df['Loading_Scenario'] == loading ]
    
    # Find circuit segments for mn, q1, md, q3, and mx by thermal, voltage, and both (min of thermal and voltage)
    t_df = find_limits_by_type(load_df, 'IC_Thermal_kW')
    v_df = find_limits_by_type(load_df, 'IC_Voltage_kW')
    b_df = pd.merge(t_df, v_df, how='inner', on='LimType', suffixes=('_t','_v'))
    b_df['IC_Both_kW'] = b_df.apply(
        lambda row: min(row['IC_Thermal_kW'],row['IC_Voltage_kW']), axis=1)
    # WARNING: There are cases where kW values are equal for different segments, selection in those cases is arbitrary
    b_df['LineSectionID'] = b_df.apply(
        lambda row: row['LineSectionID_t'] if row['IC_Thermal_kW']==row['IC_Both_kW']
        else row['LineSectionID_v'], axis=1)
    b_df['Limit'] = b_df.apply(
        lambda row: 'Thermal' if row['IC_Thermal_kW']==row['IC_Both_kW']
        else 'Voltage', axis=1)
    
    # Find thermal limit kW for each month and hour of selected thermal segments
    t_df = t_df.melt(
        id_vars=['LineSectionID', 'LimType'], 
        value_vars=['IC_Thermal_kW'], value_name='Avg_Limit_kW'
        )
    t_df['Limit'] = t_df['variable'].apply(lambda x: x.split('_')[1])
    t_df.drop(columns='variable', inplace=True)
    t_hr_df = pd.merge(
        t_df[['LineSectionID','LimType','Limit']],
        load_df[['LineSectionID','Month','Hour','IC_Thermal_kW']],
        how='left',on='LineSectionID')
    t_hr_df['LimType'] = t_hr_df['LimType'].apply(lambda x: 't_'+x)
    t_hr_df.rename(columns={'IC_Thermal_kW':'Limit_kW'}, inplace=True)
    
    # Find voltage limit kW for each month and hour of selected voltage segments
    v_df = v_df.melt(
        id_vars=['LineSectionID', 'LimType'], 
        value_vars=['IC_Voltage_kW'], value_name='Avg_Limit_kW'
        )
    v_df['Limit'] = v_df['variable'].apply(lambda x: x.split('_')[1])
    v_df.drop(columns='variable', inplace=True)
    v_hr_df = pd.merge(
        v_df[['LineSectionID','LimType','Limit']],
        load_df[['LineSectionID','Month','Hour','IC_Voltage_kW']],
        how='left',on='LineSectionID')
    v_hr_df['LimType'] = v_hr_df['LimType'].apply(lambda x: 'v_'+x)
    v_hr_df.rename(columns={'IC_Voltage_kW':'Limit_kW'}, inplace=True)
    
    # Find thermal limit kW for each month and hour of selected both segments
    b_df = b_df.melt(
        id_vars=['LineSectionID', 'LimType', 'Limit'], 
        value_vars=['IC_Both_kW'], value_name='Avg_Limit_kW'
        )
    bt_df = b_df[ b_df['Limit']=='Thermal' ]
    bt_hr_df = pd.merge(
        bt_df[['LineSectionID','LimType']],
        load_df[['LineSectionID','Month','Hour','IC_Thermal_kW']],
        how='left',on='LineSectionID')
    bt_hr_df.rename(columns={'IC_Thermal_kW':'Limit_kW'}, inplace=True)
    # Find voltage limit kW for each month and hour of selected both segments
    bv_df = b_df[ b_df['Limit']=='Voltage' ]
    bv_hr_df = pd.merge(
        bv_df[['LineSectionID','LimType']],
        load_df[['LineSectionID','Month','Hour','IC_Voltage_kW']],
        how='left',on='LineSectionID')
    bv_hr_df.rename(columns={'IC_Voltage_kW':'Limit_kW'}, inplace=True)
    # Combine thermal and voltage limits for selected both segments
    b_hr_df = pd.concat([bt_hr_df, bv_hr_df], axis=0)
    b_hr_df['Limit'] = 'Both'
    b_hr_df['LimType'] = b_hr_df['LimType'].apply(lambda x: 'b_'+x)
    
    # Combine all limits and add feeder ID and loading scenario
    lim_hr_df = pd.concat([t_hr_df, v_hr_df, b_hr_df], axis=0)
    lim_hr_df['FeederID'] = feeder_id
    lim_hr_df['Loading_Scenario'] = loading
    
    # Remove null values and convert to int
    lim_hr_df = lim_hr_df[ pd.notnull(lim_hr_df['Limit_kW']) ]
    lim_hr_df['Limit_kW'] = lim_hr_df['Limit_kW'].astype(int)
    
    return lim_hr_df


def process_file_type_2(df, feeder_id, scenario_ls=[]):
    if len(scenario_ls)==0:
        scenario_ls = list(df['Loading_Scenario'].unique())
    else:
        scenario_ls = scenario_ls
    df_ls = []
    for loading in scenario_ls:
        df0 = find_limits_by_scenario(df, feeder_id, loading)
        df_ls.append(df0)
    df = pd.concat(df_ls, axis=0)
    return df


def main(save=True):
    start_time = dt.datetime.now()
    master_df = pd.DataFrame(
        columns=['Month','Hour','FeederID','LineSectionID','Limit','Limit_kW','LimType']
        )
    print('Processing feeder data. Feeder IDs listed below:')
    for file in file_ls:
        feeder_id = file.split('.')[0]
        print(feeder_id)
        feeder_df = pd.read_csv(os.path.join(source_dir, file))
        cols = feeder_df.columns
        if 'Loading_Scenario' in cols:
            # Only use “high” loading scenario (90th percentile)
            lim_hr_df = process_file_type_2(feeder_df, feeder_id, scenario_ls=[90])
        else:
            lim_hr_df = process_file_type_1(feeder_df, feeder_id)
        master_df = pd.concat([master_df, lim_hr_df], axis=0)
    # feedlim = master_df.copy()
    # NOTE: Dropped loading scenario and line section ID
    rename_dict = {
        'FeederID': 'feeder_id', 
        # 'Limit': 'limit', 
        'LimType': 'limit_type', 
        'Month': 'month', 
        'Hour': 'hour', 
        # 'LineSectionID': 'linesection_id', 
        'Limit_kW': 'limit_kw', 
        # 'Loading_Scenario': 'loading_scenario'
        }
    # rename columns
    feedlim = master_df.rename(columns=rename_dict)
    # pivot
    feedlim = feedlim.pivot_table(values='limit_kw', index=['feeder_id', 'month', 'hour'], columns='limit_type', aggfunc='mean')
    print(feedlim.columns)
    # rename columns again
    new_col_dict = {}
    for col in feedlim.columns:
        new_col_dict[col] = 'limit_'+col+'_kw'
    feedlim = feedlim.rename(columns=new_col_dict)
    # reset index
    feedlim = feedlim.reset_index()
    
    feedlim_cls = pd.DataFrame(feedlim.dtypes).reset_index().rename(columns={'index': 'Variable', 0: 'Class'})
    
    if save:
        print('Done. Saving feedlim file under: '+save_dir)
        feedlim.to_csv(os.path.join(save_dir, 'feedlim.csv'), index=False)
        feedlim_cls.to_csv(os.path.join(save_dir, 'feedlim_cls.csv'), index=False)
    end_time = dt.datetime.now()
    print('Total runtime:')
    print(end_time - start_time)
    return feedlim

if __name__ == "__main__":
    df = main()
    # df = main(save=False)