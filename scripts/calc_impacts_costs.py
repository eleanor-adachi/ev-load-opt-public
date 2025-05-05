# -*- coding: utf-8 -*-
"""
calc_impacts_costs.py

Created on Sun Jan 19 16:42:10 2025

@author: elean

Calculate cost of grid upgrades in dollar per kW ($/kW) by project type 
based on 2023 PG&E Distribution Deferral Opportunity Report (DDOR) and 
generation capacity avoided costs based on 2024 Avoided Cost Calculator

Instructions for downloading 2023 DDOR file:
    1. Go to https://pgera.azurewebsites.net/Regulation/search
    2. From Case dropdown menu, select "DER Modernize Electric Grid OIR"
    3. From Party dropdown menu, select "PGE"
    4. Inputs dates from "08/15/23" to "08/15/23"
    5. Click Search
    6. Download "Attachment 2: PG&E's 2023 DDOR Report Public Appendices"
    7. Unzip and extract files
    8. Save PGE_2023_DDOR_Appendix_B_Candidate Deferrals_Public.xlsx under data\raw_data\ddor

Instructions for downloading 2024 Avoided Cost Calculator (ACC):
    1. Go to https://www.cpuc.ca.gov/dercosteffectiveness
    2. Download "2024 ACC Electric Model v1b"
    3. Save 2024 ACC Electric Model v1b.xlsb under data\raw_data\acc

"""
# Import libraries
import pandas as pd
import numpy as np
import re
import json
import time
import os

# Inputs
param_file = r'..\parameters\combined_scenarios_2.json'


#%% Adjust results from net peak minimization for August only

def adj_gn0_np100_Aug_files(
        feedlim_type:str, gn_file:str=None, subneed_file:str=None, 
        np_file:str=None, gn0_file:str=None, sn0_file:str=None, save:bool=True
        ):
    '''
    Adjust results from net peak minimization for August only

    Parameters
    ----------
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    gn_file : str, default None
        Name or path of gridneed file for alpha=1.
    subneed_file : str, default None
        Name or path of subneed file for alpha=1.
    np_file : str, default None
        Name or path of netpeak file for alpha=1.
    gn0_file : str, default None
        Name or path of gridneed file for alpha=0.
    sn0_file : str, default None
        Name or path of subneed file for alpha=0.
    save : bool, default True
        Save adjusted DataFrames to CSV.

    Returns
    -------
    List of adjusted DataFrames: gridneed, subneed, netpeak
    '''
    # read net peak, grid need and subneed for alpha=1
    alpha = 1 # 100% weight on net peak
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    ft = feedlim_type
    if np_file is None:
        np_file = fr'..\results\gn{gn_pct}_np{np_pct}\netpeak_gn{gn_pct}_np{np_pct}_{ft}_Aug.csv'
    np_kw = pd.read_csv(np_file) # NOTE, net peak file requires no adjustment
    if gn_file is None:
        gn_file = fr'..\results\gn{gn_pct}_np{np_pct}\gridneed_gn{gn_pct}_np{np_pct}_{ft}_Aug.csv'
    gn_kw = pd.read_csv(gn_file)
    if subneed_file is None:
        subneed_file = fr'..\results\gn{gn_pct}_np{np_pct}\subneed_gn{gn_pct}_np{np_pct}_{ft}_Aug.csv'
    subneed = pd.read_csv(subneed_file)
    
    # read grid need and subneed for alpha=0
    alpha0 = 0 # 100% weight on grid need
    np_pct0 = int(round(alpha0*100, 0))
    gn_pct0 = 100 - np_pct0
    if gn0_file is None:
        gn0_file = fr'..\results\gn{gn_pct0}_np{np_pct0}\gridneed_gn{gn_pct0}_np{np_pct0}_{ft}.csv'
    gn_kw0 = pd.read_csv(gn0_file)
    if sn0_file is None:
        sn0_file = fr'..\results\gn{gn_pct0}_np{np_pct0}\subneed_gn{gn_pct0}_np{np_pct0}_{ft}.csv'
    subneed0 = pd.read_csv(sn0_file)
    
    # check that gn DataFrames are same dimensions and have same columns and feeder IDs
    if gn_kw.shape != gn_kw0.shape:
        raise ValueError('gridneed DataFrames are different dimensions')
    if not all(gn_kw.columns == gn_kw0.columns):
        raise ValueError('gridneed column names do not match')
    if not all(gn_kw.loc[:, 'feeder_id'] == gn_kw0.loc[:, 'feeder_id']):
        raise ValueError('gridneed feeder IDs do not match')
    
    # construct element-wise maximum of both DataFrames (fine to include feeder IDs too)
    gn_kw_adj_values = np.maximum(gn_kw.values, gn_kw0.values)
    gn_kw_adj = pd.DataFrame(data=gn_kw_adj_values, columns=gn_kw.columns)
    
    # check that subneed DataFrames are same dimensions and have same columns and substation IDs
    if subneed.shape != subneed0.shape:
        raise ValueError('subneed DataFrames are different dimensions')
    if not all(subneed.columns == subneed0.columns):
        raise ValueError('subneed column names do not match')
    if not all(subneed.loc[:, 'sub_id'] == subneed0.loc[:, 'sub_id']):
        raise ValueError('subneed substation IDs do not match')
    
    # construct element-wise maximum of both DataFrames (fine to include feeder IDs too)
    subneed_adj_values = np.maximum(subneed.values, subneed0.values)
    subneed_adj = pd.DataFrame(data=subneed_adj_values, columns=subneed.columns)
    
    if save:
        gn_kw_adj.to_csv(fr'..\results\gn{gn_pct}_np{np_pct}\gridneed_gn{gn_pct}_np{np_pct}_{ft}.csv', index=False)
        subneed_adj.to_csv(fr'..\results\gn{gn_pct}_np{np_pct}\subneed_gn{gn_pct}_np{np_pct}_{ft}.csv', index=False)
        np_kw.to_csv(fr'..\results\gn{gn_pct}_np{np_pct}\netpeak_gn{gn_pct}_np{np_pct}_{ft}.csv', index=False)
    
    return gn_kw_adj, subneed_adj, np_kw


#%% Count distribution grid upgrades

# def make_gn_ct_df(gn_kw_df):
def count_gn(gn_kw_df):
    '''
    Counts total number of feeder upgrades (grid need) for each scenario and year.
    '''
    df = gn_kw_df.copy()
    df = df.set_index('feeder_id')
    gn_kw_cols = list(df.columns)
    gn_ct_df = (df > 0).sum().to_frame(name='count').T
    gn_ct_cols = [re.sub('kW', 'ct', col) for col in gn_kw_cols]
    gn_ct_df.columns = gn_ct_cols
    return gn_ct_df


def make_gncount(alpha:float, feedlim_type:str, save:bool=True):
    '''
    Makes a DataFrame of total number of feeder upgrades (grid need) for each scenario and year.
    
    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    save : bool, default True
        Set to True to save output as CSV.
        
    Returns
    -------
    pd.DataFrame
    '''
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    df = pd.read_csv(r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv'.format(gn_pct, np_pct, feedlim_type))
    # gn_ct = make_gn_ct_df(df)
    gn_ct = count_gn(df)
    if save:
        gn_ct.to_csv(r'..\results\gn{0}_np{1}\gncount_gn{0}_np{1}_{2}.csv'.format(gn_pct, np_pct, feedlim_type), index=False)
    return gn_ct


def count_subneed(subneed_df):
    '''
    Counts total number of substation upgrades for each scenario and year.
    '''
    df = subneed_df.copy()
    df = df.set_index('sub_id')
    sub_kw_cols = list(df.columns)
    subcount_df = (df > 0).sum().to_frame(name='count').T
    subcount_cols = [re.sub('kW', 'ct', col) for col in sub_kw_cols]
    subcount_df.columns = subcount_cols
    return subcount_df


def make_subcount(alpha:float, feedlim_type:str, save:bool=True):
    '''
    Makes a DataFrame of total number of substation upgrades for each scenario and year.
    
    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    save : bool, default True
        Set to True to save output as CSV.
        
    Returns
    -------
    pd.DataFrame
    '''
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    df = pd.read_csv(r'..\results\gn{0}_np{1}\subneed_gn{0}_np{1}_{2}.csv'.format(gn_pct, np_pct, feedlim_type))
    subcount_df = count_subneed(df)
    if save:
        subcount_df.to_csv(r'..\results\gn{0}_np{1}\subcount_gn{0}_np{1}_{2}.csv'.format(gn_pct, np_pct, feedlim_type), index=False)
    return subcount_df


#%% Calculate cost of distribution grid upgrades

# create functions for aggfunc in calc_ddorval
def q1(vals:pd.Series):
    '''Function for first quartile'''
    return vals.quantile(.25)


def q3(vals:pd.Series):
    '''Function for third quartile'''
    return vals.quantile(.75)


def calc_ddorval(project_types:list=['Feeder', 'Bank and Feeder'], save:bool=False):
    '''
    Calculate min, 25th percentile, median, 75th percentile, and max cost per 
    kW ($/kW) for specific types of grid upgrades based on 2023 PG&E DDOR data

    Parameters
    ----------
    project_types : list, optional
        List of project types to calculate costs for. The default is ['Feeder', 'Bank and Feeder'].
    save : bool, default False
        Set to True to save DataFrames to CSV.

    Returns
    -------
    Dictionary of DataFrames

    '''
    # read data
    ddor_file = r'..\data\raw_data\ddor\PGE_2023_DDOR_Appendix_B_Candidate Deferrals_Public.xlsx'
    ddor = pd.read_excel(ddor_file, sheet_name='Appendix B Candidate Deferral', header=7)
    
    # check that all Grid Need expressed in MW
    all_mw = (list(ddor['Grid Need Unit'].unique()) == ['MW'])
    if not(all_mw):
        print('WARNING some Grid Need is not in MW')
        print(list(ddor['Grid Need Unit'].unique()))
    
    # filter out Customer Confidential rows & convert to numeric
    ddor = ddor[ ddor['Grid Need ']!='CC' ]
    ddor['gn_MW'] = pd.to_numeric(ddor['Grid Need '])
    
    # calculcate dolperkW
    # Note that $/kW = ($k)/MW since unit cost is expressed in $k
    ddor['dolperkW'] = ddor['Unit Cost of Traditional Mitigation ($k)']/ddor['gn_MW']
    
    # assign grid need buckets
    bins = [0,1,2,4,8,np.inf]
    group_names = ['0to1MW', '1to2MW', '2to4MW', '4to8MW', '8plusMW']
    ddor['gneedbucket'] = pd.cut(ddor['gn_MW'], bins, labels=group_names, include_lowest=True)
    
    # create dict for column renaming
    col_dict = {
        'min': 'mn_dolperkW',
        'q1': 'q1_dolperkW',
        'median': 'md_dolperkW',
        'q3': 'q3_dolperkW',
        'max': 'mx_dolperkW'
        }
    
    # initialize dictionary for DDOR values by project type
    ddorval_dict = {}
    
    # calculate costs by project type
    for p_type in project_types:
        ddor0 =  ddor[ ddor['Project Type']==p_type ]
        ddor_dolperkW = ddor0.pivot_table(
            values='dolperkW', index='gneedbucket', 
            aggfunc=['min', q1, 'median', q3, 'max'],
            observed=False
            )
        sub_group_idx_ls = [ group_names.index(x) for x in list(ddor_dolperkW.index) ]
        for i in range(len(group_names)):
            group = group_names[i]
            if (group=='0to1MW') and (p_type=='Feeder'):
                ddor_dolperkW.loc['0to1MW', :] = ddor_dolperkW.loc['1to2MW', :] * 1.4 # use 1.4x 1to2 MW costs
            elif (group=='0to1MW') and (p_type=='Bank and Feeder'):
                ratio = (ddor_dolperkW.loc['0to1MW','median'] / ddor_dolperkW.loc['1to2MW','median']).iloc[0]
                ddor_dolperkW.loc['0to1MW', :] = ddor_dolperkW.loc['1to2MW', :] * ratio # use median ratio
            # substitute row with least distance
            elif group not in ddor_dolperkW.index:
                sub_group_idx = min(sub_group_idx_ls, key=lambda x:abs(x-i))
                sub_group = group_names[ sub_group_idx ]
                ddor_dolperkW.loc[group, :] = ddor_dolperkW.loc[sub_group, :]
        ddor_dolperkW = ddor_dolperkW.sort_index()
        ddor_dolperkW.columns = ddor_dolperkW.columns.droplevel(1)
        ddor_dolperkW = ddor_dolperkW.rename(columns=col_dict)
        ddorval_dict[ p_type ] = ddor_dolperkW.copy()
        
        if save:
            if p_type == 'Feeder':
                ddor_dolperkW.to_csv(r'..\data\ddorvalcirc_dolperkW_2023.csv')
            elif p_type == 'Bank and Feeder':
                ddor_dolperkW.to_csv(r'..\data\ddorvalbank_dolperkW_2023.csv')
    
    return ddorval_dict


def calc_feed_upgrd_cost(alpha:float, feedlim_type:str, cost_type:str, gridneed_file:str=None, by_feeder:bool=False):
    '''
    Calculate cost of feeder upgrades for given alpha

    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    cost_type : str
        Type of cost to calculate. Options: mn, q1, md, q3, mx
    gridneed_file : str, default None
        Name or path of gridneed file
    by_feeder : bool, default False
        Returns total costs by scenario and year if True, else returns cost by feeder

    Returns
    -------
    DataFrame of costs for scenarios and years.

    '''
    if cost_type is None:
        raise ValueError('cost_type must be a str from this list: mn, q1, md, q3, mx')
    # calculate cost in $/kW for feeder upgrades
    ddorval_dict = calc_ddorval(project_types=['Feeder'], save=False)
    ddor = ddorval_dict['Feeder']
    
    # calculate net peak and grid need percent weight
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    
    # read grid need
    ft = feedlim_type
    if gridneed_file is None:
        gridneed_file = fr'..\results\gn{gn_pct}_np{np_pct}\gridneed_gn{gn_pct}_np{np_pct}_{ft}.csv'
    feedneed = pd.read_csv(gridneed_file)
    feedneed = feedneed.set_index('feeder_id')
    feedneed = feedneed.astype(float)
    
    # initialize cost DataFrame
    feedneed_cols = list(feedneed.columns)
    feed_cost_cols = [re.sub('gn', 'feedupgrd', col) for col in feedneed_cols]
    feed_cost_cols = [re.sub('kW', 'dol', col) for col in feed_cost_cols]
    feed_cost = pd.DataFrame(columns=feed_cost_cols)
    
    # calculate costs
    cost_type_col = cost_type + '_dolperkW'
    for i in range(len(feedneed_cols)):
        feedneed_col = feedneed_cols[i]
        feed_cost_col = feed_cost_cols[i]
        
        feedneed0 = feedneed[[ feedneed_col ]].copy()
        feedneed0 = feedneed0.reset_index()
        
        # assign grid need buckets
        bins = [-1, 0, 1000, 2000, 4000, 8000, np.inf]
        group_names = ['0MW', '0to1MW', '1to2MW', '2to4MW', '4to8MW', '8plusMW']
        feedneed0['gneedbucket'] = pd.cut(feedneed0[feedneed_col], bins, labels=group_names, include_lowest=True)
        
        # merge in costs; inner merge to take out 0MW aka no upgrade
        feedneed0 = pd.merge(feedneed0, ddor, how='inner', on='gneedbucket')
        
        # multiply by specified cost column
        feedneed0[feed_cost_col] = feedneed0[feedneed_col] * feedneed0[cost_type_col]
        
        feed_cost0 = feedneed0[['feeder_id', feed_cost_col]]
        
        # add column to feed_cost
        if feed_cost.empty:
            feed_cost = feed_cost0.copy()
        else:
            feed_cost = pd.merge(feed_cost, feed_cost0, how='outer', on='feeder_id')
    
    if not(by_feeder):
        feed_cost = feed_cost.set_index('feeder_id').sum()
        feed_cost = feed_cost.to_frame(name='Dollars ($)').T
        
    return feed_cost


def calc_gn_ct_by_bucket(alpha:float, feedlim_type:str, gridneed_file:str=None):
    '''
    Calculate number of feeder upgrades in each bucket for given alpha

    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    gridneed_file : str, default None
        Name or path of gridneed file

    Returns
    -------
    DataFrame of counts by bucket for scenarios and years.

    '''    
    # calculate net peak and grid need percent weight
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    
    # read grid need
    ft = feedlim_type
    if gridneed_file is None:
        gridneed_file = fr'..\results\gn{gn_pct}_np{np_pct}\gridneed_gn{gn_pct}_np{np_pct}_{ft}.csv'
    gn_kw = pd.read_csv(gridneed_file)
    gn_kw = gn_kw.set_index('feeder_id')
    gn_kw = gn_kw.astype(float)
    
    # initialize cost DataFrame
    gn_kw_cols = list(gn_kw.columns)
    gn_bkt_ct_cols = [re.sub('kW', 'ct', col) for col in gn_kw_cols]
    gn_bkt_ct = pd.DataFrame(columns=gn_bkt_ct_cols)
    
    # count feeders in each bucket in each scenario-year combination
    for i in range(len(gn_kw_cols)):
        gn_kw_col = gn_kw_cols[i]
        gn_bkt_ct_col = gn_bkt_ct_cols[i]
        
        gn_kw0 = gn_kw[[ gn_kw_col ]].copy()
        gn_kw0 = gn_kw0.reset_index()
        
        # assign grid need buckets
        bins = [-1, 0, 1000, 2000, 4000, 8000, np.inf]
        group_names = ['0MW', '0to1MW', '1to2MW', '2to4MW', '4to8MW', '8plusMW']
        gn_kw0[gn_bkt_ct_col] = pd.cut(gn_kw0[gn_kw_col], bins, labels=group_names, include_lowest=True)
        
        # make pivot table: count feeders in each bucket
        gn_bkt_ct0 = gn_kw0.pivot_table(index=gn_bkt_ct_col, values='feeder_id', aggfunc='count', observed=False)
        gn_bkt_ct0 = gn_bkt_ct0.rename(columns={'feeder_id':gn_bkt_ct_col})
        gn_bkt_ct0.index.name = 'gneedbucket'
        
        # add column to gn_bkt_ct
        if gn_bkt_ct.empty:
            gn_bkt_ct = gn_bkt_ct0.copy()
        else:
            gn_bkt_ct = pd.merge(gn_bkt_ct, gn_bkt_ct0, left_index=True, right_index=True)
        
    return gn_bkt_ct


def calc_sub_upgrd_cost(alpha:float, feedlim_type:str, cost_type:str, subneed_file:str=None, by_sub:bool=False):
    '''
    Calculate cost of substation upgrades for given alpha

    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    cost_type : str
        Type of cost to calculate. Options: mn, q1, md, q3, mx
    subneed_file : str, default None
        Name or path of subneed file
    by_sub : bool, default False
        Returns total costs by scenario and year if True, else returns cost by substation

    Returns
    -------
    DataFrame of costs for scenarios and years.

    '''
    if cost_type is None:
        raise ValueError('cost_type must be a str from this list: mn, q1, md, q3, mx')
    # calculate cost in $/kW for substation upgrades
    ddorval_dict = calc_ddorval(project_types=['Bank and Feeder'], save=False)
    ddor = ddorval_dict['Bank and Feeder']
    
    # read subneed
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    ft = feedlim_type
    if subneed_file is None:
        subneed_file = fr'..\results\gn{gn_pct}_np{np_pct}\subneed_gn{gn_pct}_np{np_pct}_{ft}.csv'
    subneed = pd.read_csv(subneed_file)
    subneed = subneed.set_index('sub_id')
    subneed = subneed.astype(float)
    
    # initialize cost DataFrame
    subneed_cols = list(subneed.columns)
    sub_cost_cols = [re.sub('subneed', 'subupgrd', col) for col in subneed_cols]
    sub_cost_cols = [re.sub('kW', 'dol', col) for col in sub_cost_cols]
    sub_cost = pd.DataFrame(columns=sub_cost_cols)
    
    # calculate costs
    cost_type_col = cost_type + '_dolperkW'
    for i in range(len(subneed_cols)):
        subneed_col = subneed_cols[i]
        sub_cost_col = sub_cost_cols[i]
        
        subneed0 = subneed[[ subneed_col ]].copy()
        subneed0 = subneed0.reset_index()
        
        # assign grid need buckets
        bins = [-1, 0, 1000, 2000, 4000, 8000, np.inf]
        group_names = ['0MW', '0to1MW', '1to2MW', '2to4MW', '4to8MW', '8plusMW']
        subneed0['gneedbucket'] = pd.cut(subneed0[subneed_col], bins, labels=group_names, include_lowest=True)
        
        # merge in costs; inner merge to take out 0MW aka no upgrade
        subneed0 = pd.merge(subneed0, ddor, how='inner', on='gneedbucket')
        
        # multiply by specified cost column
        subneed0[sub_cost_col] = subneed0[subneed_col] * subneed0[cost_type_col]
        
        sub_cost0 = subneed0[['sub_id', sub_cost_col]]
        
        # add column to sub_cost
        if sub_cost.empty:
            sub_cost = sub_cost0.copy()
        else:
            sub_cost = pd.merge(sub_cost, sub_cost0, how='outer', on='sub_id')
    
    if not(by_sub):
        sub_cost = sub_cost.set_index('sub_id').sum()
        sub_cost = sub_cost.to_frame(name='Dollars ($)').T
        
    return sub_cost


def calc_gn_cost(alpha:float, feedlim_type:str, cost_type:str, gridneed_file:str=None, subneed_file:str=None, save:bool=False):
    '''
    Calculate cost of feeder and substation upgrades for given alpha

    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    cost_type : str
        Type of cost to calculate. Options: mn, q1, md, q3, mx.
    gridneed_file : str, default None
        Name or path of grid need (feeder upgrade) file.
    subneed_file : str, default None
        Name or path of subneed file
    save : bool, default False
        Save results if True.

    Returns
    -------
    DataFrame of costs for scenarios and years.

    '''
    # Calculate feeder upgrade costs
    feed_cost = calc_feed_upgrd_cost(alpha, feedlim_type, cost_type, gridneed_file, by_feeder=False)
    # Calculate substation upgrade costs
    sub_cost = calc_sub_upgrd_cost(alpha, feedlim_type, cost_type, subneed_file, by_sub=False)
    
    # Combine feeder and substation upgrade costs
    gn_cost_values = feed_cost.values + sub_cost.values
    gn_cost_cols = [re.sub('feedupgrd', 'gn', col) for col in feed_cost.columns]
    gn_cost = pd.DataFrame(gn_cost_values, columns=gn_cost_cols)
    
    if save:
        # calculate net peak and grid need percent weight
        np_pct = int(round(alpha*100, 0))
        gn_pct = 100 - np_pct
        ft = feedlim_type
        gn_cost.to_csv(fr'..\results\gn{gn_pct}_np{np_pct}\gncost_gn{gn_pct}_np{np_pct}_{ft}.csv', index=False)
    
    return gn_cost


#%% Calculate generation capacity avoided costs

def get_acc_val(yr_ls:list=[2030, 2040, 2050]):
    '''
    Get generation capacity avoided costs from 2024 Avoided Cost Calculator

    Parameters
    ----------
    yr_ls : list, default [2030, 2040, 2050]
        Years to retrieve generation capacity avoided costs for from ACC.
    
    Returns
    -------
    DataFrame

    '''
    # read data
    acc_file = r'..\data\raw_data\acc\2024 ACC Electric Model v1b.xlsb'
    acc = pd.read_excel(acc_file, sheet_name='Generation Capacity', index_col=[0,1], header=2, nrows=4)
    
    # extract first row, TRC in $/kW-yr for Dollar Year, for given years
    acc = acc[yr_ls].iloc[0:1]
    
    # drop first level of MultiIndex
    acc.index = acc.index.droplevel(level=0)
    
    return acc


def create_np_kw_inc(np_kw, interp=True, interp_start=2024, interp_end=2050):
    '''
    Create np_kw_inc. If interp, interpolate between years for the same scenario.

    Parameters
    ----------
    np_kw : DataFrame
        DataFrame of net peak for each year and scenario in kilowatts.
    interp : bool
        Interpolate values between interp_start and interp_end if True.
    interp_start : int
        Start year for interpolation, inclusive. np_kw_inc assumed to be 0.
    interp_end : int
        End year for interpolation, inclusive.

    Returns
    -------
    DataFrame.

    '''
    np_kw_inc = np_kw.copy()
    np_kw_inc = np_kw_inc.drop(columns='np_kW') # drop status quo

    # calculate increase in net peak for each scenario & year compared to status quo
    for col in np_kw_inc.columns:
        np_kw_inc[col] = np_kw_inc[col] - np_kw['np_kW']
    
    # interpolate between years
    if interp:        
        # get list of scenarios
        combined_sc_ls = list(set(map(lambda x: x.split('_')[1], np_kw_inc.columns)))
        combined_sc_ls.sort()
        
        # store original version
        np_kw_inc_orig = np_kw_inc.copy()
        
        # initialize new version of np_kw_inc
        np_kw_inc = pd.DataFrame()
        for sc in combined_sc_ls:
            for opt_status in ['pre', 'post']:
                # filter columns
                if opt_status == 'pre':
                    keep_cols = list(filter(
                        lambda x: (x.split('_')[1]==sc) and not(x.endswith('opt')), 
                        np_kw_inc_orig.columns)
                        )
                elif opt_status == 'post':
                    keep_cols = list(filter(
                        lambda x: (x.split('_')[1]==sc) and x.endswith('opt'), 
                        np_kw_inc_orig.columns)
                        )
                temp_df = np_kw_inc_orig[ keep_cols ].copy()
                
                # reindex years
                yr_cols = list(map(lambda x: int(x.split('_')[2]), temp_df.columns))
                temp_df.columns = yr_cols
                temp_df = temp_df.reindex(columns=range(interp_start, interp_end+1))
                
                # set interp_start value to 0
                temp_df.loc[0,interp_start] = 0
                
                # linearly interpolate values
                temp_df = temp_df.interpolate(method='linear', axis=1)
                
                # rename columns
                if opt_status == 'pre':
                    new_cols = list(map(lambda x: f'npinc_{sc}_{x}_kW', temp_df.columns))
                elif opt_status == 'post':
                    new_cols = list(map(lambda x: f'npinc_{sc}_{x}_kW_opt', temp_df.columns))
                temp_df.columns = new_cols
                
                # concat
                if np_kw_inc.empty:
                    np_kw_inc = temp_df.copy()
                else:
                    np_kw_inc = pd.concat([np_kw_inc, temp_df], axis=1)
        
        # sort columns
        np_kw_inc_cols = list(np_kw_inc.columns)
        np_kw_inc_cols.sort()
        np_kw_inc = np_kw_inc[ np_kw_inc_cols ]
    
    # if not interpolating
    else:
        # rename columns
        np_kw_inc.columns = [re.sub('np', 'npinc', col) for col in np_kw_inc.columns]
    
    return np_kw_inc
            

def calc_np_inc_cost(alpha:float, feedlim_type:str, netpeak_file:str=None, cumulative:bool=True, save:bool=False):
    '''
    Calculate cost of generation and storage capacity costs due to increase in 
    net peak for given alpha

    Parameters
    ----------
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: 'mn', '10', 'q1', 'md', 'q3', '90', 'mx'.
    netpeak_file : str, default None
        Name or path of netpeak file.
    cumulative : bool, default True
        Calculate cumulative costs if True.
    save : bool, default False
        Save results if True.

    Returns
    -------
    DataFrame of costs for scenarios and years.

    '''
    # calculate avoided generation capacity cost in $/kW-yr
    acc = get_acc_val(yr_ls=range(2024,2051))
    
    # calculate net peak and grid need percent weight
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    
    # read net peak
    ft = feedlim_type
    if netpeak_file is None:
        netpeak_file = fr'..\results\gn{gn_pct}_np{np_pct}\netpeak_gn{gn_pct}_np{np_pct}_{ft}.csv'
    np_kw = pd.read_csv(netpeak_file)
    np_kw = np_kw.astype(float)
    
    # create np_kw_inc
    np_kw_inc = create_np_kw_inc(np_kw, interp=True, interp_start=2024, interp_end=2050)
    
    # initialize np_inc_cost_dict
    np_kw_inc_cols = list(np_kw_inc.columns)
    np_inc_cost_cols = [re.sub('kW', 'dol', col) for col in np_kw_inc_cols]
    np_inc_cost_dict = {}
    # calculate increase in net peak for each scenario & year compared to status quo
    for i in range(len(np_kw_inc_cols)):
        np_kw_inc_col = np_kw_inc_cols[i]
        np_inc_cost_col = np_inc_cost_cols[i]
        
        # calculate cost for a given year
        yr = int(np_kw_inc_col.split('_')[2])
        np_inc_cost0 = acc[yr].iloc[0] * np_kw_inc[np_kw_inc_col].iloc[0]
        
        # add to dictionary
        np_inc_cost_dict[np_inc_cost_col] = np_inc_cost0
    
    # convert to DataFrame
    np_inc_cost = pd.Series(np_inc_cost_dict)
    np_inc_cost = np_inc_cost.to_frame(name='Dollars ($)').T
    
    # find cumulative costs accrued between original years
    if cumulative:
        # get original lists of years and scenarios; skip first column (np_kW)
        yr_ls = list(set(map(lambda x: int(x.split('_')[2]), np_kw.columns[1:])))
        yr_ls.sort()
        combined_sc_ls = list(set(map(lambda x: x.split('_')[1], np_kw.columns[1:])))
        combined_sc_ls.sort()
        
        # store original version
        np_inc_cost_orig = np_inc_cost.copy()
        
        # initialize new version of np_inc_cost
        np_inc_cost = pd.DataFrame()
        for sc in combined_sc_ls:
            for opt_status in ['pre', 'post']:
                # filter columns
                if opt_status == 'pre':
                    keep_cols = list(filter(
                        lambda x: (x.split('_')[1]==sc) and not(x.endswith('opt')), 
                        np_inc_cost_orig.columns)
                        )
                elif opt_status == 'post':
                    keep_cols = list(filter(
                        lambda x: (x.split('_')[1]==sc) and x.endswith('opt'), 
                        np_inc_cost_orig.columns)
                        )
                temp_df = np_inc_cost_orig[ keep_cols ].copy()
                
                # Calculate cumulative costs
                temp_df = temp_df.cumsum(axis=1)
                
                # Select columns for original years
                keep_cols = list(filter(lambda x: int(x.split('_')[2]) in yr_ls, temp_df.columns))
                temp_df = temp_df[ keep_cols ]
                
                # concat
                if np_inc_cost.empty:
                    np_inc_cost = temp_df.copy()
                else:
                    np_inc_cost = pd.concat([np_inc_cost, temp_df], axis=1)
    
    # reorder columns
    cols = list(np_inc_cost.columns)
    cols.sort()
    np_inc_cost = np_inc_cost[ cols ]
    
    if save:
        ft = feedlim_type
        np_inc_cost.to_csv(fr'..\results\gn{gn_pct}_np{np_pct}\npinccost_gn{gn_pct}_np{np_pct}_{ft}.csv', index=False)
    
    return np_inc_cost


#%%

if __name__ == "__main__":
    # inputs
    alpha = 1 # 100% weight on net peak
    feedlim_type = 'md'
    force_refresh = True
    
    # check to see if adj files exist
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    ft = feedlim_type
    adj_files_exist = (
        os.path.exists(fr'..\results\gn{gn_pct}_np{np_pct}\gridneed_gn{gn_pct}_np{np_pct}_{ft}.csv')
        and os.path.exists(fr'..\results\gn{gn_pct}_np{np_pct}\subneed_gn{gn_pct}_np{np_pct}_{ft}.csv')
        and os.path.exists(fr'..\results\gn{gn_pct}_np{np_pct}\netpeak_gn{gn_pct}_np{np_pct}_{ft}.csv')
        )
    if force_refresh or not adj_files_exist:
        print('Adjusting net peak minimization results for August only...')
        gn_kw_adj, subneed_adj, np_kw = adj_gn0_np100_Aug_files(feedlim_type)
        time.sleep(5)
    
    # count distribution grid upgrades
    gn_ct = make_gncount(alpha, feedlim_type, save=True)
    subcount = make_subcount(alpha, feedlim_type, save=True)
    
    # calculate costs
    print('Calculating costs...')
    cost_type = 'md'
    # gn_cost_0 = calc_gn_cost(0, feedlim_type, cost_type, save=False)
    # gn_cost_1 = calc_gn_cost(1, feedlim_type, cost_type, save=False)
    # np_inc_cost_0 = calc_np_inc_cost(0, feedlim_type, save=False)
    # np_inc_cost_1 = calc_np_inc_cost(1, feedlim_type, save=False)
    gn_cost_0 = calc_gn_cost(0, feedlim_type, cost_type, save=True)
    gn_cost_1 = calc_gn_cost(1, feedlim_type, cost_type, save=True)
    np_inc_cost_0 = calc_np_inc_cost(0, feedlim_type, save=True)
    np_inc_cost_1 = calc_np_inc_cost(1, feedlim_type, save=True)