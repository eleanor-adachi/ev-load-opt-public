# -*- coding: utf-8 -*-
"""
optimize_ev_load.py

Created on Tue Jul 16 21:39:13 2024

@author: elean

Contains minimize_grid_need, minimize_net_peak_annual, and other functions 
used by run_optimization_scenarios.ipynb

"""
# import packages
import numpy as np
import pandas as pd
import cvxpy as cp
import os
import re


# default parameters
EVres_chg_h_def = [*range(0,8), *range(19,24)]
EVcom_chg_h_def = range(9,18)


def check_dir(dir_path:str, mkdir:bool=False) -> bool:
    '''
    Check if specified directory exists and makes directory if mkdir=True.

    Parameters
    ----------
    dir_path : str
        Directory path whose existence will be checked.
    mkdir : bool, optional
        Makes directory if mkdir=True. The default is False.

    Returns
    -------
    bool
        True if specified directory already existed.

    '''
    check = os.path.isdir(dir_path)
    if not(check) and mkdir:
        os.mkdir(dir_path)
        print('Created new directory: ', dir_path)
    return check


def make_gn_kw_df(df, opt=False):
    '''
    Creates a DataFrame of grid need values indexed by feeder_id
    
    df can be either input_df or output_df from run_optimization_scenarios.ipynb
    '''
    if opt:
        df['hourly_need'] = df['EVres_opt'] + df['EVcom_opt'] + df['other'] - df['limit']
    else:
        df['hourly_need'] = df['EVres'] + df['EVcom'] + df['other'] - df['limit']
    gn_kw_df = pd.pivot_table(df, values='hourly_need', index='feeder_id', aggfunc='max')
    gn_kw_df = gn_kw_df.reset_index()
    gn_kw_df = gn_kw_df.rename(columns={'hourly_need':'gn_kW'})
    if opt:
        gn_kw_df = gn_kw_df.rename(columns={'gn_kW':'gn_kW_opt'})
    return gn_kw_df


def make_subneed_df(output_df, subrat, subload):
    '''
    Creates a DataFrame of necessary substation upgrades in kW indexed by sub_id.

    Parameters
    ----------
    output_df : DataFrame
        DataFrame of optimization outputs.
    subrat : DataFrame
        DataFrame of substation ratings.
    subload : DataFrame
        DataFrame of existing substation load.

    Returns
    -------
    DataFrame of necessary substation upgrades in kW indexed by sub_id.
    '''
    # add substation rating in kW
    subrat['subrating_kW'] = subrat['subrating_MW'] * 1000
    
    # create copy of output_df
    df = output_df.copy()
    
    # add sub_id to df
    df.loc[:,'sub_id'] = df.loc[:,'feeder_id'].astype(str).str.zfill(9)
    df.loc[:,'sub_id'] = df.loc[:,'sub_id'].apply(lambda x: int(x[:5]))
    
    # create ldinc_cols and ldinc_opt_cols
    ldinc_cols = ['EVres', 'EVcom', 'other']
    ldinc_opt_cols = ['EVres_opt', 'EVcom_opt'] # other will be added back in merge
    
    # group pre-optimization added loads by sub_id and mhid
    subaddload = df.groupby(['sub_id', 'mhid'])[ldinc_cols].sum()
    subaddload = subaddload.reset_index()
    # group post-optimization added loads by sub_id and mhid
    subaddload_opt = df.groupby(['sub_id', 'mhid'])[ldinc_opt_cols].sum()
    subaddload_opt = subaddload_opt.reset_index()
    
    # compute hourly substation need = subaddload + subload - subrating
    hourly_need = pd.merge(
        subaddload, subaddload_opt, how='outer', 
        on=['sub_id', 'mhid']
        )
    hourly_need = pd.merge(
        hourly_need, subrat[['sub_id', 'subrating_kW']], how='left', 
        on='sub_id'
        )
    hourly_need = pd.merge(
        hourly_need, subload[['sub_id', 'mhid', 'h_kW']], how='left', 
        on=['sub_id', 'mhid']
        )
    hourly_need.loc[:,'ldinc'] = hourly_need[ldinc_cols].sum(axis=1)
    hourly_need['hourly_subneed'] = hourly_need['ldinc'] + hourly_need['h_kW'] - hourly_need['subrating_kW']
    hourly_need.loc[:,'ldinc_opt'] = hourly_need[ldinc_opt_cols].sum(axis=1)
    hourly_need['hourly_subneed_opt'] = hourly_need['ldinc_opt'] + hourly_need['h_kW'] - hourly_need['subrating_kW']
    
    # compute max of hourly need for each substation
    subneed = hourly_need.groupby('sub_id')[['hourly_subneed', 'hourly_subneed_opt']].max()
    subneed = subneed.reset_index()
    
    # floor subneed at 0
    subneed['subneed'] = subneed['hourly_subneed'].clip(lower=0)
    subneed['subneed_opt'] = subneed['hourly_subneed_opt'].clip(lower=0)
    
    # drop hourly subneed
    subneed = subneed.drop(columns=['hourly_subneed', 'hourly_subneed_opt'])
    
    return subneed


def make_np_kw_df(df, opt=False, status_quo=False):
    '''
    Creates a DataFrame of net peak values indexed by month
    
    df can be either input_df or output_df from run_optimization_scenarios.ipynb
    '''
    # calculate hourly net load
    df2 = df.copy()
    if status_quo:
        df2['hourly_net'] = df2['base'] - df2['vre']
    else:
        if opt:
            df2['hourly_net'] = df2['base'] + df2['EVres_opt'] + df2['EVcom_opt'] + df2['other'] - df2['vre']
        else:
            df2['hourly_net'] = df2['base'] + df2['EVres'] + df2['EVcom'] + df2['other'] - df2['vre']
    hourly_net = pd.pivot_table(df2, values='hourly_net', index='month', columns='hour', aggfunc='sum')
    
    # net peak Series
    np_kw_s = hourly_net.max(axis=1)
    if opt:
        np_kw_s.name = 'np_kW_opt'
    else:
        np_kw_s.name = 'np_kW'
    
    # net peak DataFrame
    np_kw_df = np_kw_s.reset_index()
    
    return np_kw_df


def make_netload_df(df, feedload, feedvre, include_opt=True):
    '''
    Creates a DataFrame of hourly net load in MW indexed by month, hour, and mhid
    
    df can be either input_df or output_df from run_optimization_scenarios.ipynb
    '''
    # merge data as needed
    df2 = df.copy()
    if 'h_kW' not in list(df2.columns):
        df2 = pd.merge(df2, feedload, how='outer', on=['feeder_id', 'month', 'hour', 'mhid'])
    if 'solar_kW' not in list(df2.columns):
        df2 = pd.merge(df2, feedvre, how='outer', on=['feeder_id', 'month', 'hour'])
    
    # calculate netload
    if include_opt:
        # calculate hourly net load in kW, then pivot
        df2['netload'] = df2['h_kW'] + df2['EVres'] + df2['EVcom'] + df2['other'] - df2['solar_kW'] - df2['wind_kW']
        df2['netload_opt'] = df2['h_kW'] + df2['EVres_opt'] + df2['EVcom_opt'] + df2['other'] - df2['solar_kW'] - df2['wind_kW']
        netload_df = pd.pivot_table(df2, index=['month', 'hour', 'mhid'], values=['netload', 'netload_opt'], aggfunc='sum')
        # convert to MW
        netload_df = netload_df/1000
        netload_df = netload_df.reset_index()
    else:
        # calculate hourly net load in kW, then pivot
        df2['netload'] = df2['h_kW'] + df2['EVres'] + df2['EVcom'] + df2['other'] - df2['solar_kW'] - df2['wind_kW']
        netload_df = pd.pivot_table(df2, index=['month', 'hour', 'mhid'], values='netload', aggfunc='sum')
        # convert to MW
        netload_df = netload_df/1000
        netload_df = netload_df.reset_index()
    return netload_df


def make_gnmhid_df(df, feedlim, include_opt=True):
    '''
    Creates a DataFrame of mhid in which highest grid need occurs
    
    df can be either input_df or output_df from run_optimization_scenarios.ipynb
    '''
    # merge data as needed
    df2 = df.copy()
    if 'limit' not in list(df2.columns):
        df2 = pd.merge(df2, feedlim, how='outer', on=['feeder_id', 'month', 'hour', 'mhid'])
        
    # reset index
    df2 = df2.reset_index(drop=True)
    
    # calculate hourly grid need in kW, find mhid of highest grid need
    if include_opt:
        # pre-optimization grid need
        df2['hourly_need'] = df2['EVres'] + df2['EVcom'] + df2['other'] - df2['limit']
        max_idx = df2.groupby('feeder_id').idxmax()['hourly_need']
        gnmhid_df = df2[['feeder_id','mhid']].iloc[max_idx]
        # post-optimization grid need
        df2['hourly_need_opt'] = df2['EVres_opt'] + df2['EVcom_opt'] + df2['other'] - df2['limit']
        max_idx_opt = df2.groupby('feeder_id').idxmax()['hourly_need_opt']
        gnmhid_df_opt = df2[['feeder_id','mhid']].iloc[max_idx_opt]
        gnmhid_df_opt = gnmhid_df_opt.rename(columns={'mhid':'mhid_opt'})
        # merge
        gnmhid_df = pd.merge(gnmhid_df, gnmhid_df_opt, how='outer', on='feeder_id')
    else:
        # pre-optimization grid need
        df2['hourly_need'] = df2['EVres'] + df2['EVcom'] + df2['other'] - df2['limit']
        max_idx = df2.groupby('feeder_id').idxmax()['hourly_need']
        gnmhid_df = df2[['feeder_id','mhid']].iloc[max_idx]

    return gnmhid_df


def make_npmhid_df(netload_df):
    '''
    Creates a DataFrame of mhid in which net peak occurs
    '''
    netload_col_ls = list(filter(lambda s: s.startswith('netload'), netload_df.columns))
    max_idx = netload_df[netload_col_ls].idxmax()
    npmhid_df = netload_df[['mhid']].iloc[max_idx]
    npmhid_idx = [re.sub('netload', 'npmhid', re.sub('_MW', '', col)) for col in netload_col_ls]
    npmhid_df.index = npmhid_idx
    npmhid_df = npmhid_df.T
    return npmhid_df


def minimize_grid_need(input_df, EVres_chg_h=EVres_chg_h_def, EVcom_chg_h=EVcom_chg_h_def):
    '''
    Optimize residential and commercial EV charging to minimize grid need 
    compared to known hosting capacity limits
    
    Returns two DataFrames, one with optimized load profiles and the other 
    with total grid need post-optimization
    
    Default residential charging hours are 12am-8am ("off-peak") per PG&E EV2-A tariff
    Default commercial charging hours are 9am-2pm ("super off-peak") per PG&E BEV tariff
    '''
    # prepare to receive outputs
    output_df = pd.DataFrame()
    gn_kw_opt_dict = {}
    gn_mhid_opt_dict = {}
    
    # get all feeders
    feeders = list(input_df.feeder_id.unique())
    
    # for each feeder
    for feeder in feeders:
        # filter input_df
        input_df0 = input_df[ input_df.feeder_id == feeder ]
        input_df0 = input_df0.reset_index(drop=True)
        
        # number of timesteps
        n = len(input_df0)
        
        # extract numpy arrays
        EVres_add_kw = input_df0.EVres.values
        EVcom_add_kw = input_df0.EVcom.values
        other_add_kw = input_df0.other.values
        limit_kw = input_df0.limit.values
        
        # GET EV CHARGING HOURS
        # create residential peak index for a SINGLE feeder
        EVres_peak_idx = ~input_df0.hour.isin(EVres_chg_h) # peak hours are NON-CHARGING hours
        n_EVres_peak = EVres_peak_idx.sum()
        # create commercial peak index for a SINGLE feeder
        EVcom_peak_idx = ~input_df0.hour.isin(EVcom_chg_h)  # peak hours are NON-CHARGING hours
        n_EVcom_peak = EVcom_peak_idx.sum()
        
        # create month indices
        month_idx_ls = []
        for i in range(12):
            month = i+1
            month_idx = input_df0.month == month
            month_idx_ls.append(month_idx)
        
        # check that shapes match
        match = np.all([EVres_add_kw.shape == EVcom_add_kw.shape, 
                        EVcom_add_kw.shape == other_add_kw.shape, 
                        other_add_kw.shape == limit_kw.shape
                        ])
        if not(match):
            print('WARNING some input shapes do not match')
        
        # define variables
        EVres_add_kw_opt = cp.Variable(n)
        EVcom_add_kw_opt = cp.Variable(n)
        gn_kw = cp.Variable(n)
        
        # define constraints
        constraints = [
            # no negative charging (V1G only, no V2G)
            EVres_add_kw_opt >= np.zeros(n),
            EVcom_add_kw_opt >= np.zeros(n),
            # no charging during "peak" hours
            EVres_add_kw_opt[EVres_peak_idx] - EVres_add_kw[EVres_peak_idx] == np.zeros(n_EVres_peak),
            EVcom_add_kw_opt[EVcom_peak_idx] - EVcom_add_kw[EVcom_peak_idx] == np.zeros(n_EVcom_peak),
            # calculate added load that exceeds hosting capacity limits, aka grid need
            gn_kw == (EVres_add_kw_opt + EVcom_add_kw_opt + other_add_kw - limit_kw),
        ]
        for i in range(12):
            month_idx = month_idx_ls[i]
            if sum(month_idx) > 0:
                # total charging energy is constant for EACH month
                constraints.append( (cp.sum(EVres_add_kw_opt[month_idx]) == cp.sum(EVres_add_kw[month_idx])) )
                constraints.append( (cp.sum(EVcom_add_kw_opt[month_idx]) == cp.sum(EVcom_add_kw[month_idx])) )

        # define problem: minimize grid need
        prob = cp.Problem(cp.Minimize(cp.max(gn_kw)), constraints)
    
        # solve problem
        gn_kw_opt = prob.solve()
    
        # save results and round to 3 decimals
        gn_kw_opt_dict[feeder] = round(gn_kw_opt, 3)
        gn_mhid_opt_dict[feeder] = input_df0['mhid'].iloc[gn_kw.value.argmax()]
        output_df0 = input_df0.copy()
        output_df0['EVres_opt'] = EVres_add_kw_opt.value
        output_df0['EVcom_opt'] = EVcom_add_kw_opt.value
        output_df0['EVres_opt'] = round(output_df0['EVres_opt'], 3)
        output_df0['EVcom_opt'] = round(output_df0['EVcom_opt'], 3)
        if output_df.empty:
            output_df = output_df0.copy()
        else:
            output_df = pd.concat([output_df, output_df0], axis=0)
    
    # convert gn_kw_opt_dict to DataFrame
    gn_kw_opt_df = pd.DataFrame.from_dict(gn_kw_opt_dict, orient='index', columns=['gn_kW_opt'])
    gn_kw_opt_df.index.name = 'feeder_id'
    gn_kw_opt_df = gn_kw_opt_df.reset_index()
    
    # convert gn_mhid_opt_dict to DataFrame
    gn_mhid_opt_df = pd.DataFrame.from_dict(gn_mhid_opt_dict, orient='index', columns=['gn_mhid_opt'])
    gn_mhid_opt_df.index.name = 'feeder_id'
    gn_mhid_opt_df = gn_mhid_opt_df.reset_index()    
    
    return output_df, gn_kw_opt_df, gn_mhid_opt_df


def minimize_net_peak(input_df, sys_input_df, EVres_chg_h=EVres_chg_h_def, EVcom_chg_h=EVcom_chg_h_def):
    '''
    Optimize residential and commercial EV charging to minimize net peak in each month
    
    Returns two DataFrames, one with optimized load profiles and the other 
    with net peak post-optimization
    
    Default residential charging hours are 12am-8am ("off-peak") per PG&E EV2-A tariff
    Default commercial charging hours are 9am-2pm ("super off-peak") per PG&E BEV tariff
    '''
    # prepare to receive outputs
    output_df = pd.DataFrame()
    np_kw_opt_dict = {}
    
    # for each month
    for i in range(12):
        month = i+1
        
        # filter input_df and sys_input_df
        input_df0 = input_df[ input_df.month == month ]
        input_df0 = input_df0[['feeder_id', 'month', 'hour', 'mhid', 'EVres', 'EVcom', 'other']] # added 10/14
        input_df0 = input_df0.reset_index(drop=True)
        sys_input_df0 = sys_input_df[ sys_input_df.month == month ]
        sys_input_df0 = sys_input_df0.sort_values('hour')
        sys_input_df0 = sys_input_df0.reset_index(drop=True)
        
        # number of hours
        m = len(set(input_df0['hour']))
        if m != len(sys_input_df0):
            print('WARNING number of hours in feeder-specific and system-wide data do not match')
        # number of feeders
        n = len(set(input_df0['feeder_id']))
        
        # pivot feeder-specific data
        EVres = pd.pivot_table(input_df0, values='EVres', index='hour', columns='feeder_id')
        EVcom = pd.pivot_table(input_df0, values='EVcom', index='hour', columns='feeder_id')
        other = pd.pivot_table(input_df0, values='other', index='hour', columns='feeder_id')
        
        # extract numpy arrays
        EVres_add_kw = EVres.values
        EVcom_add_kw = EVcom.values
        other_add_kw = other.values
        base_kw = sys_input_df0.base.values
        vre_kw = sys_input_df0.vre.values
        
        # GET EV CHARGING HOURS
        # create residential peak index for a SINGLE feeder
        EVres_peak_idx = ~EVres.index.isin(EVres_chg_h) # peak hours are NON-CHARGING hours
        n_EVres_peak = EVres_peak_idx.sum()
        # create commercial peak index for a SINGLE feeder
        EVcom_peak_idx = ~EVcom.index.isin(EVcom_chg_h)  # peak hours are NON-CHARGING hours
        n_EVcom_peak = EVcom_peak_idx.sum()
        
        # check that shapes match
        match = np.all([EVres_add_kw.sum(axis=1).shape == EVcom_add_kw.sum(axis=1).shape, 
                        EVcom_add_kw.sum(axis=1).shape == other_add_kw.sum(axis=1).shape, 
                        other_add_kw.sum(axis=1).shape == base_kw.shape,
                        base_kw.shape == vre_kw.shape
                        ])
        if not(match):
            print('WARNING some input shapes do not match')
        
        # get all feeders
        feeders = list(input_df0.feeder_id.unique())
        
        # define variables
        EVres_add_kw_opt = cp.Variable([m, n])
        EVcom_add_kw_opt = cp.Variable([m, n])
        netload_kw = cp.Variable(m)
        
        # define constraints
        constraints = [
            # no negative charging (V1G only, no V2G)
            EVres_add_kw_opt >= np.zeros([m, n]),
            EVcom_add_kw_opt >= np.zeros([m, n]),
            # no charging during "peak" hours
            EVres_add_kw_opt[EVres_peak_idx] - EVres_add_kw[EVres_peak_idx] == np.zeros([n_EVres_peak, n]),
            EVcom_add_kw_opt[EVcom_peak_idx] - EVcom_add_kw[EVcom_peak_idx] == np.zeros([n_EVcom_peak, n]),
            # calculate net load
            netload_kw == (base_kw + EVres_add_kw_opt.sum(axis=1) + EVcom_add_kw_opt.sum(axis=1) + other_add_kw.sum(axis=1) - vre_kw),
        ]
        # for each feeder
        for j in range(len(feeders)):
            # total charging energy is constant for EACH feeder (and each month)
            constraints.append( (cp.sum(EVres_add_kw_opt[:, j]) == cp.sum(EVres_add_kw[:, j])) )
            constraints.append( (cp.sum(EVcom_add_kw_opt[:, j]) == cp.sum(EVcom_add_kw[:, j])) )

        # define problem: minimize net peak
        prob = cp.Problem(cp.Minimize(cp.max(netload_kw)), constraints)
    
        # solve problem
        np_kw_opt = prob.solve()
        
        # create EVres_opt and EVcom_opt DataFrames, make vertical, and round to 3 decimals
        EVres_opt = pd.DataFrame(EVres_add_kw_opt.value, index=EVres.index, columns=EVres.columns)
        EVcom_opt = pd.DataFrame(EVcom_add_kw_opt.value, index=EVcom.index, columns=EVcom.columns)
        EVres_opt = EVres_opt.melt(ignore_index=False, value_name='EVres_opt').reset_index()
        EVcom_opt = EVcom_opt.melt(ignore_index=False, value_name='EVcom_opt').reset_index()
        EVres_opt['EVres_opt'] = round(EVres_opt['EVres_opt'], 3)
        EVcom_opt['EVcom_opt'] = round(EVcom_opt['EVcom_opt'], 3)
        
        # merge EVres_opt and EVcom_opt into output_df0, save results
        np_kw_opt_dict[month] = round(np_kw_opt, 3)
        output_df0 = input_df0.copy()
        output_df0 = pd.merge(output_df0, EVres_opt, how='outer', on=['feeder_id', 'hour'])
        output_df0 = pd.merge(output_df0, EVcom_opt, how='outer', on=['feeder_id', 'hour'])
        if output_df.empty:
            output_df = output_df0.copy()
        else:
            output_df = pd.concat([output_df, output_df0], axis=0)
    
    # convert np_kw_opt_dict to DataFrame
    np_kw_opt_df = pd.DataFrame.from_dict(np_kw_opt_dict, orient='index', columns=['np_kW_opt'])
    np_kw_opt_df.index.name = 'month'
    np_kw_opt_df = np_kw_opt_df.reset_index()
    
    return output_df, np_kw_opt_df


def minimize_net_peak_annual(input_df, sys_input_df=None, EVres_chg_h=EVres_chg_h_def, EVcom_chg_h=EVcom_chg_h_def):
    '''
    Optimize residential and commercial EV charging to minimize net peak increase for the entire year
    
    Returns two DataFrames, one with optimized load profiles and the other 
    with net peak post-optimization
    
    Default residential charging hours are 12am-8am ("off-peak") per PG&E EV2-A tariff
    Default commercial charging hours are 9am-2pm ("super off-peak") per PG&E BEV tariff
    '''
    # create sys_input_df if not provided
    if sys_input_df is None:
        sys_input_df = input_df[['month', 'hour', 'mhid', 'base', 'vre']].groupby(['month', 'hour', 'mhid']).sum()
        sys_input_df = sys_input_df.reset_index()
    
    # sort sys_input_df
    sys_input_df = sys_input_df.sort_values('mhid')
    sys_input_df = sys_input_df.reset_index(drop=True)
    
    # number of hours
    m = len(set(input_df['mhid']))
    if m != len(sys_input_df):
        print('WARNING number of month-hours in feeder-specific and system-wide data do not match')
    # number of feeders
    n = len(set(input_df['feeder_id']))
    
    # pivot feeder-specific data
    EVres = pd.pivot_table(input_df, values='EVres', index='mhid', columns='feeder_id')
    EVcom = pd.pivot_table(input_df, values='EVcom', index='mhid', columns='feeder_id')
    other = pd.pivot_table(input_df, values='other', index='mhid', columns='feeder_id')
    
    # extract numpy arrays
    EVres_add_kw = EVres.values
    EVcom_add_kw = EVcom.values
    other_add_kw = other.values
    base_kw = sys_input_df.base.values
    vre_kw = sys_input_df.vre.values
    
    # GET EV CHARGING HOURS
    h_pivot = pd.pivot_table(input_df, values='hour', index='mhid')
    # create residential peak index for a SINGLE feeder
    EVres_peak_idx = ~h_pivot.hour.isin(EVres_chg_h) # peak hours are NON-CHARGING hours
    n_EVres_peak = EVres_peak_idx.sum()
    # create commercial peak index for a SINGLE feeder
    EVcom_peak_idx = ~h_pivot.hour.isin(EVcom_chg_h)  # peak hours are NON-CHARGING hours
    n_EVcom_peak = EVcom_peak_idx.sum()
    
    # check that shapes match
    match = np.all([EVres_add_kw.sum(axis=1).shape == EVcom_add_kw.sum(axis=1).shape, 
                    EVcom_add_kw.sum(axis=1).shape == other_add_kw.sum(axis=1).shape, 
                    other_add_kw.sum(axis=1).shape == base_kw.shape,
                    base_kw.shape == vre_kw.shape
                    ])
    if not(match):
        print('WARNING some input shapes do not match')
    
    # get all feeders
    feeders = list(input_df.feeder_id.unique())
    
    # get all months
    months = list(input_df.month.unique())
    
    # calculate status quo net peak
    np_kw_df = make_np_kw_df(input_df, status_quo=True)
    np_kw_sq = np_kw_df['np_kW'].max()
    
    # define variables
    EVres_add_kw_opt = cp.Variable([m, n])
    EVcom_add_kw_opt = cp.Variable([m, n])
    netload_kw = cp.Variable(m)
    
    # define constraints
    constraints = [
        # no negative charging (V1G only, no V2G)
        EVres_add_kw_opt >= np.zeros([m, n]),
        EVcom_add_kw_opt >= np.zeros([m, n]),
        # no charging during "peak" hours
        EVres_add_kw_opt[EVres_peak_idx] - EVres_add_kw[EVres_peak_idx] == np.zeros([n_EVres_peak, n]),
        EVcom_add_kw_opt[EVcom_peak_idx] - EVcom_add_kw[EVcom_peak_idx] == np.zeros([n_EVcom_peak, n]),
        # calculate net load
        netload_kw == (base_kw + EVres_add_kw_opt.sum(axis=1) + EVcom_add_kw_opt.sum(axis=1) + other_add_kw.sum(axis=1) - vre_kw),
    ]
    # for each month
    m_pivot = pd.pivot_table(input_df, values='month', index='mhid')
    for month_num in months:
        month_idx = m_pivot.month == month_num
        # for each feeder
        for j in range(len(feeders)):
            # total charging energy is constant for EACH feeder (and each month)
            constraints.append( (cp.sum(EVres_add_kw_opt[month_idx, j]) == cp.sum(EVres_add_kw[month_idx, j])) )
            constraints.append( (cp.sum(EVcom_add_kw_opt[month_idx, j]) == cp.sum(EVcom_add_kw[month_idx, j])) )

    # define problem: minimize net peak increase
    prob = cp.Problem(cp.Minimize(cp.max(netload_kw) - np_kw_sq), constraints)

    # solve problem
    np_kw_opt = prob.solve()
    
    # create EVres_opt and EVcom_opt DataFrames, make vertical, and round to 3 decimals
    EVres_opt = pd.DataFrame(EVres_add_kw_opt.value, index=EVres.index, columns=EVres.columns)
    EVcom_opt = pd.DataFrame(EVcom_add_kw_opt.value, index=EVcom.index, columns=EVcom.columns)
    EVres_opt = EVres_opt.melt(ignore_index=False, value_name='EVres_opt').reset_index()
    EVcom_opt = EVcom_opt.melt(ignore_index=False, value_name='EVcom_opt').reset_index()
    EVres_opt['EVres_opt'] = round(EVres_opt['EVres_opt'], 3)
    EVcom_opt['EVcom_opt'] = round(EVcom_opt['EVcom_opt'], 3)
    
    # merge EVres_opt and EVcom_opt into output_df, save results
    output_df = input_df.copy()
    output_df = pd.merge(output_df, EVres_opt, how='outer', on=['feeder_id', 'mhid'])
    output_df = pd.merge(output_df, EVcom_opt, how='outer', on=['feeder_id', 'mhid'])
    
    # round np_kw_opt to 3 decimals
    np_kw_opt = round(np_kw_opt, 3)
    
    # find mhid when net peak occurs
    np_mhid = input_df['mhid'].iloc[netload_kw.value.argmax()]
    
    return output_df, np_kw_opt, np_mhid



def pareto_optimization(alpha, input_df, sys_input_df=None, EVres_chg_h=EVres_chg_h_def, EVcom_chg_h=EVcom_chg_h_def):
    '''
    Optimize residential and commercial EV charging to minimize the weighted 
    sum of (1) net peak increase, weighted with alpha, and (2) distribution grid need*, 
    weighted with 1 - alpha, where alpha is a number between 0.0 and 1.0
    
    *Both NEGATIVE and POSITIVE grid need counts in objective function
    
    Returns two DataFrames, one with optimized load profiles and the other 
    with weighted sums post-optimization
    
    Default residential charging hours are 12am-8am ("off-peak") per PG&E EV2-A tariff
    Default commercial charging hours are 9am-2pm ("super off-peak") per PG&E BEV tariff
    '''
    # shorthand for multipliers
    a = alpha # weight on net peak
    b = 1 - alpha # weight on grid need
    
    # create sys_input_df if not provided
    if sys_input_df is None:
        sys_input_df = input_df[['month', 'hour', 'mhid', 'base', 'vre']].groupby(['month', 'hour', 'mhid']).sum()
        sys_input_df = sys_input_df.reset_index()
        
    # sort sys_input_df
    sys_input_df = sys_input_df.sort_values('mhid')
    sys_input_df = sys_input_df.reset_index(drop=True)
    
    # number of month-hours
    m = len(set(input_df['mhid']))
    if m != len(sys_input_df):
        print('WARNING number of month-hours in feeder-specific and system-wide data do not match')
    # number of feeders
    n = len(set(input_df['feeder_id']))
    
    # pivot feeder-specific data
    EVres = pd.pivot_table(input_df, values='EVres', index='mhid', columns='feeder_id')
    EVcom = pd.pivot_table(input_df, values='EVcom', index='mhid', columns='feeder_id')
    other = pd.pivot_table(input_df, values='other', index='mhid', columns='feeder_id')
    limits = pd.pivot_table(input_df, values='limit', index='mhid', columns='feeder_id')
    
    # extract numpy arrays
    EVres_add_kw = EVres.values
    EVcom_add_kw = EVcom.values
    other_add_kw = other.values
    limit_kw = limits.values
    base_kw = sys_input_df.base.values
    vre_kw = sys_input_df.vre.values
    
    # GET EV CHARGING HOURS
    h_pivot = pd.pivot_table(input_df, values='hour', index='mhid')
    # create residential peak index for a SINGLE feeder
    EVres_peak_idx = ~h_pivot.hour.isin(EVres_chg_h) # peak hours are NON-CHARGING hours
    n_EVres_peak = EVres_peak_idx.sum()
    # create commercial peak index for a SINGLE feeder
    EVcom_peak_idx = ~h_pivot.hour.isin(EVcom_chg_h)  # peak hours are NON-CHARGING hours
    n_EVcom_peak = EVcom_peak_idx.sum()
    
    # check that shapes match
    match = np.all([EVres_add_kw.sum(axis=1).shape == EVcom_add_kw.sum(axis=1).shape, 
                    EVcom_add_kw.sum(axis=1).shape == other_add_kw.sum(axis=1).shape, 
                    other_add_kw.sum(axis=1).shape == limit_kw.sum(axis=1).shape,
                    limit_kw.sum(axis=1).shape == base_kw.shape,
                    base_kw.shape == vre_kw.shape
                    ])
    if not(match):
        print('WARNING some input shapes do not match')
    
    # get all feeders
    feeders = list(input_df.feeder_id.unique())
    
    # get all months
    months = list(input_df.month.unique())
    
    # calculate status quo net peak
    np_kw_df = make_np_kw_df(input_df, status_quo=True)
    np_kw_sq = np_kw_df['np_kW'].max()
    
    # define variables
    EVres_add_kw_opt = cp.Variable([m, n])
    EVcom_add_kw_opt = cp.Variable([m, n])
    netload_kw = cp.Variable(m)
    gn_kw = cp.Variable([m, n])
    
    # define constraints
    constraints = [
        # no negative charging (V1G only, no V2G)
        EVres_add_kw_opt >= np.zeros([m, n]),
        EVcom_add_kw_opt >= np.zeros([m, n]),
        # no charging during "peak" hours
        EVres_add_kw_opt[EVres_peak_idx] - EVres_add_kw[EVres_peak_idx] == np.zeros([n_EVres_peak, n]),
        EVcom_add_kw_opt[EVcom_peak_idx] - EVcom_add_kw[EVcom_peak_idx] == np.zeros([n_EVcom_peak, n]),
        # calculate net load
        netload_kw == (base_kw + EVres_add_kw_opt.sum(axis=1) + EVcom_add_kw_opt.sum(axis=1) + other_add_kw.sum(axis=1) - vre_kw),
        # calculate added load that exceeds hosting capacity limits, aka hourly grid need
        gn_kw == (EVres_add_kw_opt + EVcom_add_kw_opt + other_add_kw - limit_kw),
    ]
    # for each month
    m_pivot = pd.pivot_table(input_df, values='month', index='mhid')
    for month_num in months:
        month_idx = m_pivot.month == month_num
        # for each feeder
        for j in range(len(feeders)):
            # total charging energy is constant for EACH feeder (and each month)
            constraints.append( (cp.sum(EVres_add_kw_opt[month_idx, j]) == cp.sum(EVres_add_kw[month_idx, j])) )
            constraints.append( (cp.sum(EVcom_add_kw_opt[month_idx, j]) == cp.sum(EVcom_add_kw[month_idx, j])) )

    # define problem: multi-objective optimization (minimize net peak increase AND grid need) with edge cases
    if a == 0: # first edge case, a = 0
        prob = cp.Problem(cp.Minimize(cp.sum(cp.max(gn_kw, axis=0))), constraints)
    elif a == 1: # second edge case, a = 1
        prob = cp.Problem(cp.Minimize(cp.max(netload_kw)-np_kw_sq), constraints)
    else:
        prob = cp.Problem(cp.Minimize(a*(cp.max(netload_kw)-np_kw_sq) + b*cp.sum(cp.max(gn_kw, axis=0))), constraints)

    # solve problem
    wsum_opt = prob.solve()
    
    # create EVres_opt and EVcom_opt DataFrames, make vertical, and round to 3 decimals
    EVres_opt = pd.DataFrame(EVres_add_kw_opt.value, index=EVres.index, columns=EVres.columns)
    EVcom_opt = pd.DataFrame(EVcom_add_kw_opt.value, index=EVcom.index, columns=EVcom.columns)
    EVres_opt = EVres_opt.melt(ignore_index=False, value_name='EVres_opt').reset_index()
    EVcom_opt = EVcom_opt.melt(ignore_index=False, value_name='EVcom_opt').reset_index()
    EVres_opt['EVres_opt'] = round(EVres_opt['EVres_opt'], 3)
    EVcom_opt['EVcom_opt'] = round(EVcom_opt['EVcom_opt'], 3)
    
    # merge EVres_opt and EVcom_opt into output_df, save results
    output_df = input_df.copy()
    output_df = pd.merge(output_df, EVres_opt, how='outer', on=['feeder_id', 'mhid'])
    output_df = pd.merge(output_df, EVcom_opt, how='outer', on=['feeder_id', 'mhid'])
    
    # round wsum_opt to 3 decimals
    wsum_opt = round(wsum_opt, 3)
    
    return output_df, wsum_opt


def pareto_optimization_local(alpha, input_df, EVres_chg_h=EVres_chg_h_def, EVcom_chg_h=EVcom_chg_h_def):
    '''
    Optimize residential and commercial EV charging to minimize the weighted 
    sum of (1) LOCAL net peak, weighted with alpha, and (2) distribution grid need*, 
    weighted with 1 - alpha, where alpha is a number between 0.0 and 1.0
    
    LOCAL net peak means the net peak for each feeder; may not be coincident with system-wide net peak
    
    *Both NEGATIVE and POSITIVE grid need counts in objective function
    
    Returns two DataFrames, one with optimized load profiles and the other 
    with weighted sums post-optimization
    
    Default residential charging hours are 12am-8am ("off-peak") per PG&E EV2-A tariff
    Default commercial charging hours are 9am-2pm ("super off-peak") per PG&E BEV tariff
    '''
    # shorthand for multipliers
    a = alpha # weight on net peak
    b = 1 - alpha # weight on grid need
    
    # prepare to receive outputs
    output_df = pd.DataFrame()
    wsum_opt_dict = {}
    
    # get all feeders
    feeders = list(input_df.feeder_id.unique())
    
    # for each feeder
    for feeder in feeders:
        # filter input_df
        input_df0 = input_df[ input_df.feeder_id == feeder ]
        input_df0 = input_df0.reset_index(drop=True)
        
        # number of timesteps
        n = len(input_df0)
        
        # extract numpy arrays
        EVres_add_kw = input_df0.EVres.values
        EVcom_add_kw = input_df0.EVcom.values
        other_add_kw = input_df0.other.values
        limit_kw = input_df0.limit.values
        base_kw = input_df0.base.values
        vre_kw = input_df0.vre.values
        
        # GET EV CHARGING HOURS
        # create residential peak index for a SINGLE feeder
        EVres_peak_idx = ~input_df0.hour.isin(EVres_chg_h) # peak hours are NON-CHARGING hours
        n_EVres_peak = EVres_peak_idx.sum()
        # create commercial peak index for a SINGLE feeder
        EVcom_peak_idx = ~input_df0.hour.isin(EVcom_chg_h)  # peak hours are NON-CHARGING hours
        n_EVcom_peak = EVcom_peak_idx.sum()
        
        # create month indices
        month_idx_ls = []
        for i in range(12):
            month = i+1
            month_idx = input_df0.month == month
            month_idx_ls.append(month_idx)
        
        # check that shapes match
        match = np.all([EVres_add_kw.shape == EVcom_add_kw.shape, 
                        EVcom_add_kw.shape == other_add_kw.shape, 
                        other_add_kw.shape == limit_kw.shape,
                        limit_kw.shape == base_kw.shape,
                        base_kw.shape == vre_kw.shape
                        ])
        if not(match):
            print('WARNING some input shapes do not match')
        
        # define variables
        EVres_add_kw_opt = cp.Variable(n)
        EVcom_add_kw_opt = cp.Variable(n)
        netload_kw = cp.Variable(n)
        gn_kw = cp.Variable(n)
        
        # define constraints
        constraints = [
            # no negative charging (V1G only, no V2G)
            EVres_add_kw_opt >= np.zeros(n),
            EVcom_add_kw_opt >= np.zeros(n),
            # no charging during "peak" hours
            EVres_add_kw_opt[EVres_peak_idx] - EVres_add_kw[EVres_peak_idx] == np.zeros(n_EVres_peak),
            EVcom_add_kw_opt[EVcom_peak_idx] - EVcom_add_kw[EVcom_peak_idx] == np.zeros(n_EVcom_peak),
            # calculate added load that exceeds hosting capacity limits, aka grid need
            gn_kw == (EVres_add_kw_opt + EVcom_add_kw_opt + other_add_kw - limit_kw),
            # calculate net load profile
            netload_kw == (base_kw + EVres_add_kw_opt + EVcom_add_kw_opt + other_add_kw - vre_kw),
        ]
        for i in range(12):
            month_idx = month_idx_ls[i]
            if sum(month_idx) > 0:
                # total charging energy is constant for EACH month
                constraints.append( (cp.sum(EVres_add_kw_opt[month_idx]) == cp.sum(EVres_add_kw[month_idx])) )
                constraints.append( (cp.sum(EVcom_add_kw_opt[month_idx]) == cp.sum(EVcom_add_kw[month_idx])) )
        
        # define problem: multi-objective optimization (net peak AND grid need)
        if a == 0: # first edge case, a = 0
            prob = cp.Problem(cp.Minimize(cp.max(gn_kw)), constraints)
        elif a == 1: # second edge case, a = 1
            prob = cp.Problem(cp.Minimize(cp.max(netload_kw)), constraints)
        else:
            prob = cp.Problem(cp.Minimize(a*cp.max(netload_kw) + b*cp.max(gn_kw)), constraints)

        # solve problem
        wsum_opt = prob.solve()
    
        # save results and round to 3 decimals
        wsum_opt_dict[feeder] = round(wsum_opt, 3)
        output_df0 = input_df0.copy()
        output_df0['EVres_opt'] = EVres_add_kw_opt.value
        output_df0['EVcom_opt'] = EVcom_add_kw_opt.value
        output_df0['EVres_opt'] = round(output_df0['EVres_opt'], 3)
        output_df0['EVcom_opt'] = round(output_df0['EVcom_opt'], 3)
        if output_df.empty:
            output_df = output_df0.copy()
        else:
            output_df = pd.concat([output_df, output_df0], axis=0)
    
    # convert wsum_opt_dict to DataFrame
    wsum_opt_df = pd.DataFrame.from_dict(wsum_opt_dict, orient='index', columns=['wsum_opt'])
    wsum_opt_df.index.name = 'feeder_id'
    wsum_opt_df = wsum_opt_df.reset_index()
    
    return output_df, wsum_opt_df