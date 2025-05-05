# -*- coding: utf-8 -*-
"""
make_plots.py

Created on Fri Sep 27 18:09:19 2024

@author: elean
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from calc_impacts_costs import count_gn, count_subneed, calc_gn_cost, calc_np_inc_cost, create_np_kw_inc


def check_figures_dir(alpha:float=None, mkdir:bool=False) -> bool:
    '''
    Check if figures directory exists and makes directory if mkdir=True.

    Parameters
    ----------
    alpha : float, default None
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
        If None, makes parent figures directory and not for a specific alpha.
    mkdir : bool, optional
        Makes directory if mkdir=True. The default is False.

    Returns
    -------
    bool
        True if directory already existed.

    '''
    # create figures path
    if alpha is None:
        figures_path = r'..\figures'
    else:
        # calculate net peak and grid need percent weight
        np_pct = int(round(alpha*100, 0))
        gn_pct = 100 - np_pct
        figures_path = r'..\figures\gn%d_np%d' % (gn_pct, np_pct)
    
    # check if directory exists
    check = os.path.isdir(figures_path)
        
    # make directory
    if not(check) and mkdir:
        os.mkdir(figures_path)
        print('Created new directory: ', figures_path)
        
    return check





gn_cost_type_dict = {
    'mn': 'minimum', 
    'q1': '25th pctl', 
    'md': 'median', 
    'q3': '75th pctl', 
    'mx': 'maximum'
    }


sc_color_dict = {
    'A': sns.color_palette("PuBu", 7)[4],
    'B': sns.color_palette("Greens", 7)[4],
    'C': sns.color_palette("Reds", 7)[4],
    }


pre_plot_dict = {
    'gn_gw_sum': {
        'input_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Distribution Grid Need (GW)',
        'plot_name': r'..\figures\gn_gw_sum_preopt_{0}.png',
        },
    'np_gw_inc': {
        'input_file': r'..\results\gn{0}_np{1}\netpeak_gn{0}_np{1}_{2}.csv',
        'y_label': 'Increase in Net Peak (GW)',
        'plot_name': r'..\figures\np_gw_inc_preopt_{0}.png',
        },
    'gn_ct': {
        'input_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Number of Feeder Upgrades',
        'plot_name': r'..\figures\gn_ct_preopt_{0}.png',
        },
    'subcount': {
        'input_file': r'..\results\gn{0}_np{1}\subneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Number of Substation Upgrades',
        'plot_name': r'..\figures\subcount_preopt_{0}.png',
        },
    'gn_cost': {
        'gridneed_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'subneed_file': r'..\results\gn{0}_np{1}\subneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Cost of Grid Upgrades ($ Billion, %s)',
        'plot_name': r'..\figures\gn_cost_preopt_{0}.png',
        },
    'np_inc_cost': {
        'input_file':  r'..\results\gn{0}_np{1}\netpeak_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Generation Capacity Costs ($ Billion)',
        'plot_name': r'..\figures\np_inc_cost_preopt_{0}.png',
        },
    }


def make_pre_opt_plot(plot_type:str, feedlim_type:str, gn_cost_type:str=None, save:bool=True):
    '''
    Create plots of pre--optimization values of a specified type--distribution 
    grid need, net peak increase, number of distribution grid upgrades, 
    distribution grid upgrade costs, or cost of generation capacity 
    procurement--across scenarios and years.

    Parameters
    ----------
    plot_type : str
        Type of plot to make. Options: gn_gw_sum, np_gw_inc, gn_ct, subcount, gn_cost, np_inc_cost.
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    gn_cost_type : str, default None
        Type of cost to calculate for grid need cost. Options: mn, q1, md, q3, mx.
    save : bool, default True
        Set to True to save plot to PNG.
        
    Returns
    -------
    fig, axs

    '''
    # calculate net peak and grid need percent weight
    alpha = 0 # arbitrary--just need to get the input data (pre-optimization values same for both alphas)
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    
    # read input file
    if plot_type == 'gn_cost':
        gridneed_file = (pre_plot_dict[plot_type]['gridneed_file']).format(gn_pct, np_pct, feedlim_type)
        subneed_file = (pre_plot_dict[plot_type]['subneed_file']).format(gn_pct, np_pct, feedlim_type)
    else:
        input_file = (pre_plot_dict[plot_type]['input_file']).format(gn_pct, np_pct, feedlim_type)
        df = pd.read_csv(input_file)
    
    # create plot_s (plot series)
    if plot_type == 'gn_gw_sum':
        df = df.set_index('feeder_id')
        df = df.astype(float)
        # compute totals and convert to GW
        plot_s = df.sum(axis=0)
        plot_s = plot_s/1000/1000
    elif plot_type == 'np_gw_inc':
        df0 = df.copy()
        df = df.drop(columns='np_kW') # drop status quo
        # calculate increase in net peak for each scenario & year compared to status quo
        for col in df.columns:
            df[col] = df[col] - df0['np_kW']
        # convert to GW
        df = df/1000/1000
        # convert to series
        plot_s = df.iloc[0, :]
    elif plot_type == 'gn_ct':
        # calculate grid need count (number of grid upgrades)
        gn_ct = count_gn(df)
        # convert to series
        plot_s = gn_ct.iloc[0, :]
    elif plot_type == 'subcount':
        # calculate count of substation upgrades
        subcount = count_subneed(df)
        # convert to series
        plot_s = subcount.iloc[0, :]
    elif plot_type == 'gn_cost':
        # calculate cost of distribution grid upgrades
        gn_cost = calc_gn_cost(alpha, feedlim_type, gn_cost_type, gridneed_file=gridneed_file, subneed_file=subneed_file)
        # convert to billions ($)
        gn_cost = gn_cost/(10**9)
        # convert to series
        plot_s = gn_cost.iloc[0, :]
    elif plot_type == 'np_inc_cost':
        # calculate cost of distribution grid upgrades
        np_inc_cost = calc_np_inc_cost(alpha, feedlim_type, netpeak_file=input_file, cumulative=True)
        # convert to billions ($)
        np_inc_cost = np_inc_cost/(10**9)
        # convert to series
        plot_s = np_inc_cost.iloc[0, :]
    
    # extract scenarios and years
    combined_sc_ls = list(set(map(lambda x: x.split('_')[1], plot_s.index)))
    combined_sc_ls.sort()
    yr_ls = list(set(map(lambda x: int(x.split('_')[2]), plot_s.index)))
    yr_ls.sort()
    
    # get pre-optimization index
    pre_opt_idx = list(filter(lambda x: not(x.endswith('_opt')), plot_s.index))

    # get y-label and plot name
    y_label = pre_plot_dict[plot_type]['y_label']
    plot_name = pre_plot_dict[plot_type]['plot_name']

    # make plot
    bar_width = 0.12
    figscale = 5
    fig, ax = plt.subplots(1, 1, figsize=(figscale, figscale))
    y0 = None
    x_previous = None
    # create bar chart for each combination of dm and sc
    for j in range(len(combined_sc_ls)):
        sc_id = combined_sc_ls[j]
        stage_sc_idx = list(filter(lambda x: x.split('_')[1]==sc_id, pre_opt_idx))
        y = plot_s[ stage_sc_idx ]
        if j == 0:
            x = np.arange(len(y))
            y0 = y
        else:
            x = [x_val + bar_width for x_val in x_previous]
        ax.bar(x, y, width=bar_width, edgecolor='white', label=sc_id, color=sc_color_dict[sc_id])
        x_previous = x
    # add x-ticks on the middle of the group bars
    ax.set_xticks([r + bar_width for r in range(len(y0))], yr_ls)
    # set y-limits
    ax.set_ylim([0, 1.05*plot_s.max()])
    # add y-labels
    if plot_type == 'gn_cost':
        cost_type_label = gn_cost_type_dict[gn_cost_type]
        ax.set_ylabel(y_label % cost_type_label)
    else:
        ax.set_ylabel(y_label)
    # add legend
    ax.legend()
    # add horizontal grid lines
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='y')
    
    # save graphic
    if save:
        plt.savefig(plot_name.format(feedlim_type), bbox_inches='tight')

    # show graphic
    plt.show()
    
    return fig, ax


pre_post_plot_dict = {
    'gn_gw_sum': {
        'input_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Distribution Grid Need (GW)',
        'plot_name': r'..\figures\gn{0}_np{1}\gn_gw_sum_gn{0}_np{1}_{2}.png',
        },
    'np_gw_inc': {
        'input_file': r'..\results\gn{0}_np{1}\netpeak_gn{0}_np{1}_{2}.csv',
        'y_label': 'Increase in Net Peak (GW)',
        'plot_name': r'..\figures\gn{0}_np{1}\np_gw_inc_gn{0}_np{1}_{2}.png',
        },
    'gn_ct': {
        'input_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Number of Feeder Upgrades',
        'plot_name': r'..\figures\gn{0}_np{1}\gn_ct_gn{0}_np{1}_{2}.png',
        },
    'subcount': {
        'input_file': r'..\results\gn{0}_np{1}\subneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Number of Substation Upgrades',
        'plot_name': r'..\figures\gn{0}_np{1}\subcount_gn{0}_np{1}_{2}.png',
        },
    'gn_cost': {
        'gridneed_file': r'..\results\gn{0}_np{1}\gridneed_gn{0}_np{1}_{2}.csv',
        'subneed_file': r'..\results\gn{0}_np{1}\subneed_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Cost of Grid Upgrades ($ Billion, %s)',
        'plot_name': r'..\figures\gn{0}_np{1}\gn_cost_gn{0}_np{1}_{2}.png',
        },
    'np_inc_cost': {
        'input_file':  r'..\results\gn{0}_np{1}\netpeak_gn{0}_np{1}_{2}.csv',
        'y_label': 'Cumulative Generation Capacity Costs ($ Billion)',
        'plot_name': r'..\figures\gn{0}_np{1}\np_inc_cost_gn{0}_np{1}_{2}.png',
        },
    }


def make_pre_post_opt_plot(plot_type:str, alpha:float, feedlim_type:str, gn_cost_type:str=None, save:bool=True):
    '''
    Create plots of pre- and post-optimization values of a specified type--
    distribution grid need, net peak increase, number of distribution grid 
    upgrades, distribution grid upgrade costs, or cost of generation capacity 
    procurement--for a given alpha across scenarios and years.

    Parameters
    ----------
    plot_type : str
        Type of plot to make. Options: gn_gw_sum, np_gw_inc, gn_ct, subcount, gn_cost, np_inc_cost.
    alpha : float
        Weight for net peak in pareto optimization; weight for grid need is 1 - alpha.
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    gn_cost_type : str, default None
        Type of cost to calculate for grid need cost. Options: mn, q1, md, q3, mx.
    save : bool, default True
        Set to True to save plot to PNG.
        
    Returns
    -------
    fig, axs

    '''
    # calculate net peak and grid need percent weight
    np_pct = int(round(alpha*100, 0))
    gn_pct = 100 - np_pct
    
    # read input file
    if plot_type == 'gn_cost':
        gridneed_file = (pre_post_plot_dict[plot_type]['gridneed_file']).format(gn_pct, np_pct, feedlim_type)
        subneed_file = (pre_post_plot_dict[plot_type]['subneed_file']).format(gn_pct, np_pct, feedlim_type)
    else:
        input_file = (pre_post_plot_dict[plot_type]['input_file']).format(gn_pct, np_pct, feedlim_type)
        df = pd.read_csv(input_file)
    
    # create plot_s (plot series)
    if plot_type == 'gn_gw_sum':
        df = df.set_index('feeder_id')
        df = df.astype(float)
        # compute totals and convert to GW
        plot_s = df.sum(axis=0)
        plot_s = plot_s/1000/1000
    elif plot_type == 'np_gw_inc':
        df0 = df.copy()
        df = df.drop(columns='np_kW') # drop status quo
        # calculate increase in net peak for each scenario & year compared to status quo
        for col in df.columns:
            df[col] = df[col] - df0['np_kW']
        # convert to GW
        df = df/1000/1000
        # convert to series
        plot_s = df.iloc[0, :]
    elif plot_type == 'gn_ct':
        # calculate grid need count (number of grid upgrades)
        gn_ct = count_gn(df)
        # convert to series
        plot_s = gn_ct.iloc[0, :]
    elif plot_type == 'subcount':
        # calculate count of substation upgrades
        subcount = count_subneed(df)
        # convert to series
        plot_s = subcount.iloc[0, :]
    elif plot_type == 'gn_cost':
        # calculate cost of distribution grid upgrades
        gn_cost = calc_gn_cost(alpha, feedlim_type, gn_cost_type, gridneed_file=gridneed_file, subneed_file=subneed_file)
        # convert to billions ($)
        gn_cost = gn_cost/(10**9)
        # convert to series
        plot_s = gn_cost.iloc[0, :]
    elif plot_type == 'np_inc_cost':
        # calculate cost of distribution grid upgrades
        np_inc_cost = calc_np_inc_cost(alpha, feedlim_type, netpeak_file=input_file, cumulative=True)
        # convert to billions ($)
        np_inc_cost = np_inc_cost/(10**9)
        # convert to series
        plot_s = np_inc_cost.iloc[0, :]
    
    # extract scenarios and years
    combined_sc_ls = list(set(map(lambda x: x.split('_')[1], plot_s.index)))
    combined_sc_ls.sort()
    yr_ls = list(set(map(lambda x: int(x.split('_')[2]), plot_s.index)))
    yr_ls.sort()
    
    # get pre- and post-optimization index
    pre_opt_idx = list(filter(lambda x: not(x.endswith('_opt')), plot_s.index))
    post_opt_idx = list(filter(lambda x: x.endswith('_opt'), plot_s.index))

    # get y-label and plot name
    y_label = pre_post_plot_dict[plot_type]['y_label']
    plot_name = pre_post_plot_dict[plot_type]['plot_name']

    # make plot
    bar_width = 0.12
    stage_ls = ['Pre-Optimization', 'Post-Optimization']
    num_stages = len(stage_ls)
    figscale = 5
    fig, axs = plt.subplots(1, num_stages, figsize=(figscale*num_stages, figscale*1))
    for i in range(num_stages):
        stage = stage_ls[i]
        ax = axs[i]
        y0 = None
        x_previous = None
        # create bar chart for each combination of dm and sc
        for j in range(len(combined_sc_ls)):
            sc_id = combined_sc_ls[j]
            if stage == 'Pre-Optimization':
                stage_sc_idx = list(filter(lambda x: x.split('_')[1]==sc_id, pre_opt_idx))
            elif stage == 'Post-Optimization':
                stage_sc_idx = list(filter(lambda x: x.split('_')[1]==sc_id, post_opt_idx))
            y = plot_s[ stage_sc_idx ]
            if j == 0:
                x = np.arange(len(y))
                y0 = y
            else:
                x = [x_val + bar_width for x_val in x_previous]
            ax.bar(x, y, width=bar_width, edgecolor='white', label=sc_id, color=sc_color_dict[sc_id])
            x_previous = x
        # add subtitle
        ax.title.set_text(stage)
        # add x-ticks on the middle of the group bars
        ax.set_xticks([r + bar_width for r in range(len(y0))], yr_ls)
        # set y-limits
        ax.set_ylim([0, 1.05*plot_s.max()])
        # add y-labels if first column
        if i == 0:
            if plot_type == 'gn_cost':
                cost_type_label = gn_cost_type_dict[gn_cost_type]
                ax.set_ylabel(y_label % cost_type_label)
            else:
                ax.set_ylabel(y_label)
        # add legend
        ax.legend()
        # add horizontal grid lines
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='y')
    
    # save graphic
    if save:
        plt.savefig(plot_name.format(gn_pct, np_pct, feedlim_type), bbox_inches='tight')

    # show graphic
    plt.show()
    
    return fig, axs


color_label_dict = {
    'A': {
        2030: {
            'color': sns.color_palette("PuBu", 7)[2], 
            'label': 'Sc. A, 2030'
            },
        2040: {
            'color': sns.color_palette("PuBu", 7)[4],
            'label': 'Sc. A, 2040'
            },
        2050: {
            'color': sns.color_palette("PuBu", 7)[6],
            'label': 'Sc. A, 2050'
            }
        },
    'B': {
        2030: {
            'color': sns.color_palette("Greens", 7)[2],
            'label': 'Sc. B, 2030'
            },
        2040: {
            'color': sns.color_palette("Greens", 7)[4],
            'label': 'Sc. B, 2040'
            },
        2050: {
            'color': sns.color_palette("Greens", 7)[6],
            'label': 'Sc. B, 2050'
            }
        },
    'C': {
        2030: {
            'color': sns.color_palette("Reds", 7)[2],
            'label': 'Sc. C, 2030'
            },
        2040: {
            'color': sns.color_palette("Reds", 7)[4],
            'label': 'Sc. C, 2040'
            },
        2050: {
            'color': sns.color_palette("Reds", 7)[6],
            'label': 'Sc. C, 2050'
            }
        }
    }


def plot_gw_tradeoffs(
        feedlim_type:str, sc_ls:list=None, include_sq:bool=False, 
        save:bool=False
        ):
    '''
    Create plots of tradeoffs between net peak increase in GW (x-axis) and 
    total distribution grid upgrade capacity in GW (y-axis) for different
    scenarios, years, and values of alpha.

    Parameters
    ----------
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    sc_ls : list, default None
        List of scenarios to include. If None, includes all scenarios.
    include_sq : bool, default False
        Set to True to include pre-optimization (status quo).
    save : bool, default False
        Set to True to save plot to PNG.
        
    Returns
    -------
    fig

    '''
    # read grid need and net peak results
    gn_0 = pd.read_csv(r'..\results\gn100_np0\gridneed_gn100_np0_%s.csv' % feedlim_type)
    gn_1 = pd.read_csv(r'..\results\gn0_np100\gridneed_gn0_np100_%s.csv' % feedlim_type)
    np_0 = pd.read_csv(r'..\results\gn100_np0\netpeak_gn100_np0_%s.csv' % feedlim_type)
    np_1 = pd.read_csv(r'..\results\gn0_np100\netpeak_gn0_np100_%s.csv' % feedlim_type)
    
    # sum up grid need and convert to GW
    gn_0 = gn_0.set_index('feeder_id').astype(float)
    gn_1 = gn_1.set_index('feeder_id').astype(float)
    gn_0 = (gn_0.sum(axis=0)/1000/1000).to_frame().T
    gn_1 = (gn_1.sum(axis=0)/1000/1000).to_frame().T
    gn_0.columns = [re.sub('kW', 'GW', col) for col in gn_0.columns]
    gn_1.columns = [re.sub('kW', 'GW', col) for col in gn_1.columns]
    
    # calculate net peak increases and convert to GW
    np_inc_0 = create_np_kw_inc(np_0, interp=False)
    np_inc_1 = create_np_kw_inc(np_1, interp=False)
    np_inc_0 = np_inc_0/1000/1000
    np_inc_1 = np_inc_1/1000/1000
    np_inc_0.columns = [re.sub('kW', 'GW', col) for col in np_inc_0.columns]
    np_inc_1.columns = [re.sub('kW', 'GW', col) for col in np_inc_1.columns]
    
    # filter for pre-optimization (status quo)
    gn_sq = gn_0[ list(filter(lambda s: not(s.endswith('_opt')), gn_0.columns)) ]
    np_inc_sq = np_inc_0[ list(filter(lambda s: not(s.endswith('_opt')), np_inc_0.columns)) ]
    
    # filter for post-optimization only
    gn_0 = gn_0[ list(filter(lambda s: s.endswith('_opt'), gn_0.columns)) ]
    gn_1 = gn_1[ list(filter(lambda s: s.endswith('_opt'), gn_1.columns)) ]
    np_inc_0 = np_inc_0[ list(filter(lambda s: s.endswith('_opt'), np_inc_0.columns)) ]
    np_inc_1 = np_inc_1[ list(filter(lambda s: s.endswith('_opt'), np_inc_1.columns)) ]
    
    # get lists of years and scenarios
    yr_ls = list(set(map(lambda x: int(x.split('_')[2]), gn_0.columns)))
    yr_ls.sort()
    if sc_ls is None:
        sc_ls = list(set(map(lambda x: x.split('_')[1], gn_0.columns)))
        sc_ls.sort()
        fig_name =  r'..\figures\gn_np_gw_tradeoffs_%s.png' % feedlim_type
    else:
        fig_name = r'..\figures\gn_np_gw_tradeoffs_%s' % feedlim_type + \
            '_' + ''.join(sc_ls) + '.png'
    
    fig_scale = 5
    np_inc_max = max(np_inc_0.values.max(), np_inc_1.values.max())
    gn_max = max(gn_0.values.max(), gn_1.values.max())
    fig = plt.figure(figsize=(fig_scale*np_inc_max/gn_max, fig_scale))
    for sc in sc_ls:
        for yr in yr_ls:
            sc_yr_color = color_label_dict[sc][yr]['color']
            sc_yr_label = color_label_dict[sc][yr]['label']
            if include_sq:
                gn_col_sq = list(filter(lambda s: f'{sc}_{yr}' in s, gn_sq.columns))[0]
                np_col_sq = list(filter(lambda s: f'{sc}_{yr}' in s, np_inc_sq.columns))[0]
                plt.scatter(np_inc_sq[np_col_sq], gn_sq[gn_col_sq], s=50, marker='x', color=sc_yr_color, label=sc_yr_label+', no opt.')
            gn_col = list(filter(lambda s: f'{sc}_{yr}' in s, gn_0.columns))[0]
            np_col = list(filter(lambda s: f'{sc}_{yr}' in s, np_inc_0.columns))[0]
            plt.scatter(np_inc_0[np_col], gn_0[gn_col], s=50, facecolors='none', edgecolors=sc_yr_color, label=sc_yr_label+r', $\alpha$=0')
            plt.scatter(np_inc_1[np_col], gn_1[gn_col], s=50, marker='d', color=sc_yr_color, label=sc_yr_label+r', $\alpha$=1')
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size': 8})
    plt.xlabel('Net Peak Increase (GW)')
    plt.ylabel('Distribution Grid Upgrade (GW)')
    ax = plt.gca() # get current axes
    ax.set_aspect('equal', adjustable='box') # set aspect ratio equal
    
    if save:
        plt.savefig(fig_name, bbox_inches='tight', dpi=200)
    
    plt.show()
    
    return fig


def plot_cost_tradeoffs(
        feedlim_type:str, gn_cost_type:str, sc_ls:list=None, include_sq:bool=False, 
        save:bool=False
        ):
    '''
    Create plots of tradeoffs between cumulative generation capacity costs in 
    $ Billion (x-axis) and cumulative distribution grid upgrade costs in 
    $ Billion (y-axis) for different scenarios, years, and values of alpha.

    Parameters
    ----------
    feedlim_type : str
        Type of feeder limit. Options: mn, 10, q1, md, q3, 90, mx.
    gn_cost_type : str
        Type of cost to calculate for grid need cost. Options: mn, q1, md, q3, mx.
    sc_ls : list, default None
        List of scenarios to include. If None, includes all scenarios.
    include_sq : bool, default False
        Set to True to include pre-optimization (status quo).
    save : bool, default False
        Set to True to save plot to PNG.
        
    Returns
    -------
    fig

    '''
    # calculate costs
    gn_cost_0 = calc_gn_cost(0, feedlim_type, gn_cost_type)
    gn_cost_1 = calc_gn_cost(1, feedlim_type, gn_cost_type)
    np_inc_cost_0 = calc_np_inc_cost(0, feedlim_type, cumulative=True)
    np_inc_cost_1 = calc_np_inc_cost(1, feedlim_type, cumulative=True)
    
    # convert to billions ($)
    gn_cost_0 = gn_cost_0/(10**9)
    gn_cost_1 = gn_cost_1/(10**9)
    np_inc_cost_0 = np_inc_cost_0/(10**9)
    np_inc_cost_1 = np_inc_cost_1/(10**9)
    
    # filter for pre-optimization (status quo)
    gn_cost_sq = gn_cost_0[ list(filter(lambda s: not(s.endswith('_opt')), gn_cost_0.columns)) ]
    np_inc_cost_sq = np_inc_cost_0[ list(filter(lambda s: not(s.endswith('_opt')), np_inc_cost_0.columns)) ]
    
    # filter for post-optimization only
    gn_cost_0 = gn_cost_0[ list(filter(lambda s: s.endswith('_opt'), gn_cost_0.columns)) ]
    gn_cost_1 = gn_cost_1[ list(filter(lambda s: s.endswith('_opt'), gn_cost_1.columns)) ]
    np_inc_cost_0 = np_inc_cost_0[ list(filter(lambda s: s.endswith('_opt'), np_inc_cost_0.columns)) ]
    np_inc_cost_1 = np_inc_cost_1[ list(filter(lambda s: s.endswith('_opt'), np_inc_cost_1.columns)) ]
    
    # get lists of years and scenarios
    yr_ls = list(set(map(lambda x: int(x.split('_')[2]), gn_cost_0.columns)))
    yr_ls.sort()
    if sc_ls is None:
        sc_ls = list(set(map(lambda x: x.split('_')[1], gn_cost_0.columns)))
        sc_ls.sort()
        fig_name = r'..\figures\gn_np_cost_tradeoffs_%s.png' % feedlim_type
    else:
        fig_name = r'..\figures\gn_np_cost_tradeoffs_%s' % feedlim_type + \
            '_' + ''.join(sc_ls) + '.png'
    
    fig_scale = 5
    np_inc_cost_max = max(np_inc_cost_0.values.max(), np_inc_cost_1.values.max())
    gn_cost_max = max(gn_cost_0.values.max(), gn_cost_1.values.max())
    fig = plt.figure(figsize=(fig_scale*np_inc_cost_max/gn_cost_max, fig_scale))
    for sc in sc_ls:
        for yr in yr_ls:
            sc_yr_color = color_label_dict[sc][yr]['color']
            sc_yr_label = color_label_dict[sc][yr]['label']
            if include_sq:
                gn_col_sq = list(filter(lambda s: f'{sc}_{yr}' in s, gn_cost_sq.columns))[0]
                np_col_sq = list(filter(lambda s: f'{sc}_{yr}' in s, np_inc_cost_sq.columns))[0]
                plt.scatter(np_inc_cost_sq[np_col_sq], gn_cost_sq[gn_col_sq], s=50, marker='x', color=sc_yr_color, label=sc_yr_label+', no opt.')
            gn_col = list(filter(lambda s: f'{sc}_{yr}' in s, gn_cost_0.columns))[0]
            np_col = list(filter(lambda s: f'{sc}_{yr}' in s, np_inc_cost_0.columns))[0]
            plt.scatter(np_inc_cost_0[np_col], gn_cost_0[gn_col], s=50, facecolors='none', edgecolors=sc_yr_color, label=sc_yr_label+r', $\alpha$=0')
            plt.scatter(np_inc_cost_1[np_col], gn_cost_1[gn_col], s=50, marker='d', color=sc_yr_color, label=sc_yr_label+r', $\alpha$=1')
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), prop={'size': 8})
    plt.xlabel('Generation Capacity Costs ($ Billion)')
    plt.ylabel('Distribution Grid Upgrade Costs ($ Billion)')
    
    # Set aspect ratio and tick spacing equal
    ax = plt.gca() # get current axes
    ax.set_aspect('equal', adjustable='box') # set aspect ratio equal
    ax.apply_aspect() # for getting new ticks
    current_x_ticks = ax.get_xticks()
    current_y_ticks = ax.get_yticks()
    new_tick_spacing = max(current_x_ticks[1] - current_x_ticks[0], current_y_ticks[1] - current_y_ticks[0])
    plt.xticks(np.arange(min(current_x_ticks), max(current_x_ticks), new_tick_spacing)) 
    plt.yticks(np.arange(min(current_y_ticks), max(current_y_ticks), new_tick_spacing)) 
    
    if save:
        plt.savefig(fig_name, bbox_inches='tight', dpi=200)
    
    plt.show()
    
    return fig


#%%

if __name__ == "__main__":
    feedlim_type = 'md'
    gn_cost_type = 'md'
    save_figs = False
    
    check_figures = check_figures_dir(mkdir=save_figs)
    
    # Make pre-optimization plots
    make_pre_opt_plot('gn_gw_sum', feedlim_type, save=save_figs)
    make_pre_opt_plot('np_gw_inc', feedlim_type, save=save_figs)
    make_pre_opt_plot('gn_ct', feedlim_type, save=save_figs)
    make_pre_opt_plot('subcount', feedlim_type, save=save_figs)
    make_pre_opt_plot('gn_cost', feedlim_type, gn_cost_type='md', save=save_figs)
    make_pre_opt_plot('np_inc_cost', feedlim_type, save=save_figs)
    
    # Plot GW and cost tradeoffs
    plot_gw_tradeoffs(feedlim_type, save=save_figs)
    plot_cost_tradeoffs(feedlim_type, gn_cost_type, save=save_figs)