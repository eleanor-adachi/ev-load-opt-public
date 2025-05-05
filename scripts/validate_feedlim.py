# -*- coding: utf-8 -*-
"""
validate_feedlim.py

Created on Tue Mar  5 10:24:03 2024

@author: elean

Validate feedlim file (derived from ICA) against GNA.

GNA downloaded from PG&E Regulatory Case Documents
- Go to https://pgera.azurewebsites.net/Regulation/search
- Search:
    - Case: "DER Modernize Electric Grid OIR [R.21-06-017]"
    - Party: "PGE"
    - Date(s) from: 08/15/23 to 08/15/23
- Download "Attachment 2: PG&E's 2023 GNA Report Public Appendices" (DER-ModernizeElectricGridOIR_Report_PGE_20230815_763238.zip)
- Extract PGE_2023_GNA_Appendix_E_Public.xlsx and save under data\raw_data\gna

"""
# Standard
import os

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# INPUTS
results_dir = r'..\results'
fig_dir = r'..\figures'
feedlim_file = r'..\data\feedlim.csv'
gna_file = r'..\data\raw_data\gna\PGE_2023_GNA_Appendix_E_Public.xlsx'


def regress_ica_gna(gna_ica_df:pd.DataFrame, save:bool=False):
    '''
    Run linear regression between ICA feeder limits and GNA capacity needs and 
    make validation plots.
    
    Parameters
    ----------
    gna_ica_df : pd.DataFrame
        Merge of ica_peak_df and gna_limit_df
    save : bool, default False
        Save plots to PNG if True.

    Returns
    -------
    Regression DataFrame, Figure and axes of validation plots for ICA feeder limits.

    '''
    # Drop NA values for plotting and regressions
    gna_ica_plt_df = gna_ica_df.dropna()
    
    # Make plots
    fig, axs = plt.subplots(3, 7, sharex=True, sharey=True, figsize=(25,8))
    fig.add_subplot(111, frameon=False)
    
    row_ls = ['b', 't', 'v']
    col_ls = ['mx', '90', 'q3', 'md', 'q1', '10', 'mn']
    
    # Initialize regression DataFrame
    limit_idx = [
        'limit_b_mn_kw','limit_b_10_kw','limit_b_q1_kw','limit_b_md_kw',
        'limit_b_q3_kw','limit_b_90_kw','limit_b_mx_kw', 
        'limit_t_mn_kw','limit_t_10_kw','limit_t_q1_kw','limit_t_md_kw',
        'limit_t_q3_kw','limit_t_90_kw','limit_t_mx_kw', 
        'limit_v_mn_kw','limit_v_10_kw','limit_v_q1_kw','limit_v_md_kw',
        'limit_v_q3_kw','limit_v_90_kw','limit_v_mx_kw', 
        ]
    reg_df = pd.DataFrame(index=limit_idx, columns=['intercept', 'slope', 'R2'])
    
    for i in range(len(row_ls)):
        row_label = row_ls[i]
        for j in range(len(col_ls)):
            col_label = col_ls[j]
            ica_label = 'limit_%s_%s_kw' % (row_label, col_label)
            x = gna_ica_plt_df['GNA_Limit']
            y = gna_ica_plt_df[ica_label]
            axs[i, j].plot(x, y, 'o', markersize=2)
            m, b = np.polyfit(x, y, 1)
            axs[i, j].plot(x, m*x+b, color='red')
            rsq = r2_score(y, m*x+b)
            reg_df.loc[ica_label, 'intercept'] = b
            reg_df.loc[ica_label, 'slope'] = m
            reg_df.loc[ica_label, 'R2'] = rsq
            axs[i, j].text(
                0.1, 0.8, r'$R^2=$'+str(round(rsq,2)), color='red', 
                transform=axs[i, j].transAxes
                )
            if i == 0:
                axs[i, j].set_title(col_label)
            if j == len(col_ls) - 1:
                axs[i, j].text(
                    1.05, 0.5, row_label, verticalalignment='center', rotation=270,
                    transform=axs[i, j].transAxes
                    )
    
    # Hide tick params of big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Difference between facility rating and facility load in 2023 GNA (MW)', fontsize=16)
    plt.ylabel('Annual load hosting capacity limit from ICA (MW)', fontsize=16)
    
    # Save figure
    if save:
        plt.savefig(os.path.join(fig_dir, 'feedlim_validation_plots.png'), bbox_inches='tight')
    
    plt.show()
    return reg_df, fig, axs


def validate_feedlim(feedlim_file:str, gna_file:str, save:bool=False):
    '''
    Validate ICA feeder limits against GNA feeder capacity needs.
    
    Parameters
    ----------
    feedlim_file : str
        Filename of ICA feeder limits.
    gna_file : str
        Filename of GNA Bank & Feeder Capacity Needs.
    save : bool, default False
        Save results as Excel and validation plots as PNG.
    
    Returns
    -------
    Dictionary of DataFrames, figure, axes.

    '''
    # Read in feedlim
    ica_df = pd.read_csv(feedlim_file)

    # Read in GNA
    gna_df = pd.read_excel(gna_file, header=[6,7])
    
    # Combine multiindex column names (append years from level 0)
    gna_cols_0 = list(gna_df.columns.get_level_values(0))
    gna_cols_1 = list(gna_df.columns.get_level_values(1))
    new_gna_cols_0 = list(map(lambda x: str(x)+' ' if type(x)==int else '', gna_cols_0))
    new_gna_cols = [x+y for x,y in zip(new_gna_cols_0, gna_cols_1)]
    gna_df.columns = new_gna_cols
    
    # PREPARE ICA FOR COMPARISON
    # Get minimum values of Limit_kW (peak loading) across months-hours for each FeederID and LimType
    limit_cols = [
        'limit_b_mn_kw','limit_b_10_kw','limit_b_q1_kw','limit_b_md_kw',
        'limit_b_q3_kw','limit_b_90_kw','limit_b_mx_kw', 
        'limit_t_mn_kw','limit_t_10_kw','limit_t_q1_kw','limit_t_md_kw',
        'limit_t_q3_kw','limit_t_90_kw','limit_t_mx_kw', 
        'limit_v_mn_kw','limit_v_10_kw','limit_v_q1_kw','limit_v_md_kw',
        'limit_v_q3_kw','limit_v_90_kw','limit_v_mx_kw', 
        ]
    ica_peak_df = ica_df.pivot_table(values=limit_cols, index='feeder_id', aggfunc='min')
    ica_peak_df = ica_peak_df/1000
    
    # Summary statistics for ICA peak loading limits
    ica_peak_stats = ica_peak_df.describe()
    ica_peak_stats = ica_peak_stats.rename(index={'50%':'median'})
    
    # PREPARE GNA FOR COMPARISON
    # Filter for feeders only
    gna_df = gna_df[ gna_df['Facility Type']=='Feeder' ]
    # Remove any feeders where ID is not a number
    gna_df = gna_df[ pd.to_numeric(gna_df['Facility ID formatted'], errors='coerce').notnull() ]
    # Get difference between 2023 Facility Rating (MW) and 2023 Facility Loading (MW)
    gna_df['2023 Facility Rating (MW)'] = pd.to_numeric(gna_df['2023 Facility Rating (MW)'], errors='coerce')
    gna_df['2023 Facility Loading (MW)'] = pd.to_numeric(gna_df['2023 Facility Loading (MW)'], errors='coerce')
    # Multiply by 1000 to get difference in kW
    gna_df['GNA_Limit'] = gna_df['2023 Facility Rating (MW)'] - gna_df['2023 Facility Loading (MW)']
    # Keep only FeederID and GNA_Limit_kW
    gna_limit_df = gna_df[['Facility ID formatted', 'GNA_Limit']]
    gna_limit_df.rename(columns={'Facility ID formatted':'FeederID'}, inplace=True)
    gna_limit_df.set_index('FeederID', inplace=True)
    gna_limit_df.sort_index(inplace=True)
    # Summary statistics for GNA limits
    gna_limit_stats = gna_limit_df.describe()
    gna_limit_stats = gna_limit_stats.rename(index={'50%':'median'})
    
    # COMPUTE COLUMN-WISE DIFFERENCES BETWEEN ICA AND GNA
    # Difference for each FeederID and LimType (ICA)
    diff_df = pd.DataFrame(columns=ica_peak_df.columns)
    for col in ica_peak_df.columns:
        # Positive means GNA limit is higher than ICA limit (more headroom available / less deficiency)
        # Negative means GNA limit is lower than ICA limit (less headroom available / more deficiency)
        diff_df[col] = gna_limit_df['GNA_Limit'] - ica_peak_df[col]
    # Summary statistics for GNA-ICA differences
    diff_stats = diff_df.describe()
    diff_stats = diff_stats.rename(index={'50%':'median'})
    
    # COMPUTE COLUMN-WISE REGRESSIONS BETWEEN ICA AND GNA
    # Merge datasets
    gna_ica_df = pd.concat([ica_peak_df, gna_limit_df], axis=1)
    # Make regressions and plots
    reg_df, fig, axs = regress_ica_gna(gna_ica_df, save=save)
    
    # Results dictionary
    results_dict = {}
    results_dict['Available MW'] = gna_ica_df
    results_dict['Diff statistics'] = diff_stats
    results_dict['ICA peak statistics'] = ica_peak_stats
    results_dict['GNA statistics'] = gna_limit_stats
    results_dict['ICA-GNA regressions'] = reg_df
    
    # Save results
    if save:
        output_filename = os.path.join(results_dir, 'feedlim_validation.xlsx')
        with pd.ExcelWriter(output_filename, mode='w', engine='openpyxl') as writer:
            gna_ica_df.to_excel(writer, sheet_name='Available MW')
            diff_stats.to_excel(writer, sheet_name='Diff statistics')
            ica_peak_stats.to_excel(writer, sheet_name='ICA peak statistics')
            gna_limit_stats.to_excel(writer, sheet_name='GNA statistics')
            reg_df.to_excel(writer, sheet_name='ICA-GNA regressions')
        
    return results_dict, fig, axs

#%%

if __name__ == "__main__":
    save = True
    results_dict, fig, axs = validate_feedlim(feedlim_file, gna_file, save=save)