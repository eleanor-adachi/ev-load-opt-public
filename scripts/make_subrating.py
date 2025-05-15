# -*- coding: utf-8 -*-
"""
make_subrating.py

Created on Sun Feb 23 17:56:02 2025

@author: elean

Go to https://iopscience.iop.org/article/10.1088/2634-4505/ac949c/meta
Download Supplementary Information

From "Substation capacity" on page 17:
"Substation transformer bank capacities are available from PG&E’s DDOR report 
(PG&E 2021b). In cases where a given substation has multiple transformer banks, 
we add the capacities (in MW) of those banks together to calculate a total 
substation capacity. (On average, there are approximately 1.78 reported banks 
per substation.) This step is necessary because we only have information for 
which substation (i.e., not which specific substation bank) serves a particular 
circuit. This approach may underestimate substation upgrade needs as it is 
possible that capacity is not favorably distributed across the substation banks 
that serve specific circuits."

"""

import os
import pandas as pd

# Inputs
gna_file = r'..\data\raw_data\gna\PGE_2023_GNA_Appendix_E_Public.xlsx'
gna_sheet = 'GNA - Bank & Feeder Capacity'
data_dir = r'..\data'

def make_subrating(gna_file, gna_sheet, save=True, save_dir=''):
    '''Create a csv with maximum ratings of substations'''
    # Read GNA file
    gna_df = pd.read_excel(gna_file, sheet_name=gna_sheet, header=[6,7])
    
    # Combine multiindex column names (append years from level 0)
    gna_cols_0 = list(gna_df.columns.get_level_values(0))
    gna_cols_1 = list(gna_df.columns.get_level_values(1))
    new_gna_cols_0 = list(map(lambda x: str(x)+' ' if type(x)==int else '', gna_cols_0))
    new_gna_cols = [x+y for x,y in zip(new_gna_cols_0, gna_cols_1)]
    gna_df.columns = new_gna_cols
    
    # Filter gna_df for substation transformer banks and desired columns
    bank_df = gna_df[ gna_df['Facility Type']=='Bank' ]
    bank_df = bank_df[[
        'Distribution Planning Regions', 'Division', 'Facility Name', 
        'Facility ID formatted', '2023 Facility Rating (MW)'
        ]]
    
    # Add substation ID
    bank_df['temp'] = bank_df['Facility ID formatted'].astype(str).str.zfill(7)
    bank_df['sub_id'] = bank_df['temp'].apply(lambda x: x[:-2])
    bank_df['sub_id'] = pd.to_numeric(bank_df['sub_id'])
    bank_df['CC'] = bank_df['2023 Facility Rating (MW)'].apply(lambda x: x=='CC')
    bank_df['2023 Facility Rating (MW)'] = pd.to_numeric(
        bank_df['2023 Facility Rating (MW)'], errors='coerce'
        )
    bank_df = bank_df.drop(columns='temp')
    
    # Compute substation rating = sum of ratings across banks at substation
    subrat = bank_df.pivot_table(
        values = ['2023 Facility Rating (MW)', 'Facility Name', 'CC'], 
        index = 'sub_id', 
        aggfunc = {
            '2023 Facility Rating (MW)':'sum',
            'Facility Name':'count',
            'CC':'sum'
            }
        )
    subrat = subrat.reset_index()
    subrat = subrat.rename(
        columns={
            '2023 Facility Rating (MW)':'subrating_MW',
            'Facility Name':'Bank Count', 
            'CC':'CC Bank Count'
            })
    
    # remove substations where one or more banks is CC
    subrat = subrat[ subrat['CC Bank Count']==0 ]
    
    # keep only sub_id and subrating_MW
    subrat = subrat[['sub_id', 'subrating_MW']]

    # Save results
    if save:
        subrat.to_csv(os.path.join(save_dir, 'subrating.csv'), index=False)

    return subrat


#%%

if __name__ == "__main__":
    subrat = make_subrating(gna_file, gna_sheet, save=True, save_dir=data_dir)