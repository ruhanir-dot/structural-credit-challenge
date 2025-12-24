"""
Entry point for IMPROVED Merton model with Exponential Smoothing.

Run with: python -m improved
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from naive_model.model import MertonModel
from naive_model.calibration import calibrate_asset_parameters
from naive_model.risk_measures import compute_risk_measures


def main():
    """
    IMPROVED Merton Model with Exponential Smoothing
    
    This function:
    1. Runs baseline calibration (same as naive model)
    2. Applies exponential smoothing to PDs 
    3. Compares baseline vs improved results
    """
    print("="*80)
    print("IMPROVED MERTON MODEL - WITH EXPONENTIAL SMOOTHING")
    print("="*80)

    # Dataset selection
    USE_REAL_DATA = True  # toggle depending on if ur using real or synthetic
    SMOOTHING_ALPHA = 0.1  # smoothing parameter (0.1=heavy, 0.5=light)
    
    if USE_REAL_DATA:
        data_path = 'data/real'
        
        # outstanding shares
        shares_outstanding = {
            'AAPL': 17.00e9,    # post 4-for-1 split (Aug 2020)
            'JPM': 3.07e9,      # no split
            'TSLA': 0.96e9,     # post 5-for-1 split (Aug 2020)
            'XOM': 4.23e9,      # no split
            'F': 3.92e9         # no split
        }
        print(f"Dataset: REAL data")
        print(f"Smoothing parameter: {SMOOTHING_ALPHA}")
    else:
        data_path = 'data/synthetic'
        shares_outstanding = None
        print(f"Dataset: SYNTHETIC data")
        print(f"Smoothing parameter: {SMOOTHING_ALPHA}")
    
    # loading in data
    equity_prices = pd.read_csv(f'{data_path}/equity_prices.csv', parse_dates=['date'])
    equity_vol = pd.read_csv(f'{data_path}/equity_vol.csv', parse_dates=['date'])
    debt = pd.read_csv(f'{data_path}/debt_quarterly.csv', parse_dates=['date'])
    risk_free = pd.read_csv(f'{data_path}/risk_free.csv', parse_dates=['date'])
    
    # set parameters 
    T = 1.0 # time to maturity is 1 year

    # get unique firms 
    firms = equity_prices['firm_id'].unique()

    # place to store all results
    results = []

    for firm_id in firms: 

        # firm specific data
        firm_equity = equity_prices[equity_prices['firm_id'] == firm_id]
        firm_vol = equity_vol[equity_vol['firm_id'] == firm_id]
        firm_debt_quarterly = debt[debt['firm_id'] == firm_id]
        firm_rf = risk_free

        if USE_REAL_DATA:
            # real data use merge_asof for proper forward-fill
            equity_dates = pd.Series(firm_equity['date'].unique()).sort_values()
            firm_debt_daily = pd.merge_asof(
                pd.DataFrame({'date': equity_dates}),
                firm_debt_quarterly[['date', 'debt']].sort_values('date'),
                on='date',
                direction='forward'
            )
            if firm_debt_daily['debt'].isna().any():
                firm_debt_daily['debt'] = firm_debt_daily['debt'].bfill()
        else:
            # synthetic data
            firm_debt_daily = firm_debt_quarterly.set_index('date')['debt'].reindex(
                firm_equity['date'].unique(),
                method='ffill'
            ).ffill().reset_index()
            firm_debt_daily.columns = ['date', 'debt']

        # merge data together
        firm_data = firm_equity.merge(firm_vol, on=['date', 'firm_id'])
        firm_data = firm_data.merge(firm_debt_daily, on='date', how='left')
        firm_data = firm_data.merge(firm_rf, on='date', how='left')

        # for each date for this firm
        for _, row in firm_data.iterrows():
            date = row['date']
            E = row['equity_price']
            sigma_E = row['equity_vol']
            D = row['debt']
            r = row['risk_free_rate']
            
            if USE_REAL_DATA:
                E = E * shares_outstanding[firm_id] / 1e6  # type: ignore
                D = D
            
            # check for missing or invalid data
            if pd.isna([E, sigma_E, D, r]).any() or E <= 0 or sigma_E <= 0 or D <= 0:
                results.append({
                    'date': date,
                    'firm_id': firm_id,
                    'E': E,
                    'sigma_E': sigma_E,
                    'D': D,
                    'r': r,
                    'V': np.nan,
                    'sigma_V': np.nan,
                    'DD': np.nan,
                    'PD_raw': np.nan,        # Changed: renamed from 'PD'
                    'PD_smoothed': np.nan,   # NEW: will fill later
                    'success': False
                })
                continue
            
            # calibrate
            V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
            
            if V is None:
                results.append({
                    'date': date,
                    'firm_id': firm_id,
                    'E': E,
                    'sigma_E': sigma_E,
                    'D': D,
                    'r': r,
                    'V': np.nan,
                    'sigma_V': np.nan,
                    'DD': np.nan,
                    'PD_raw': np.nan,
                    'PD_smoothed': np.nan,
                    'success': False
                })
                continue
            
            # compute risk measures
            risk = compute_risk_measures(V, D, T, r, sigma_V)
            
            # store results
            results.append({
                'date': date,
                'firm_id': firm_id,
                'E': E,
                'sigma_E': sigma_E,
                'D': D,
                'r': r,
                'V': V,
                'sigma_V': sigma_V,
                'DD': risk['DD'],
                'PD_raw': risk['PD'],      # Raw PD from baseline
                'PD_smoothed': np.nan,     # Will compute next
                'success': True
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # apply exponential smoothing    
    for firm_id in firms:
        # Get successful results for this firm in time order
        firm_mask = (results_df['firm_id'] == firm_id) & (results_df['success'] == True)
        firm_indices = results_df[firm_mask].sort_values('date').index
        
        if len(firm_indices) > 0:
            # get raw PDs
            raw_pds = results_df.loc[firm_indices, 'PD_raw']
            
            # apply exponential smoothing
            smoothed_pds = raw_pds.ewm(alpha=SMOOTHING_ALPHA).mean()
            
            # store back
            results_df.loc[firm_indices, 'PD_smoothed'] = smoothed_pds.values
    

    # create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # save to CSV
    results_df.to_csv('outputs/improved_model_results.csv', index=False)
    
    print("\n")
    print("----- results -----")
    
    total = len(results_df)
    successful = results_df['success'].sum()
    
    print(f"\nTotal observations: {total}")
    print(f"Successful calibrations: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed calibrations: {total - successful}")
    

if __name__ == "__main__":
    main()