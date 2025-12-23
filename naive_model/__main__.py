"""
Entry point for baseline Merton model.

Run with: python -m basemodel
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
    Main entry point for baseline model.
    
    This function should:
    1. Load data (equity prices, equity vol, debt, risk-free rates)
    2. For each firm and date, calibrate asset value and volatility
    3. Compute risk measures (DD, PD)
    4. Output results (print or save to file)
    """
    print("Baseline Merton Model")
    print("=" * 60)
    
    # TODO: Load data from data/real/ or data/synthetic/

    # Loading in Data
    equity_prices = pd.read_csv('data/synthetic/equity_prices.csv', parse_dates=['date'])
    equity_vol = pd.read_csv('data/synthetic/equity_vol.csv', parse_dates=['date'])
    debt = pd.read_csv('data/synthetic/debt_quarterly.csv', parse_dates=['date'])
    risk_free = pd.read_csv('data/synthetic/risk_free.csv', parse_dates=['date'])


    # TODO: For each firm and date:
    # 1. Get equity value (E), equity volatility (sigma_E), debt (D), risk-free rate (r)
    # 2. Set time to maturity (T, e.g., 1.0 year)
    # 3. Calibrate: V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
    # 4. Compute: DD, PD = compute_risk_measures(V, D, T, r, sigma_V)
    # 5. Store results

    # set parameters 
    T = 1.0 # time to maturity is 1 year

    # get unique firms 
    firms = equity_prices['firm_id'].unique()

    # place to store all results
    results =[]


    for firm_id in firms: 

        # firm specific data
        firm_equity = equity_prices[equity_prices['firm_id'] == firm_id]
        firm_vol = equity_vol[equity_vol['firm_id'] == firm_id]
        firm_debt_quarterly= debt[debt['firm_id'] == firm_id]
        firm_rf = risk_free

        # align debt date
        firm_debt_daily = firm_debt_quarterly.set_index('date')['debt'].reindex(
            firm_equity['date'].unique(),
            method='ffill'
        ).reset_index()
        firm_debt_daily.columns = ['date', 'debt']

        # merge data together
        firm_data = firm_equity.merge(firm_vol, on=['date', 'firm_id'])
        firm_data = firm_data.merge(firm_debt_daily, on='date', how='left')
        firm_data = firm_data.merge(firm_rf, on='date', how='left')

        # for each date for this firm
        for _, row in firm_data.iterrows():
            # get equity value (E), equity volatility (sigma_E), debt (D), risk-free rate (r)
            date = row['date']
            E = row['equity_price']
            sigma_E = row['equity_vol']
            D = row['debt']
            r = row['risk_free_rate']
            
            # check for missing or invalid data
            if pd.isna([E, sigma_E, D, r]).any() or E <= 0 or sigma_E <= 0 or D <= 0:
                # store the failed results
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
                    'PD': np.nan,
                    'success': False
                })
                continue
            
            # calibrate!
            V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
            
            if V is None:
                # for when calibration failed
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
                    'PD': np.nan,
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
                'PD': risk['PD'],
                'success': True
            })
    
    # TODO: Output results
    # Example:
    # results_df = pd.DataFrame(results)
    # results_df.to_csv('outputs/baseline_results.csv', index=False)
    # print("\nResults saved to outputs/baseline_results.csv")
    
    # output results
    results_df = pd.DataFrame(results)
    
    # create output directory if doesnt exist
    Path('outputs').mkdir(exist_ok=True)
    
    # save to CSV, commented it don't want multiple rn
    # results_df.to_csv('outputs/naive_results.csv', index=False)
    
    print("\n")
    print("----- results -----")
    
    total = len(results_df)
    successful = results_df['success'].sum()
    
    print(f"\nTotal observations: {total}")
    print(f"Successful calibrations: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed calibrations: {total - successful} ({(total-successful)/total*100:.1f}%)")
    
    if successful > 0:
        success_df = results_df[results_df['success'] == True]
        print(f"\nCalibrated Values (successful only):")
        print(f"  Mean V: ${success_df['V'].mean():.2f}")
        print(f"  Mean o_V: {success_df['sigma_V'].mean():.2%}")
        print(f"  Mean DD: {success_df['DD'].mean():.2f}")
        print(f"  Mean PD: {success_df['PD'].mean():.2%}")


if __name__ == "__main__":
    main()

