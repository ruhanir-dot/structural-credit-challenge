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

    # Load data from data/real/ or data/synthetic/

    # Dataset selection, real or synthetic 
    USE_REAL_DATA = True  # toggle depending on if ur using real or synthetic
    
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
        print("REAL data")
    else:
        data_path = 'data/synthetic'
        shares_outstanding = None
        print("SYNTHETIC data")
    
    # loading in data depending on USE_REAL_DATA
    equity_prices = pd.read_csv(f'{data_path}/equity_prices.csv', parse_dates=['date'])
    equity_vol = pd.read_csv(f'{data_path}/equity_vol.csv', parse_dates=['date'])
    debt = pd.read_csv(f'{data_path}/debt_quarterly.csv', parse_dates=['date'])
    risk_free = pd.read_csv(f'{data_path}/risk_free.csv', parse_dates=['date'])
    

    # For each firm and date:
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

        if USE_REAL_DATA:
            # filter debt data for current firm
            firm_debt_quarterly = debt[debt['firm_id'] == firm_id]

            if len(firm_debt_quarterly) == 0:
                print(f"Warning: No debt data for firm {firm_id}")
                equity_dates = pd.Series(firm_equity['date'].unique()).sort_values()
                firm_debt_daily = pd.DataFrame({'date': equity_dates, 'debt': np.nan})
            else:
                equity_dates = pd.Series(firm_equity['date'].unique()).sort_values()
                firm_debt_sorted = firm_debt_quarterly[['date', 'debt']].sort_values('date')
                firm_debt_daily = pd.merge_asof(
                    pd.DataFrame({'date': equity_dates}),
                    firm_debt_sorted,
                    on='date',
                    direction='backward'
                )
                firm_debt_daily['debt'] = firm_debt_daily['debt'].ffill()
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
            # get equity value (E), equity volatility (sigma_E), debt (D), risk-free rate (r)
            date = row['date']
            E = row['equity_price']
            sigma_E = row['equity_vol']
            D = row['debt']
            r = row['risk_free_rate']
            
            if USE_REAL_DATA:
                E = E * shares_outstanding[firm_id] / 1e6  # type: ignore # market cap in millions
                D = D  # already in millions
            
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
    
    
    # output results
    results_df = pd.DataFrame(results)
    
    # create output directory if doesnt exist
    Path('outputs').mkdir(exist_ok=True)
    
    # save to CSV
    results_df.to_csv('outputs/naive_model_results.csv', index=False)
    
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
    
    # per-firm breakdown
    if successful > 0:
        print("\n")
        print("----- Firm breakdown -----")
    
        for firm in firms:
            firm_data = results_df[(results_df['firm_id'] == firm) & (results_df['success'] == True)]
        
            if len(firm_data) > 0:
                # calculate metrics
                leverage = (firm_data['D'] / firm_data['E']).mean()
                mean_pd = firm_data['PD'].mean()
                mean_dd = firm_data['DD'].mean()
                mean_e = firm_data['E'].mean()
                mean_v = firm_data['V'].mean()
            
                print(f"\n{firm}:")
                print(f"  Observations: {len(firm_data)}/{len(results_df[results_df['firm_id'] == firm])} successful")
                print(f"  Mean E (market cap): ${mean_e:,.0f}M")
                print(f"  Mean V (asset value): ${mean_v:,.0f}M")
                print(f"  Mean Leverage (D/E): {leverage:.3f} ({leverage*100:.1f}%)")
                print(f"  Mean PD: {mean_pd:.2%}")
                print(f"  Mean DD: {mean_dd:.2f}")


if __name__ == "__main__":
    main()
