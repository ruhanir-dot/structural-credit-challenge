import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# make an output directory
os.makedirs('outputs', exist_ok=True)

# load our datasets in, and make sure data column is in datatime type
equity = pd.read_csv('data/synthetic/equity_prices.csv', parse_dates=['date'])
equity_vol = pd.read_csv('data/synthetic/equity_vol.csv', parse_dates=['date'])
debt = pd.read_csv('data/synthetic/debt_quarterly.csv', parse_dates=['date'])
rf = pd.read_csv('data/synthetic/risk_free.csv', parse_dates = ['date'])

class EDA:
    @staticmethod
    def equity_data_info(equity: pd.DataFrame):
        print("---- Equity EDA ----")
        print(f"{len(equity)} X {len(list(equity.columns))}")
        print(equity.groupby('firm_id')['equity_price'].describe())
        print("\n")
    
    @staticmethod
    def equity_vol_info(equity_vol: pd.DataFrame):
        print("---- Equity Volatility EDA ----")
        print(f"{len(equity_vol)} X {len(list(equity_vol.columns))}")
        print(equity_vol.groupby('firm_id')['equity_vol'].describe())
        print("\n")
    
    @staticmethod
    def debt_data_info(debt: pd.DataFrame):
        print("---- Debt Data EDA ----")
        print(f"{len(debt)} X {len(list(debt.columns))}")
        print(f"# of Quarters: {debt['date'].nunique()}")
        print(debt.groupby('firm_id')['debt'].describe())
        print("\n")
    
    @staticmethod
    def risk_free_info(rf: pd.DataFrame):
        print("---- Risk Free Rate EDA ----")
        print(f"{len(rf)} X {len(list(rf.columns))}")
        print(f"   mean: {rf['risk_free_rate'].mean():.4f}")
        print(f"   min: {rf['risk_free_rate'].min():.4f}")
        print(f"   max: {rf['risk_free_rate'].max():.4f}")
        print(f"   std: {rf['risk_free_rate'].std():.4f}")

EDA.equity_data_info(equity)
EDA.equity_vol_info(equity_vol)
EDA.debt_data_info(debt)
EDA.risk_free_info(rf)
        


# checking simmple data alignment across datasets for a sample firm and dates, can i find values for the date i want
print("\n ----- Data Alignment Check ----- ")

sample_firm = equity['firm_id'].iloc[0] # getting first firm

mid_point = len(equity[equity['firm_id'] == sample_firm]) // 2 # getting mid point index of firm 

sample_dates = equity[equity['firm_id'] == sample_firm]['date'].iloc[mid_point:mid_point+3] # getting 3 sample dates around mid point 

print(f"\nData Alignment: {sample_firm}")
print(f"Dates: {[d.date() for d in sample_dates]}\n")

for sample_date in sample_dates:
    e = equity[(equity['firm_id'] == sample_firm) & (equity['date'] == sample_date)] # equity price at date 
    v = equity_vol[(equity_vol['firm_id'] == sample_firm) & (equity_vol['date'] == sample_date)] # equity vol at date
    d = debt[(debt['firm_id'] == sample_firm) & (debt['date'] <= sample_date)].sort_values('date').tail(1) # most recent debt
    r = rf[rf['date'] == sample_date] # risk free rate at date
    
    print(f"Date: {sample_date.date()}")
    if not e.empty:
        print(f"Equity Price: ${e['equity_price'].values[0]:.2f}")
    else:
        print(f"Equity Price: MISSING")
    
    if not v.empty:
        print(f"Equity Vol: {v['equity_vol'].values[0]:.4f}")
    else:
        print(f"Equity Vol: MISSING")
    
    if not d.empty:
        print(f"Debt (as of {d['date'].values[0]}): ${d['debt'].values[0]:,.0f}")
    else:
        print(f"Debt: MISSING")
    
    if not r.empty:
        print(f"Risk-free Rate: {r['risk_free_rate'].values[0]:.4f}")
    else:
        print(f"Risk-free Rate: MISSING")
    print()

# End of eda.py

