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
        



