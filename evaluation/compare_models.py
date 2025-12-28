"""
Comparison script between naive and improved models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load results from both models."""
    naive = pd.read_csv('outputs/naive_model_results.csv', parse_dates=['date'])
    improved = pd.read_csv('outputs/improved_model_results.csv', parse_dates=['date'])
    
    # filter successful calibrations only
    naive = naive[naive['success'] == True]
    improved = improved[improved['success'] == True]
    
    # for improved model, use PD_smoothed as the main PD
    improved['PD'] = improved['PD_smoothed']
    
    return naive, improved



def compare_time_series_stability(naive, improved):
    """
    Compare time-series stability of risk measures.
    
    Lower standard deviation = more stable = better (usually)
    """
    print("\n" + "="*60)
    print("Time-Series Stability Comparison")
    print("="*60)
    
    # group by firm and compute standard deviation of PD
    naive_stability = naive.groupby('firm_id')['PD'].std()
    improved_stability = improved.groupby('firm_id')['PD'].std()
    
    print("\n----- PD Standard Deviation (lower is more stable): ------")
    print(f"\nNaive Model:")
    print(naive_stability)
    print(f"\nImproved Model:")
    print(improved_stability)
    
    print(f"\nAverage PD Std Dev:")
    print(f"  Naive:    {naive_stability.mean():.4f}")
    print(f"  Improved: {improved_stability.mean():.4f}")
    print(f"  Change:   {improved_stability.mean() - naive_stability.mean():.4f}")
    
    # calculate improvement percentage
    improvement_pct = (naive_stability - improved_stability) / naive_stability * 100
    print(f"\nImprovement by Firm:")
    for firm in improvement_pct.index:
        print(f"  {firm}: {improvement_pct[firm]:.1f}% reduction")
    
    return naive_stability, improved_stability


def compare_cross_sectional_ranking(naive, improved):
    """
    Compare cross-sectional risk ranking.
    
    Do firms with higher leverage have higher PD? (They should!)
    """
    print("\n" + "="*60)
    print("Cross-Sectional Risk Ranking")
    print("="*60)
    
    # get average PD per firm
    naive_avg_pd = naive.groupby('firm_id')['PD'].mean().sort_values(ascending=False)
    improved_avg_pd = improved.groupby('firm_id')['PD'].mean().sort_values(ascending=False)
    
    print("\n----- Average PD by Firm (sorted, highest to lowest): ------")
    print(f"\nNaive Model:")
    print(naive_avg_pd)
    print(f"\nImproved Model:")
    print(improved_avg_pd)
    
    # compare with leverage
    print("\nLeverage Comparison:")
    for firm in naive_avg_pd.index:
        firm_data = naive[naive['firm_id'] == firm]
        avg_leverage = (firm_data['D'] / firm_data['E']).mean()
        naive_pd = naive_avg_pd[firm]
        improved_pd = improved_avg_pd[firm]
        print(f"  {firm}: Leverage={avg_leverage:.2f}, Naive PD={naive_pd:.2%}, Improved PD={improved_pd:.2%}")


def plot_all_firms_pd(naive, improved):
    """
    Plot PD comparison for all firms on a single plot.
    Shows raw (naive) vs improved (smoothed) for each firm.
    """
    firms = ['AAPL', 'JPM', 'TSLA', 'XOM', 'F']
    colors = ["#3c6b8d", "#d79c62a6", "#a02c2c", "#74b29271", "#7c67bd"]
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('Raw vs Improved Default Probability - All Firms', 
                 fontsize=16, fontweight='bold')
    
    for idx, (firm, color) in enumerate(zip(firms, colors)):
        ax = axes[idx]
        
        # get firm data
        naive_firm = naive[naive['firm_id'] == firm].sort_values('date')
        improved_firm = improved[improved['firm_id'] == firm].sort_values('date')
        
        # plot raw and improved
        ax.plot(naive_firm['date'], naive_firm['PD'] * 100, 
                color=color, linewidth=1.0, alpha=0.4, label='Raw (Naive)', linestyle='-')
        ax.plot(improved_firm['date'], improved_firm['PD'] * 100, 
                color=color, linewidth=2.0, alpha=0.9, label='Improved (Smoothed)', linestyle='-')
        
        # calculate improvement
        raw_std = naive_firm['PD'].std() * 100
        smooth_std = improved_firm['PD'].std() * 100
        ax.set_title(f'{firm}', fontweight='bold', fontsize=12, loc='left')
        ax.set_ylabel('Default Probability (%)', fontsize=10)
    
        
        if idx == len(firms) - 1:
            ax.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/comparison_all_firms_pd.png', dpi=300, bbox_inches='tight')

def compare_improvement_metrics(naive, improved):
    """
    Compare improvement metrics for Table 2.
    
    Calculates:
    1. Coefficient of Variation (CV = σ/μ)
    2. Mean Daily Change (in percentage points)
    3. Reduction percentages for both metrics
    """
    print("\n" + "="*60)
    print("Improvement Metrics for table 2")
    print("="*60)
    
    # calculate coefficient of variation st.dev/mean

    naive_mean = naive.groupby('firm_id')['PD'].mean()
    naive_std = naive.groupby('firm_id')['PD'].std()
    naive_cv = naive_std / naive_mean
    
    improved_mean = improved.groupby('firm_id')['PD'].mean()
    improved_std = improved.groupby('firm_id')['PD'].std()
    improved_cv = improved_std / improved_mean
    
    print("\n----- Coefficient of Variation (CV = σ/μ): ------")
    print("\nNaive Model:")
    print(naive_cv)
    print("\nImproved Model:")
    print(improved_cv)
    
    print(f"\nAverage CV:")
    print(f"  Naive:    {naive_cv.mean():.4f}")
    print(f"  Improved: {improved_cv.mean():.4f}")
    print(f"  Change:   {improved_cv.mean() - naive_cv.mean():.4f}")
    
    # calculate CV reduction percentage
    cv_improvement_pct = (naive_cv - improved_cv) / naive_cv * 100
    print(f"\nCV Reduction by Firm:")
    for firm in cv_improvement_pct.index:
        print(f"  {firm}: {cv_improvement_pct[firm]:.1f}% reduction")
    
    # calculate Mean Daily Change (in percentage points)
    firms = ['AAPL', 'JPM', 'TSLA', 'XOM', 'F']
    naive_daily = {}
    improved_daily = {}
    
    for firm in firms:
        # firm data sorted by date
        naive_firm = naive[naive['firm_id'] == firm].sort_values('date')
        improved_firm = improved[improved['firm_id'] == firm].sort_values('date')
        
        # calculate mean daily change (convert to percentage points)
        naive_daily[firm] = np.mean(np.abs(np.diff(naive_firm['PD'].values))) * 100
        improved_daily[firm] = np.mean(np.abs(np.diff(improved_firm['PD'].values))) * 100
    
    # convert to Series for display
    naive_daily_series = pd.Series(naive_daily, name='Daily_Change')
    improved_daily_series = pd.Series(improved_daily, name='Daily_Change')
    
    print("\n----- Mean Daily Change (percentage points): ------")
    print("\nNaive Model:")
    print(naive_daily_series)
    print("\nImproved Model:")
    print(improved_daily_series)
    
    print(f"\nAverage Daily Change:")
    print(f"  Naive:    {naive_daily_series.mean():.4f} percentage points")
    print(f"  Improved: {improved_daily_series.mean():.4f} percentage points")
    print(f"  Change:   {improved_daily_series.mean() - naive_daily_series.mean():.4f} percentage points")
    
    # calculate daily change reduction percentage
    daily_improvement_pct = (naive_daily_series - improved_daily_series) / naive_daily_series * 100
    print(f"\nDaily Change Reduction by Firm:")
    for firm in daily_improvement_pct.index: # type: ignore
        print(f"  {firm}: {daily_improvement_pct[firm]:.1f}% reduction")
    
    print("\n----- Summary: ------")
    print(f"Average CV reduction:           {cv_improvement_pct.mean():.1f}%")
    print(f"Average daily change reduction: {daily_improvement_pct.mean():.1f}%")
    
    return naive_cv, improved_cv, naive_daily_series, improved_daily_series


def main():
    """Main comparison function."""
    print("Model Comparison")
    print("="*60)
    
    # Load results
    try:
        naive, improved = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run both models and saved results to outputs/")
        return
    
    #compare metrics
    compare_time_series_stability(naive, improved)
    compare_cross_sectional_ranking(naive, improved)
    compare_improvement_metrics(naive, improved)
    
   
    plot_all_firms_pd(naive, improved)
    


if __name__ == "__main__":
    main()