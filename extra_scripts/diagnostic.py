"""
Visualization: PD Instability Across All Firms
Creates comprehensive plots showing unstable PD behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Load results
results = pd.read_csv('/Users/ruhanirekhi/Documents/structural-credit-challenge/outputs/naive_model_results.csv', parse_dates=['date'])
results = results[results['success'] == True].sort_values(by=['firm_id', 'date']) # type: ignore

firms = ['AAPL', 'JPM', 'TSLA', 'XOM', 'F']
colors = ["#3c6b8d", "#d79c62a6", "#a02c2c", "#74b29271", "#7c67bd"]


# figure 1 - time series of default probabilties

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

fig.suptitle('Baseline Model Weakness: Unstable Default Probabilities', 
             fontsize=16, fontweight='bold', y=0.995)

for i, (firm, color) in enumerate(zip(firms, colors)):
    row = i // 2
    col = i % 2
    
    ax = fig.add_subplot(gs[row, col])
    
    firm_data = results[results['firm_id'] == firm].sort_values('date')
    
    # plot PD over time
    ax.plot(firm_data['date'], firm_data['PD'] * 100, 
            color=color, linewidth=1.5, alpha=0.8)
    
    # highlight COVID period
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30'),  # type: ignore
           alpha=0.15, color='red', label='COVID Crash')
    
    # calculate statistics
    mean_pd = firm_data['PD'].mean() * 100
    std_pd = firm_data['PD'].std() * 100
    cv = std_pd / mean_pd if mean_pd > 0 else 0
    

    # add stats box
    stats_text = f"Mean: {mean_pd:.2f}%\nStd: {std_pd:.2f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'{firm}: PD Over Time (CV={cv:.2f})', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Default Probability (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)


plt.savefig('outputs/pd_instability_timeseries.png', 
            dpi=300, bbox_inches='tight')


# figure 2 - quantitative evidence of pd instability

fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.suptitle('Quantitative Evidence of PD Instability', 
             fontsize=16, fontweight='bold')

# prepare data
stability_metrics = []
for firm in firms:
    firm_data = results[results['firm_id'] == firm].sort_values('date')
    
    mean_pd = firm_data['PD'].mean()
    std_pd = firm_data['PD'].std()
    cv = std_pd / mean_pd if mean_pd > 0 else 0
    
    daily_changes = firm_data['PD'].diff().abs()
    mean_change = daily_changes.mean() * 100
    max_change = daily_changes.max() * 100
    
    # direction changes
    pd_diff = firm_data['PD'].diff()
    direction_changes = ((pd_diff > 0) != (pd_diff.shift(1) > 0)).sum()
    pct_direction_changes = direction_changes / len(firm_data) * 100
    
    stability_metrics.append({
        'firm': firm,
        'cv': cv,
        'mean_change': mean_change,
        'max_change': max_change,
        'direction_changes': pct_direction_changes
    })

metrics_df = pd.DataFrame(stability_metrics)

# ---- coefficient of variation plot ----
ax = axes[0]
bars = ax.bar(metrics_df['firm'], metrics_df['cv'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='High Instability (CV=1.0)')
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Moderate (CV=0.5)')

# add value label on bar
for bar, cv in zip(bars, metrics_df['cv']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cv:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Coefficient of Variation (CV)\n >1.0 (Unstable)', 
            fontweight='bold', fontsize=11)
ax.set_ylabel('CV = Std Dev / Mean', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(metrics_df['cv']) * 1.2)

# ---- max daily jump plot ----
ax = axes[1]
bars = ax.bar(metrics_df['firm'], metrics_df['max_change'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Extreme (>2%pts)')

# add value labels
for bar, val in zip(bars, metrics_df['max_change']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Maximum Daily PD Jump', 
            fontweight='bold', fontsize=11)
ax.set_ylabel('Max Daily Change (%pts)', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/pd_instability_metrics.png', 
            dpi=300, bbox_inches='tight')
