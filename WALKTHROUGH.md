# Candidate Walkthrough Guide

This guide walks you through the take-home assignment step-by-step from a candidate's perspective.

## Step 1: Setup and Data

1. **Set up environment** (see README Setup section)

2. **Explore the data** (data is already provided in `data/` directory):
   ```python
   import pandas as pd
   
   # Check what you're working with
   equity = pd.read_csv('data/synthetic/equity_prices.csv', parse_dates=['date'])
   print(equity.head())
   print(f"Firms: {equity['firm_id'].unique()}")
   ```

## Step 2: Study the Mathematical Background

**Important**: Do NOT look at the reference implementation yet! Implement everything yourself first.

Review the Mathematical Background section in README.md:
- Black-Scholes call option formula
- Calibration methodology (solving system of equations)
- Distance-to-default formula
- Default probability formula

**Key insight**: You'll need to solve a system of two equations:

$$E = \text{BlackScholes}(V, D, T, r, \sigma_V)$$

$$\sigma_E \cdot E = \frac{\partial E}{\partial V} \cdot \sigma_V \cdot V$$

where $\frac{\partial E}{\partial V} = \Phi(d_1)$ is the option delta.

Use `scipy.optimize.fsolve` to solve this system.

## Step 3: Implement Baseline Model

### 3.1 Complete the Functions

Start with `naive_model/model.py`:
- Implement `black_scholes_call()` - use the formula from Mathematical Background
- Implement `black_scholes_delta()` - derivative w.r.t. underlying price (needed for calibration)
- Implement `MertonModel.equity_value()` - calls Black-Scholes
- Implement `MertonModel.equity_volatility()` - uses delta relationship (∂E/∂V = Φ(d₁))

### 3.2 Implement Calibration

In `naive_model/calibration.py`:
- Set up the system of equations (equity value + equity volatility)
- Use `scipy.optimize.fsolve` to solve
- Handle edge cases (negative values, calibration failures)

### 3.3 Implement Risk Measures

In `naive_model/risk_measures.py`:
- `distance_to_default()` - use formula from Mathematical Background
- `default_probability()` - use formula from Mathematical Background

### 3.4 Complete the Main Function

In `naive_model/__main__.py`, you need to:

1. **Load and align data**:
   ```python
   # Load all data
   equity_prices = pd.read_csv('data/synthetic/equity_prices.csv', parse_dates=['date'])
   equity_vol = pd.read_csv('data/synthetic/equity_vol.csv', parse_dates=['date'])
   debt = pd.read_csv('data/synthetic/debt_quarterly.csv', parse_dates=['date'])
   risk_free = pd.read_csv('data/synthetic/risk_free.csv', parse_dates=['date'])
   
   # Align debt to daily (forward-fill quarterly values)
   debt_daily = debt.set_index('date').reindex(
       equity_prices['date'].unique(), 
       method='ffill'
   ).reset_index()
   ```

2. **For each firm and date, calibrate and compute**:
   ```python
   results = []
   T = 1.0  # Time to maturity (1 year)
   
   for firm_id in equity_prices['firm_id'].unique():
       firm_equity = equity_prices[equity_prices['firm_id'] == firm_id]
       firm_vol = equity_vol[equity_vol['firm_id'] == firm_id]
       firm_debt = debt_daily[debt_daily['firm_id'] == firm_id]
       firm_rf = risk_free
       
       for date in firm_equity['date']:
           # Get values for this date
           E = firm_equity[firm_equity['date'] == date]['equity_price'].values[0]
           sigma_E = firm_vol[firm_vol['date'] == date]['equity_vol'].values[0]
           D = firm_debt[firm_debt['date'] == date]['debt'].values[0]
           r = firm_rf[firm_rf['date'] == date]['risk_free_rate'].values[0]
           
           # Calibrate
           V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
           
           # Compute risk measures
           risk = compute_risk_measures(V, D, T, r, sigma_V)
           
           results.append({
               'date': date,
               'firm_id': firm_id,
               'V': V,
               'sigma_V': sigma_V,
               'DD': risk['DD'],
               'PD': risk['PD']
           })
   ```

3. **Save results**:
   ```python
   results_df = pd.DataFrame(results)
   results_df.to_csv('outputs/naive_results.csv', index=False)
   ```

## Step 4: Diagnose Baseline Weaknesses

Run your baseline on real data and look for:
- **Unstable PD values** (jumping around)
- **Implausible asset values** (negative, too high/low)
- **Poor risk ranking** (distressed firms don't have highest PD)
- **Sensitivity issues** (small changes in inputs cause large changes in outputs)

**Document your findings** - this will go in your report!

## Step 5: Implement Improvement

1. **Copy baseline to improved**:
   ```bash
   cp -r naive_model/* improved/
   ```

2. **Modify the model** based on your diagnosis:
   - Example: If default-only-at-maturity is the issue → implement first-passage time
   - Example: If volatility is unstable → implement stochastic volatility
   - Keep it simple! One core change.

3. **Update all functions** to use your improvement

4. **Run and save results**:
   ```bash
   python -m improved
   # Should save to outputs/improved_results.csv
   ```

## Step 6: Evaluate and Compare

Create a comparison script (e.g., `evaluation/compare_models.py`):

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
naive = pd.read_csv('outputs/naive_results.csv', parse_dates=['date'])
improved = pd.read_csv('outputs/improved_results.csv', parse_dates=['date'])

# Merge on date and firm_id
comparison = naive.merge(
    improved, 
    on=['date', 'firm_id'], 
    suffixes=('_naive', '_improved')
)

# Compute metrics
# Example: Time-series stability (lower std is better)
naive_pd_std = comparison.groupby('firm_id')['PD_naive'].std()
improved_pd_std = comparison.groupby('firm_id')['PD_improved'].std()

print("PD Stability (std):")
print(f"Naive: {naive_pd_std.mean():.4f}")
print(f"Improved: {improved_pd_std.mean():.4f}")

# Example: Cross-sectional ranking
# Check if distressed firms have higher PD
```

## Step 7: Write Report

Structure your report (in `report/` directory):

1. **Introduction**: Brief overview
2. **Model Formulation**: 
   - Baseline Merton model equations
   - Your improvement (what changed, why)
3. **Calibration Methodology**: How you solve for V and sigma_V
4. **Empirical Setup**: 
   - Data description
   - Which firms/dates you used
   - Assumptions (T=1.0, etc.)
5. **Baseline Diagnosis**: 
   - What weaknesses you found
   - Evidence (plots, statistics)
6. **Improved Model Results**: 
   - Quantitative comparison
   - Why it's better
7. **Limitations**: What your model doesn't capture
8. **Conclusion**: Summary

## Common Pitfalls to Avoid

1. **Not testing on synthetic data first** - Start simple!
2. **Forgetting to align debt data** - Quarterly needs forward-fill
3. **Using wrong time to maturity** - Be consistent (T=1.0 is common)
4. **Not handling calibration failures** - Some dates may fail, handle gracefully
5. **Over-complicating the improvement** - One simple change is better than many
6. **Not documenting assumptions** - State all your choices
7. **Look-ahead bias** - Don't use future data to predict past

## Expected Output Format

Your results CSV should have columns:
- `date`: Evaluation date
- `firm_id`: Firm identifier
- `V`: Estimated asset value
- `sigma_V`: Estimated asset volatility
- `DD`: Distance-to-default
- `PD`: Default probability

## Questions to Answer in Your Report

1. What systematic weakness did you identify in the baseline?
2. What improvement did you implement and why?
3. How do you measure "better" performance?
4. What evidence supports your improvement?
5. What are the limitations of your approach?

Good luck!

