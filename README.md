# Structural Credit Modeling Take-Home  
**Black–Scholes Applied to Corporate Liabilities**

## Overview

This take-home exercise evaluates your ability to reason about financial models **beyond textbook usage**.

You will build a simple structural credit model, apply it to real firm data, diagnose where it fails, and implement a **minimal, principled improvement** supported by empirical evidence.

This is **not** a test of formula memorization or software scale.
We care most about:

- Modeling judgment *(the art of knowing when to bend the rules)*
- Numerical discipline
- Empirical reasoning
- Clarity of technical communication

---

## Objective

You are asked to:

1. Implement a **baseline structural credit model** (Merton, 1974) in which a firm's equity is modeled as a call option on its assets.
2. Calibrate **unobservable firm asset value and asset volatility** using observable equity prices, equity volatility, debt, and risk-free rates.
3. Apply the model to real firms and **identify systematic weaknesses** in its behavior.
4. Propose and implement **one minimal, well-justified model improvement**.
5. Demonstrate quantitatively that the improved model performs better than the baseline under a clearly defined evaluation criterion.
6. Clearly document assumptions, methodology, results, and limitations in a concise technical report.

---

## Background (Conceptual)

### Why equity behaves like a call option

At a debt maturity date $T$:

- If firm assets $V_T > D$, debt holders are paid in full and equity holders receive $V_T - D$.
- If $V_T \le D$, the firm defaults and equity holders receive nothing.

The equity payoff is therefore:

$$\max(V_T - D, 0)$$

This payoff is identical to a **European call option** on firm assets with strike equal to the debt face value.
This observation underlies the Merton (1974) structural credit model.

---

## Mathematical Background

### Merton Model Formulation

The Merton model assumes that firm asset value $V_t$ follows a geometric Brownian motion:

$$dV_t = r V_t dt + \sigma_V V_t dW_t$$

where:
- $r$ is the risk-free rate (assumed constant)
- $\sigma_V$ is the asset volatility (assumed constant)
- $dW_t$ is a Wiener process

Under the risk-neutral measure, the asset value at maturity $T$ is lognormally distributed:

$$V_T = V_0 \exp\left((r - \frac{1}{2}\sigma_V^2)T + \sigma_V \sqrt{T} Z\right)$$

where $Z \sim \mathcal{N}(0,1)$.

### Equity Valuation

Equity is valued as a European call option on firm assets using the Black-Scholes formula:

$$E_t = V_t \Phi(d_1) - D e^{-r(T-t)} \Phi(d_2)$$

where:
- $E_t$ is the market value of equity at time $t$
- $D$ is the face value of debt
- $\Phi(\cdot)$ is the standard normal cumulative distribution function
- $d_1 = \frac{\ln(V_t/D) + (r + \sigma_V^2/2)(T-t)}{\sigma_V \sqrt{T-t}}$
- $d_2 = d_1 - \sigma_V \sqrt{T-t}$

### Calibration Methodology

The unobservable parameters $V_t$ and $\sigma_V$ must be calibrated from observable equity data. This requires solving a system of two equations:

1. **Equity value equation**: $E_t = \text{BlackScholes}(V_t, D, T-t, r, \sigma_V)$

2. **Equity volatility equation**: 
   $$\sigma_E E_t = \frac{\partial E}{\partial V} \sigma_V V_t$$
   
   where $\frac{\partial E}{\partial V} = \Phi(d_1)$ is the option delta.

This system is solved iteratively using numerical methods (e.g., `scipy.optimize.fsolve`).

### Distance-to-Default

Distance-to-default (DD) measures how many standard deviations the expected asset value is above the default threshold:

$$\text{DD} = \frac{E[V_T] - D}{\text{std}(V_T)}$$

where:
- $E[V_T] = V_0 e^{rT}$ (expected asset value at maturity)
- $\text{std}(V_T) = V_0 e^{rT} \sqrt{e^{\sigma_V^2 T} - 1}$ (standard deviation)

A higher DD indicates lower default risk.

### Default Probability

The risk-neutral default probability is the probability that $V_T < D$:

$$\text{PD} = \Phi(-d_2) = \Phi\left(-\frac{\ln(V_0/D) + (r - \sigma_V^2/2)T}{\sigma_V \sqrt{T}}\right)$$

where $\Phi$ is the standard normal CDF.

---

## Setup

### Environment Setup

1. **Create a conda environment** (recommended):
   ```bash
   conda create -n quant_takehome python=3.9
   conda activate quant_takehome
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `numpy`, `scipy`, `pandas` (numerical computing)
   - `matplotlib`, `seaborn` (visualization)
   - `jupyter` (optional, for notebooks)
   - `yfinance`, `fredapi` (for fetching real data)


### Data

All required data is already provided in the `data/` directory:
- `data/synthetic/` - Synthetic data with known parameters (for validation)
- `data/real/` - Real firm data from financial APIs

**Note**: Data generation scripts are available in `baseline/` with `DELETE_` prefix for reference only. You do not need to run them.

### Validate Setup

You can test your setup by running the starter code:

```bash
python -m naive_model
```

This will show TODO messages until you complete the implementation.

### Load and Explore Data

```python
import pandas as pd

# Load equity prices
equity = pd.read_csv('data/real/equity_prices.csv', parse_dates=['date'])

# Load debt (quarterly)
debt = pd.read_csv('data/real/debt_quarterly.csv', parse_dates=['date'])

# Load risk-free rates
rf = pd.read_csv('data/real/risk_free.csv', parse_dates=['date'])

# Explore the data
print(equity.head())
print(f"Firms: {equity['firm_id'].unique()}")
print(f"Date range: {equity['date'].min()} to {equity['date'].max()}")
```

See `DATA.md` for detailed data documentation.

**Example**: See `examples/data_alignment_example.py` for how to align quarterly debt with daily equity data.

---

## Baseline Model (Required)

The baseline model is the **Merton structural credit model**, with the following assumptions:

- Firm asset value follows a geometric Brownian motion.
- Default occurs **only at maturity $T$** if asset value is below debt.
- Equity is treated as a European call option on firm assets.
- Interest rates and asset volatility are constant.
- The firm's capital structure is represented by a **single debt face-value proxy**.

### Required baseline outputs

For each firm and evaluation date, your baseline implementation must produce:

- Estimated firm asset value $V_t$
- Estimated firm asset volatility $\sigma_V$
- Distance-to-default (DD)
- Default probability (PD), or an equivalent implied credit-risk measure

---

## Model Improvement (Required)

After implementing and evaluating the baseline model, you must identify **at least one systematic failure or vulnerability**.
*(Because no model is perfect... except maybe this one in our dreams)*

Examples include (but are not limited to):

- Default only occurring at maturity
- Excessive sensitivity to maturity or volatility assumptions
- Poor behavior for highly leveraged or distressed firms
- Numerical instability or implausible risk dynamics

You must then:

1. Propose **one primary model improvement**
2. Implement it
3. Demonstrate empirically that it improves performance relative to the baseline

### Implementation Structure

Your solution should be organized as follows:

```
naive_model/        # Your baseline Merton model implementation
├── __init__.py
├── __main__.py     # Entry point: python -m naive_model
├── model.py        # Merton model implementation
├── calibration.py  # Asset value/volatility calibration
└── risk_measures.py # DD, PD calculations

improved/           # Your improved model implementation
                    # (Copy from naive_model/ and modify)
├── __init__.py
├── __main__.py     # Entry point: python -m improved
├── model.py        # Improved model implementation
├── calibration.py  # Calibration for improved model
└── risk_measures.py # Risk measures for improved model
```

**Starter files are provided** in `naive_model/` directory. You should:
- Complete the baseline implementation in `naive_model/`
- Copy `naive_model/` to `improved/` and modify it with your improvement
- Ensure both can be run directly: `python -m naive_model` and `python -m improved`

### Constraints

- The improvement must be **minimal** (modify one core assumption).
- The improvement must be **well-justified**.
- Complexity is **not** rewarded. *(Keep it simple and stupid)*
- Both implementations should produce comparable outputs (asset value, volatility, DD, PD)

---

## Evaluation: What Does “Better” Mean?

You must define what “better performance” means and provide evidence.

Acceptable evaluation criteria include:

- Improved alignment with provided benchmark credit signals (if used)
- Improved cross-sectional risk ranking across firms
- Improved time-series stability of risk measures
- Reduction of implausible or unstable behavior
- Improved out-of-sample or rolling-window performance

Your evaluation must include:

- A clearly defined metric
- A comparison between baseline and improved models
- Economic interpretation of results

---

## Data

All required data is provided in the `data/` directory:

```text
data/
├── equity_prices.csv
├── equity_vol.csv
├── debt_quarterly.csv
└── risk_free.csv
```

- **Equity data** is daily.
- **Debt** is reported quarterly and must be aligned appropriately.
- A **single debt proxy** is used by design.
- **Risk-free rates** are provided as a constant or a time series.

Precise definitions, units, and alignment rules are specified in `DATA.md`.

### Data usage rules

- Use **only** the provided data unless explicitly justified.
- No look-ahead bias or future information leakage.
- All simplifying assumptions must be stated and defended.

---

## Reference Implementation (For Validation Only)

The `baseline/` directory contains a **reference implementation for validation purposes only**:

```text
baseline/
├── DELETE_synthetic_test.py      # Data generator (for reference only)
└── DELETE_generate_real_data.py  # Data fetcher (for reference only)
```
---

## Repository Structure (Expected)

You may organize your solution as you see fit. A typical structure might include:

```text
src/          # model and calibration code
evaluation/   # metrics and comparisons
outputs/      # tables and figures
report/       # final report
```
Clarity and reproducibility matter more than structure.

---

## Deliverables

Your submission must include:

### 1. Code

- Fully reproducible code
- Clear instructions or a single entrypoint to run experiments
- Reasonable runtime on a standard laptop

### 2. Technical Report (PDF or Markdown)

The report should include:

- Model formulation and assumptions  
- Calibration methodology  
- Empirical setup and data description  
- Diagnosis of baseline model weaknesses  
- Description of the proposed improvement  
- Quantitative results and evaluation  
- Limitations and possible extensions  

### 3. Output Artifacts

- Tables and/or figures supporting your conclusions  
- Clear labeling and economic interpretation  

---

## Getting Started

**New to this?** See `WALKTHROUGH.md` for a step-by-step guide

### Quick Start

1. Fork this repository.
2. Clone your fork locally.
3. Create a Python environment (see Setup section above).
4. Install dependencies: `pip install -r requirements.txt`
5. Complete the baseline in `naive_model/` (see WALKTHROUGH.md)
6. Implement your improvement in `improved/`

### Setup Instructions

1. Fork this repository.
2. Clone your fork locally.
3. Create a Python environment (virtualenv or conda).
4. Install dependencies listed in `requirements.txt`.
5. **Implement the baseline model in `naive_model/`** (see WALKTHROUGH.md).

---
## Submission Instructions

1. Push all code, report, and output artifacts to your fork.
2. Provide:
   - A link to your fork, **or**
   - A compressed archive of the repository
3. Include the exact command(s) required to reproduce your results.

---

## Evaluation Criteria

Submissions will be evaluated on:

- Correctness and numerical discipline  
- Quality of model diagnosis  
- Soundness of the proposed improvement  
- Strength of empirical evidence  
- Clarity of technical communication  

Thoughtful, well-reasoned solutions are valued over complexity.

---

## References

- Merton, R. (1974). *On the Pricing of Corporate Debt*  
- Black, F., & Cox, J. (1976). *Valuing Corporate Securities*  
- Crosbie, P., & Bohn, J. (KMV). *Modeling Default Risk*
