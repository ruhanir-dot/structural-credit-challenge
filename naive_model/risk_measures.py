"""
Risk Measures: Distance-to-Default and Default Probability

Compute credit risk measures from calibrated asset parameters.
"""

import numpy as np
from scipy.stats import norm


def distance_to_default(V, D, T, r, sigma_V):
    """
    Calculate distance-to-default (DD).
    
    DD measures how many standard deviations the asset value is above
    the default threshold.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Distance-to-default
    """
    # TODO: Implement distance-to-default calculation
    # Hint: See Mathematical Background section in README.md
    # DD = (E[V_T] - D) / std(V_T)
    # where E[V_T] = V * exp(r*T)
    # and std(V_T) = V * exp(r*T) * sqrt(exp(sigma_V^2*T) - 1)

    if V <= 0 or sigma_V <= 0 or T<= 0 or D <= 0:
        return np.nan
    
    expected_V_T = V * np.exp(r *T)
    std_V_T = V* np.exp(r * T) * np.sqrt(np.exp((sigma_V ** 2) * T) - 1)

    if std_V_T == 0: 
        return np.nan
    
    d_to_d = (expected_V_T - D)/ std_V_T

    return d_to_d

def default_probability(V, D, T, r, sigma_V):
    """
    Calculate risk-neutral default probability (PD).
    
    The probability that asset value at maturity will be below debt.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Default probability (between 0 and 1)
    """
    # TODO: Implement default probability calculation
    # Hint: See Mathematical Background section in README.md
    # PD = Phi(-d2) where d2 = (ln(V/D) + (r - sigma_V^2/2)*T) / (sigma_V*sqrt(T))
    
    d2 = (np.log(V/D) + (r - (sigma_V**2)/2) * T) / (sigma_V * np.sqrt(T))

    default_prob = norm.cdf(-d2)    

    return default_prob

def compute_risk_measures(V, D, T, r, sigma_V):
    """
    Compute both distance-to-default and default probability.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    dict
        Dictionary with 'DD' and 'PD' keys
    """
    DD = distance_to_default(V, D, T, r, sigma_V)
    PD = default_probability(V, D, T, r, sigma_V)
    
    return {
        'DD': DD,
        'PD': PD
    }
