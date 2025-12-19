"""
Merton Structural Credit Model

Implement the baseline Merton (1974) model here.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option price.
    
    Parameters:
    -----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Call option price
    """
    # TODO: Implement Black-Scholes call option formula
    # Hint: Use the formula from the Mathematical Background section
    
    # calculating d1 and d2
    d1 = (np.log(S/K) + (r + (sigma**2)/2) * T) / sigma * np.sqrt(T)
    d2 = d1 - (sigma * np.sqrt(T))

    call_price = (S * norm.cdf(d1)) - (K * np.e**(-r *T) * norm.cdf(d2))

    return call_price

def black_scholes_vega(S, K, T, r, sigma):
    """
    Vega (sensitivity to volatility) of Black-Scholes call option.
    
    Parameters:
    -----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Vega of the call option
    """
    # TODO: Implement Black-Scholes vega

    raise NotImplementedError("Implement Black-Scholes vega")


class MertonModel:
    """
    Baseline Merton structural credit model.
    
    Assumptions:
    - Firm asset value follows geometric Brownian motion
    - Default occurs only at maturity T if V_T < D
    - Equity is a European call option on firm assets
    """
    
    def __init__(self, T=1.0):
        """
        Initialize Merton model.
        
        Parameters:
        -----------
        T : float
            Time to maturity (years)
        """
        self.T = T
    
    def equity_value(self, V, D, r, sigma_V):
        """
        Calculate equity value as call option on assets.
        
        Parameters:
        -----------
        V : float
            Current asset value
        D : float
            Debt face value
        r : float
            Risk-free rate
        sigma_V : float
            Asset volatility
        
        Returns:
        --------
        float
            Equity value
        """
        # TODO: Implement equity valuation
        # Hint: Use black_scholes_call with V as underlying, D as strike
        raise NotImplementedError("Implement equity valuation")
    
    def equity_volatility(self, V, D, r, sigma_V, E):
        """
        Calculate equity volatility from asset volatility.
        
        Parameters:
        -----------
        V : float
            Current asset value
        D : float
            Debt face value
        r : float
            Risk-free rate
        sigma_V : float
            Asset volatility
        E : float
            Equity value
        
        Returns:
        --------
        float
            Equity volatility
        """
        # TODO: Implement equity volatility relationship
        # Hint: Use vega and the relationship: sigma_E * E = vega * sigma_V * V
        raise NotImplementedError("Implement equity volatility calculation")

