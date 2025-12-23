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
    raise NotImplementedError("Implement Black-Scholes call option pricing")


def black_scholes_delta(S, K, T, r, sigma):
    """
    Delta (sensitivity to underlying price) of Black-Scholes call option.
    
    Delta measures how much the option price changes when the underlying price changes.
    For a call option: delta = ∂E/∂V = Φ(d₁)
    
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
        Delta of the call option (between 0 and 1)
    """
    # TODO: Implement Black-Scholes delta
    # Hint: Calculate d1 first, then delta = Φ(d₁) = norm.cdf(d1)
    # d1 = (ln(S/K) + (r + sigma²/2)*T) / (sigma * sqrt(T))
    raise NotImplementedError("Implement Black-Scholes delta")


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
        # Hint: Use black_scholes_delta and the relationship: sigma_E * E = delta * sigma_V * V
        # where delta = ∂E/∂V = Φ(d₁) is the option delta
        raise NotImplementedError("Implement equity volatility calculation")

