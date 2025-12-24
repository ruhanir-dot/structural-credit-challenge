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
    # edge cases
    if S<= 0 or sigma <= 0 or T<=0:
        return 0.0
    if K <= 0: 
        return S

    # calculating d1 and d2
    d1 = (np.log(S/K) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    # calculating call price
    call_price = (S * norm.cdf(d1,0,1)) - (K * np.exp(-r *T) * norm.cdf(d2,0,1))

    return max(0, call_price) # make sure non negative

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
    # edge cases
    if S<= 0 or sigma <= 0 or T<=0 or K <=0:
        return 0.0

    # calculating d1
    d1 = (np.log(S/K) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))

    # vega formula S*N'(d1)sqrt(T) derivative of normal cdf is pdf 
    # but in this secnario, not calculating vega rather delta for merton model

    delta = norm.cdf(d1,0,1)

    return delta


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

        E = black_scholes_call(S=V, K = D, T= self.T, r= r, sigma = sigma_V)
        return E
        
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
        
        if E <= 0: 
            return 0.0

        option_delta = black_scholes_delta(S=V, K=D, T=self.T, r=r, sigma=sigma_V)
        
        sigma_E = (option_delta * sigma_V * V) / E

        return sigma_E
