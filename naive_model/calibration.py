"""
Asset Value and Volatility Calibration

Calibrate unobservable asset value (V) and asset volatility (sigma_V)
from observable equity value (E) and equity volatility (sigma_E).
"""

import numpy as np
from scipy.optimize import fsolve

from naive_model.model import black_scholes_call, black_scholes_delta


def calibrate_asset_parameters(E, sigma_E, D, T, r, V0=None, sigma_V0=None):
    """
    Calibrate asset value (V) and asset volatility (sigma_V) from equity data.
    
    This solves the system of equations:
    1. E = BlackScholes(V, D, T, r, sigma_V)
    2. sigma_E * E = (∂E/∂V) * sigma_V * V
    
    where ∂E/∂V = Φ(d₁) is the option delta.
    
    Parameters:
    -----------
    E : float
        Market value of equity
    sigma_E : float
        Equity volatility (annualized)
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    V0 : float, optional
        Initial guess for asset value (default: E + D)
    sigma_V0 : float, optional
        Initial guess for asset volatility (default: sigma_E * E / (E + D))
    
    Returns:
    --------
    tuple (V, sigma_V)
        Estimated asset value and asset volatility
    """
    # TODO: Implement calibration
    # Hint: Use scipy.optimize.fsolve to solve the system of equations
    
    # handle edge case of input 
    if E <= 0 or sigma_E <= 0 or D < 0 or T<=0:
        return None,None

    if V0 is None:
        V0 = E + D  # Simple initial guess

    if sigma_V0 is None:
        # de-lever equity volatility for intial guess
        sigma_V0 = sigma_E * E / (E + D) if (E + D) > 0 else sigma_E
        sigma_V0 = np.clip(sigma_V0, 0.01, 0.99) # setting bounds
    
    def equations(params):
        """
        System of equations to solve.
        
        Returns:
        --------
        list [eq1, eq2]
            Residuals that should be zero at solution
        """
        V, sigma_V = params
        # lets start with edge case of invalid values 
        if V <= 0 or sigma_V <= 0:
            return [1e10, 1e10]  # large penalty for invalid values
        
        # TODO: Equation 1: Equity value equals call option value
        E_calc = black_scholes_call(V, D, T, r, sigma_V)
        eq1 = E_calc - E
        
        # TODO: Equation 2: Equity volatility relationship
        # delta = black_scholes_delta(V, D, T, r, sigma_V)
        # E_vol_calc = (delta * sigma_V * V) / E if E > 0 else 0
        # eq2 = E_vol_calc - sigma_E
        delta = black_scholes_delta(S=V, K=D, T=T, r=r, sigma=sigma_V)
        left = sigma_E * E
        right = delta * sigma_V * V
        eq2 = right - left
        
        return [eq1, eq2]
    
    # TODO: Solve the system
    try:
        result = fsolve(equations, [V0, sigma_V0], xtol=1e-6, full_output= True)
        info = result[1]
        ier = result[2]
        
        V, sigma_V = result[0]

        # check covergence
        if ier != 1:
            return None, None
        
        if V <= 0 or sigma_V <= 0:
            return None, None 
        
        if V < E * 1.01: # V should be larger than E by small margin
            return None, None
        
        if sigma_V>2.0 or sigma_V < 0.0001:
            return None, None
        
        return V, sigma_V
    except Exception as e:
        return None, None
    

def calibrate_with_validation(E, sigma_E, D, T, r, verbose=False):
    """
    Calibrate with additional validation and diagnostics.
    
    Parameters:
    -----------
    E, sigma_E, D, T, r : float
        Same as calibrate_asset_parameters
    verbose : bool
        Print diagnostic information
    
    Returns:
    --------
    dict with keys:
        'V': Calibrated asset value (or None)
        'sigma_V': Calibrated asset volatility (or None)
        'success': Boolean indicating success
        'message': Diagnostic message
    """
    result = {
        'V': None,
        'sigma_V': None,
        'success': False,
        'message': ''
    }
    
    # calibrate
    V, sigma_V = calibrate_asset_parameters(E, sigma_E, D, T, r)
    
    if V is None or sigma_V is None:
        result['message'] = 'Calibration failed to converge'
        if verbose:
            print(f"Calibration failed: E=${E:.2f}, σ_E={sigma_E:.2%}")
        return result
    
    # verify, recomputing E and sigma_E
    E_check = black_scholes_call(V, D, T, r, sigma_V)
    delta = black_scholes_delta(V, D, T, r, sigma_V)
    sigma_E_check = (delta * sigma_V * V) / E_check if E_check > 0 else 0
    
    # check error
    E_error = abs(E_check - E) / E
    sigma_E_error = abs(sigma_E_check - sigma_E) / sigma_E
    
    if E_error > 0.01 or sigma_E_error > 0.01:  # More than 1% error
        result['message'] = f'Poor match: E_error={E_error:.2%}, σ_E_error={sigma_E_error:.2%}'
        if verbose:
            print(f"Error{result['message']}")
        return result
    
    # sucessful case
    result['V'] = V
    result['sigma_V'] = sigma_V
    result['success'] = True
    result['message'] = 'Calibration successful'
    
    if verbose:
        print(f"Success: V=${V:.2f}, σ_V={sigma_V:.2%} (errors: E={E_error:.3%}, σ_E={sigma_E_error:.3%})")
    
    return result
