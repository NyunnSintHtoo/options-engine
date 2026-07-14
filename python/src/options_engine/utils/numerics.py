"""
Numerical utilities for options pricing: robust cumulative normal, stability helpers.

Design:
- Use scipy.special.ndtr (cumulative standard normal via erfc)
- Document accuracy (>1e-14 relative error everywhere)
- Handle edge cases: deep ITM/OTM, near-zero vol, 0DTE
- Mitigate catastrophic cancellation in theta and Greeks
"""

import numpy as np
from scipy import special
from scipy import stats


def cumulative_normal(x: np.ndarray) -> np.ndarray:
    """
    Cumulative standard normal distribution N(x).

    Uses scipy.special.ndtr via erfc for robust computation.
    Accurate to >1e-14 relative error across the domain.

    Args:
        x: Input value(s), any shape.

    Returns:
        N(x): Cumulative probability, same shape as input.
    """
    return special.ndtr(x)


def standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal probability density function phi(x) = exp(-x^2/2) / sqrt(2*pi).

    Args:
        x: Input value(s), any shape.

    Returns:
        phi(x): Probability density, same shape as input.
    """
    return stats.norm.pdf(x)


def validate_option_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    option_type: str = "call",
) -> None:
    """
    Validate option pricing inputs; raise clear errors.

    Args:
        S: Spot price (must be > 0).
        K: Strike price (must be > 0).
        T: Time to expiry in years (must be >= 0).
        r: Risk-free rate (can be negative, common in some markets).
        vol: Volatility (annualized, must be > 0).
        option_type: "call" or "put".

    Raises:
        ValueError: If any input is invalid.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive; got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive; got {K}")
    if T < 0:
        raise ValueError(f"Time to expiry T must be non-negative; got {T}")
    if vol <= 0:
        raise ValueError(f"Volatility vol must be positive; got {vol}")
    if option_type.lower() not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'; got {option_type}")
    if not np.isfinite(r):
        raise ValueError(f"Risk-free rate r must be finite; got {r}")


def compute_d1_d2(
    S,
    K,
    T,
    r,
    vol,
):
    """
    Compute d1 and d2 for Black-Scholes formula.

    d1 = (ln(S/K) + (r + 0.5*vol^2)*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.

    Returns:
        (d1, d2) tuple.
    """
    sqrt_T = np.sqrt(T)
    # Handle 0DTE: use np.any() for array-safe check
    is_zero_dte = sqrt_T == 0
    if np.any(is_zero_dte):
        # 0DTE: d1 and d2 become infinite if S != K, undefined if S == K
        # For vectorized operation, compute normally and replace later
        vol_sqrt_T = np.where(is_zero_dte, 1.0, vol * sqrt_T)  # Avoid division by zero
    else:
        vol_sqrt_T = vol * sqrt_T

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T

    # Replace 0DTE values with symbolic representation
    if np.any(is_zero_dte):
        # For 0DTE: d1 = +inf if S > K, -inf if S < K, else 0
        d1 = np.where(is_zero_dte, np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0)), d1)
        d2 = np.where(is_zero_dte, np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0)), d2)

    return d1, d2


def moneyness_level(S: float, K: float) -> str:
    """
    Classify moneyness: deep OTM, OTM, ATM, ITM, deep ITM.

    Args:
        S: Spot price.
        K: Strike price.

    Returns:
        Moneyness classification string.
    """
    m = S / K
    if m < 0.9:
        return "deep_otm"
    elif m < 0.98:
        return "otm"
    elif m <= 1.02:
        return "atm"
    elif m <= 1.1:
        return "itm"
    else:
        return "deep_itm"
