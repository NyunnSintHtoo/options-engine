"""
Black-Scholes-Merton closed-form pricing for European options.

Formulas:
- d1 = (ln(S/K) + (r + 0.5*vol^2)*T) / (vol*sqrt(T))
- d2 = d1 - vol*sqrt(T)
- Call = S*N(d1) - K*exp(-r*T)*N(d2)
- Put = K*exp(-r*T)*N(-d2) - S*N(-d1)

With continuous dividend yield q:
- Call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
- Put = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

Design:
- Vectorized: accept scalars, arrays, compute in parallel
- Numerically stable: use robust cumulative normal
- Input validation: clear error messages
- Cost-of-carry: optionally pass q (dividend yield or convenience yield)
"""

import numpy as np
from options_engine.utils.numerics import (
    cumulative_normal,
    standard_normal_pdf,
    validate_option_inputs,
    compute_d1_d2,
)


def bsm_call(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Black-Scholes-Merton European call price.

    C = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)

    Args:
        S: Spot price (scalar or array).
        K: Strike price (scalar or array).
        T: Time to expiry in years (scalar or array).
        r: Risk-free rate (scalar or array).
        vol: Volatility, annualized (scalar or array).
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        Call price(s), same shape as broadcasted inputs.

    Raises:
        ValueError: If any input is invalid.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    # Broadcast to common shape
    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    # Validate inputs (vectorized)
    if np.any(S <= 0):
        raise ValueError("Spot price S must be positive")
    if np.any(K <= 0):
        raise ValueError("Strike price K must be positive")
    if np.any(T < 0):
        raise ValueError("Time to expiry T must be non-negative")
    if np.any(vol <= 0):
        raise ValueError("Volatility vol must be positive")

    # Handle 0DTE edge case
    is_zero_dte = T == 0
    if np.any(is_zero_dte):
        # At expiry, option value is intrinsic
        intrinsic = np.maximum(S - K, 0.0)
        if np.isscalar(T):
            return intrinsic
        result = np.where(is_zero_dte, intrinsic, np.full_like(S, np.nan))
        # Continue computation for non-zero T
        nonzero_mask = ~is_zero_dte
        if not np.any(nonzero_mask):
            return result if result.shape == () else result

    # Compute d1, d2
    d1, d2 = compute_d1_d2(S, K, T, r, vol)

    # BSM call formula with dividend yield
    discount_S = np.exp(-dividend_yield * T)
    discount_K = np.exp(-r * T)

    call_price = discount_S * S * cumulative_normal(d1) - discount_K * K * cumulative_normal(d2)

    # Replace 0DTE values with intrinsic
    if np.any(is_zero_dte):
        intrinsic = np.maximum(S - K, 0.0)
        call_price = np.where(is_zero_dte, intrinsic, call_price)

    return call_price.item() if call_price.shape == () else call_price


def bsm_put(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Black-Scholes-Merton European put price.

    P = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

    Args:
        S: Spot price (scalar or array).
        K: Strike price (scalar or array).
        T: Time to expiry in years (scalar or array).
        r: Risk-free rate (scalar or array).
        vol: Volatility, annualized (scalar or array).
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        Put price(s), same shape as broadcasted inputs.

    Raises:
        ValueError: If any input is invalid.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    # Broadcast to common shape
    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    # Validate inputs
    if np.any(S <= 0):
        raise ValueError("Spot price S must be positive")
    if np.any(K <= 0):
        raise ValueError("Strike price K must be positive")
    if np.any(T < 0):
        raise ValueError("Time to expiry T must be non-negative")
    if np.any(vol <= 0):
        raise ValueError("Volatility vol must be positive")

    # Handle 0DTE edge case
    is_zero_dte = T == 0
    if np.any(is_zero_dte):
        intrinsic = np.maximum(K - S, 0.0)
        if np.isscalar(T):
            return intrinsic
        result = np.where(is_zero_dte, intrinsic, np.full_like(S, np.nan))
        nonzero_mask = ~is_zero_dte
        if not np.any(nonzero_mask):
            return result if result.shape == () else result

    # Compute d1, d2
    d1, d2 = compute_d1_d2(S, K, T, r, vol)

    # BSM put formula with dividend yield
    discount_S = np.exp(-dividend_yield * T)
    discount_K = np.exp(-r * T)

    put_price = discount_K * K * cumulative_normal(-d2) - discount_S * S * cumulative_normal(-d1)

    # Replace 0DTE values with intrinsic
    if np.any(is_zero_dte):
        intrinsic = np.maximum(K - S, 0.0)
        put_price = np.where(is_zero_dte, intrinsic, put_price)

    return put_price.item() if put_price.shape == () else put_price


def bsm_european_bounds(
    S: float,
    K: float,
    T: float,
    r: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    European call and put price bounds (no-arbitrage).

    Returns:
        ((call_lower, call_upper), (put_lower, put_upper)) tuple.
    """
    discount_factor = np.exp(-r * T)
    call_lower = max(0.0, S - K * discount_factor)
    call_upper = S
    put_lower = max(0.0, K * discount_factor - S)
    put_upper = K * discount_factor
    return (call_lower, call_upper), (put_lower, put_upper)


def bsm_american_bounds(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    American call and put price bounds.

    American call is at least as valuable as European call, at most S.
    American put is at least as valuable as intrinsic (K - S), at most K.

    Returns:
        ((call_lower, call_upper), (put_lower, put_upper)) tuple.
    """
    european_call = bsm_call(S, K, T, r, vol, dividend_yield=dividend_yield)
    intrinsic_call = max(0.0, S - K)
    call_lower = max(intrinsic_call, european_call)
    call_upper = S

    european_put = bsm_put(S, K, T, r, vol, dividend_yield=dividend_yield)
    intrinsic_put = max(0.0, K - S)
    put_lower = max(intrinsic_put, european_put)
    put_upper = K

    return (call_lower, call_upper), (put_lower, put_upper)
