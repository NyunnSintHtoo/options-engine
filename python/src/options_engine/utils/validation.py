"""
Validation utilities for options pricing: put-call parity, bounds checks, arbitrage detection.
"""

import numpy as np


def put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-10,
) -> tuple[bool, float]:
    """
    Verify put-call parity for European options: C - P = S - K*exp(-r*T)

    Args:
        call_price: European call price.
        put_price: European put price.
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        tol: Absolute tolerance for parity check.

    Returns:
        (is_valid, parity_error) tuple.
        is_valid: True if |LHS - RHS| < tol.
        parity_error: |C - P - (S - K*exp(-r*T))|
    """
    discount_factor = np.exp(-r * T)
    lhs = call_price - put_price
    rhs = S - K * discount_factor
    parity_error = abs(lhs - rhs)
    is_valid = parity_error < tol
    return is_valid, parity_error


def european_call_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """
    European call bounds: max(0, S - K*exp(-r*T)) <= C <= S

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    discount_factor = np.exp(-r * T)
    lower = max(0.0, S - K * discount_factor)
    upper = S
    return lower, upper


def european_put_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """
    European put bounds: max(0, K*exp(-r*T) - S) <= P <= K*exp(-r*T)

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    discount_factor = np.exp(-r * T)
    lower = max(0.0, K * discount_factor - S)
    upper = K * discount_factor
    return lower, upper


def american_call_bounds(
    S: float, K: float, T: float, r: float, vol: float, dividend_yield: float = 0.0
) -> tuple[float, float]:
    """
    American call bounds: European_call_lower <= AC <= S

    Note: American call on non-dividend-paying stock = European call.
    With dividends, American call value is bounded below by European call and above by S.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield (used for bound estimation).

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    # Lower bound: American call >= European call (or intrinsic if deep ITM)
    from options_engine.models.black_scholes import bsm_call

    european_lower = bsm_call(S, K, T, r, vol, dividend_yield=dividend_yield)
    intrinsic = max(0.0, S - K)
    lower = max(intrinsic, european_lower)
    upper = S
    return lower, upper


def american_put_bounds(
    S: float, K: float, T: float, r: float, vol: float, dividend_yield: float = 0.0
) -> tuple[float, float]:
    """
    American put bounds: max(K - S, European_put_lower) <= AP <= K

    American put is always worth at least its intrinsic value (K - S).

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield (used for bound estimation).

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    from options_engine.models.black_scholes import bsm_put

    european_value = bsm_put(S, K, T, r, vol, dividend_yield=dividend_yield)
    intrinsic = max(0.0, K - S)
    lower = max(intrinsic, european_value)
    upper = K
    return lower, upper


def check_option_bounds(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    option_type: str = "call",
    is_american: bool = False,
    tol: float = 1e-10,
) -> tuple[bool, str]:
    """
    Check if an option price respects bounds.

    Args:
        price: Option price to check.
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        option_type: "call" or "put".
        is_american: If True, use American bounds; else European.
        tol: Tolerance for bound checks.

    Returns:
        (is_valid, message) tuple.
    """
    if is_american:
        if option_type.lower() == "call":
            lower, upper = american_call_bounds(S, K, T, r, vol)
        else:
            lower, upper = american_put_bounds(S, K, T, r, vol)
    else:
        if option_type.lower() == "call":
            lower, upper = european_call_bounds(S, K, T, r)
        else:
            lower, upper = european_put_bounds(S, K, T, r)

    if price < lower - tol:
        return False, f"{option_type} price {price:.6f} below lower bound {lower:.6f}"
    if price > upper + tol:
        return False, f"{option_type} price {price:.6f} above upper bound {upper:.6f}"

    return True, "OK"
