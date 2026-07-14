"""
Analytic Greeks for European options: delta, gamma, vega, theta, rho, vanna, volga, charm.

All Greeks are first/second-order partial derivatives of option price w.r.t. market parameters.
This module provides closed-form formulas where they exist.

Stability notes:
- Delta: straightforward, numeric stable
- Gamma: proportional to phi(d1) / (S*vol*sqrt(T)); watch for vol or T near zero
- Vega: proportional to S*phi(d1)*sqrt(T); also watch for extreme values
- Theta: risk of cancellation between vol decay and discount term; compute separately
- Rho: straightforward, numeric stable
- Vanna, Volga, Charm: second-order; stable but expensive to compute
"""

import numpy as np
from options_engine.utils.numerics import (
    cumulative_normal,
    standard_normal_pdf,
    compute_d1_d2,
)


def delta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    option_type: str = "call",
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option delta: dPrice/dS.

    Call delta = exp(-q*T) * N(d1)
    Put delta = -exp(-q*T) * N(-d1) = exp(-q*T) * (N(d1) - 1)

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield.

    Returns:
        Delta values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    # Handle 0DTE
    if np.any(T == 0):
        is_zero_dte = T == 0
        # At expiry: delta = 1 if S > K (call) / delta = 0 if S <= K (call)
        d_dte = np.where(S > K, 1.0, 0.0)
        if np.isscalar(T) and np.isscalar(K):
            return d_dte
        result = np.where(is_zero_dte, d_dte, np.full_like(S, np.nan, dtype=float))
        nonzero_mask = ~is_zero_dte
        if not np.any(nonzero_mask):
            return result.item() if result.shape == () else result

    d1, _ = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)

    if option_type.lower() == "call":
        delta_val = discount_S * cumulative_normal(d1)
    else:
        delta_val = discount_S * (cumulative_normal(d1) - 1.0)

    return delta_val.item() if delta_val.shape == () else delta_val


def gamma(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option gamma: d²Price/dS² = dDelta/dS.

    Gamma = exp(-q*T) * phi(d1) / (S * vol * sqrt(T))

    Same for calls and puts.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield.

    Returns:
        Gamma values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    sqrt_T = np.sqrt(T)

    # Handle 0DTE and near-zero vol
    if np.any(sqrt_T == 0) or np.any(vol == 0):
        return np.zeros_like(S, dtype=float)

    d1, _ = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)

    gamma_val = discount_S * standard_normal_pdf(d1) / (S * vol * sqrt_T)

    return gamma_val.item() if gamma_val.shape == () else gamma_val


def vega(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option vega: dPrice/dVol.

    Vega = S * exp(-q*T) * phi(d1) * sqrt(T) / 100

    Note: We divide by 100 to get vega per 1% change in vol (common market convention).
    To get raw vega (per 1.0 change), multiply by 100.

    Same for calls and puts.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield.

    Returns:
        Vega values (per 1% vol change).
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    sqrt_T = np.sqrt(T)

    # Handle 0DTE
    if np.any(sqrt_T == 0):
        is_zero_dte = sqrt_T == 0
        vega_val = np.zeros_like(S, dtype=float)
        if np.isscalar(T):
            return vega_val
        vega_val = np.where(is_zero_dte, 0.0, np.full_like(S, np.nan, dtype=float))
        nonzero_mask = ~is_zero_dte
        if not np.any(nonzero_mask):
            return vega_val.item() if vega_val.shape == () else vega_val

    d1, _ = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)

    # Raw vega: S * exp(-q*T) * phi(d1) * sqrt(T)
    vega_val = S * discount_S * standard_normal_pdf(d1) * sqrt_T / 100.0

    return vega_val.item() if vega_val.shape == () else vega_val


def theta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    option_type: str = "call",
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option theta: dPrice/dT (time decay).

    Design: Compute theta carefully to avoid cancellation.
    - Theta_vol: -S * exp(-q*T) * phi(d1) * vol / (2 * sqrt(T))
    - Theta_discount: For calls: +r*K*exp(-r*T)*N(d2) - q*S*exp(-q*T)*N(d1)
                       For puts: -r*K*exp(-r*T)*N(-d2) + q*S*exp(-q*T)*N(-d1)

    Note: Theta is reported as dPrice/dT, i.e., price change per day. Traders often
    quote theta as negative daily decay. We report the mathematical theta (sign as-is).

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield.

    Returns:
        Theta values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    sqrt_T = np.sqrt(T)

    # Handle 0DTE
    if np.any(sqrt_T == 0):
        return np.zeros_like(S, dtype=float)

    d1, d2 = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)
    discount_K = np.exp(-r * T)

    # Vol decay term
    theta_vol = -S * discount_S * standard_normal_pdf(d1) * vol / (2.0 * sqrt_T)

    # Discount/carry term
    if option_type.lower() == "call":
        theta_discount = (
            r * K * discount_K * cumulative_normal(d2)
            - dividend_yield * S * discount_S * cumulative_normal(d1)
        )
    else:
        theta_discount = (
            -r * K * discount_K * cumulative_normal(-d2)
            + dividend_yield * S * discount_S * cumulative_normal(-d1)
        )

    theta_val = theta_vol + theta_discount

    return theta_val.item() if theta_val.shape == () else theta_val


def rho(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    option_type: str = "call",
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option rho: dPrice/dr (interest rate sensitivity).

    Call rho = K * T * exp(-r*T) * N(d2) / 100
    Put rho = -K * T * exp(-r*T) * N(-d2) / 100

    Divided by 100 to report rho per 1% change in rates (market convention).

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield.

    Returns:
        Rho values (per 1% rate change).
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    _, d2 = compute_d1_d2(S, K, T, r, vol)
    discount_K = np.exp(-r * T)

    if option_type.lower() == "call":
        rho_val = K * T * discount_K * cumulative_normal(d2) / 100.0
    else:
        rho_val = -K * T * discount_K * cumulative_normal(-d2) / 100.0

    return rho_val.item() if rho_val.shape == () else rho_val


def vanna(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option vanna: d²Price/dS/dVol (delta-vega interaction).

    Vanna = -exp(-q*T) * phi(d1) * d2 / vol / 100

    Same for calls and puts. Divided by 100 for per-1% vol convention.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield.

    Returns:
        Vanna values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    # Handle zero vol
    if np.any(vol == 0):
        return np.zeros_like(S, dtype=float)

    d1, d2 = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)

    vanna_val = -discount_S * standard_normal_pdf(d1) * d2 / vol / 100.0

    return vanna_val.item() if vanna_val.shape == () else vanna_val


def volga(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option volga: d²Price/dVol² (vega convexity).

    Volga = S * exp(-q*T) * phi(d1) * d1 * d2 / vol² / 10000

    Same for calls and puts. Divided by 10000 for per-1% vol convention (squared).

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        dividend_yield: Continuous dividend yield.

    Returns:
        Volga values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    # Handle zero vol
    if np.any(vol == 0):
        return np.zeros_like(S, dtype=float)

    d1, d2 = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)

    volga_val = (
        S
        * discount_S
        * standard_normal_pdf(d1)
        * d1
        * d2
        / (vol**2)
        / 10000.0
    )

    return volga_val.item() if volga_val.shape == () else volga_val


def charm(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    vol: np.ndarray | float,
    option_type: str = "call",
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Option charm: d²Price/dT/dS (delta decay).

    Design: Avoid cancellation by computing separate terms.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield.

    Returns:
        Charm values.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    S, K, T, r, vol, dividend_yield = np.broadcast_arrays(
        S, K, T, r, vol, dividend_yield
    )

    sqrt_T = np.sqrt(T)

    # Handle 0DTE
    if np.any(sqrt_T == 0):
        return np.zeros_like(S, dtype=float)

    d1, d2 = compute_d1_d2(S, K, T, r, vol)
    discount_S = np.exp(-dividend_yield * T)
    discount_K = np.exp(-r * T)

    # Common term
    common = standard_normal_pdf(d1) / (2.0 * S * vol * sqrt_T)

    if option_type.lower() == "call":
        charm_val = (
            -discount_S * (dividend_yield - r * d2 / sqrt_T) * common
        )
    else:
        charm_val = (
            discount_S * (dividend_yield + r * d2 / sqrt_T) * common
        )

    return charm_val.item() if charm_val.shape == () else charm_val
